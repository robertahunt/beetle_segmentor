from PyQt5 import QtWidgets, QtGui, QtCore
import os
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from copy import deepcopy

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import (
    felzenszwalb,
    watershed,
    slic,
    quickshift,
    mark_boundaries,
)
from PyQt5.QtWidgets import QLabel, QApplication
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QImage, QPainter, QPixmap, QCursor
import qimage2ndarray as q2np


class ImageViewer(QtWidgets.QGraphicsView):
    factor = 2.0

    def __init__(
        self,
        ROOT_FOLDER,
        IMAGE_FOLDER,
        FINAL_MASKS_FOLDER,
        EXISTING_MASKS_FOLDER,
        SUBSET=[],
        NUM_CLASSES=1,
        parent=None,
    ):
        super().__init__(parent)
        self.just_drag = False
        self.setMouseTracking(True)
        self.zoomed = 1
        self.parent = parent
        self.ROOT_FOLDER = ROOT_FOLDER
        self.FINAL_MASKS_FOLDER = FINAL_MASKS_FOLDER
        self.EXISTING_MASKS_FOLDER = EXISTING_MASKS_FOLDER
        self.IMAGE_FOLDER = IMAGE_FOLDER
        self.SUBSET = SUBSET
        self.NUM_CLASSES = NUM_CLASSES
        self.OUTLINE_THICKNESS = 3
        self.blur=False
        self.current_class = 1
        self.mask_colors = (np.array(plt.get_cmap('Dark2').colors)*255).astype('uint8')

        assert len(self.mask_colors) >= self.NUM_CLASSES, "Too many classes for the number of colors we have!! oh no!"
        
        self.viewport().setCursor(QCursor(Qt.ArrowCursor))
        self.files = sorted(os.listdir(self.IMAGE_FOLDER))

        self.file_no = 0

        final_masks = [
            os.path.basename(x)
            for x in glob(os.path.join(self.FINAL_MASKS_FOLDER, "*.png"))
        ]
        # remove any files that don't have masks
        self.files = [x for x in self.files if os.path.exists(os.path.join(self.EXISTING_MASKS_FOLDER, x))]

        # remove any files that already have final segmentations
        self.files = [x for x in self.files if (x not in final_masks) or (x in self.SUBSET)]
        if len(self.SUBSET):
            self.files = [x for x in self.files if x in self.SUBSET]

        self.brushSize = 20

        self.setRenderHints(
            QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform
        )
        self.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.setBackgroundRole(QtGui.QPalette.Dark)

        self.scene = QtWidgets.QGraphicsScene()
        self.setScene(self.scene)

        self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self._pixmap_item.mousePressEvent = self._mousePressEvent
        self._pixmap_item.mouseMoveEvent = self._mouseMoveEvent
        self._pixmap_item.mouseReleaseEvent = self._mouseReleaseEvent
        self.scene.addItem(self._pixmap_item)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

        self.viewport().installEventFilter(self)

    def wheelEvent(self, event):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ShiftModifier:
            self.brushSize = max(self.brushSize + int(event.angleDelta().y() / 120), 1)
            self.refreshCursorPixmap()
            self.refreshPixmap()
            event.accept()

    def get_viewport_pos(self, event):
        pos = event.pos()
        offsets = self.mapToScene(self.viewport().geometry()).boundingRect()

        x = int(offsets.x() + pos.x() / self.zoomed)
        y = int(offsets.y() + pos.y() / self.zoomed)
        return x, y

    def resetManualHistToCursor(self):
        self.manual_backgrounds_hist = self.manual_backgrounds_hist[: self.history_cursor]
        self.manual_foregrounds_hist = self.manual_foregrounds_hist[: self.history_cursor]

    def setHistToCursor(self):
        self.manual_foreground = self.manual_foregrounds_hist[self.history_cursor - 1].copy()
        self.manual_background = self.manual_backgrounds_hist[self.history_cursor - 1].copy()

        self._mask = self.load_mask(self.maskPath)
        self._mask[np.where(self.manual_background > 0)] = 0
        self._mask[np.where(self.manual_foreground > 0)] = self.manual_foreground[np.where(self.manual_foreground > 0)]
        self.refreshManualAnnotationsPixmap()
        self.refreshMaskOutlinePixmap()
        self.refreshMaskPixmap()

    def eventFilter(self, source, event):
        if event.type() == QEvent.MouseMove:
            x, y = self.get_viewport_pos(event)
            self.cursor = np.zeros((self.img_h, self.img_w))#np.array([[0] * self.img_w] * self.img_h)
            if self.parent.side_bar_widget.use_brush.isChecked():
                self.cursor = cv2.circle(
                    self.cursor.astype("uint8"),
                    (x, y),
                    self.brushSize // 2,
                    255,
                    1,
                )
            else:
                if (y < self.superpixelContours.shape[0]) & (x < self.superpixelContours.shape[1]):
                    segment = self.superpixelContours[y, x]
                    self.cursor[np.where(self.superpixelContours == segment)] = 255
            self.refreshCursorPixmap()

        return super(ImageViewer, self).eventFilter(source, event)

    def _mouseMoveEvent(self, event):
        if self.parent.side_bar_widget.use_brush.isChecked():
            if (event.buttons() & Qt.RightButton) and self.drawing:
                pos = event.pos()
                self.manual_background = cv2.line(
                    self.manual_background.astype("uint8"),
                    (int(self.lastPoint.x()), int(self.lastPoint.y())),
                    (int(pos.x()), int(pos.y())),
                    255,
                    self.brushSize,
                )
                self.manual_foreground = cv2.line(
                    self.manual_foreground.astype("uint8"),
                    (int(self.lastPoint.x()), int(self.lastPoint.y())),
                    (int(pos.x()), int(pos.y())),
                    0,
                    self.brushSize,
                )
                self.lastPoint = event.pos()
                self.refreshManualAnnotationsPixmap()
                self.refreshMaskOutlinePixmap()
                self.refreshMaskPixmap()

            if (event.buttons() & Qt.LeftButton) and self.drawing:
                pos = event.pos()
                self.manual_foreground = cv2.line(
                    self.manual_foreground.astype("uint8"),
                    (int(self.lastPoint.x()), int(self.lastPoint.y())),
                    (int(pos.x()), int(pos.y())),
                    self.current_class,
                    self.brushSize,
                )
                self.manual_background = cv2.line(
                    self.manual_background.astype("uint8"),
                    (int(self.lastPoint.x()), int(self.lastPoint.y())),
                    (int(pos.x()), int(pos.y())),
                    0,
                    self.brushSize,
                )
                self.lastPoint = event.pos()
                self.refreshManualAnnotationsPixmap()
                self.refreshMaskOutlinePixmap()
                self.refreshMaskPixmap()


    def _mouseReleaseEvent(self, event):
        if (
            (event.button() == Qt.LeftButton)
            or (event.button() == Qt.RightButton)
            and (self.just_drag == False)
        ):
            self.drawing = False
            self.manual_foregrounds_hist += [self.manual_foreground.copy()]
            self.manual_backgrounds_hist += [self.manual_background.copy()]
            self.history_cursor = len(self.manual_foregrounds_hist)
        elif event.button() == Qt.MidButton:
            # for changing back to Qt.OpenHandCursor
            self.viewport().setCursor(Qt.OpenHandCursor)
            handmade_event = QtGui.QMouseEvent(
                QEvent.MouseButtonRelease,
                QtCore.QPointF(event.pos()),
                Qt.LeftButton,
                event.buttons(),
                Qt.KeyboardModifiers(),
            )
            self.just_drag = True
            self.mouseReleaseEvent(handmade_event)
            self.just_drag = False

        self.refreshManualAnnotationsPixmap()
        self.refreshMaskOutlinePixmap()
        self.refreshMaskPixmap()

    def _mousePressEvent(self, event):
        x = int(event.pos().x())
        y = int(event.pos().y())
        if (event.button() == Qt.LeftButton) and (self.just_drag == False):

            self.resetManualHistToCursor()
            self.drawing = True
            if self.parent.side_bar_widget.use_brush.isChecked():
                self.lastPoint = event.pos()
                self.manual_background = cv2.line(
                    self.manual_background.astype("uint8"),
                    (int(self.lastPoint.x()), int(self.lastPoint.y())),
                    (int(self.lastPoint.x()), int(self.lastPoint.y())),
                    0,
                    self.brushSize,
                )
                self.manual_foreground = cv2.line(
                    self.manual_foreground.astype("uint8"),
                    (int(self.lastPoint.x()), int(self.lastPoint.y())),
                    (int(self.lastPoint.x()), int(self.lastPoint.y())),
                    self.current_class,
                    self.brushSize,
                )
            else:
                if (y < self.superpixelContours.shape[0]) & (x < self.superpixelContours.shape[1]):
                    segment = self.superpixelContours[y, x]
                    self.manual_background[np.where(self.superpixelContours == segment)] = 0
                    self.manual_foreground[np.where(self.superpixelContours == segment)] = self.current_class
        elif (event.button() == Qt.RightButton) and (self.just_drag == False):
            self.resetManualHistToCursor()
            self.drawing = True

            if self.parent.side_bar_widget.use_brush.isChecked():
                self.lastPoint = event.pos()
                self.manual_foreground = cv2.line(
                    self.manual_foreground.astype("uint8"),
                    (int(self.lastPoint.x()), int(self.lastPoint.y())),
                    (int(self.lastPoint.x()), int(self.lastPoint.y())),
                    0,
                    self.brushSize,
                )
                self.manual_background = cv2.line(
                    self.manual_background.astype("uint8"),
                    (int(self.lastPoint.x()), int(self.lastPoint.y())),
                    (int(self.lastPoint.x()), int(self.lastPoint.y())),
                    255,
                    self.brushSize,
                )
            else:
                if (y < self.superpixelContours.shape[0]) & (x < self.superpixelContours.shape[1]):
                    segment = self.superpixelContours[y, x]
                    self.manual_foreground[np.where(self.superpixelContours == segment)] = 0
                    self.manual_background[np.where(self.superpixelContours == segment)] = 255
        elif event.button() == Qt.MidButton:
            self.viewport().setCursor(Qt.ClosedHandCursor)
            self.original_event = event
            handmade_event = QtGui.QMouseEvent(
                QEvent.MouseButtonPress,
                QtCore.QPointF(event.pos()),
                Qt.LeftButton,
                event.buttons(),
                Qt.KeyboardModifiers(),
            )
            self.just_drag = True
            self.mousePressEvent(handmade_event)
            self.just_drag = False
        self._mask[np.where(self.manual_foreground > 0)] = self.manual_foreground[np.where(self.manual_foreground > 0)]
        self._mask[np.where(self.manual_background > 0)] = 0
        self.refreshManualAnnotationsPixmap()
        self.refreshMaskOutlinePixmap()
        self.refreshMaskPixmap()

    def load_image(self, filePath, opacity=0.5):
        self.filePath = filePath
        self.fileName = os.path.basename(filePath)
        self.parent.side_bar_widget.id_label.setText(self.fileName)
        final_seg_fp = os.path.join(self.FINAL_MASKS_FOLDER, self.fileName)
        if os.path.exists(final_seg_fp):
            self.maskPath = final_seg_fp
            self.parent.side_bar_widget.has_final_seg.setText("Has Final Mask: True")
        else:
            self.maskPath = os.path.join(self.EXISTING_MASKS_FOLDER, self.fileName)
            self.parent.side_bar_widget.has_final_seg.setText("Has Final Mask: False")

        self.imgPixmap = QImage(filePath)
        self.img_w, self.img_h = self.imgPixmap.size().width(), self.imgPixmap.size().height()
        self.manual_foreground = np.array([[0] * self.img_w] * self.img_h)
        self.manual_background = np.array([[0] * self.img_w] * self.img_h)

        self.manual_foregrounds_hist = [self.manual_foreground.copy()]
        self.manual_backgrounds_hist = [self.manual_background.copy()]
        self.history_cursor = 1
        self.cursor = np.zeros((self.img_h, self.img_w))#np.array([[0] * self.img_w] * self.img_h)
        

        if self.imgPixmap.isNull():
            print("Could not find image, ", filePath)
            return False

        self.parent.side_bar_widget.load_img_meta()

        self.parent.side_bar_widget.perc_done.setText(
            "Perc Done: %s" % (self.file_no / len(self.files))
        )


        self.img = cv2.cvtColor(deepcopy(q2np.byte_view(self.imgPixmap)[:, :, :3]), cv2.COLOR_BGR2RGB)
        if self.blur:
            self.img = cv2.blur(self.img, (5,5))

        if os.path.exists(self.maskPath):
            self._mask = self.load_mask(self.maskPath)
        else:
            self._mask = np.zeros((self.img_h, self.img_w)).astype('uint8')
        self.refreshCursorPixmap()
        self.refreshManualAnnotationsPixmap()
        self.refreshMaskPixmap()
        self.refreshMaskOutlinePixmap()
        self.refreshSuperpixelContoursPixmap()

        return True

    def load_next_img(self):
        self.file_no = min(self.file_no + 1, len(self.files))
        self.load_image(os.path.join(self.IMAGE_FOLDER, self.files[self.file_no]))

    def load_prev_img(self):
        self.file_no = max(self.file_no - 1, 0)
        self.load_image(os.path.join(self.IMAGE_FOLDER, self.files[self.file_no]))

    def refreshPixmap(self):
        show_outline = self.parent.side_bar_widget.outline_visible.isChecked()
        opacity = self.parent.side_bar_widget.slider_mask_opacity.value() / 100
        show_mask = self.parent.side_bar_widget.mask_visible.isChecked()
        show_manual_annotations = (self.parent.side_bar_widget.show_manual_annotations.isChecked())
        use_brush = self.parent.side_bar_widget.use_brush.isChecked()
        
        self.imgPixmap = q2np.array2qimage(self.img)
        painter = QPainter()
        painter.begin(self.imgPixmap)
        if show_mask and hasattr(self, 'maskPixmap'):
            painter.setOpacity(opacity)
            painter.drawImage(0, 0, self.maskPixmap)
        if show_outline and hasattr(self, 'maskOutlinePixmap'):
            painter.drawImage(0 ,0, self.maskOutlinePixmap)
        if show_manual_annotations and hasattr(self, 'manualAnnotationsPixmap'):
            painter.drawImage(0 ,0, self.manualAnnotationsPixmap)
        if not use_brush and hasattr(self, 'superpixelContoursPixmap'):
            painter.drawImage(0, 0, self.superpixelContoursPixmap)
        painter.drawImage(0 ,0, self.cursorPixmap)
        painter.end()

        self._pixmap_item.setPixmap(QPixmap.fromImage(self.imgPixmap))

    def refreshCursorPixmap(self):
        cur = np.stack((self.cursor,self.cursor,self.cursor,self.cursor*0.5),axis=2)
        self.cursorPixmap = q2np.array2qimage(cur)
        self.refreshPixmap()

    def refreshMaskOutlinePixmap(self):
        outline = np.zeros((self.img_h, self.img_w, 3))
        for class_id, color in zip(range(1,self.NUM_CLASSES), self.mask_colors):
            color = tuple(color.astype('float'))
            if (self._mask == class_id).sum() > 0:
                contours, _ = cv2.findContours((self._mask == class_id).astype('uint8'), 1, 2)
                outline = cv2.drawContours(outline, contours, -1, color, self.OUTLINE_THICKNESS)

        self.maskOutlinePixmap = q2np.array2qimage(outline)
        self.refreshPixmap()

    def refreshMaskPixmap(self):
        self.maskPixmap = q2np.array2qimage(self._mask)
        self.refreshPixmap()
        
    def refreshManualAnnotationsPixmap(self):
        alpha = np.zeros(self.manual_foreground.shape)
        alpha[np.where((self.manual_foreground + self.manual_background) > 0)] = 255

        manual_annot = np.zeros((self.img_h, self.img_w, 3))
        for class_id, color in zip(range(1,self.NUM_CLASSES), self.mask_colors):
            color = tuple(color.astype('float'))
            if (self._mask == class_id).sum() > 0:
                contours, _ = cv2.findContours((self._mask == class_id).astype('uint8'), 1, 2)
                manual_annot = cv2.drawContours(manual_annot, contours, -1, color, -1)

        self.manualAnnotationsPixmap = q2np.array2qimage(manual_annot)
        self.refreshPixmap()

    def refreshSuperpixelContoursPixmap(self):
        superpixel_scale = int(self.parent.side_bar_widget.superpixel_scale.currentText())
        superpixel_sigma = (
                self.parent.side_bar_widget.superpixel_sigma.value() / 100
            )
        superpixel_compactness = (
                self.parent.side_bar_widget.superpixel_compactness.value()
            )
        self.superpixelContours = slic(
                    self.img,
                    n_segments=max(1, superpixel_scale),
                    compactness=max(1, superpixel_compactness),
                    sigma=max(0.001, superpixel_sigma / 25),
                    start_label=1,
                )
        self.superpixelContoursOutlines = np.zeros(self.img.shape)
        self.superpixelContoursOutlines = (mark_boundaries(self.superpixelContoursOutlines, self.superpixelContours, mode="thin") * 255).astype("uint8")
        alpha = np.zeros(self._mask.shape)
        alpha[np.where(self.superpixelContoursOutlines.sum(axis=2) > 0)] = 255
        self.superpixelContoursOutlines = np.concatenate((self.superpixelContoursOutlines, np.expand_dims(alpha,axis=2)), axis=2)
        self.superpixelContoursPixmap = q2np.array2qimage(self.superpixelContoursOutlines)
        self.refreshPixmap()
        
        
    def load_mask(self, mask_path):
        mask = QImage(mask_path)

        if mask.isNull():
            print("Could not find mask, ", self.maskPath)
            mask = np.zeros(self.img.shape[:2]).astype("uint8")
        else:
            mask = q2np.byte_view(mask).astype('uint8')

        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        if mask.shape != self.img.shape[:2]:
            new_mask = np.zeros(self.img.shape[:2])
            w = min(self.img.shape[0], mask.shape[0])
            h = min(self.img.shape[1], mask.shape[1])
            new_mask[:w, :h] = mask[:w, :h]
            mask = new_mask.astype("uint8")

        assert mask.shape == self.img.shape[:2]

        mask[np.where(self.manual_background == 255)] = 0
        mask[np.where(self.manual_foreground > 0)] = self.manual_foreground[np.where(self.manual_foreground > 0)]
        self._mask = mask
        return self._mask

    def save_mask(self):
        fn = self.files[self.file_no]
        fp = os.path.join(self.FINAL_MASKS_FOLDER, fn)
        cv2.imwrite(fp, self._mask)

    def shiftMask(self, i=0, j=0):
        self.mask_shift_i += i
        self.mask_shift_j += j
        self.refreshMaskOutlinePixmap()
        self.refreshMaskPixmap()

    def zoomIn(self):
        self.zoom(self.factor)

    def zoomOut(self):
        self.zoom(1 / self.factor)

    def zoom(self, f):
        self.zoomed = f * self.zoomed
        self.scale(f, f)

    def resetZoom(self):
        self.resetTransform()

    def fitToWindow(self):
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def changeCurrentClass(self, value):
        self.current_class = int(self.parent.side_bar_widget.current_class.currentText())
        self.refreshMaskOutlinePixmap()
        self.refreshMaskPixmap()

    def changeOpacity(self, value):
        self.refreshMaskOutlinePixmap()
        self.refreshMaskPixmap()

    def changeMaskClass(self, value):
        self.refreshMaskOutlinePixmap()
        self.refreshMaskPixmap()

    def toggleMaskVisibility(self, value):
        self.refreshMaskOutlinePixmap()
        self.refreshMaskPixmap()

    def toggleOutlineVisibility(self, value):
        self.refreshMaskOutlinePixmap()
        self.refreshMaskPixmap()

    def toggleManualAnnotations(self, value):
        self.refreshPixmap()

    def toggleBrush(self, value):
        self.refreshCursorPixmap()

    def zoomToMask(self, value):
        pass

    def changeSuperpixelScale(self, value):
        self.refreshSuperpixelContoursPixmap()
        self.parent.side_bar_widget.scale_label.setText(
            "Superpixel Scale: %s"
            % int(self.parent.side_bar_widget.superpixel_scale.currentText())
        )

    def changeSuperpixelSigma(self, value):
        self.refreshSuperpixelContoursPixmap()
        self.parent.side_bar_widget.sigma_label.setText(
            "Superpixel Sigma: %s / 25"
            % self.parent.side_bar_widget.superpixel_sigma.value()
        )

    def changeSuperpixelCompactness(self, value):
        self.refreshSuperpixelContoursPixmap()
        self.parent.side_bar_widget.compact_label.setText(
            "Superpixel Compactness: %s"
            % self.parent.side_bar_widget.superpixel_compactness.value()
        )

    def shiftMaskUp(self):
        self.shiftMask(i=-1, j=0)
        self.refreshMaskOutlinePixmap()
        self.refreshMaskPixmap()

    def shiftMaskLeft(self):
        self.shiftMask(i=0, j=-1)
        self.refreshMaskOutlinePixmap()
        self.refreshMaskPixmap()

    def shiftMaskRight(self):
        self.shiftMask(i=0, j=1)
        self.refreshMaskOutlinePixmap()
        self.refreshMaskPixmap()

    def shiftMaskDown(self):
        self.shiftMask(i=1, j=0)
        self.refreshMaskOutlinePixmap()
        self.refreshMaskPixmap()

    def decreaseBrushSize(self):
        self.brushSize = max(1, self.brushSize - 1)

    def increaseBrushSize(self):
        self.brushSize += 1
