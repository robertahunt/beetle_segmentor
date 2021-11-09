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
from PyQt5.QtGui import QImage, QPainter, QPixmap
import qimage2ndarray as q2np


def stratified_species_shuffle(species_df, return_species_df=False):
    sps = species_df.copy()
    sps = sps.sample(frac=1, random_state=42)
    sps["count"] = 1
    species_group = (
        sps.groupby(["species"]).apply(lambda x: x["count"].cumsum()).reset_index()
    )
    sps["species_group"] = species_group.set_index("level_1")["count"]
    sps = sps.sort_values("species_group")
    if return_species_df == True:
        return list(sps["fn"].values), sps
    else:
        return list(sps["fn"].values)


class ImageViewer(QtWidgets.QGraphicsView):
    factor = 2.0

    def __init__(
        self,
        ROOT_FOLDER,
        IMAGE_FOLDER,
        FINAL_MASKS_FOLDER,
        EXISTING_MASKS_FOLDER,
        parent=None,
    ):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.zoomed = 1
        self.parent = parent
        self.ROOT_FOLDER = ROOT_FOLDER
        self.FINAL_SEGS_FOLDER = FINAL_MASKS_FOLDER
        self.maskFilePath = EXISTING_MASKS_FOLDER
        self.IMAGE_FOLDER = IMAGE_FOLDER
        #
        self.files = sorted(os.listdir(self.IMAGE_FOLDER))
        # self.files = [x for x in self.files if int(x.split('/')[-1].split('_')[1]) < 200]
        # self.sps = pd.read_csv(os.path.join(self.ROOT_FOLDER, "bounding_boxes.csv"))
        # species = [
        #    "Rugilus similis",
        #    "Rugilus geniculatus",
        #    "Scopaeus minimus",
        #    "Tetartopeus rufonitidus",
        #    "Lathrobium castaneipenne",
        #    "Lathrobium pallidipenne",
        #    "Lathrobium longulum",
        #    "Neobisnius procerulus",
        #    "Philonthus spinipes",
        #    "Philonthus umbratilis",
        #    "Bisnius nigriventris",
        #    "Philonthus pseudovarians",
        #    "Gabrius piliger",
        #    "Tasgius pedator",
        #    "Tasgius winkleri",
        #    "Platydracus chalcocephalus",
        #    "Quedius invreae",
        #    "Quedius nemoralis",
        # ]
        # self.sps = self.sps[self.sps["species"].map(lambda x: x in species)]
        self.file_no = 0
        # self.files, self.sps = stratified_species_shuffle(
        #    self.sps, return_species_df=True
        # )

        final_segs = [
            os.path.basename(x)
            for x in glob(os.path.join(self.FINAL_SEGS_FOLDER, "*.jpg"))
        ]
        # remove any files that already have final segmentations
        self.files = [x for x in self.files if x not in final_segs]

        self.brushSize = 2

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
        # self.scene.installEventFilter(self)

    def wheelEvent(self, event):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ShiftModifier:
            self.brushSize = max(self.brushSize + int(event.angleDelta().y() / 120), 1)
            self.refreshImage()
            event.accept()

    def get_viewport_pos(self, event):
        pos = event.pos()
        offsets = self.mapToScene(self.viewport().geometry()).boundingRect()

        x = int(offsets.x() + pos.x() / self.zoomed)
        y = int(offsets.y() + pos.y() / self.zoomed)
        return x, y

    def resetManualHistToCursor(self):
        self.manual_backgrounds_hist = self.manual_backgrounds_hist[
            : self.history_cursor
        ]
        self.manual_beetles_hist = self.manual_beetles_hist[: self.history_cursor]

    def setHistToCursor(self):
        self.manual_beetle = self.manual_beetles_hist[self.history_cursor - 1].copy()
        self.manual_background = self.manual_backgrounds_hist[
            self.history_cursor - 1
        ].copy()
        self.refreshImage()

    def eventFilter(self, source, event):
        if event.type() == QEvent.MouseMove:
            x, y = self.get_viewport_pos(event)
            self.cursor = np.array([[0] * self.img_w] * self.img_h)
            if self.parent.side_bar_widget.use_brush.isChecked():
                self.cursor = cv2.circle(
                    self.cursor.astype("uint8"),
                    (x, y),
                    self.brushSize // 2,
                    255,
                    1,
                )
            else:
                if (y < self.segments.shape[0]) & (x < self.segments.shape[1]):
                    segment = self.segments[y, x]
                    self.cursor[np.where(self.segments == segment)] = 255
            self.refreshImage()

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
                self.manual_beetle = cv2.line(
                    self.manual_beetle.astype("uint8"),
                    (int(self.lastPoint.x()), int(self.lastPoint.y())),
                    (int(pos.x()), int(pos.y())),
                    0,
                    self.brushSize,
                )
                self.lastPoint = event.pos()
                self.refreshImage()

            if (event.buttons() & Qt.LeftButton) and self.drawing:
                pos = event.pos()
                self.manual_beetle = cv2.line(
                    self.manual_beetle.astype("uint8"),
                    (int(self.lastPoint.x()), int(self.lastPoint.y())),
                    (int(pos.x()), int(pos.y())),
                    255,
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
                self.refreshImage()

        self.refreshImage()

    def _mouseReleaseEvent(self, event):
        if (event.button() == Qt.LeftButton) or (event.button() == Qt.RightButton):
            self.drawing = False
            self.manual_beetles_hist += [self.manual_beetle.copy()]
            self.manual_backgrounds_hist += [self.manual_background.copy()]
            self.history_cursor = len(self.manual_beetles_hist)
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
            self.mouseReleaseEvent(handmade_event)

    def _mousePressEvent(self, event):
        x = int(event.pos().x())
        y = int(event.pos().y())
        if event.button() == Qt.LeftButton:

            self.resetManualHistToCursor()
            self.drawing = True
            if self.parent.side_bar_widget.use_brush.isChecked():
                self.lastPoint = event.pos()
                self.manual_beetle = cv2.line(
                    self.manual_beetle.astype("uint8"),
                    (int(self.lastPoint.x()), int(self.lastPoint.y())),
                    (int(self.lastPoint.x()), int(self.lastPoint.y())),
                    255,
                    self.brushSize,
                )
                self.manual_background = cv2.line(
                    self.manual_background.astype("uint8"),
                    (int(self.lastPoint.x()), int(self.lastPoint.y())),
                    (int(self.lastPoint.x()), int(self.lastPoint.y())),
                    0,
                    self.brushSize,
                )
            else:
                if (y < self.segments.shape[0]) & (x < self.segments.shape[1]):
                    segment = self.segments[y, x]
                    self.manual_beetle[np.where(self.segments == segment)] = 255
                    self.manual_background[np.where(self.segments == segment)] = 0
        elif event.button() == Qt.RightButton:
            self.resetManualHistToCursor()
            self.drawing = True

            if self.parent.side_bar_widget.use_brush.isChecked():
                self.lastPoint = event.pos()
                self.manual_beetle = cv2.line(
                    self.manual_beetle.astype("uint8"),
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
                if (y < self.segments.shape[0]) & (x < self.segments.shape[1]):
                    segment = self.segments[y, x]
                    self.manual_beetle[np.where(self.segments == segment)] = 0
                    self.manual_background[np.where(self.segments == segment)] = 255
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
            self.mousePressEvent(handmade_event)
        self.refreshImage()

    def load_image(self, filePath, opacity=0.5, mask_threshold=0.5):
        self.filePath = filePath
        self.fileName = os.path.basename(filePath)
        self.parent.side_bar_widget.id_label.setText(self.fileName)
        final_seg_fp = os.path.join(self.FINAL_SEGS_FOLDER, self.fileName)
        if os.path.exists(final_seg_fp):
            self.maskPath = final_seg_fp
            self.parent.side_bar_widget.has_final_seg.setText("Has Final Seg: True")
        else:
            self.maskPath = os.path.join(self.maskFilePath, self.fileName)
            self.parent.side_bar_widget.has_final_seg.setText("Has Final Seg: False")

        self.img = QImage(filePath)
        self.img_w, self.img_h = self.img.size().width(), self.img.size().height()
        self.manual_beetle = np.array([[0] * self.img_w] * self.img_h)
        self.manual_background = np.array([[0] * self.img_w] * self.img_h)

        self.manual_beetles_hist = [self.manual_beetle.copy()]
        self.manual_backgrounds_hist = [self.manual_background.copy()]
        self.history_cursor = 1
        self.cursor = np.array([[0] * self.img_w] * self.img_h)

        if self.img.isNull():
            print("Could not find image, ", filePath)
            return False

        self.mask_shift_i = 0
        self.mask_shift_j = 0
        self.refreshImage(refresh_segments=True)
        self.parent.side_bar_widget.load_img_meta()

        # meta = os.listdir(os.path.join(self.ROOT_FOLDER, "All_meta"))
        # meta = [x for x in meta if x.replace(".json", ".jpg") in self.files]
        self.parent.side_bar_widget.perc_done.setText(
            "Perc Done: %s" % (self.file_no / len(self.files))
        )

        # no_species_left = len(
        #    self.sps.iloc[self.file_no :]["species"].drop_duplicates()
        # )

        # self.parent.side_bar_widget.species_left.setText(
        #    "No. Sps Left: %s" % no_species_left
        # )

        return True

    def load_next_img(self):
        self.file_no = min(self.file_no + 1, len(self.files))

        self.load_image(os.path.join(self.IMAGE_FOLDER, self.files[self.file_no]))

    def load_prev_img(self):
        self.file_no = max(self.file_no - 1, 0)
        self.load_image(os.path.join(self.IMAGE_FOLDER, self.files[self.file_no]))

    def refreshImage(
        self,
        opacity=None,
        mask_threshold=None,
        show_mask=None,
        show_outline=None,
        show_manual_annotations=None,
        use_brush=None,
        superpixel_scale=None,
        superpixel_sigma=None,
        superpixel_compactness=None,
        refresh_segments=False,
    ):
        if opacity is None:
            opacity = self.parent.side_bar_widget.slider_mask_opacity.value() / 100
        if mask_threshold is None:
            mask_threshold = (
                self.parent.side_bar_widget.segmentation_mask_threshold.value() / 100
            )
        if show_mask is None:
            show_mask = self.parent.side_bar_widget.mask_visible.isChecked()
        if show_outline is None:
            show_outline = self.parent.side_bar_widget.outline_visible.isChecked()
        if show_manual_annotations is None:
            show_manual_annotations = (
                self.parent.side_bar_widget.show_manual_annotations.isChecked()
            )
        if use_brush is None:
            use_brush = self.parent.side_bar_widget.use_brush.isChecked()
        if superpixel_scale is None:
            superpixel_scale = self.parent.side_bar_widget.superpixel_scale.value()
        if superpixel_sigma is None:
            superpixel_sigma = (
                self.parent.side_bar_widget.superpixel_sigma.value() / 100
            )
        if superpixel_compactness is None:
            superpixel_compactness = (
                self.parent.side_bar_widget.superpixel_compactness.value()
            )

        self.img = QImage(self.filePath)

        self.img = q2np.byte_view(self.img)[:, :, :3]
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        self._mask = self.load_mask(self.maskPath, mask_threshold)

        if use_brush:
            self.img[np.where(self.cursor == 255)] = np.multiply(
                0.25, self.img[np.where(self.cursor == 255)]
            ) + np.multiply(0.75, [0, 0, 0])
        else:
            gradient = sobel(rgb2gray(self.img))
            # self.segments = watershed(
            #    gradient,
            #    markers=max(10, superpixel_scale),
            #    compactness=superpixel_sigma / 1000,
            # )
            if refresh_segments:
                self.segments = slic(
                    self.img,
                    n_segments=max(1, superpixel_scale),
                    compactness=max(1, superpixel_compactness),
                    sigma=max(0.001, superpixel_sigma),
                    start_label=1,
                )
            self.img = np.multiply(0.75, self.img) + np.multiply(
                0.25,
                (mark_boundaries(self.img, self.segments, mode="thin") * 255).astype(
                    "uint8"
                ),
            )

            self.img[np.where(self.cursor == 255)] = np.multiply(
                0.5, self.img[np.where(self.cursor == 255)]
            ) + np.multiply(0.5, [160, 0, 160])

        if show_outline:
            contours, heirarchy = cv2.findContours(
                self._mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(self.img, contours, -1, (0, 0, 255), 1)

        if show_manual_annotations:
            self.img[np.where(self.manual_beetle == 255)] = np.multiply(
                0.5, self.img[np.where(self.manual_beetle == 255)]
            ) + np.multiply(0.5, [255, 0, 0])
            self.img[np.where(self.manual_background == 255)] = np.multiply(
                0.5, self.img[np.where(self.manual_background == 255)]
            ) + np.multiply(0.5, [0, 255, 0])

        self.img = q2np.array2qimage(self.img)
        if show_mask:
            mask = q2np.array2qimage(self._mask)
            painter = QPainter()
            painter.begin(self.img)
            painter.setOpacity(opacity)
            painter.drawImage(0, 0, mask)
            painter.end()

        self._pixmap_item.setPixmap(QPixmap.fromImage(self.img))

    def load_mask(self, mask_path, threshold):
        mask = QImage(mask_path)

        if mask.isNull():
            print("Could not find mask, ", self.maskPath)
            mask = np.zeros(self.img.shape[:2]).astype("uint8")
        else:
            mask = q2np.byte_view(mask)
            mask = ((mask > (255 * threshold)) * 255).astype("uint8")

        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        if mask.shape != self.img.shape[:2]:
            new_mask = np.zeros(self.img.shape[:2])
            w = min(self.img.shape[0], mask.shape[0])
            h = min(self.img.shape[1], mask.shape[1])
            new_mask[:w, :h] = mask[:w, :h]
            mask = new_mask.astype("uint8")

        assert mask.shape == self.img.shape[:2]

        mask = np.roll(mask, self.mask_shift_i, axis=0)
        mask = np.roll(mask, self.mask_shift_j, axis=1)

        mask[np.where(self.manual_beetle == 255)] = 255
        mask[np.where(self.manual_background == 255)] = 0

        return mask

    def save_mask(self):
        fn = self.files[self.file_no]
        fp = os.path.join(self.FINAL_SEGS_FOLDER, fn)
        cv2.imwrite(fp, self._mask)

    def shiftMask(self, i=0, j=0):
        self.mask_shift_i += i
        self.mask_shift_j += j

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

    def changeOpacity(self, value):
        self.refreshImage(opacity=value / 100)

    def changeMaskThreshold(self, value):
        self.refreshImage(mask_threshold=value / 100)

    def toggleMaskVisibility(self, value):
        self.refreshImage(show_mask=value)

    def toggleOutlineVisibility(self, value):
        self.refreshImage(show_outline=value)

    def toggleManualAnnotations(self, value):
        self.refreshImage(show_manual_annotations=value)

    def toggleBrush(self, value):
        self.refreshImage(use_brush=value, refresh_segments=True)

    def changeSuperpixelScale(self, value):
        self.refreshImage(superpixel_scale=value, refresh_segments=True)
        self.parent.side_bar_widget.scale_label.setText(
            "Superpixel Scale: %s"
            % self.parent.side_bar_widget.superpixel_scale.value()
        )

    def changeSuperpixelSigma(self, value):
        self.refreshImage(superpixel_sigma=value / 25, refresh_segments=True)
        self.parent.side_bar_widget.sigma_label.setText(
            "Superpixel Sigma: %s / 25"
            % self.parent.side_bar_widget.superpixel_sigma.value()
        )

    def changeSuperpixelCompactness(self, value):
        self.refreshImage(superpixel_compactness=value, refresh_segments=True)
        self.parent.side_bar_widget.compact_label.setText(
            "Superpixel Compactness: %s"
            % self.parent.side_bar_widget.superpixel_compactness.value()
        )

    def shiftMaskUp(self):
        self.shiftMask(i=-1, j=0)
        self.refreshImage()

    def shiftMaskLeft(self):
        self.shiftMask(i=0, j=-1)
        self.refreshImage()

    def shiftMaskRight(self):
        self.shiftMask(i=0, j=1)
        self.refreshImage()

    def shiftMaskDown(self):
        self.shiftMask(i=1, j=0)
        self.refreshImage()

    def decreaseBrushSize(self):
        self.brushSize = max(1, self.brushSize - 1)

    def increaseBrushSize(self):
        self.brushSize += 1
