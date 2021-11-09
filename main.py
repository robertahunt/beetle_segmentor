from PyQt5 import QtWidgets, QtGui, QtCore
import os

import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QApplication,
)
from PyQt5.QtCore import Qt, QEvent
from sidebar import SideBarWidget
from image_viewer import ImageViewer

# name of the masks, if already using preexisting needs to be the same name as the images
ROOT_FOLDER = "/mnt/newterra/share/04. Seafile/03. Exploration/02. Create Beetle Dataset/test"
IMAGE_FOLDER = os.path.join(ROOT_FOLDER, "image_folder")
EXISTING_MASKS_FOLDER = os.path.join(ROOT_FOLDER, "mask_folder")
FINAL_MASKS_FOLDER = os.path.join(ROOT_FOLDER, "final_mask_folder")

assert os.path.exists(ROOT_FOLDER)
assert os.path.exists(IMAGE_FOLDER)
assert os.path.exists(EXISTING_MASKS_FOLDER)
assert os.path.exists(FINAL_MASKS_FOLDER)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._title = (
            "Robertas Super cool Segmenter that is totally not a waste of time"
        )
        self.setWindowTitle(self._title)
        self._main = QWidget()
        self.setCentralWidget(self._main)
        self.setMouseTracking(True)
        main_layout = QHBoxLayout(self._main)

        self.view = ImageViewer(parent=self, IMAGE_FOLDER=IMAGE_FOLDER,ROOT_FOLDER=ROOT_FOLDER, FINAL_MASKS_FOLDER=FINAL_MASKS_FOLDER, EXISTING_MASKS_FOLDER=EXISTING_MASKS_FOLDER)
        main_layout.addWidget(self.view)

        self.side_bar_widget = SideBarWidget(self)
        main_layout.addWidget(self.side_bar_widget)

        self.createActions()
        self.createMenus()

        self.resize(1640, 1480)

        self.installEventFilter(self)
        self.updateActions()
        self.view.load_image(os.path.join(self.view.IMAGE_FOLDER, self.view.files[0]))

        self.view.zoomIn()

    def eventFilter(self, source, event):
        if event.type() == QEvent.KeyPress:  # QKeySequence(Qt.CTRL + Qt.Key_N):
            modifiers = QApplication.keyboardModifiers()
            if event.key() == Qt.Key_L:
                self.side_bar_widget.save_img_meta()
                self.view.save_mask()
                self.view.load_next_img()
            if event.key() == Qt.Key_J:
                self.side_bar_widget.save_img_meta()
                self.view.save_mask()
                self.view.load_prev_img()
            if event.key() == Qt.Key_1:
                self.side_bar_widget.okay_for_study.toggle()
            if event.key() == Qt.Key_2:
                self.side_bar_widget.genitalia_separated.toggle()
            if event.key() == Qt.Key_3:
                self.side_bar_widget.missing_body_parts.toggle()
            if event.key() == Qt.Key_4:
                self.side_bar_widget.body_parts_separated_from_body.toggle()
            if event.key() == Qt.Key_5:
                self.side_bar_widget.standard_pose.toggle()
            if event.key() == Qt.Key_6:
                self.side_bar_widget.good_segmentation.toggle()
            if event.key() == Qt.Key_7:
                self.side_bar_widget.bad_bounding_box.toggle()
            if event.key() == Qt.Key_I:
                self.side_bar_widget.mask_visible.toggle()
            if event.key() == Qt.Key_K:
                self.side_bar_widget.outline_visible.toggle()
            if event.key() == Qt.Key_Comma:
                self.side_bar_widget.show_manual_annotations.toggle()
            if event.key() == Qt.Key_M:
                self.side_bar_widget.use_brush.toggle()
            if (
                (event.key() == Qt.Key_Z)
                and (modifiers & Qt.ControlModifier)
                and not (modifiers & Qt.ShiftModifier)
            ):
                # undo last manual annotation
                if self.view.history_cursor == 0:
                    print("Cannot undo... nothing to undo")
                else:
                    self.view.history_cursor -= 1
                self.view.setHistToCursor()

            if (
                (event.key() == Qt.Key_Z)
                and (modifiers & Qt.ControlModifier)
                and (modifiers & Qt.ShiftModifier)
            ):
                # redo manual annotation

                if self.view.history_cursor == len(self.view.manual_beetles_hist):
                    print("Cannot redo... nothing to redo")
                else:
                    self.view.history_cursor += 1
                print(self.view.history_cursor)
                self.view.setHistToCursor()

        return super(MainWindow, self).eventFilter(source, event)

    def open(self):
        image_formats = " ".join(
            [
                "*." + image_format.data().decode()
                for image_format in QtGui.QImageReader.supportedImageFormats()
            ]
        )

        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Open Image"),
            QtCore.QDir.currentPath(),
            self.tr("Image Files({})".format(image_formats)),
        )
        if fileName:
            is_loaded = self.view.load_image(fileName)
            self.fitToWindowAct.setEnabled(is_loaded)
            self.updateActions()

    def fitToWindow(self):
        if self.fitToWindowAct.isChecked():
            self.view.fitToWindow()
        else:
            self.view.resetZoom()
        self.updateActions()

    def about(self):
        QtWidgets.QMessageBox.about(
            self,
            "ImageViewer",
            "ImageViewer",
        )

    def createActions(self):
        self.openAct = QtWidgets.QAction(
            "&Open...", self, shortcut="Ctrl+O", triggered=self.open
        )
        self.exitAct = QtWidgets.QAction(
            "E&xit", self, shortcut="Ctrl+Q", triggered=self.close
        )
        self.zoomInAct = QtWidgets.QAction(
            self.tr("Zoom &In (25%)"),
            self,
            shortcut="Ctrl++",
            enabled=False,
            triggered=self.view.zoomIn,
        )
        self.zoomOutAct = QtWidgets.QAction(
            self.tr("Zoom &Out (25%)"),
            self,
            shortcut="Ctrl+-",
            enabled=False,
            triggered=self.view.zoomOut,
        )
        self.normalSizeAct = QtWidgets.QAction(
            self.tr("&Normal Size"),
            self,
            shortcut="Ctrl+S",
            enabled=False,
            triggered=self.view.resetZoom,
        )
        self.fitToWindowAct = QtWidgets.QAction(
            self.tr("&Fit to Window"),
            self,
            enabled=False,
            checkable=True,
            shortcut="Ctrl+F",
            triggered=self.fitToWindow,
        )
        self.aboutAct = QtWidgets.QAction(self.tr("&About"), self, triggered=self.about)
        self.aboutQtAct = QtWidgets.QAction(
            self.tr("About &Qt"), self, triggered=QtWidgets.QApplication.aboutQt
        )

    def createMenus(self):
        self.fileMenu = QtWidgets.QMenu(self.tr("&File"), self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QtWidgets.QMenu(self.tr("&View"), self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.helpMenu = QtWidgets.QMenu(self.tr("&Help"), self)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
