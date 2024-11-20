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
ROOT_FOLDER = "/mnt/newterra/Dropbox/05. 2024 Postdoc/04. ACCESS 2024/2_live_bees"
IMAGE_FOLDER = os.path.join(ROOT_FOLDER, "6_cropped_and_flipped")
EXISTING_MASKS_FOLDER = os.path.join(ROOT_FOLDER, "9_UNet_predictions_resized")
FINAL_MASKS_FOLDER = os.path.join(ROOT_FOLDER, "9_UNet_predictions_resized_manual_corrections")
SUBSET = [
'Hive01_Sheet_02_slideww43_left.png',
 'Hive01_Sheet_02_slideww43_right.png',
 'Hive01_Sheet_03_slideww21-x_left.png',
 'Hive02_Sheet_01_slide23_left.png',
 'Hive02_Sheet_01_slide45_right.png',
 'Hive02_Sheet_02_slide103_left.png',
 'Hive02_Sheet_02_slide105_right.png',
 'Hive02_Sheet_03_slideww47_left.png',
 'Hive02_Sheet_03_slideww70_left.png',
 'Hive03_Sheet_01_slide115_left.png',
 'Hive03_Sheet_01_slideww45_right.png',
 'Hive04_Sheet_01_slide38_right.png',
 'Hive04_Sheet_01_slideww28_left.png',
 'Hive04_Sheet_02_slide38_right.png',
 'Hive04_Sheet_02_slideww28_left.png',
 'Hive05_Sheet_01_slideww82_left.png',
 'Hive06_Sheet_02_slide62_right.png',
 'Hive06_Sheet_03_slide115_left.png',
 'Hive06_Sheet_03_slide80_left.png',
 'Hive06_Sheet_04_slide125_right.png',
 'Hive07_Sheet_01_slide33_left.png',
 'Hive07_Sheet_04_slide120_left.png',
 'Hive07_Sheet_04_slide122_left.png',
 'Hive07_Sheet_04_slide129_left.png',
 'Hive07_Sheet_05_slide160_left.png',
 'Hive07_Sheet_05_slide162_left.png',
 'Hive07_Sheet_05_slide71_left.png',
 'Hive08_Sheet_01_slide20_left.png',
 'Hive08_Sheet_01_slide22_left.png',
 'Hive08_Sheet_02_slide72_left.png',
 'Hive08_Sheet_03_slide124_left.png',
 'Hive08_Sheet_04_slide138_left.png',
 'Hive08_Sheet_04_slide148_left.png',
 'Hive08_Sheet_05_slide74_left.png',
 'Hive08_Sheet_06_slide133_left.png',
 'Hive08_Sheet_06_slide77_right.png',
 'Hive09_Sheet_03_slide104_right.png',
 'Hive09_Sheet_03_slide78_right.png',
 'Hive09_Sheet_03_slide87_left.png',
 'Hive09_Sheet_03_slide87_right.png',
 'Hive09_Sheet_03_slide95_left.png',
 'Hive09_Sheet_05_slide161_right.png',
 'Hive10_Sheet_04_slide124_left.png',
 'Hive10_Sheet_05_slide146_left.png',
 'Hive12_Sheet_01_slide14_left.png',
 'Hive12_Sheet_02_slideww28_right.png',
 'Hive13_Sheet_02_slideww24_left.png',
 'Hive13_Sheet_02_slideww32_left.png',
 'Hive13_Sheet_02_slideww32_right.png',
 'Hive17_Sheet_01_slide32_left.png',
 'Hive17_Sheet_01_slide3_left.png',
 'Hive18_Sheet_02_slide32_right.png',
 'Hive19_Sheet_02_slide48_right.png',
 'Hive19_Sheet_02_slide55_right.png',
 'Hive19_Sheet_03_slide43_left.png',
 'Hive19_Sheet_03_slide43_right.png',
 'Hive19_Sheet_03_slide64_left.png',
 'Hive19_Sheet_03_slide70_right.png',
 'Hive19_Sheet_03_slide73_right.png',
 'Hive20_Sheet_01_slide28_left.png',
 'Hive20_Sheet_02_slide47_right.png',
 'Hive20_Sheet_02_slide52_right.png',
 'Hive20_Sheet_02_slide59_right.png',
 'Hive20_Sheet_03_slide78_left.png',
 'Hive21_Sheet_01_slide12_right.png',
 'Hive21_Sheet_01_slideww04_left.png',
 'Hive22_Sheet_01_slide32_left.png',
 'Hive22_Sheet_01_slide45_right.png',
 'Hive22_Sheet_01_slideww32_left.png',
 'Hive22_Sheet_01_slideww32_right.png',
 'Hive23_Sheet_01_slide23_right.png',
 'Hive23_Sheet_01_slideww10_left.png',
 'Hive23_Sheet_01_slideww10_right.png',
 'Hive24_Sheet_01_slide25_right.png',
 'Hive24_Sheet_01_slide27_left.png',
 'Hive24_Sheet_01_slide3_right.png',
 'Hive24_Sheet_01_slideww18_left.png',
 'Hive24_Sheet_01_slideww18_right.png',
 'Hive25_Sheet_01_slide30_left.png',
 'Hive25_Sheet_01_slide51_right.png',
 'Hive25_Sheet_01_slide60_right.png',
 'Hive25_Sheet_01_slideww27_left.png',
 'Hive26_Sheet_02_slide31_left.png',
 'Hive26_Sheet_02_slide44_left.png',
 'Hive26_Sheet_03_slide93_left.png',
 'Hive27_Sheet_01_slide21_left.png',
 'Hive27_Sheet_02_slide84_left.png',
 'Hive27_Sheet_02_slide84_right.png',
 'Hive28_Sheet_03_slide65_left.png',
 'Hive28_Sheet_03_slide65_right.png',
 'Hive28_Sheet_03_slide72_right.png',
 'Hive28_Sheet_03_slide75_left.png',
 'Hive29_Sheet_02_slide48_right.png',
 'Hive29_Sheet_02_slide52_left.png',
 'Hive29_Sheet_02_slide52_right.png',
 'Hive30_Sheet_02_slide58_left.png',
 'Hive30_Sheet_02_slide70_left.png',
 'Hive30_Sheet_02_slide70_right.png',
 'Hive31_Sheet_01_slide51_left.png',
 'Hive31_Sheet_01_slide51_right.png',
 'Hive32_Sheet_01_slide22_left.png',
 'Hive32_Sheet_01_slide35_left.png',
 'Hive32_Sheet_01_slide44_left.png',
 'Hive32_Sheet_01_slide44_right.png',
 'Hive32_Sheet_01_slide57_right.png',
 'Hive32_Sheet_01_slide74_right.png',
 'Hive32_Sheet_01_slideww11_right.png',
 'Hive32_Sheet_01_slideww13_left.png',
 'Hive32_Sheet_01_slideww13_right.png',
 'Hive32_Sheet_01_slideww18_left.png',
 'Hive32_Sheet_01_slideww18_right.png',
 'Hive32_Sheet_01_slideww1_left.png',
 'Hive32_Sheet_01_slideww26_right.png',
 'Hive32_Sheet_01_slideww44_left.png',
 'Hive33_Sheet_01_slideww3_left.png',
 'Hive33_Sheet_01_slideww49_left.png',
 'Hive33_Sheet_01_slideww49_right.png',
 'Hive34_Sheet_01_slide18_right.png',
 'Hive34_Sheet_01_slide27_left.png',
 'Hive34_Sheet_01_slide27_right.png',
 'Hive34_Sheet_01_slideww53_left.png',
 'Hive34_Sheet_01_slideww53_right.png',
 'Hive36_Sheet_01_slide32_right.png',
 'Hive36_Sheet_02_slide62_left.png',
 'Hive36_Sheet_02_slide67_left.png',
 'Hive36_Sheet_03_slide75_left.png',
 'Hive36_Sheet_04_slide140_left.png',
 'Hive37_Sheet_01_slide2_left.png',
 'Hive37_Sheet_02_slide40_right.png',
 'Hive38_Sheet_01_slide16_right.png',
 'Hive38_Sheet_01_slide24_left.png',
 'Hive38_Sheet_01_slide35_right.png',
 'Hive38_Sheet_02_slide75_right.png',
 'Hive38_Sheet_02_slide78_right.png',
 'Hive39_Sheet_02_slide51_right.png',
 'Hive39_Sheet_02_slide59_left.png',
 'Hive39_Sheet_03_slide92_right.png',
 'Hive40_Sheet_02_slide67_left.png',
 'Hive40_Sheet_03_slide84_left.png',
 'Hive40_Sheet_04_slide121_left.png',
 'Hive40_Sheet_04_slideww40_left.png',
 'HiveC1_Sheet_01_slide24_right.png',
 'HiveC1_Sheet_02_slide43_left.png',
 'HiveC1_Sheet_02_slide45_left.png',
 'HiveC1_Sheet_02_slide45_right.png',
 'HiveC2_Sheet_01_slide16_left.png',
 'HiveC2_Sheet_01_slide7_right.png',
 'HiveC2_Sheet_02_slide25_left.png',
 'HiveC2_Sheet_02_slide29_right.png',
 'HiveC2_Sheet_02_slide43_right.png',
 'HiveC2_Sheet_02_slide46_right.png',
 'HiveC4_Sheet_01_slide27_left.png',
 'HiveC4_Sheet_01_slide44_left.png',
 'HiveC5_Sheet_02_slide43_right.png',
 'HiveC5_Sheet_02_slide44_right.png'
 ] 
NUM_CLASSES = 7



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

        self.view = ImageViewer(parent=self, IMAGE_FOLDER=IMAGE_FOLDER,
                                ROOT_FOLDER=ROOT_FOLDER, 
                                FINAL_MASKS_FOLDER=FINAL_MASKS_FOLDER, 
                                EXISTING_MASKS_FOLDER=EXISTING_MASKS_FOLDER, 
                                SUBSET=SUBSET,
                                NUM_CLASSES=NUM_CLASSES)
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

                if self.view.history_cursor == len(self.view.manual_foregrounds_hist):
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
