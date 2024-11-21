import os
import json
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QSlider,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QPushButton,
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from PyQt5.QtGui import QIntValidator




class SideBarWidget(QWidget):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.parent = parent
        self.META_FOLDER = os.path.join(self.parent.view.ROOT_FOLDER, "03_meta")
        if not os.path.exists(self.META_FOLDER):
            os.mkdir(self.META_FOLDER)
        self.setFixedWidth(200)
        layout = QVBoxLayout(self)

        self.id_label = QLabel(f"Id:")
        layout.addWidget(self.id_label)

        self.perc_done = QLabel(f"Perc Done:")
        layout.addWidget(self.perc_done)

        self.has_final_seg = QLabel(f"Has Final Seg:")
        layout.addWidget(self.has_final_seg)

        self.species_left = QLabel(f"Commands: J and L to go between files.\nCtrl+/- to zoom in or out\n Mousewheel to increase brush size")
        layout.addWidget(self.species_left)

        brush_size_increase = QPushButton("Increase Brush Size", self)
        brush_size_increase.clicked.connect(self.parent.view.increaseBrushSize)
        layout.addWidget(brush_size_increase)

        brush_size_decrease = QPushButton("Decrease Brush Size", self)
        brush_size_decrease.clicked.connect(self.parent.view.decreaseBrushSize)
        layout.addWidget(brush_size_decrease)

        # up_button = QPushButton("^", self)
        # up_button.clicked.connect(self.parent.view.shiftMaskUp)
        # layout.addWidget(up_button)

        # left_button = QPushButton("<", self)
        # left_button.clicked.connect(self.parent.view.shiftMaskLeft)
        # layout.addWidget(left_button)

        # right_button = QPushButton(">", self)
        # right_button.clicked.connect(self.parent.view.shiftMaskRight)
        # layout.addWidget(right_button)

        # down_button = QPushButton("_", self)
        # down_button.clicked.connect(self.parent.view.shiftMaskDown)
        # layout.addWidget(down_button)

        current_class = QComboBox()
        current_class.addItems(list(map(str, range(1,self.parent.view.NUM_CLASSES))))
        current_class.setCurrentIndex(0)
        self.current_class = current_class
        self.current_class_label = QLabel(
            "Current Annotation Class: %s" % self.current_class.currentText() #.value*
        )
        layout.addWidget(self.current_class_label)
        layout.addWidget(self.current_class)
        self.current_class.currentIndexChanged.connect(self.parent.view.changeCurrentClass)

        label = QLabel("Mask Opacity:")
        layout.addWidget(label)

        slider = QSlider(Qt.Horizontal)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QSlider.TicksBothSides)
        slider.setMaximum(100)
        slider.setMinimum(0)
        slider.setSingleStep(5)
        slider.setValue(50)
        self.slider_mask_opacity = slider
        layout.addWidget(self.slider_mask_opacity)
        self.slider_mask_opacity.valueChanged.connect(self.parent.view.changeOpacity)

        self.mask_visible = QCheckBox("Show Mask")
        self.mask_visible.setChecked(True)
        self.mask_visible.toggled.connect(self.parent.view.toggleMaskVisibility)
        layout.addWidget(self.mask_visible)

        self.outline_visible = QCheckBox("Show Outline")
        self.outline_visible.setChecked(True)
        self.outline_visible.toggled.connect(self.parent.view.toggleOutlineVisibility)
        layout.addWidget(self.outline_visible)

        self.show_manual_annotations = QCheckBox("Show Annotations")
        self.show_manual_annotations.setChecked(True)
        self.show_manual_annotations.toggled.connect(
            self.parent.view.toggleManualAnnotations
        )
        layout.addWidget(self.show_manual_annotations)

        self.use_brush = QCheckBox("Brush instead of Superpixel Fill")
        self.use_brush.setChecked(False)
        self.use_brush.toggled.connect(self.parent.view.toggleBrush)
        layout.addWidget(self.use_brush)

        self.zoom_to_mask = QCheckBox("Zoom to Mask?")
        self.zoom_to_mask.setChecked(False)
        self.zoom_to_mask.toggled.connect(self.parent.view.zoomToMask)
        layout.addWidget(self.zoom_to_mask)

        superpixel_scale = QComboBox()
        superpixel_scale.addItems(['2','4','8','16','32','64','128','256','512','1024','2048'])
        superpixel_scale.setCurrentIndex(6)

        self.superpixel_scale = superpixel_scale
        self.scale_label = QLabel(
            "Superpixel No. Segments: %s" % self.superpixel_scale.currentText() #.value*
        )
        layout.addWidget(self.scale_label)
        layout.addWidget(self.superpixel_scale)
        self.superpixel_scale.currentIndexChanged.connect(self.parent.view.changeSuperpixelScale)

        superpixel_compactness = QSlider(Qt.Horizontal)
        superpixel_compactness.setFocusPolicy(Qt.StrongFocus)
        superpixel_compactness.setTickPosition(QSlider.TicksBothSides)
        superpixel_compactness.setMaximum(100)
        superpixel_compactness.setMinimum(0)
        superpixel_compactness.setSingleStep(5)
        superpixel_compactness.setValue(15)
        self.superpixel_compactness = superpixel_compactness
        self.compact_label = QLabel(
            "Superpixel Compactness: %s" % self.superpixel_compactness.value()
        )
        layout.addWidget(self.compact_label)
        layout.addWidget(self.superpixel_compactness)
        self.superpixel_compactness.valueChanged.connect(
            self.parent.view.changeSuperpixelCompactness
        )

        superpixel_sigma = QSlider(Qt.Horizontal)
        superpixel_sigma.setFocusPolicy(Qt.StrongFocus)
        superpixel_sigma.setTickPosition(QSlider.TicksBothSides)
        superpixel_sigma.setMaximum(100)
        superpixel_sigma.setMinimum(0)
        superpixel_sigma.setSingleStep(5)
        superpixel_sigma.setValue(50)
        self.superpixel_sigma = superpixel_sigma
        self.sigma_label = QLabel("Superpixel Sigma: %s" % superpixel_sigma.value())
        layout.addWidget(self.sigma_label)
        layout.addWidget(self.superpixel_sigma)
        self.superpixel_sigma.valueChanged.connect(
            self.parent.view.changeSuperpixelSigma
        )


        slider = QSlider(Qt.Horizontal)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QSlider.TicksBothSides)
        slider.setMaximum(100)
        slider.setMinimum(0)
        slider.setSingleStep(5)
        slider.setValue(50)

        label = QLabel("Rotation:")
        layout.addWidget(label)
        self.rotation = QLineEdit()
        layout.addWidget(self.rotation)
        self.rotation.setText("0")



    def get_or_set_meta(self, img_meta, key, default_value):
        if img_meta.get(key, None) is None:
            img_meta[key] = default_value
            exec(
                f"self.{key}.setStyleSheet('QCheckBox' '{{' 'background : yellow;' '}}')"
            )
        else:
            exec(f"self.{key}.setStyleSheet('QCheckBox' '{{' '' '}}')")
        exec(f"global kind; kind = self.{key}.__class__.__name__")
        val = img_meta[key]

        if kind == "QCheckBox":
            exec(f"self.{key}.setChecked(%s)" % img_meta[key])
        elif kind == "QLineEdit":
            exec(f"self.{key}.setText('{val}')")
        elif kind == "QComboBox":
            exec(f"self.{key}.setCurrentText('%s')" % img_meta[key])
        elif kind == "QSlider":
            exec(f"self.{key}.setValue(%s)" % img_meta[key])
        else:
            wtf

    def save_img_meta(self):
        fp = os.path.join(
            self.META_FOLDER,
            self.parent.view.files[self.parent.view.file_no].replace("jpg", "json")
        )
        if os.path.exists(fp):
            with open(fp, "r") as f:
                img_meta = json.load(f)
        else:
            img_meta = {}
        img_meta["rotation"] = self.rotation.text()

        json_formatted_str = json.dumps(img_meta, indent=4)

        with open(fp, "w") as f:
            f.write(json_formatted_str)

    def load_img_meta(self):
        fp = os.path.join(
            self.META_FOLDER
            , self.parent.view.files[self.parent.view.file_no].replace("jpg", "json")
        )
        if not os.path.exists(fp):
            img_meta = {}
        else:
            with open(fp, "r") as f:
                img_meta = json.load(f)
        self.get_or_set_meta(img_meta, "rotation", "0")
