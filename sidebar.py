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

        superpixel_scale = QSlider(Qt.Horizontal)
        superpixel_scale.setFocusPolicy(Qt.StrongFocus)
        superpixel_scale.setTickPosition(QSlider.TicksBothSides)
        superpixel_scale.setMaximum(1000)
        superpixel_scale.setMinimum(0)
        superpixel_scale.setSingleStep(5)
        superpixel_scale.setValue(400)
        self.superpixel_scale = superpixel_scale
        self.scale_label = QLabel(
            "Superpixel No. Segments: %s" % self.superpixel_scale.value()
        )
        layout.addWidget(self.scale_label)
        layout.addWidget(self.superpixel_scale)
        self.superpixel_scale.valueChanged.connect(
            self.parent.view.changeSuperpixelScale
        )

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

        label = QLabel("Mask Threshold:")
        layout.addWidget(label)

        slider = QSlider(Qt.Horizontal)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QSlider.TicksBothSides)
        slider.setMaximum(100)
        slider.setMinimum(0)
        slider.setSingleStep(5)
        slider.setValue(50)
        self.segmentation_mask_threshold = slider
        layout.addWidget(self.segmentation_mask_threshold)
        self.segmentation_mask_threshold.valueChanged.connect(
            self.parent.view.changeMaskThreshold
        )

        label = QLabel("Rotation:")
        layout.addWidget(label)
        self.rotation = QLineEdit()
        layout.addWidget(self.rotation)
        self.rotation.setText("0")

        self.okay_for_study = QCheckBox("1 Okay for my study")
        layout.addWidget(self.okay_for_study)

        self.genitalia_separated = QCheckBox("2 Genitalia Separated")
        layout.addWidget(self.genitalia_separated)

        self.missing_body_parts = QCheckBox("3 Missing Body Parts")
        layout.addWidget(self.missing_body_parts)

        self.body_parts_separated_from_body = QCheckBox(
            "4 Body parts separated from body"
        )
        layout.addWidget(self.body_parts_separated_from_body)

        self.standard_pose = QCheckBox("5 Standard Pose")
        layout.addWidget(self.standard_pose)

        self.good_segmentation = QCheckBox("6 Good Segmentation")
        layout.addWidget(self.good_segmentation)

        self.bad_bounding_box = QCheckBox("7 Bad Bounding Box")
        layout.addWidget(self.bad_bounding_box)

        label = QLabel("Orientation:")
        layout.addWidget(label)
        self.orientation = QComboBox()
        self.orientation.addItems(["dorsal", "ventral", "other"])
        layout.addWidget(self.orientation)

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
        img_meta["okay_for_study"] = self.okay_for_study.isChecked()
        img_meta["genitalia_separated"] = self.genitalia_separated.isChecked()
        img_meta[
            "segmentation_mask_threshold"
        ] = self.segmentation_mask_threshold.value()
        img_meta["bad_bounding_box"] = self.bad_bounding_box.isChecked()
        img_meta["missing_body_parts"] = self.missing_body_parts.isChecked()
        img_meta[
            "body_parts_separated_from_body"
        ] = self.body_parts_separated_from_body.isChecked()
        img_meta["orientation"] = self.orientation.currentText()
        img_meta["standard_pose"] = self.standard_pose.isChecked()
        img_meta["good_segmentation"] = self.good_segmentation.isChecked()

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
        self.get_or_set_meta(img_meta, "okay_for_study", False)
        self.get_or_set_meta(img_meta, "missing_body_parts", False)
        self.get_or_set_meta(img_meta, "body_parts_separated_from_body", False)
        self.get_or_set_meta(img_meta, "orientation", "dorsal")
        self.get_or_set_meta(img_meta, "standard_pose", True)
        self.get_or_set_meta(img_meta, "good_segmentation", True)
        self.get_or_set_meta(img_meta, "genitalia_separated", False)
        self.get_or_set_meta(img_meta, "segmentation_mask_threshold", 50)
        self.get_or_set_meta(img_meta, "bad_bounding_box", False)
