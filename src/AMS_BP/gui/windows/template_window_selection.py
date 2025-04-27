import json
from pathlib import Path

import tomli
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .configuration_window import ConfigEditor

TEMPLATE_DIR = Path(__file__).parent.parent.parent / "resources" / "template_configs"
METADATA_PATH = TEMPLATE_DIR / "metadata_configs.json"


class TemplateSelectionWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Choose a Simulation Template")
        self.setMinimumSize(700, 500)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("<h2>Select a Template to Begin</h2>"))

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)

        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            for key, entry in metadata.items():
                card = self.create_template_card(entry)
                content_layout.addWidget(card)
        except Exception as e:
            error_label = QLabel(f"Failed to load templates: {e}")
            layout.addWidget(error_label)

        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)
        self.setLayout(layout)

    def create_template_card(self, entry: dict) -> QWidget:
        group = QGroupBox(entry["label"])
        group.setMinimumHeight(150)
        group.setMaximumHeight(200)
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(15)

        # Image
        img_label = QLabel()
        img_path = TEMPLATE_DIR / entry["image"]
        if img_path.exists():
            pixmap = QPixmap(str(img_path)).scaled(
                150,
                150,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            img_label.setPixmap(pixmap)
        else:
            img_label.setText("[Missing image assets]")
        img_label.setFixedSize(150, 150)
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(img_label)

        # Description + Button (in a VBox)
        vbox = QVBoxLayout()
        vbox.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Scrollable description
        desc_scroll = QScrollArea()
        desc_scroll.setFixedHeight(100)  # Height for the scroll area
        desc_scroll.setWidgetResizable(True)

        desc_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        desc_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        description_label = QLabel(entry.get("description", ""))
        description_label.setWordWrap(True)
        description_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        description_label.setContentsMargins(5, 5, 5, 5)
        description_label.setSizePolicy(
            description_label.sizePolicy().horizontalPolicy(),
            QSizePolicy.Policy.Maximum,
        )

        desc_scroll.setWidget(description_label)
        vbox.addWidget(desc_scroll)

        btn = QPushButton("Use This Template")
        btn.clicked.connect(
            lambda _, config=entry["config"]: self.load_template(config)
        )
        vbox.addWidget(btn)

        layout.addLayout(vbox)

        group.setLayout(layout)
        return group

    def load_template(self, config_filename: str):
        config_path = TEMPLATE_DIR / config_filename
        try:
            with open(config_path, "rb") as f:
                config = tomli.load(f)

            self.launch_config_editor(config)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load template:\n{e}")

    def launch_config_editor(self, config_dict: dict):
        self.editor = ConfigEditor()
        self.editor.set_data(config_dict)
        self.editor.show()
        self.close()
