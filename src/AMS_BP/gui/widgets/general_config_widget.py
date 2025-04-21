import re
from pathlib import Path

from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLineEdit,
    QWidget,
)


class GeneralConfigWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QFormLayout()

        self.version = QLineEdit()
        self.version.setPlaceholderText("e.g., 1.0")
        layout.addRow("Version:", self.version)

        self.length_unit = QComboBox()
        self.length_unit.addItems(["um"])
        layout.addRow("Length Unit:", self.length_unit)

        self.time_unit = QComboBox()
        self.time_unit.addItems(["ms"])
        layout.addRow("Time Unit:", self.time_unit)

        self.diffusion_unit = QComboBox()
        self.diffusion_unit.addItems(["um^2/s"])
        layout.addRow("Diffusion Unit:", self.diffusion_unit)

        self.setLayout(layout)

    def is_valid(self):
        return bool(re.match(r"^\d+\.\d+$", self.version.text()))

    def get_data(self):
        return {
            "version": self.version.text(),
            "length_unit": self.length_unit.currentText(),
            "time_unit": self.time_unit.currentText(),
            "diffusion_unit": self.diffusion_unit.currentText(),
        }

    def get_help_path(self) -> Path:
        return Path(__file__).parent.parent / "help_docs" / "general_help.md"
