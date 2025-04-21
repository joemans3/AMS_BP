from pathlib import Path

from PyQt6.QtWidgets import (
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QWidget,
)


class OutputConfigWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QFormLayout()

        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("e.g., ./results/")
        layout.addRow("Output Path:", self.output_path)

        self.output_name = QLineEdit()
        self.output_name.setPlaceholderText("e.g., simulation_output")
        layout.addRow("Output Name:", self.output_name)

        self.subsegment_type = QLineEdit()
        self.subsegment_type.setPlaceholderText("Not implemented")
        self.subsegment_type.setDisabled(True)
        layout.addRow("Subsegment Type:", self.subsegment_type)

        self.subsegment_number = QSpinBox()
        self.subsegment_number.setMinimum(0)
        self.subsegment_number.setMaximum(999)
        self.subsegment_number.setDisabled(True)
        layout.addRow("Subsegment Number:", self.subsegment_number)

        self.setLayout(layout)

    def is_valid(self):
        return bool(self.output_path.text().strip()) and bool(
            self.output_name.text().strip()
        )

    def get_data(self):
        return {
            "output_path": self.output_path.text(),
            "output_name": self.output_name.text(),
            "subsegment_type": self.subsegment_type.text(),  # Optional, disabled for now
            "subsegment_number": self.subsegment_number.value(),  # Optional, disabled for now
        }

    def set_data(self, data: dict):
        if "output_path" in data:
            self.output_path.setText(data["output_path"])
        if "output_name" in data:
            self.output_name.setText(data["output_name"])
        if "subsegment_type" in data:
            self.subsegment_type.setText(data["subsegment_type"])
        if "subsegment_number" in data:
            self.subsegment_number.setValue(data["subsegment_number"])

    def get_help_path(self) -> Path:
        return Path(__file__).parent.parent / "help_docs" / "output_help.md"
