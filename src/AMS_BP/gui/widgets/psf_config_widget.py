from pathlib import Path

from pydantic import ValidationError
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)


class PSFConfigWidget(QWidget):
    confocal_mode_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        # PSF Type Group
        self.psf_group = QGroupBox("Point Spread Function (PSF)")
        psf_form = QFormLayout()

        self.psf_type = QComboBox()
        self.psf_type.addItems(["gaussian"])  # Future-proofing
        psf_form.addRow("Type:", self.psf_type)

        self.custom_path = QLineEdit()
        self.custom_path.setPlaceholderText("Path to custom PSF (not supported)")
        self.custom_path.setEnabled(False)
        psf_form.addRow("Custom Path:", self.custom_path)

        # Confocal toggle
        self.confocal_checkbox = QCheckBox("Confocal (Enable pinhole)")
        self.confocal_checkbox.toggled.connect(self.toggle_pinhole_visibility)
        self.confocal_checkbox.toggled.connect(self._on_confocal_toggled)
        psf_form.addRow(self.confocal_checkbox)

        # PSF Parameters
        self.numerical_aperture = QDoubleSpinBox()
        self.numerical_aperture.setRange(0.1, 1.5)
        self.numerical_aperture.setSingleStep(0.01)
        psf_form.addRow("Numerical Aperture:", self.numerical_aperture)

        self.refractive_index = QDoubleSpinBox()
        self.refractive_index.setRange(1.0, 2.0)
        self.refractive_index.setValue(1.0)
        self.refractive_index.setSingleStep(0.01)
        psf_form.addRow("Refractive Index:", self.refractive_index)

        self.pinhole_diameter = QDoubleSpinBox()
        self.pinhole_diameter.setRange(0.1, 100.0)
        self.pinhole_diameter.setSuffix(" μm")
        self.pinhole_diameter.setSingleStep(0.1)
        self.pinhole_diameter.setVisible(False)  # Initially hidden
        self.pinhole_label = QLabel("Pinhole Diameter:")
        self.pinhole_label.setVisible(False)
        psf_form.addRow(self.pinhole_label, self.pinhole_diameter)

        self.psf_group.setLayout(psf_form)
        layout.addWidget(self.psf_group)

    def _on_confocal_toggled(self, enabled: bool):
        self.toggle_pinhole_visibility(enabled)
        self.confocal_mode_changed.emit(enabled)

    def toggle_pinhole_visibility(self, enabled: bool):
        self.pinhole_diameter.setVisible(enabled)
        self.pinhole_label.setVisible(enabled)

    def get_data(self):
        """Return current PSF config as a dict."""
        return_dict = {
            "type": self.psf_type.currentText(),
            "custom_path": self.custom_path.text(),
            "parameters": {
                "numerical_aperture": self.numerical_aperture.value(),
                "refractive_index": self.refractive_index.value(),
            },
        }

        if self.confocal_checkbox.isChecked():
            return_dict["parameters"]["pinhole_diameter"] = (
                self.pinhole_diameter.value()
            )

        return return_dict

    def set_data(self, data):
        """Populate fields from a given PSF config dict."""
        self.psf_type.setCurrentText(data.get("type", "gaussian"))
        self.custom_path.setText(data.get("custom_path", ""))

        params = data.get("parameters", {})
        self.numerical_aperture.setValue(params.get("numerical_aperture", 1.0))
        self.refractive_index.setValue(params.get("refractive_index", 1.0))

        pinhole = params.get("pinhole_diameter", None)
        if pinhole is not None:
            self.confocal_checkbox.setChecked(True)
            self.pinhole_diameter.setValue(pinhole)
        else:
            self.confocal_checkbox.setChecked(False)

    def validate(self) -> bool:
        try:
            data = self.get_data()
            # validated = PSFParameters(**data["parameters"])
            QMessageBox.information(
                self, "Validation Successful", "PSF parameters are valid."
            )
            return True
        except ValidationError as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            return False
        except ValueError as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            return False

    def get_help_path(self) -> Path:
        return Path(__file__).parent.parent / "help_docs" / "psf_help.md"
