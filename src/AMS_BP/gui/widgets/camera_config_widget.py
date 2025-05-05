from pathlib import Path

from pydantic import ValidationError
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...core.configio.convertconfig import create_quantum_efficiency_from_config
from ...core.optics.camera.detectors import (
    CMOSDetector,
)
from .utility_widgets.spectrum_widget import SpectrumEditorDialog


class CameraConfigWidget(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        form = QFormLayout()

        # Camera type (Only "CMOS" is available)
        self.camera_type = QComboBox()
        self.camera_type.addItems(["CMOS"])
        form.addRow("Camera Type:", self.camera_type)

        # Pixel count (width and height)
        self.pixel_width = QSpinBox()
        self.pixel_width.setRange(1, 10000)
        self.pixel_height = QSpinBox()
        self.pixel_height.setRange(1, 10000)
        form.addRow(
            "Pixel Count (Width x Height):",
            self._hbox([self.pixel_width, self.pixel_height]),
        )

        # Pixel detector size
        self.pixel_detector_size = QDoubleSpinBox()
        self.pixel_detector_size.setRange(0, 100)
        form.addRow("Pixel Detector Size (µm):", self.pixel_detector_size)

        # Magnification
        self.magnification = QSpinBox()
        self.magnification.setRange(1, 100)
        form.addRow("Magnification:", self.magnification)

        # Dark current
        self.dark_current = QDoubleSpinBox()
        self.dark_current.setRange(0, 1000)
        form.addRow("Dark Current (electrons/pixel/sec):", self.dark_current)

        # Readout noise
        self.readout_noise = QDoubleSpinBox()
        self.readout_noise.setRange(0, 1000)
        form.addRow("Readout Noise (electrons RMS):", self.readout_noise)

        # Bit depth
        self.bit_depth = QSpinBox()
        self.bit_depth.setRange(8, 16)
        form.addRow("Bit Depth:", self.bit_depth)

        # Sensitivity
        self.sensitivity = QDoubleSpinBox()
        self.sensitivity.setRange(0, 100)
        form.addRow("Sensitivity (electrons/ADU):", self.sensitivity)

        # Base ADU
        self.base_adu = QSpinBox()
        self.base_adu.setRange(0, 65535)
        form.addRow("Base ADU:", self.base_adu)

        # Binning size
        self.binning_size = QSpinBox()
        self.binning_size.setRange(1, 10)
        form.addRow("Binning Size:", self.binning_size)

        # Quantum efficiency
        self.quantum_efficiency_button = QPushButton("Edit Quantum Efficiency")
        self.quantum_efficiency_button.clicked.connect(self.edit_quantum_efficiency)
        form.addRow("Quantum Efficiency:", self.quantum_efficiency_button)

        layout.addLayout(form)
        self.setLayout(layout)

        self.validate_button = QPushButton("Validate Parameters")
        self.validate_button.clicked.connect(self.validate)
        layout.addWidget(self.validate_button)

    def _hbox(self, widgets):
        box = QHBoxLayout()
        for w in widgets:
            box.addWidget(w)
        container = QWidget()
        container.setLayout(box)
        return container

    def edit_quantum_efficiency(self):
        # Open the SpectrumEditorDialog for editing quantum efficiency
        dialog = SpectrumEditorDialog(
            parent=self,
            wavelengths=[],  # Pass an empty list or preloaded wavelengths
            intensities=[],  # Pass an empty list or preloaded intensities
            intensity_name="QE",
        )

        if dialog.exec():
            # Handle the updated quantum efficiency data (wavelengths, intensities)
            self.quantum_efficiency_data = {
                "wavelengths": dialog.wavelengths,
                "quantum_efficiency": dialog.intensities,
            }

    def validate(self) -> bool:
        try:
            data = self.get_data()

            # Convert QE before passing
            qe = create_quantum_efficiency_from_config(data["quantum_efficiency"])

            camera_data = {
                k: v for k, v in data.items() if k not in {"type", "quantum_efficiency"}
            }
            camera_data["pixel_size"] = (
                camera_data["pixel_detector_size"] / camera_data["magnification"]
            )

            # Attempt to construct the detector
            detector = CMOSDetector(**camera_data)

            # If no exception, validation is successful
            QMessageBox.information(
                self, "Validation Successful", "Camera parameters are valid."
            )
            return True

        except ValidationError as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            return False
        except Exception as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            return False

    def get_data(self):
        camera_data = {
            "type": self.camera_type.currentText(),
            "pixel_count": [self.pixel_width.value(), self.pixel_height.value()],
            "pixel_detector_size": self.pixel_detector_size.value(),
            "magnification": self.magnification.value(),
            "dark_current": self.dark_current.value(),
            "readout_noise": self.readout_noise.value(),
            "bit_depth": self.bit_depth.value(),
            "sensitivity": self.sensitivity.value(),
            "base_adu": self.base_adu.value(),
            "binning_size": self.binning_size.value(),
            "quantum_efficiency": convert_dict_to_2_list(self.quantum_efficiency_data),
        }
        return camera_data

    def set_data(self, data: dict):
        # Camera type
        camera_type = data.get("type", "CMOS")
        idx = self.camera_type.findText(camera_type)
        if idx >= 0:
            self.camera_type.setCurrentIndex(idx)

        # Pixel count
        pixel_count = data.get("pixel_count", [512, 512])
        if len(pixel_count) == 2:
            self.pixel_width.setValue(pixel_count[0])
            self.pixel_height.setValue(pixel_count[1])

        self.pixel_detector_size.setValue(data.get("pixel_detector_size", 6.5))
        self.magnification.setValue(data.get("magnification", 60))
        self.dark_current.setValue(data.get("dark_current", 1.0))
        self.readout_noise.setValue(data.get("readout_noise", 1.0))
        self.bit_depth.setValue(data.get("bit_depth", 16))
        self.sensitivity.setValue(data.get("sensitivity", 0.5))
        self.base_adu.setValue(data.get("base_adu", 100))
        self.binning_size.setValue(data.get("binning_size", 1))

        # Quantum efficiency
        raw_qe = data.get("quantum_efficiency", [])
        wavelengths = [pair[0] for pair in raw_qe]
        efficiencies = [pair[1] for pair in raw_qe]
        self.quantum_efficiency_data = {
            "wavelengths": wavelengths,
            "quantum_efficiency": efficiencies,
        }

    def get_help_path(self) -> Path:
        return Path(__file__).parent.parent / "help_docs" / "detector_help.md"


def convert_dict_to_2_list(dict_c):
    wl_qe = []
    wl = dict_c["wavelengths"]
    qe = dict_c["quantum_efficiency"]
    for i in range(len(wl)):
        lista = [wl[i], qe[i]]
        wl_qe.append(lista)
    return wl_qe
