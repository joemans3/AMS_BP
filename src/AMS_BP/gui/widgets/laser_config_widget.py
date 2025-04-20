from pydantic import ValidationError
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...optics.lasers.laser_profiles import LaserParameters


class LaserConfigWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        form = QFormLayout()

        self.validate_button = QPushButton("Validate")
        self.validate_button.clicked.connect(self.validate)
        layout.addWidget(self.validate_button)

        self.num_lasers = QSpinBox()
        self.num_lasers.setRange(1, 10)
        self.num_lasers.setValue(2)
        self.num_lasers.valueChanged.connect(self.update_laser_tabs)
        form.addRow("Number of Lasers:", self.num_lasers)

        self.active_lasers = []

        self.laser_tabs = QTabWidget()
        self.update_laser_tabs()

        layout.addLayout(form)
        layout.addWidget(self.laser_tabs)
        self.setLayout(layout)

    def validate(self) -> bool:
        try:
            data = self.get_data()
            validated = LaserParameters(**data)
            QMessageBox.information(
                self, "Validation Successful", "Laser parameters are valid."
            )
            return True
        except ValidationError as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            return False
        except ValueError as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            return False

    def update_laser_tabs(self):
        # Update the number of lasers based on user input
        num_lasers = self.num_lasers.value()

        # Add tabs for each laser
        while self.laser_tabs.count() < num_lasers:
            self.add_laser_tab(self.laser_tabs.count())

        while self.laser_tabs.count() > num_lasers:
            self.laser_tabs.removeTab(self.laser_tabs.count() - 1)

        self.laser_tabs.setCurrentIndex(0)

    def add_laser_tab(self, index):
        tab = QWidget()
        layout = QFormLayout()

        # Laser name
        laser_name = QLineEdit()
        layout.addRow(f"Laser {index + 1} Name:", laser_name)

        # Laser type
        laser_type = QComboBox()
        laser_type.addItems(["widefield", "gaussian", "hilo"])
        layout.addRow(f"Laser {index + 1} Type:", laser_type)

        # Laser preset
        laser_preset = QLineEdit()
        layout.addRow(f"Laser {index + 1} Preset:", laser_preset)

        # Laser-specific parameters
        power = QDoubleSpinBox()
        power.setRange(0, 100)
        layout.addRow(f"Laser {index + 1} Power (W):", power)

        wavelength = QSpinBox()
        wavelength.setRange(100, 10000)
        layout.addRow(f"Laser {index + 1} Wavelength (nm):", wavelength)

        beam_width = QDoubleSpinBox()
        beam_width.setRange(0, 1000)
        layout.addRow(f"Laser {index + 1} Beam Width (µm):", beam_width)

        numerical_aperture = QDoubleSpinBox()
        numerical_aperture.setRange(0, 2)
        layout.addRow(f"Laser {index + 1} Numerical Aperture:", numerical_aperture)

        refractive_index = QDoubleSpinBox()
        refractive_index.setRange(1, 2)
        layout.addRow(f"Laser {index + 1} Refractive Index:", refractive_index)

        # Inclination angle (only for HiLo laser type)
        inclination_angle = QDoubleSpinBox()
        inclination_angle.setRange(0, 90)
        inclination_angle.setEnabled(False)  # Initially disabled
        layout.addRow(f"Laser {index + 1} Inclination Angle (°):", inclination_angle)

        # Enable/disable inclination angle based on laser type
        laser_type.currentIndexChanged.connect(
            lambda: self.toggle_inclination(laser_type, inclination_angle)
        )

        tab.setLayout(layout)
        self.laser_tabs.addTab(tab, f"Laser {index + 1}")

        # Store widget references for later validation
        self.active_lasers.append(laser_name)

    def toggle_inclination(self, laser_type, inclination_angle):
        """Enable or disable inclination angle field based on laser type."""
        if laser_type.currentText() == "hilo":
            inclination_angle.setEnabled(True)
        else:
            inclination_angle.setEnabled(False)

    def get_data(self):
        lasers_data = []

        for i in range(self.num_lasers.value()):
            laser_data = {
                "name": self.laser_tabs.widget(i).findChild(QLineEdit).text(),
                "type": self.laser_tabs.widget(i).findChild(QComboBox).currentText(),
                "preset": self.laser_tabs.widget(i).findChild(QLineEdit).text(),
                "parameters": {
                    "power": self.laser_tabs.widget(i)
                    .findChild(QDoubleSpinBox)
                    .value(),
                    "wavelength": self.laser_tabs.widget(i).findChild(QSpinBox).value(),
                    "beam_width": self.laser_tabs.widget(i)
                    .findChild(QDoubleSpinBox)
                    .value(),
                    "numerical_aperture": self.laser_tabs.widget(i)
                    .findChild(QDoubleSpinBox)
                    .value(),
                    "refractive_index": self.laser_tabs.widget(i)
                    .findChild(QDoubleSpinBox)
                    .value(),
                    "inclination_angle": self.laser_tabs.widget(i)
                    .findChild(QDoubleSpinBox)
                    .value(),
                },
            }
            lasers_data.append(laser_data)

        return {
            "active": [laser.name for laser in self.active_lasers],
            "lasers": lasers_data,
        }
