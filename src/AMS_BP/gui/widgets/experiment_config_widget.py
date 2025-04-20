from typing import List

from pydantic import BaseModel, ValidationError
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


# Pydantic Model for Experiment Parameters
class ExperimentParameters(BaseModel):
    name: str
    description: str
    experiment_type: str
    z_position: List[float]  # Array of floats (time-series will be one element)
    laser_names_active: List[str]  # List of strings
    laser_powers_active: List[float]  # List of floats
    laser_positions_active: List[List[float]]  # List of lists of floats
    xyoffset: List[float]  # List of floats
    exposure_time: int  # Exposure time in ms
    interval_time: int  # Interval time in ms


class ExperimentConfigWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.laser_power_widgets = {}
        self.laser_position_widgets = {}

        layout = QVBoxLayout()
        form = QFormLayout()

        # Experiment Info
        self.name_field = QLineEdit()
        form.addRow("Experiment Name:", self.name_field)

        self.desc_field = QLineEdit()
        form.addRow("Description:", self.desc_field)

        self.type_field = QComboBox()
        self.type_field.addItems(["time-series", "z-stack"])
        form.addRow("Experiment Type:", self.type_field)

        # Z Position (just one for time-series)
        self.z_position_field = QDoubleSpinBox()
        self.z_position_field.setRange(-1e5, 1e5)
        form.addRow("Z Position:", self.z_position_field)

        # XY Offset
        self.xyoffset = [QDoubleSpinBox() for _ in range(2)]
        for box in self.xyoffset:
            box.setRange(-1e5, 1e5)
        form.addRow("XY Offset (x, y):", self._hbox(self.xyoffset))

        # Exposure and Interval
        self.exposure = QSpinBox()
        self.exposure.setRange(0, 10000)
        form.addRow("Exposure Time (ms):", self.exposure)

        self.interval = QSpinBox()
        self.interval.setRange(0, 10000)
        form.addRow("Interval Time (ms):", self.interval)

        layout.addLayout(form)

        # Laser Tabs
        self.laser_tabs = QTabWidget()
        layout.addWidget(QLabel("Active Laser Parameters:"))
        layout.addWidget(self.laser_tabs)

        # Validate Button
        self.validate_button = QPushButton("Validate")
        self.validate_button.clicked.connect(self.validate)
        layout.addWidget(self.validate_button)

        self.setLayout(layout)

    def _hbox(self, widgets):
        box = QHBoxLayout()
        for w in widgets:
            box.addWidget(w)
        container = QWidget()
        container.setLayout(box)
        return container

    def set_active_lasers(self, laser_names: List[str]):
        self.laser_tabs.clear()
        self.laser_power_widgets.clear()
        self.laser_position_widgets.clear()

        for name in laser_names:
            tab = QWidget()
            form = QFormLayout()

            # Power
            power = QDoubleSpinBox()
            power.setRange(0, 1e5)
            form.addRow("Power (W):", power)

            # Position
            pos_spins = [QDoubleSpinBox() for _ in range(3)]
            for s in pos_spins:
                s.setRange(-1e5, 1e5)
            form.addRow("Position (x, y, z):", self._hbox(pos_spins))

            tab.setLayout(form)
            self.laser_tabs.addTab(tab, name)

            self.laser_power_widgets[name] = power
            self.laser_position_widgets[name] = pos_spins

    def get_data(self):
        return {
            "name": self.name_field.text(),
            "description": self.desc_field.text(),
            "experiment_type": self.type_field.currentText(),
            "z_position": [self.z_position_field.value()],
            "laser_names_active": list(self.laser_power_widgets.keys()),
            "laser_powers_active": [
                w.value() for w in self.laser_power_widgets.values()
            ],
            "laser_positions_active": [
                [w.value() for w in self.laser_position_widgets[name]]
                for name in self.laser_position_widgets
            ],
            "xyoffset": [w.value() for w in self.xyoffset],
            "exposure_time": self.exposure.value(),
            "interval_time": self.interval.value(),
        }

    def validate(self) -> bool:
        try:
            data = self.get_data()
            validated = ExperimentParameters(**data)
            QMessageBox.information(
                self, "Validation Successful", "Experiment parameters are valid."
            )
            return True
        except ValidationError as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            return False
        except ValueError as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            return False
