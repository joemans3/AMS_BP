from pathlib import Path
from typing import List

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
        self.z_position_inputs: List[QDoubleSpinBox] = []

        self.z_position_container = QWidget()
        self.z_position_layout = QVBoxLayout()
        self.z_position_container.setLayout(self.z_position_layout)
        form.addRow("Z Position(s):", self.z_position_container)

        self.add_z_button = QPushButton("Add Z-Position")
        self.remove_z_button = QPushButton("Remove Z-Position")
        self.remove_z_button.clicked.connect(self.remove_z_position_field)
        self.add_z_button.clicked.connect(self.add_z_position_field)
        self.type_field.currentTextChanged.connect(self.update_z_position_mode)
        z_button_row = QHBoxLayout()
        z_button_row.addWidget(self.add_z_button)
        z_button_row.addWidget(self.remove_z_button)
        layout.addLayout(z_button_row)
        # XY Offset
        self.xyoffset = [QDoubleSpinBox() for _ in range(2)]
        for box in self.xyoffset:
            box.setRange(-1e5, 1e5)
        form.addRow("XY Offset (x, y):", self._hbox(self.xyoffset))

        # Exposure and Interval wrapped in a QWidget
        self.timing_widget = QWidget()
        timing_layout = QFormLayout(self.timing_widget)

        self.exposure = QSpinBox()
        self.exposure.setRange(0, 100000000)
        timing_layout.addRow("Exposure Time (ms):", self.exposure)

        self.interval = QSpinBox()
        self.interval.setRange(0, 100000000)
        timing_layout.addRow("Interval Time (ms):", self.interval)

        form.addRow(self.timing_widget)

        self.update_z_position_mode(self.type_field.currentText())
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

    def update_z_position_mode(self, mode: str):
        # Clear existing
        for i in reversed(range(self.z_position_layout.count())):
            item = self.z_position_layout.itemAt(i).widget()
            if item:
                item.setParent(None)
        self.z_position_inputs.clear()

        if mode == "time-series":
            # One input only
            z_input = QDoubleSpinBox()
            z_input.setRange(-1e5, 1e5)
            self.z_position_inputs.append(z_input)
            self.z_position_layout.addWidget(z_input)
            self.add_z_button.setVisible(False)
            self.add_z_button.setVisible(False)
            self.remove_z_button.setVisible(False)
            self.timing_widget.setVisible(False)
        else:
            # Start with two for z-stack
            for _ in range(2):
                self.add_z_position_field()
            self.add_z_button.setVisible(True)
            self.add_z_button.setVisible(True)
            self.remove_z_button.setVisible(True)
            self.timing_widget.setVisible(True)

    def remove_z_position_field(self):
        if len(self.z_position_inputs) > 1:
            z_widget = self.z_position_inputs.pop()
            self.z_position_layout.removeWidget(z_widget)
            z_widget.setParent(None)
        else:
            QMessageBox.warning(
                self, "Cannot Remove", "At least one Z-position is required."
            )

    def add_z_position_field(self):
        z_input = QDoubleSpinBox()
        z_input.setRange(-1e5, 1e5)
        self.z_position_inputs.append(z_input)
        self.z_position_layout.addWidget(z_input)

    def _hbox(self, widgets):
        box = QHBoxLayout()
        for w in widgets:
            box.addWidget(w)
        container = QWidget()
        container.setLayout(box)
        return container

    def set_active_lasers(self, laser_names: List[str], powers=None, positions=None):
        self.laser_tabs.clear()
        self.laser_power_widgets.clear()
        self.laser_position_widgets.clear()

        powers = powers or [0.0] * len(laser_names)
        positions = positions or [[0.0, 0.0, 0.0] for _ in laser_names]

        for i, name in enumerate(laser_names):
            tab = QWidget()
            form = QFormLayout()

            # Power
            power = QDoubleSpinBox()
            power.setValue(powers[i] if i < len(powers) else 0.0)
            form.addRow("Power (W):", power)

            # Position
            pos_spins = [QDoubleSpinBox() for _ in range(3)]
            for j, s in enumerate(pos_spins):
                s.setValue(
                    positions[i][j]
                    if i < len(positions) and j < len(positions[i])
                    else 0.0
                )
            form.addRow("Position (x, y, z):", self._hbox(pos_spins))

            tab.setLayout(form)
            self.laser_tabs.addTab(tab, name)

            self.laser_power_widgets[name] = power
            self.laser_position_widgets[name] = pos_spins

    def get_data(self):
        data = {
            "name": self.name_field.text(),
            "description": self.desc_field.text(),
            "experiment_type": self.type_field.currentText(),
            "z_position": [z.value() for z in self.z_position_inputs],
            "laser_names_active": list(self.laser_power_widgets.keys()),
            "laser_powers_active": [
                w.value() for w in self.laser_power_widgets.values()
            ],
            "laser_positions_active": [
                [w.value() for w in self.laser_position_widgets[name]]
                for name in self.laser_position_widgets
            ],
            "xyoffset": [w.value() for w in self.xyoffset],
        }

        if data["experiment_type"] == "z-stack":
            data["exposure_time"] = self.exposure.value()
            data["interval_time"] = self.interval.value()

        return data

    def set_data(self, data: dict):
        self.name_field.setText(data.get("name", ""))
        self.desc_field.setText(data.get("description", ""))

        experiment_type = data.get("experiment_type", "time-series")
        idx = self.type_field.findText(experiment_type)
        if idx >= 0:
            self.type_field.setCurrentIndex(idx)

        self.update_z_position_mode(experiment_type)
        z_positions = data.get("z_position", [])
        if experiment_type == "time-series" and z_positions:
            self.z_position_inputs[0].setValue(z_positions[0])
        elif experiment_type == "z-stack":
            for i in range(len(self.z_position_inputs), len(z_positions)):
                self.add_z_position_field()
            for i, val in enumerate(z_positions):
                if i < len(self.z_position_inputs):
                    self.z_position_inputs[i].setValue(val)

        # XY Offset
        xyoffset = data.get("xyoffset", [0, 0])
        for i in range(min(2, len(xyoffset))):
            self.xyoffset[i].setValue(xyoffset[i])

        # Exposure and Interval only for z-stack
        if experiment_type == "z-stack":
            self.exposure.setValue(data.get("exposure_time", 100))
            self.interval.setValue(data.get("interval_time", 0))
        else:
            self.exposure.setValue(0)
            self.interval.setValue(0)

        # Laser data in one call
        laser_names = data.get("laser_names_active", [])
        powers = data.get("laser_powers_active", [])
        positions = data.get("laser_positions_active", [])
        self.set_active_lasers(laser_names, powers=powers, positions=positions)

    def validate(self) -> bool:
        try:
            from ...configio.convertconfig import create_experiment_from_config

            data = self.get_data()
            config_dict = {"experiment": data}

            # This function will raise if the dataclass constructor or __post_init__ fails
            configEXP, funcEXP = create_experiment_from_config(config_dict)

            QMessageBox.information(
                self, "Validation Successful", "Experiment parameters are valid."
            )
            return True

        except Exception as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            return False

    def get_help_path(self) -> Path:
        return Path(__file__).parent.parent / "help_docs" / "experiment_help.md"
