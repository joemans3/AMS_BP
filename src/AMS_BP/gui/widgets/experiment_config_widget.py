from typing import List

from pydantic import BaseModel, ValidationError
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
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
        layout = QVBoxLayout()
        form = QFormLayout()

        self.validate_button = QPushButton("Validate")
        self.validate_button.clicked.connect(self.validate)
        layout.addWidget(self.validate_button)

        # Experiment Name
        self.experiment_name = QLineEdit()
        form.addRow("Experiment Name:", self.experiment_name)

        # Description
        self.experiment_description = QLineEdit()
        form.addRow("Description:", self.experiment_description)

        # Experiment Type
        self.experiment_type = QComboBox()
        self.experiment_type.addItems(["time-series", "z-stack"])
        self.experiment_type.currentTextChanged.connect(self.update_z_position_field)
        form.addRow("Experiment Type:", self.experiment_type)

        # Z Positions (Dynamic Input)
        self.z_position_inputs = []  # Will hold QDoubleSpinBox widgets dynamically
        self.z_position_label = QLineEdit()  # Default label for z-position
        self.add_z_position_button = QPushButton("Add Z-Position")
        self.add_z_position_button.clicked.connect(self.add_z_position_field)

        self.update_z_position_field(
            self.experiment_type.currentText()
        )  # Initialize based on the current type

        form.addRow(self.z_position_label)
        form.addRow(self.add_z_position_button)

        # Laser Names (Array of Strings)
        self.laser_names = QLineEdit()  # Comma-separated string input
        form.addRow("Active Laser Names:", self.laser_names)

        # Laser Powers (Array of Numbers)
        self.laser_powers = []  # Dynamic array of QDoubleSpinBox
        self.laser_powers.append(QDoubleSpinBox())  # Always just one for time-series
        for power in self.laser_powers:
            power.setRange(0, 1e5)
        form.addRow("Laser Powers (W):", self._hbox(self.laser_powers))

        # Laser Positions (Array of Arrays of Numbers)
        self.laser_positions = []  # Dynamic array of QDoubleSpinBox arrays (x, y, z)
        self.laser_positions.append(
            [QDoubleSpinBox() for _ in range(3)]
        )  # Always just one laser
        for pos in self.laser_positions[0]:
            pos.setRange(-1e5, 1e5)
        form.addRow(
            "Laser Positions (x,y,z):",
            self._hbox([self._hbox(self.laser_positions[0])]),
        )

        # XY Offsets (Array of Numbers)
        self.xy_offset = [QDoubleSpinBox() for _ in range(2)]
        for offset in self.xy_offset:
            offset.setRange(-1e5, 1e5)
        form.addRow("XY Offsets:", self._hbox(self.xy_offset))

        # Exposure Time
        self.exposure_time = QSpinBox()
        self.exposure_time.setRange(0, 10)
        form.addRow("Exposure Time (ms):", self.exposure_time)

        # Interval Time
        self.interval_time = QSpinBox()
        self.interval_time.setRange(0, 10)
        form.addRow("Interval Time (ms):", self.interval_time)

        layout.addLayout(form)
        self.setLayout(layout)

    def update_z_position_field(self, experiment_type):
        """Dynamically update the Z Position field based on experiment type"""
        if experiment_type == "time-series":
            # For time-series, only one z-position field is required
            self.z_position_inputs.clear()
            self.z_position_label.setText("Z Position:")
            self.z_position_inputs.append(self.create_z_position_input())
            self.add_z_position_button.setEnabled(
                False
            )  # Disable Add button for time-series
        else:
            # For z-stack, allow multiple z-position fields
            self.z_position_label.setText("Z Positions (multiple allowed):")
            self.add_z_position_button.setEnabled(True)  # Enable Add button for z-stack

    def add_z_position_field(self):
        """Adds a new Z-position field dynamically when the user clicks the button"""
        z_position_input = self.create_z_position_input()
        self.z_position_inputs.append(z_position_input)
        self.layout().itemAt(1).widget().addWidget(
            z_position_input
        )  # Add to the layout dynamically

    def create_z_position_input(self):
        """Helper function to create a Z-position input"""
        z_position_input = QDoubleSpinBox()
        z_position_input.setRange(-1e5, 1e5)
        return z_position_input

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

    def _hbox(self, widgets):
        box = QHBoxLayout()
        for w in widgets:
            box.addWidget(w)
        container = QWidget()
        container.setLayout(box)
        return container

    def get_data(self):
        experiment_type = self.experiment_type.currentText()

        # Collecting z-positions, one for time-series, or all for z-stack
        z_position = [z.value() for z in self.z_position_inputs]

        return {
            "name": self.experiment_name.text(),
            "description": self.experiment_description.text(),
            "experiment_type": experiment_type,
            "z_position": z_position,
            "laser_names_active": self.laser_names.text().split(","),
            "laser_powers_active": [power.value() for power in self.laser_powers],
            "laser_positions_active": [
                [pos.value() for pos in pos_set] for pos_set in self.laser_positions
            ],
            "xyoffset": [offset.value() for offset in self.xy_offset],
            "exposure_time": self.exposure_time.value(),
            "interval_time": self.interval_time.value(),
        }
