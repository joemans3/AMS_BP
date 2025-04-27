from pathlib import Path

from pydantic import ValidationError
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...core.configio.configmodels import MoleculeParameters
from ...core.configio.convertconfig import create_dataclass_schema


class MoleculeConfigWidget(QWidget):
    # Signal to notify when molecule count changes
    molecule_count_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        self.setLayout(self.main_layout)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.setLayout(self.main_layout)

        # Number of molecule types spinner
        self.num_types_layout = QHBoxLayout()
        self.num_types_label = QLabel("Number of molecule types:")
        self.num_types_spinner = QSpinBox()
        self.num_types_spinner.setRange(1, 10)
        self.num_types_spinner.setValue(1)
        self.num_types_spinner.valueChanged.connect(self._on_molecule_count_changed)
        self.num_types_layout.addWidget(self.num_types_label)
        self.num_types_layout.addWidget(self.num_types_spinner)
        self.num_types_layout.addStretch()

        self.main_layout.addLayout(self.num_types_layout)

        # Create tab widget to hold molecule type configs
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # Initialize with one molecule type
        self.molecule_type_widgets = []
        self.update_molecule_types(1)

        # Add the validate button at the bottom
        self.validate_button = QPushButton("Validate Parameters")
        self.validate_button.clicked.connect(self.validate)
        self.main_layout.addWidget(self.validate_button)

    def _on_molecule_count_changed(self, count):
        self.update_molecule_types(count)
        self.molecule_count_changed.emit(count)

    def set_molecule_count(self, count):
        if self.num_types_spinner.value() != count:
            self.num_types_spinner.blockSignals(True)
            self.num_types_spinner.setValue(count)
            self.num_types_spinner.blockSignals(False)
            self.update_molecule_types(count)
            self.molecule_count_changed.emit(count)

    def update_molecule_types(self, num_types):
        # Store current data if any
        current_data = self.get_data() if self.molecule_type_widgets else None

        # Remove existing tabs
        while self.tabs.count() > 0:
            self.tabs.removeTab(0)

        self.molecule_type_widgets = []

        # Create new tabs
        for i in range(num_types):
            molecule_widget = MoleculeTypeWidget(i)
            self.molecule_type_widgets.append(molecule_widget)
            self.tabs.addTab(molecule_widget, f"Molecule Type {i+1}")

        # Restore data only if tab count matches current data
        if current_data and len(current_data["num_molecules"]) == num_types:
            self.set_data_from_widget_selection(current_data)

    def validate(self) -> bool:
        try:
            data = self.get_data()

            # This will validate the schema using the backend logic

            _ = create_dataclass_schema(MoleculeParameters, data)

            QMessageBox.information(
                self, "Validation Successful", "Molecule parameters are valid."
            )
            return True
        except ValidationError as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            return False
        except ValueError as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            return False

    def set_data(self, config: dict):
        """
        Load molecule configuration from TOML config format.
        """
        try:
            validated = create_dataclass_schema(MoleculeParameters, config)

            # Determine number of types
            num_types = len(validated.num_molecules)
            self.set_molecule_count(num_types)

            # Build compatible format to use set_data_from_widget_state()
            parsed = {
                "num_molecules": validated.num_molecules,
                "track_type": validated.track_type,
                "diffusion_coefficient": validated.diffusion_coefficient,
                "hurst_exponent": validated.hurst_exponent,
                "allow_transition_probability": validated.allow_transition_probability,
                "transition_matrix_time_step": validated.transition_matrix_time_step,
                "diffusion_transition_matrix": validated.diffusion_transition_matrix,
                "hurst_transition_matrix": validated.hurst_transition_matrix,
                "state_probability_diffusion": validated.state_probability_diffusion,
                "state_probability_hurst": validated.state_probability_hurst,
            }

            self.set_data_from_widget_selection(parsed)

        except Exception as e:
            print(f"[MoleculeConfigWidget] Failed to load from config: {e}")

    def get_data(self):
        num_molecules = []
        track_type = []
        diffusion_coefficient = []
        hurst_exponent = []
        allow_transition_probability = []
        transition_matrix_time_step = []
        diffusion_transition_matrix = []
        hurst_transition_matrix = []
        state_probability_diffusion = []
        state_probability_hurst = []

        for widget in self.molecule_type_widgets:
            type_data = widget.get_data()

            num_molecules.append(type_data["num_molecules"])
            track_type.append(type_data["track_type"])
            diffusion_coefficient.append(type_data["diffusion_coefficient"])
            hurst_exponent.append(type_data["hurst_exponent"])
            allow_transition_probability.append(
                type_data["allow_transition_probability"]
            )
            transition_matrix_time_step.append(type_data["transition_matrix_time_step"])
            diffusion_transition_matrix.append(type_data["diffusion_transition_matrix"])
            hurst_transition_matrix.append(type_data["hurst_transition_matrix"])
            state_probability_diffusion.append(type_data["state_probability_diffusion"])
            state_probability_hurst.append(type_data["state_probability_hurst"])

        # Normalize shapes: make sure every list has the same shape
        def ensure_nested_list_shape(lst, expected_len, default_val):
            return [x if len(x) > 0 else [default_val] for x in lst]

        def pad_matrix_list(mat_list, expected_size):
            result = []
            for mat in mat_list:
                if len(mat) == 0:
                    result.append([[1.0]])
                else:
                    result.append(mat)
            return result

        diffusion_coefficient = ensure_nested_list_shape(
            diffusion_coefficient, len(num_molecules), 1.0
        )
        hurst_exponent = ensure_nested_list_shape(
            hurst_exponent, len(num_molecules), 0.5
        )
        state_probability_diffusion = ensure_nested_list_shape(
            state_probability_diffusion, len(num_molecules), 1.0
        )
        state_probability_hurst = ensure_nested_list_shape(
            state_probability_hurst, len(num_molecules), 1.0
        )

        diffusion_transition_matrix = pad_matrix_list(
            diffusion_transition_matrix, len(num_molecules)
        )
        hurst_transition_matrix = pad_matrix_list(
            hurst_transition_matrix, len(num_molecules)
        )
        return {
            "num_molecules": num_molecules,
            "track_type": track_type,
            "diffusion_coefficient": diffusion_coefficient,
            "hurst_exponent": hurst_exponent,
            "allow_transition_probability": allow_transition_probability,
            "transition_matrix_time_step": transition_matrix_time_step,
            "diffusion_transition_matrix": diffusion_transition_matrix,
            "hurst_transition_matrix": hurst_transition_matrix,
            "state_probability_diffusion": state_probability_diffusion,
            "state_probability_hurst": state_probability_hurst,
        }

    def set_data_from_widget_selection(self, data):
        num_types = min(len(data["num_molecules"]), len(self.molecule_type_widgets))
        self.num_types_spinner.setValue(num_types)

        for i in range(num_types):
            type_data = {
                "num_molecules": data["num_molecules"][i],
                "track_type": data["track_type"][i],
                "diffusion_coefficient": data["diffusion_coefficient"][i],
                "hurst_exponent": data["hurst_exponent"][i],
                "allow_transition_probability": data["allow_transition_probability"][i],
                "transition_matrix_time_step": data["transition_matrix_time_step"][i],
                "diffusion_transition_matrix": data["diffusion_transition_matrix"][i],
                "hurst_transition_matrix": data["hurst_transition_matrix"][i],
                "state_probability_diffusion": data["state_probability_diffusion"][i],
                "state_probability_hurst": data["state_probability_hurst"][i],
            }
            self.molecule_type_widgets[i].set_data(type_data)

    def get_help_path(self) -> Path:
        return Path(__file__).parent.parent / "help_docs" / "molecule_help.md"


class MoleculeTypeWidget(QWidget):
    def __init__(self, type_index):
        super().__init__()
        self.type_index = type_index

        # Create a scroll area to handle potentially large configs
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)

        self.container = QWidget()
        self.layout = QVBoxLayout(self.container)

        # Basic parameters
        self.form_layout = QFormLayout()

        # Number of molecules
        self.num_molecules = QSpinBox()
        self.num_molecules.setRange(1, 100000)
        self.num_molecules.setValue(100)
        self.form_layout.addRow("Number of molecules:", self.num_molecules)

        # Track type
        self.track_type = QComboBox()
        self.track_type.addItems(["constant", "fbm"])
        self.track_type.currentTextChanged.connect(self.update_visibility)
        self.form_layout.addRow("Track type:", self.track_type)

        # Transition probability
        self.allow_transition = QCheckBox("Allow transition probability")
        self.allow_transition.setChecked(False)
        self.allow_transition.stateChanged.connect(self.update_visibility)
        self.form_layout.addRow("", self.allow_transition)

        # Transition matrix time step
        self.transition_time_step = QSpinBox()
        self.transition_time_step.setRange(1, 10000)
        self.transition_time_step.setValue(100)
        self.transition_time_step.setSuffix(" ms")
        self.form_layout.addRow(
            "Transition matrix time step:", self.transition_time_step
        )

        self.layout.addLayout(self.form_layout)

        # Diffusion coefficient section
        self.diffusion_group = QWidget()
        self.diffusion_layout = QVBoxLayout(self.diffusion_group)

        self.diffusion_header = QHBoxLayout()
        self.diffusion_label = QLabel("<b>Diffusion Coefficients</b>")
        self.diffusion_count = QSpinBox()
        self.diffusion_count.setValue(1)
        self.diffusion_count.valueChanged.connect(self.update_diffusion_coefficients)
        self.diffusion_header.addWidget(self.diffusion_label)
        self.diffusion_header.addWidget(QLabel("Number of states:"))
        self.diffusion_header.addWidget(self.diffusion_count)
        self.diffusion_header.addStretch()

        self.diffusion_layout.addLayout(self.diffusion_header)

        self.diffusion_grid = QGridLayout()
        self.diffusion_grid.addWidget(QLabel("Coefficient (μm²/s)"), 0, 0)
        self.diffusion_grid.addWidget(
            QLabel("Initial Fraction in this State (0-1)"), 0, 1
        )

        self.diffusion_coefficients = []
        self.diffusion_amounts = []

        self.update_diffusion_coefficients(1)
        self.diffusion_layout.addLayout(self.diffusion_grid)

        # Transition matrix for diffusion
        self.diffusion_matrix_widget = TransitionMatrixWidget("Diffusion Coefficient")
        self.diffusion_layout.addWidget(self.diffusion_matrix_widget)

        self.layout.addWidget(self.diffusion_group)

        # Hurst exponent section (visible only for fbm)
        self.hurst_group = QWidget()
        self.hurst_layout = QVBoxLayout(self.hurst_group)

        self.hurst_header = QHBoxLayout()
        self.hurst_label = QLabel("<b>Hurst Exponents</b>")
        self.hurst_count = QSpinBox()
        self.hurst_count.setRange(1, 10)  # set minimum to 1
        self.hurst_count.setValue(1)
        self.hurst_count.valueChanged.connect(self.update_hurst_exponents)
        self.hurst_header.addWidget(self.hurst_label)
        self.hurst_header.addWidget(QLabel("Number of states:"))
        self.hurst_header.addWidget(self.hurst_count)
        self.hurst_header.addStretch()

        self.hurst_layout.addLayout(self.hurst_header)

        self.hurst_grid = QGridLayout()
        self.hurst_grid.addWidget(QLabel("Exponent (0-1)"), 0, 0)
        self.hurst_grid.addWidget(QLabel("Initial Fraction in this State (0-1)"), 0, 1)

        self.hurst_exponents = []
        self.hurst_amounts = []

        self.update_hurst_exponents(1)
        self.hurst_layout.addLayout(self.hurst_grid)

        # Transition matrix for Hurst
        self.hurst_matrix_widget = TransitionMatrixWidget("Hurst Exponent")
        self.hurst_layout.addWidget(self.hurst_matrix_widget)

        self.layout.addWidget(self.hurst_group)

        # Set the container as the scroll area widget
        self.scroll.setWidget(self.container)

        # Main layout for this widget
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.scroll)

        # Connect signals for dependency updates
        self.diffusion_count.valueChanged.connect(
            lambda val: self.diffusion_matrix_widget.update_matrix_size(val)
        )
        self.hurst_count.valueChanged.connect(
            lambda val: self.hurst_matrix_widget.update_matrix_size(val)
        )

        # Update visibility based on initial values
        self.update_visibility()

    def update_visibility(self):
        track = self.track_type.currentText()
        is_fbm = track == "fbm"
        is_constant = track == "constant"
        allow_transitions = self.allow_transition.isChecked()

        # Hide/show all motion-related groups
        self.diffusion_group.setVisible(not is_constant)
        self.hurst_group.setVisible(is_fbm and not is_constant)

        # Hide/show allow_transition checkbox and transition step field
        self.allow_transition.setVisible(not is_constant)
        label = self.form_layout.labelForField(self.transition_time_step)
        if label:
            label.setVisible(not is_constant and allow_transitions)
        self.transition_time_step.setVisible(not is_constant and allow_transitions)

        # Transition matrices visibility
        self.diffusion_matrix_widget.setVisible(not is_constant and allow_transitions)
        self.hurst_matrix_widget.setVisible(
            not is_constant and is_fbm and allow_transitions
        )

    def update_diffusion_coefficients(self, count):
        # Store current total amount
        current_total = (
            sum(spin.value() for spin in self.diffusion_amounts)
            if self.diffusion_amounts
            else 1.0
        )

        # Clear existing items
        while len(self.diffusion_coefficients) > count:
            # Remove the last row
            self.diffusion_grid.removeWidget(self.diffusion_coefficients[-1])
            self.diffusion_grid.removeWidget(self.diffusion_amounts[-1])
            self.diffusion_coefficients[-1].deleteLater()
            self.diffusion_amounts[-1].deleteLater()
            self.diffusion_coefficients.pop()
            self.diffusion_amounts.pop()

        # Add new items if needed
        while len(self.diffusion_coefficients) < count:
            row = len(self.diffusion_coefficients) + 1
            coeff = QDoubleSpinBox()
            coeff.setRange(0, 1000)
            coeff.setValue(1.0)
            coeff.setSuffix(" μm²/s")
            coeff.setDecimals(3)

            amount = QDoubleSpinBox()
            amount.setRange(0, 1)
            amount.setValue(1.0 / count)  # Default value will be adjusted later
            amount.setSingleStep(0.1)
            amount.setDecimals(2)
            amount.valueChanged.connect(self.normalize_diffusion_amounts)

            self.diffusion_grid.addWidget(coeff, row, 0)
            self.diffusion_grid.addWidget(amount, row, 1)

            self.diffusion_coefficients.append(coeff)
            self.diffusion_amounts.append(amount)

        # Normalize the amounts to sum to 1
        self.normalize_diffusion_amounts()

    def normalize_diffusion_amounts(self):
        # Get the sender (which spin box was changed)
        sender = self.sender()

        # Skip if this wasn't triggered by a spin box change
        if not sender or sender not in self.diffusion_amounts:
            # Equal distribution
            if self.diffusion_amounts:
                for amount in self.diffusion_amounts:
                    amount.blockSignals(True)
                    amount.setValue(1.0 / len(self.diffusion_amounts))
                    amount.blockSignals(False)
            return

        # Get total and adjust other values proportionally
        total = sum(spin.value() for spin in self.diffusion_amounts)

        if total > 0:
            # If total > 1, scale down others
            if total > 1.0:
                # How much we need to reduce by
                excess = total - 1.0

                # Get the sum of other amounts
                other_sum = total - sender.value()

                # Adjust each amount proportionally
                for amount in self.diffusion_amounts:
                    if amount != sender:
                        if other_sum > 0:
                            # Reduce proportionally
                            amount.blockSignals(True)
                            new_value = max(
                                0,
                                amount.value()
                                - (excess * (amount.value() / other_sum)),
                            )
                            amount.setValue(new_value)
                            amount.blockSignals(False)
                        else:
                            # If other sum is 0, we can't adjust proportionally
                            amount.blockSignals(True)
                            amount.setValue(0)
                            amount.blockSignals(False)

    def update_hurst_exponents(self, count):
        # Store current total amount
        current_total = (
            sum(spin.value() for spin in self.hurst_amounts)
            if self.hurst_amounts
            else 1.0
        )

        # Clear existing items
        while len(self.hurst_exponents) > count:
            # Remove the last row
            self.hurst_grid.removeWidget(self.hurst_exponents[-1])
            self.hurst_grid.removeWidget(self.hurst_amounts[-1])
            self.hurst_exponents[-1].deleteLater()
            self.hurst_amounts[-1].deleteLater()
            self.hurst_exponents.pop()
            self.hurst_amounts.pop()

        # Add new items if needed
        while len(self.hurst_exponents) < count:
            row = len(self.hurst_exponents) + 1
            exp = QDoubleSpinBox()
            exp.setRange(0, 1)
            exp.setValue(0.5)
            exp.setSingleStep(0.1)
            exp.setDecimals(2)

            amount = QDoubleSpinBox()
            amount.setRange(0, 1)
            amount.setValue(1.0 / count)  # Default value will be adjusted later
            amount.setSingleStep(0.1)
            amount.setDecimals(2)
            amount.valueChanged.connect(self.normalize_hurst_amounts)

            self.hurst_grid.addWidget(exp, row, 0)
            self.hurst_grid.addWidget(amount, row, 1)

            self.hurst_exponents.append(exp)
            self.hurst_amounts.append(amount)

        # Normalize the amounts to sum to 1
        self.normalize_hurst_amounts()

    def normalize_hurst_amounts(self):
        # Get the sender (which spin box was changed)
        sender = self.sender()

        # Skip if this wasn't triggered by a spin box change
        if not sender or sender not in self.hurst_amounts:
            # Equal distribution
            if self.hurst_amounts:
                for amount in self.hurst_amounts:
                    amount.blockSignals(True)
                    amount.setValue(1.0 / len(self.hurst_amounts))
                    amount.blockSignals(False)
            return

        # Get total and adjust other values proportionally
        total = sum(spin.value() for spin in self.hurst_amounts)

        if total > 0:
            # If total > 1, scale down others
            if total > 1.0:
                # How much we need to reduce by
                excess = total - 1.0

                # Get the sum of other amounts
                other_sum = total - sender.value()

                # Adjust each amount proportionally
                for amount in self.hurst_amounts:
                    if amount != sender:
                        if other_sum > 0:
                            # Reduce proportionally
                            amount.blockSignals(True)
                            new_value = max(
                                0,
                                amount.value()
                                - (excess * (amount.value() / other_sum)),
                            )
                            amount.setValue(new_value)
                            amount.blockSignals(False)
                        else:
                            # If other sum is 0, we can't adjust proportionally
                            amount.blockSignals(True)
                            amount.setValue(0)
                            amount.blockSignals(False)

    def get_data(self):
        # Get diffusion coefficients and state probabilities
        diff_coeff = [spin.value() for spin in self.diffusion_coefficients]
        diff_prob = [spin.value() for spin in self.diffusion_amounts]

        # Get Hurst exponents and state probabilities (only if visible)
        hurst_exp = (
            [spin.value() for spin in self.hurst_exponents]
            if self.hurst_group.isVisible()
            else []
        )
        hurst_prob = (
            [spin.value() for spin in self.hurst_amounts]
            if self.hurst_group.isVisible()
            else []
        )

        # Transition matrices
        diff_matrix = self.diffusion_matrix_widget.get_matrix()
        hurst_matrix = (
            self.hurst_matrix_widget.get_matrix()
            if self.hurst_group.isVisible()
            else []
        )

        return {
            "num_molecules": self.num_molecules.value(),
            "track_type": self.track_type.currentText(),
            "diffusion_coefficient": diff_coeff,
            "hurst_exponent": hurst_exp,
            "allow_transition_probability": self.allow_transition.isChecked(),
            "transition_matrix_time_step": self.transition_time_step.value(),
            "diffusion_transition_matrix": diff_matrix,
            "hurst_transition_matrix": hurst_matrix,
            "state_probability_diffusion": diff_prob,
            "state_probability_hurst": hurst_prob,
        }

    def set_data(self, data):
        self.num_molecules.setValue(data["num_molecules"])

        index = self.track_type.findText(data["track_type"])
        if index >= 0:
            self.track_type.setCurrentIndex(index)

        self.allow_transition.setChecked(data["allow_transition_probability"])
        self.transition_time_step.setValue(data["transition_matrix_time_step"])

        # Diffusion Coefficients
        diff_count = len(data["diffusion_coefficient"])
        self.diffusion_count.setValue(diff_count)
        for i in range(diff_count):
            if i < len(self.diffusion_coefficients):
                self.diffusion_coefficients[i].setValue(
                    data["diffusion_coefficient"][i]
                )
                self.diffusion_amounts[i].setValue(
                    data["state_probability_diffusion"][i]
                )

        # Hurst Exponents (only if track_type == "fbm")
        if data["track_type"] == "fbm":
            hurst_count = max(1, len(data["hurst_exponent"]))  # ensure at least 1
            self.hurst_count.setValue(hurst_count)
            for i in range(hurst_count):
                if i < len(self.hurst_exponents):
                    self.hurst_exponents[i].setValue(data["hurst_exponent"][i])
                    self.hurst_amounts[i].setValue(data["state_probability_hurst"][i])

            self.hurst_matrix_widget.set_matrix(data["hurst_transition_matrix"])
        else:
            self.hurst_count.setValue(0)

        # Set diffusion matrix
        self.diffusion_matrix_widget.set_matrix(data["diffusion_transition_matrix"])

        # Refresh UI visibility
        self.update_visibility()


class TransitionMatrixWidget(QWidget):
    def __init__(self, title):
        super().__init__()
        self.layout = QVBoxLayout(self)

        self.layout.addWidget(QLabel(f"<b>{title} Transition Matrix</b>"))
        self.layout.addWidget(
            QLabel(
                "Probabilities of transitioning between states (rows must sum to 1.0):"
            )
        )

        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.layout.addWidget(self.grid_container)

        self.spinboxes = []
        self.update_matrix_size(1)

    def update_matrix_size(self, size):
        # Clear existing grid
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.spinboxes = []

        # Add column headers (to state)
        for i in range(size):
            self.grid_layout.addWidget(QLabel(f"To {i+1}"), 0, i + 1)

        # Add row headers (from state)
        for i in range(size):
            self.grid_layout.addWidget(QLabel(f"From {i+1}"), i + 1, 0)

        # Create transition probability spinboxes
        for i in range(size):
            row = []
            for j in range(size):
                spin = QDoubleSpinBox()
                spin.setRange(0, 1)
                spin.setValue(
                    1.0 / size if i == j else 0.0
                )  # Default to self-transitions
                spin.setSingleStep(0.1)
                spin.setDecimals(2)
                self.grid_layout.addWidget(spin, i + 1, j + 1)
                row.append(spin)
            self.spinboxes.append(row)

    def get_matrix(self):
        matrix = []
        for row in self.spinboxes:
            matrix.append([spin.value() for spin in row])
        return matrix

    def set_matrix(self, matrix):
        size = len(matrix)
        if size != len(self.spinboxes):
            self.update_matrix_size(size)

        for i in range(size):
            for j in range(min(size, len(matrix[i]))):
                self.spinboxes[i][j].setValue(matrix[i][j])
