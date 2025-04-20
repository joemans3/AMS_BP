from pydantic import ValidationError
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .utility_widgets.spectrum_widget import SpectrumEditorDialog


class FluorophoreConfigWidget(QWidget):
    # Signal to notify when molecule count changes
    mfluorophore_count_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.fluorophore_widgets = []
        self._updating_count = False
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        instructions = QLabel(
            "Configure fluorophores and their respective states and transitions."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Controls for fluorophore count
        controls_layout = QHBoxLayout()
        self.fluorophore_count = QSpinBox()
        self.fluorophore_count.setRange(1, 10)
        self.fluorophore_count.setValue(1)
        self.fluorophore_count.valueChanged.connect(self._on_fluorophore_count_changed)

        controls_layout.addWidget(QLabel("Number of Fluorophores:"))
        controls_layout.addWidget(self.fluorophore_count)
        layout.addLayout(controls_layout)

        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        self.validate_button = QPushButton("Validate Parameters")
        self.validate_button.clicked.connect(self.validate)
        layout.addWidget(self.validate_button)

        self.setLayout(layout)
        self.update_fluorophore_count(1)

    def _on_fluorophore_count_changed(self, count):
        if not self._updating_count:
            self.update_fluorophore_count(count)
            # Emit signal to notify other widgets
            self.mfluorophore_count_changed.emit(count)

    def set_mfluorophore_count(self, count):
        """Public method to be called by other widgets to update fluorophore molecule count"""
        if self.fluorophore_count.value() != count:
            self._updating_count = True
            self.fluorophore_count.setValue(count)
            self.update_fluorophore_count(count)
            self._updating_count = False

    def update_fluorophore_count(self, count):
        current_count = self.tab_widget.count()

        for i in range(current_count, count):
            self.add_fluorophore_tab(i)

        while self.tab_widget.count() > count:
            self.tab_widget.removeTab(count)
            if self.fluorophore_widgets:
                self.fluorophore_widgets.pop()

    def add_fluorophore_tab(self, index):
        fluor_widget = QWidget()
        layout = QVBoxLayout(fluor_widget)

        form = QFormLayout()

        name_input = QLineEdit()
        form.addRow("Name:", name_input)

        initial_state_input = QLineEdit()
        form.addRow("Initial State:", initial_state_input)

        layout.addLayout(form)

        # === STATES ===
        states_box = QGroupBox("States")
        states_layout = QVBoxLayout()
        states_controls = QHBoxLayout()

        add_state_btn = QPushButton("Add State")
        remove_state_btn = QPushButton("Remove Last State")
        states_controls.addWidget(add_state_btn)
        states_controls.addWidget(remove_state_btn)

        states_layout.addLayout(states_controls)

        state_container = QVBoxLayout()
        states_layout.addLayout(state_container)
        states_box.setLayout(states_layout)

        # === TRANSITIONS ===
        transitions_box = QGroupBox("Transitions")
        transitions_layout = QVBoxLayout()
        transitions_controls = QHBoxLayout()

        add_transition_btn = QPushButton("Add Transition")
        remove_transition_btn = QPushButton("Remove Last Transition")
        transitions_controls.addWidget(add_transition_btn)
        transitions_controls.addWidget(remove_transition_btn)

        transitions_layout.addLayout(transitions_controls)

        transition_container = QVBoxLayout()
        transitions_layout.addLayout(transition_container)
        transitions_box.setLayout(transitions_layout)

        # Add to main layout
        layout.addWidget(states_box)
        layout.addWidget(transitions_box)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(fluor_widget)

        self.tab_widget.addTab(scroll, f"Fluorophore {index + 1}")

        # Tracking widget refs
        widget_refs = {
            "name": name_input,
            "initial_state": initial_state_input,
            "state_container": state_container,
            "transition_container": transition_container,
            "states": [],
            "transitions": [],
        }

        self.fluorophore_widgets.append(widget_refs)

        # Initial state and transition
        self.add_state_group(widget_refs)
        self.add_transition_group(widget_refs)

        # Button logic
        add_state_btn.clicked.connect(lambda: self.add_state_group(widget_refs))
        remove_state_btn.clicked.connect(
            lambda: self.remove_last_group(widget_refs["states"])
        )

        add_transition_btn.clicked.connect(
            lambda: self.add_transition_group(widget_refs)
        )
        remove_transition_btn.clicked.connect(
            lambda: self.remove_last_group(widget_refs["transitions"])
        )

    def add_state_group(self, widget_refs):
        layout = widget_refs["state_container"]
        group = QGroupBox(f"State {len(widget_refs['states']) + 1}")
        form = QFormLayout()

        name = QLineEdit()
        state_type = QComboBox()
        state_type.addItems(["fluorescent", "dark", "bleached"])
        form.addRow("Name:", name)
        form.addRow("State Type:", state_type)

        # === Parameter container ===
        param_container = QWidget()
        param_form = QFormLayout(param_container)

        quantum_yield = QDoubleSpinBox()
        quantum_yield.setRange(0, 1)
        quantum_yield.setDecimals(3)
        param_form.addRow("Quantum Yield:", quantum_yield)

        extinction = QDoubleSpinBox()
        extinction.setRange(0, 1e6)
        extinction.setSuffix(" M⁻¹cm⁻¹")
        param_form.addRow("Extinction Coefficient:", extinction)

        lifetime = QDoubleSpinBox()
        lifetime.setRange(0, 100)
        lifetime.setSuffix(" s")
        param_form.addRow("Fluorescent Lifetime:", lifetime)

        excitation_spectrum_button = QPushButton("Edit Spectrum")
        param_form.addRow("Excitation Spectrum:", excitation_spectrum_button)
        excitation_spectrum_data = {"wavelengths": [], "intensities": []}
        excitation_spectrum_button.clicked.connect(
            lambda: self.edit_spectrum(excitation_spectrum_data)
        )

        emission_spectrum_button = QPushButton("Edit Spectrum")
        param_form.addRow("Emission Spectrum:", emission_spectrum_button)
        emission_spectrum_data = {"wavelengths": [], "intensities": []}
        emission_spectrum_button.clicked.connect(
            lambda: self.edit_spectrum(emission_spectrum_data)
        )

        # Add conditional param container to main form
        form.addRow(param_container)

        group.setLayout(form)
        layout.addWidget(group)

        # === Visibility logic ===
        def update_param_visibility(state: str):
            param_container.setVisible(state == "fluorescent")

        state_type.currentTextChanged.connect(update_param_visibility)
        update_param_visibility(state_type.currentText())

        # === Store all widgets ===
        widget_refs["states"].append(
            {
                "group": group,
                "name": name,
                "type": state_type,
                "param_container": param_container,
                "quantum_yield": quantum_yield,
                "extinction": extinction,
                "lifetime": lifetime,
                "excitation_spectrum_button": excitation_spectrum_button,
                "emission_spectrum_button": emission_spectrum_button,
                "excitation_spectrum_data": excitation_spectrum_data,
                "emission_spectrum_data": emission_spectrum_data,
            }
        )

    def add_transition_group(self, widget_refs):
        layout = widget_refs["transition_container"]
        group = QGroupBox(f"Transition {len(widget_refs['transitions']) + 1}")
        form = QFormLayout()

        from_state = QLineEdit()
        to_state = QLineEdit()
        photon_dependent = QComboBox()
        photon_dependent.addItems(["True", "False"])
        base_rate = QDoubleSpinBox()
        base_rate.setRange(0, 1e6)
        base_rate.setSuffix(" 1/s")

        form.addRow("From State:", from_state)
        form.addRow("To State:", to_state)
        form.addRow("Photon Dependent:", photon_dependent)
        form.addRow("Base Rate:", base_rate)

        # === Spectrum container ===
        spectrum_container = QWidget()
        spectrum_form = QFormLayout(spectrum_container)

        activation_spectrum_button = QPushButton("Edit Spectrum")
        spectrum_form.addRow("Activation Spectrum:", activation_spectrum_button)
        activation_spectrum_data = {"wavelengths": [], "intensities": []}
        activation_spectrum_button.clicked.connect(
            lambda: self.edit_spectrum(activation_spectrum_data)
        )

        # Add to form
        form.addRow(spectrum_container)
        group.setLayout(form)
        layout.addWidget(group)

        # === Visibility logic ===
        def update_spectrum_visibility(val: str):
            spectrum_container.setVisible(val == "True")

        photon_dependent.currentTextChanged.connect(update_spectrum_visibility)
        update_spectrum_visibility(photon_dependent.currentText())

        # === Store everything ===
        widget_refs["transitions"].append(
            {
                "group": group,
                "from_state": from_state,
                "to_state": to_state,
                "photon_dependent": photon_dependent,
                "base_rate": base_rate,
                "spectrum_container": spectrum_container,
                "activation_spectrum_button": activation_spectrum_button,
                "activation_spectrum_data": activation_spectrum_data,
            }
        )

    def remove_last_group(self, group_list):
        if group_list:
            widget = group_list.pop()
            widget["group"].deleteLater()

    def edit_spectrum(self, spectrum_data):
        dialog = SpectrumEditorDialog(
            parent=self,
            wavelengths=spectrum_data.get("wavelengths", []),
            intensities=spectrum_data.get("intensities", []),
        )

        if dialog.exec():
            spectrum_data["wavelengths"] = dialog.wavelengths
            spectrum_data["intensities"] = dialog.intensities

    def get_data(self):
        # Placeholder: implement data collection matching your model
        return {}

    def validate(self) -> bool:
        try:
            data = self.get_data()
            # validated = FluorophoreParameters(**data)
            QMessageBox.information(
                self, "Validation Successful", "Parameters are valid."
            )
            return True
        except ValidationError as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            return False
