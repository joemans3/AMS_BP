from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...core.configio.convertconfig import create_fluorophores_from_config
from .utility_widgets.scinotation_widget import scientific_input_field
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
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

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

        quantum_yield = scientific_input_field(0, 1, default=0.8)
        param_form.addRow("Quantum Yield:", quantum_yield)

        extinction = scientific_input_field(-1e12, 1e12, default=1e5)
        param_form.addRow("Extinction Coefficient:", extinction)

        lifetime = scientific_input_field(1e-10, 1, default=1e-8)
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
        base_rate = scientific_input_field(0, 1e6, default=1000.0)
        form.addRow("From State:", from_state)
        form.addRow("To State:", to_state)
        form.addRow("Photon Dependent:", photon_dependent)
        form.addRow("Base Rate:", base_rate)

        # === Spectrum container ===
        spectrum_container = QWidget()
        spectrum_form = QFormLayout(spectrum_container)

        activation_spectrum_button = QPushButton("Edit Spectrum")
        activation_spectrum_data = {"wavelengths": [], "intensities": []}
        activation_spectrum_button.clicked.connect(
            lambda: self.edit_spectrum(activation_spectrum_data)
        )
        spectrum_form.addRow("Activation Spectrum:", activation_spectrum_button)

        quantum_yield = scientific_input_field(0, 1, default=0.7)
        spectrum_form.addRow("Quantum Yield:", quantum_yield)

        extinction_coefficient = scientific_input_field(-1e12, 1e12, default=1e5)
        spectrum_form.addRow("Extinction Coefficient:", extinction_coefficient)
        form.addRow(spectrum_container)

        group.setLayout(form)
        layout.addWidget(group)

        # === Visibility logic ===
        def update_visibility(val: str):
            is_photon = val == "True"
            spectrum_container.setVisible(is_photon)
            base_rate.setVisible(not is_photon)

        photon_dependent.currentTextChanged.connect(update_visibility)
        update_visibility(photon_dependent.currentText())  # initial state

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
                "quantum_yield": quantum_yield,
                "extinction_coefficient": extinction_coefficient,
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

    def get_data(self) -> dict:
        data = {
            "num_of_fluorophores": len(self.fluorophore_widgets),
            "fluorophore_names": [],
        }

        for fluor in self.fluorophore_widgets:
            name = fluor["name"].text().strip()
            if not name:
                raise ValueError("Each fluorophore must have a name.")

            data["fluorophore_names"].append(name)

            fluor_data = {
                "name": name,
                "initial_state": fluor["initial_state"].text().strip(),
                "states": {},
                "transitions": {},
            }

            # States
            for state in fluor["states"]:
                state_name = state["name"].text().strip()
                state_type = state["type"].currentText()

                state_data = {
                    "name": state_name,
                    "state_type": state_type,
                }

                if state_type == "fluorescent":
                    state_data.update(
                        {
                            "quantum_yield": float(state["quantum_yield"].text()),
                            "extinction_coefficient": float(state["extinction"].text()),
                            "fluorescent_lifetime": float(state["lifetime"].text()),
                            "excitation_spectrum": {
                                "wavelengths": state["excitation_spectrum_data"][
                                    "wavelengths"
                                ],
                                "intensities": state["excitation_spectrum_data"][
                                    "intensities"
                                ],
                            },
                            "emission_spectrum": {
                                "wavelengths": state["emission_spectrum_data"][
                                    "wavelengths"
                                ],
                                "intensities": state["emission_spectrum_data"][
                                    "intensities"
                                ],
                            },
                        }
                    )

                fluor_data["states"][state_name] = state_data

            # Transitions
            for trans in fluor["transitions"]:
                from_state = trans["from_state"].text().strip()
                to_state = trans["to_state"].text().strip()
                key = f"{from_state}_to_{to_state}"

                photon_dependent = trans["photon_dependent"].currentText() == "True"
                transition_data = {
                    "from_state": from_state,
                    "to_state": to_state,
                    "photon_dependent": photon_dependent,
                }

                if photon_dependent:
                    transition_data["spectrum"] = {
                        "wavelengths": trans["activation_spectrum_data"]["wavelengths"],
                        "intensities": trans["activation_spectrum_data"]["intensities"],
                        "extinction_coefficient": float(
                            trans["extinction_coefficient"].text()
                        ),
                        "quantum_yield": float(trans["quantum_yield"].text()),
                    }
                else:
                    transition_data["base_rate"] = float(trans["base_rate"].text())

                fluor_data["transitions"][key] = transition_data

            data[name] = fluor_data

        return data

    def set_data(self, data: dict):
        """Populate the UI from TOML-based fluorophore config data."""
        fluor_names = data.get("fluorophore_names", [])
        self.set_mfluorophore_count(len(fluor_names))

        for i, name in enumerate(fluor_names):
            fluor_data = data.get(name, {})
            widget_refs = self.fluorophore_widgets[i]
            widget_refs["name"].setText(name)

            widget_refs["initial_state"].setText(fluor_data.get("initial_state", {}))

            # === Load States ===
            widget_refs["group"] = []
            states = fluor_data.get("states", {})
            for j, (state_name, state_data) in enumerate(states.items()):
                if j >= len(widget_refs["states"]):
                    self.add_state_group(widget_refs)
                state_widget = widget_refs["states"][j]

                state_widget["name"].setText(state_data["name"])
                idx = state_widget["type"].findText(state_data["state_type"])
                if idx != -1:
                    state_widget["type"].setCurrentIndex(idx)

                if state_data["state_type"] == "fluorescent":
                    state_widget["quantum_yield"].setText(
                        str(state_data.get("quantum_yield", 0.0))
                    )
                    state_widget["extinction"].setText(
                        str(state_data.get("extinction_coefficient", 0.0))
                    )
                    state_widget["lifetime"].setText(
                        str(state_data.get("fluorescent_lifetime", 0.0))
                    )

                    # Spectra
                    state_widget["excitation_spectrum_data"]["wavelengths"] = (
                        state_data.get("excitation_spectrum", {}).get("wavelengths", [])
                    )
                    state_widget["excitation_spectrum_data"]["intensities"] = (
                        state_data.get("excitation_spectrum", {}).get("intensities", [])
                    )

                    state_widget["emission_spectrum_data"]["wavelengths"] = (
                        state_data.get("emission_spectrum", {}).get("wavelengths", [])
                    )
                    state_widget["emission_spectrum_data"]["intensities"] = (
                        state_data.get("emission_spectrum", {}).get("intensities", [])
                    )

            # === Load Transitions ===
            transitions = fluor_data.get("transitions", {})
            for j, (key, trans_data) in enumerate(transitions.items()):
                if j >= len(widget_refs["transitions"]):
                    self.add_transition_group(widget_refs)
                trans_widget = widget_refs["transitions"][j]

                trans_widget["from_state"].setText(trans_data["from_state"])
                trans_widget["to_state"].setText(trans_data["to_state"])
                is_photon = trans_data.get("photon_dependent", False)
                idx = trans_widget["photon_dependent"].findText(str(is_photon))
                if idx != -1:
                    trans_widget["photon_dependent"].setCurrentIndex(idx)

                if is_photon:
                    spectrum = trans_data.get("spectrum", {})
                    trans_widget["activation_spectrum_data"]["wavelengths"] = (
                        spectrum.get("wavelengths", [])
                    )
                    trans_widget["activation_spectrum_data"]["intensities"] = (
                        spectrum.get("intensities", [])
                    )
                    trans_widget["extinction_coefficient"].setText(
                        str(spectrum.get("extinction_coefficient", 0.0))
                    )
                    trans_widget["quantum_yield"].setText(
                        str(spectrum.get("quantum_yield", 0.0))
                    )
                else:
                    trans_widget["base_rate"].setText(
                        str(trans_data.get("base_rate", 0.0))
                    )

    def validate(self) -> bool:
        try:
            data = self.get_data()
            # Try to build fluorophores with the backend logic

            create_fluorophores_from_config({"fluorophores": data})

            QMessageBox.information(
                self, "Validation Successful", "Fluorophore parameters are valid."
            )
            return True

        except ValueError as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            return False
        except Exception as e:
            QMessageBox.critical(self, "Unexpected Error", str(e))
            return False

    def get_help_path(self) -> Path:
        return Path(__file__).parent.parent / "help_docs" / "fluorophore_help.md"
