from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...core.configio.convertconfig import create_lasers_from_config


class LaserConfigWidget(QWidget):
    laser_names_updated = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.laser_name_widgets = []
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        form = QFormLayout()

        self.num_lasers = QSpinBox()
        self.num_lasers.setRange(1, 10)
        self.num_lasers.setValue(2)
        self.num_lasers.valueChanged.connect(self.update_laser_tabs)
        form.addRow("Number of Lasers:", self.num_lasers)

        self.laser_widgets = []

        self.laser_tabs = QTabWidget()
        self.update_laser_tabs()
        self.emit_active_lasers()

        layout.addLayout(form)
        layout.addWidget(self.laser_tabs)
        self.setLayout(layout)

        self.validate_button = QPushButton("Validate Parameters")
        self.validate_button.clicked.connect(self.validate)
        layout.addWidget(self.validate_button)

    def set_confocal_mode(self, enabled: bool):
        for i in range(self.laser_tabs.count()):
            tab = self.laser_tabs.widget(i)

            laser_type: QComboBox = tab.findChild(QComboBox)
            beam_width: QDoubleSpinBox = tab.findChildren(QDoubleSpinBox)[
                1
            ]  # Assumes 2nd spinbox is beam_width

            if enabled:
                # Force laser type to "gaussian" and disable editing
                laser_type.setCurrentText("gaussian")
                laser_type.setEnabled(False)

                # Hide beam width
                beam_width.hide()
                label = tab.layout().labelForField(beam_width)
                if label:
                    label.hide()
            else:
                # Enable laser type editing
                laser_type.setEnabled(True)

                # Show beam width
                beam_width.show()
                label = tab.layout().labelForField(beam_width)
                if label:
                    label.show()

    def validate(self) -> bool:
        try:
            data = self.get_data()
            create_lasers_from_config({"lasers": data})

            QMessageBox.information(
                self, "Validation Successful", "Laser parameters are valid."
            )
            return True

        except ValueError as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            return False
        except Exception as e:
            QMessageBox.critical(self, "Unexpected Error", str(e))
            return False

    def emit_active_lasers(self):
        names = []
        for i in range(self.laser_tabs.count()):
            tab = self.laser_tabs.widget(i)
            name_field = tab.findChild(QLineEdit)
            if name_field:
                names.append(name_field.text())
        self.laser_names_updated.emit([n for n in names if n])

    def update_laser_tabs(self):
        self.laser_tabs.clear()
        self.laser_widgets = []

        for i in range(self.num_lasers.value()):
            self.add_laser_tab(i)

        self.emit_active_lasers()

    def add_laser_tab(self, index):
        tab = QWidget()
        layout = QFormLayout()

        # Create widgets
        laser_name = QLineEdit()
        laser_type = QComboBox()
        laser_type.addItems(["widefield", "gaussian", "hilo"])
        laser_preset = QLineEdit()
        power = QDoubleSpinBox()
        power.setRange(0, 100000)
        wavelength = QSpinBox()
        wavelength.setRange(100, 10000)
        beam_width = QDoubleSpinBox()
        beam_width.setRange(0, 1000)
        numerical_aperture = QDoubleSpinBox()
        numerical_aperture.setRange(0, 2)
        refractive_index = QDoubleSpinBox()
        refractive_index.setRange(1, 2)
        inclination_angle = QDoubleSpinBox()
        inclination_angle.setRange(0, 90)
        inclination_angle.setEnabled(False)

        # Form layout
        layout.addRow(f"Laser {index + 1} Name:", laser_name)
        layout.addRow(f"Laser {index + 1} Type:", laser_type)
        layout.addRow(f"Laser {index + 1} Preset:", laser_preset)
        layout.addRow(f"Laser {index + 1} Power (W):", power)
        layout.addRow(f"Laser {index + 1} Wavelength (nm):", wavelength)
        layout.addRow(f"Laser {index + 1} Beam Width (µm):", beam_width)
        layout.addRow(f"Laser {index + 1} Numerical Aperture:", numerical_aperture)
        layout.addRow(f"Laser {index + 1} Refractive Index:", refractive_index)
        layout.addRow(f"Laser {index + 1} Inclination Angle (°):", inclination_angle)

        # Logic for hilo
        laser_type.currentTextChanged.connect(
            lambda val: inclination_angle.setEnabled(val == "hilo")
        )
        inclination_angle.setEnabled(laser_type.currentText() == "hilo")

        tab.setLayout(layout)
        self.laser_tabs.addTab(tab, f"Laser {index + 1}")

        # Track widgets
        self.laser_widgets.append(
            {
                "name": laser_name,
                "type": laser_type,
                "preset": laser_preset,
                "power": power,
                "wavelength": wavelength,
                "beam_width": beam_width,
                "numerical_aperture": numerical_aperture,
                "refractive_index": refractive_index,
                "inclination_angle": inclination_angle,
            }
        )

        # Update signal
        laser_name.textChanged.connect(self.emit_active_lasers)

    def toggle_inclination(self, laser_type, inclination_angle):
        """Enable or disable inclination angle field based on laser type."""
        if laser_type.currentText() == "hilo":
            inclination_angle.setEnabled(True)
        else:
            inclination_angle.setEnabled(False)

    def set_data(self, data: dict):
        # Determine how many lasers we need to configure
        active_names = data.get("active", [])
        self.num_lasers.setValue(len(active_names))
        self.update_laser_tabs()

        for i, name in enumerate(active_names):
            if i >= len(self.laser_widgets):
                break  # Extra safety

            widgets = self.laser_widgets[i]
            widgets["name"].setText(name)

            laser_info = data.get(name, {})
            widgets["type"].setCurrentText(laser_info.get("type", "widefield"))
            widgets["preset"].setText(laser_info.get("preset", ""))

            params = laser_info.get("parameters", {})

            widgets["power"].setValue(params.get("power", 0.0))
            widgets["wavelength"].setValue(params.get("wavelength", 488))
            widgets["beam_width"].setValue(params.get("beam_width", 1.0))
            widgets["numerical_aperture"].setValue(
                params.get("numerical_aperture", 1.4)
            )
            widgets["refractive_index"].setValue(params.get("refractive_index", 1.518))

            if laser_info.get("type") == "hilo":
                widgets["inclination_angle"].setValue(
                    params.get("inclination_angle", 45.0)
                )
                widgets["inclination_angle"].setEnabled(True)
            else:
                widgets["inclination_angle"].setEnabled(False)

        self.emit_active_lasers()

    def get_data(self) -> dict:
        lasers_section = {}
        active_names = []

        for widgets in self.laser_widgets:
            name = widgets["name"].text().strip()
            if not name:
                raise ValueError("All lasers must have a name.")

            active_names.append(name)

            laser_type = widgets["type"].currentText()
            preset = widgets["preset"].text().strip()

            params = {
                "power": widgets["power"].value(),
                "wavelength": widgets["wavelength"].value(),
                "beam_width": widgets["beam_width"].value(),
                "numerical_aperture": widgets["numerical_aperture"].value(),
                "refractive_index": widgets["refractive_index"].value(),
            }

            if laser_type == "hilo":
                params["inclination_angle"] = widgets["inclination_angle"].value()

            lasers_section[name] = {
                "type": laser_type,
                "preset": preset,
                "parameters": params,
            }

        return {
            "active": active_names,
            **lasers_section,
        }

    def get_help_path(self) -> Path:
        return Path(__file__).parent.parent / "help_docs" / "laser_help.md"
