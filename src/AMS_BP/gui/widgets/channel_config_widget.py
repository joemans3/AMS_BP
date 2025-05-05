from pathlib import Path

from pydantic import ValidationError
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

from ...core.configio.convertconfig import create_channels


class ChannelConfigWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.channel_widgets = []

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        form = QFormLayout()

        self.num_channels = QSpinBox()
        self.num_channels.setRange(1, 10)
        self.num_channels.setValue(2)
        self.num_channels.valueChanged.connect(self.update_channel_tabs)
        form.addRow("Number of Channels:", self.num_channels)

        self.channel_tabs = QTabWidget()
        self.update_channel_tabs()

        layout.addLayout(form)
        layout.addWidget(self.channel_tabs)
        self.setLayout(layout)

        self.validate_button = QPushButton("Validate Parameters")
        self.validate_button.clicked.connect(self.validate)
        layout.addWidget(self.validate_button)

    def validate(self) -> bool:
        try:
            data = self.get_data()

            # Full simulation-level validation
            channels = create_channels({"channels": data})

            if len(channels.names) != channels.num_channels:
                raise ValueError("Channel count does not match number of names.")

            QMessageBox.information(
                self, "Validation Successful", "Channel parameters are valid."
            )
            return True

        except (ValidationError, ValueError, KeyError) as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            return False
        except Exception as e:
            QMessageBox.critical(self, "Unexpected Error", str(e))
            return False

    def update_channel_tabs(self):
        self.channel_tabs.clear()
        self.channel_widgets = []

        for i in range(self.num_channels.value()):
            self.add_channel_tab(i)

        self.channel_tabs.setCurrentIndex(0)

    def add_channel_tab(self, index):
        tab = QWidget()
        layout = QFormLayout()

        # Channel name
        channel_name = QLineEdit()
        layout.addRow(f"Channel {index + 1} Name:", channel_name)

        # Split efficiency
        split_eff = QDoubleSpinBox()
        split_eff.setRange(0.0, 1.0)
        split_eff.setValue(1.0)
        layout.addRow(f"Channel {index + 1} Split Efficiency:", split_eff)

        # Excitation filter
        exc_name = QLineEdit()
        exc_type = QComboBox()
        exc_type.addItems(["bandpass", "allow_all"])
        exc_center = QSpinBox()
        exc_center.setRange(0, 10000)
        exc_bandwidth = QSpinBox()
        exc_bandwidth.setRange(0, 10000)
        exc_trans = QDoubleSpinBox()
        exc_trans.setRange(0.0, 1.0)
        exc_points = QSpinBox()
        exc_points.setRange(1, 10000)

        layout.addRow("Excitation Name:", exc_name)
        layout.addRow("Excitation Type:", exc_type)
        layout.addRow("Excitation Center (nm):", exc_center)
        layout.addRow("Excitation Bandwidth (nm):", exc_bandwidth)
        layout.addRow("Excitation Transmission Peak:", exc_trans)
        layout.addRow("Excitation Points:", exc_points)

        # Emission filter
        em_name = QLineEdit()
        em_type = QComboBox()
        em_type.addItems(["bandpass", "allow_all"])
        em_center = QSpinBox()
        em_center.setRange(0, 10000)
        em_bandwidth = QSpinBox()
        em_bandwidth.setRange(0, 10000)
        em_trans = QDoubleSpinBox()
        em_trans.setRange(0.0, 1.0)
        em_points = QSpinBox()
        em_points.setRange(1, 10000)

        exc_type.currentTextChanged.connect(
            lambda val: self.toggle_filter_fields(
                val, exc_center, exc_bandwidth, exc_trans, exc_points
            )
        )
        em_type.currentTextChanged.connect(
            lambda val: self.toggle_filter_fields(
                val, em_center, em_bandwidth, em_trans, em_points
            )
        )

        # Call once to initialize visibility
        self.toggle_filter_fields(
            exc_type.currentText(), exc_center, exc_bandwidth, exc_trans, exc_points
        )
        self.toggle_filter_fields(
            em_type.currentText(), em_center, em_bandwidth, em_trans, em_points
        )

        layout.addRow("Emission Name:", em_name)
        layout.addRow("Emission Type:", em_type)
        layout.addRow("Emission Center (nm):", em_center)
        layout.addRow("Emission Bandwidth (nm):", em_bandwidth)
        layout.addRow("Emission Transmission Peak:", em_trans)
        layout.addRow("Emission Points:", em_points)

        tab.setLayout(layout)
        self.channel_tabs.addTab(tab, f"Channel {index + 1}")

        # Store all widget references
        widgets = {
            "channel_name": channel_name,
            "split_efficiency": split_eff,
            "exc_name": exc_name,
            "exc_type": exc_type,
            "exc_center": exc_center,
            "exc_bandwidth": exc_bandwidth,
            "exc_trans": exc_trans,
            "exc_points": exc_points,
            "em_name": em_name,
            "em_type": em_type,
            "em_center": em_center,
            "em_bandwidth": em_bandwidth,
            "em_trans": em_trans,
            "em_points": em_points,
        }

        self.channel_widgets.append(widgets)

    def toggle_filter_fields(
        self,
        filter_type,
        center_field,
        bandwidth_field,
        transmission_field,
        points_field,
    ):
        is_allow_all = filter_type == "allow_all"
        center_field.setEnabled(not is_allow_all)
        bandwidth_field.setEnabled(not is_allow_all)
        transmission_field.setEnabled(not is_allow_all)
        points_field.setEnabled(not is_allow_all)

    def get_data(self) -> dict:
        num_channels = self.num_channels.value()
        data = {
            "num_of_channels": num_channels,
            "channel_names": [],
            "split_efficiency": [],
            "filters": {},
        }

        for i, widgets in enumerate(self.channel_widgets):
            name = widgets["channel_name"].text().strip()
            if not name:
                raise ValueError(f"Channel {i + 1} name is required.")

            data["channel_names"].append(name)
            data["split_efficiency"].append(widgets["split_efficiency"].value())

            # Excitation
            exc_type = widgets["exc_type"].currentText()
            excitation = {
                "name": widgets["exc_name"].text(),
                "type": exc_type,
            }
            if exc_type == "bandpass":
                excitation.update(
                    {
                        "center_wavelength": widgets["exc_center"].value(),
                        "bandwidth": widgets["exc_bandwidth"].value(),
                        "transmission_peak": widgets["exc_trans"].value(),
                        "points": widgets["exc_points"].value(),
                    }
                )
            else:
                excitation["points"] = widgets["exc_points"].value()

            # Emission
            em_type = widgets["em_type"].currentText()
            emission = {
                "name": widgets["em_name"].text(),
                "type": em_type,
            }
            if em_type == "bandpass":
                emission.update(
                    {
                        "center_wavelength": widgets["em_center"].value(),
                        "bandwidth": widgets["em_bandwidth"].value(),
                        "transmission_peak": widgets["em_trans"].value(),
                        "points": widgets["em_points"].value(),
                    }
                )
            else:
                emission["points"] = widgets["em_points"].value()

            data["filters"][name] = {
                "filter_set_name": f"{name.capitalize()} Filter Set",
                "filter_set_description": f"Sample {name.capitalize()} filter set configuration",
                "excitation": excitation,
                "emission": emission,
            }

        return data

    def set_data(self, data: dict):
        num_channels = data.get("num_of_channels", 0)
        self.num_channels.setValue(num_channels)
        self.update_channel_tabs()

        channel_names = data.get("channel_names", [])
        split_efficiencies = data.get("split_efficiency", [])
        filters = data.get("filters", {})

        for i, widgets in enumerate(self.channel_widgets):
            if i >= len(channel_names):
                break

            name = channel_names[i]
            widgets["channel_name"].setText(name)

            if i < len(split_efficiencies):
                widgets["split_efficiency"].setValue(split_efficiencies[i])

            filter_data = filters.get(name, {})

            # Excitation
            excitation = filter_data.get("excitation", {})
            widgets["exc_name"].setText(excitation.get("name", ""))
            widgets["exc_type"].setCurrentText(excitation.get("type", "bandpass"))

            widgets["exc_center"].setValue(excitation.get("center_wavelength", 0))
            widgets["exc_bandwidth"].setValue(excitation.get("bandwidth", 0))
            widgets["exc_trans"].setValue(excitation.get("transmission_peak", 0.0))
            widgets["exc_points"].setValue(excitation.get("points", 1))

            # Emission
            emission = filter_data.get("emission", {})
            widgets["em_name"].setText(emission.get("name", ""))
            widgets["em_type"].setCurrentText(emission.get("type", "bandpass"))

            widgets["em_center"].setValue(emission.get("center_wavelength", 0))
            widgets["em_bandwidth"].setValue(emission.get("bandwidth", 0))
            widgets["em_trans"].setValue(emission.get("transmission_peak", 0.0))
            widgets["em_points"].setValue(emission.get("points", 1))

            # Apply visibility logic
            self.toggle_filter_fields(
                widgets["exc_type"].currentText(),
                widgets["exc_center"],
                widgets["exc_bandwidth"],
                widgets["exc_trans"],
                widgets["exc_points"],
            )
            self.toggle_filter_fields(
                widgets["em_type"].currentText(),
                widgets["em_center"],
                widgets["em_bandwidth"],
                widgets["em_trans"],
                widgets["em_points"],
            )

    def get_help_path(self) -> Path:
        return Path(__file__).parent.parent / "help_docs" / "channels_help.md"
