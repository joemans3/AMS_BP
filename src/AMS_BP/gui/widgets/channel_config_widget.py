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

from ...optics.filters.channels.channelschema import Channels


class ChannelConfigWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        form = QFormLayout()

        self.validate_button = QPushButton("Validate")
        self.validate_button.clicked.connect(self.validate)
        layout.addWidget(self.validate_button)

        self.num_channels = QSpinBox()
        self.num_channels.setRange(1, 10)
        self.num_channels.setValue(2)
        self.num_channels.valueChanged.connect(self.update_channel_tabs)
        form.addRow("Number of Channels:", self.num_channels)

        self.channel_names = []
        self.split_efficiency = []

        self.channel_tabs = QTabWidget()
        self.update_channel_tabs()

        layout.addLayout(form)
        layout.addWidget(self.channel_tabs)
        self.setLayout(layout)

    def validate(self) -> bool:
        try:
            data = self.get_data()
            validated = Channels(**data)
            QMessageBox.information(
                self, "Validation Successful", "Channel parameters are valid."
            )
            return True
        except ValidationError as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            return False
        except ValueError as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            return False

    def update_channel_tabs(self):
        # Update the number of channels based on user input
        num_channels = self.num_channels.value()

        # Add tabs for each channel
        while self.channel_tabs.count() < num_channels:
            self.add_channel_tab(self.channel_tabs.count())

        while self.channel_tabs.count() > num_channels:
            self.channel_tabs.removeTab(self.channel_tabs.count() - 1)

        self.channel_tabs.setCurrentIndex(0)

    def add_channel_tab(self, index):
        tab = QWidget()
        layout = QFormLayout()

        # Channel name
        channel_name = QLineEdit()
        layout.addRow(f"Channel {index + 1} Name:", channel_name)

        # Split efficiency
        split_efficiency = QDoubleSpinBox()
        split_efficiency.setRange(0, 1)
        split_efficiency.setValue(1.0)
        layout.addRow(f"Channel {index + 1} Split Efficiency:", split_efficiency)

        # Filter set configuration (Excitation and Emission filters)
        filters_layout = QFormLayout()

        # Excitation Filter
        exc_name = QLineEdit()
        exc_type = QComboBox()
        exc_type.addItems(["allow_all", "bandpass"])
        exc_points = QSpinBox()
        exc_center = QSpinBox()
        exc_bandwidth = QSpinBox()
        exc_bandwidth.setRange(0, 10000)
        exc_transmission = QDoubleSpinBox()
        exc_transmission.setRange(0, 1)
        exc_center.setRange(0, 10000)
        exc_points.setRange(1, 10000)
        exc_type.currentIndexChanged.connect(
            lambda: self.toggle_filter_fields(
                exc_type, exc_points, exc_center, exc_bandwidth, exc_transmission
            )
        )
        layout.addRow(f"Excitation Filter Name (Channel {index + 1}):", exc_name)
        layout.addRow(f"Excitation Filter Type (Channel {index + 1}):", exc_type)
        layout.addRow(f"Excitation Filter Center (Channel {index + 1}):", exc_center)
        layout.addRow(
            f"Excitation Filter Bandwidth (Channel {index + 1}):", exc_bandwidth
        )
        layout.addRow(
            f"Excitation Filter Transmission Peak (Channel {index + 1}):",
            exc_transmission,
        )
        layout.addRow(f"Excitation Filter Points (Channel {index + 1}):", exc_points)

        # Emission Filter
        em_name = QLineEdit()
        em_type = QComboBox()
        em_type.addItems(["allow_all", "bandpass"])
        em_center = QSpinBox()
        em_center.setRange(0, 10000)
        em_bandwidth = QSpinBox()
        em_bandwidth.setRange(0, 10000)
        em_transmission = QDoubleSpinBox()
        em_transmission.setRange(0, 1)
        em_points = QSpinBox()
        em_points.setRange(1, 10000)
        em_type.currentIndexChanged.connect(
            lambda: self.toggle_filter_fields(
                em_type, em_points, em_center, em_bandwidth, em_transmission
            )
        )
        layout.addRow(f"Emission Filter Name (Channel {index + 1}):", em_name)
        layout.addRow(f"Emission Filter Type (Channel {index + 1}):", em_type)
        layout.addRow(f"Emission Filter Center (Channel {index + 1}):", em_center)
        layout.addRow(f"Emission Filter Bandwidth (Channel {index + 1}):", em_bandwidth)
        layout.addRow(
            f"Emission Filter Transmission Peak (Channel {index + 1}):", em_transmission
        )
        layout.addRow(f"Emission Filter Points (Channel {index + 1}):", em_points)

        tab.setLayout(layout)
        self.channel_tabs.addTab(tab, f"Channel {index + 1}")

        # Store the widget references for later validation
        self.channel_names.append(channel_name)
        self.split_efficiency.append(split_efficiency)

    def toggle_filter_fields(
        self,
        filter_type,
        points_field,
        center_field=None,
        bandwidth_field=None,
        transmission_field=None,
    ):
        """Toggle visibility of filter fields based on selected filter type."""
        is_allow_all = filter_type.currentText() == "allow_all"
        if center_field:
            center_field.setEnabled(not is_allow_all)
        if bandwidth_field:
            bandwidth_field.setEnabled(not is_allow_all)
        if transmission_field:
            transmission_field.setEnabled(not is_allow_all)
        points_field.setEnabled(not is_allow_all)

    def get_data(self):
        channels_data = []

        for i in range(self.num_channels.value()):
            channel_data = {
                "name": self.channel_names[i].text(),
                "split_efficiency": self.split_efficiency[i].value(),
                "filters": {
                    "excitation": {
                        "name": self.channel_tabs.widget(i).findChild(QLineEdit).text(),
                        "type": self.channel_tabs.widget(i)
                        .findChild(QComboBox)
                        .currentText(),
                        "points": self.channel_tabs.widget(i)
                        .findChild(QSpinBox)
                        .value(),
                    },
                    "emission": {
                        "name": self.channel_tabs.widget(i)
                        .findChild(QLineEdit, "em_name")
                        .text(),
                        "type": self.channel_tabs.widget(i)
                        .findChild(QComboBox, "em_type")
                        .currentText(),
                        "center_wavelength": self.channel_tabs.widget(i)
                        .findChild(QSpinBox, "em_center")
                        .value(),
                        "bandwidth": self.channel_tabs.widget(i)
                        .findChild(QSpinBox, "em_bandwidth")
                        .value(),
                        "transmission_peak": self.channel_tabs.widget(i)
                        .findChild(QDoubleSpinBox, "em_transmission")
                        .value(),
                        "points": self.channel_tabs.widget(i)
                        .findChild(QSpinBox, "em_points")
                        .value(),
                    },
                },
            }

            channels_data.append(channel_data)

        return {"num_of_channels": self.num_channels.value(), "channels": channels_data}
