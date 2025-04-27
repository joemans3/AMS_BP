from pathlib import Path

from PyQt6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...core.configio.configmodels import GlobalParameters


class GlobalConfigWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        form = QFormLayout()

        # Sample plane dimensions
        self.sample_plane_width = QSpinBox()
        self.sample_plane_width.setRange(1, 1000)
        self.sample_plane_width.setSuffix(" μm")

        self.sample_plane_height = QSpinBox()
        self.sample_plane_height.setRange(1, 1000)
        self.sample_plane_height.setSuffix(" μm")

        plane_dim_layout = QHBoxLayout()
        plane_dim_layout.addWidget(self.sample_plane_width)
        plane_dim_layout.addWidget(self.sample_plane_height)
        plane_dim_container = QWidget()
        plane_dim_container.setLayout(plane_dim_layout)
        form.addRow("Sample Plane Dimensions (W × H):", plane_dim_container)

        # Cycle count
        self.cycle_count = QSpinBox()
        self.cycle_count.setRange(1, 1000)
        form.addRow("Cycle Count:", self.cycle_count)

        # Exposure time
        self.exposure_time = QSpinBox()
        self.exposure_time.setRange(1, 1000)
        self.exposure_time.setSuffix(" ms")
        form.addRow("Exposure Time:", self.exposure_time)

        # Interval time
        self.interval_time = QSpinBox()
        self.interval_time.setRange(0, 1000)
        self.interval_time.setSuffix(" ms")
        form.addRow("Interval Time:", self.interval_time)

        # Oversample motion time
        self.oversample_motion_time = QSpinBox()
        self.oversample_motion_time.setRange(0, 1000)
        self.oversample_motion_time.setSuffix(" ms")
        form.addRow("Oversample Motion Time:", self.oversample_motion_time)

        # Add form to main layout
        layout.addLayout(form)

        # Set default values
        self.set_defaults()

        self.setLayout(layout)

        # Validation button
        self.validate_button = QPushButton("Validate Parameters")
        self.validate_button.clicked.connect(self.validate)
        layout.addWidget(self.validate_button)

    def set_defaults(self):
        """Set default values for the form fields"""
        self.sample_plane_width.setValue(50)  # 1000 μm
        self.sample_plane_height.setValue(50)  # 1000 μm
        self.cycle_count.setValue(5)
        self.exposure_time.setValue(100)  # 100 ms
        self.interval_time.setValue(0)  # 50 ms
        self.oversample_motion_time.setValue(1)  # 10 ms

    def set_data(self, data: dict):
        """
        Populate the global parameter fields using data from a config dictionary.
        """
        try:
            # Sample plane dimensions
            dims = data.get("sample_plane_dim", [50, 50])
            if isinstance(dims, list) and len(dims) == 2:
                self.sample_plane_width.setValue(dims[0])
                self.sample_plane_height.setValue(dims[1])

            # Other scalar fields
            self.cycle_count.setValue(data.get("cycle_count", 5))
            self.exposure_time.setValue(data.get("exposure_time", 100))
            self.interval_time.setValue(data.get("interval_time", 0))
            self.oversample_motion_time.setValue(data.get("oversample_motion_time", 1))

        except Exception as e:
            print(f"[GlobalConfigWidget] Failed to load config: {e}")

    def get_data(self):
        """Collect all form data and return as a dictionary"""
        return {
            "sample_plane_dim": [
                self.sample_plane_width.value(),
                self.sample_plane_height.value(),
            ],
            "cycle_count": self.cycle_count.value(),
            "exposure_time": self.exposure_time.value(),
            "interval_time": self.interval_time.value(),
            "oversample_motion_time": self.oversample_motion_time.value(),
        }

    def validate(self) -> bool:
        try:
            data = self.get_data()

            GlobalParameters(**data)

            QMessageBox.information(
                self, "Validation Successful", "Global parameters are valid."
            )
            return True

        except TypeError as e:
            QMessageBox.critical(
                self, "Validation Error", f"Missing or invalid fields: {str(e)}"
            )
            return False
        except ValueError as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            return False
        except Exception as e:
            QMessageBox.critical(self, "Unexpected Error", str(e))
            return False

    def get_help_path(self) -> Path:
        return Path(__file__).parent.parent / "help_docs" / "global_help.md"
