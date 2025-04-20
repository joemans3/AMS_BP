from PyQt6.QtWidgets import QMessageBox, QPushButton, QTabWidget, QVBoxLayout, QWidget

from .help_window import HelpWindow
from .widgets.camera_config_widget import CameraConfigWidget
from .widgets.cell_config_widget import CellConfigWidget
from .widgets.channel_config_widget import ChannelConfigWidget
from .widgets.condensate_config_widget import CondensateConfigWidget
from .widgets.flurophore_config_widget import FluorophoreConfigWidget
from .widgets.general_config_widget import GeneralConfigWidget
from .widgets.global_config_widget import GlobalConfigWidget
from .widgets.laser_config_widget import LaserConfigWidget
from .widgets.molecule_config_widget import MoleculeConfigWidget
from .widgets.output_config_widget import OutputConfigWidget
from .widgets.psf_config_widget import PSFConfigWidget


class ConfigEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulation Configuration Editor")

        # Create the main layout
        layout = QVBoxLayout()

        # Create the tab widget and individual tab widgets
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(
            QTabWidget.TabPosition.North
        )  # Adjust tab position (Top)
        self.tabs.setTabsClosable(False)  # Optionally, disable closing tabs

        # Create and add tabs for each section
        self.general_tab = GeneralConfigWidget()
        self.global_tab = GlobalConfigWidget()
        self.cell_tab = CellConfigWidget()
        self.molecule_tab = MoleculeConfigWidget()
        self.condensate_tab = CondensateConfigWidget()
        self.output_tab = OutputConfigWidget()
        self.fluorophore_tab = FluorophoreConfigWidget()
        self.psf_tab = PSFConfigWidget()
        self.laser_tab = LaserConfigWidget()
        self.channel_tab = ChannelConfigWidget()
        self.detector_tab = CameraConfigWidget()

        self.tabs.addTab(self.general_tab, "General")
        self.tabs.addTab(self.global_tab, "Global Parameters")
        self.tabs.addTab(self.cell_tab, "Cell Parameters")
        self.tabs.addTab(self.molecule_tab, "Molecule Parameters")
        self.tabs.addTab(self.condensate_tab, "Condensate Parameters")
        self.tabs.addTab(self.fluorophore_tab, "Define Flurophores")
        self.tabs.addTab(self.detector_tab, "Camera Parameters")
        self.tabs.addTab(self.psf_tab, "PSF Parameters")
        self.tabs.addTab(self.laser_tab, "Laser Parameters")
        self.tabs.addTab(self.channel_tab, "Channels Parameters")
        self.tabs.addTab(self.output_tab, "Saving Instructions")

        # Optionally, add some spacing to the layout for a cleaner appearance
        layout.addWidget(self.tabs)

        # Create and add the save and help buttons
        self.save_button = QPushButton("Save Configuration")
        self.save_button.clicked.connect(self.save_config)
        layout.addWidget(self.save_button)

        self.help_button = QPushButton("Help")
        self.help_button.clicked.connect(self.show_help)
        layout.addWidget(self.help_button)

        # Set layout for the main window
        self.setLayout(layout)

    def show_help(self):
        current_widget = self.tabs.currentWidget()
        if hasattr(current_widget, "get_help_path"):
            help_path = current_widget.get_help_path()
            if help_path.exists():
                help_window = HelpWindow(help_path, self)
                help_window.exec()
                return

        QMessageBox.warning(self, "Help", "Help content not found for this section.")

    def save_config(self):
        """Collect data from all tabs and save the configuration."""
        try:
            # Validate all tabs first
            global_valid = self.global_tab.validate()
            cell_valid = self.cell_tab.validate()
            molecule_valid = self.molecule_tab.validate()
            condensate_valid = self.condensate_tab.validate()

            if all([global_valid, cell_valid, molecule_valid, condensate_valid]):
                # If all validations pass, collect the data
                general_data = self.general_tab.get_data()
                global_data = self.global_tab.get_data()
                cell_data = self.cell_tab.get_data()
                molecule_data = self.molecule_tab.get_data()
                condensate_data = self.condensate_tab.get_data()

                # Combine into a complete configuration
                config = {
                    **general_data,
                    "global_params": global_data,
                    "cell": cell_data,
                    "molecule": molecule_data,
                    "condensate": condensate_data,
                }

                print("Config to save:", config)
                # TODO: Replace with actual file saving code

                QMessageBox.information(
                    self, "Success", "Configuration has been saved successfully."
                )
            else:
                QMessageBox.warning(
                    self,
                    "Validation Error",
                    "Please correct the errors in all tabs before saving.",
                )
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"An error occurred while saving: {str(e)}"
            )
