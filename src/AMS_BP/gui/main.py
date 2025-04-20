import tomlkit
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

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

        # Create the main layout for the window
        layout = QVBoxLayout()

        # Create a horizontal layout for the dropdown and the tab index label
        dropdown_layout = QHBoxLayout()

        # Add a QLabel for the instruction/title about the dropdown
        dropdown_title = QLabel(
            "Use the dropdown below to set the parameters for each tab:"
        )
        dropdown_layout.addWidget(dropdown_title)

        # Create a QComboBox (dropdown menu) for selecting tabs
        self.dropdown = QComboBox()
        self.dropdown.addItems(
            [
                "General",
                "Global Parameters",
                "Cell Parameters",
                "Molecule Parameters",
                "Condensate Parameters",
                "Define Fluorophores",
                "Camera Parameters",
                "PSF Parameters",
                "Laser Parameters",
                "Channels Parameters",
                "Saving Instructions",
            ]
        )
        self.dropdown.currentIndexChanged.connect(
            self.on_dropdown_change
        )  # Connect to the change event

        # Create a QLabel for displaying the current tab index
        self.tab_index_label = QLabel("1/10")

        # Add the dropdown and label to the layout
        dropdown_layout.addWidget(self.dropdown)
        dropdown_layout.addWidget(self.tab_index_label)

        # Add the dropdown layout to the main layout
        layout.addLayout(dropdown_layout)

        # Create a QStackedWidget to hold the content for each "tab"
        self.stacked_widget = QStackedWidget()

        # Initialize the widgets for each "tab"
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

        # Add each tab's widget to the stacked widget
        self.stacked_widget.addWidget(self.general_tab)
        self.stacked_widget.addWidget(self.global_tab)
        self.stacked_widget.addWidget(self.cell_tab)
        self.stacked_widget.addWidget(self.molecule_tab)
        self.stacked_widget.addWidget(self.condensate_tab)
        self.stacked_widget.addWidget(self.fluorophore_tab)
        self.stacked_widget.addWidget(self.detector_tab)
        self.stacked_widget.addWidget(self.psf_tab)
        self.stacked_widget.addWidget(self.laser_tab)
        self.stacked_widget.addWidget(self.channel_tab)
        self.stacked_widget.addWidget(self.output_tab)

        # Set the stacked widget as the central widget
        layout.addWidget(self.stacked_widget)

        # Create and add the save and help buttons at the bottom
        self.save_button = QPushButton("Save Configuration")
        self.save_button.clicked.connect(self.save_config)
        layout.addWidget(self.save_button)

        self.help_button = QPushButton("Help")
        self.help_button.clicked.connect(self.show_help)
        layout.addWidget(self.help_button)

        # Set the layout for the main window
        self.setLayout(layout)

        # Set initial display
        self.on_dropdown_change(0)  # Show the first tab (index 0)

    def on_dropdown_change(self, index):
        """Change the displayed widget based on the dropdown selection."""
        self.stacked_widget.setCurrentIndex(index)
        # Update the tab index label (1-based index)
        total_tabs = (
            self.dropdown.count()
        )  # Corrected way to get the total number of items
        self.tab_index_label.setText(f"{index + 1}/{total_tabs}")

    def show_help(self):
        current_widget = self.stacked_widget.currentWidget()
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

                # Open the file dialog to select where to save the file
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Configuration", "", "TOML Files (*.toml);;All Files (*)"
                )

                if file_path:
                    # Ensure the file has a .toml extension
                    if not file_path.endswith(".toml"):
                        file_path += ".toml"

                    # Use tomlkit to write the configuration to the file
                    # Create a TOML document with the provided configuration
                    toml_doc = tomlkit.document()
                    for key, value in config.items():
                        toml_doc[key] = value

                    # Write to the file
                    with open(file_path, "w") as f:
                        tomlkit.dump(toml_doc, f)

                    QMessageBox.information(
                        self, "Success", "Configuration has been saved successfully."
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Save Error",
                        "No file selected. The configuration was not saved.",
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
