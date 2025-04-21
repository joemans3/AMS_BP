import tomlkit
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .help_window import HelpWindow
from .widgets.camera_config_widget import CameraConfigWidget
from .widgets.cell_config_widget import CellConfigWidget
from .widgets.channel_config_widget import ChannelConfigWidget
from .widgets.condensate_config_widget import CondensateConfigWidget
from .widgets.experiment_config_widget import ExperimentConfigWidget
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
                "Experiment Builder",
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
        self.experiment_tab = ExperimentConfigWidget()

        # connections
        # PSF -> confocal -> lasers
        self.psf_tab.confocal_mode_changed.connect(self.laser_tab.set_confocal_mode)
        # === Molecule -> Fluorophore & Condensate ===
        self.molecule_tab.molecule_count_changed.connect(
            self.fluorophore_tab.set_mfluorophore_count
        )
        self.molecule_tab.molecule_count_changed.connect(
            self.condensate_tab.set_molecule_count
        )
        # === Fluorophore -> Molecule & Condensate ===
        self.fluorophore_tab.mfluorophore_count_changed.connect(
            self.molecule_tab.set_molecule_count
        )
        self.fluorophore_tab.mfluorophore_count_changed.connect(
            self.condensate_tab.set_molecule_count
        )
        # === Condensate -> Molecule & Fluorophore ===
        self.condensate_tab.molecule_count_changed.connect(
            self.molecule_tab.set_molecule_count
        )
        self.condensate_tab.molecule_count_changed.connect(
            self.fluorophore_tab.set_mfluorophore_count
        )
        # === Laser -> Experiment
        self.laser_tab.laser_names_updated.connect(
            self.experiment_tab.set_active_lasers
        )

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
        self.stacked_widget.addWidget(self.experiment_tab)

        # Set the stacked widget as the central widget
        layout.addWidget(self.stacked_widget)

        # Create and add the save and help buttons at the bottom
        self.save_button = QPushButton("Ready to save configuration?")
        self.save_button.clicked.connect(self.save_config)
        layout.addWidget(self.save_button)

        self.preview_button = QPushButton("Preview Configuration TOML")
        self.preview_button.clicked.connect(self.preview_config)
        layout.addWidget(self.preview_button)

        self.help_button = QPushButton("Get Help on this section")
        self.help_button.clicked.connect(self.show_help)
        layout.addWidget(self.help_button)

        # Set the layout for the main window
        self.setLayout(layout)

        # Set initial display
        self.on_dropdown_change(0)  # Show the first tab (index 0)

    def preview_config(self):
        """Preview the full TOML config in a dialog before saving."""
        try:
            # Step 1: Validate tabs
            if not self.validate_all_tabs():
                QMessageBox.warning(
                    self, "Validation Error", "Fix errors before preview."
                )
                return

            # Step 2: Collect config data
            config = self.collect_all_config()

            # Step 3: Convert to TOML string
            toml_doc = tomlkit.document()
            for key, value in config.items():
                toml_doc[key] = value
            toml_str = tomlkit.dumps(toml_doc)

            # Step 4: Show preview dialog
            preview_dialog = QDialog(self)
            preview_dialog.setWindowTitle("Preview TOML Configuration")
            layout = QVBoxLayout(preview_dialog)
            text_edit = QTextEdit()
            text_edit.setPlainText(toml_str)
            text_edit.setReadOnly(True)
            layout.addWidget(text_edit)
            preview_dialog.resize(800, 600)
            preview_dialog.exec()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Preview failed: {str(e)}")

    def build_toml_doc(self, config: dict) -> tomlkit.TOMLDocument:
        """Build a TOML document from the configuration dictionary."""
        doc = tomlkit.document()
        for key, val in config.items():
            doc[key] = val
        return doc

    def on_dropdown_change(self, index):
        """Change the displayed widget based on the dropdown selection."""
        self.stacked_widget.setCurrentIndex(index)
        # Update the tab index label (1-based index)
        total_tabs = (
            self.dropdown.count()
        )  # Corrected way to get the total number of items
        self.tab_index_label.setText(f"{index + 1}/{total_tabs}")

    def validate_all_tabs(self) -> bool:
        return all(
            [
                self.global_tab.validate(),
                self.cell_tab.validate(),
                self.molecule_tab.validate(),
                self.condensate_tab.validate(),
                self.fluorophore_tab.validate(),
                self.psf_tab.validate(),
                self.laser_tab.validate(),
                self.channel_tab.validate(),
                self.detector_tab.validate(),
                self.experiment_tab.validate(),
            ]
        )

    def collect_all_config(self) -> dict:
        return {
            **self.general_tab.get_data(),
            "Global_Parameters": self.global_tab.get_data(),
            "Cell_Parameters": self.cell_tab.get_data(),
            "Molecule_Parameters": self.molecule_tab.get_data(),
            "Condensate_Parameters": self.condensate_tab.get_data(),
            "Output_Parameters": self.output_tab.get_data(),
            "fluorophores": self.fluorophore_tab.get_data(),
            "psf": self.psf_tab.get_data(),
            "lasers": self.laser_tab.get_data(),
            "channels": self.channel_tab.get_data(),
            "camera": self.detector_tab.get_data(),
            "experiment": self.experiment_tab.get_data(),
        }

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
            # Validate all required tabs

            if self.validate_all_tabs:
                config = self.collect_all_config()
                toml_doc = self.build_toml_doc(config)

                # Ask user where to save
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Configuration", "", "TOML Files (*.toml);;All Files (*)"
                )

                if file_path:
                    if not file_path.endswith(".toml"):
                        file_path += ".toml"

                    with open(file_path, "w") as f:
                        tomlkit.dump(toml_doc, f)

                    QMessageBox.information(
                        self, "Success", "Configuration has been saved successfully."
                    )
                else:
                    QMessageBox.warning(
                        self, "Save Cancelled", "No file selected. Save was aborted."
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
