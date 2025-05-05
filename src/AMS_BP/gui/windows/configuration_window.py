from pathlib import Path

import tomli
import tomlkit
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..widgets.camera_config_widget import CameraConfigWidget
from ..widgets.cell_config_widget import CellConfigWidget
from ..widgets.channel_config_widget import ChannelConfigWidget
from ..widgets.condensate_config_widget import CondensateConfigWidget
from ..widgets.experiment_config_widget import ExperimentConfigWidget
from ..widgets.flurophore_config_widget import FluorophoreConfigWidget
from ..widgets.general_config_widget import GeneralConfigWidget
from ..widgets.global_config_widget import GlobalConfigWidget
from ..widgets.laser_config_widget import LaserConfigWidget
from ..widgets.molecule_config_widget import MoleculeConfigWidget
from ..widgets.output_config_widget import OutputConfigWidget
from ..widgets.psf_config_widget import PSFConfigWidget
from .help_window import HelpWindow


class ConfigEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulation Configuration Editor")

        # === Main horizontal layout: [Side Navigation | Content Area] ===
        main_layout = QHBoxLayout(self)

        # === Sidebar: Section Navigation ===
        self.nav_list = QListWidget()
        self.sections = [
            "General",
            "Global Parameters",
            "Cell Parameters",
            "Molecule Parameters",
            "Condensate Parameters",
            "Define fluorophores",
            "Camera Parameters",
            "PSF Parameters",
            "Laser Parameters",
            "Channels Parameters",
            "Saving Instructions",
            "Experiment Builder",
        ]
        self.nav_list.addItems(self.sections)
        self.nav_list.setFixedWidth(220)
        self.nav_list.setSpacing(4)
        self.nav_list.setCurrentRow(0)
        self.nav_list.currentRowChanged.connect(self.on_tab_selected)
        main_layout.addWidget(self.nav_list)

        # === Right panel layout ===
        right_panel = QVBoxLayout()

        # Step/breadcrumb label
        self.step_label = QLabel()
        self.step_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        right_panel.addWidget(self.step_label)

        # === Stack of config widgets ===
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        right_panel.addWidget(self.stacked_widget)

        # === Buttons at the bottom ===
        self.save_button = QPushButton("Ready to save configuration?")
        self.save_button.clicked.connect(self.save_config)
        right_panel.addWidget(self.save_button)

        self.preview_button = QPushButton("Preview Configuration TOML")
        self.preview_button.clicked.connect(self.preview_config)
        right_panel.addWidget(self.preview_button)

        self.help_button = QPushButton("Get Help on this section")
        self.help_button.clicked.connect(self.show_help)
        right_panel.addWidget(self.help_button)

        main_layout.addLayout(right_panel)

        # === Create tab content widgets ===
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

        # === Widget interconnections ===
        self.psf_tab.confocal_mode_changed.connect(self.laser_tab.set_confocal_mode)
        self.psf_tab.confocal_mode_changed.connect(
            self.experiment_tab.set_scanning_mode
        )

        self.molecule_tab.molecule_count_changed.connect(
            self.fluorophore_tab.set_mfluorophore_count
        )
        self.molecule_tab.molecule_count_changed.connect(
            self.condensate_tab.set_molecule_count
        )

        self.fluorophore_tab.mfluorophore_count_changed.connect(
            self.molecule_tab.set_molecule_count
        )
        self.fluorophore_tab.mfluorophore_count_changed.connect(
            self.condensate_tab.set_molecule_count
        )

        self.condensate_tab.molecule_count_changed.connect(
            self.molecule_tab.set_molecule_count
        )
        self.condensate_tab.molecule_count_changed.connect(
            self.fluorophore_tab.set_mfluorophore_count
        )

        self.laser_tab.laser_names_updated.connect(
            self.experiment_tab.set_active_lasers
        )

        # === Add tab widgets to the stack ===
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

        # Final layout and window size
        self.setLayout(main_layout)
        self.setMinimumSize(1100, 750)
        self.resize(1250, 850)

        # Initial tab display
        self.on_tab_selected(0)

    def set_data(self, config: dict):
        if "Cell_Parameters" in config:
            self.cell_tab.set_data(config["Cell_Parameters"])

        if "Global_Parameters" in config:
            self.global_tab.set_data(config["Global_Parameters"])

        if "Molecule_Parameters" in config:
            self.molecule_tab.set_data(config["Molecule_Parameters"])

        if "fluorophores" in config:
            self.fluorophore_tab.set_data(config["fluorophores"])

        if "Condensate_Parameters" in config:
            self.condensate_tab.set_data(config["Condensate_Parameters"])

        if "Output_Parameters" in config:
            self.output_tab.set_data(config["Output_Parameters"])

        if "lasers" in config:
            self.laser_tab.set_data(config["lasers"])

        if "experiment" in config:
            self.experiment_tab.set_data(config["experiment"])

        if "channels" in config:
            self.channel_tab.set_data(config["channels"])

        if "camera" in config:
            self.detector_tab.set_data(config["camera"])

        if "psf" in config:
            self.psf_tab.set_data(config["psf"])

    def preview_config(self):
        """Preview the full TOML config in a dialog before saving."""
        try:
            if not self.validate_all_tabs():
                QMessageBox.warning(
                    self, "Validation Error", "Fix errors before preview."
                )
                return

            config = self.collect_all_config()

            toml_doc = tomlkit.document()
            for key, value in config.items():
                toml_doc[key] = value
            toml_str = tomlkit.dumps(toml_doc)

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

    def on_tab_selected(self, index: int):
        """Change the displayed widget and update breadcrumb/step label."""
        self.stacked_widget.setCurrentIndex(index)
        total = self.nav_list.count()
        current = self.nav_list.item(index).text()
        self.step_label.setText(f"Step {index + 1}/{total} — {current}")

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

    def load_config_from_toml(self, path: Path):
        with open(path, "rb") as f:
            config = tomli.load(f)
        self.set_data(config)
