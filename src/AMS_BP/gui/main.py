import webbrowser
from pathlib import Path
from zipfile import ZipFile

import napari
import tifffile
from PyQt6.QtCore import QSettings, Qt, QThread
from PyQt6.QtGui import QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..tools.logging.logutil import LoggerManager
from ..tools.logging.setup_run_directory import setup_run_directory
from .sim_worker import SimulationWorker
from .widgets.utility_widgets.toggleswitch_widget import ToggleSwitch
from .windows.logging_window import LogWindow
from .windows.template_window_selection import TemplateSelectionWindow

LOGO_PATH = str(Path(__file__).parent / "assets" / "drawing.svg")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Welcome to AMS!")

        # Set up the main layout
        layout = QVBoxLayout()

        self.logo_label = QLabel()  # Label to hold the logo
        self.set_svg_logo(LOGO_PATH)  # Set the SVG logo
        layout.addWidget(self.logo_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.maintainer_label = QLabel(
            "Maintainer: Baljyot Parmar \n baljyotparmar@hotmail.com"
        )
        self.maintainer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.maintainer_label)
        self.lab_label = QLabel(
            "Brought to you by: " + '<a href="https://weberlab.ca">The WeberLab</a>'
        )
        self.lab_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lab_label.setOpenExternalLinks(True)  # Enable external links
        self.lab_label.linkActivated.connect(self.on_link_activated)
        layout.addWidget(self.lab_label)

        # Button to open the Configuration Creation window
        self.config_button = QPushButton("Create Configuration File")
        self.config_button.clicked.connect(self.open_config_editor)
        layout.addWidget(self.config_button)

        # Create a central widget and set the layout
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Button to run simulation with a TOML config file
        self.run_sim_button = QPushButton("Run Simulation from Config")
        self.run_sim_button.clicked.connect(self.run_simulation_from_config)
        layout.addWidget(self.run_sim_button)

        # Button to open Napari viewer
        self.view_button = QPushButton("Visualize Microscopy Data (Napari)")
        self.view_button.clicked.connect(self.open_napari_viewer)
        layout.addWidget(self.view_button)

        self.package_logs_button = QPushButton("Package Logs for Sharing")
        self.package_logs_button.clicked.connect(self.package_logs)
        layout.addWidget(self.package_logs_button)

        # Load theme preference
        self.settings = QSettings("AMS", "AMSConfig")
        theme_pref = self.settings.value("theme", "light")

        # Add toggle switch with label
        self.theme_toggle = ToggleSwitch(checked=(theme_pref == "dark"))
        self.theme_toggle.toggled.connect(self.toggle_theme)
        self.theme_label = QLabel("Dark Mode" if theme_pref == "dark" else "Light Mode")
        layout.addWidget(self.theme_label)
        layout.addWidget(self.theme_toggle, alignment=Qt.AlignmentFlag.AlignCenter)

        # Apply initial theme
        self.apply_theme(theme_pref)

    def package_logs(self):
        log_dir = Path.home() / "AMS_runs"

        folder_paths = QFileDialog.getExistingDirectory(
            self,
            "Select Folder Containing Run Logs",
            str(log_dir),
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )

        if not folder_paths:
            return

        # For now, let's treat this as a single run_* folder being selected.

        run_dir = Path(folder_paths)
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            QMessageBox.warning(
                self, "Invalid Selection", "Please select a valid run_* folder."
            )
            return

        zip_path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save Zipped Folder As",
            str(log_dir / f"{run_dir.name}.zip"),
            "Zip Archive (*.zip)",
        )

        if not zip_path_str:
            return

        zip_path = Path(zip_path_str)
        if not zip_path.suffix == ".zip":
            zip_path = zip_path.with_suffix(".zip")

        try:
            with ZipFile(zip_path, "w") as archive:
                for path in run_dir.rglob("*"):
                    archive.write(path, arcname=path.relative_to(run_dir.parent))

            QMessageBox.information(
                self,
                "Logs Packaged",
                f"Folder '{run_dir.name}' successfully packaged to:\n{zip_path}",
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to package folder:\n{e}")

    def run_simulation_from_config(self):
        config_path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Select Configuration File",
            "",
            "TOML Files (*.toml);;All Files (*)",
        )

        if config_path_str:
            config_path = Path(config_path_str)

            # Structured run dir setup
            run_info = setup_run_directory(config_path)
            log_path = run_info["log_file"]

            # Create logger
            self.logger_manager = LoggerManager(log_path)
            self.logger_manager.start()

            # Log window
            self.log_window = LogWindow(self)
            self.log_window.show()

            # Connect logger to GUI window
            # Get emitter from logger
            emitter = self.logger_manager.get_emitter()

            # Hook up emitter to the GUI
            emitter.message.connect(self.log_window.append_text)

            self.sim_worker = SimulationWorker(
                config_path,
                emitter,
                cancel_callback=lambda: self.log_window.cancel_requested,
            )

            self.sim_thread = QThread()
            self.sim_worker.moveToThread(self.sim_thread)

            self.sim_thread.started.connect(self.sim_worker.run)
            self.sim_worker.finished.connect(self.sim_thread.quit)
            self.sim_worker.finished.connect(self.sim_worker.deleteLater)
            self.sim_thread.finished.connect(self.sim_thread.deleteLater)

            self.sim_worker.finished.connect(lambda: print("Simulation finished."))
            self.sim_worker.finished.connect(self.logger_manager.stop)

            def handle_finished():
                if self.sim_worker.failed:
                    self.log_window.mark_failure()
                else:
                    self.log_window.mark_success()

                self.log_window.cancel_button.setText("Close")
                self.log_window.cancel_button.clicked.disconnect()
                self.log_window.cancel_button.clicked.connect(self.log_window.close)

            self.sim_worker.finished.connect(handle_finished)
            self.sim_worker.error_occurred.connect(emitter.message.emit)

            self.sim_thread.start()

    def open_napari_viewer(self):
        """Open a file dialog to select a microscopy image and visualize it with Napari."""
        # Allow user to select an image file
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Microscopy Image",
            "",
            "Image Files (*.tif *.tiff *.nd2 *.png *.jpg *.zarr);;All Files (*)",
        )

        if file_path:
            try:
                image = tifffile.imread(file_path)

                # Open Napari viewer and display the image
                viewer = napari.Viewer()
                viewer.add_image(image, name=Path(file_path).stem)
                napari.run()

            except Exception as e:
                print(f"Failed to open image: {e}")

    def set_svg_logo(self, svg_path):
        """Set an SVG logo to the QLabel, maintaining the aspect ratio."""
        renderer = QSvgRenderer(svg_path)
        if renderer.isValid():
            # Get the size of the SVG image
            image_size = renderer.defaultSize()

            # Create a QPixmap to hold the rendered SVG with the same size as the SVG image
            pixmap = QPixmap(image_size * 2)
            pixmap.fill(Qt.GlobalColor.transparent)  # Fill the pixmap with transparency

            # Use QPainter to paint the SVG onto the pixmap
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()

            # Scale the pixmap to fit the desired size while maintaining the aspect ratio
            scaled_pixmap = pixmap.scaled(
                200,
                200,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            # Set the scaled pixmap as the QLabel content
            self.logo_label.setPixmap(scaled_pixmap)
        else:
            print("Failed to load SVG file.")

    def toggle_theme(self, is_dark: bool):
        theme = "dark" if is_dark else "light"
        self.settings.setValue("theme", theme)
        self.apply_theme(theme)
        self.theme_label.setText("Dark Mode" if is_dark else "Light Mode")

    def apply_theme(self, theme: str):
        theme_file = Path(__file__).parent / "themes" / f"{theme}_theme.qss"
        try:
            with open(theme_file, "r") as f:
                stylesheet = f.read()
                QApplication.instance().setStyleSheet(stylesheet)
        except Exception as e:
            QMessageBox.warning(self, "Theme Error", f"Failed to load theme:\n{e}")

    def open_config_editor(self):
        """Launch template selection first, then open ConfigEditor."""
        self.template_window = TemplateSelectionWindow()
        self.template_window.show()

    def on_link_activated(self, url):
        """Handle the link activation (clicking the hyperlink)."""
        webbrowser.open(url)  # Open the URL in the default web browser
