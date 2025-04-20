import webbrowser
from pathlib import Path

import napari
import tifffile
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .configuration_window import ConfigEditor
from .logging_window import LogWindow
from .sim_worker import SimulationWorker

LOGO_PATH = str(Path(__file__).parent / "assets" / "drawing.svg")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Welcome to AMS!")

        # Set up the main layout
        layout = QVBoxLayout()

        # Add logo as a placeholder (SVG format)
        self.logo_label = QLabel()  # Label to hold the logo
        self.set_svg_logo(LOGO_PATH)  # Set the SVG logo
        layout.addWidget(self.logo_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Add the maintainer's name under the image
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

    def run_simulation_from_config(self):
        config_path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Select Configuration File",
            "",
            "TOML Files (*.toml);;All Files (*)",
        )

        if config_path_str:
            config_path = Path(config_path_str)

            # Create and show log window
            self.log_window = LogWindow(self)
            self.log_window.show()

            # Set up thread + worker
            self.sim_thread = QThread()
            self.sim_worker = SimulationWorker(config_path, self.log_window)
            self.sim_worker.moveToThread(self.sim_thread)

            # Connect thread/worker signals
            self.sim_thread.started.connect(self.sim_worker.run)
            self.sim_worker.finished.connect(self.sim_thread.quit)
            self.sim_worker.finished.connect(self.sim_worker.deleteLater)
            self.sim_thread.finished.connect(self.sim_thread.deleteLater)

            self.sim_worker.log_message.connect(self.log_window.append_text)
            self.sim_worker.error_occurred.connect(self.log_window.append_text)

            # Auto-close log window and reset on completion
            def cleanup():
                self.log_window.progress.setRange(0, 1)
                self.log_window.progress.setValue(1)
                self.log_window.cancel_button.setText("Close")
                self.log_window.cancel_button.clicked.disconnect()
                self.log_window.cancel_button.clicked.connect(self.log_window.close)

            self.sim_worker.finished.connect(cleanup)

            # Start the simulation
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
                # Load the image (expand here if you want ND2 or Zarr support)
                image = tifffile.imread(file_path)

                # Open Napari viewer and display the image
                viewer = napari.Viewer()
                viewer.add_image(image, name=Path(file_path).stem)
                napari.run()

            except Exception as e:
                print(f"Failed to open image: {e}")

    def set_svg_logo(self, svg_path):
        """Set an SVG logo to the QLabel, maintaining the aspect ratio."""
        # Create a QSvgRenderer to render the SVG
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

    def open_config_editor(self):
        """Open the ConfigEditor window."""
        self.config_editor_window = ConfigEditor()
        self.config_editor_window.show()  # Open the ConfigEditor window as a new window

    def on_link_activated(self, url):
        """Handle the link activation (clicking the hyperlink)."""
        webbrowser.open(url)  # Open the URL in the default web browser
