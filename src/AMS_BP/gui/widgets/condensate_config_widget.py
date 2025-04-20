from pydantic import ValidationError
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...configio.configmodels import CondensateParameters


class CondensateConfigWidget(QWidget):
    # Signal to notify when molecule count changes
    molecule_count_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.molecule_tabs = []
        self.condensate_widgets = []
        self._updating_molecule_count = False  # Flag to prevent recursion
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Instructions label
        instructions = QLabel(
            "Configure parameters for different molecule types and their condensates. "
            "Each tab represents a different molecule type."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Controls for adding/removing molecule types
        controls_layout = QHBoxLayout()

        self.molecule_count = QSpinBox()
        self.molecule_count.setRange(1, 10)
        self.molecule_count.setValue(1)
        self.molecule_count.valueChanged.connect(self._on_molecule_count_changed)
        controls_layout.addWidget(QLabel("Number of Molecule Types:"))
        controls_layout.addWidget(self.molecule_count)

        layout.addLayout(controls_layout)

        # Tab widget for molecule types
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Density difference
        density_layout = QHBoxLayout()
        density_layout.addWidget(QLabel("Density Difference:"))

        self.density_dif_widget = QDoubleSpinBox()
        self.density_dif_widget.setRange(0, 100)
        self.density_dif_widget.setValue(1.0)
        self.density_dif_widget.setDecimals(3)
        density_layout.addWidget(self.density_dif_widget)

        layout.addLayout(density_layout)

        # Validation button
        self.validate_button = QPushButton("Validate Parameters")
        self.validate_button.clicked.connect(self.validate)
        layout.addWidget(self.validate_button)

        # Initialize with one molecule type
        self.update_molecule_count(1)

        self.setLayout(layout)

    def _on_molecule_count_changed(self, count):
        """Handle molecule count change internally and emit signal"""
        if not self._updating_molecule_count:
            self.update_molecule_count(count)
            # Emit signal to notify other widgets
            self.molecule_count_changed.emit(count)

    def set_molecule_count(self, count):
        """Public method to be called by other widgets to update molecule count"""
        if self.molecule_count.value() != count:
            self._updating_molecule_count = True
            self.molecule_count.setValue(count)
            self.update_molecule_count(count)
            self._updating_molecule_count = False

    def update_molecule_count(self, count):
        """Update the number of molecule tabs"""
        current_count = self.tab_widget.count()

        # Add new tabs if needed
        for i in range(current_count, count):
            self.add_molecule_tab(i)

        # Remove excess tabs if needed
        while self.tab_widget.count() > count:
            self.tab_widget.removeTab(count)
            if self.condensate_widgets:
                self.condensate_widgets.pop()

    def add_molecule_tab(self, index):
        """Add a new tab for a molecule type"""
        molecule_widget = QWidget()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        layout = QVBoxLayout(molecule_widget)

        # Controls for condensate count
        condensate_controls = QHBoxLayout()
        condensate_count = QSpinBox()
        condensate_count.setRange(1, 20)
        condensate_count.setValue(1)

        condensate_controls.addWidget(QLabel("Number of Condensates:"))
        condensate_controls.addWidget(condensate_count)
        layout.addLayout(condensate_controls)

        condensate_container = QVBoxLayout()
        layout.addLayout(condensate_container)

        # Add first condensate
        condensate_widgets = []
        self.add_condensate_group(0, condensate_widgets, condensate_container)

        condensate_count.valueChanged.connect(
            lambda count: self.update_condensate_count(
                count, condensate_widgets, condensate_container
            )
        )

        # Store the condensate widgets for this molecule
        self.condensate_widgets.append(condensate_widgets)

        molecule_widget.setLayout(layout)
        scroll_area.setWidget(molecule_widget)

        self.tab_widget.addTab(scroll_area, f"Molecule Type {index + 1}")

    def add_condensate_group(self, index, condensate_widgets, condensate_container):
        """Add a group of widgets for a single condensate"""
        group = QGroupBox(f"Condensate {index + 1}")
        form = QFormLayout()

        # Initial center
        center_layout = QHBoxLayout()
        center_x = QDoubleSpinBox()
        center_y = QDoubleSpinBox()
        center_z = QDoubleSpinBox()

        for spinbox in [center_x, center_y, center_z]:
            spinbox.setRange(-1000, 1000)
            spinbox.setDecimals(2)
            spinbox.setSuffix(" μm")
            center_layout.addWidget(spinbox)

        form.addRow("Initial Center (x, y, z):", self._make_container(center_layout))

        # Initial scale
        scale = QDoubleSpinBox()
        scale.setRange(0.01, 100)
        scale.setValue(1.0)
        scale.setDecimals(2)
        scale.setSuffix(" μm")
        form.addRow("Initial Scale:", scale)

        # Diffusion coefficient
        diffusion = QDoubleSpinBox()
        diffusion.setRange(0, 100)
        diffusion.setValue(1.0)
        diffusion.setDecimals(3)
        diffusion.setSuffix(" μm²/s")
        form.addRow("Diffusion Coefficient:", diffusion)

        # Hurst exponent
        hurst = QDoubleSpinBox()
        hurst.setRange(0, 1)
        hurst.setValue(0.5)
        hurst.setDecimals(2)
        form.addRow("Hurst Exponent:", hurst)

        group.setLayout(form)

        condensate_container.addWidget(group)

        # Store the widgets
        condensate_data = {
            "center": [center_x, center_y, center_z],
            "scale": scale,
            "diffusion": diffusion,
            "hurst": hurst,
            "group": group,
        }
        condensate_widgets.append(condensate_data)

    def update_condensate_count(self, count, condensate_widgets, condensate_container):
        """Update the number of condensate groups"""
        current_count = len(condensate_widgets)

        # Add new condensates if needed
        for i in range(current_count, count):
            self.add_condensate_group(i, condensate_widgets, condensate_container)

        # Remove excess condensates if needed
        while len(condensate_widgets) > count:
            removed = condensate_widgets.pop()
            removed["group"].deleteLater()

    def _make_container(self, layout):
        """Helper to create a container widget for a layout"""
        container = QWidget()
        container.setLayout(layout)
        return container

    def get_data(self) -> dict:
        initial_centers: list[list[list[float]]] = []
        initial_scale: list[list[float]] = []
        diffusion_coefficient: list[list[float]] = []
        hurst_exponent: list[list[float]] = []

        # For each molecule type (i.e., each tab)
        for molecule_widgets in self.condensate_widgets:
            molecule_centers: list[list[float]] = []
            molecule_scales: list[float] = []
            molecule_diffusions: list[float] = []
            molecule_hursts: list[float] = []

            # For each condensate in that molecule type
            for condensate in molecule_widgets:
                center = [spin.value() for spin in condensate["center"]]
                molecule_centers.append(center)
                molecule_scales.append(condensate["scale"].value())
                molecule_diffusions.append(condensate["diffusion"].value())
                molecule_hursts.append(condensate["hurst"].value())

            initial_centers.append(molecule_centers)
            initial_scale.append(molecule_scales)
            diffusion_coefficient.append(molecule_diffusions)
            hurst_exponent.append(molecule_hursts)

        # Replicate density difference per molecule type
        density_value = self.density_dif_widget.value()
        num_molecule_types = len(self.condensate_widgets)
        density_dif = [density_value for _ in range(num_molecule_types)]

        return {
            "initial_centers": initial_centers,
            "initial_scale": initial_scale,
            "diffusion_coefficient": diffusion_coefficient,
            "hurst_exponent": hurst_exponent,
            "density_dif": density_dif,
        }

    def validate(self) -> bool:
        """Validate the form data against the CondensateParameters model"""
        try:
            data = self.get_data()
            validated = CondensateParameters(**data)
            QMessageBox.information(
                self, "Validation Successful", "Condensate parameters are valid."
            )
            return True
        except ValidationError as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            return False
        except ValueError as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            return False
