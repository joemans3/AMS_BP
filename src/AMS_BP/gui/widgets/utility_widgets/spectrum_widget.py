# spectrum_editor.py
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class SpectrumEditorDialog(QDialog):
    def __init__(
        self,
        parent=None,
        wavelengths=None,
        intensities=None,
        intensity_name="Intensity",
    ):
        super().__init__(parent)
        self.setWindowTitle("Edit Spectrum")
        self.resize(600, 400)
        self.wavelengths = wavelengths or []
        self.intensities = intensities or []
        self.intensity_name = intensity_name

        self.setup_ui()
        self.populate_table()
        self.update_plot()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Wavelength (nm)", self.intensity_name])
        self.table.cellChanged.connect(self.update_plot)
        layout.addWidget(self.table)

        btns = QHBoxLayout()
        add_btn = QPushButton("Add Row")
        remove_btn = QPushButton("Remove Selected")
        btns.addWidget(add_btn)
        btns.addWidget(remove_btn)
        layout.addLayout(btns)

        add_btn.clicked.connect(self.add_row)
        remove_btn.clicked.connect(self.remove_selected_row)

        # Spectrum preview
        self.figure = Figure(figsize=(4, 2))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # OK / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def populate_table(self):
        for w, i in zip(self.wavelengths, self.intensities):
            self.add_row(w, i)

    def add_row(self, wavelength="", intensity=""):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(wavelength)))
        self.table.setItem(row, 1, QTableWidgetItem(str(intensity)))

    def remove_selected_row(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)
            self.update_plot()

    def get_spectrum(self):
        wavelengths = []
        intensities = []
        for row in range(self.table.rowCount()):
            try:
                w = float(self.table.item(row, 0).text())
                i = float(self.table.item(row, 1).text())
                wavelengths.append(w)
                intensities.append(i)
            except (ValueError, AttributeError):
                raise ValueError(f"Invalid entry at row {row + 1}")
        return wavelengths, intensities

    def update_plot(self):
        try:
            wavelengths, intensities = self.get_spectrum()
        except ValueError:
            return  # Avoid crashing on invalid input

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(wavelengths, intensities, marker="o", linestyle="-", color="blue")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel(self.intensity_name)
        ax.set_title("Spectrum Preview")
        ax.grid(True)
        self.canvas.draw()

    def accept(self):
        try:
            self.wavelengths, self.intensities = self.get_spectrum()
            super().accept()
        except ValueError as e:
            QMessageBox.critical(self, "Invalid Data", str(e))
