from pathlib import Path

from pydantic import ValidationError
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...core.cells import create_cell
from ...core.configio.configmodels import CellParameters


class CellConfigWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        form = QFormLayout()

        self.cell_type = QComboBox()
        self.cell_type.addItems(
            ["SphericalCell", "RodCell", "RectangularCell", "OvoidCell"]
        )
        self.cell_type.currentIndexChanged.connect(self.update_params)
        form.addRow("Cell Type:", self.cell_type)

        self.param_stack = QStackedWidget()
        self.param_widgets = {
            "SphericalCell": self.make_spherical_widget(),
            "RodCell": self.make_rod_widget(),
            "RectangularCell": self.make_rect_widget(),
            "OvoidCell": self.make_ovoid_widget(),
        }

        for widget in self.param_widgets.values():
            self.param_stack.addWidget(widget)

        layout.addLayout(form)
        layout.addWidget(self.param_stack)
        self.setLayout(layout)

        self.validate_button = QPushButton("Validate Parameters")
        self.validate_button.clicked.connect(self.validate)
        layout.addWidget(self.validate_button)

    def validate(self) -> bool:
        try:
            data = self.get_data()

            # Validate the Pydantic model first
            cell_params = CellParameters(**data)

            # Try creating the cell with backend logic
            create_cell(cell_params.cell_type, cell_params.params)

            # Success
            QMessageBox.information(
                self, "Validation Successful", "Cell parameters are valid."
            )
            return True

        except ValidationError as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            return False
        except Exception as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            return False

    def update_params(self, index):
        self.param_stack.setCurrentIndex(index)

    def make_spherical_widget(self):
        widget = QWidget()
        form = QFormLayout()
        self.center_s = [QDoubleSpinBox() for _ in range(3)]
        for c in self.center_s:
            c.setRange(0, 1e5)
        self.radius_s = QDoubleSpinBox()
        self.radius_s.setRange(0, 1e4)
        form.addRow("Center (x,y,z):", self._hbox(self.center_s))
        form.addRow("Radius:", self.radius_s)
        widget.setLayout(form)
        return widget

    def make_rod_widget(self):
        widget = QWidget()
        form = QFormLayout()
        self.center_r = [QDoubleSpinBox() for _ in range(3)]
        self.direction_r = [QDoubleSpinBox() for _ in range(3)]
        self.height_r = QDoubleSpinBox()
        self.radius_r = QDoubleSpinBox()
        for w in self.center_r + self.direction_r:
            w.setRange(0, 1e5)
        form.addRow("Center:", self._hbox(self.center_r))
        form.addRow("Direction:", self._hbox(self.direction_r))
        form.addRow("Height:", self.height_r)
        form.addRow("Radius:", self.radius_r)
        widget.setLayout(form)
        return widget

    def make_rect_widget(self):
        widget = QWidget()
        form = QFormLayout()
        self.bounds = [QDoubleSpinBox() for _ in range(6)]
        for b in self.bounds:
            b.setRange(0, 1e5)
        form.addRow("Bounds [xmin,xmax,ymin,ymax,zmin,zmax]:", self._hbox(self.bounds))
        widget.setLayout(form)
        return widget

    def make_ovoid_widget(self):
        widget = QWidget()
        form = QFormLayout()
        self.center_o = [QDoubleSpinBox() for _ in range(3)]
        self.xradius = QDoubleSpinBox()
        self.yradius = QDoubleSpinBox()
        self.zradius = QDoubleSpinBox()
        for c in self.center_o:
            c.setRange(0, 1e5)
        form.addRow("Center:", self._hbox(self.center_o))
        form.addRow("X Radius:", self.xradius)
        form.addRow("Y Radius:", self.yradius)
        form.addRow("Z Radius:", self.zradius)
        widget.setLayout(form)
        return widget

    def _hbox(self, widgets):
        box = QHBoxLayout()
        for w in widgets:
            box.addWidget(w)
        container = QWidget()
        container.setLayout(box)
        return container

    def get_data(self):
        ctype = self.cell_type.currentText()
        if ctype == "SphericalCell":
            return {
                "cell_type": ctype,
                "params": {
                    "center": [s.value() for s in self.center_s],
                    "radius": self.radius_s.value(),
                },
            }
        elif ctype == "RodCell":
            return {
                "cell_type": ctype,
                "params": {
                    "center": [s.value() for s in self.center_r],
                    "direction": [s.value() for s in self.direction_r],
                    "height": self.height_r.value(),
                    "radius": self.radius_r.value(),
                },
            }
        elif ctype == "RectangularCell":
            return {
                "cell_type": ctype,
                "params": {"bounds": [b.value() for b in self.bounds]},
            }
        elif ctype == "OvoidCell":
            return {
                "cell_type": ctype,
                "params": {
                    "center": [s.value() for s in self.center_o],
                    "xradius": self.xradius.value(),
                    "yradius": self.yradius.value(),
                    "zradius": self.zradius.value(),
                },
            }
        else:
            return {"cell_type": ctype, "params": {}}

    def set_data(self, data: dict):
        try:
            cell_type = data.get("cell_type", "RodCell")
            self.cell_type.setCurrentText(cell_type)

            params = data.get("params", {})

            if cell_type == "SphericalCell":
                center = params.get("center", [0, 0, 0])
                radius = params.get("radius", 1.0)
                for i, val in enumerate(center):
                    self.center_s[i].setValue(val)
                self.radius_s.setValue(radius)

            elif cell_type == "RodCell":
                center = params.get("center", [0, 0, 0])
                direction = params.get("direction", [1, 0, 0])
                height = params.get("height", 1.0)
                radius = params.get("radius", 0.5)
                for i, val in enumerate(center):
                    self.center_r[i].setValue(val)
                for i, val in enumerate(direction):
                    self.direction_r[i].setValue(val)
                self.height_r.setValue(height)
                self.radius_r.setValue(radius)

            elif cell_type == "RectangularCell":
                bounds = params.get("bounds", [0, 10, 0, 10, 0, 10])
                for i, val in enumerate(bounds):
                    self.bounds[i].setValue(val)

            elif cell_type == "OvoidCell":
                center = params.get("center", [0, 0, 0])
                xr = params.get("xradius", 1.0)
                yr = params.get("yradius", 1.0)
                zr = params.get("zradius", 1.0)
                for i, val in enumerate(center):
                    self.center_o[i].setValue(val)
                self.xradius.setValue(xr)
                self.yradius.setValue(yr)
                self.zradius.setValue(zr)

        except Exception as e:
            print(f"[CellConfigWidget] Failed to load data: {e}")

    def get_help_path(self) -> Path:
        return Path(__file__).parent.parent / "help_docs" / "cell_help.md"
