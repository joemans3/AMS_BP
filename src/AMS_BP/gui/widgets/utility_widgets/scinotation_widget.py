from typing import Optional

from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QLineEdit,
)


def scientific_input_field(
    minimum: float = -1e100,
    maximum: float = 1e100,
    default: Optional[float] = None,
    decimals: int = 10,
) -> QLineEdit:
    line = QLineEdit()
    validator = QDoubleValidator(minimum, maximum, decimals)
    validator.setNotation(QDoubleValidator.Notation.ScientificNotation)
    line.setValidator(validator)
    if default is not None:
        line.setText(f"{default:.3e}")
    return line
