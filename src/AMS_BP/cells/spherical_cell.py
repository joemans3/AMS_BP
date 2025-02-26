from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pyvista as pv

from .base_cell import BaseCell
from .typedefs import Vector3D


@dataclass
class SphericalCell(BaseCell):
    """
    Represents a spherical cell in 3D space, centered around Z=0.

    Attributes:
        center (Tuple[float,float,float]): center coordinate of the sphere
        radius (float): Radius of the sphere
    """

    center: Tuple[float, float, float]
    radius: float


def make_SphericalCell(
    center: Tuple[float, float, float], radius: float
) -> SphericalCell:
    return SphericalCell(
        mesh=pv.Sphere(radius=radius, center=center), center=center, radius=radius
    )


@dataclass
class SphericalCellParams:
    center: Vector3D
    radius: float

    @classmethod
    def validate_center(cls, value):
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 3:
            raise ValueError("center must be a 3D vector")

    @classmethod
    def validate_radius(cls, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("radius must be a positive number")
