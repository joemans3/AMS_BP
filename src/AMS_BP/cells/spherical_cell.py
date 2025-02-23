from dataclasses import dataclass
from typing import Tuple

import pyvista as pv

from .base_cell import BaseCell


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
