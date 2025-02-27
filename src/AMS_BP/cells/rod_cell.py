from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pyvista as pv
from typing_extensions import List

from .base_cell import BaseCell
from .typedefs import Vector3D


@dataclass
class RodCell(BaseCell):
    """
    Represents a rod-like cell in 3D space.

    Attributes:
        center (np.ndarray): The (x, y, z) coordinates of the cell's center in XYZ plane
        direction (np.ndarray): direction vector of the orientation of the RodCell
        height (float): length of the rod, NOT including end caps
        radius (float): Radius of both the cylindrical body and hemispheres

        +

        pyvista mesh for the BaseCell
    """

    center: np.ndarray | List[float] | Tuple
    direction: np.ndarray | List[float] | Tuple
    height: float
    radius: float


def make_RodCell(
    center: np.ndarray | List[float] | Tuple,
    direction: np.ndarray | List[float] | Tuple,
    height: float,
    radius: float,
) -> RodCell:
    """
    Create a capsule (cylinder with spherical caps) shape.

    Args:
        center: Center point of the capsule
        direction: Direction vector of the capsule axis
        radius: Radius of both cylinder and spherical caps
        height: Height of the cylindrical portion (excluding caps)

    Returns:
        PVShape3D: Capsule shape instance
    """
    capsule = pv.Capsule(
        center=center, direction=direction, radius=radius, cylinder_length=height
    )

    return RodCell(
        mesh=capsule, center=center, direction=direction, height=height, radius=radius
    )


@dataclass
class RodCellParams:
    center: Vector3D
    direction: Vector3D
    height: float
    radius: float

    @classmethod
    def validate_center(cls, value):
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 3:
            raise ValueError("center must be a 3D vector")

    @classmethod
    def validate_direction(cls, value):
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 3:
            raise ValueError("direction must be a 3D vector")

    @classmethod
    def validate_height(cls, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("height must be a positive number")

    @classmethod
    def validate_radius(cls, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("radius must be a positive number")
