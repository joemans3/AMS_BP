from dataclasses import dataclass

import numpy as np
import pyvista as pv

from .base_cell import BaseCell


@dataclass
class OvoidCell(BaseCell):
    """
    Represents an ovoid (ellipsoidal) cell in 3D space, centered around Z=0.
    Attributes:
        center (np.ndarray): The (x, y, z) coordinates of the cell's center in XYZ
        direction (np.ndarray): direction vector of the orientation of the ovoid
        xradius (float): Radius along the X axis
        yradius (float): Radius along the Y axis
        zradius (float): Radius along the Z axis
    """

    center: np.ndarray
    direction: np.ndarray
    xradius: float
    yradius: float
    zradius: float


def make_OvoidCell(
    center: np.ndarray,
    direction: np.ndarray,
    xradius: float,
    yradius: float,
    zradius: float,
) -> OvoidCell:
    return OvoidCell(
        mesh=pv.ParametricEllipsoid(
            xradius=xradius,
            yradius=yradius,
            zradius=zradius,
            center=center,
            direction=direction,
        ),
        center=center,
        direction=direction,
        xradius=xradius,
        yradius=yradius,
        zradius=zradius,
    )
