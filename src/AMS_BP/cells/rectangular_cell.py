from dataclasses import dataclass

import numpy as np
import pyvista as pv

from .base_cell import BaseCell


@dataclass
class RectangularCell(BaseCell):
    """
    Represents a rectangular cell in 3D space.

    Attributes:
        bounds (np.ndarray):
            [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
    """

    bounds: np.ndarray


def make_RectangularCell(bounds: np.ndarray) -> RectangularCell:
    """
    Parameters:
    -----------
    bounds (np.ndarray):
        [[xmin,xmax],[ymin,ymax],[zmin,zmax]]

    Returns:
    --------
    RectangularCell object
    """
    # valudate bounds
    assert bounds.shape == (3, 2)
    for j in bounds:
        assert j[1] > j[0]
    pv_bounds = bounds.flatten()
    rec = pv.Box(bounds=pv_bounds)
    return RectangularCell(mesh=rec, bounds=bounds)


@dataclass
class RectangularCellParams:
    bounds: np.ndarray

    @classmethod
    def validate_bounds(cls, value):
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise ValueError("bounds must be an array-like object")

        # Convert to numpy array if needed
        if not isinstance(value, np.ndarray):
            value = np.array(value)

        # Check shape
        if value.shape != (2, 3):
            raise ValueError("bounds must be a 2x3 array (min and max points)")

        # Check min < max
        for i in range(3):
            if value[0, i] >= value[1, i]:
                raise ValueError(
                    f"Min bound must be less than max bound for dimension {i}"
                )
