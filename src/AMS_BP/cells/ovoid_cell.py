from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .base_cell import BaseCell


@dataclass
class OvoidCell(BaseCell):
    """
    Represents an ovoid (ellipsoidal) cell in 3D space, centered around Z=0.
    Attributes:
        origin (np.ndarray): The (x, y) coordinates of the cell's center in XY plane
        radius_x (float): Radius along the X axis
        radius_y (float): Radius along the Y axis
        radius_z (float): Radius along the Z axis
    """

    radius_x: float
    radius_y: float
    radius_z: float

    def validate_specific(self) -> None:
        """Validate ovoid-specific parameters."""
        if self.radius_x <= 0 or self.radius_y <= 0 or self.radius_z <= 0:
            raise ValueError("All radii must be positive")

    def calculate_volume(self) -> float:
        """Calculate the volume of the ovoid."""
        return (4 / 3) * np.pi * self.radius_x * self.radius_y * self.radius_z

    @property
    def center(self) -> np.ndarray:
        """Get the center point of the ovoid."""
        return np.array([self.origin[0], self.origin[1], 0.0])

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the minimum and maximum points that define the ovoid's bounding box.
        Returns:
            Tuple containing (min_point, max_point)
        """
        min_point = np.array(
            [
                self.origin[0] - self.radius_x,
                self.origin[1] - self.radius_y,
                -self.radius_z,  # Z extends downward from 0
            ]
        )
        max_point = np.array(
            [
                self.origin[0] + self.radius_x,
                self.origin[1] + self.radius_y,
                self.radius_z,  # Z extends upward from 0
            ]
        )
        return min_point, max_point

    def contains_point(self, point: np.ndarray) -> bool:
        """
        Check if a point lies within the ovoid.
        Args:
            point: A 3D point to check
        Returns:
            bool: True if the point is inside the ovoid, False otherwise
        """
        point = np.array(point)
        if point.shape != (3,):
            raise ValueError("Point must be a 3D point")

        # Transform point to be relative to center
        relative_point = point - self.center

        # For point (x,y,z) to be inside ellipsoid: (x/a)^2 + (y/b)^2 + (z/c)^2 <= 1
        # where a,b,c are the radii along each axis
        return (
            (relative_point[0] / self.radius_x) ** 2
            + (relative_point[1] / self.radius_y) ** 2
            + (relative_point[2] / self.radius_z) ** 2
        ) <= 1.0
