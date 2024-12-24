from dataclasses import dataclass
import numpy as np
from .base_cell import BaseCell
from typing import Tuple


@dataclass
class SphericalCell(BaseCell):
    """
    Represents a spherical cell in 3D space, centered around Z=0.

    Attributes:
        origin (np.ndarray): The (x, y) coordinates of the cell's center in XY plane
        radius (float): Radius of the sphere
    """

    radius: float

    def _validate_specific(self) -> None:
        """Validate sphere-specific parameters."""
        if self.radius <= 0:
            raise ValueError("Radius must be positive")

    def _calculate_volume(self) -> float:
        """Calculate the volume of the sphere."""
        return (4 / 3) * np.pi * self.radius**3

    @property
    def center(self) -> np.ndarray:
        """Get the center point of the sphere."""
        return np.array([self.origin[0], self.origin[1], 0.0])

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the minimum and maximum points that define the sphere's bounding box.

        Returns:
            Tuple containing (min_point, max_point)
        """
        min_point = np.array(
            [
                self.origin[0] - self.radius,
                self.origin[1] - self.radius,
                -self.radius,  # Z extends downward from 0
            ]
        )

        max_point = np.array(
            [
                self.origin[0] + self.radius,
                self.origin[1] + self.radius,
                self.radius,  # Z extends upward from 0
            ]
        )

        return min_point, max_point

    def contains_point(self, point: np.ndarray) -> bool:
        """
        Check if a point lies within the sphere.

        Args:
            point: A 3D point to check

        Returns:
            bool: True if the point is inside the sphere, False otherwise
        """
        point = np.array(point)
        if point.shape != (3,):
            raise ValueError("Point must be a 3D point")

        # Calculate distance from center to point
        center_3d = self.center
        return np.linalg.norm(point - center_3d) <= self.radius
