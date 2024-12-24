from dataclasses import dataclass
import numpy as np
from .base_cell import BaseCell
from typing import Tuple


@dataclass
class RodCell(BaseCell):
    """
    Represents a rod-like cell in 3D space.

    Attributes:
        origin (np.ndarray): The (x, y) coordinates of the cell's origin in XY plane
        length (float): Total length of the rod (including end caps)
        radius (float): Radius of both the cylindrical body and hemispheres
    """

    length: float
    radius: float

    def _validate_specific(self) -> None:
        """Validate rod-specific parameters."""
        if self.length <= 0:
            raise ValueError("Length must be positive")
        if self.radius <= 0:
            raise ValueError("Radius must be positive")

    def _calculate_volume(self) -> float:
        """Calculate the volume of the rod."""
        cylinder_volume = np.pi * self.radius**2 * (self.length - 2 * self.radius)
        sphere_volume = (4 / 3) * np.pi * self.radius**3
        return cylinder_volume + sphere_volume

    @property
    def center(self) -> np.ndarray:
        """Get the center point of the rod."""
        return np.array([self.origin[0] + self.length / 2, self.origin[1], 0.0])

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the minimum and maximum points that define the rod's bounding box.

        Returns:
            Tuple containing (min_point, max_point)
        """
        min_point = np.array(
            [
                self.origin[0],
                self.origin[1] - self.radius,
                -self.radius,  # Z extends downward from 0
            ]
        )

        max_point = np.array(
            [
                self.origin[0] + self.length,
                self.origin[1] + self.radius,
                self.radius,  # Z extends upward from 0
            ]
        )

        return min_point, max_point

    def contains_point(self, point: np.ndarray) -> bool:
        """
        Check if a point lies within the rod.

        Args:
            point: A 3D point to check

        Returns:
            bool: True if the point is inside the rod, False otherwise
        """
        point = np.array(point)
        if point.shape != (3,):
            raise ValueError("Point must be a 3D point")

        # Project point onto XY plane for cylinder check
        x, y, z = point - np.array([self.origin[0], self.origin[1], 0])

        # Check if point is within the length bounds
        if x < 0 or x > self.length:
            return False

        # If point is within the cylindrical section
        if self.radius <= x <= (self.length - self.radius):
            # Check distance from central axis
            return np.sqrt(y**2 + z**2) <= self.radius

        # Check left hemisphere (if x < radius)
        elif x < self.radius:
            center = np.array([self.radius, 0, 0])
            return np.linalg.norm(np.array([x, y, z]) - center) <= self.radius

        # Check right hemisphere (if x > length - radius)
        else:
            center = np.array([self.length - self.radius, 0, 0])
            return np.linalg.norm(np.array([x, y, z]) - center) <= self.radius
