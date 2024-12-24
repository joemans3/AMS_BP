from dataclasses import dataclass
import numpy as np
from .base_cell import BaseCell
from typing import Tuple


@dataclass
class RectangularCell(BaseCell):
    """
    Represents a rectangular cell in 3D space.

    Attributes:
        origin (np.ndarray): The (x, y) coordinates of the cell's origin in XY plane
        dimensions (np.ndarray): The (length, width, height) of the cell
    """

    dimensions: np.ndarray

    def _validate_specific(self) -> None:
        """Validate rectangle-specific parameters."""
        self.dimensions = np.array(self.dimensions, dtype=float)
        if self.dimensions.shape != (3,):
            raise ValueError("Dimensions must be a 3D vector (length, width, height)")
        if np.any(self.dimensions <= 0):
            raise ValueError("All dimensions must be positive")

    def _calculate_volume(self) -> float:
        """Calculate the volume of the rectangular cell."""
        return float(np.prod(self.dimensions))

    @property
    def center(self) -> np.ndarray:
        """Get the center point of the rectangular cell."""
        return np.array(
            [
                self.origin[0] + self.dimensions[0] / 2,
                self.origin[1] + self.dimensions[1] / 2,
                0.0,
            ]
        )

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the minimum and maximum points that define the cell's bounds.

        Returns:
            Tuple containing (min_point, max_point)
        """
        min_point = np.array(
            [
                self.origin[0],
                self.origin[1],
                -self.dimensions[2] / 2,  # Z extends downward from 0
            ]
        )

        max_point = np.array(
            [
                self.origin[0] + self.dimensions[0],
                self.origin[1] + self.dimensions[1],
                self.dimensions[2] / 2,  # Z extends upward from 0
            ]
        )

        return min_point, max_point

    def contains_point(self, point: np.ndarray) -> bool:
        """
        Check if a point lies within the rectangular cell.

        Args:
            point: A 3D point to check

        Returns:
            bool: True if the point is inside the cell, False otherwise
        """
        point = np.array(point)
        if point.shape != (3,):
            raise ValueError("Point must be a 3D point")

        min_point, max_point = self.get_bounds()
        return np.all(point >= min_point) and np.all(point <= max_point)
