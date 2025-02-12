from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .base_cell import BaseCell


@dataclass
class BuddingCell(BaseCell):
    """
    Represents a budding yeast cell composed of two connected ovoids (mother and bud).
    The cells are connected at a "neck" region, with the bud growing from the mother cell.

    Attributes:
        origin (np.ndarray): The (x, y) coordinates of the mother cell's center in XY plane
        mother_radius_x (float): Mother cell radius along X axis
        mother_radius_y (float): Mother cell radius along Y axis
        mother_radius_z (float): Mother cell radius along Z axis
        bud_radius_x (float): Bud radius along X axis
        bud_radius_y (float): Bud radius along Y axis
        bud_radius_z (float): Bud radius along Z axis
        bud_angle (float): Angle in radians from x-axis where bud emerges
        bud_distance (float): Distance between mother and bud centers
        neck_radius (float): Radius of the connecting neck region
    """

    mother_radius_x: float
    mother_radius_y: float
    mother_radius_z: float
    bud_radius_x: float
    bud_radius_y: float
    bud_radius_z: float
    bud_angle: float  # in radians
    bud_distance: float
    neck_radius: float

    def validate_specific(self) -> None:
        """Validate budding cell-specific parameters."""
        if any(
            r <= 0
            for r in [
                self.mother_radius_x,
                self.mother_radius_y,
                self.mother_radius_z,
                self.bud_radius_x,
                self.bud_radius_y,
                self.bud_radius_z,
                self.bud_distance,
                self.neck_radius,
            ]
        ):
            raise ValueError("All radii and distances must be positive")

        # Check if bud is too far to maintain connection
        max_distance = max(self.mother_radius_x, self.mother_radius_y) + max(
            self.bud_radius_x, self.bud_radius_y
        )
        if self.bud_distance > max_distance:
            raise ValueError("Bud distance too large for cells to remain connected")

        if self.neck_radius > min(
            min(self.mother_radius_x, self.mother_radius_y),
            min(self.bud_radius_x, self.bud_radius_y),
        ):
            raise ValueError("Neck radius cannot be larger than smallest cell radius")

    @property
    def bud_center(self) -> np.ndarray:
        """Calculate the center point of the bud cell."""
        dx = self.bud_distance * np.cos(self.bud_angle)
        dy = self.bud_distance * np.sin(self.bud_angle)
        return np.array([self.origin[0] + dx, self.origin[1] + dy, 0.0])

    @property
    def mother_center(self) -> np.ndarray:
        """Get the center point of the mother cell."""
        return np.array([self.origin[0], self.origin[1], 0.0])

    def calculate_volume(self) -> float:
        """
        Calculate the approximate volume of the budding cell.
        This is an approximation that considers the two ovoids and subtracts
        the overlapping region at the neck.
        """
        # Volume of mother cell
        mother_volume = (
            (4 / 3)
            * np.pi
            * (self.mother_radius_x * self.mother_radius_y * self.mother_radius_z)
        )

        # Volume of bud
        bud_volume = (
            (4 / 3)
            * np.pi
            * (self.bud_radius_x * self.bud_radius_y * self.bud_radius_z)
        )

        # Approximate volume of the neck region (cylinder)
        neck_height = self.neck_radius * 2  # approximate height
        neck_volume = np.pi * self.neck_radius**2 * neck_height

        # Total volume (adding neck to account for smooth transition)
        return mother_volume + bud_volume + neck_volume

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the minimum and maximum points that define the cell's bounding box.
        """
        bud_center = self.bud_center

        # Find the extremes considering both cells
        min_x = min(
            self.origin[0] - self.mother_radius_x, bud_center[0] - self.bud_radius_x
        )
        max_x = max(
            self.origin[0] + self.mother_radius_x, bud_center[0] + self.bud_radius_x
        )
        min_y = min(
            self.origin[1] - self.mother_radius_y, bud_center[1] - self.bud_radius_y
        )
        max_y = max(
            self.origin[1] + self.mother_radius_y, bud_center[1] + self.bud_radius_y
        )
        min_z = min(-self.mother_radius_z, -self.bud_radius_z)
        max_z = max(self.mother_radius_z, self.bud_radius_z)

        return (np.array([min_x, min_y, min_z]), np.array([max_x, max_y, max_z]))

    def contains_point(self, point: np.ndarray) -> bool:
        """
        Check if a point lies within the budding cell.
        A point is considered inside if it's in either the mother cell,
        the bud, or the neck region connecting them.
        """
        point = np.array(point)
        if point.shape != (3,):
            raise ValueError("Point must be a 3D point")

        # Check if point is in mother cell
        relative_to_mother = point - self.mother_center
        in_mother = (
            (relative_to_mother[0] / self.mother_radius_x) ** 2
            + (relative_to_mother[1] / self.mother_radius_y) ** 2
            + (relative_to_mother[2] / self.mother_radius_z) ** 2
        ) <= 1.0

        # Check if point is in bud
        relative_to_bud = point - self.bud_center
        in_bud = (
            (relative_to_bud[0] / self.bud_radius_x) ** 2
            + (relative_to_bud[1] / self.bud_radius_y) ** 2
            + (relative_to_bud[2] / self.bud_radius_z) ** 2
        ) <= 1.0

        # Check if point is in neck region
        if not (in_mother or in_bud):
            # Calculate vector from mother to bud center
            mother_to_bud = self.bud_center - self.mother_center
            mother_to_bud_normalized = mother_to_bud / np.linalg.norm(mother_to_bud)

            # Project point onto mother-bud axis
            point_relative_to_mother = point - self.mother_center
            projection_length = np.dot(
                point_relative_to_mother, mother_to_bud_normalized
            )

            # Find closest point on mother-bud axis
            projection = (
                self.mother_center + projection_length * mother_to_bud_normalized
            )

            # Check if point is within neck region
            distance_to_axis = np.linalg.norm(point - projection)
            projection_ratio = projection_length / self.bud_distance

            # Point is in neck if it's between mother and bud centers and within neck radius
            in_neck = (
                0 <= projection_ratio <= 1 and distance_to_axis <= self.neck_radius
            )
            return in_neck

        return in_mother or in_bud
