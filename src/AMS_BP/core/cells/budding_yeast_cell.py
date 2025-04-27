from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pyvista as pv
from boundedfbm.cells.base_cell import BaseCell
from boundedfbm.cells.typedefs import Vector3D


@dataclass
class BuddingCell(BaseCell):
    """
    Represents a budding yeast cell composed of two connected ovoids (mother and bud).
    The cells are connected at a "neck" region, with the bud growing from the mother cell.

    Attributes:
        center (np.ndarray): The (x, y, z) coordinates of the mother cell's center in XYZ plane
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

    center: np.ndarray | List[float] | Tuple
    mother_radius_x: float
    mother_radius_y: float
    mother_radius_z: float
    bud_radius_x: float
    bud_radius_y: float
    bud_radius_z: float
    bud_angle: float
    bud_distance: float
    neck_radius: float

    def contains_point_fallback(
        self, x: float, y: float, z: float, tolerance: float = 1e-3
    ) -> bool:
        """
        Determines if a given point is inside the BuddingCell.
        A point is inside if it's within the mother cell, the bud, or the neck region.

        Args:
            point (np.ndarray | List[float] | Tuple): The (x, y, z) coordinates of the point to check

        Returns:
            bool: True if the point is inside the cell, False otherwise
        """
        # Ensure point is a numpy array for vector operations
        point = np.array([x, y, z])
        mother_center = np.array(self.center)

        # Calculate bud center based on angle and distance
        bud_x = mother_center[0] + self.bud_distance * np.cos(self.bud_angle)
        bud_y = mother_center[1] + self.bud_distance * np.sin(self.bud_angle)
        bud_center = np.array([bud_x, bud_y, mother_center[2]])

        # Check if point is inside the mother cell (scaled ellipsoid)
        x, y, z = point - mother_center
        mother_distance_squared = (
            (x / self.mother_radius_x) ** 2
            + (y / self.mother_radius_y) ** 2
            + (z / self.mother_radius_z) ** 2
        )
        if mother_distance_squared <= 1:
            return True

        # Check if point is inside the bud (scaled ellipsoid)
        x, y, z = point - bud_center
        bud_distance_squared = (
            (x / self.bud_radius_x) ** 2
            + (y / self.bud_radius_y) ** 2
            + (z / self.bud_radius_z) ** 2
        )
        if bud_distance_squared <= 1:
            return True

        # Check if point is inside the neck region
        # First, project the point onto the line between mother and bud centers
        mother_to_bud_vec = bud_center - mother_center
        mother_to_bud_length = np.linalg.norm(mother_to_bud_vec)
        mother_to_bud_unit = mother_to_bud_vec / mother_to_bud_length

        mother_to_point_vec = point - mother_center
        projection_length = np.dot(mother_to_point_vec, mother_to_bud_unit)

        # Calculate minimum distance from point to the center line
        if 0 <= projection_length <= mother_to_bud_length:
            projection_point = mother_center + projection_length * mother_to_bud_unit
            distance_to_line = np.linalg.norm(point - projection_point)

            # The neck is modeled as a truncated cone
            # Interpolate the radius at this point along the neck
            if projection_length <= self.mother_radius_x:
                # Within mother cell radius from the center
                local_radius = self.neck_radius
            elif projection_length >= mother_to_bud_length - self.bud_radius_x:
                # Within bud cell radius from the bud center
                local_radius = self.neck_radius
            else:
                local_radius = self.neck_radius

            return distance_to_line <= local_radius

        return False


def make_BuddingCell(
    center: np.ndarray | List[float] | Tuple,
    mother_radius_x: float,
    mother_radius_y: float,
    mother_radius_z: float,
    bud_radius_x: float,
    bud_radius_y: float,
    bud_radius_z: float,
    bud_angle: float,
    bud_distance: float,
    neck_radius: float,
) -> BuddingCell:
    """
    Create a budding yeast cell using PyVista meshes.

    Args:
        center: Center point of the mother cell
        mother_radius_x/y/z: Radii of the mother cell along each axis
        bud_radius_x/y/z: Radii of the bud cell along each axis
        bud_angle: Angle in radians from x-axis where bud emerges
        bud_distance: Distance between mother and bud centers
        neck_radius: Radius of the connecting neck region

    Returns:
        BuddingCell: Instance with PyVista mesh
    """
    # Validate inputs
    center = np.array(center)
    if center.shape != (3,):
        raise ValueError("Center must be a 3D point")

    # Calculate bud center
    bud_center = np.array(
        [
            center[0] + bud_distance * np.cos(bud_angle),
            center[1] + bud_distance * np.sin(bud_angle),
            center[2],
        ]
    )

    # Create mother cell ellipsoid
    mother = pv.ParametricEllipsoid(
        xradius=mother_radius_x,
        yradius=mother_radius_y,
        zradius=mother_radius_z,
        center=center,
    )

    # Create bud cell ellipsoid
    bud = pv.ParametricEllipsoid(
        xradius=bud_radius_x,
        yradius=bud_radius_y,
        zradius=bud_radius_z,
        center=bud_center,
    )

    # Create neck region (cylinder)
    # Calculate direction vector from mother to bud
    direction = bud_center - center
    direction = direction / np.linalg.norm(direction)

    # Create cylinder for neck
    cylinder = pv.Cylinder(
        center=(center + bud_center) / 2,  # Midpoint
        direction=direction,
        radius=neck_radius,
        height=bud_distance,
    )

    # Combine shapes using boolean operations
    # First combine mother and neck
    mother_and_neck = mother.boolean_union(cylinder)

    # Then add the bud
    complete_cell = mother_and_neck.boolean_union(bud)

    # Clean up the mesh
    complete_cell = complete_cell.clean()
    complete_cell = complete_cell.fill_holes(1)

    # Verify mesh integrity
    edges = complete_cell.extract_feature_edges(
        feature_edges=False, manifold_edges=False
    )
    assert edges.n_cells == 0, "Mesh has non-manifold edges"

    return BuddingCell(
        mesh=complete_cell,
        center=center,
        mother_radius_x=mother_radius_x,
        mother_radius_y=mother_radius_y,
        mother_radius_z=mother_radius_z,
        bud_radius_x=bud_radius_x,
        bud_radius_y=bud_radius_y,
        bud_radius_z=bud_radius_z,
        bud_angle=bud_angle,
        bud_distance=bud_distance,
        neck_radius=neck_radius,
    )


@dataclass
class BuddingCellParams:
    center: Vector3D
    mother_radius_x: float
    mother_radius_y: float
    mother_radius_z: float
    bud_radius_x: float
    bud_radius_y: float
    bud_radius_z: float
    bud_angle: float
    bud_distance: float
    neck_radius: float

    @classmethod
    def validate_center(cls, value):
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 3:
            raise ValueError("center must be a 3D vector")

    @classmethod
    def validate_mother_radius_x(cls, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("mother_radius_x must be a positive number")

    @classmethod
    def validate_mother_radius_y(cls, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("mother_radius_y must be a positive number")

    @classmethod
    def validate_mother_radius_z(cls, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("mother_radius_z must be a positive number")

    @classmethod
    def validate_bud_radius_x(cls, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("bud_radius_x must be a positive number")

    @classmethod
    def validate_bud_radius_y(cls, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("bud_radius_y must be a positive number")

    @classmethod
    def validate_bud_radius_z(cls, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("bud_radius_z must be a positive number")

    @classmethod
    def validate_bud_angle(cls, value):
        if not isinstance(value, (int, float)):
            raise ValueError("bud_angle must be a number")

    @classmethod
    def validate_bud_distance(cls, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("bud_distance must be a positive number")

    @classmethod
    def validate_neck_radius(cls, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("neck_radius must be a positive number")
