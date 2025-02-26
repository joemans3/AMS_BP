from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pyvista as pv

from .base_cell import BaseCell
from .typedefs import Vector3D


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
