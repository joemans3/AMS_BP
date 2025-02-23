from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pyvista as pv
from typing_extensions import List

from .base_cell import BaseCell


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
    # Normalize direction vector
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)

    # Create cylinder for the body
    cylinder = pv.Cylinder(
        center=center, direction=direction, radius=radius, height=height
    )

    # Create spheres for the caps
    half_height = height / 2
    sphere1_center = np.array(center) + direction * half_height
    sphere2_center = np.array(center) - direction * half_height

    sphere1 = pv.Sphere(radius=radius, center=sphere1_center)

    sphere2 = pv.Sphere(radius=radius, center=sphere2_center)

    # Combine the shapes using boolean operations
    capsule = cylinder.triangulate().boolean_union(sphere1).boolean_union(sphere2)
    capsule = capsule.clean()
    capsule = capsule.fill_holes(1)
    edges = capsule.extract_feature_edges(feature_edges=False, manifold_edges=False)
    assert edges.n_cells == 0, "Mesh has non-manifold edges"
    return RodCell(
        mesh=capsule, center=center, direction=direction, height=height, radius=radius
    )
