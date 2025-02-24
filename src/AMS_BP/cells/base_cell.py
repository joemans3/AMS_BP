from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pyvista as pv


@dataclass
class BaseCell(ABC):
    """
    Abstract base class for all cell types.
    """

    mesh: pv.DataSet

    def __post_init__(self):
        """Validate inputs and convert to numpy arrays if needed."""

        self._validate_specific()
        self._volume = self._calculate_volume()

    def _validate_specific(self) -> None:
        """Validate cell-specific parameters."""
        # Make sure the mesh is triangulated
        if not self.mesh.is_all_triangles:
            self.mesh = self.mesh.triangulate()

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """
        Check if point is inside the shape.

        Args:
            x, y, z: Coordinates of the point to test

        Returns:
            bool: True if point is inside or on boundary
        """
        # Use the select_enclosed_points filter which is more reliable
        point = np.array([[x, y, z]])
        points = pv.PolyData(point)

        # Use select_enclosed_points to check containment
        enclosed = points.select_enclosed_points(self.mesh, tolerance=1e-1)
        mask = enclosed["SelectedPoints"][0]
        return bool(mask)

    def reflecting_point(
        self,
        x1: float,
        y1: float,
        z1: float,
        x2: float,
        y2: float,
        z2: float,
        max_iterations: int = 10,
    ) -> Tuple[float, float, float]:
        """
        Reflect a point to the nearest boundary if it's outside the shape.

        Args:
            x1, y1, z1: A reference point inside the shape.
            x2, y2, z2: The candidate point to reflect.
            max_iterations: Maximum number of reflections to prevent infinite loops.

        Returns:
            A tuple (xr, yr, zr) representing the reflected point inside the shape.
        """
        if not self.contains_point(x1, y1, z1):
            raise ValueError(
                f"Reference point ({x1}, {y1}, {z1}) must be inside the shape."
            )

        # If already inside, return the point
        if self.contains_point(x2, y2, z2):
            return (x2, y2, z2)

        p1 = np.array([x1, y1, z1])
        p2 = np.array([x2, y2, z2])

        for _ in range(max_iterations):
            ray_direction = p2 - p1
            ray_length = np.linalg.norm(ray_direction)

            if ray_length == 0:
                return tuple(p1)  # No movement needed

            ray_direction /= ray_length  # Normalize direction

            # Perform ray tracing
            hit_points, _ = self.mesh.ray_trace(p1, p2, first_point=False)
            hit_points_shape = hit_points.shape
            # find the first non p1 point in hit_points
            if hit_points_shape[0] == 0:
                raise ValueError("Ray did not hit the mesh. Reflection failed.")
            if hit_points_shape[0] > 1:
                # if more than one point, take the first one which is not p1
                hit_points = hit_points[1]
            else:
                hit_points = hit_points[0]
            intersection = hit_points

            # Get the normal at the hit point
            cell_ids = self.mesh.find_closest_cell([intersection])
            if len(cell_ids) == 0:
                raise ValueError(
                    "Could not determine the cell at the intersection point."
                )

            normal = self.mesh.cell_normals[cell_ids[0]]

            normal /= np.linalg.norm(normal)  # Normalize the normal vector

            # Reflect the direction
            ray_direction = ray_direction - 2 * np.dot(ray_direction, normal) * normal

            # Compute the new candidate point
            p2 = intersection + ray_direction * (
                ray_length - np.linalg.norm(intersection - p1)
            )

            # Check if it's inside
            if self.contains_point(*p2):
                return tuple(p2)

        raise RuntimeError("Max iterations reached. Reflection may not have converged.")

    def _visualize(self, points: Optional[List[Tuple[float, float, float]]] = None):
        """Visualize the capsule and optional points."""
        plotter = pv.Plotter()
        plotter.add_mesh(self.mesh, style="wireframe")

        if points:
            point_cloud = pv.PolyData(np.array(points))
            plotter.add_mesh(point_cloud, color="red", point_size=10)

        # axis
        plotter.add_axes()
        plotter.show()

    def _calculate_volume(self) -> float:
        self._volume = self.mesh.volume
        return self._volume

    def _calculate_bounds(self) -> Tuple[float, float, float, float, float, float]:
        self._bounds = self.mesh.bounds
        return self._bounds

    @property
    def bounds(self):
        """The bounding box -> [xmin,xmax,ymin,ymax,zmin,zmax]"""
        return self._bounds

    @property
    def volume(self) -> float:
        """Get the pre-calculated volume of the cell."""
        return self._volume
