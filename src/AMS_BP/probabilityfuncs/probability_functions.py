"""
Top Hat Probability Function Module
===================================

This module defines a class for handling the probability function of multiple top-hat-shaped subspaces
within a larger spatial environment. A "top-hat" distribution is a flat or constant distribution within a
defined subspace and zero outside of it, commonly used to model regions with a uniform density surrounded
by an area with a different (typically lower) density.

Since top-hat distributions are not continuous or analytical probability distributions, their probability
must be computed manually. This module provides a class, `multiple_top_hat_probability`, to handle the
calculation and retrieval of the probability values based on input positions. The probability is computed
as a constant value inside the top-hat subspaces and a different constant value outside them.

Key Features:
-------------
- Probability calculation within and outside defined subspaces.
- Support for multiple top-hat subspaces, each defined by its center and radius.
- Ability to update parameters and recalculate probabilities as needed.

Usage:
------
An instance of the `multiple_top_hat_probability` class is initialized with the number of subspaces,
their centers, radii, density difference, and overall space size. Once initialized, the object can be
called with a position to return the probability at that location.

Example:
```python
prob_func = multiple_top_hat_probability(
    num_subspace=3,
    subspace_centers=np.array([[1, 1], [2, 2], [3, 3]]),
    subspace_radius=np.array([1.0, 0.5, 0.75]),
    density_dif=0.2,
    cell=BaseCell type
)

prob = prob_func(np.array([1.5, 1.5]))

Note:
-----
After initialization, do not change the parameters directly. Use the update_parameters method to modify any values.
"""

from collections.abc import Callable
from typing import Tuple

import numpy as np

from ..cells import BaseCell


def generate_points_from_cls(
    pdf: Callable,
    total_points: int,
    volume: float,
    bounds: Tuple[float, float, float, float, float, float],
    density_dif: float,
) -> np.ndarray:
    """
    Generates random (x, y, z) points using the accept/reject method based on a given distribution.

    Parameters:
    -----------
    pdf : callable
        Probability density function to sample from.
    total_points : int
        Number of points to generate.
    bound : list with the following
        min_x : float
            Minimum x value for sampling.
        max_x : float
            Maximum x value for sampling.
        min_y : float
            Minimum y value for sampling.
        max_y : float
            Maximum y value for sampling.
        min_z : float
            Minimum z value for sampling.
        max_z : float
            Maximum z value for sampling.
    volume : float,
        volume of region sampling
    density_dif : float
        Scaling factor for density differences.

    Returns:
    --------
    np.ndarray
        Array of generated (x, y, z) points.
    """
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    xyz_coords = []
    while len(xyz_coords) < total_points:
        # generate candidate variable
        var = np.random.uniform([min_x, min_y, min_z], [max_x, max_y, max_z])
        # generate varibale to condition var1
        var2 = np.random.uniform(0, 1)
        # apply condition
        pdf_val = pdf(var)
        if var2 < ((1.0 / density_dif) * volume) * pdf_val:
            xyz_coords.append(var)
    return np.array(xyz_coords)


class multiple_top_hat_probability:
    """Class for the probability function of multiple top hats within different cell types."""

    def __init__(
        self,
        num_subspace: int,
        subspace_centers: np.ndarray,
        subspace_radius: np.ndarray,
        density_dif: float,
        cell: BaseCell,
    ) -> None:
        """
        Initialize the probability function.

        Parameters:
        -----------
        num_subspace : int
            Number of subspaces
        subspace_centers : np.ndarray
            Centers of each subspace (shape: [num_subspace, 3])
        subspace_radius : np.ndarray
            Radius of each subspace
        density_dif : float
            Difference in density between subspaces and non-subspaces
        cell : BaseCell
            Cell object defining the boundary
        """
        self.num_subspace = num_subspace
        self.subspace_centers = np.array(subspace_centers)
        self.subspace_radius = np.array(subspace_radius)
        self.density_dif = density_dif
        self.cell = cell

        # Calculate probabilities using cell's volume property
        total_volume = self.cell.volume
        self.subspace_probability = self._calculate_subspace_probability(
            total_volume, self.density_dif
        )
        self.non_subspace_probability = self._calculate_non_subspace_probability(
            total_volume, self.density_dif, self.num_subspace, self.subspace_radius
        )

    def __call__(self, position: np.ndarray, **kwargs) -> float:
        """Returns the probability given a coordinate"""
        if not isinstance(position, np.ndarray):
            raise TypeError("Position must be a numpy array.")

        # First check if point is within the cell
        if not self.cell.contains_point(*position):
            return 0.0

        # Then check if point is within any subspace
        for i in range(self.num_subspace):
            if (
                np.linalg.norm(position - self.subspace_centers[i])
                <= self.subspace_radius[i]
            ):
                return self.subspace_probability

        return self.non_subspace_probability

    def _calculate_subspace_probability(
        self, total_volume: float, density_dif: float
    ) -> float:
        """Calculate probability within subspaces"""
        return density_dif / total_volume

    def _calculate_non_subspace_probability(
        self,
        total_volume: float,
        density_dif: float,
        num_subspace: int,
        subspace_radius: np.ndarray,
    ) -> float:
        """Calculate probability outside subspaces"""
        total_subspace_volume = (
            num_subspace * (4 / 3) * np.pi * np.mean(subspace_radius) ** 3
        )
        remaining_volume = total_volume - total_subspace_volume

        if remaining_volume <= 0:
            return 0.0

        return 1.0 / total_volume

    @property
    def num_subspace(self) -> int:
        """Returns the number of subspaces."""
        return self._num_subspace

    @num_subspace.setter
    def num_subspace(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError("Number of subspaces must be an integer.")
        self._num_subspace = value

    @property
    def subspace_centers(self) -> np.ndarray:
        """Returns the centers of the subspaces."""
        return self._subspace_centers

    @subspace_centers.setter
    def subspace_centers(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("Subspace centers must be a numpy array.")
        self._subspace_centers = value

    @property
    def subspace_radius(self) -> np.ndarray:
        """Returns the radius of the subspaces."""
        return self._subspace_radius

    @subspace_radius.setter
    def subspace_radius(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("Subspace radius must be a numpy array.")
        self._subspace_radius = value

    @property
    def density_dif(self) -> float:
        """Returns the difference in density between the subspaces and the rest of the space."""
        return self._density_dif

    @density_dif.setter
    def density_dif(self, value: float) -> None:
        self._density_dif = value

    @property
    def cell(self) -> BaseCell:
        """Returns the cell object."""
        return self._cell

    @cell.setter
    def cell(self, value: BaseCell) -> None:
        self._cell = value

    @property
    def subspace_probability(self) -> float:
        return self._subspace_probability

    @subspace_probability.setter
    def subspace_probability(self, value: float) -> None:
        self._subspace_probability = value

    @property
    def non_subspace_probability(self) -> float:
        """Returns the probability of the non-subspaces."""
        return self._non_subspace_probability

    @non_subspace_probability.setter
    def non_subspace_probability(self, value: float) -> None:
        self._non_subspace_probability = value

    def update_parameters(
        self,
        num_subspace: int | None = None,
        subspace_centers: np.ndarray | None = None,
        subspace_radius: np.ndarray | None = None,
        density_dif: float | None = None,
        cell: BaseCell | None = None,
    ) -> None:
        """Updates the parameters of the probability function."""
        if num_subspace is not None:
            self.num_subspace = num_subspace
        if subspace_centers is not None:
            self.subspace_centers = subspace_centers
        if subspace_radius is not None:
            self.subspace_radius = subspace_radius
        if density_dif is not None:
            self.density_dif = density_dif
        if cell is not None:
            self.cell = cell

        self.subspace_probability = self._calculate_subspace_probability(
            self.cell.volume, self.density_dif
        )
        self.non_subspace_probability = self._calculate_non_subspace_probability(
            self.cell.volume, self.density_dif, self.num_subspace, self.subspace_radius
        )
