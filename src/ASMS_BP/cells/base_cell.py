from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class BaseCell(ABC):
    """
    Abstract base class for all cell types.

    Attributes:
        origin (np.ndarray): The (x, y) coordinates of the cell's origin in XY plane
    """

    origin: np.ndarray

    def __post_init__(self):
        """Validate inputs and convert to numpy arrays if needed."""
        self.origin = np.array(self.origin, dtype=float)
        if self.origin.shape != (2,):
            raise ValueError("Origin must be a 2D point (x,y)")
        self._validate_specific()
        self._volume = self._calculate_volume()

    @abstractmethod
    def _validate_specific(self) -> None:
        """Validate cell-specific parameters."""
        pass

    @abstractmethod
    def _calculate_volume(self) -> float:
        """Calculate the volume of the cell."""
        pass

    @property
    @abstractmethod
    def center(self) -> np.ndarray:
        """Get the center point of the cell."""
        pass

    @abstractmethod
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the minimum and maximum points that define the cell's bounds."""
        pass

    @abstractmethod
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if a point lies within the cell."""
        pass

    @property
    def volume(self) -> float:
        """Get the pre-calculated volume of the cell."""
        return self._volume
