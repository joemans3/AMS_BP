from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class QuantumEfficiency:
    """
    Represents the quantum efficiency curve of a detector.

    The wavelength values should be specified in nanometers (nm).
    """

    wavelength_qe: Dict[float, float] = field(default_factory=dict)
    _cached_wavelength_qe: Dict[float, float] = field(init=False)

    def __post_init__(self):
        """
        Validate the quantum efficiency values and wavelengths in nanometers.
        Initialize numpy arrays for faster interpolation.
        """
        # Validate all values first
        invalid_wavelengths = [w for w in self.wavelength_qe if not 100 <= w <= 1100]
        if invalid_wavelengths:
            raise ValueError(
                f"Wavelengths must be between 100-1100 nm, got {invalid_wavelengths}"
            )

        invalid_qes = [qe for qe in self.wavelength_qe.values() if not 0 <= qe <= 1]
        if invalid_qes:
            raise ValueError(
                f"Quantum efficiencies must be between 0 and 1, got {invalid_qes}"
            )

        self._cached_wavelength_qe = deepcopy(self.wavelength_qe)
        # Sort and store as numpy arrays for faster operations
        sorted_items = sorted(self.wavelength_qe.items())
        self._wavelengths = np.array([x[0] for x in sorted_items])
        self._efficiencies = np.array([x[1] for x in sorted_items])

    def get_qe(self, wavelength: float) -> float:
        """
        Get the quantum efficiency for a specific wavelength using linear interpolation.
        Stores interpolated values in the dictionary for future lookups.

        Args:
            wavelength: The wavelength in nanometers

        Returns:
            Interpolated quantum efficiency value between 0 and 1
        """
        try:
            return self._cached_wavelength_qe[wavelength]
        except KeyError:
            # Quick return for out-of-bounds values
            if wavelength < self._wavelengths[0] or wavelength > self._wavelengths[-1]:
                return 0.0

            # Calculate interpolated value
            qe = float(np.interp(wavelength, self._wavelengths, self._efficiencies))

            # Store the interpolated value for future use
            self._cached_wavelength_qe[wavelength] = qe
            return qe
