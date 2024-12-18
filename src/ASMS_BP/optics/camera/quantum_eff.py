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
    
    def __post_init__(self):
        """Validate the quantum efficiency values and wavelengths in nanometers."""
        for wavelength, qe in self.wavelength_qe.items():

            if not 100 <= wavelength <= 1100:
                raise ValueError(f"Wavelength must be between 100-1100 nm, got {wavelength}")
            if not 0 <= qe <= 1:
                raise ValueError(f"Quantum efficiency must be between 0 and 1, got {qe}")
    
    def get_qe(self, wavelength: float) -> float:
        """
        Get the quantum efficiency for a specific wavelength using linear interpolation.
        
        Args:
            wavelength: The wavelength in nanometers
            
        Returns:
            Interpolated quantum efficiency value between 0 and 1
        """
        if wavelength in self.wavelength_qe:
            return self.wavelength_qe[wavelength]
        
        # Find nearest wavelengths for interpolation
        wavelengths = np.array(list(self.wavelength_qe.keys()))
        if wavelength < wavelengths.min() or wavelength > wavelengths.max():
            return 0.0  # Outside the defined range
            
        # Linear interpolation
        idx = np.searchsorted(wavelengths, wavelength)
        x1, x2 = wavelengths[idx-1], wavelengths[idx]
        y1, y2 = self.wavelength_qe[x1], self.wavelength_qe[x2]
        
        return y1 + (y2 - y1) * (wavelength - x1) / (x2 - x1)
