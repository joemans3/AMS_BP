from dataclasses import dataclass
from functools import cache
from typing import Optional, Tuple

import numpy as np


@dataclass
class PSFParameters:
    """Parameters for PSF generation"""

    wavelength: float  # in nm
    numerical_aperture: float
    pixel_size: float  # in um
    z_step: float  # axial step size in um
    refractive_index: float = 1.0

    @property
    def wavelength_um(self) -> float:
        """Convert wavelength from nm to um"""
        return self.wavelength / 1000.0


class PSFEngine:
    """Engine for generating various microscope Point Spread Functions"""

    def __init__(self, params: PSFParameters):
        self.params = params

    def _sigma_calc_xy(self) -> float:
        return 0.61 * self.params.wavelength_um / (self.params.numerical_aperture)

    def _sigma_calc_z(self) -> float:
        return (
            2.0
            * self.params.wavelength_um
            * self.params.refractive_index
            / (self.params.numerical_aperture**2)
        )

    def generate_grid(self, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate coordinate grids for PSF calculation"""
        y, x = np.indices(size)
        center_y, center_x = [(s - 1) / 2 for s in size]
        y = (y - center_y) * self.params.pixel_size
        x = (x - center_x) * self.params.pixel_size
        return x, y

    def calculate_psf_size(self, z_size: Optional[int] = None) -> Tuple[int, ...]:
        """
        Calculate appropriate PSF size based on physical parameters

        Args:
            z_size: Optional number of z-planes for 3D PSF

        Returns:
            Tuple of dimensions (z,y,x) or (y,x) for the PSF calculation
        """
        # Calculate radius of Airy disk first zero (in um)
        r_airy = self._sigma_calc_xy()

        # Use 3x this radius to capture important features
        r_psf = 2 * r_airy
        # Convert to pixels (round up to odd number to maintain central pixel)
        pixels_needed = int(np.ceil(r_psf / self.params.pixel_size))
        if pixels_needed % 2 == 0:
            pixels_needed += 1

        if z_size is not None:
            # For 3D PSFs, ensure z_size is odd
            sigma_z = self._sigma_calc_z()
            pixels_needed_z = 2 * sigma_z
            pixels_needed_z = int(np.ceil(pixels_needed_z / z_size))
            if pixels_needed_z % 2 == 0:
                pixels_needed_z += 1
            return (pixels_needed_z, pixels_needed, pixels_needed)

        return (pixels_needed, pixels_needed)

    @cache
    def psf_z(self, z_val: float) -> np.ndarray:
        """Generate z=z_val Gaussian approximation of PSF
        returned normalized values"""
        psf_size = self.calculate_psf_size()

        x, y = self.generate_grid(psf_size)

        sigma_xy = self._sigma_calc_xy() / 2.355
        sigma_z = self._sigma_calc_z() / 2.355

        psf = np.exp(
            -0.5 * ((x / sigma_xy) ** 2 + (y / sigma_xy) ** 2 + (z_val / sigma_z) ** 2)
        )
        return psf  # * self._gaussian_3d_normalization_A(sigma_z = sigma_z, sigma_x = sigma_xy, sigma_y = sigma_xy)

    @cache
    def psf_z_xy0(self, z_val: float) -> np.ndarray:
        """Generate z=z_val Gaussian approximation of PSF with x=y=0
        returned normalized values"""

        sigma_z = self._sigma_calc_z() / 2.355

        psf = np.exp(-0.5 * (z_val / sigma_z) ** 2)
        return psf

    @cache
    def _3d_normalization_A(
        self, sigma_z: float, sigma_x: float, sigma_y: float
    ) -> float:
        return 1.0 / (((2.0 * np.pi) ** (3.0 / 2.0)) * sigma_x * sigma_y * sigma_z)

    @cache
    def _2d_normalization_A(self, sigma_x: float, sigma_y: float) -> float:
        return 1.0 / ((2.0 * np.pi) * sigma_x * sigma_y)

    @staticmethod
    def normalize_psf(psf: np.ndarray, mode: str = "sum") -> np.ndarray:
        """
        Normalize PSF with different schemes

        Args:
            psf: Input PSF array
            mode: Normalization mode ('sum', 'max', 'energy')
                - 'sum': Normalize so sum equals 1 (energy conservation)
                - 'max': Normalize so maximum equals 1
                - 'energy': Normalize so squared sum equals 1

        Returns:
            np.ndarray: Normalized PSF

        Raises:
            ValueError: If unknown normalization mode is specified
        """
        # check if all zeros
        psf_sum = np.sum(psf)
        if not psf_sum:
            return psf
        if mode == "sum":
            return psf / psf_sum
        elif mode == "max":
            return psf / np.max(psf)
        elif mode == "energy":
            return psf / np.sqrt(np.sum(psf**2))
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")
