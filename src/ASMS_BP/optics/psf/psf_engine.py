from dataclasses import dataclass
from functools import cache, cached_property, lru_cache
from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PSFParameters:
    """Parameters for PSF (Point Spread Function) generation.

    Attributes:
        wavelength: Light wavelength in nanometers
        numerical_aperture: Numerical aperture of the optical system
        pixel_size: Size of pixels in micrometers
        z_step: Axial step size in micrometers
        refractive_index: Refractive index of the medium (default: 1.0 for air)
    """

    wavelength: float
    numerical_aperture: float
    pixel_size: float
    z_step: float
    refractive_index: float = 1.0

    # def __post_init__(self) -> None:
    #     """Validate parameters after initialization."""
    #     if any(
    #         param <= 0
    #         for param in (
    #             self.wavelength,
    #             self.numerical_aperture,
    #             self.pixel_size,
    #             self.z_step,
    #             self.refractive_index,
    #         )
    #     ):
    #         raise ValueError("All parameters must be positive numbers")
    #     if self.numerical_aperture >= self.refractive_index:
    #         raise ValueError("Numerical aperture must be less than refractive index")

    @cached_property
    def wavelength_um(self) -> float:
        """Wavelength in micrometers."""
        return self.wavelength / 1000.0


class PSFEngine:
    """Engine for generating various microscope Point Spread Functions.

    This class implements calculations for both 2D and 3D Point Spread Functions
    using Gaussian approximations.
    """

    def __init__(self, params: PSFParameters):
        """Initialize PSF engine with given parameters."""
        self.params = params
        self._initialize_calculations()

    def _initialize_calculations(self) -> None:
        """Initialize commonly used calculations."""
        self._sigma_xy = _calculate_sigma_xy(
            self.params.wavelength_um, self.params.numerical_aperture
        )
        self._sigma_z = _calculate_sigma_z(
            self.params.wavelength_um,
            self.params.numerical_aperture,
            self.params.refractive_index,
        )
        self._psf_size = calculate_psf_size(
            sigma_xy=self._sigma_xy,
            pixel_size=self.params.pixel_size,
            sigma_z=self._sigma_z,
        )
        self._grid_xy = _generate_grid(self._psf_size, self.params.pixel_size)

        # Pre-calculate normalized sigma values
        self._norm_sigma_xy = self._sigma_xy / 2.355
        self._norm_sigma_z = self._sigma_z / 2.355

    @lru_cache(maxsize=128)
    def psf_z(self, z_val: float) -> NDArray[np.float64]:
        """Generate z=z_val Gaussian approximation of PSF.

        Args:
            z_val: Z-position in micrometers

        Returns:
            2D array containing the PSF at given z position
        """
        x, y = self._grid_xy

        # Vectorized calculation
        r_squared = (x / self._norm_sigma_xy) ** 2 + (y / self._norm_sigma_xy) ** 2
        z_term = (z_val / self._norm_sigma_z) ** 2
        return np.exp(-0.5 * (r_squared + z_term))

    @lru_cache(maxsize=128)
    def psf_z_xy0(self, z_val: float) -> float:
        """Generate z=z_val Gaussian approximation of PSF with x=y=0.

        Args:
            z_val: Z-position in micrometers

        Returns:
            PSF value at x=y=0 and given z position
        """
        return np.exp(-0.5 * (z_val / self._norm_sigma_z) ** 2)

    @cache
    def _3d_normalization_A(
        self, sigma_z: float, sigma_x: float, sigma_y: float
    ) -> float:
        return 1.0 / (((2.0 * np.pi) ** (3.0 / 2.0)) * sigma_x * sigma_y * sigma_z)

    @cache
    def _2d_normalization_A(self, sigma_x: float, sigma_y: float) -> float:
        return 1.0 / ((2.0 * np.pi) * sigma_x * sigma_y)

    @staticmethod
    def normalize_psf(
        psf: NDArray[np.float64], mode: Literal["sum", "max", "energy"] = "sum"
    ) -> NDArray[np.float64]:
        """Normalize PSF with different schemes.

        Args:
            psf: Input PSF array
            mode: Normalization mode
                - 'sum': Normalize so sum equals 1 (energy conservation)
                - 'max': Normalize so maximum equals 1
                - 'energy': Normalize so squared sum equals 1

        Returns:
            Normalized PSF array

        Raises:
            ValueError: If unknown normalization mode is specified
        """
        if not np.any(psf):  # Check if array is all zeros
            return psf

        normalizers = {
            "sum": np.sum,
            "max": np.max,
            "energy": lambda x: np.sqrt(np.sum(x**2)),
        }

        try:
            normalizer = normalizers[mode]
            return psf / normalizer(psf)
        except KeyError:
            raise ValueError(
                f"Unknown normalization mode: {mode}. Valid modes: {list(normalizers.keys())}"
            )


@cache
def _calculate_sigma_xy(wavelength_um: float, numerical_aperture: float) -> float:
    """Calculate lateral sigma value."""
    return 0.61 * wavelength_um / numerical_aperture


@cache
def _calculate_sigma_z(
    wavelength_um: float, numerical_aperture: float, refractive_index: float
) -> float:
    """Calculate axial sigma value."""
    return 2.0 * wavelength_um * refractive_index / (numerical_aperture**2)


@cache
def _generate_grid(
    size: Tuple[int, int], pixel_size: float
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate coordinate grids for PSF calculation.

    Args:
        size: Tuple of (height, width) for the grid

    Returns:
        Tuple of x and y coordinate arrays
    """
    y, x = np.ogrid[: size[0], : size[1]]
    center_y, center_x = [(s - 1) / 2 for s in size]
    y = (y - center_y) * pixel_size
    x = (x - center_x) * pixel_size
    return x, y


@cache
def calculate_psf_size(
    sigma_xy: float, pixel_size: float, sigma_z: float, z_size: Optional[int] = None
) -> Tuple[int, ...]:
    """Calculate appropriate PSF size based on physical parameters.

    Args:
        z_size: Optional number of z-planes for 3D PSF

    Returns:
        Tuple of dimensions (z,y,x) or (y,x) for the PSF calculation
    """
    # Calculate radius to capture important features (2x Airy radius)
    r_psf = 2 * sigma_xy

    # Convert to pixels and ensure odd number
    pixels_xy = int(np.ceil(r_psf / pixel_size))
    pixels_xy += (pixels_xy + 1) % 2

    if z_size is not None:
        pixels_z = int(np.ceil(2 * sigma_z / z_size))
        pixels_z += (pixels_z + 1) % 2
        return (pixels_z, pixels_xy, pixels_xy)

    return (pixels_xy, pixels_xy)
