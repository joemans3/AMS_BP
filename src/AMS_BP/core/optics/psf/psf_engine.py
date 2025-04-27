from dataclasses import dataclass
from functools import cache, cached_property, lru_cache
from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

AIRYFACTOR = 1.0


@dataclass(frozen=True)
class PSFParameters:
    """Parameters for emission PSF (Point Spread Function) at the detector.


    This class defines parameters that determine how light from a point source
    (e.g., a fluorescent molecule) diffracts through the collection optics
    to form a pattern at the detector.

    Attributes:
        emission_wavelength: Emission wavelength in nanometers
        numerical_aperture: Numerical aperture of the collection objective
        pixel_size: Size of pixels in micrometers at the detector
        z_step: Axial step size in micrometers
        refractive_index: Refractive index of the medium (default: 1.0 for air)
        pinhole_diameter: Diameter of the pinhole in micrometers (default: None for widefield)
                         The pinhole spatially filters the emitted light before it reaches
                         the detector.
    """

    emission_wavelength: float
    numerical_aperture: float
    pixel_size: float
    z_step: float
    refractive_index: float = 1.0
    pinhole_diameter: Optional[float] = None  # um

    @cached_property
    def wavelength_um(self) -> float:
        """Emission wavelength in micrometers."""
        return self.emission_wavelength / 1000.0

    @cached_property
    def pinhole_radius(self) -> Optional[float]:
        """Pinhole radius in micrometers."""
        return (
            self.pinhole_diameter / 2.0 if self.pinhole_diameter is not None else None
        )


class PSFEngine:
    """Engine for calculating emission light PSF at the detector.

    This class calculates how light from a point source (like a fluorescent molecule)
    spreads due to diffraction through the collection optics to form a pattern at
    the detector. For confocal systems, it can include the effect of a pinhole
    that spatially filters the light before detection.

    Note: This PSF describes only the diffraction of emitted light through the
    collection optics. While a confocal microscope uses focused illumination to
    excite molecules, that illumination pattern does not affect how the emitted
    light diffracts to form the PSF we calculate here.
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
        self._norm_sigma_xy = self._sigma_xy / 2.0
        self._norm_sigma_z = self._sigma_z / 2.0

        # Generate pinhole mask if specified
        if self.params.pinhole_radius is not None:
            if self.params.pinhole_radius < AIRYFACTOR * self._sigma_xy:
                RuntimeWarning(
                    f"Pinhole size ({self.params.pinhole_radius} um) is smaller than {AIRYFACTOR} times the Airy lobe. This will diffract the emission light in the pinhole; an ideal pinhole size for this setup is {self._sigma_xy} um."
                )
                #
                # raise ValueError(
                #     f"Pinhole size ({self.params.pinhole_radius} um) is smaller than {AIRYFACTOR} times the Airy lobe. This will diffract the emission light in the pinhole; an ideal pinhole size for this setup is {self._sigma_xy} um."
                # )
            self._pinhole_mask = self._generate_pinhole_mask()
        else:
            self._pinhole_mask = None

    def _generate_pinhole_mask(self) -> NDArray[np.float64]:
        """Generate a binary mask representing the pinhole's spatial filtering.
        The pinhole is centered on the grid, blocking emission light based on position
        in the image plane, affecting what portion of the diffracted light reaches
        the detector.
        """
        x, y = self._grid_xy

        # Calculate the grid center
        x_center = (x.max() + x.min()) / 2
        y_center = (y.max() + y.min()) / 2

        # Calculate radial distance from grid center
        r = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)

        return (r <= self.params.pinhole_radius).astype(np.float64)

    @lru_cache(maxsize=128)
    def psf_z(
        self, x_val: float, y_val: float, z_val: float, norm_scale: bool = True
    ) -> NDArray[np.float64]:
        """Calculate the PSF at the detector for a point source at z_val.

        This represents how light from a point source at position z_val
        diffracts through the collection optics to form a pattern at the
        detector. If a pinhole is present, it spatially filters this pattern.

        Args:
            x_val: x-position of the point source in micrometers
            y_val: y-position of the point source in micrometers
            z_val: Z-position of the point source in micrometers

        Returns:
            2D array containing the light intensity pattern at the detector
        """
        x, y = self._grid_xy
        sigma_xy_z_squared = (self._norm_sigma_xy**2) * (
            1 + (z_val / self._norm_sigma_z) ** 2
        )

        # Calculate how light from the point source diffracts through collection optics
        r_squared = (x - x_val % self.params.pixel_size) ** 2 + (
            y - y_val % self.params.pixel_size
        ) ** 2
        psf_at_detector = np.exp(-0.5 * (r_squared / sigma_xy_z_squared))

        if norm_scale:
            psf_at_detector = self.normalize_psf(
                psf_at_detector, mode="sum"
            ) * self.psf_z_xy0(z_val)

        if self._pinhole_mask is not None:
            # Apply pinhole's spatial filtering
            return psf_at_detector * self._pinhole_mask

        return psf_at_detector

    @lru_cache(maxsize=128)
    def psf_z_xy0(self, z_val: float) -> float:
        """Calculate the PSF intensity at the center of the detector.

        For a point source at z_val, this gives the intensity of light
        that reaches the detector center (x=y=0). This point is always
        within the pinhole if one is present.

        Args:
            z_val: Z-position of the point source in micrometers

        Returns:
            Light intensity at detector center
        """
        z_term = (z_val / self._norm_sigma_z) ** 2
        return np.exp(-0.5 * z_term)

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
    r_psf = 3 * sigma_xy

    # Convert to pixels and ensure odd number
    pixels_xy = int(np.ceil(r_psf / pixel_size))
    pixels_xy += (pixels_xy + 1) % 2

    if z_size is not None:
        pixels_z = int(np.ceil(2 * sigma_z / z_size))
        pixels_z += (pixels_z + 1) % 2
        return (pixels_z, pixels_xy, pixels_xy)

    return (pixels_xy, pixels_xy)
