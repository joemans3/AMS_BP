from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np


class Units(Enum):
    """Enumeration of supported units for laser parameters."""

    MICRONS = "µm"
    WATTS = "W"


@dataclass
class LaserParameters:
    """
    Parameters defining a laser beam.

    All spatial parameters are in microns unless otherwise specified.
    """

    wavelength: float  # Wavelength in nanometers
    power: Union[float, Callable[[float], float]]  # Power in watts
    beam_width: float  # 1/e² beam width at waist in microns
    numerical_aperture: Optional[float] = None  # NA of focusing lens
    position: Union[Tuple[float, float, float], Callable[[float], np.ndarray]] = (
        0.0,
        0.0,
        0.0,
    )
    refractive_index: float = 1.0  # Refractive index of medium

    def __post_init__(self):
        """Validate parameters after initialization."""
        self._validate_parameters()
        self._compute_derived_parameters()
        self.max_power = self.power

    def _validate_parameters(self):
        """Validate input parameters."""
        if self.wavelength <= 0:
            raise ValueError("Wavelength must be positive")

        if self.beam_width <= 0:
            raise ValueError("Beam width must be positive")

        if isinstance(self.power, (int, float)) and self.power <= 0:
            raise ValueError("Power must be positive")

        if self.numerical_aperture is not None:
            if not 0 < self.numerical_aperture <= self.refractive_index:
                raise ValueError(f"NA must be between 0 and {self.refractive_index}")

            # Check diffraction limit
            diffraction_limit = self.diffraction_limited_width
            if self.beam_width < diffraction_limit:
                raise ValueError(
                    f"Beam width ({self.beam_width:.2f} µm) cannot be smaller than "
                    f"the diffraction limit ({diffraction_limit:.2f} µm) "
                    f"for NA={self.numerical_aperture}"
                )

        if self.refractive_index <= 0:
            raise ValueError("Refractive index must be positive")

    def _compute_derived_parameters(self):
        """Compute derived beam parameters."""
        self.wavelength_microns = self.wavelength / 1000
        self.k = (
            2 * np.pi * self.refractive_index / self.wavelength_microns
        )  # Wave number
        self.rayleigh_range = (
            self.refractive_index * np.pi * self.beam_width**2 / self.wavelength_microns
        )

        if isinstance(self.power, (int, float)):
            self.peak_intensity = 2 * self.power / (np.pi * self.beam_width**2)

    @property
    def diffraction_limited_width(self) -> Optional[float]:
        """Calculate diffraction-limited 1/e² beam width in microns."""
        if self.numerical_aperture is None:
            return None
        return self.wavelength / (np.pi * self.numerical_aperture * 1000)

    def get_power(self, t: float) -> float:
        """
        Get power at time t.

        Args:
            t: Time in seconds

        Returns:
            Power in watts
        """
        if callable(self.power):
            power = self.power(t)
            if power < 0:
                raise ValueError("Laser Power Cannot be Negative")
            return power
        return self.power

    def get_position(self, t: float) -> Tuple[float, float, float]:
        """
        Get beam position at time t.

        Args:
            t: Time in seconds

        Returns:
            Tuple of (x, y, z) coordinates in microns
        """
        if callable(self.position):
            return self.position(t)
        return self.position


class LaserProfile(ABC):
    """Abstract base class for laser beam profiles."""

    def __init__(self, params: LaserParameters):
        self.params = params

    @abstractmethod
    def calculate_intensity(
        self,
        x: np.ndarray | float,
        y: np.ndarray | float,
        z: np.ndarray | float,
        t: float,
    ) -> np.ndarray:
        """
        Calculate the intensity distribution at given coordinates and time.

        Args:
            x: X coordinates in microns (3D array)
            y: Y coordinates in microns (3D array)
            z: Z coordinates in microns (3D array)
            t: Time in milliseconds

        Returns:
            3D array of intensities in W/m²
        """
        pass

    def get_beam_width(self, z: float) -> float:
        """
        Calculate beam width at distance z from waist.

        Args:
            z: Distance from beam waist in microns

        Returns:
            Beam width in microns
        """
        return self.params.beam_width * np.sqrt(
            1 + (z / self.params.rayleigh_range) ** 2
        )

    def get_radius_of_curvature(self, z: float) -> float:
        """
        Calculate radius of curvature at distance z.

        Args:
            z: Distance from beam waist in microns

        Returns:
            Radius of curvature in microns
        """
        if z == 0:
            return float("inf")
        return z * (1 + (self.params.rayleigh_range / z) ** 2)

    def get_gouy_phase(self, z: float) -> float:
        """
        Calculate Gouy phase at distance z.

        Args:
            z: Distance from beam waist in microns

        Returns:
            Gouy phase in radians
        """
        return np.arctan(z / self.params.rayleigh_range)

    def get_intensity_map(
        self,
        volume_size: Tuple[float, float, float],
        voxel_size: float,
        t: float,
        center: Tuple[float, float, float] = (0, 0, 0),
    ) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Generate a discretized intensity map for the given volume at time t.

        Args:
            volume_size: (x_size, y_size, z_size) in microns
            voxel_size: Size of each voxel in microns
            t: Time in seconds
            center: Center coordinates of the volume

        Returns:
            Dictionary containing:
                'intensity': 3D array of intensities
                'coordinates': Dictionary of coordinate arrays
        """
        # Create 3D coordinate grid
        nx = int(volume_size[0] / voxel_size)
        ny = int(volume_size[1] / voxel_size)
        nz = int(volume_size[2] / voxel_size)

        x = np.linspace(
            -volume_size[0] / 2 + center[0], volume_size[0] / 2 + center[0], nx
        )
        y = np.linspace(
            -volume_size[1] / 2 + center[1], volume_size[1] / 2 + center[1], ny
        )
        z = np.linspace(
            -volume_size[2] / 2 + center[2], volume_size[2] / 2 + center[2], nz
        )

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        intensity = self.calculate_intensity(X, Y, Z, t)

        return {"intensity": intensity, "coordinates": {"x": x, "y": y, "z": z}}


class GaussianBeam(LaserProfile):
    """3D Gaussian laser beam profile with time dependence."""

    def calculate_intensity(
        self,
        x: np.ndarray | float,
        y: np.ndarray | float,
        z: np.ndarray | float,
        t: float,
    ) -> np.ndarray:
        # Get time-dependent parameters
        power = self.params.get_power(t)
        pos = self.params.get_position(t)

        # Shift coordinates based on current beam position
        x_shifted = x - pos[0]
        y_shifted = y - pos[1]
        z_shifted = z - pos[2]

        # Calculate beam parameters at z
        w_z = self.get_beam_width(z_shifted)
        # R_z = self.get_radius_of_curvature(z_shifted)

        # Peak intensity at beam waist
        I0 = 2 * power / (np.pi * (self.params.beam_width / 2) ** 2)

        # Radial distance squared
        r_squared = x_shifted**2 + y_shifted**2

        # # Calculate z-dependent intensity with phase terms
        # phase_terms = (
        #     self.params.k * z_shifted
        #     - self.get_gouy_phase(z_shifted)
        #     + self.params.k * r_squared / (2 * R_z)
        # )

        # More accurate Gaussian intensity distribution
        return (
            I0 * (self.params.beam_width / w_z) ** 2 * np.exp(-2 * r_squared / w_z**2)
            # * np.cos(phase_terms)
        )

    def calculate_intensity_(
        self,
        x: np.ndarray | float,
        y: np.ndarray | float,
        z: np.ndarray | float,
        t: float,
    ) -> np.ndarray:
        """
        Calculate the Gaussian beam intensity distribution.

        Args:
            x: X coordinates in microns (3D array)
            y: Y coordinates in microns (3D array)
            z: Z coordinates in microns (3D array)
            t: Time in seconds

        Returns:
            3D array of intensities in W/um²

        Note:
            Uses the paraxial approximation valid for low NA
        """
        # Get time-dependent parameters
        power = self.params.get_power(t)
        pos = self.params.get_position(t)

        # Shift coordinates based on current beam position
        x_shifted = x - pos[0]
        y_shifted = y - pos[1]
        z_shifted = z - pos[2]

        # Calculate radial distance squared
        r_squared = x_shifted**2 + y_shifted**2

        # Calculate z-dependent beam width
        w_z = self.get_beam_width(z_shifted)

        # Calculate peak intensity (z-dependent)
        I0 = 2 * power / (np.pi * (self.params.beam_width / 2.0) ** 2)
        I0_z = I0 * (self.params.beam_width / w_z) ** 2

        # Calculate phase terms if needed
        # phase = self.params.k * z_shifted - self.get_gouy_phase(z_shifted)
        # R_z = self.get_radius_of_curvature(z_shifted)
        # phase += self.params.k * r_squared / (2 * R_z)

        # Calculate Gaussian intensity with z-dependence
        return I0_z * np.exp(-2.0 * r_squared / (w_z**2))


class WidefieldBeam(LaserProfile):
    """
    Widefield illumination profile where the laser beam is focused at the back focal plane
    of the objective to create uniform illumination across the field of view.

    The intensity distribution is approximately uniform in the XY plane with some
    falloff at the edges due to the finite numerical aperture. The axial (Z) intensity
    distribution is modeled based on the microscope's optical depth of field.
    """

    def __init__(self, params: LaserParameters):
        """
        Initialize widefield beam profile.
        """
        super().__init__(params)

        if self.params.numerical_aperture is None:
            raise ValueError(
                "Numerical aperture must be specified for widefield illumination"
            )

        # Calculate maximum radius of illumination at focal plane
        self.max_radius = (
            self.params.beam_width
            * self.params.numerical_aperture
            / self.params.refractive_index
        )

        # Calculate optical depth of field
        wavelength_microns = self.params.wavelength_microns  # Convert nm to µm
        na = self.params.numerical_aperture
        n = self.params.refractive_index

        # Wave optics DoF
        self.dof = (wavelength_microns * n) / (na * na)

        # print(f"Optical DoF: {self.dof:.2f} µm")

    def _calculate_dof_profile(self, z: np.ndarray | float) -> np.ndarray:
        """
        Calculate the intensity profile based on optical depth of field.

        Uses a smooth transition function to model intensity falloff
        outside the depth of field.

        Args:
            z: Distance from focal plane in microns

        Returns:
            Intensity scaling factor between 0 and 1
        """
        # Use error function for smooth transition at DoF boundaries
        # Scale factor determines how sharp the transition is
        scale_factor = 2.0  # Adjust this to change transition sharpness

        # Normalize z by DoF and create smooth falloff
        normalized_z = scale_factor * (np.abs(z) - self.dof / 2) / self.dof

        # Use sigmoid function for smooth transition
        return 1 / (1 + np.exp(normalized_z))

    def calculate_intensity(
        self,
        x: np.ndarray | float,
        y: np.ndarray | float,
        z: np.ndarray | float,
        t: float,
    ) -> np.ndarray:
        """
        Calculate the widefield illumination intensity distribution.

        Args:
            x: X coordinates in microns (3D array)
            y: Y coordinates in microns (3D array)
            z: Z coordinates in microns (3D array)
            t: Time in seconds

        Returns:
            3D array of intensities in W/µm²
        """
        # Get time-dependent parameters
        power = self.params.get_power(t)
        pos = self.params.get_position(t)

        # Shift coordinates based on current beam position
        x_shifted = x - pos[0]
        y_shifted = y - pos[1]
        z_shifted = z - pos[2]

        # Calculate radial distance from optical axis
        r = np.sqrt(x_shifted**2 + y_shifted**2)

        # Calculate base intensity (power divided by illumination area)
        base_intensity = power / (np.pi * self.max_radius**2)

        # Apply radial intensity profile with smooth falloff at edges
        edge_width = self.max_radius * 0.1  # 10% of max radius
        radial_profile = 0.5 * (1 - np.tanh((r - self.max_radius) / edge_width))
        # Apply DoF-based axial intensity profile
        axial_profile = self._calculate_dof_profile(z_shifted)

        # Combine profiles
        return base_intensity * radial_profile * axial_profile


# Example usage
if __name__ == "__main__":
    # Create parameters for a typical microscope objective
    params = LaserParameters(
        wavelength=488,  # 488 nm
        power=0.001,  # 1 mW
        beam_width=0.25,  # 250 nm
        numerical_aperture=1.4,
        refractive_index=1.518,  # Oil immersion
    )

    # Create beam object
    beam = GaussianBeam(params)

    # Get intensity map
    result = beam.get_intensity_map(
        volume_size=(5, 5, 10),  # 5x5x10 microns
        voxel_size=0.1,  # 100 nm voxels
        t=0,  # t=0 seconds
    )

    # print(f"Beam waist: {params.beam_width:.3f} µm")
    # print(f"Rayleigh range: {params.rayleigh_range:.3f} µm")
    # print(f"Diffraction limit: {params.diffraction_limited_width:.3f} µm")


class HiLoBeam(LaserProfile):
    """
    Highly Inclined Laminated Optical (HiLo) illumination profile.

    HiLo microscopy uses an oblique, tilted illumination angle to reduce
    out-of-focus background while maintaining high contrast for thin specimens.
    """

    def __init__(self, params: LaserParameters, inclination_angle: float):
        """
        Initialize HiLo beam profile.

        Args:
            params: LaserParameters for the beam
            inclination_angle: Angle of illumination from optical axis (in degrees)
        """
        super().__init__(params)

        # Validate numerical aperture
        if params.numerical_aperture is None:
            raise ValueError(
                "Numerical aperture must be specified for HiLo illumination"
            )

        # Convert angle to radians
        self.inclination_angle = np.deg2rad(inclination_angle)

        # Calculate effective illumination parameters
        self.effective_na = min(
            params.numerical_aperture,
            params.refractive_index * np.sin(self.inclination_angle),
        )

        # Calculate illumination characteristics
        wavelength_microns = params.wavelength / 1000.0
        self.lateral_resolution = 0.61 * wavelength_microns / self.effective_na
        self.axial_resolution = wavelength_microns / (2 * self.effective_na**2)

        # print(
        #     f"HiLo Illumination - Inclination Angle: {np.rad2deg(self.inclination_angle):.2f}°"
        # )
        # print(f"Effective NA: {self.effective_na:.3f}")
        # print(f"Lateral Resolution: {self.lateral_resolution:.3f} µm")
        # print(f"Axial Resolution: {self.axial_resolution:.3f} µm")

    def calculate_intensity(
        self,
        x: np.ndarray | float,
        y: np.ndarray | float,
        z: np.ndarray | float,
        t: float,
    ) -> np.ndarray:
        """
        Calculate the HiLo illumination intensity distribution.

        Args:
            x: X coordinates in microns (3D array)
            y: Y coordinates in microns (3D array)
            z: Z coordinates in microns (3D array)
            t: Time in seconds

        Returns:
            3D array of intensities in W/µm²
        """
        # Get time-dependent parameters
        power = self.params.get_power(t)
        pos = self.params.get_position(t)

        # Shift coordinates based on current beam position
        x_shifted = x - pos[0]
        y_shifted = y - pos[1]
        z_shifted = z - pos[2]

        # Calculate radial distance from optical axis
        r = np.sqrt(x_shifted**2 + y_shifted**2)

        # Base beam parameters
        w0 = self.params.beam_width  # Beam waist
        zR = self.params.rayleigh_range  # Rayleigh range

        # Inclined illumination projection
        # Modify z to account for tilted illumination
        z_inclined = z_shifted * np.cos(self.inclination_angle)

        # Calculate beam width at inclined z
        w_z = w0 * np.sqrt(1 + (z_inclined / zR) ** 2)

        # Peak intensity calculation
        I0 = 2 * power / (np.pi * w0**2)

        # Gaussian beam intensity with inclination
        intensity = (
            I0
            * (w0 / w_z) ** 2  # Beam width scaling
            * np.exp(-2 * r**2 / w_z**2)  # Gaussian radial profile
        )

        # Lamination effect: attenuate out-of-focus regions
        lamination_factor = np.exp(-np.abs(z_shifted) / (2 * self.axial_resolution))

        return intensity * lamination_factor


class ConfocalBeam(LaserProfile):
    """
    Confocal microscopy beam profile with point scanning and pinhole characteristics.

    Implements key optical principles of confocal microscopy:
    - Point scanning illumination
    - Pinhole-based rejection of out-of-focus light
    - Depth-resolved imaging capabilities
    """

    def __init__(
        self,
        params: LaserParameters,
        pinhole_diameter: float,  # Pinhole diameter in microns
        scanning_mode: str = "point",  # 'point' or 'line'
        line_orientation: str = "horizontal",  # 'horizontal' or 'vertical'
    ):
        """
        Initialize Confocal beam profile.

        Args:
            params: LaserParameters for the beam
            pinhole_diameter: Diameter of the detection pinhole in microns
            scanning_mode: Scanning method ('point' or 'line')
            line_orientation: Orientation for line scanning
        """
        super().__init__(params)

        # Validate numerical aperture
        if params.numerical_aperture is None:
            raise ValueError(
                "Numerical aperture must be specified for confocal microscopy"
            )

        # Pinhole and optical characteristics
        self.pinhole_diameter = pinhole_diameter
        self.scanning_mode = scanning_mode
        self.line_orientation = line_orientation

        # Calculate optical parameters
        wavelength_microns = params.wavelength / 1000.0
        na = params.numerical_aperture

        # Theoretical resolution calculations
        self.lateral_resolution = 0.61 * wavelength_microns / na
        self.axial_resolution = 0.5 * wavelength_microns / (na**2)

        # Pinhole transmission calculation
        # Airy disk radius calculation
        self.airy_radius = 1.22 * wavelength_microns / (2 * na)

        # Transmission through pinhole
        def pinhole_transmission(z):
            """
            Calculate pinhole transmission as a function of z-position.
            Uses an error function to model smooth transition.
            """
            # Normalized z-position relative to focal plane
            z_norm = z / self.axial_resolution

            # Smooth transition function
            return 0.5 * (1 + np.tanh(-z_norm))

        self.pinhole_transmission = pinhole_transmission

        # print("Confocal Microscopy Configuration:")
        # print(f"  Scanning Mode: {scanning_mode}")
        # print(f"  Pinhole Diameter: {pinhole_diameter:.2f} µm")
        # print(f"  Lateral Resolution: {self.lateral_resolution:.3f} µm")
        # print(f"  Axial Resolution: {self.axial_resolution:.3f} µm")
        # print(f"  Airy Disk Radius: {self.airy_radius:.3f} µm")

    def calculate_intensity(
        self,
        x: np.ndarray | float,
        y: np.ndarray | float,
        z: np.ndarray | float,
        t: float,
    ) -> np.ndarray:
        """
        Calculate the confocal illumination intensity distribution.

        Args:
            x: X coordinates in microns (3D array)
            y: Y coordinates in microns (3D array)
            z: Z coordinates in microns (3D array)
            t: Time in seconds

        Returns:
            3D array of intensities in W/µm²
        """
        # Get time-dependent parameters
        power = self.params.get_power(t)
        pos = self.params.get_position(t)

        # Shift coordinates based on current beam position
        x_shifted = x - pos[0]
        y_shifted = y - pos[1]
        z_shifted = z - pos[2]

        # Base beam parameters
        w0 = self.params.beam_width  # Beam waist
        zR = self.params.rayleigh_range  # Rayleigh range

        # Calculate beam width at z
        w_z = w0 * np.sqrt(1 + (z_shifted / zR) ** 2)

        # Peak intensity calculation
        I0 = 2 * power / (np.pi * w0**2)

        # Scanning mode intensity modification
        if self.scanning_mode == "point":
            # Point scanning: standard Gaussian beam
            radial_intensity = (
                I0
                * (w0 / w_z) ** 2
                * np.exp(-2 * (x_shifted**2 + y_shifted**2) / w_z**2)
            )
        elif self.scanning_mode == "line":
            # Line scanning: different intensity distribution
            if self.line_orientation == "horizontal":
                line_intensity = (
                    I0 * (w0 / w_z) ** 2 * np.exp(-2 * y_shifted**2 / w_z**2)
                )
                radial_intensity = line_intensity
            else:  # vertical line scanning
                line_intensity = (
                    I0 * (w0 / w_z) ** 2 * np.exp(-2 * x_shifted**2 / w_z**2)
                )
                radial_intensity = line_intensity
        else:
            raise ValueError(f"Unknown scanning mode: {self.scanning_mode}")

        # Pinhole transmission effect
        pinhole_effect = self.pinhole_transmission(z_shifted)

        # Final intensity calculation
        return radial_intensity * pinhole_effect
