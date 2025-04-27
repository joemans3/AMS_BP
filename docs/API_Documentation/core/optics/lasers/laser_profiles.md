# Module: `laser_profiles.py`

This module provides a comprehensive set of classes and utilities for modeling laser beam profiles, including Gaussian, widefield, HiLo, and confocal beam profiles. The module is designed to be flexible and extensible, allowing for the simulation of various laser beam characteristics and their interactions with optical systems.

## Table of Contents

- [Overview](#overview)
- [Classes](#classes)
  - [Units](#units)
  - [LaserParameters](#laserparameters)
  - [LaserProfile](#laserprofile)
  - [GaussianBeam](#gaussianbeam)
  - [WidefieldBeam](#widefieldbeam)
  - [HiLoBeam](#hiloBeam)
  - [ConfocalBeam](#confocalbeam)
- [Example Usage](#example-usage)

## Overview

The `laser_profiles.py` module is designed to model laser beam profiles with a focus on optical microscopy applications. It includes classes for defining laser parameters, abstract base classes for laser profiles, and concrete implementations for specific types of laser beams. The module leverages NumPy for numerical computations and supports time-dependent parameters, making it suitable for dynamic simulations.

## Classes

### `Units`

```python
class Units(Enum):
    """Enumeration of supported units for laser parameters."""

    MICRONS = "µm"
    WATTS = "W"
```

- **Description**: Enumeration of supported units for laser parameters.
- **Fields**:
  - `MICRONS`: Represents microns.
  - `WATTS`: Represents watts.

### `LaserParameters`

```python
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
```

- **Description**: A dataclass that encapsulates the parameters defining a laser beam.
- **Fields**:
  - `wavelength`: Wavelength of the laser in nanometers.
  - `power`: Power of the laser in watts, or a callable that returns power as a function of time.
  - `beam_width`: 1/e² beam width at the waist in microns.
  - `numerical_aperture`: Numerical aperture of the focusing lens (optional).
  - `position`: Position of the beam as a tuple of (x, y, z) coordinates in microns, or a callable that returns the position as a function of time.
  - `refractive_index`: Refractive index of the medium.

- **Methods**:
  - `__post_init__`: Validates and computes derived parameters after initialization.
  - `_validate_parameters`: Validates the input parameters.
  - `_compute_derived_parameters`: Computes derived beam parameters such as the Rayleigh range and wave number.
  - `diffraction_limited_width`: Calculates the diffraction-limited 1/e² beam width in microns.
  - `get_power`: Returns the power at a given time.
  - `get_position`: Returns the beam position at a given time.

### `LaserProfile`

```python
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
```

- **Description**: Abstract base class for laser beam profiles.
- **Methods**:
  - `calculate_intensity`: Abstract method to calculate the intensity distribution at given coordinates and time.
  - `get_beam_width`: Calculates the beam width at a distance z from the waist.
  - `get_radius_of_curvature`: Calculates the radius of curvature at a distance z.
  - `get_gouy_phase`: Calculates the Gouy phase at a distance z.
  - `get_intensity_map`: Generates a discretized intensity map for the given volume at time t.

### `GaussianBeam`

```python
class GaussianBeam(LaserProfile):
    """3D Gaussian laser beam profile with time dependence."""

    def calculate_intensity(
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
        pass
```

- **Description**: Implements a 3D Gaussian laser beam profile with time dependence.
- **Methods**:
  - `calculate_intensity`: Calculates the Gaussian beam intensity distribution.

### `WidefieldBeam`

```python
class WidefieldBeam(LaserProfile):
    """
    Widefield illumination profile where the laser beam is focused at the back focal plane
    of the objective to create uniform illumination across the field of view.
    """

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
        pass
```

- **Description**: Implements a widefield illumination profile.
- **Methods**:
  - `calculate_intensity`: Calculates the widefield illumination intensity distribution.

### `HiLoBeam`

```python
class HiLoBeam(LaserProfile):
    """
    Highly Inclined Laminated Optical (HiLo) illumination profile.

    HiLo microscopy uses an oblique, tilted illumination angle to reduce
    out-of-focus background while maintaining high contrast for thin specimens.
    """

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
        pass
```

- **Description**: Implements a HiLo illumination profile.
- **Methods**:
  - `calculate_intensity`: Calculates the HiLo illumination intensity distribution.

### `ConfocalBeam`

```python
class ConfocalBeam(LaserProfile):
    """
    Confocal microscopy beam profile with point scanning and pinhole characteristics.

    Implements key optical principles of confocal microscopy:
    - Point scanning illumination
    - Pinhole-based rejection of out-of-focus light
    - Depth-resolved imaging capabilities
    """

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
        pass
```

- **Description**: Implements a confocal microscopy beam profile.
- **Methods**:
  - `calculate_intensity`: Calculates the confocal illumination intensity distribution.

## Example Usage

```python
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
```

This example demonstrates how to create a `LaserParameters` object, instantiate a `GaussianBeam`, and generate an intensity map for a given volume.