# `psf_engine.py` Documentation

This module provides functionality for generating Point Spread Functions (PSFs) for microscopy using Gaussian approximations. It includes classes and functions to define PSF parameters, initialize calculations, and generate 2D and 3D PSFs.

## Table of Contents

- [Classes](#classes)
  - [PSFParameters](#psfparameters)
  - [PSFEngine](#psfengine)
- [Functions](#functions)
  - [_calculate_sigma_xy](#_calculate_sigma_xy)
  - [_calculate_sigma_z](#_calculate_sigma_z)
  - [_generate_grid](#_generate_grid)
  - [calculate_psf_size](#calculate_psf_size)
  - [_generate_pinhole_mask](#_generate_pinhole_mask)

---

## Classes

### `PSFParameters`
#### Description
This is a frozen dataclass (enum-ish?) with the following parameters / cached properties
```python
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
```
### `PSFEngine`
#### Methods

- **`__init__`**: Initializes the PSF engine with the given parameters.
- **`_initialize_calculations`**: Initializes commonly used calculations for PSF generation.
- **`psf_z`**: Generates a 2D Gaussian approximation of the PSF at a specific z-position.
- **`psf_z_xy0`**: Generates a Gaussian approximation of the PSF at a specific z-position with x=y=0.
- **`_3d_normalization_A`**: Computes the normalization factor for a 3D Gaussian PSF.
- **`_2d_normalization_A`**: Computes the normalization factor for a 2D Gaussian PSF.
- **`normalize_psf`**: Normalizes the PSF using different schemes.
- **`_generate_pinhole_mask`**: Generate a binary mask representing the pinhole's spatial filtering.

---

## Functions

### `_calculate_sigma_xy`



#### Description

Calculates the lateral sigma value for the PSF.

#### Parameters

- **`wavelength_um`**: Wavelength in micrometers.
- **`numerical_aperture`**: Numerical aperture of the optical system.

#### Returns

- **`float`**: Lateral sigma value.

---

### `_calculate_sigma_z`

#### Description

Calculates the axial sigma value for the PSF.

#### Parameters

- **`wavelength_um`**: Wavelength in micrometers.
- **`numerical_aperture`**: Numerical aperture of the optical system.
- **`refractive_index`**: Refractive index of the medium.

#### Returns

- **`float`**: Axial sigma value.

---

### `_generate_grid`


#### Description

Generates coordinate grids for PSF calculation.

#### Parameters

- **`size`**: Tuple of (height, width) for the grid.
- **`pixel_size`**: Size of pixels in micrometers.

#### Returns

- **`Tuple[NDArray[np.float64], NDArray[np.float64]]`**: Tuple of x and y coordinate arrays.

---

### `calculate_psf_size`


#### Description

Calculates the appropriate PSF size based on physical parameters.

#### Parameters

- **`sigma_xy`**: Lateral sigma value.
- **`pixel_size`**: Size of pixels in micrometers.
- **`sigma_z`**: Axial sigma value.
- **`z_size`**: Optional number of z-planes for 3D PSF.

#### Returns

- **`Tuple[int, ...]`**: Tuple of dimensions (z, y, x) or (y, x) for the PSF calculation.

---

### `_generate_pinhole_mask`


#### Description

Generate a binary mask representing the pinhole's spatial filtering.

The pinhole blocks emission light based on position in the image plane,
affecting what portion of the diffracted light reaches the detector.

#### Parameters
#### Returns

- **`NDArray[np.float64]`**: Same dimensions as the psf generated but with binary values (0,1) indicating the transmittance of the psf due to the pinhole. Note, if the pinhole size is smaller than 1*airy disk diameter, then diffraction due to the pinhole is NOT non negligiable and diffraction effects will be introduced. A ValueError is thrown if this is the case.   