# Documentation for `filters.py`

This module provides classes and functions for representing and creating optical filters, including bandpass, top-hat, and all-pass filters. The module uses `numpy` for numerical operations and `pydantic` for data validation and serialization.

---

## Type Aliases

### `CustomNDarray`

```python
CustomNDarray: TypeAlias = NDArray[np.float64]
```

A type alias for a `numpy` array of `float64` values.

---

## Classes

### `FilterSpectrum`

```python
class FilterSpectrum(BaseModel):
    """Represents the spectral characteristics of an optical filter"""
```

#### Fields

- **`wavelengths`**: `NDArray[np.float64]`  
  Wavelengths in nanometers.

- **`transmission`**: `NDArray[np.float64]`  
  Transmission values (0-1).

- **`cached_transmissions`**: `Dict[float, float]`  
  A dictionary to cache computed transmission values for future lookups. This field is excluded from serialization.

- **`name`**: `str`  
  The name of the filter.

#### Validators

- **`validate_transmission`**: Ensures that all transmission values are between 0 and 1. Raises a `ValueError` if invalid values are found.

- **`validate_wavelengths`**: Ensures that all wavelengths are positive and strictly increasing. Raises a `ValueError` if invalid values are found.

- **`validate_array_lengths`**: Ensures that the `wavelengths` and `transmission` arrays have the same length. Raises a `ValueError` if they do not match.

#### Methods

- **`find_transmission(wavelength: float) -> float`**  
  Finds the transmission value for a given wavelength using interpolation. Caches computed values for future lookups.

  - **Args**:
    - `wavelength`: The wavelength in nanometers.

  - **Returns**:
    - The interpolated transmission value between 0 and 1.

#### Configuration

- **`Config`**: Allows arbitrary types to be used in the model.

---

### `FilterSet`

```python
class FilterSet(BaseModel):
    """Represents a complete filter set (excitation, dichroic, emission)"""
```

#### Fields

- **`excitation`**: `FilterSpectrum`  
  The excitation filter.

- **`dichroic`**: `FilterSpectrum`  
  The dichroic filter.

- **`emission`**: `FilterSpectrum`  
  The emission filter.

- **`name`**: `str`  
  The name of the filter set (default: "Generic Filter Set").

---

## Functions

### `create_bandpass_filter`

```python
def create_bandpass_filter(
    center_wavelength: float,
    bandwidth: float,
    transmission_peak: float = 0.95,
    points: int = 1000,
    name: Optional[str] = None,
) -> FilterSpectrum:
    """
    Create a gaussian-shaped bandpass filter
    """
```

Creates a Gaussian-shaped bandpass filter.

- **Args**:
  - `center_wavelength`: Center wavelength in nanometers.
  - `bandwidth`: FWHM bandwidth in nanometers.
  - `transmission_peak`: Peak transmission (0-1) (default: 0.95).
  - `points`: Number of points in the spectrum (default: 1000).
  - `name`: Optional name for the filter.

- **Returns**:
  - A `FilterSpectrum` object representing the bandpass filter.

---

### `create_tophat_filter`

```python
def create_tophat_filter(
    center_wavelength: float,
    bandwidth: float,
    transmission_peak: float = 0.95,
    edge_steepness: float = 5.0,
    points: int = 1000,
    name: Optional[str] = None,
) -> FilterSpectrum:
    """
    Create a top-hat (rectangular) shaped filter with smooth edges
    """
```

Creates a top-hat (rectangular) shaped filter with smooth edges.

- **Args**:
  - `center_wavelength`: Center wavelength in nanometers.
  - `bandwidth`: Width of the passband in nanometers (FWHM).
  - `transmission_peak`: Peak transmission (0-1) (default: 0.95).
  - `edge_steepness`: Controls the sharpness of the edges (higher = sharper) (default: 5.0).
  - `points`: Number of points in the spectrum (default: 1000).
  - `name`: Optional name for the filter.

- **Returns**:
  - A `FilterSpectrum` object representing the top-hat filter.

---

### `create_allow_all_filter`

```python
def create_allow_all_filter(points: int, name: Optional[str] = None) -> FilterSpectrum:
    """
    Create a filter that allows all wavelengths
    """
```

Creates a filter that allows all wavelengths.

- **Args**:
  - `points`: Number of points in the spectrum.
  - `name`: Optional name for the filter.

- **Returns**:
  - A `FilterSpectrum` object representing the all-pass filter.

---

## Usage Example

```python
# Create a bandpass filter
bandpass_filter = create_bandpass_filter(center_wavelength=500, bandwidth=50)

# Create a top-hat filter
tophat_filter = create_tophat_filter(center_wavelength=600, bandwidth=40)

# Create an all-pass filter
all_pass_filter = create_allow_all_filter(points=500)

# Create a filter set
filter_set = FilterSet(
    excitation=bandpass_filter,
    dichroic=tophat_filter,
    emission=all_pass_filter,
    name="My Filter Set"
)

# Find transmission for a specific wavelength
transmission = filter_set.excitation.find_transmission(wavelength=520)
print(transmission)
```

---

## Notes

- The `FilterSpectrum` class uses `pydantic` for data validation, ensuring that wavelengths and transmission values are valid.
- The `cached_transmissions` field in `FilterSpectrum` improves performance by caching interpolated transmission values.
- The `create_*_filter` functions provide convenient ways to generate common filter shapes.