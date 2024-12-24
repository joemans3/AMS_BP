# Documentation for `detectors.py`

## Overview

The `detectors.py` module provides a framework for simulating different types of microscopy camera detectors, such as EMCCD and CMOS detectors. It includes functionality for calculating photon noise, converting electrons to digital counts (ADU), and capturing frames with various detector parameters.

## Functions

### `photon_noise`

```python
@overload
def photon_noise(photons: float) -> float: ...
@overload
def photon_noise(photons: np.ndarray) -> np.ndarray: ...
def photon_noise(photons: np.ndarray | float) -> np.ndarray | float:
    """
    Calculate photons with Poisson noise.
    """
    # non-negative floats
    return np.random.poisson(lam=photons)
```

#### Description

The `photon_noise` function calculates photons with Poisson noise, which is a common model for photon detection in microscopy. The function supports both scalar and array inputs.

#### Parameters

- **photons**: `float` or `np.ndarray`  
  The number of photons (or array of photon counts) to which Poisson noise will be applied.

#### Returns

- `float` or `np.ndarray`  
  The photon count(s) with Poisson noise added.

---

## Classes

### `Detector`

```python
class Detector(ABC):
    """Base class for microscopy camera detectors."""
```

#### Description

The `Detector` class is an abstract base class that defines the common interface and behavior for all types of microscopy camera detectors. It includes methods for initializing detector parameters, converting electrons to digital counts, and capturing frames.

#### Attributes

- **pixel_size**: `float`  
  Size of each pixel in microns.
- **dark_current**: `float`  
  Dark current in electrons/pixel/second.
- **readout_noise**: `float`  
  RMS readout noise in electrons.
- **pixel_count**: `Tuple[int, int]`  
  Tuple of (width, height) in pixels.
- **pixel_detector_size**: `float | int`  
  Size of the detector in microns.
- **magnification**: `float | int`  
  Magnification of the microscope.
- **bit_depth**: `int` (default: `16`)  
  Number of bits for analog-to-digital conversion.
- **sensitivity**: `float` (default: `1.0`)  
  Conversion gain in electrons/ADU.
- **base_adu**: `int` (default: `100`)  
  Base ADU to avoid negative values due to photon arrival rate.
- **binning_size**: `int` (default: `1`)  
  Binning size for combining pixels.

#### Methods

- **`__init__`**

  ```python
  def __init__(
      self,
      pixel_size: float,
      dark_current: float,
      readout_noise: float,
      pixel_count: Tuple[int, int],
      pixel_detector_size: float | int,
      magnification: float | int,
      bit_depth: int = 16,
      sensitivity: float = 1.0,
      base_adu: int = 100,
      binning_size: int = 1,
  ):
  ```

  Initializes the detector with the given parameters.

- **`base_frame`**

  ```python
  def base_frame(self, base_adu: int) -> np.ndarray:
  ```

  Creates a base frame with the specified base ADU value.

- **`electrons_to_counts`**

  ```python
  def electrons_to_counts(self, electrons: np.ndarray) -> np.ndarray:
  ```

  Converts electrons to digital counts (ADU).

- **`clipADU`**

  ```python
  def clipADU(self, counts: np.ndarray) -> np.ndarray:
  ```

  Clips the digital counts to the valid range based on the bit depth.

- **`capture_frame`** (Abstract Method)

  ```python
  @abstractmethod
  def capture_frame(self, photons: np.ndarray, exposure_time: float) -> np.ndarray:
  ```

  Captures a frame with the detector. This method must be implemented by subclasses.

---

### `EMCCDDetector`

```python
class EMCCDDetector(Detector):
    """Electron Multiplying CCD detector implementation."""
```

#### Description

The `EMCCDDetector` class extends the `Detector` class to implement an Electron Multiplying CCD (EMCCD) detector. It includes additional parameters and functionality specific to EMCCD detectors, such as electron multiplication gain and clock-induced charge.

#### Attributes

- **em_gain**: `float`  
  Electron multiplication gain.
- **clock_induced_charge**: `float`  
  Clock-induced charge in electrons/pixel/frame.

#### Methods

- **`__init__`**

  ```python
  def __init__(
      self,
      pixel_size: float,
      dark_current: float,
      readout_noise: float,
      pixel_count: Tuple[int, int],
      em_gain: float,
      clock_induced_charge: float,
      pixel_detector_size: float | int,
      magnification: float | int,
      sensitivity: float = 1.0,
      bit_depth: int = 16,
      base_adu: int = 0,
      binning_size: int = 1,
  ):
  ```

  Initializes the EMCCD detector with the given parameters.

- **`capture_frame`**

  ```python
  def capture_frame(self, photons: np.ndarray, exposure_time: float) -> np.ndarray:
  ```

  Captures a frame with the EMCCD detector, applying electron multiplication gain and clock-induced charge.

---

### `CMOSDetector`

```python
class CMOSDetector(Detector):
    """CMOS detector implementation."""
```

#### Description

The `CMOSDetector` class extends the `Detector` class to implement a CMOS detector. It includes functionality specific to CMOS detectors.

#### Methods

- **`__init__`**

  ```python
  def __init__(
      self,
      pixel_size: float,
      dark_current: float,
      readout_noise: float,
      pixel_count: Tuple[int, int],
      pixel_detector_size: float | int,
      magnification: float | int,
      sensitivity: float = 1.0,
      bit_depth: int = 16,
      base_adu: int = 0,
      binning_size: int = 1,
  ):
  ```

  Initializes the CMOS detector with the given parameters.

- **`capture_frame`**

  ```python
  def capture_frame(self, photons: np.ndarray, exposure_time: float) -> np.ndarray:
  ```

  Captures a frame with the CMOS detector.

---

### `create_binning_function`

```python
def create_binning_function(input_shape, binning_size, mode="sum"):
```

#### Description

Creates an optimized partial function for binning arrays of a specific shape.

#### Parameters

- **input_shape**: `tuple`  
  Shape of the input arrays that will be binned.
- **binning_size**: `int`  
  Size of the binning window (e.g., 2 for 2x2 binning).
- **mode**: `str` (default: `"sum"`)  
  Method for binning. Currently only supports `"sum"`.

#### Returns

- `function`  
  A specialized function that only takes an array as input and performs binning.

---
