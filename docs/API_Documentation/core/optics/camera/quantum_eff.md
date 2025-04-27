
# Documentation for `quantum_eff.py`

## Overview

The `quantum_eff.py` module provides a class for representing and working with the quantum efficiency (QE) curve of a detector. The quantum efficiency is a measure of how effectively a detector converts incident photons into detectable electrons.

## Classes

### `QuantumEfficiency`

```python
@dataclass
class QuantumEfficiency:
    """
    Represents the quantum efficiency curve of a detector.

    The wavelength values should be specified in nanometers (nm).
    """
```

#### Description

The `QuantumEfficiency` class represents the quantum efficiency curve of a detector. It allows for the specification of quantum efficiency values at specific wavelengths and provides functionality for interpolating the quantum efficiency at arbitrary wavelengths.

#### Attributes

- **wavelength_qe**: `Dict[float, float]`  
  A dictionary mapping wavelengths (in nanometers) to their corresponding quantum efficiency values.

#### Methods

- **`__post_init__`**

  ```python
  def __post_init__(self):
  ```

  Validates the quantum efficiency values and wavelengths, and initializes numpy arrays for faster interpolation.

- **`get_qe`**

  ```python
  def get_qe(self, wavelength: float) -> float:
  ```

  Gets the quantum efficiency for a specific wavelength using linear interpolation.

#### Parameters

- **wavelength**: `float`  
  The wavelength in nanometers.

#### Returns

- `float`  
  The interpolated quantum efficiency value between 0 and 1.