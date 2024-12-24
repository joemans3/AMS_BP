## `flurophore_schema.py`

This module defines the schema and classes for modeling fluorophores and their properties, including wavelength-dependent spectral data, states, and state transitions.

### Constants

- **`CS_COEFF`**: A constant derived from the natural logarithm of 10, multiplied by $$10^3$$ and divided by Avogadro's number ($$N_A$$).

### Functions

#### `normalize_values(values: List[float]) -> List[float]`

Normalizes a list of values such that they sum to 1.

- **Parameters**:
  - `values`: A list of floats to be normalized.
- **Returns**: A list of normalized floats.

### Classes

#### `WavelengthDependentBase`

Base class for wavelength-dependent data.

- **Attributes**:
  - `wavelengths`: List of wavelengths (nm).
  - `values`: List of corresponding values.
  - `cache_values`: Dictionary caching values for specific wavelengths.

- **Methods**:
  - `model_post_init(__context: Any) -> None`: Initializes the cache with values at each wavelength.
  - `validate_lengths(cls, v, info)`: Validates that `wavelengths` and `values` have the same length.
  - `get_value(wavelength: float) -> float`: Retrieves or interpolates the value at a specific wavelength.

#### `SpectralData`

Extends `WavelengthDependentBase` for wavelength-dependent spectral data.

- **Attributes**:
  - `values`: List of intensities (0-1), aliased as `intensities`.

- **Methods**:
  - `get_intensity(wavelength: float) -> float`: Retrieves the intensity at a specific wavelength.

#### `WavelengthDependentProperty`

Extends `WavelengthDependentBase` for wavelength-dependent properties.

#### `StateType`

Enum class representing types of fluorophore states.

- **Values**:
  - `FLUORESCENT`: Fluorescent state.
  - `DARK`: Dark state.
  - `BLEACHED`: Bleached state.

#### `State`

Represents a single state of a fluorophore.

- **Attributes**:
  - `name`: Name of the state.
  - `state_type`: Type of the state (from `StateType`).
  - `excitation_spectrum`: Optional `SpectralData` for excitation.
  - `emission_spectrum`: Optional `SpectralData` for emission.
  - `quantum_yield_lambda_val`: Optional quantum yield at a specific wavelength (0-1).
  - `quantum_yield`: Optional wavelength-dependent quantum yield.
  - `extinction_coefficient_lambda_val`: Optional extinction coefficient at a specific wavelength (M⁻¹cm⁻¹).
  - `extinction_coefficient`: Optional wavelength-dependent extinction coefficient.
  - `ex_max`: Wavelength (nm) at maximum excitation.
  - `em_max`: Wavelength (nm) at maximum emission.
  - `molar_cross_section`: Optional wavelength-dependent molar cross-section.
  - `fluorescent_lifetime`: Optional fluorescent lifetime (1/s).
  - `fluorescent_lifetime_inverse`: Inverse of fluorescent lifetime.

- **Methods**:
  - `model_post_init(__context)`: Initializes derived properties like `ex_max`, `em_max`, and expands values to spectra.
  - `_val_ratio_expand(val: float, wavelength: float, base_spectrum: WavelengthDependentBase) -> WavelengthDependentProperty`: Expands a value to a spectrum based on a ratio.

#### `StateTransition`

Represents a transition between states.

- **Attributes**:
  - `from_state`: Name of the starting state.
  - `to_state`: Name of the target state.
  - `spectrum`: Optional `SpectralData` for the activation spectrum.
  - `extinction_coefficient_lambda_val`: Optional extinction coefficient at a specific wavelength (M⁻¹cm⁻¹).
  - `extinction_coefficient`: Optional wavelength-dependent extinction coefficient.
  - `cross_section`: Optional wavelength-dependent cross-section.
  - `base_rate`: Optional base transition rate (1/s).
  - `quantum_yield`: Optional quantum yield for state change.

- **Methods**:
  - `model_post_init(__context)`: Initializes derived properties like `extinction_coefficient` and `cross_section`.
  - `rate() -> Callable`: Returns a function to calculate the activation rate at a specific wavelength and intensity.
  - `_val_ratio_expand(val: float, wavelength: float, base_spectrum: WavelengthDependentBase) -> WavelengthDependentProperty`: Expands a value to a spectrum based on a ratio.

#### `Fluorophore`

Represents a complete fluorophore model.

- **Attributes**:
  - `name`: Name of the fluorophore.
  - `states`: Dictionary of states, keyed by state name.
  - `transitions`: Dictionary of state transitions.
  - `initial_state`: The initial state of the fluorophore.

- **Methods**:
  - `validate_states(cls, v)`: Validates that at least one state is fluorescent and state names are unique.
  - `validate_transitions(cls, v, info)`: Validates that transitions reference valid states.
  - `_find_transitions(statename: str) -> List[StateTransition]`: Finds all transitions originating from a specific state.

---
