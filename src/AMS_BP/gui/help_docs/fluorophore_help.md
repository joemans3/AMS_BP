# Fluorophore Configuration Help

This section defines the **fluorophores** used in the simulation, including their photophysical states and transition dynamics.

Each fluorophore can contain one or more states, and transitions between them. States can be fluorescent, dark (non-emissive), or bleached (permanently off).

---

## Top-Level Fields

### `num_of_fluorophores`
- **Type**: Integer
- **Description**: Total number of distinct fluorophore types.

### `fluorophore_names`
- **Type**: List of Strings
- **Description**: Names identifying each fluorophore. These must be unique.

---

## Per-Fluorophore Fields

Each fluorophore has its own section named after its `name`.

### `name`
- **Type**: String
- **Description**: Redundant copy of the fluorophore name.

### `initial_state`
- **Type**: String
- **Description**: The state in which each fluorophore begins the simulation. Must match one of the defined state names.

---

## States Section

Each fluorophore contains a `states` section, defining its possible photophysical states.

### `states`
- **Type**: Dictionary
- **Key**: State name (e.g., `"on"`, `"off"`, `"bleached"`)
- **Value**: Dictionary with the following fields:

#### `name`
- **Type**: String
- **Description**: Name of the state.

#### `state_type`
- **Type**: Enum
- **Options**: `"fluorescent"`, `"dark"`, `"bleached"`
- **Description**: Type of the photophysical state.

#### If `state_type == "fluorescent"`, the following fields are required:

##### `quantum_yield`
- **Type**: Float
- **Range**: [0, 1]
- **Description**: Quantum yield of emission.

##### `extinction_coefficient`
- **Type**: Float
- **Unit**: M⁻¹·cm⁻¹
- **Description**: Molar extinction coefficient for light absorption.

##### `fluorescent_lifetime`
- **Type**: Float
- **Unit**: Seconds
- **Description**: Mean lifetime of the fluorophore before emitting a photon.

##### `excitation_spectrum`
- **Type**: Dictionary
- **Fields**:
  - `wavelengths`: List of Floats (nm)
  - `intensities`: List of Floats (normalized to [0, 1])
- **Description**: Defines the excitation efficiency across the spectrum.

##### `emission_spectrum`
- **Type**: Dictionary
- **Fields**:
  - `wavelengths`: List of Floats (nm)
  - `intensities`: List of Floats (normalized to [0, 1])
- **Description**: Defines the emission intensity across the spectrum.

---

## Transitions Section

Each fluorophore can also define transitions between states.

### `transitions`
- **Type**: Dictionary
- **Key**: Transition label (e.g., `"on_to_off"`)
- **Value**: Dictionary with:

#### `from_state`, `to_state`
- **Type**: Strings
- **Description**: Names of the source and target states.

#### `photon_dependent`
- **Type**: Boolean
- **Description**: Whether this transition depends on light absorption.

#### If `photon_dependent == False`:

##### `base_rate`
- **Type**: Float
- **Unit**: s⁻¹
- **Description**: Constant rate of transition (Poisson process).

#### If `photon_dependent == True`:

##### `spectrum`
- **Type**: Dictionary
- **Fields**:
  - `wavelengths`: List of Floats (nm)
  - `intensities`: List of Floats (normalized)
  - `quantum_yield`: Float [0, 1]
  - `extinction_coefficient`: Float (M⁻¹·cm⁻¹)
- **Description**: Defines the absorption characteristics that trigger the transition.

---

## Notes

- State and transition names must be unique within a fluorophore.
- Spectral data should cover the same wavelength range for excitation and emission (typically 300–800 nm).
- Transitions allow for modeling of photoswitching, blinking, and photobleaching.

