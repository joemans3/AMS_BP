# Molecule Configuration Help

This section allows you to define the behavior and motion characteristics of each molecular species in the simulation. Each molecule type can follow different track models and include multiple diffusive or subdiffusive states.

---

## `num_molecules`
- **Type**: Array of Integers
- **Description**: The number of molecules for each defined type. The number of entries in this array defines the number of molecule types in the simulation.

---

## `track_type`
- **Type**: Array of Strings
- **Enum**: `["constant", "fbm"]`
- **Description**:
  - `constant`: Molecules exhibit fixed diffusion.
  - `fbm` (Fractional Brownian Motion): Molecules follow anomalous diffusion governed by a Hurst exponent.

---

## `diffusion_coefficient`
- **Type**: Array of Arrays of Floats
- **Units**: `μm²/s`
- **Description**: Specifies the diffusion coefficients for each molecule type. Each type can have one or more diffusion states.

---

## `diffusion_track_amount`
- **Type**: Array of Arrays of Floats
- **Description**: Initial distribution (probabilities) of molecules across each diffusion state. Must sum to 1.0 per molecule type.

---

## `hurst_exponent`
- **Type**: Array of Arrays of Floats
- **Range**: [0, 1]
- **Description**: Only used for `fbm` track types. Defines the Hurst exponents controlling subdiffusive behavior.

---

## `hurst_track_amount`
- **Type**: Array of Arrays of Floats
- **Description**: Initial distribution of molecules across each Hurst exponent state. Must sum to 1.0. Ignored for `constant` track types.

---

## `allow_transition_probability`
- **Type**: Array of Booleans
- **Description**: Whether molecules of each type can switch between diffusion or hurst states during the simulation.

---

## `transition_matrix_time_step`
- **Type**: Array of Integers
- **Units**: `ms`
- **Description**: The timestep interval at which transitions between states are evaluated. Only relevant when `allow_transition_probability` is `true`.

---

## `diffusion_transition_matrix`
- **Type**: Array of 2D Arrays (Per Molecule Type)
- **Description**: Transition probability matrices for diffusion coefficients. Each row must sum to 1. Only used if transitions are allowed.

---

## `hurst_transition_matrix`
- **Type**: Array of 2D Arrays (Per Molecule Type)
- **Description**: Transition probability matrices for hurst states. Only used for `fbm` track types and when transitions are allowed.

---

## Notes

- If `track_type` is `constant`, the hurst-related fields will be ignored and hidden in the UI.
- If a molecule type has only one state, the associated transition matrix is a 1×1 matrix with value 1.0.
- State probability fields (`diffusion_track_amount`, `hurst_track_amount`) should match the number of corresponding states.

