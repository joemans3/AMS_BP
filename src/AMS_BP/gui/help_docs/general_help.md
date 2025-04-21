# General Configuration Help

This section defines the global settings that apply to **all units and measurements** throughout the simulation.

These values are typically fixed and should not be changed unless the simulation framework explicitly supports it.

---

## Fields

### `version`
- **Type**: String
- **Format**: `X.Y` (e.g., `0.1`, `1.0`)
- **Description**: Version of the configuration schema being used. Ensures compatibility between your configuration file and the simulation engine.

---

### `length_unit`
- **Type**: String
- **Allowed Values**: `"um"` (micrometers)
- **Description**: The base unit of length for the simulation. All spatial dimensions (e.g., cell size, laser beam width, positions) are expressed in micrometers.

---

### `time_unit`
- **Type**: String
- **Allowed Values**: `"ms"` (milliseconds)
- **Description**: The unit of time for all time-related parameters like exposure, interval, and simulation durations.

---

### `diffusion_unit`
- **Type**: String
- **Allowed Values**: `"um^2/s"`
- **Description**: The unit for diffusion coefficients. All diffusion values are given in square micrometers per second.

---

## Notes

- These units are **not editable** in most configurations and are standardized to simplify parameter consistency.
- All widgets and backend systems interpret numeric values based on these settings.

