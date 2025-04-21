# Experiment Configuration Help

This section defines the structure of the **experiment** to be simulated. It includes information about the type of acquisition, the z-plane positions, and the laser activation strategy.

---

## General Fields

### `name`
- **Type**: String
- **Description**: The name of the experiment. This will be used to label output files and for documentation purposes.

### `description`
- **Type**: String
- **Description**: A short description of the experiment setup or purpose.

---

## Experiment Type

### `experiment_type`
- **Type**: String
- **Enum**: `["time-series", "z-stack"]`
- **Description**: Determines the acquisition mode:
  - `time-series`: Single z-plane acquired over time.
  - `z-stack`: Multiple z-planes acquired in a single timepoint sweep.

---

## Z Positions

### `z_position`
- **Type**:
  - Float for `time-series`
  - List of Floats for `z-stack`
- **Unit**: Micrometers (µm)
- **Description**:
  - In `time-series`, a single z-position is specified.
  - In `z-stack`, multiple z-positions are given, each corresponding to a slice in the stack.

---

## Laser Activation

### `laser_names_active`
- **Type**: List of Strings
- **Description**: Names of the lasers to be activated during this experiment. These must match the names defined in the laser configuration section.

### `laser_powers_active`
- **Type**: List of Floats
- **Unit**: Watts
- **Description**: The power levels for each laser, in the same order as `laser_names_active`.

### `laser_positions_active`
- **Type**: List of Lists `[x, y, z]`
- **Unit**: Micrometers (µm)
- **Description**: The position of each active laser beam in space. Each laser has a position defined as `[x, y, z]`.

---

## XY Offset

### `xyoffset`
- **Type**: List `[x, y]`
- **Unit**: Micrometers (µm)
- **Description**: The lateral offset applied to the sample plane during acquisition.

---

## Time Parameters (only for `z-stack`)

### `exposure_time`
- **Type**: Integer
- **Unit**: Milliseconds (ms)
- **Description**: Exposure duration per z-slice.

### `interval_time`
- **Type**: Integer
- **Unit**: Milliseconds (ms)
- **Description**: Time between capturing consecutive z-slices.

> ⚠️ These time fields are only used for `z-stack` experiments.  
> In `time-series` mode, time control should be handled globally under **Global Parameters**.

---

### Notes

- The number of `laser_names_active`, `laser_powers_active`, and `laser_positions_active` must match.
- Laser names must be defined in the Laser Configuration section.
- For `z-stack`, the total time per stack is `exposure_time × len(z_position) + interval_time × (len(z_position) - 1)`.

