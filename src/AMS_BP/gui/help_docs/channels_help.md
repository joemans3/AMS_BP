# Channels Configuration Help

This section allows you to define the optical channels used in the simulation, including filters for excitation and emission as well as split efficiencies.

## Fields

### `num_of_channels`
- **Type**: Integer
- **Description**: Number of optical channels. Each channel is assigned its own tab.

### `channel_names`
- **Type**: Array of Strings
- **Description**: Names of each channel (e.g., `"Channel 1"`, `"GFP"`, `"Cy5"`). Must be unique.

### `split_efficiency`
- **Type**: Array of Numbers
- **Range**: `0.0` – `1.0`
- **Description**: Efficiency of the beam splitter for each channel. Determines what fraction of signal is passed to the channel.

---

## Filter Set Configuration

Each channel defines a **filter set** with excitation and emission filters.

### `filter_set_name`
- **Type**: String
- **Description**: Descriptive name for the filter set.

### `filter_set_description`
- **Type**: String
- **Description**: Optional details about the purpose of the filter set.

---

## Excitation Filter

### `name`
- **Type**: String
- **Description**: Name of the excitation filter.

### `type`
- **Type**: String
- **Options**: `"bandpass"`, `"allow_all"`
- **Description**:
  - `"bandpass"` filters light around a central wavelength.
  - `"allow_all"` passes all wavelengths without filtering.

### `center_wavelength` *(only if type is `"bandpass"`)*  
- **Type**: Integer  
- **Unit**: Nanometers (nm)  
- **Description**: Central wavelength of the bandpass filter.

### `bandwidth` *(only if type is `"bandpass"`)*  
- **Type**: Integer  
- **Unit**: Nanometers (nm)  
- **Description**: Width of the bandpass range.

### `transmission_peak` *(only if type is `"bandpass"`)*  
- **Type**: Float  
- **Range**: `0.0` – `1.0`  
- **Description**: Maximum transmission efficiency.

### `points`
- **Type**: Integer  
- **Description**: Number of discrete sampling points in the filter curve.

---

## Emission Filter

Identical in structure to the Excitation filter.

### Notes:
- `"allow_all"` disables wavelength-specific fields.
- Bandpass filters should be configured carefully to avoid overlap between channels.
- Points determine how finely the spectrum is resolved during simulation.

