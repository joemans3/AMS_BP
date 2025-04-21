# Global Parameters Help

This section defines the overarching timing and physical layout parameters of the simulation environment. These values affect **all molecules and the simulation timeline**.

---

## Fields

### `sample_plane_dim`
- **Type**: Array of Numbers `[width, height]`
- **Units**: Micrometers (`μm`)
- **Description**: Specifies the width and height of the simulation's imaging plane. This defines the observable region for molecule motion and fluorescence.

---

### `cycle_count`
- **Type**: Integer
- **Description**: The number of complete exposure–interval cycles to simulate. For example, a cycle count of 5 with 100ms exposure and 50ms interval simulates a total of 750ms.

---

### `exposure_time`
- **Type**: Integer
- **Units**: Milliseconds (`ms`)
- **Description**: Duration for which the camera exposes per frame. Affects motion blur and signal strength.

---

### `interval_time`
- **Type**: Integer
- **Units**: Milliseconds (`ms`)
- **Description**: Time between exposures. Combined with exposure time, it defines the total frame interval.

---

### `oversample_motion_time`
- **Type**: Integer
- **Units**: Milliseconds (`ms`)
- **Description**: The time resolution used for simulating motion in a sub-frame manner. Useful for accurate motion blur or faster dynamics within a single exposure.

---

## Notes

- Exposure and interval times together define the temporal resolution of the simulation.
- Oversample motion time enables fine-grained molecule dynamics between frames.

