# Module: `experiments.py`

This module provides functionality for defining and running experiments in a simulated microscopy environment. It includes dataclasses for configuring experiments and functions to execute these experiments using a virtual microscope.

## Imports

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..metadata.metadata import MetaData
from ..sim_microscopy import VirtualMicroscope
```

## Dataclasses

### `BaseExpConfig`

Base configuration class for experiments.

**Attributes:**
- `name` (`str`): Name of the experiment.
- `description` (`str`): Description of the experiment.

```python
@dataclass
class BaseExpConfig:
    name: str
    description: str
```

### `TimeSeriesExpConfig`

Configuration class for time series experiments.

**Attributes:**
- `z_position` (`float`): Z-position for the experiment.
- `laser_names_active` (`List[str]`): List of active laser names.
- `laser_powers_active` (`List[float]`): List of active laser powers.
- `laser_positions_active` (`List`): List of active laser positions.
- `xyoffset` (`Tuple[float, float]`): XY offset for the experiment.
- `exposure_time` (`Optional[int]`): Exposure time (optional).
- `interval_time` (`Optional[int]`): Interval time (optional).
- `duration_time` (`Optional[int]`): Duration time (optional).

**Methods:**
- `__post_init__()`: Validates the configuration and sets up laser powers and positions.

```python
@dataclass
class TimeSeriesExpConfig(BaseExpConfig):
    z_position: float
    laser_names_active: List[str]
    laser_powers_active: List[float]
    laser_positions_active: List
    xyoffset: Tuple[float, float]

    exposure_time: Optional[int] = None
    interval_time: Optional[int] = None
    duration_time: Optional[int] = None

    def __post_init__(self):
        len_ln = len(self.laser_names_active)
        len_lpow = len(self.laser_powers_active)
        len_lpos = len(self.laser_positions_active)
        if len_ln != len_lpos or len_ln != len_lpow:
            raise ValueError(
                f"Length mismatch among lists: "
                f"laser_names_active({len_ln}), "
                f"laser_powers_active({len_lpow}), "
                f"laser_positions_active({len_lpos})"
            )
        if self.exposure_time or self.interval_time or self.duration_time:
            raise ValueError(
                "Please do not define exposure_time, interval_time, or duration_time in a time series experiment component. Use the GlobalParameters to set this."
            )
        self.laser_powers = {
            self.laser_names_active[i]: self.laser_powers_active[i]
            for i in range(len(self.laser_names_active))
        }
        self.laser_positions = {
            self.laser_names_active[i]: self.laser_positions_active[i]
            for i in range(len(self.laser_names_active))
        }
```

### `zStackExpConfig`

Configuration class for z-stack experiments.

**Attributes:**
- `z_position` (`List[float]`): List of Z-positions for the experiment.
- `laser_names_active` (`List[str]`): List of active laser names.
- `laser_powers_active` (`List[float]`): List of active laser powers.
- `laser_positions_active` (`List`): List of active laser positions.
- `xyoffset` (`Tuple[float, float]`): XY offset for the experiment.
- `exposure_time` (`int`): Exposure time.
- `interval_time` (`int`): Interval time.

**Methods:**
- `__post_init__()`: Validates the configuration and sets up laser powers and positions.

```python
@dataclass
class zStackExpConfig(BaseExpConfig):
    z_position: List[float]
    laser_names_active: List[str]
    laser_powers_active: List[float]
    laser_positions_active: List
    xyoffset: Tuple[float, float]

    exposure_time: int
    interval_time: int

    def __post_init__(self):
        len_ln = len(self.laser_names_active)
        len_lpow = len(self.laser_powers_active)
        len_lpos = len(self.laser_positions_active)
        if len_ln != len_lpos or len_ln != len_lpow:
            raise ValueError(
                f"Length mismatch among lists: "
                f"laser_names_active({len_ln}), "
                f"laser_powers_active({len_lpow}), "
                f"laser_positions_active({len_lpos})"
            )
        self.laser_powers = {
            self.laser_names_active[i]: self.laser_powers_active[i]
            for i in range(len(self.laser_names_active))
        }
        self.laser_positions = {
            self.laser_names_active[i]: self.laser_positions_active[i]
            for i in range(len(self.laser_names_active))
        }
```

## Functions

### `timeseriesEXP`

Runs a time series experiment using the provided virtual microscope and configuration.

**Parameters:**
- `microscope` (`VirtualMicroscope`): The virtual microscope instance.
- `config` (`TimeSeriesExpConfig`): Configuration for the time series experiment.

**Returns:**
- `Tuple[np.ndarray, MetaData]`: A tuple containing the frames and metadata.

```python
def timeseriesEXP(
    microscope: VirtualMicroscope,
    config: TimeSeriesExpConfig,
) -> Tuple[np.ndarray, MetaData]:
    frames, metadata = microscope.run_sim(
        z_val=config.z_position,
        laser_power=config.laser_powers,
        laser_position=config.laser_positions,
        xyoffset=config.xyoffset,
        duration_total=config.duration_time,
        exposure_time=config.exposure_time,
        interval_time=config.interval_time,
    )
    return np.array([frames]), metadata
```

### `zseriesEXP`

Runs a z-series (z-stack) experiment using the provided virtual microscope and configuration.

**Parameters:**
- `microscope` (`VirtualMicroscope`): The virtual microscope instance.
- `config` (`zStackExpConfig`): Configuration for the z-stack experiment.

**Returns:**
- `Tuple[np.ndarray, MetaData]`: A tuple containing the frames and metadata.

```python
def zseriesEXP(
    microscope: VirtualMicroscope,
    config: zStackExpConfig,
) -> Tuple[np.ndarray, MetaData]:
    frames = []
    for i in config.z_position:
        f, m = microscope.run_sim(
            z_val=i,
            laser_power=config.laser_powers,
            laser_position=config.laser_positions,
            xyoffset=config.xyoffset,
            duration_total=config.exposure_time + config.interval_time,
            exposure_time=config.exposure_time,
            interval_time=config.interval_time,
        )
        frames.append(f)
    metadata = m
    return np.array(frames), metadata
```

## Summary

This module provides a structured way to define and run experiments in a simulated microscopy environment. The `TimeSeriesExpConfig` and `zStackExpConfig` classes allow for detailed configuration of experiments, while the `timeseriesEXP` and `zseriesEXP` functions enable the execution of these experiments using a virtual microscope.