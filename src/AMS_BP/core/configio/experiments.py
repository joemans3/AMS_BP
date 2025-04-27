from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..metadata.metadata import MetaData
from ..sim_microscopy import VirtualMicroscope


@dataclass
class BaseExpConfig:
    name: str
    description: str


@dataclass
class TimeSeriesExpConfig(BaseExpConfig):
    z_position: float
    laser_names_active: List[str]
    laser_powers_active: List[float]
    laser_positions_active: List
    xyoffset: Tuple[float, float]
    scanning: bool = False

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


@dataclass
class zStackExpConfig(BaseExpConfig):
    z_position: List[float]
    laser_names_active: List[str]
    laser_powers_active: List[float]
    laser_positions_active: List
    xyoffset: Tuple[float, float]

    exposure_time: int
    interval_time: int

    scanning: bool = False

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
        scanning=config.scanning,
    )
    return np.array([frames]), metadata


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
            scanning=config.scanning,
        )
        frames.append(f)
    # m.Channel = {"name": microscope.channels.names}
    # m.TimeIncrementUnit = None
    # m.TimeIncrement = None
    metadata = m
    return np.array(frames), metadata
