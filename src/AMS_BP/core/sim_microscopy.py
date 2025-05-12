from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from ..utils.util_functions import ms_to_seconds
from .configio.configmodels import ConfigList
from .metadata.metadata import MetaData
from .optics.camera.detectors import Detector
from .optics.camera.quantum_eff import QuantumEfficiency
from .optics.filters.channels.channelschema import Channels
from .optics.lasers import LaserProfile
from .optics.psf import PSFEngine
from .photophysics.photon_physics import (
    AbsorptionPhysics,
    EmissionPhysics,
    incident_photons,
)
from .photophysics.state_kinetics import StateTransitionCalculator
from .sample.flurophores.flurophore_schema import StateType, WavelengthDependentProperty
from .sample.sim_sampleplane import EMPTY_STATE_HISTORY_DICT, SamplePlane


class VirtualMicroscope:
    def __init__(
        self,
        camera: Tuple[Detector, QuantumEfficiency],
        sample_plane: SamplePlane,
        lasers: Dict[str, LaserProfile],
        channels: Channels,
        psf: Callable[[float | int, Optional[float | int]], PSFEngine],
        config: ConfigList,
        collection_efficiency: float = 1,
        start_time: int = 0,
    ):
        # Core components
        self.camera = camera[0]
        self.qe = camera[1]
        self.sample_plane = sample_plane
        self.lasers = lasers
        self.channels = channels
        self.psf = psf
        self._time = start_time  # ms
        self.config = config
        self.collection_efficiency = collection_efficiency

        # Cached initial configuration
        self._cached_initial_config()
        self._set_laser_position_center_cell()

    def _set_laser_position_center_cell(self) -> None:
        # center of cell
        cell_bounds = np.array(self.sample_plane.fov_bounds)

        x_c = (cell_bounds[0][1] - cell_bounds[0][0]) / 2.0
        y_c = (cell_bounds[1][1] - cell_bounds[1][0]) / 2.0
        z_c = (cell_bounds[2][1] - cell_bounds[2][0]) / 2.0

        for laser in self.lasers.keys():
            self.lasers[laser].params.position = np.array([x_c, y_c, z_c])

    def _cached_initial_config(self) -> None:
        """Cache the initial configuration of the microscope"""
        self.initial_config = {
            "camera": self.camera,
            "qe": self.qe,
            "sample_plane": self.sample_plane,
            "lasers": self.lasers,
            "channels": self.channels,
            "psf": self.psf,
            "config": self.config,
            "_time": self._time,
        }

    def _set_laser_powers(
        self, laser_power: Dict[str, Union[float, Callable[[float], float]]]
    ) -> None:
        if laser_power is not None:
            for laser in laser_power.keys():
                if isinstance(self.lasers[laser].params.power, float) and isinstance(
                    laser_power[laser], float
                ):
                    if laser_power[laser] > self.lasers[laser].params.max_power:
                        raise ValueError(
                            "Provided laser power for laser: {} nm, is larger than the maximum power: {}".format(
                                self.lasers[laser].params.wavelength,
                                self.lasers[laser].params.max_power,
                            )
                        )
                self.lasers[laser].params.power = laser_power[laser]

    def _set_laser_positions(
        self,
        laser_positions: Dict[
            str,
            Union[
                Tuple[float, float, float],
                Callable[[float], Tuple[float, float, float]],
            ],
        ],
    ) -> None:
        if laser_positions is not None:
            for laser in laser_positions.keys():
                self.lasers[laser].params.position = laser_positions[laser]

    def run_sim(
        self,
        z_val: float,  # um
        laser_power: Dict[
            str, Union[float, Callable[[float], float]]  # power or f(t) -> power
        ],  # str = lasername, float = power in W
        xyoffset: Tuple[
            float, float
        ],  # location of the bottom left corner of the field of view -> sample -> camera
        laser_position: Optional[
            Dict[
                str,  # laser name
                Union[
                    Tuple[float, float, float],  # x, y, z
                    Callable[[float], Tuple[float, float, float]],  # f(t) -> x, y, z
                ],
            ]
        ] = None,  # str = lasername, Tuple = x, y, z in um at the sample plane
        duration_total: Optional[int] = None,  # ms
        exposure_time: Optional[int] = None,
        interval_time: Optional[int] = None,
        scanning: Optional[
            bool
        ] = False,  # True if scanning -> laser will autoposition to the xy location of the fluorophore but at the z_val of the plane being imaged and NOT the z value of the fluorophore.
        # as a consequence the "effective" time spent by the laser of on the fluorophore position is not the dwell time of traditional confocal, but "always on". TODO: better fix.
    ) -> Tuple[np.ndarray, MetaData]:
        self._set_laser_powers(laser_power=laser_power)
        if laser_position is not None:
            self._set_laser_positions(laser_positions=laser_position)

        duration_total, exposure_time, interval_time = timeValidator(
            self.config.GlobalParameter.exposure_time,
            self.config.GlobalParameter.interval_time,
            self.sample_plane.dt,
            self.sample_plane.t_end,
            self._time,
            {
                "exposure_time": exposure_time,
                "interval_time": interval_time,
                "total_time": duration_total,
            },
        )

        timestoconsider, frame_list, max_frame = generate_sampling_pattern(
            exposure_time,
            interval_time,
            self._time,
            self._time + duration_total,
            self.sample_plane.dt,
        )

        mapSC = {}
        for i in range(self.channels.num_channels):
            mapSC[self.channels.names[i]] = mapSampleCamera(
                sampleplane=self.sample_plane,
                camera=self.camera,
                xyoffset=xyoffset,
                frames=max_frame,
            )
        image_stack = []
        channel_names = []

        # for each object find its location and the excitation laser intensity (after applying excitation filter)

        for time_index, time in enumerate(timestoconsider):
            # transmission rate (1/s) at the current time for each filterset (channel) for each flurophore
            for objID, fluorObj in self.sample_plane._objects.items():
                # find the current state history of the fluorophore
                # (State, random_Val, list[StateTransition])
                statehist = fluorObj.state_history[time]
                # if not aval transitions go next
                # this assumes that the bleached state is the only state from which there are no more transitions
                if (not statehist[2]) and (
                    statehist[0].state_type != StateType.FLUORESCENT
                ):
                    fluorObj.state_history[time + self.sample_plane.dt] = statehist
                    continue

                # flurophore position
                florPos = np.copy(fluorObj.position_history[time])
                # make z relative to the z_Val of the stage
                florPos[2] -= z_val

                # overlap in the transmission of the lasers into the sample-plane
                # intensity
                def laser_intensities_gen(
                    florPos: list | np.ndarray, time: int | float
                ) -> dict:
                    laser_intensities = {}
                    for laserID in laser_power.keys():
                        if scanning:
                            self.lasers[laserID].params.position = (
                                *florPos[:2],
                                0,
                            )  # z value is 0 since this is the neew focus plane
                        laser_intensities[laserID] = {
                            "wavelength": self.lasers[laserID].params.wavelength,
                            "intensity": (
                                self.channels.filtersets[
                                    0  # assumes all lasers are on at once in all channels
                                ].excitation.find_transmission(
                                    self.lasers[laserID].params.wavelength
                                )
                                * self.lasers[laserID].calculate_intensity(
                                    x=florPos[0], y=florPos[1], z=florPos[2], t=time
                                )  # W/um²
                            ),
                        }
                    return laser_intensities

                state_calculator_setup = StateTransitionCalculator(
                    fluorObj, self.sample_plane.dt_s, time, laser_intensities_gen
                )
                # ({state.name: [total_time_in_state, laser_intensities]}, final_state (State)
                fluorescent_state_history, final_state, ernomsg = (
                    state_calculator_setup()
                )

                if frame_list[time_index] != 0:
                    for (
                        fstateNAME,
                        time_and_laser_intensities,
                    ) in fluorescent_state_history.items():
                        deltaTime, laser_intensities = time_and_laser_intensities
                        fstate = fluorObj.fluorophore.states[fstateNAME]

                        wl_t = []
                        int_t = []
                        for i in laser_intensities.values():
                            wl_t.append(i["wavelength"])
                            int_t.append(i["intensity"])

                        absorb_photon_cl = AbsorptionPhysics(
                            excitation_spectrum=fstate.excitation_spectrum,
                            intensity_incident=WavelengthDependentProperty(
                                wavelengths=wl_t, values=int_t
                            ),
                            absorb_cross_section_spectrum=WavelengthDependentProperty(
                                wavelengths=wl_t,
                                values=[
                                    fstate.molar_cross_section.get_value(wl)  # pyright: ignore
                                    for wl in wl_t
                                ],
                            ),
                            fluorescent_lifetime_inverse=fstate.fluorescent_lifetime_inverse,
                        )
                        absorbed_photon_rate = absorb_photon_cl.absorbed_photon_rate()

                        for channel_num in range(self.channels.num_channels):
                            emission_photon_cl = EmissionPhysics(
                                emission_spectrum=fstate.emission_spectrum,
                                quantum_yield=fstate.quantum_yield,
                                transmission_filter=self.channels.filtersets[
                                    channel_num
                                ].emission,
                            )

                            transmission_photon_rate = emission_photon_cl.transmission_photon_rate(
                                emission_photon_rate_lambda=emission_photon_cl.emission_photon_rate(
                                    total_absorbed_rate=absorbed_photon_rate
                                    * self.channels.splitting_efficiency[channel_num]
                                )
                            )

                            inc = incident_photons(
                                transmission_photon_rate,
                                self.qe,
                                self.psf,
                                florPos,
                            )
                            _, psfs = inc.incident_photons_calc(
                                deltaTime, self.collection_efficiency
                            )
                            for ipsf in psfs:
                                mapSC[self.channels.names[channel_num]].add_psf_frame(
                                    ipsf, florPos[:2], frame_list[time_index]
                                )

                statehist_updated = (
                    final_state,
                    EMPTY_STATE_HISTORY_DICT,
                    [
                        stateTrans
                        for stateTrans in fluorObj.fluorophore.transitions.values()
                        if stateTrans.from_state == final_state.name
                    ],
                )
                fluorObj.state_history[time + self.sample_plane.dt] = statehist_updated

        # use photon frames to make digital image
        exposure_time_sc = ms_to_seconds(exposure_time)
        for channel_name in mapSC.keys():
            frames = [
                self.camera.capture_frame(elpho, exposure_time=exposure_time_sc)
                for elpho in mapSC[channel_name].holdframe.frames
            ]
            image_stack.append(frames)
            channel_names.append(channel_name)
        self._time += duration_total
        metadata = MetaData(
            notes="Not implimented",
            axes="ZCTYX",
            TimeIncrement=interval_time + exposure_time,
            TimeIncrementUnit="ms",
            PhysicalSizeX=self.camera.pixel_size * 1e-6,
            PhysicalSizeXUnit="m",
            PhysicalSizeY=self.camera.pixel_size * 1e-6,
            PhysicalSizeYUnit="m",
            Channel={"Name": self.channels.names},
        )

        # return frames in the format ZCTYX
        # Z is infered from the stack so [np.array(image_stack)] is one Z stack
        return np.array(image_stack), metadata

    def reset_to_initial_config(self) -> bool:
        """Reset to initial configuration."""
        for key, value in self.initial_config.items():
            setattr(self, key, value)
        return True


@dataclass
class mapSampleCamera:
    """Maps the location on the x,y detector grid where the sample plane resides,
    Determines the index x,y of the bottom left corner of the detector grid at which the psf array starts"""

    sampleplane: SamplePlane
    camera: Detector
    xyoffset: Tuple[float, float]  # in um
    frames: int = 1

    def __post_init__(self):
        self.holdframe = PhotonFrameContainer(
            [self.camera.base_frame(base_adu=0) for _ in range(self.frames)]
        )

    def get_pixel_indices(self, x: float, y: float) -> Tuple[int, int]:
        # convert the x,y position of the sample plane to the detector grid
        # the offset is in the reference frame of the sample plane
        # can be negative
        return (
            int((x - self.xyoffset[0]) / self.camera.pixel_size),
            int((y - self.xyoffset[1]) / self.camera.pixel_size),
        )

    def add_psf_frame(
        self, psf: np.ndarray, mol_pos: Tuple[float, float], frame_num: int
    ) -> None:
        psf_shape = psf.shape
        frame_shape = self.holdframe.frames[frame_num - 1].shape
        # find the pixel indices of the psf
        x, y = self.get_pixel_indices(mol_pos[0], mol_pos[1])
        # find the bottom left corner of the psf
        x0 = x - psf_shape[0] // 2
        y0 = y - psf_shape[1] // 2

        # Calculate the overlapping region between PSF and frame
        # For the frame
        frame_x_start = max(0, x0)
        frame_y_start = max(0, y0)
        frame_x_end = min(frame_shape[0], x0 + psf_shape[0])
        frame_y_end = min(frame_shape[1], y0 + psf_shape[1])

        # If there's no overlap, return
        if frame_x_end <= frame_x_start or frame_y_end <= frame_y_start:
            return

        # Calculate corresponding region in the PSF
        psf_x_start = max(0, -x0)
        psf_y_start = max(0, -y0)
        psf_x_end = psf_shape[0] - max(0, (x0 + psf_shape[0]) - frame_shape[0])
        psf_y_end = psf_shape[1] - max(0, (y0 + psf_shape[1]) - frame_shape[1])

        # Add the overlapping region of the PSF to the frame
        self.holdframe.frames[frame_num - 1][
            frame_y_start:frame_y_end, frame_x_start:frame_x_end
        ] += psf[psf_y_start:psf_y_end, psf_x_start:psf_x_end]

    def get_frame(self, frame_num: int) -> np.ndarray:
        return self.holdframe.frames[frame_num]


@dataclass
class PhotonFrameContainer:
    """Container for the frames of the simulation"""

    frames: List[np.ndarray]

    def __iter__(self):
        return iter(self.frames)

    def __len__(self):
        return len(self.frames)


def generate_sampling_pattern(
    exposure_time, interval_time, start_time, end_time, oversample_motion_time
) -> Tuple[List[int], List[int], int]:
    """
    Generate a sampling pattern based on exposure and interval times.

    Args:
    - exposure_time: Duration of each exposure
    - interval_time: Duration between exposures
    - start_time: Beginning of the sampling period
    - end_time: End of the sampling period
    - oversample_motion_time: Time resolution for oversampling

    Returns:
    - times: List of sampling times
    - sample_bool: List indicating frame numbers or intervals

    Notes:
    - Think of this as a cyclic pattern:
    * * * * * - - - * * * * * - - -  * * * * * - - -
    |______________|
        Cycle 1
    - * = Exposure period
    - - = Interval period
    iterating over the sampling states allows to find the sample state which is exposed or not using modulo operations (cycle % (exposure + interval))
    """
    int_e_o = int(exposure_time / oversample_motion_time)
    int_f_i_o = int((end_time - start_time) / oversample_motion_time)
    int_i_o = int(interval_time / oversample_motion_time)

    times = []
    sample_bool = []
    frame_num = 1

    for counter_times in range(int_f_i_o):
        # Determine current cycle state
        exposure_cycle = counter_times % (int_e_o + int_i_o)

        if exposure_cycle < int_e_o:
            # During exposure period
            sample_bool.append(frame_num)
        else:
            # During interval period
            sample_bool.append(0)

        # Reset frame when a full cycle is complete
        if (counter_times + 1) % (int_e_o + int_i_o) == 0:
            frame_num += 1

        times.append(counter_times * oversample_motion_time + start_time)

    return times, sample_bool, frame_num - 1


def timeValidator(
    oexposure_time: int,
    ointerval_time: int,
    oversample_motion_time: int,
    ototal_time: int,
    current_time: int,
    state_arr: Dict[
        Literal["exposure_time", "interval_time", "total_time"], int | None
    ],
) -> Tuple[int, int, int]:
    duration_total = state_arr.get("total_time", ototal_time) or ototal_time
    exposure_time = state_arr.get("exposure_time", oexposure_time) or oexposure_time
    interval_time = state_arr.get("interval_time", ointerval_time) or ointerval_time

    if (current_time + duration_total) > ototal_time:
        raise ValueError(
            f"Duration_total: {duration_total + current_time}, is larger than the largest simulated: {ototal_time}. Choose a smaller value or rerun the microscope setup with a longer duration."
        )

    if exposure_time < oversample_motion_time and exposure_time > 0:
        raise ValueError(
            f"Exposure time: {exposure_time}, is smaller than the original oversample_motion_time: {oversample_motion_time}"
        )
    if interval_time < oversample_motion_time and interval_time > 0:
        raise ValueError(
            f"Interval time: {interval_time}, is smaller than the original oversample_motion_time: {oversample_motion_time}"
        )

    if duration_total % (exposure_time + interval_time) != 0:
        raise ValueError(
            f"Total duration: {duration_total} needs to be a multiple of exposure time: {exposure_time} + interval time: {interval_time}."
        )
    return duration_total, exposure_time, interval_time
