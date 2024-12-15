from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from SMS_BP_adv.optics.camera.detectors import Detector
from SMS_BP_adv.optics.camera.quantum_eff import QuantumEfficiency
from SMS_BP_adv.photophysics.photon_physics import (
    AbsorptionPhysics,
    EmissionPhysics,
    incident_photons,
)

from .configio.configmodels import ConfigList
from .optics.camera import CMOSDetector, EMCCDDetector
from .optics.filters import FilterSet
from .optics.lasers import LaserProfile
from .optics.psf import PSFEngine
from .probabilityfuncs.markov_chain import rate_to_probability
from .sample.flurophores.flurophore_schema import StateType, WavelengthDependentProperty
from .sample.sim_sampleplane import EMPTY_STATE_HISTORY_DICT, SamplePlane


class VirtualMicroscope:
    def __init__(
        self,
        camera: Tuple[EMCCDDetector | CMOSDetector, QuantumEfficiency],
        sample_plane: SamplePlane,
        lasers: Dict[str, LaserProfile],
        filterset: FilterSet,
        psf: Callable[[float | int, Optional[float | int]], PSFEngine],
        config: ConfigList,
        start_time: int = 0,
    ):
        # Core components
        self.camera = camera[0]
        self.qe = camera[1]
        self.sample_plane = sample_plane
        self.lasers = lasers
        self.filterset = filterset
        self.psf = psf
        self._time = start_time  # ms
        self.config = config

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
            "filterset": self.filterset,
            "psf": self.psf,
            "config": self.config,
            "_time": self._time,
        }

    def _set_laser_powers(self, laser_power: Dict[str, float]) -> None:
        if laser_power is not None:
            for laser in laser_power.keys():
                if isinstance(self.lasers[laser].params.power, float):
                    if laser_power[laser] > self.lasers[laser].params.max_power:
                        raise ValueError(
                            "Provided laser power for laser: {} nm, is larger than the maximum power: {}".format(
                                self.lasers[laser].params.wavelength,
                                self.lasers[laser].params.max_power,
                            )
                        )
                self.lasers[laser].params.power = laser_power[laser]

    def _set_laser_positions(
        self, laser_positions: Dict[str, Tuple[float, float, float]]
    ) -> None:
        if laser_positions is not None:
            for laser in laser_positions.keys():
                self.lasers[laser].params.position = laser_positions[laser]

    def aquire_time_series(
        self,
        z_val: float,  # um
        laser_power: Dict[str, float],  # str = lasername, float = power in W
        xyoffset: Tuple[
            float, float
        ],  # location of the bottom left corner of the field of view -> sample -> camera
        laser_position: Dict[str, Tuple[float, float, float]]
        | None,  # str = lasername, Tuple = x, y, z in um at the sample plane
        duration_total: Optional[int] = None,  # ms
    ) -> np.ndarray:
        self._set_laser_powers(laser_power=laser_power)
        if laser_position is not None:
            self._set_laser_positions(laser_positions=laser_position)

        if duration_total is None:
            duration_total = self.sample_plane.t_end

        timestoconsider, frame_list, max_frame = generate_sampling_pattern(
            self.config.GlobalParameters.exposure_time,
            self.config.GlobalParameters.interval_time,
            self._time,
            self._time + duration_total,
            self.config.GlobalParameters.oversample_motion_time,
        )
        mapSC = mapSampleCamera(
            sampleplane=self.sample_plane,
            camera=self.camera,
            xyoffset=xyoffset,
            frames=max_frame,
        )
        print(timestoconsider, frame_list, max_frame)

        # for each object find its location and the excitation laser intensity (after applying excitation filter)
        for time_index, time in enumerate(timestoconsider):
            # transmission rate (1/s) at the current time for each filterset (channel) for each flurophore
            for objID, fluorObj in self.sample_plane._objects.items():
                # flurophore position
                florPos = fluorObj.position_history[time]
                # make z relative to the z_Val of the stage
                florPos[2] -= z_val

                # find the current state history of the fluorophore
                statehist = fluorObj.state_history[time]

                # (State, photon_count, list[StateTransition])
                # overlap in the transmission of the lasers into the sample-plane
                # intensity
                laser_intensities = {}
                for laserID in laser_power.keys():
                    laser_intensities[laserID] = {
                        "wavelength": self.lasers[laserID].params.wavelength,
                        "intensity": (
                            self.filterset.excitation.find_transmission(
                                self.lasers[laserID].params.wavelength
                            )
                            * self.lasers[laserID].calculate_intensity(
                                x=florPos[0], y=florPos[1], z=florPos[2], t=time
                            )  # W/um²
                        ),
                    }

                # make an array of all the possible states
                statearr = [strans.to_state for strans in statehist[2]]
                stateTransitionMatrix = [
                    sum(
                        strans.activation_rate()(
                            laser["wavelength"], laser["intensity"]
                        )
                        for laser in laser_intensities.values()
                    )
                    for strans in statehist[2]
                ]

                stateTransitionMatrix = [
                    rate_to_probability(i, self.sample_plane.dt * (1e-3))
                    for i in stateTransitionMatrix
                ]  # input required is 1/s and s. time is in ms, rate is in 1/s from the .activation_rate() method
                # make the self->self state the rest of the probability
                nothing_change_prob = 1.0 - np.sum(stateTransitionMatrix)
                statearr = [statehist[0].name] + statearr
                stateTransitionMatrix = [nothing_change_prob] + list(
                    stateTransitionMatrix
                )

                assert np.sum(stateTransitionMatrix) == 1.0

                next_state_index = np.random.choice(
                    np.arange(len(statearr)), p=stateTransitionMatrix
                )

                # initialize None transmission_photon_rate
                transmission_photon_rate = None
                if (
                    fluorObj.fluorophore.states[statearr[next_state_index]].state_type
                    == StateType.FLUORESCENT
                ):
                    if (
                        fluorObj.fluorophore.states[
                            statearr[next_state_index]
                        ].quantum_yield
                        is None
                    ):
                        raise ValueError(
                            "Fluorescent states must have spectral data, quantum yield, extinction coefficient, and photon budget"
                        )
                    # make the absorb photon class
                    wl_t = []
                    int_t = []
                    for i in laser_intensities.values():
                        wl_t.append(i["wavelength"])
                        int_t.append(i["intensity"])

                    absorb_photon_cl = AbsorptionPhysics(
                        intensity_spectrum=fluorObj.fluorophore.states[
                            statearr[next_state_index]
                        ].excitation_spectrum,
                        intensity_incident=WavelengthDependentProperty(
                            wavelengths=wl_t, values=int_t
                        ),
                        absorb_cross_section_spectrum=WavelengthDependentProperty(
                            wavelengths=wl_t,
                            values=[
                                fluorObj.fluorophore.states[
                                    statearr[next_state_index]
                                ].molar_cross_section.get_value(wl)  # pyright: ignore
                                for wl in wl_t
                            ],
                        ),
                    )

                    emision_photon_cl = EmissionPhysics(
                        emission_spectrum=fluorObj.fluorophore.states[  # pyright: ignore
                            statearr[next_state_index]
                        ].emission_spectrum,
                        quantum_yield=fluorObj.fluorophore.states[  # pyright: ignore
                            statearr[next_state_index]
                        ].quantum_yield,
                        transmission_filter=self.filterset.emission,
                    )

                    transmission_photon_rate = emision_photon_cl.transmission_photon_rate(
                        emission_photon_rate_lambda=emision_photon_cl.emission_photon_rate(
                            total_absorbed_rate=absorb_photon_cl.absorbed_photon_rate()
                        )
                    )
                # if no fluorescent state the transmission_photon_rate is None -> update state history with the empty.
                if transmission_photon_rate is None:
                    statehist_updated = (
                        fluorObj.fluorophore.states[statearr[next_state_index]],
                        EMPTY_STATE_HISTORY_DICT,
                        [
                            stateTrans
                            for stateTrans in fluorObj.fluorophore.transitions.values()
                            if stateTrans.from_state
                            == fluorObj.fluorophore.states[
                                statearr[next_state_index]
                            ].name
                        ],
                    )
                else:
                    # build transmission_dict
                    transmission_dict = {self.filterset.name: transmission_photon_rate}
                    statehist_updated = (
                        fluorObj.fluorophore.states[statearr[next_state_index]],
                        transmission_dict,
                        [
                            stateTrans
                            for stateTrans in fluorObj.fluorophore.transitions.values()
                            if stateTrans.from_state
                            == fluorObj.fluorophore.states[
                                statearr[next_state_index]
                            ].name
                        ],
                    )
                fluorObj.state_history[time + self.sample_plane.dt] = statehist_updated

                if frame_list[time_index] != 0:
                    if transmission_photon_rate is None:
                        inc_photons = None
                        psfs = None
                    else:
                        inc = incident_photons(
                            transmission_photon_rate,
                            self.qe,
                            self.psf,
                            florPos,
                        )
                        inc_photons, psfs = inc.incident_photons_calc(
                            self.sample_plane.dt
                        )
                        for ipsf in psfs:
                            mapSC.add_psf_frame(
                                ipsf, florPos[:2], frame_list[time_index]
                            )

        # use photon frames to make digital image
        frames = [
            self.camera.capture_frame(
                elpho, exposure_time=self.config.GlobalParameters.exposure_time * 1e-3
            )
            for elpho in mapSC.holdframe.frames
        ]
        return np.array(frames)

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
        # find the pixel indices of the psf
        x, y = self.get_pixel_indices(mol_pos[0], mol_pos[1])
        # find the bottom left corner of the psf
        x0 = x - int(psf.shape[0] / 2)
        y0 = y - int(psf.shape[1] / 2)
        # add the psf to the frame
        self.holdframe.frames[frame_num - 1][
            y0 : y0 + psf.shape[0], x0 : x0 + psf.shape[1]
        ] += psf

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

        times.append(counter_times * oversample_motion_time)

    return times, sample_bool, frame_num - 1
