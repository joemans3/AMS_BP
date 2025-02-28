"""
simulate_foci.py
================
This file contains the necessary classes and functions to simulate foci dynamics in space, particularly within cell simulations.

Author: Baljyot Singh Parmar

Classes:
--------
- Track_generator: A class to generate tracks of foci movements in a cell space with or without transitions.
"""

import random
from typing import overload

import numpy as np
from boundedfbm.motion.FBM import FBM_BP

from ..cells import BaseCell


class Track_generator:
    """
    A class to generate tracks of foci movements in a simulated cell space.

    Parameters:
    -----------
    cell : BaseCell
        Cell object defining the space for track generation
    oversample_motion_time : int | float
        Time for oversampling motion in milliseconds.
    """

    def __init__(
        self,
        cell: BaseCell,
        total_time: int | float,
        oversample_motion_time: int | float,
    ) -> None:
        self.cell = cell
        self._allowable_cell_types()

        self.oversample_motion_time = oversample_motion_time  # in ms
        # total time in ms is the exposure time + interval time * (cycle_count) / oversample_motion_time
        # in ms
        self.total_time = total_time

    def _allowable_cell_types(self):
        # only allow rectangular cells for now
        # if not isinstance(self.cell, RectangularCell):
        #     raise ValueError(
        #         "Only rectangular cells are supported for track generation"
        #     )
        pass

    def track_generation_no_transition(
        self,
        diffusion_coefficient: float,  # um^2/s
        hurst_exponent: float,
        track_length: int,
        initials: np.ndarray,
        start_time: int | float,
    ) -> dict:
        """
        Simulates the track generation with no transition between the diffusion coefficients and the hurst exponents
        namely, this means each track has a unique diffusion coefficient and hurst exponent
        This simulation is confined to the cell space and the axial range of the cell

        Parameters:
        -----------
        diffusion_coefficient : float
            diffusion coefficient for the track
        hurst_exponent : float
            hurst exponent for the track
        track_length : int
            track_length for the track
        initials : array-like
            [[x,y,z]] coordinates of the initial positions of the track
        start_time : int
            time at which the track start (this is not the frame, and needs to be converted to the frame using the exposure time and interval time and the oversample motion time)
        Returns:
        --------
        dict-like with format: {"xy":xyz,"frames":frames,"diffusion_coefficient":diffusion_coefficient,"hurst":hurst_exponent,"initial":initial}
        """
        # initialize the fbm class
        # make self.space_lim relative to the initial position, using self.space_lim define the 0 to be initial position
        if np.shape(initials) == (2,):
            # change the shape to (3,)
            initials = np.array([initials[0], initials[1], 0])
        # convert the diffusion_coefficients
        # diffusion_coefficient = self._convert_diffcoef_um2s_um2xms(
        #     diffusion_coefficient
        # )
        fbm = FBM_BP(
            n=track_length,
            dt=self.oversample_motion_time / 1000.0,
            hurst_parameters=[hurst_exponent],
            diffusion_parameters=[diffusion_coefficient],
            diffusion_parameter_transition_matrix=[1],
            hurst_parameter_transition_matrix=[1],
            state_probability_diffusion=[1],
            state_probability_hurst=[1],
            cell=self.cell,
            initial_position=initials,
        )
        xyz = fbm.fbm(dims=3)
        # make the times starting from the starting time
        track_times = np.arange(
            start_time,
            (track_length) * self.oversample_motion_time,
            self.oversample_motion_time,
        )
        track_xyz = xyz
        # create the dict
        track_data = {
            "xy": track_xyz,
            "frames": track_times,
            "diffusion_coefficient": fbm._diff_a_n,
            "hurst": fbm._hurst_n,
            "initial": initials,
        }
        # construct the dict
        return track_data

    def track_generation_with_transition(
        self,
        diffusion_transition_matrix: np.ndarray | list,
        hurst_transition_matrix: np.ndarray | list,
        diffusion_parameters: np.ndarray | list,  # um^2/s
        hurst_parameters: np.ndarray | list,
        diffusion_state_probability: np.ndarray | list,
        hurst_state_probability: np.ndarray | list,
        track_length: int,
        initials: np.ndarray,
        start_time: int | float,
    ) -> dict:
        """
        Genereates the track data with transition between the diffusion coefficients and the hurst exponents

        Parameters:
        -----------
        diffusion_transition_matrix : array-like
            transition matrix for the diffusion coefficients
        hurst_transition_matrix : array-like
            transition matrix for the hurst exponents
        diffusion_parameters : array-like
            diffusion coefficients for the tracks
        hurst_parameters : array-like
            hurst exponents for the tracks
        diffusion_state_probability : array-like
            probabilities for the diffusion coefficients
        hurst_state_probability : array-like
            probabilities for the hurst exponents
        track_length : int
            track_length for the track
        initials : array-like
            [[x,y,z]] coordinates of the initial positions of the track
        start_time : int
            time at which the track start (this is not the frame, and needs to be converted to the frame using the exposure time and interval time and the oversample motion time)

        Returns:
        --------
        dict-like with format: {"xy":xyz,"frames":frames,"diffusion_coefficient":diffusion_coefficient,"hurst":hurst_exponent,"initial":initial}
        """
        # make self.space_lim relative to the initial position, using self.space_lim define the 0 to be initial position
        # self.space_lim is in general shape (3,2) while the initials is in shape (3,)
        # make sure the - operator is broadcasted correctly
        if np.shape(initials) == (2,):
            # change the shape to (3,)
            initials = np.array([initials[0], initials[1], 0])
        # subtract each element of the first dimension of self.space_lim by the first element of initials

        # convert the diffusion_coefficients
        # diffusion_parameters = self._convert_diffcoef_um2s_um2xms(diffusion_parameters)
        # initialize the fbm class
        fbm = FBM_BP(
            n=track_length,
            dt=self.oversample_motion_time / 1000.0,
            hurst_parameters=hurst_parameters,
            diffusion_parameters=diffusion_parameters,
            diffusion_parameter_transition_matrix=diffusion_transition_matrix,
            hurst_parameter_transition_matrix=hurst_transition_matrix,
            state_probability_diffusion=diffusion_state_probability,
            state_probability_hurst=hurst_state_probability,
            cell=self.cell,
            initial_position=initials,
        )
        xyz = fbm.fbm(dims=3)
        # make the times starting from the starting time
        track_times = np.arange(
            start_time,
            track_length * self.oversample_motion_time,
            self.oversample_motion_time,
        )
        track_xyz = xyz
        # create the dict
        track_data = {
            "xy": track_xyz,
            "frames": track_times,
            "diffusion_coefficient": fbm._diff_a_n,
            "hurst": fbm._hurst_n,
            "initial": initials,
        }
        # construct the dict
        return track_data

    def track_generation_constant(
        self, track_length: int, initials: np.ndarray, start_time: int
    ) -> dict:
        """
        Generate a constant track (no movement).

        Parameters:
        -----------
        track_length : int
            mean track length, in this case the track length is constant with this mean
        initials : array-like
            [[x,y,z]] coordinates of the initial positions of the track
        starting_time : int
            time at which the track start (this is not the frame, and needs to be converted to the frame using the exposure time and interval time and the oversample motion time)

        Returns:
        --------
        dict-like with format: {"xy":xyz,"frames":frames,"diffusion_coefficient":diffusion_coefficient,"hurst":hurst_exponent,"initial":initial}
        """
        # make the times starting from the starting time
        track_times = np.arange(
            start_time,
            track_length * self.oversample_motion_time,
            self.oversample_motion_time,
        )
        # make the track x,y,z from the initial positions
        track_xyz = np.tile(initials, (len(track_times), 1))
        # construct the dict
        track_data = {
            "xy": track_xyz,
            "frames": track_times,
            "diffusion_coefficient": 0,
            "hurst": 0,
            "initial": initials,
        }
        return track_data

    @overload
    def _convert_diffcoef_um2s_um2xms(self, diffusion_coefficient: float) -> float: ...
    @overload
    def _convert_diffcoef_um2s_um2xms(
        self, diffusion_coefficient: np.ndarray
    ) -> np.ndarray: ...
    @overload
    def _convert_diffcoef_um2s_um2xms(self, diffusion_coefficient: list) -> list: ...
    def _convert_diffcoef_um2s_um2xms(
        self, diffusion_coefficient: float | np.ndarray | list
    ) -> float | np.ndarray | list:
        """converts um^2/s diffusion_coefficient into um^2/ x ms
        x = amount of ms
        ms = milliseconds

        x ms = self.oversample_motion_time (in ms, int)"""
        if isinstance(diffusion_coefficient, (np.ndarray, float)):
            return (
                1.0 / (1000.0 / self.oversample_motion_time)
            ) * diffusion_coefficient
        elif isinstance(diffusion_coefficient, list):
            return [
                (1.0 / (1000.0 / self.oversample_motion_time)) * i
                for i in diffusion_coefficient
            ]
        else:
            raise TypeError(f"Unsupported type: {type(diffusion_coefficient)}")

    def _convert_time_to_frame(
        self, time: int, exposure_time: int, interval_time: int
    ) -> int:
        """
        Parameters:
        -----------
        time : int
            time in ms

        Returns:
        --------
        int: frame number
        """
        return int(
            (time * self.oversample_motion_time) / (exposure_time + interval_time)
        )

    def _convert_frame_to_time(
        self, frame: int, exposure_time: int, interval_time: int
    ) -> int:
        """
        Parameters:
        -----------
        frame : int
            frame number

        Returns:
        --------
        int: time in ms
        """
        return int((frame * (exposure_time + interval_time)))


def _initialize_points_per_time(total_time: int, oversample_motion_time: int) -> dict:
    """Initialize empty points per time dictionary.

    Returns
    -------
    dict
        Empty dictionary with keys for each time point
    """
    return {
        str(i): []
        for i in np.arange(
            0, total_time + oversample_motion_time, oversample_motion_time
        )
    }


def _update_points_per_time(points_per_time: dict, track: dict) -> None:
    """Update points per time dictionary with new track data.

    Parameters
    ----------
    points_per_time : dict
        Dictionary to update
    track : dict
        Track data to add
    """
    for frame, position in zip(track["frames"], track["xy"]):
        points_per_time[str(frame)].append(position)


def _generate_constant_tracks(
    track_generator: Track_generator,
    track_lengths: list | np.ndarray | int,
    initial_positions: np.ndarray,
    starting_frames: int = 0,
) -> tuple[dict, dict]:
    """Generate tracks with constant parameters."""
    if isinstance(track_lengths, int):
        track_lengths = np.full(len(initial_positions), track_lengths)
    if isinstance(starting_frames, int):
        starting_frames = np.full(len(initial_positions), starting_frames)

    tracks = {}
    points_per_time = _initialize_points_per_time(
        track_generator.total_time, track_generator.oversample_motion_time
    )
    for i in range(len(track_lengths)):
        tracks[i] = track_generator.track_generation_constant(
            track_length=track_lengths[i],
            initials=initial_positions[i],
            start_time=starting_frames[i],
        )
        _update_points_per_time(points_per_time, tracks[i])

    return tracks, points_per_time


def _generate_no_transition_tracks(
    track_generator: Track_generator,
    track_lengths: list | np.ndarray | int,
    initial_positions: np.ndarray,
    starting_frames: int,
    diffusion_parameters: np.ndarray,
    hurst_parameters: np.ndarray,
) -> tuple[dict, dict]:
    """Generate tracks without state transitions.

    Parameters
    ----------
    track_generator : sf.Track_generator
        Track generator instance
    track_lengths : list | np.ndarray | int
        Track lengths
    initial_positions : np.ndarray
        Initial positions
    starting_frames : int
        Starting frames
    diffusion_parameters : np.ndarray
        Diffusion parameters
    hurst_parameters : np.ndarray
        Hurst parameters

    Returns
    -------
    tuple[dict, dict]
        Tracks dictionary and points per time dictionary
    """
    if isinstance(track_lengths, int):
        track_lengths = np.full(len(initial_positions), track_lengths)
    if isinstance(starting_frames, int):
        starting_frames = np.full(len(initial_positions), starting_frames)

    tracks = {}
    points_per_time = _initialize_points_per_time(
        track_generator.total_time, track_generator.oversample_motion_time
    )

    for i in range(len(track_lengths)):
        # Randomly select diffusion coefficient and hurst exponent indices
        diff_idx = random.randint(0, len(diffusion_parameters) - 1)
        hurst_idx = random.randint(0, len(hurst_parameters) - 1)

        # Generate track with selected parameters
        tracks[i] = track_generator.track_generation_no_transition(
            track_length=track_lengths[i],
            initials=initial_positions[i],
            start_time=starting_frames[i],
            diffusion_coefficient=diffusion_parameters[diff_idx],
            hurst_exponent=hurst_parameters[hurst_idx],
        )
        _update_points_per_time(points_per_time, tracks[i])

    return tracks, points_per_time


def _generate_transition_tracks(
    track_generator: Track_generator,
    track_lengths: list | np.ndarray | int,
    initial_positions: np.ndarray,
    starting_frames: int,
    diffusion_parameters: np.ndarray,
    hurst_parameters: np.ndarray,
    diffusion_transition_matrix: np.ndarray,
    hurst_transition_matrix: np.ndarray,
    diffusion_state_probability: np.ndarray,
    hurst_state_probability: np.ndarray,
) -> tuple[dict, dict]:
    """Generate tracks with state transitions.

    Parameters
    ----------
    track_generator : sf.Track_generator
        Track generator instance
    track_lengths : list | np.ndarray | int
        Track lengths
    initial_positions : np.ndarray
        Initial positions
    starting_frames : int
        Starting frames
    diffusion_parameters : np.ndarray
        Diffusion parameters
    hurst_parameters : np.ndarray
        Hurst parameters
    diffusion_transition_matrix : np.ndarray
        Diffusion transition matrix
    hurst_transition_matrix : np.ndarray
        Hurst transition matrix
    diffusion_state_probability : np.ndarray
        Diffusion state probability
    hurst_state_probability : np.ndarray
        Hurst state probability

    Returns
    -------
    tuple[dict, dict]
        Tracks dictionary and points per time dictionary
    """
    if isinstance(track_lengths, int):
        track_lengths = np.full(len(initial_positions), track_lengths)
    if isinstance(starting_frames, int):
        starting_frames = np.full(len(initial_positions), starting_frames)

    tracks = {}
    points_per_time = _initialize_points_per_time(
        track_generator.total_time, track_generator.oversample_motion_time
    )

    for i in range(len(track_lengths)):
        # Generate track with transitions
        tracks[i] = track_generator.track_generation_with_transition(
            diffusion_transition_matrix=diffusion_transition_matrix,
            hurst_transition_matrix=hurst_transition_matrix,
            diffusion_parameters=diffusion_parameters,
            hurst_parameters=hurst_parameters,
            diffusion_state_probability=diffusion_state_probability,
            hurst_state_probability=hurst_state_probability,
            track_length=track_lengths[i],
            initials=initial_positions[i],
            start_time=starting_frames[i],
        )
        _update_points_per_time(points_per_time, tracks[i])

    return tracks, points_per_time


def _convert_tracks_to_trajectory(tracks: dict) -> dict:
    """Convert tracks to trajectory."""
    return {i: tracks["xy"][j] for j, i in enumerate(tracks["frames"])}
