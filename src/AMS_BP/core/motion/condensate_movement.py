"""
Contains class for storing condensate data. Condensates are defined as spherical always; defined by a
center (x,y,z), radius (r), and time (t). The complete description of the condensate at any time (t) is:
(x,y,z,r,t).

Usage:
------
    Initialize the class as follows:
        condensate = Condensate(**{
            "initial_position":np.array([0, 0, 0]),
            "initial_time":0,
            "diffusion_coefficient":0,
            "hurst_exponent":0,
            "units_time":'ms',
            "units_position":'um',
            "condensate_id":0,
            "initial_scale":0,
            "oversample_motion_time":20,
        })
    Call the class object as follows to get the position and scale of the condensate at a given time:
        condensate(times, time_unit) -> dict{"Position":np.ndarray, "Scale":float}
"""

from typing import Optional

import numpy as np

from ...utils.decorators import cache
from ..cells import BaseCell
from .track_gen import Track_generator


def create_condensate_dict(
    initial_centers: np.ndarray,
    initial_scale: np.ndarray,
    diffusion_coefficient: np.ndarray,
    hurst_exponent: np.ndarray,
    cell: BaseCell,
    **kwargs,
) -> dict:
    """
    Creates a dictionary of condensates for simulation.

    Parameters:
    -----------
    initial_centers : np.ndarray
        Array of shape (num_condensates, 2) representing the initial centers of the condensates.
    initial_scale : np.ndarray
        Array of shape (num_condensates, 2) representing the initial scales of the condensates.
    diffusion_coefficient : np.ndarray
        Array of shape (num_condensates, 2) representing the diffusion coefficients of the condensates.
    hurst_exponent : np.ndarray
        Array of shape (num_condensates, 2) representing the Hurst exponents of the condensates.
    cell : BaseCell
        The cell that contains the condensates.
    **kwargs : dict
        Additional arguments passed to `Condensate` class.

        oversample_motion_time : int
            smallest time unit for motion (time resolution for motion) (ms)

    Returns:
    --------
    dict
        A dictionary of `Condensate` objects with keys as condensate IDs.
    """
    # check the length of diffusion_coefficient to find the number of condensates
    num_condensates = len(diffusion_coefficient)
    condensates = {}
    units_time = kwargs.get("units_time", ["ms"] * num_condensates)
    for i in range(num_condensates):
        condensates[str(i)] = Condensate(
            initial_position=initial_centers[i],
            initial_scale=initial_scale[i],
            diffusion_coefficient=diffusion_coefficient[i],
            hurst_exponent=hurst_exponent[i],
            condensate_id=int(str(i)),
            units_time=units_time[i],
            cell=cell,
        )
    return condensates


class Condensate:
    """Condensate class for storing condensate data.

    Parameters:
    -----------
    initial_position: np.ndarray = np.array([0, 0, 0])
        Initial position of the condensate.
    initial_time: float = 0
        Initial time of the condensates.
    diffusion_coefficient: float = 0
        Diffusion coefficient of the condensate.
    hurst_exponent: float = 0
        Hurst exponent of the condensate.
    units_time: str = 's'
        Units of time. Units work as follows: in the class reference frame, start from 0 and iterate by 1 each time.
        For a units_time of "ms", 1 represents 1ms.
        For a units_time of "s", 1 represents 1s.
        For a units_time of "20ms", 1 represents 20ms.
    units_position: str = 'um'
        Units of position.
    condensate_id: int = 0
        ID of the condensate.
    initial_scale: float = 0
        Initial scale of the condensate.
    cell: BaseCell = None
        The cell that contains the condensates.
    oversample_motion_time: int = None
        motion resolution

    """

    def __init__(
        self,
        initial_position: np.ndarray = np.array([0, 0, 0]),
        initial_time: int = 0,
        diffusion_coefficient: float = 0,  # same units as position and time
        hurst_exponent: float = 0,  # 0<hurst_exponent<1
        units_time: str = "ms",
        units_position: str = "um",
        condensate_id: int = 0,
        initial_scale: float = 0,
        cell: Optional[BaseCell] = None,
        oversample_motion_time: Optional[int] = None,
    ):
        self.initial_position = (
            np.array(initial_position)
            if not isinstance(initial_position, np.ndarray)
            else initial_position
        )
        self.initial_time = (
            int(initial_time) if not isinstance(initial_time, int) else initial_time
        )
        self.diffusion_coefficient = (
            np.array(diffusion_coefficient)
            if not isinstance(diffusion_coefficient, np.ndarray)
            else diffusion_coefficient
        )
        self.hurst_exponent = (
            np.array(hurst_exponent)
            if not isinstance(hurst_exponent, np.ndarray)
            else hurst_exponent
        )
        self.units_time = units_time
        self.units_position = units_position
        self.condensate_id = condensate_id
        self.initial_scale = initial_scale

        self.cell = cell
        self.dim = self.initial_position.shape[0]

        self.oversample_motion_time = oversample_motion_time
        # initialize the properties of the condensate
        self._initialize_properties()

    def _initialize_properties(self) -> None:
        """Initializes the properties of the condensate."""
        self.times = np.array([self.initial_time])
        self.condensate_positions = np.array([self.initial_position])
        self.scale = np.array([self.initial_scale])

    @property
    def times(self) -> np.ndarray:
        """Returns the times of the condensate."""
        return self._times

    @times.setter
    def times(self, value) -> None:
        # make sure this is a numpy array
        if not isinstance(value, np.ndarray):
            raise TypeError("Times must be a numpy array.")
        self._times = value

    @property
    def condensate_positions(self) -> np.ndarray:
        """Returns the positions of the condensate."""
        # make sure this is a numpy array and that it is the same dimension as the initial position
        return self._condensate_positions

    @condensate_positions.setter
    def condensate_positions(self, value) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("Condensate positions must be a numpy array.")
        if value.shape[1] != self.dim:
            raise ValueError(
                "Condensate positions must be the same dimension as the initial position."
            )
        self._condensate_positions = value

    @property
    def scale(self) -> np.ndarray:
        """Returns the scale of the condensate."""
        return self._scale

    @scale.setter
    def scale(self, value) -> None:
        self._scale = value

    def add_positions(
        self, time: np.ndarray, position: np.ndarray, scale: np.ndarray
    ) -> None:
        """Adds positions to the condensate.

        Parameters:
        -----------
        time: np.ndarray
            Times at which to add positions.
        position: np.ndarray
            Positions to add to the condensate.
        scale: np.ndarray
            Scale to add to the condensate.
        """
        self.times = np.append(self.times, time)
        self.condensate_positions = np.append(
            self.condensate_positions, position, axis=0
        )
        self.scale = np.append(self.scale, scale)

    @cache
    def __call__(self, time: int, time_unit: str) -> dict:
        """Returns the position and scale of the condensate at a given time.

        Parameters:
        -----------
        time: float
            Time at which to return the position of the condensate. User needs to convert to the reference frame of the condensate class.
        time_unit: str
            Units of time.
            Just to make sure the user is aware of the conversion they need to do to get into the reference frame of the condensate class.

        Returns:
        --------
        Dict of the position and scale of the condensate at the given time.
            Keys:
                Position: np.ndarray
                    Position of the condensate at the given time.
                Scale: float
                    Scale of the condensate at the given time.
        """
        if time_unit != self.units_time:
            # raise error that you need to ask for the time units in the condensates reference frame
            raise ValueError("Time units do not match to the condensate.")
        # check if the _condensate_positions exists
        if not hasattr(self, "_condensate_positions"):
            # if it doesn't then we need to generate the condensate positions
            self.times = np.array([self.initial_time])
            self.condensate_positions = np.array([self.initial_position])
            self.scale = np.array([self.initial_scale])
        # if the time larger than the last time in the condensate_positions then we need to generate more positions
        if time > self.times[-1]:
            self.generate_condensate_positions(time)

        return {
            "Position": self.condensate_positions[self.times == time][0],
            "Scale": self.scale[self.times == time][0],
        }

    def generate_condensate_positions(self, time: int) -> None:
        """Generates the condensate positions up to a given time.

        Parameters:
        -----------
        time: int
            Time up to which to generate the condensate positions.
        """
        # find the time difference
        time_difference = time - self.times[-1]
        # make a time array starting from the last time +1 and goin to the time inclusive
        time_array = np.arange(
            self.times[-1] + self.oversample_motion_time,
            time + self.oversample_motion_time,
        )

        track_generator = Track_generator(
            cell=self.cell,
            total_time=time,
            oversample_motion_time=self.oversample_motion_time,
        )
        track = track_generator.track_generation_no_transition(
            diffusion_coefficient=self.diffusion_coefficient,
            hurst_exponent=self.hurst_exponent,
            track_length=int(time_difference / self.oversample_motion_time),
            initials=self.condensate_positions[-1],
            start_time=self.times[-1],
        )
        track_xyz = track["xy"][:]
        # take all the x,y,z
        track_xyz = track_xyz[:, :]
        # get the scale for the time array and positions
        scales = self.calculate_scale(time_array, track_xyz)
        # add the positions to the condensate_positions
        self.add_positions(time_array, track_xyz, scales)

    def calculate_scale(self, time: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Calculates the scale of the condensate at a given time.

        Parameters:
        -----------
        time: np.ndarray
            Times at which to calculate the scale.
        position: np.ndarray
            Positions at which to calculate the scale.
        """
        # find the last scale in the scale array
        last_scale = self.scale[-1]
        # make array of length time with the last scale
        scale = np.full(time.shape, last_scale)
        return scale
