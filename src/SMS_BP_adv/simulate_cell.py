import json
import os
import pickle
import random
from typing import Union, TypedDict, List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import skimage as skimage
from PIL import Image
from scipy.linalg import fractional_matrix_power

from .utils.decorators import cache
from .montecarlo.probability_functions import multiple_top_hat_probability as pf
from .motion.track_gen import Track_generator as sf
from .config.config_schema import SimulationConfig
from .config.json_validator_converter import (
    load_validate_and_convert,
    validate_and_convert,
)
from .utils.util_functions import (
    convert_arrays_to_lists,
    convert_lists_to_arrays,
    save_tiff,
    sub_segment,
    make_directory_structure,
)

# Type definitions for better clarity
class TrackData(TypedDict):
    frames: List[int]
    xy: List[List[float]]

class PointsPerTime(TypedDict):
    time: List[List[float]]

@dataclass
class TrackParameters:
    """Parameters for track generation"""
    track_lengths: np.ndarray
    starting_frames: np.ndarray
    initial_positions: np.ndarray
    total_time: int

class Simulate_cells:
    def __init__(self, init_dict_json: dict | str | SimulationConfig):
        """Initialize a cell simulator with the given configuration.
        
        Parameters
        ----------
        init_dict_json : Union[dict, str, SimulationConfig]
            Either:
            - A dictionary containing simulation parameters
            - A path to a JSON config file
            - A SimulationConfig object
            
        See sim_config.md for detailed parameter specifications.
        
        Example
        -------
        >>> simulator = Simulate_cells("config.json")
        >>> simulator = Simulate_cells({"Global_Parameters": {...}})
        """
        # Load and validate configuration
        self.simulation_config = self._load_configuration(init_dict_json)
        self.simulation_config.make_array()

        # Initialize time-related parameters
        self._initialize_time_parameters()
        
        # Convert units for various parameters
        self._initialize_unit_conversions()
        
        # Handle transition matrices if enabled
        if self.simulation_config.Track_Parameters.allow_transition_probability:
            self._initialize_transition_matrices()

    def _load_configuration(self, config_source: dict | str | SimulationConfig) -> SimulationConfig:
        """Load and validate the simulation configuration."""
        if isinstance(config_source, str):
            return load_validate_and_convert(config_source)
        elif isinstance(config_source, dict):
            return validate_and_convert(config_source)
        return config_source

    def _initialize_time_parameters(self) -> None:
        """Initialize all time-related simulation parameters."""
        # Store the times
        self.frame_count = self.simulation_config.Global_Parameters.frame_count
        self.interval_time = int(self.simulation_config.Global_Parameters.interval_time)
        self.oversample_motion_time = int(self.simulation_config.Global_Parameters.oversample_motion_time)
        self.exposure_time = int(self.simulation_config.Global_Parameters.exposure_time)
        
        # Calculate total time and track length
        self.total_time = self._convert_frame_to_time(
            self.frame_count, self.exposure_time, self.interval_time, self.oversample_motion_time
        )
        self.track_length_mean = self._convert_frame_to_time(
            self.simulation_config.Track_Parameters.track_length_mean,
            self.exposure_time, self.interval_time, self.oversample_motion_time
        )

    def _initialize_unit_conversions(self) -> None:
        """Convert all parameters to appropriate units."""
        # Update diffusion coefficients
        self.track_diffusion_updated = self._update_units(
            self.simulation_config.Track_Parameters.diffusion_coefficient,
            "um^2/s", "pix^2/(oversample_motion_time)ms)"
        )
        self.condensate_diffusion_updated = self._update_units(
            self.simulation_config.Condensate_Parameters.diffusion_coefficient,
            "um^2/s", "pix^2/(oversample_motion_time)ms)"
        )
        
        # Update spatial parameters
        self.pixel_size_pix = self._update_units(
            self.simulation_config.Global_Parameters.pixel_size, "um", "pix"
        )
        self.axial_detection_range_pix = self._update_units(
            self.simulation_config.Global_Parameters.axial_detection_range, "um", "pix"
        )
        self.psf_sigma_pix = self._update_units(
            self.simulation_config.Global_Parameters.psf_sigma, "um", "pix"
        )

    def _initialize_transition_matrices(self) -> None:
        """Initialize transition matrices for diffusion and hurst parameters."""
        self.transition_matrix_time_step = self.simulation_config.Track_Parameters.transition_matrix_time_step
        
        # Validate matrix dimensions
        self._validate_transition_matrices()
        
        # Initialize diffusion transition matrix
        self.diffusion_transition_matrix = self._compute_transition_matrix(
            self.simulation_config.Track_Parameters.diffusion_transition_matrix,
            len(self.simulation_config.Track_Parameters.diffusion_coefficient)
        )
        
        # Initialize hurst transition matrix
        self.hurst_transition_matrix = self._compute_transition_matrix(
            self.simulation_config.Track_Parameters.hurst_transition_matrix,
            len(self.simulation_config.Track_Parameters.hurst_exponent)
        )

    def _convert_frame_to_time(
        self,
        frame: int,
        exposure_time: int,
        interval_time: int,
        oversample_motion_time: int,
    ) -> int:
        """Convert the frame number to time.
        
        Parameters
        ----------
        frame : int
            frame number
        exposure_time : int
            exposure time
        interval_time : int
            interval time
        oversample_motion_time : int
            oversample motion time
        
        Returns
        -------
        int
            time in the appropriate units
        """
        return int((frame * (exposure_time + interval_time)) / oversample_motion_time)

    def _update_units(
        self, unit: np.ndarray | float | int, orig_type: str, update_type: str
    ) -> float | np.ndarray | None:
        """Update the unit from one type to another.
        
        Handles various unit conversions needed for simulation, including:
        - Length units (nm, μm, pixels)
        - Time units (ms, s)
        - Diffusion coefficients (μm²/s to pixel²/ms)
        
        Parameters
        ----------
        unit : Union[float, np.ndarray, int]
            Value(s) to be converted. Can be single value or array.
        orig_type : str
            Original unit type. Supported values:
            - "nm", "um", "pix" (length units)
            - "ms", "s" (time units)
            - "um^2/s" (diffusion units)
        update_type : str
            Target unit type. Must be compatible with orig_type.
            
        Returns
        -------
        Union[float, np.ndarray, None]
            Converted value(s) in new units
            Returns None if conversion is not supported
            
        Examples
        --------
        >>> # Convert 1 μm to pixels
        >>> pixels = simulator._update_units(1.0, "um", "pix")
        
        >>> # Convert diffusion coefficient
        >>> d_conv = simulator._update_units(0.1, "um^2/s", "pix^2/(oversample_motion_time)ms)")
        
        Notes
        -----
        - Length conversions use pixel_size from configuration
        - Time conversions use standard ms/s relationships
        - Diffusion coefficient conversions combine both spatial and temporal scaling
        """
        unit = np.array(unit)
        if orig_type == "nm":
            if update_type == "pix":
                return unit / self.simulation_config.Global_Parameters.pixel_size
        elif orig_type == "pix":
            if update_type == "nm":
                return unit * self.simulation_config.Global_Parameters.pixel_size
        elif orig_type == "ms":
            if update_type == "s":
                return unit / 1000.0
        elif orig_type == "s":
            if update_type == "ms":
                return unit * 1000.0
        elif orig_type == "um^2/s":
            if update_type == "pix^2/(oversample_motion_time)ms)":
                return (
                    unit
                    * (1.0 / (self.simulation_config.Global_Parameters.pixel_size**2))
                    * (
                        self.simulation_config.Global_Parameters.oversample_motion_time
                        / 1000.0
                    )
                )
        if orig_type == "um":
            if update_type == "pix":
                return unit / self.simulation_config.Global_Parameters.pixel_size

    def _check_init_dict(self) -> bool:
        """Docstring for _check_init_dict: check the init_dict for the required keys, and if they are consistent with other keys

        Parameters:
        -----------
        None

        Returns:
        --------
        bool: True if the init_dict is correct

        Raises:
        -------
        InitializationKeys: if the init_dict does not have the required keys
        InitializationValues: if the init_dict values are not consistent with each other
        """
        # check if the init_dict has the required keys
        # TODO
        return True

    def _read_json(self, json_file: str) -> dict:
        """Docstring for _read_json: read the json file and return the dictionary
        Parameters:
        -----------
        json_file : str
            path to the json file

        Returns:
        --------
        dict
            dictionary of parameters
        """
        # Open the json file
        with open(json_file) as f:
            # Load the json file
            data = json.load(f)
        # Function to recursively convert lists to NumPy arrays

        def convert_lists_to_arrays(obj):
            if isinstance(obj, list):
                return np.array(obj)
            elif isinstance(obj, dict):
                return {k: convert_lists_to_arrays(v) for k, v in obj.items()}
            else:
                return obj

        # Convert lists to NumPy arrays
        data = convert_lists_to_arrays(data)
        return data

    def _define_space(
        self, dims: tuple[int, int] = (100, 100), movie_frames: int = 500
    ) -> np.ndarray:
        """Docstring for _define_space: make the empty space for simulation
        Parameters:
        -----------
        dims : tuple, Default = (100,100)
            dimensions of the space to be simulated
        movie_frames : int, Default = 500
            number of frames to be simulated
        Returns:
        --------
        space : array-like, shape = (movie_frames,dims[0],dims[1])
            empty space for simulation
        """
        space = np.zeros((movie_frames, dims[0], dims[1]))
        return space

    def _convert_track_dict_points_per_frame(
        self, tracks: dict, movie_frames: int
    ) -> dict:
        """
        Convert the track dictionary into a dictionary of points per frame.

        Parameters:
        -----------
        tracks : dict
            Dictionary of tracks where keys are track numbers and values are track data.
        movie_frames : int
            Number of frames in the movie.

        Returns:
        --------
        dict
            Dictionary where keys are frame numbers and values are lists of (x, y, z) tuples representing points.
        """
        points_per_frame = dict(
            zip(
                [str(i) for i in range(movie_frames)], [[] for i in range(movie_frames)]
            )
        )
        for i, j in tracks.items():
            for k in range(len(j["frames"])):
                points_per_frame[str(j["frames"][k])].append(j["xy"][k])

        return points_per_frame

    def _convert_track_dict_msd(self, tracks: dict) -> dict:
        """
        Convert the track dictionary to a format required for the MSD (mean square displacement) function.

        Parameters:
        -----------
        tracks : dict
            Dictionary of tracks where keys are track numbers and values are track data.

        Returns:
        --------
        dict
            Dictionary where keys are track numbers and values are lists of (x, y, T) tuples representing track points.
        """
        track_msd = {}
        for i, j in tracks.items():
            # make a (x,y,T) tuple for each point
            track_msd[i] = []
            for k in range(len(j["xy"])):
                track_msd[i].append((j["xy"][k][0], j["xy"][k][1], j["frames"][k]))
            # add this to the dictionary
            track_msd[i] = np.array(track_msd[i])
        return track_msd

    def _create_track_pop_dict(self, simulation_cube: np.ndarray) -> tuple[dict, dict]:
        """Create tracks and points per time for cell simulation.
        
        This method orchestrates the entire track generation process, including:
        1. Initializing track parameters
        2. Setting up condensates
        3. Generating initial positions
        4. Creating tracks based on configuration
        
        Parameters
        ----------
        simulation_cube : np.ndarray
            Empty space for the simulation, shape (frames, height, width)
            
        Returns
        -------
        tuple[dict, dict]
            - tracks: Dictionary mapping track IDs to track data
              Format: {track_id: {"frames": [...], "xy": [...]}}
            - points_per_time: Dictionary mapping time points to position lists
              Format: {time_point: [[x1, y1], [x2, y2], ...]}
            
        Examples
        --------
        >>> simulator = Simulate_cells(config)
        >>> space = np.zeros((100, 512, 512))
        >>> tracks, points = simulator._create_track_pop_dict(space)
        >>> print(f"Generated {len(tracks)} tracks")
        """
        # Initialize track parameters
        track_params = self._initialize_track_parameters()
        
        # Initialize condensates and sampling function
        condensates = self._initialize_condensates()
        sampling_function = self._create_sampling_function(condensates)
        
        # Generate initial positions
        initial_positions = self._generate_initial_positions(
            track_params.starting_frames, 
            condensates, 
            sampling_function
        )
        
        # Create tracks based on configuration
        tracks, points_per_time = self._generate_tracks(
            track_params._replace(initial_positions=initial_positions)
        )
        
        return tracks, points_per_time

    def _initialize_track_parameters(self) -> TrackParameters:
        """Initialize basic track parameters."""
        # Get track lengths
        track_lengths = sf.get_lengths(
            track_distribution=self.simulation_config.Track_Parameters.track_distribution,
            track_length_mean=self.track_length_mean,
            total_tracks=self.simulation_config.Track_Parameters.num_tracks,
        )
        
        # Adjust track lengths to not exceed total time
        track_lengths = np.minimum(track_lengths, self.total_time - 1)
        
        # Generate starting frames
        starting_frames = np.array(
            [random.randint(0, self.total_time - length) for length in track_lengths]
        )
        
        return TrackParameters(
            track_lengths=track_lengths,
            starting_frames=starting_frames,
            initial_positions=None,
            total_time=self.total_time
        )

    def _initialize_condensates(self) -> dict:
        """Initialize condensates with given parameters."""
        vol_cell = self._calculate_cell_volume()
        
        return sf.create_condensate_dict(
            initial_centers=np.array(
                self.simulation_config.Condensate_Parameters.initial_centers
            ),
            initial_scale=np.array(
                self.simulation_config.Condensate_Parameters.initial_scale
            ),
            diffusion_coefficient=np.array(self.condensate_diffusion_updated),
            hurst_exponent=np.array(
                self.simulation_config.Condensate_Parameters.hurst_exponent
            ),
            units_time=self._get_time_units(),
            cell_space=self.simulation_config.Cell_Parameters.cell_space,
            cell_axial_range=self.simulation_config.Cell_Parameters.cell_axial_radius,
        )

    def _create_sampling_function(self, condensates: dict) -> pf.multiple_top_hat_probability:
        """Create sampling function for initial positions."""
        vol_cell = self._calculate_cell_volume()
        
        return pf.multiple_top_hat_probability(
            num_subspace=len(self.condensate_diffusion_updated),
            subspace_centers=self.simulation_config.Condensate_Parameters.initial_centers,
            subspace_radius=self.simulation_config.Condensate_Parameters.initial_scale,
            density_dif=self.simulation_config.Condensate_Parameters.density_dif,
            space_size=np.array(vol_cell),
        )

    def _generate_initial_positions(
        self, 
        starting_frames: np.ndarray, 
        condensates: dict,
        sampling_function: pf.multiple_top_hat_probability
    ) -> np.ndarray:
        """Generate initial positions for all tracks."""
        initials = np.zeros((self.simulation_config.Track_Parameters.num_tracks, 3))
        
        for i in range(self.simulation_config.Track_Parameters.num_tracks):
            condensate_positions = self._get_condensate_positions(
                condensates, starting_frames[i]
            )
            sampling_function.update_parameters(subspace_centers=condensate_positions)
            
            initials[i] = self._sample_initial_position(sampling_function)
        
        # Ensure 3D coordinates
        if initials.shape[1] == 2:
            initials = np.hstack((
                initials,
                np.zeros((self.simulation_config.Track_Parameters.num_tracks, 1))
            ))
        
        return initials

    def _generate_tracks(self, track_params: TrackParameters) -> tuple[dict, dict]:
        """Generate tracks based on configuration type.
        
        This method serves as a factory for track generation, selecting the appropriate
        generation method based on configuration parameters:
        - Constant: Uses fixed parameters throughout track
        - No transition: Random but fixed parameters per track
        - Transition: Parameters can change during track lifetime
        
        Parameters
        ----------
        track_params : TrackParameters
            Dataclass containing:
            - track_lengths: Length of each track
            - starting_frames: Starting frame for each track
            - initial_positions: Initial position of each track
            - total_time: Total simulation time
            
        Returns
        -------
        tuple[dict, dict]
            - tracks: Dictionary of track data
              Format: {track_id: {"frames": [...], "xy": [...]}}
            - points_per_time: Dictionary of points at each time
              Format: {time_point: [[x1, y1], [x2, y2], ...]}
            
        Notes
        -----
        Track generation type is determined by:
        1. track_type parameter ("constant")
        2. allow_transition_probability parameter (True/False)
        
        The appropriate generation method is selected automatically based on these parameters.
        """
        track_generator = self._create_track_generator()
        
        if self.simulation_config.Track_Parameters.track_type == "constant":
            return self._generate_constant_tracks(track_generator, track_params)
        elif not self.simulation_config.Track_Parameters.allow_transition_probability:
            return self._generate_no_transition_tracks(track_generator, track_params)
        else:
            return self._generate_transition_tracks(track_generator, track_params)

    def _create_track_generator(self) -> sf.Track_generator:
        """Create track generator with current parameters."""
        return sf.Track_generator(
            cell_space=self.simulation_config.Cell_Parameters.cell_space,
            cell_axial_range=self.simulation_config.Cell_Parameters.cell_axial_radius,
            frame_count=self.frame_count,
            exposure_time=self.exposure_time,
            interval_time=self.interval_time,
            oversample_motion_time=self.oversample_motion_time,
        )

    # Helper methods for track generation based on type
    def _generate_constant_tracks(
        self, 
        track_generator: sf.Track_generator, 
        track_params: TrackParameters
    ) -> tuple[dict, dict]:
        """Generate tracks with constant parameters."""
        tracks = {}
        points_per_time = self._initialize_points_per_time()
        
        for i in range(self.simulation_config.Track_Parameters.num_tracks):
            tracks[i] = track_generator.track_generation_constant(
                track_length=track_params.track_lengths[i],
                initials=track_params.initial_positions[i],
                starting_time=track_params.starting_frames[i],
            )
            self._update_points_per_time(points_per_time, tracks[i])
        
        return tracks, points_per_time

    # Similar methods for no_transition and transition tracks...

    def _create_map(
        self, initial_map: np.ndarray, points_per_frame: dict, axial_function: str
    ) -> np.ndarray:
        """
        Create the simulation map from points per frame.

        Parameters:
        -----------
        initial_map : np.ndarray
            Empty space for the simulation.
        points_per_frame : dict
            Dictionary of points per frame, where keys are frame numbers and values are point coordinates.
        axial_function : str
            The function used to generate axial intensity.

        Returns:
        --------
        np.ndarray
            Updated simulation map.
        """
        for i in range(initial_map.shape[0]):
            # if empty points_per_frame for frame i then do some house keeping
            if len(points_per_frame[str(i)]) == 0:
                abs_axial_position = (
                    1.0
                    * self.simulation_config.Global_Parameters.point_intensity
                    * self.oversample_motion_time
                    / self.exposure_time
                )
                points_per_frame_xyz = np.array(points_per_frame[str(i)])
                points_per_frame_xyz = np.array(points_per_frame_xyz)
            else:
                abs_axial_position = (
                    1.0
                    * self.simulation_config.Global_Parameters.point_intensity
                    * sf.axial_intensity_factor(
                        np.abs(np.array(points_per_frame[str(i)])[:, 2]),
                        detection_range=self.axial_detection_range_pix,
                        func=self.simulation_config.Global_Parameters.axial_function,
                    )
                    * self.oversample_motion_time
                    / self.exposure_time
                )
                points_per_frame_xyz = np.array(points_per_frame[str(i)])[:, :2]
            initial_map[i], _ = sf.generate_map_from_points(
                points_per_frame_xyz,
                point_intensity=abs_axial_position,
                map=initial_map[i],
                movie=True,
                base_noise=self.simulation_config.Global_Parameters.base_noise,
                psf_sigma=self.psf_sigma_pix,
            )
        return initial_map


    def _point_per_time_selection(self, points_per_time: dict) -> dict:
        """
        Select points per frame for the simulation, considering only points during the exposure time.

        Parameters:
        -----------
        points_per_time : dict
            Dictionary of points per time, where keys are frame numbers and values are lists of points.

        Returns:
        --------
        dict
            Dictionary of points per frame, filtered by exposure time.
        """
        # The tracks and points_per_time are already created, so we just need to convert the points_per_time to points_per_frame
        # we only select the points which are in every exposure time ignoring the interval time inbetween the exposure time
        points_per_frame = dict(
            zip(
                [str(i) for i in range(self.frame_count)],
                [[] for i in range(self.frame_count)],
            )
        )
        exposure_counter = 0
        interval_counter = 0
        frame_counter = 0
        for i in range(int(self.total_time)):
            if (
                exposure_counter < int(self.exposure_time / self.oversample_motion_time)
            ) and (
                interval_counter
                <= int(self.interval_time / self.oversample_motion_time)
            ):
                # append the points to the points_per_frame
                if len(points_per_time[str(i)]) != 0:
                    for k in range(len(points_per_time[str(i)])):
                        points_per_frame[str(frame_counter)].append(
                            points_per_time[str(i)][k])
                # increment the exposure_counter
                exposure_counter += 1
            elif (
                exposure_counter
                == int(self.exposure_time / self.oversample_motion_time)
            ) and (
                interval_counter < int(self.interval_time / self.oversample_motion_time)
            ):
                # increment the interval_counter
                interval_counter += 1
            if (
                exposure_counter
                == int(self.exposure_time / self.oversample_motion_time)
            ) and (
                interval_counter
                == int(self.interval_time / self.oversample_motion_time)
            ):
                # reset the counters
                exposure_counter = 0
                interval_counter = 0
                frame_counter += 1
        return points_per_frame

    def get_cell(self) -> dict:
        """Docstring for get_cell: get the cell simulation
        Parameters:
        -----------
        None
        Returns:
        --------
        cell : dict
            dictionary of the cell simulation, keys = "map","tracks","points_per_frame"
        """
        # create the space for the simulation
        space = self._define_space(
            dims=self.simulation_config.Global_Parameters.field_of_view_dim,
            movie_frames=self.frame_count,
        )
        # create the tracks and points_per_time
        tracks, points_per_time = self._create_track_pop_dict(space)
        points_per_frame = self._point_per_time_selection(points_per_time)

        # update the space
        space_updated = self._create_map(
            initial_map=space,
            points_per_frame=points_per_frame,
            axial_function=self.simulation_config.Global_Parameters.axial_function,
        )
        return {
            "map": space_updated,
            "tracks": tracks,
            "points_per_frame": points_per_frame,
        }

    def get_and_save_sim(
        self,
        cd: str,
        img_name: str,
        subsegment_type: str,
        subsegment_num: int,
        **kwargs,
    ) -> None:
        """Docstring for make_directory_structure: make the directory structure for the simulation and save the image + the data and parameters
        Also perform the subsegmentation and save the subsegments in the appropriate directory
        Parameters:
        -----------
        cd : str
            directory to save the simulation
        img_name : str
            name of the image
        img : array-like
            image to be subsegmented
        subsegment_type : str
            type of subsegmentation to be performed, currently only "mean" is supported
        subsegment_num : int
            number of subsegments to be created
        **kwargs : dict
            dictionary of keyword arguments
        KWARGS:
        -------
        data : dict, Default = None
            dictionary of data to be saved, Keys = "map","tracks","points_per_frame" Values = array-like.
            See the return of the function simulate_cell_tracks for more details
        parameters : dict, Default = self.simulation_config
        Returns:
        --------
        none
        """
        # run the sim
        sim = self.get_cell()
        # update the kwargs with the data
        kwargs["data"] = sim
        kwargs["parameters"] = self.simulation_config
        # make the directory structure
        _ = make_directory_structure(
            cd, img_name, sim["map"], subsegment_type, subsegment_num, **kwargs
        )
        return None

    @property
    def condensates(self) -> dict:
        return self._condensates

    @condensates.setter
    def condensates(self, condensates: dict):
        self._condensates = condensates

    @decorators.deprecated(
        "This function is not useful, but is still here for a while in case I need it later"
    )
    def _format_points_per_frame(self, points_per_frame):
        """
        Docstring for _format_points_per_frame: format the points per frame dictionary so that for each key the set of tracks in it are
        converted to a numpy array of N x 2 where N is the total amount of points in that frame. You don't need this function.

        Parameters:
        -----------
        points_per_frame : dict
            keys = str(i) for i in range(self.total_time), values = list of tracks, which are collections of [x,y] coordinates

        Returns:
        --------
        points_per_frame : dict
            keys = str(i) for i in range(movie_frames), values = numpy array of N x 2 where N is the total amount of points in that frame

        """
        for i in points_per_frame.keys():
            # each value is a list of K lists that are composed of M x 2 arrays where M can be different for each list
            # we want to convert this to a numpy array of N x 2 where N is the total amount of points in that frame
            point_holder = []
            for j in points_per_frame[i]:
                point_holder.append(j)
            points_per_frame[i] = np.array(point_holder)
        return points_per_frame

    def _validate_transition_matrices(self) -> None:
        """Validate the dimensions and properties of transition matrices.
        
        Performs two types of validation:
        1. Dimension validation: Ensures matrices match the number of states
        2. Probability validation: Ensures rows sum to 1 (valid probability distributions)
        
        Raises
        ------
        ValueError
            If any of the following conditions are not met:
            - Matrix dimensions don't match number of states
            - Matrix rows don't sum to 1 (within numerical precision)
            - Matrix is not square
            
        Notes
        -----
        For a valid transition matrix:
        - Each row represents probabilities of transitioning from one state
        - All elements must be non-negative
        - Each row must sum to 1
        - Dimensions must match number of possible states
        
        Examples
        --------
        Valid transition matrix for 2 states:
        [[0.9, 0.1],
         [0.2, 0.8]]
        
        Invalid transition matrix (rows don't sum to 1):
        [[0.8, 0.1],
         [0.2, 0.7]]
        """
        diffusion_matrix = self.simulation_config.Track_Parameters.diffusion_transition_matrix
        hurst_matrix = self.simulation_config.Track_Parameters.hurst_transition_matrix
        
        # Check matrix dimensions match number of states
        expected_diffusion_dim = len(self.simulation_config.Track_Parameters.diffusion_coefficient)
        expected_hurst_dim = len(self.simulation_config.Track_Parameters.hurst_exponent)
        
        if diffusion_matrix.shape != (expected_diffusion_dim, expected_diffusion_dim):
            raise ValueError(
                f"Diffusion transition matrix shape {diffusion_matrix.shape} "
                f"doesn't match number of diffusion states {expected_diffusion_dim}"
            )
        
        if hurst_matrix.shape != (expected_hurst_dim, expected_hurst_dim):
            raise ValueError(
                f"Hurst transition matrix shape {hurst_matrix.shape} "
                f"doesn't match number of Hurst states {expected_hurst_dim}"
            )
        
        # Verify each row sums to 1 (probability requirement)
        for matrix, name in [(diffusion_matrix, "Diffusion"), (hurst_matrix, "Hurst")]:
            row_sums = np.sum(matrix, axis=1)
            if not np.allclose(row_sums, 1.0):
                raise ValueError(
                    f"{name} transition matrix rows must sum to 1. "
                    f"Current row sums: {row_sums}"
                )

    def _compute_transition_matrix(
        self, base_matrix: np.ndarray, expected_dim: int
    ) -> np.ndarray:
        """Compute the transition matrix for the given time step.
        
        Parameters
        ----------
        base_matrix : np.ndarray
            The initial transition matrix
        expected_dim : int
            Expected dimension of the matrix
            
        Returns
        -------
        np.ndarray
            Computed transition matrix for the current time step
            
        Notes
        -----
        Uses the fractional matrix power to compute transition probabilities
        for arbitrary time steps.
        """
        # Verify matrix is square
        if base_matrix.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Matrix shape {base_matrix.shape} doesn't match "
                f"expected dimensions ({expected_dim}, {expected_dim})"
            )
        
        # Compute matrix power based on time step
        power = 1.0 / self.transition_matrix_time_step
        return fractional_matrix_power(base_matrix, power)

    def _generate_no_transition_tracks(
        self, 
        track_generator: sf.Track_generator, 
        track_params: TrackParameters
    ) -> tuple[dict, dict]:
        """Generate tracks without state transitions.
        
        Parameters
        ----------
        track_generator : sf.Track_generator
            Track generator instance
        track_params : TrackParameters
            Parameters for track generation
            
        Returns
        -------
        tuple[dict, dict]
            Tracks dictionary and points per time dictionary
        """
        tracks = {}
        points_per_time = self._initialize_points_per_time()
        
        for i in range(self.simulation_config.Track_Parameters.num_tracks):
            # Randomly select diffusion coefficient and hurst exponent indices
            diff_idx = random.randint(
                0, len(self.simulation_config.Track_Parameters.diffusion_coefficient) - 1
            )
            hurst_idx = random.randint(
                0, len(self.simulation_config.Track_Parameters.hurst_exponent) - 1
            )
            
            # Generate track with selected parameters
            tracks[i] = track_generator.track_generation_no_transition(
                track_length=track_params.track_lengths[i],
                initials=track_params.initial_positions[i],
                starting_time=track_params.starting_frames[i],
                diffusion_coefficient=self.track_diffusion_updated[diff_idx],
                hurst_exponent=self.simulation_config.Track_Parameters.hurst_exponent[hurst_idx]
            )
            self._update_points_per_time(points_per_time, tracks[i])
        
        return tracks, points_per_time

    def _generate_transition_tracks(
        self, 
        track_generator: sf.Track_generator, 
        track_params: TrackParameters
    ) -> tuple[dict, dict]:
        """Generate tracks with state transitions.
        
        Parameters
        ----------
        track_generator : sf.Track_generator
            Track generator instance
        track_params : TrackParameters
            Parameters for track generation
            
        Returns
        -------
        tuple[dict, dict]
            Tracks dictionary and points per time dictionary
        """
        tracks = {}
        points_per_time = self._initialize_points_per_time()
        
        for i in range(self.simulation_config.Track_Parameters.num_tracks):
            # Randomly select initial states
            initial_diff_state = random.randint(
                0, len(self.simulation_config.Track_Parameters.diffusion_coefficient) - 1
            )
            initial_hurst_state = random.randint(
                0, len(self.simulation_config.Track_Parameters.hurst_exponent) - 1
            )
            
            # Generate track with transitions
            tracks[i] = track_generator.track_generation_transition(
                track_length=track_params.track_lengths[i],
                initials=track_params.initial_positions[i],
                starting_time=track_params.starting_frames[i],
                diffusion_coefficient=self.track_diffusion_updated,
                hurst_exponent=self.simulation_config.Track_Parameters.hurst_exponent,
                diffusion_transition_matrix=self.diffusion_transition_matrix,
                hurst_transition_matrix=self.hurst_transition_matrix,
                initial_diff_state=initial_diff_state,
                initial_hurst_state=initial_hurst_state,
                transition_matrix_time_step=self.transition_matrix_time_step
            )
            self._update_points_per_time(points_per_time, tracks[i])
        
        return tracks, points_per_time

    def _initialize_points_per_time(self) -> dict:
        """Initialize empty points per time dictionary.
        
        Returns
        -------
        dict
            Empty dictionary with keys for each time point
        """
        return {
            str(i): [] 
            for i in range(self.total_time)
        }

    def _update_points_per_time(self, points_per_time: dict, track: TrackData) -> None:
        """Update points per time dictionary with new track data.
        
        Parameters
        ----------
        points_per_time : dict
            Dictionary to update
        track : TrackData
            Track data to add
        """
        for frame, position in zip(track["frames"], track["xy"]):
            points_per_time[str(frame)].append(position)

    def _calculate_cell_volume(self) -> np.ndarray:
        """Calculate the cell volume based on configuration parameters.
        
        Returns
        -------
        np.ndarray
            Array containing cell dimensions [x, y, z]
        """
        return np.array([
            self.simulation_config.Cell_Parameters.cell_space,
            self.simulation_config.Cell_Parameters.cell_space,
            self.simulation_config.Cell_Parameters.cell_axial_radius * 2
        ])

    def _get_time_units(self) -> float:
        """Get time units for simulation.
        
        Returns
        -------
        float
            Time units in appropriate scale
        """
        return self.oversample_motion_time / 1000.0  # Convert to seconds

    def _get_condensate_positions(self, condensates: dict, time_point: int) -> np.ndarray:
        """Get condensate positions at a specific time point.
        
        Parameters
        ----------
        condensates : dict
            Dictionary containing condensate information
        time_point : int
            Time point to get positions for
        
        Returns
        -------
        np.ndarray
            Array of condensate positions
        """
        positions = []
        for condensate in condensates.values():
            # Get position at specific time point
            if time_point < len(condensate['trajectory']):
                positions.append(condensate['trajectory'][time_point])
            else:
                # Use last known position if time_point exceeds trajectory length
                positions.append(condensate['trajectory'][-1])
        return np.array(positions)

    def _sample_initial_position(
        self, 
        sampling_function: pf.multiple_top_hat_probability
    ) -> np.ndarray:
        """Sample initial position using the provided sampling function.
        
        Repeatedly samples positions until a valid position within cell boundaries
        is found. This ensures all initial positions are physically meaningful.
        
        Parameters
        ----------
        sampling_function : pf.multiple_top_hat_probability
            Function that generates sample positions based on probability distribution
            Must implement .sample() method returning position coordinates
            
        Returns
        -------
        np.ndarray
            Valid position coordinates within cell boundaries
            Format: [x, y] or [x, y, z] depending on dimensionality
            
        Notes
        -----
        - Uses rejection sampling to ensure valid positions
        - May take multiple attempts to find valid position
        - Validates positions using _is_position_valid method
        
        See Also
        --------
        _is_position_valid : Method used to validate sampled positions
        """
        # Sample until we get a valid position
        while True:
            position = sampling_function.sample()
            # Check if position is within cell boundaries
            if self._is_position_valid(position):
                return position

    def _is_position_valid(self, position: np.ndarray) -> bool:
        """Check if a position is within valid cell boundaries.
        
        Parameters
        ----------
        position : np.ndarray
            Position coordinates to check
        
        Returns
        -------
        bool
            True if position is valid, False otherwise
        """
        cell_space = self.simulation_config.Cell_Parameters.cell_space
        cell_axial_radius = self.simulation_config.Cell_Parameters.cell_axial_radius
        
        # Check x and y boundaries
        if not (0 <= position[0] <= cell_space and 0 <= position[1] <= cell_space):
            return False
        
        # Check z boundary if it exists
        if len(position) > 2:
            if not (-cell_axial_radius <= position[2] <= cell_axial_radius):
                return False
        
        return True
