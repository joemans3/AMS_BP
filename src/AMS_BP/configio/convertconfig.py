from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import tomli
from pydantic import BaseModel

from ..cells import RectangularCell
from ..cells.base_cell import BaseCell
from ..motion import Track_generator, create_condensate_dict
from ..motion.track_gen import (
    _convert_tracks_to_trajectory,
    _generate_constant_tracks,
    _generate_no_transition_tracks,
    _generate_transition_tracks,
)
from ..optics.camera.detectors import CMOSDetector, Detector, EMCCDDetector
from ..optics.camera.quantum_eff import QuantumEfficiency
from ..optics.filters import (
    FilterSet,
    FilterSpectrum,
    create_allow_all_filter,
    create_bandpass_filter,
    create_tophat_filter,
)
from ..optics.filters.channels.channelschema import Channels
from ..optics.lasers.laser_profiles import (
    GaussianBeam,
    HiLoBeam,
    LaserParameters,
    LaserProfile,
    WidefieldBeam,
)
from ..optics.psf.psf_engine import PSFEngine, PSFParameters
from ..probabilityfuncs.markov_chain import change_prob_time
from ..probabilityfuncs.probability_functions import (
    generate_points_from_cls as gen_points,
)
from ..probabilityfuncs.probability_functions import multiple_top_hat_probability as tp
from ..sample.flurophores.flurophore_schema import (
    Fluorophore,
    SpectralData,
    State,
    StateTransition,
    StateType,
)
from ..sample.sim_sampleplane import SamplePlane, SampleSpace
from ..sim_microscopy import VirtualMicroscope
from .configmodels import (
    CellParameters,
    CondensateParameters,
    ConfigList,
    GlobalParameters,
    MoleculeParameters,
    OutputParameters,
)
from .experiments import (
    BaseExpConfig,
    TimeSeriesExpConfig,
    timeseriesEXP,
    zseriesEXP,
    zStackExpConfig,
)

FILTERSET_BASE = ["excitation", "emission", "dichroic"]


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and parse a TOML configuration file.

    Args:
        config_path: Path to the TOML configuration file (can be string or Path object)

    Returns:
        Dict[str, Any]: Parsed configuration dictionary

    Raises:
        FileNotFoundError: If the config file doesn't exist
        tomli.TOMLDecodeError: If the TOML file is invalid
    """
    # Convert string path to Path object if necessary
    path = Path(config_path) if isinstance(config_path, str) else config_path

    # Check if file exists
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    # Load and parse TOML file
    try:
        with open(path, "rb") as f:
            return tomli.load(f)
    except tomli.TOMLDecodeError as e:
        raise tomli.TOMLDecodeError(f"Error parsing TOML file {path}: {str(e)}")


class ConfigLoader:
    def __init__(self, config_path: Union[str, Path, dict]):
        # if exists, load config, otherwise raise error
        if isinstance(config_path, dict):
            self.config = config_path
        elif not Path(config_path).exists():
            print(f"Configuration file not found: {config_path}")
            self.config_path = None
        else:
            self.config_path = config_path
            self.config = load_config(config_path)

    def _reload_config(self):
        if self.config_path is not None:
            self.config = load_config(config_path=self.config_path)

    def create_dataclass_schema(
        self, dataclass_schema: type[BaseModel], config: Dict[str, Any]
    ) -> BaseModel:
        """
        Populate a dataclass schema with configuration data.
        """
        return dataclass_schema(**config)

    def populate_dataclass_schema(self) -> None:
        """
        Populate a dataclass schema with configuration data.
        """
        self.global_params = self.create_dataclass_schema(
            GlobalParameters, self.config["Global_Parameters"]
        )
        self.cell_params = self.create_dataclass_schema(
            CellParameters, self.config["Cell_Parameters"]
        )
        self.molecule_params = self.create_dataclass_schema(
            MoleculeParameters, self.config["Molecule_Parameters"]
        )
        self.condensate_params = self.create_dataclass_schema(
            CondensateParameters, self.config["Condensate_Parameters"]
        )
        self.output_params = self.create_dataclass_schema(
            OutputParameters, self.config["Output_Parameters"]
        )

    def create_experiment_from_config(
        self, config: Dict[str, Any]
    ) -> Tuple[BaseExpConfig, Callable]:
        configEXP = deepcopy(config["experiment"])
        if configEXP.get("experiment_type") == "time-series":
            del configEXP["experiment_type"]
            tconfig = TimeSeriesExpConfig(**configEXP)
            callableEXP = timeseriesEXP
        elif configEXP.get("experiment_type") == "z-stack":
            del configEXP["experiment_type"]
            tconfig = zStackExpConfig(**configEXP)
            callableEXP = zseriesEXP
        else:
            raise TypeError("Experiment is not supported")
        return tconfig, callableEXP

    def create_fluorophores_from_config(
        self, config: Dict[str, Any]
    ) -> List[Fluorophore]:
        # Extract fluorophore section
        fluor_config = config.get("fluorophores", {})
        if not fluor_config:
            raise ValueError("No fluorophores configuration found in config")
        num_fluorophores = fluor_config["num_of_fluorophores"]
        fluorophore_names = fluor_config["fluorophore_names"]
        fluorophores = []
        for i in range(num_fluorophores):
            fluorophores.append(
                self.create_fluorophore_from_config(fluor_config[fluorophore_names[i]])
            )
        return fluorophores

    def create_fluorophore_from_config(self, config: Dict[str, Any]) -> Fluorophore:
        """
        Create a fluorophore instance from a configuration dictionary.

        Args:
            config: Dictionary containing the full configuration (typically loaded from TOML)

        Returns:
            Fluorophore: A Fluorophore instance with the loaded configuration
        """
        # Extract fluorophore section
        fluor_config = config
        if not fluor_config:
            raise ValueError("No fluorophore configuration found.")

        # Build states
        states = {}
        for state_name, state_data in fluor_config.get("states", {}).items():
            # Create spectral data if present
            excitation_spectrum = (
                SpectralData(
                    wavelengths=state_data.get("excitation_spectrum", {}).get(
                        "wavelengths", []
                    ),
                    intensities=state_data.get("excitation_spectrum", {}).get(
                        "intensities", []
                    ),
                )
                if "excitation_spectrum" in state_data
                else None
            )

            emission_spectrum = (
                SpectralData(
                    wavelengths=state_data.get("emission_spectrum", {}).get(
                        "wavelengths", []
                    ),
                    intensities=state_data.get("emission_spectrum", {}).get(
                        "intensities", []
                    ),
                )
                if "emission_spectrum" in state_data
                else None
            )

            extinction_coefficient = state_data.get("extinction_coefficient")
            quantum_yield = state_data.get("quantum_yield")
            molar_cross_section = state_data.get("molar_cross_section")
            fluorescent_lifetime = state_data.get("fluorescent_lifetime")

            # Create state
            state = State(
                name=state_data["name"],
                state_type=StateType(state_data["state_type"]),
                excitation_spectrum=excitation_spectrum,
                emission_spectrum=emission_spectrum,
                quantum_yield_lambda_val=quantum_yield,
                extinction_coefficient_lambda_val=extinction_coefficient,
                molar_cross_section=molar_cross_section,
                quantum_yield=None,
                extinction_coefficient=None,
                fluorescent_lifetime=fluorescent_lifetime,
            )
            states[state.name] = state

        initial_state = None
        state_list = []
        for state in states.values():
            state_list.append(state.name)
            if state.name == fluor_config["initial_state"]:
                initial_state = state

        if initial_state is None:
            raise ValueError(
                f"Inital state must be a valid name from the provided states: {state_list}."
            )

        # Build transitions
        transitions = {}
        for _, trans_data in fluor_config.get("transitions", {}).items():
            if trans_data.get("photon_dependent", False):
                transition = StateTransition(
                    from_state=trans_data["from_state"],
                    to_state=trans_data["to_state"],
                    spectrum=SpectralData(
                        wavelengths=trans_data.get("spectrum")["wavelengths"],
                        intensities=trans_data.get("spectrum")["intensities"],
                    ),
                    extinction_coefficient_lambda_val=trans_data.get("spectrum")[
                        "extinction_coefficient"
                    ],
                    extinction_coefficient=None,
                    cross_section=None,
                    base_rate=None,
                    quantum_yield=trans_data.get("spectrum")["quantum_yield"],
                )
            else:
                transition = StateTransition(
                    from_state=trans_data["from_state"],
                    to_state=trans_data["to_state"],
                    base_rate=trans_data.get("base_rate", None),
                    spectrum=None,
                    extinction_coefficient_lambda_val=None,
                    extinction_coefficient=None,
                    cross_section=None,
                    quantum_yield=None,
                )
            transitions[transition.from_state + transition.to_state] = transition

        # Create and return fluorophore
        return Fluorophore(
            name=fluor_config["name"],
            states=states,
            transitions=transitions,
            initial_state=initial_state,
        )

    def create_psf_from_config(
        self, config: Dict[str, Any]
    ) -> Tuple[Callable, Dict[str, Any]]:
        """
        Create a PSF engine instance from a configuration dictionary.

        Args:
            config: Dictionary containing the full configuration (typically loaded from TOML)

        Returns:
            Tuple[Callable, Optional[Dict]]: A tuple containing:
                - Partial_PSFEngine partial funcion of PSFEngine. Called as f(wavelength, z_step)
                    - Parameters:
                        - wavelength (int, float) in nm
                            - wavelength of the emitted light from the sample after emission filters
                        - z_step (int, float) in um
                            - z_step used to parameterize the psf grid.
                - Additional PSF-specific parameters (like custom path if specified)
        """
        # Extract PSF section
        psf_config = config.get("psf", {})
        if not psf_config:
            raise ValueError("No PSF configuration found in config")

        # Extract parameters section
        params_config = psf_config.get("parameters", {})
        if not params_config:
            raise ValueError("No PSF parameters found in config")
        pixel_size = self._find_pixel_size(
            config["camera"]["magnification"], config["camera"]["pixel_detector_size"]
        )

        def Partial_PSFengine(
            wavelength: int | float, z_step: Optional[int | float] = None
        ):
            # Create PSFParameters instance
            parameters = PSFParameters(
                emission_wavelength=wavelength,
                numerical_aperture=float(params_config["numerical_aperture"]),
                pixel_size=pixel_size,
                z_step=float(params_config["z_step"]) if z_step is None else z_step,
                refractive_index=float(params_config.get("refractive_index", 1.0)),
                pinhole_diameter=params_config.get("pinhole_diameter", None),
            )

            # Create PSF engine
            psf_engine = PSFEngine(parameters)
            return psf_engine

        # Extract additional configuration
        additional_config = {
            "type": psf_config.get("type", "gaussian"),
            "custom_path": psf_config.get("custom_path", ""),
        }

        return Partial_PSFengine, additional_config

    @staticmethod
    def _find_pixel_size(magnification: float, pixel_detector_size: float) -> float:
        return pixel_detector_size / magnification

    def create_laser_from_config(
        self, laser_config: Dict[str, Any], preset: str
    ) -> LaserProfile:
        """
        Create a laser profile instance from a configuration dictionary.

        Args:
            laser_config: Dictionary containing the laser configuration
            preset: Name of the laser preset (e.g., 'blue', 'green', 'red')

        Returns:
            LaserProfile: A LaserProfile instance with the loaded configuration
        """
        # Extract laser parameters
        params_config = laser_config.get("parameters", {})
        if not params_config:
            raise ValueError(f"No parameters found for laser: {preset}")

        # Create LaserParameters instance
        parameters = LaserParameters(
            power=float(params_config["power"]),
            wavelength=float(params_config["wavelength"]),
            beam_width=float(params_config["beam_width"]),
            numerical_aperture=float(params_config.get("numerical_aperture")),
            refractive_index=float(params_config.get("refractive_index", 1.0)),
        )

        # Create appropriate laser profile based on type
        laser_type = laser_config.get("type", "gaussian").lower()

        if laser_type == "gaussian":
            return GaussianBeam(parameters)
        if laser_type == "widefield":
            return WidefieldBeam(parameters)
        if laser_type == "hilo":
            try:
                params_config.get("inclination_angle")
            except KeyError:
                raise KeyError("HiLo needs inclination angle. Currently not provided")
            return HiLoBeam(parameters, float(params_config["inclination_angle"]))
        else:
            raise ValueError(f"Unknown laser type: {laser_type}")

    def create_lasers_from_config(
        self, config: Dict[str, Any]
    ) -> Dict[str, LaserProfile]:
        """
        Create multiple laser profile instances from a configuration dictionary.

        Args:
            config: Dictionary containing the full configuration (typically loaded from TOML)

        Returns:
            Dict[str, LaserProfile]: Dictionary mapping laser names to their profile instances
        """
        # Extract lasers section
        lasers_config = config.get("lasers", {})
        if not lasers_config:
            raise ValueError("No lasers configuration found in config")

        # Get active lasers
        active_lasers = lasers_config.get("active", [])
        if not active_lasers:
            raise ValueError("No active lasers specified in configuration")

        # Create laser profiles for each active laser
        laser_profiles = {}
        for laser_name in active_lasers:
            laser_config = lasers_config.get(laser_name)
            if not laser_config:
                raise ValueError(f"Configuration not found for laser: {laser_name}")

            laser_profiles[laser_name] = self.create_laser_from_config(
                laser_config, laser_name
            )

        return laser_profiles

    def create_filter_spectrum_from_config(
        self, filter_config: Dict[str, Any]
    ) -> FilterSpectrum:
        """
        Create a filter spectrum from configuration dictionary.

        Args:
            filter_config: Dictionary containing filter configuration

        Returns:
            FilterSpectrum: Created filter spectrum instance
        """
        filter_type = filter_config.get("type", "").lower()

        if filter_type == "bandpass":
            return create_bandpass_filter(
                center_wavelength=float(filter_config["center_wavelength"]),
                bandwidth=float(filter_config["bandwidth"]),
                transmission_peak=float(filter_config.get("transmission_peak", 0.95)),
                points=int(filter_config.get("points", 1000)),
                name=filter_config.get("name"),
            )
        elif filter_type == "tophat":
            return create_tophat_filter(
                center_wavelength=float(filter_config["center_wavelength"]),
                bandwidth=float(filter_config["bandwidth"]),
                transmission_peak=float(filter_config.get("transmission_peak", 0.95)),
                edge_steepness=float(filter_config.get("edge_steepness", 5.0)),
                points=int(filter_config.get("points", 1000)),
                name=filter_config.get("name"),
            )
        elif filter_type == "allow_all":
            return create_allow_all_filter(
                points=int(filter_config.get("points", 1000)),
                name=filter_config.get("name"),
            )

        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

    def create_filter_set_from_config(self, config: Dict[str, Any]) -> FilterSet:
        """
        Create a filter set from configuration dictionary.

        Args:
            config: Dictionary containing the full configuration (typically loaded from TOML)

        Returns:
            FilterSet: Created filter set instance
        """
        # Extract filters section
        filters_config = config
        if not filters_config:
            raise ValueError("No filters configuration found in config")

        missing = []
        for base_filter in FILTERSET_BASE:
            if base_filter not in filters_config:
                print(f"Missing {base_filter} filter in filter set; using base config")
                missing.append(base_filter)

        if missing:
            for base_filter in missing:
                filters_config[base_filter] = {
                    "type": "allow_all",
                    "points": 1000,
                    "name": f"{base_filter} filter",
                }

        # Create filter components
        excitation = self.create_filter_spectrum_from_config(
            filters_config["excitation"]
        )
        emission = self.create_filter_spectrum_from_config(filters_config["emission"])
        dichroic = self.create_filter_spectrum_from_config(filters_config["dichroic"])

        # Create filter set
        return FilterSet(
            name=filters_config.get("filter_set_name", "Custom Filter Set"),
            excitation=excitation,
            emission=emission,
            dichroic=dichroic,
        )

    def create_channels(self, config: Dict[str, Any]) -> Channels:
        # Extract channel section
        channel_config = config.get("channels", {})
        if not channel_config:
            raise ValueError("No channels configuration found in config")
        channel_filters = []
        channel_num = int(channel_config.get("num_of_channels"))
        channel_names = channel_config.get("channel_names")
        split_eff = channel_config.get("split_efficiency")
        for i in range(channel_num):
            channel_filters.append(
                self.create_filter_set_from_config(
                    channel_config.get("filters").get(channel_names[i])
                )
            )
        channels = Channels(
            filtersets=channel_filters,
            num_channels=channel_num,
            splitting_efficiency=split_eff,
            names=channel_names,
        )
        return channels

    def create_quantum_efficiency_from_config(
        self, qe_data: List[List[float]]
    ) -> QuantumEfficiency:
        """
        Create a QuantumEfficiency instance from configuration data.

        Args:
            qe_data: List of [wavelength, efficiency] pairs

        Returns:
            QuantumEfficiency: Created quantum efficiency instance
        """
        # Convert list of pairs to dictionary
        wavelength_qe = {pair[0]: pair[1] for pair in qe_data}
        return QuantumEfficiency(wavelength_qe=wavelength_qe)

    def create_detector_from_config(
        self, config: Dict[str, Any]
    ) -> Tuple[Detector, QuantumEfficiency]:
        """
        Create a detector instance from a configuration dictionary.

        Args:
            config: Dictionary containing the full configuration (typically loaded from TOML)

        Returns:
            Tuple[Detector, QuantumEfficiency]: A tuple containing:
                - Detector instance with the loaded configuration
                - QuantumEfficiency instance for the detector
        """
        # Extract camera section
        camera_config = config.get("camera", {})
        if not camera_config:
            raise ValueError("No camera configuration found in config")

        # Create quantum efficiency curve
        qe_data = camera_config.get("quantum_efficiency", [])
        quantum_efficiency = self.create_quantum_efficiency_from_config(qe_data)

        pixel_size = self._find_pixel_size(
            camera_config["magnification"], camera_config["pixel_detector_size"]
        )
        # Extract common parameters
        common_params = {
            "pixel_size": pixel_size,
            "dark_current": float(camera_config["dark_current"]),
            "readout_noise": float(camera_config["readout_noise"]),
            "pixel_count": tuple([int(i) for i in camera_config["pixel_count"]]),
            "bit_depth": int(camera_config.get("bit_depth", 16)),
            "sensitivity": float(camera_config.get("sensitivity", 1.0)),
            "pixel_detector_size": float(camera_config["pixel_detector_size"]),
            "magnification": float(camera_config["magnification"]),
            "base_adu": int(camera_config["base_adu"]),
            "binning_size": int(camera_config["binning_size"]),
        }

        # Create appropriate detector based on type
        camera_type = camera_config.get("type", "").upper()

        if camera_type == "CMOS":
            detector = CMOSDetector(**common_params)
        elif camera_type == "EMCCD":
            # Extract EMCCD-specific parameters
            em_params = {
                "em_gain": float(camera_config.get("em_gain", 300)),
                "clock_induced_charge": float(
                    camera_config.get("clock_induced_charge", 0.002)
                ),
            }
            detector = EMCCDDetector(
                **common_params,
                em_gain=em_params["em_gain"],
                clock_induced_charge=em_params["clock_induced_charge"],
            )
        else:
            raise ValueError(f"Unsupported camera type: {camera_type}")

        return detector, quantum_efficiency

    def duration_time_validation_experiments(self, configEXP) -> bool:
        if configEXP.exposure_time:
            if len(configEXP.z_position) * (
                configEXP.exposure_time + configEXP.interval_time
            ) > self.config["Global_Parameters"]["cycle_count"] * (
                self.config["Global_Parameters"]["exposure_time"]
                + self.config["Global_Parameters"]["interval_time"]
            ):
                print(
                    f"Z-series parameters overriding the set Global_parameters. cycle_count: {len(configEXP.z_position)}, exposure_time: {configEXP.exposure_time}, and interval_time: {configEXP.interval_time}."
                )
                self.config["Global_Parameters"]["cycle_count"] = len(
                    configEXP.z_position
                )
                self.config["Global_Parameters"]["exposure_time"] = (
                    configEXP.exposure_time
                )
                self.config["Global_Parameters"]["interval_time"] = (
                    configEXP.interval_time
                )

                return False
            else:
                return True
        else:
            return True

    def setup_microscope(self) -> dict:
        # config of experiment

        configEXP, funcEXP = self.create_experiment_from_config(config=self.config)
        self.duration_time_validation_experiments(configEXP)
        # find the larger of the two duration times.
        # base config
        self.populate_dataclass_schema()
        base_config = ConfigList(
            CellParameters=self.cell_params,
            MoleculeParameters=self.molecule_params,
            GlobalParameters=self.global_params,
            CondensateParameters=self.condensate_params,
            OutputParameters=self.output_params,
        )

        # fluorophore config
        fluorophores = self.create_fluorophores_from_config(self.config)
        # psf config
        psf, psf_config = self.create_psf_from_config(self.config)
        # lasers config
        lasers = self.create_lasers_from_config(self.config)
        # channels config
        channels = self.create_channels(self.config)
        # detector config
        detector, qe = self.create_detector_from_config(self.config)

        # make cell
        cell = make_cell(cell_params=base_config.CellParameters)

        # make initial sample plane
        sample_plane = make_sample(
            global_params=base_config.GlobalParameters,
            cell_params=base_config.CellParameters,
        )

        # make condensates_dict
        condensates_dict = make_condensatedict(
            condensate_params=base_config.CondensateParameters, cell=cell
        )

        # make sampling function
        sampling_functions = make_samplingfunction(
            condensate_params=base_config.CondensateParameters, cell=cell
        )

        # create initial positions
        initial_molecule_positions = gen_initial_positions(
            molecule_params=base_config.MoleculeParameters,
            cell=cell,
            condensate_params=base_config.CondensateParameters,
            sampling_functions=sampling_functions,
        )

        # create the track generator
        track_generators = create_track_generator(
            global_params=base_config.GlobalParameters, cell=cell
        )

        # get all the tracks
        tracks, points_per_time = get_tracks(
            molecule_params=base_config.MoleculeParameters,
            global_params=base_config.GlobalParameters,
            initial_positions=initial_molecule_positions,
            track_generator=track_generators,
        )

        # add tracks to sample
        sample_plane = add_tracks_to_sample(
            tracks=tracks, sample_plane=sample_plane, fluorophore=fluorophores
        )

        vm = VirtualMicroscope(
            camera=(detector, qe),
            sample_plane=sample_plane,
            lasers=lasers,
            channels=channels,
            psf=psf,
            config=base_config,
        )
        return_dict = {
            "microscope": vm,
            "base_config": base_config,
            "psf": psf,
            "psf_config": psf_config,
            "channels": channels,
            "lasers": lasers,
            "sample_plane": sample_plane,
            "tracks": tracks,
            "points_per_time": points_per_time,
            "condensate_dict": condensates_dict,
            "cell": cell,
            "experiment_config": configEXP,
            "experiment_func": funcEXP,
        }
        return return_dict


def make_cell(cell_params) -> BaseCell:
    # make cell
    cell_origin = (cell_params.cell_space[0][0], cell_params.cell_space[1][0])
    cell_dimensions = (
        cell_params.cell_space[0][1] - cell_params.cell_space[0][0],
        cell_params.cell_space[1][1] - cell_params.cell_space[1][0],
        cell_params.cell_axial_radius * 2,
    )
    cell = RectangularCell(origin=cell_origin, dimensions=cell_dimensions)

    return cell


def make_sample(global_params, cell_params) -> SamplePlane:
    sample_space = SampleSpace(
        x_max=global_params.sample_plane_dim[0],
        y_max=global_params.sample_plane_dim[1],
        z_max=cell_params.cell_axial_radius,
        z_min=-cell_params.cell_axial_radius,
    )

    # total time
    totaltime = int(
        global_params.cycle_count
        * (global_params.exposure_time + global_params.interval_time)
    )
    # initialize sample plane
    sample_plane = SamplePlane(
        sample_space=sample_space,
        fov=(
            (0, global_params.sample_plane_dim[0]),
            (0, global_params.sample_plane_dim[1]),
            (-cell_params.cell_axial_radius, cell_params.cell_axial_radius),
        ),
        oversample_motion_time=global_params.oversample_motion_time,
        t_end=totaltime,
    )
    return sample_plane


def make_condensatedict(condensate_params, cell) -> List[dict]:
    condensates_dict = []
    for i in range(len(condensate_params.initial_centers)):
        condensates_dict.append(
            create_condensate_dict(
                initial_centers=condensate_params.initial_centers[i],
                initial_scale=condensate_params.initial_scale[i],
                diffusion_coefficient=condensate_params.diffusion_coefficient[i],
                hurst_exponent=condensate_params.hurst_exponent[i],
                cell=cell,
            )
        )
    return condensates_dict


def make_samplingfunction(condensate_params, cell) -> List[Callable]:
    sampling_functions = []
    for i in range(len(condensate_params.initial_centers)):
        sampling_functions.append(
            tp(
                num_subspace=len(condensate_params.initial_centers[i]),
                subspace_centers=condensate_params.initial_centers[i],
                subspace_radius=condensate_params.initial_scale[i],
                density_dif=condensate_params.density_dif[i],
                cell=cell,
            )
        )
    return sampling_functions


def gen_initial_positions(molecule_params, cell, condensate_params, sampling_functions):
    initials = []
    for i in range(len(molecule_params.num_molecules)):
        num_molecules = molecule_params.num_molecules[i]
        initial_positions = gen_points(
            pdf=sampling_functions[i],
            total_points=num_molecules,
            min_x=cell.origin[0],
            max_x=cell.origin[0] + cell.dimensions[0],
            min_y=cell.origin[1],
            max_y=cell.origin[1] + cell.dimensions[1],
            min_z=-cell.dimensions[2] / 2,
            max_z=cell.dimensions[2] / 2,
            density_dif=condensate_params.density_dif[i],
        )
        initials.append(initial_positions)
    return initials


def create_track_generator(global_params, cell):
    totaltime = int(
        global_params.cycle_count
        * (global_params.exposure_time + global_params.interval_time)
    )
    # make track generator
    track_generator = Track_generator(
        cell=cell,
        cycle_count=totaltime / global_params.oversample_motion_time,
        exposure_time=global_params.exposure_time,
        interval_time=global_params.interval_time,
        oversample_motion_time=global_params.oversample_motion_time,
    )
    return track_generator


def get_tracks(molecule_params, global_params, initial_positions, track_generator):
    totaltime = int(
        global_params.cycle_count
        * (global_params.exposure_time + global_params.interval_time)
    )
    tracks_collection = []
    points_per_time_collection = []

    for i in range(len(initial_positions)):
        if molecule_params.track_type[i] == "constant":
            tracks, points_per_time = _generate_constant_tracks(
                track_generator,
                int(totaltime / global_params.oversample_motion_time),
                initial_positions[i],
                0,
            )
        elif molecule_params.allow_transition_probability[i]:
            tracks, points_per_time = _generate_transition_tracks(
                track_generator=track_generator,
                track_lengths=int(totaltime / global_params.oversample_motion_time),
                initial_positions=initial_positions[i],
                starting_frames=0,
                diffusion_parameters=molecule_params.diffusion_coefficient[i],
                hurst_parameters=molecule_params.hurst_exponent[i],
                diffusion_transition_matrix=change_prob_time(
                    molecule_params.diffusion_transition_matrix[i],
                    molecule_params.transition_matrix_time_step[i],
                    global_params.oversample_motion_time,
                ),
                hurst_transition_matrix=change_prob_time(
                    molecule_params.hurst_transition_matrix[i],
                    molecule_params.transition_matrix_time_step[i],
                    global_params.oversample_motion_time,
                ),
                diffusion_state_probability=molecule_params.state_probability_diffusion[
                    i
                ],
                hurst_state_probability=molecule_params.state_probability_hurst[i],
            )
        else:
            tracks, points_per_time = _generate_no_transition_tracks(
                track_generator=track_generator,
                track_lengths=int(totaltime / global_params.oversample_motion_time),
                initial_positions=initial_positions[i],
                starting_frames=0,
                diffusion_parameters=molecule_params.diffusion_coefficient[i],
                hurst_parameters=molecule_params.hurst_exponent[i],
            )

        tracks_collection.append(tracks)
        points_per_time_collection.append(points_per_time)

    return tracks_collection, points_per_time_collection


def add_tracks_to_sample(tracks, sample_plane, fluorophore, ID_counter=0):
    counter = ID_counter
    for track_type in range(len(tracks)):
        for j in tracks[track_type].values():
            sample_plane.add_object(
                object_id=str(counter),
                position=j["xy"][0],
                fluorophore=fluorophore[track_type],
                trajectory=_convert_tracks_to_trajectory(j),
            )
            counter += 1
    return sample_plane
