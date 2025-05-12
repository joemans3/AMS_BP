from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tomli
from pydantic import BaseModel

from ...core.optics.collection_efficiency import (
    collection_efficiency_isotropic_emission,
)
from ..cells import BaseCell, create_cell
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


# Helper function to load config
def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and parse a TOML configuration file.
    """
    path = Path(config_path) if isinstance(config_path, str) else config_path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with open(path, "rb") as f:
            return tomli.load(f)
    except tomli.TOMLDecodeError as e:
        raise tomli.TOMLDecodeError(f"Error parsing TOML file {path}: {str(e)}")


# Function to populate the dataclass schema
def populate_dataclass_schema(
    config: Dict[str, Any],
) -> Tuple[
    GlobalParameters,
    CellParameters,
    MoleculeParameters,
    CondensateParameters,
    OutputParameters,
]:
    global_params = create_dataclass_schema(
        GlobalParameters, config["Global_Parameters"]
    )
    cell_params = create_dataclass_schema(CellParameters, config["Cell_Parameters"])
    molecule_params = create_dataclass_schema(
        MoleculeParameters, config["Molecule_Parameters"]
    )
    condensate_params = create_dataclass_schema(
        CondensateParameters, config["Condensate_Parameters"]
    )
    output_params = create_dataclass_schema(
        OutputParameters, config["Output_Parameters"]
    )

    return global_params, cell_params, molecule_params, condensate_params, output_params


def create_dataclass_schema(
    dataclass_schema: type[BaseModel], config: Dict[str, Any]
) -> BaseModel:
    """Populate a dataclass schema with configuration data."""
    return dataclass_schema(**config)


# Function to create experiment from config
def create_experiment_from_config(
    config: Dict[str, Any],
) -> Tuple[BaseExpConfig, Callable]:
    """Create experiment config and associated callable from configuration."""
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


# Function to create fluorophores from config
def create_fluorophores_from_config(config: Dict[str, Any]) -> List[Fluorophore]:
    fluor_config = config.get("fluorophores", {})
    if not fluor_config:
        raise ValueError("No fluorophores configuration found in config")
    num_fluorophores = fluor_config["num_of_fluorophores"]
    fluorophore_names = fluor_config["fluorophore_names"]
    return [
        create_fluorophore_from_config(fluor_config[fluorophore_names[i]])
        for i in range(num_fluorophores)
    ]


# Function to create a single fluorophore from config
def create_fluorophore_from_config(config: Dict[str, Any]) -> Fluorophore:
    fluor_config = config
    states = create_states_from_config(fluor_config)
    initial_state = get_initial_state(states, fluor_config)
    transitions = create_transitions_from_config(fluor_config)
    return Fluorophore(
        name=fluor_config["name"],
        states=states,
        transitions=transitions,
        initial_state=initial_state,
    )


def create_states_from_config(fluor_config: Dict[str, Any]) -> Dict[str, State]:
    """Create states from fluorophore configuration."""
    states = {}
    for state_name, state_data in fluor_config.get("states", {}).items():
        states[state_name] = create_state(state_data)
    return states


def create_state(state_data: Dict[str, Any]) -> State:
    """Create a single state from configuration."""
    excitation_spectrum = get_spectral_data(state_data, "excitation_spectrum")
    emission_spectrum = get_spectral_data(state_data, "emission_spectrum")
    return State(
        name=state_data["name"],
        state_type=StateType(state_data["state_type"]),
        excitation_spectrum=excitation_spectrum,
        emission_spectrum=emission_spectrum,
        quantum_yield_lambda_val=state_data.get("quantum_yield"),
        extinction_coefficient_lambda_val=state_data.get("extinction_coefficient"),
        fluorescent_lifetime=state_data.get("fluorescent_lifetime"),
    )


def get_spectral_data(state_data: Dict[str, Any], key: str) -> Optional[SpectralData]:
    """Retrieve spectral data for excitation/emission."""
    spectrum_data = state_data.get(key)
    if spectrum_data:
        return SpectralData(
            wavelengths=spectrum_data.get("wavelengths", []),
            intensities=spectrum_data.get("intensities", []),
        )
    return None


def get_initial_state(states: Dict[str, State], fluor_config: Dict[str, Any]) -> State:
    """Get initial state for fluorophore."""
    initial_state = None
    state_list = list(states.keys())
    for state in states.values():
        if state.name == fluor_config["initial_state"]:
            initial_state = state
    if initial_state is None:
        raise ValueError(f"Initial state must be one of: {state_list}.")
    return initial_state


def create_transitions_from_config(
    fluor_config: Dict[str, Any],
) -> Dict[str, StateTransition]:
    """Create state transitions from configuration."""
    transitions = {}
    for _, trans_data in fluor_config.get("transitions", {}).items():
        transitions[trans_data["from_state"] + trans_data["to_state"]] = (
            create_transition(trans_data)
        )
    return transitions


def create_transition(trans_data: Dict[str, Any]) -> StateTransition:
    """Create a single state transition."""
    if trans_data.get("photon_dependent", False):
        return StateTransition(
            from_state=trans_data["from_state"],
            to_state=trans_data["to_state"],
            spectrum=SpectralData(
                wavelengths=trans_data["spectrum"]["wavelengths"],
                intensities=trans_data["spectrum"]["intensities"],
            ),
            extinction_coefficient_lambda_val=trans_data["spectrum"][
                "extinction_coefficient"
            ],
            quantum_yield=trans_data["spectrum"]["quantum_yield"],
        )
    else:
        return StateTransition(
            from_state=trans_data["from_state"],
            to_state=trans_data["to_state"],
            base_rate=trans_data.get("base_rate"),
        )


# Function to create PSF engine from config
def create_psf_from_config(config: Dict[str, Any]) -> Tuple[Callable, Dict[str, Any]]:
    """Create a PSF engine instance from a configuration dictionary."""
    psf_config = config.get("psf", {})
    if not psf_config:
        raise ValueError("No PSF configuration found in config")

    params_config = psf_config.get("parameters", {})
    if not params_config:
        raise ValueError("No PSF parameters found in config")

    pixel_size = find_pixel_size(
        config["camera"]["magnification"], config["camera"]["pixel_detector_size"]
    )

    def Partial_PSFengine(
        wavelength: int | float, z_step: Optional[int | float] = None
    ):
        parameters = PSFParameters(
            emission_wavelength=wavelength,
            numerical_aperture=float(params_config["numerical_aperture"]),
            pixel_size=pixel_size,
            z_step=float(params_config["z_step"]) if z_step is None else z_step,
            refractive_index=float(params_config.get("refractive_index", 1.0)),
            pinhole_diameter=params_config.get("pinhole_diameter", None),
        )
        psf_engine = PSFEngine(parameters)
        return psf_engine

    additional_config = {
        "type": psf_config.get("type", "gaussian"),
        "custom_path": psf_config.get("custom_path", ""),
        "psf_config": psf_config,
    }

    return Partial_PSFengine, additional_config


def create_collection_efficiency(config: Dict[str, Any]) -> float:
    na = config.get("psf").get("parameters").get("numerical_aperture")
    n = config.get("psf").get("parameters").get("refractive_index")
    col_eff = collection_efficiency_isotropic_emission(na=na, n=n)
    return col_eff


# Helper function to find pixel size
def find_pixel_size(magnification: float, pixel_detector_size: float) -> float:
    return pixel_detector_size / magnification


# Function to create a laser from config
def create_laser_from_config(laser_config: Dict[str, Any], preset: str) -> LaserProfile:
    """Create a laser profile instance from a configuration dictionary."""
    params_config = laser_config.get("parameters", {})
    if not params_config:
        raise ValueError(f"No parameters found for laser: {preset}")

    parameters = LaserParameters(
        power=float(params_config["power"]),
        wavelength=float(params_config["wavelength"]),
        beam_width=float(params_config["beam_width"]),
        numerical_aperture=float(params_config.get("numerical_aperture")),
        refractive_index=float(params_config.get("refractive_index", 1.0)),
    )

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


# Function to create lasers from config
def create_lasers_from_config(config: Dict[str, Any]) -> Dict[str, LaserProfile]:
    """Create multiple laser profile instances from a configuration dictionary."""
    lasers_config = config.get("lasers", {})
    if not lasers_config:
        raise ValueError("No lasers configuration found in config")

    active_lasers = lasers_config.get("active", [])
    if not active_lasers:
        raise ValueError("No active lasers specified in configuration")

    laser_profiles = {}
    for laser_name in active_lasers:
        laser_config = lasers_config.get(laser_name)
        if not laser_config:
            raise ValueError(f"Configuration not found for laser: {laser_name}")

        laser_profiles[laser_name] = create_laser_from_config(laser_config, laser_name)

    return laser_profiles


# Function to create filter spectrum from config
def create_filter_spectrum_from_config(filter_config: Dict[str, Any]) -> FilterSpectrum:
    """Create a filter spectrum from configuration dictionary."""
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


# Function to create filter set from config
def create_filter_set_from_config(config: Dict[str, Any]) -> FilterSet:
    """Create a filter set from configuration dictionary."""
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

    excitation = create_filter_spectrum_from_config(filters_config["excitation"])
    emission = create_filter_spectrum_from_config(filters_config["emission"])
    dichroic = create_filter_spectrum_from_config(filters_config["dichroic"])

    return FilterSet(
        name=filters_config.get("filter_set_name", "Custom Filter Set"),
        excitation=excitation,
        emission=emission,
        dichroic=dichroic,
    )


# Function to create channels from config
def create_channels(config: Dict[str, Any]) -> Channels:
    """Create channels from configuration."""
    channel_config = config.get("channels", {})
    if not channel_config:
        raise ValueError("No channels configuration found in config")

    channel_filters = []
    channel_num = int(channel_config.get("num_of_channels"))
    channel_names = channel_config.get("channel_names")
    split_eff = channel_config.get("split_efficiency")

    for i in range(channel_num):
        channel_filters.append(
            create_filter_set_from_config(
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


# Function to create quantum efficiency from config
def create_quantum_efficiency_from_config(
    qe_data: List[List[float]],
) -> QuantumEfficiency:
    """Create a QuantumEfficiency instance from configuration data."""
    wavelength_qe = {pair[0]: pair[1] for pair in qe_data}
    return QuantumEfficiency(wavelength_qe=wavelength_qe)


# Function to create detector from config
def create_detector_from_config(
    config: Dict[str, Any],
) -> Tuple[Detector, QuantumEfficiency]:
    """Create a detector instance from a configuration dictionary."""
    camera_config = config.get("camera", {})
    if not camera_config:
        raise ValueError("No camera configuration found in config")

    qe_data = camera_config.get("quantum_efficiency", [])
    quantum_efficiency = create_quantum_efficiency_from_config(qe_data)

    pixel_size = find_pixel_size(
        camera_config["magnification"], camera_config["pixel_detector_size"]
    )

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

    camera_type = camera_config.get("type", "").upper()

    if camera_type == "CMOS":
        detector = CMOSDetector(**common_params)
    elif camera_type == "EMCCD":
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


# Function to validate the experiment duration time
def duration_time_validation_experiments(configEXP, global_params) -> bool:
    if configEXP.exposure_time:
        if len(configEXP.z_position) * (
            configEXP.exposure_time + configEXP.interval_time
        ) > global_params.cycle_count * (
            global_params.exposure_time + global_params.interval_time
        ):
            print(
                f"Z-series parameters overriding the set Global_parameters. cycle_count: {len(configEXP.z_position)}, exposure_time: {configEXP.exposure_time}, and interval_time: {configEXP.interval_time}."
            )
            global_params.cycle_count = len(configEXP.z_position)
            global_params.exposure_time = configEXP.exposure_time
            global_params.interval_time = configEXP.interval_time

            return False
        else:
            return True
    else:
        return True


# Function to create base config from parameters
def create_base_config(
    global_params, cell_params, molecule_params, condensate_params, output_params
):
    return ConfigList(
        CellParameter=cell_params,
        MoleculeParameter=molecule_params,
        GlobalParameter=global_params,
        CondensateParameter=condensate_params,
        OutputParameter=output_params,
    )


# Function to create sample plane
def make_sample(global_params: GlobalParameters, cell: BaseCell) -> SamplePlane:
    bounds = cell.boundingbox
    sample_space = SampleSpace(
        x_max=global_params.sample_plane_dim[0],
        y_max=global_params.sample_plane_dim[1],
        z_max=bounds[-1],
        z_min=bounds[-2],
    )

    # Total time
    totaltime = int(
        global_params.cycle_count
        * (global_params.exposure_time + global_params.interval_time)
    )

    # Initialize sample plane
    sample_plane = SamplePlane(
        sample_space=sample_space,
        fov=(
            (0, global_params.sample_plane_dim[0]),
            (0, global_params.sample_plane_dim[1]),
            (bounds[-2], bounds[-1]),
        ),
        oversample_motion_time=global_params.oversample_motion_time,
        t_end=totaltime,
    )
    return sample_plane


# Function to create cell
def make_cell(cell_params) -> BaseCell:
    cell = create_cell(cell_params.cell_type, cell_params.params)
    return cell


# Function to create condensates dict
def make_condensatedict(
    condensate_params: CondensateParameters, cell: BaseCell
) -> List[dict]:
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


# Function to create sampling functions
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


# Function to generate initial positions for molecules
def gen_initial_positions(
    molecule_params: MoleculeParameters,
    cell: BaseCell,
    condensate_params: CondensateParameters,
    sampling_functions: List[Callable],
) -> List:
    initials = []
    for i in range(len(molecule_params.num_molecules)):
        num_molecules = molecule_params.num_molecules[i]
        initial_positions = gen_points(
            pdf=sampling_functions[i],
            total_points=num_molecules,
            volume=cell.volume,
            bounds=cell.boundingbox,
            density_dif=condensate_params.density_dif[i],
        )
        initials.append(initial_positions)
    return initials


# Function to create track generator
def create_track_generator(
    global_params: GlobalParameters, cell: BaseCell
) -> Track_generator:
    totaltime = int(
        global_params.cycle_count
        * (global_params.exposure_time + global_params.interval_time)
    )
    track_generator = Track_generator(
        cell=cell,
        total_time=totaltime,
        oversample_motion_time=global_params.oversample_motion_time,
    )
    return track_generator


# Function to get tracks
def get_tracks(
    molecule_params: MoleculeParameters,
    global_params: GlobalParameters,
    initial_positions: List,
    track_generator: Track_generator,
) -> Tuple[List, List]:
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
                    np.array(molecule_params.diffusion_transition_matrix[i]),
                    molecule_params.transition_matrix_time_step[i],
                    global_params.oversample_motion_time,
                ),
                hurst_transition_matrix=change_prob_time(
                    np.array(molecule_params.hurst_transition_matrix[i]),
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


# Function to add tracks to sample plane
def add_tracks_to_sample(
    tracks: List,
    sample_plane: SamplePlane,
    fluorophore: List[Fluorophore],
    ID_counter=0,
) -> SamplePlane:
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


# Function to set up the virtual microscope and return the configuration dictionary
def setup_microscope(config: Dict[str, Any]) -> dict:
    global_params, cell_params, molecule_params, condensate_params, output_params = (
        populate_dataclass_schema(config)
    )
    configEXP, funcEXP = create_experiment_from_config(config=config)
    duration_time_validation_experiments(configEXP, global_params)
    base_config = create_base_config(
        global_params, cell_params, molecule_params, condensate_params, output_params
    )

    # fluorophore config
    fluorophores = create_fluorophores_from_config(config)
    # psf config
    psf, psf_config = create_psf_from_config(config)
    collection_efficiency = create_collection_efficiency(config)
    # lasers config
    lasers = create_lasers_from_config(config)
    # channels config
    channels = create_channels(config)
    # detector config
    detector, qe = create_detector_from_config(config)

    # make cell
    cell = make_cell(base_config.CellParameter)

    # make initial sample plane
    sample_plane = make_sample(base_config.GlobalParameter, cell)

    # make condensates_dict
    condensates_dict = make_condensatedict(base_config.CondensateParameter, cell)

    # make sampling function
    sampling_functions = make_samplingfunction(base_config.CondensateParameter, cell)

    # create initial positions
    initial_molecule_positions = gen_initial_positions(
        base_config.MoleculeParameter,
        cell,
        base_config.CondensateParameter,
        sampling_functions,
    )

    # create the track generator
    track_generators = create_track_generator(base_config.GlobalParameter, cell)

    # get all the tracks
    tracks, points_per_time = get_tracks(
        base_config.MoleculeParameter,
        base_config.GlobalParameter,
        initial_molecule_positions,
        track_generators,
    )

    # add tracks to sample
    sample_plane = add_tracks_to_sample(tracks, sample_plane, fluorophores)

    vm = VirtualMicroscope(
        camera=(detector, qe),
        sample_plane=sample_plane,
        lasers=lasers,
        channels=channels,
        psf=psf,
        config=base_config,
        collection_efficiency=collection_efficiency,
    )

    return {
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
