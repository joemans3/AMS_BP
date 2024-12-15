from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import tomli
from pydantic import BaseModel

from ..optics.camera.detectors import CMOSDetector, Detector, EMCCDDetector
from ..optics.camera.quantum_eff import QuantumEfficiency
from ..optics.filters import (
    FilterSet,
    FilterSpectrum,
    create_allow_all_filter,
    create_bandpass_filter,
    create_tophat_filter,
)
from ..optics.lasers.laser_profiles import (
    GaussianBeam,
    LaserParameters,
    LaserProfile,
    WidefieldBeam,
)
from ..optics.psf.psf_engine import PSFEngine, PSFParameters
from ..sample.flurophores.flurophore_schema import (
    Fluorophore,
    SpectralData,
    State,
    StateTransition,
    StateType,
)
from .configmodels import (
    CellParameters,
    CondensateParameters,
    ConfigList,
    GlobalParameters,
    MoleculeParameters,
    OutputParameters,
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
    def __init__(self, config_path: Union[str, Path]):
        # if exists, load config, otherwise raise error
        if not Path(config_path).exists():
            print(f"Configuration file not found: {config_path}")
            self.config_path = None
        else:
            self.config_path = config_path
            self.config = load_config(config_path)
            self.populate_dataclass_schema()

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

    def create_fluorophore_from_config(self, config: Dict[str, Any]) -> Fluorophore:
        """
        Create a fluorophore instance from a configuration dictionary.

        Args:
            config: Dictionary containing the full configuration (typically loaded from TOML)

        Returns:
            Fluorophore: A Fluorophore instance with the loaded configuration
        """
        # Extract fluorophore section
        fluor_config = config.get("fluorophore", {})
        if not fluor_config:
            raise ValueError("No fluorophore configuration found in config")

        # Build states
        states = {}
        for state_name, state_data in fluor_config.get("states", {}).items():
            # Create spectral data if present
            excitation_spectrum = None
            emission_spectrum = None
            extinction_coefficient = None
            quantum_yield = None
            molar_cross_section = None

            if "excitation_spectrum" in state_data:
                excitation_spectrum = SpectralData(
                    wavelengths=state_data["excitation_spectrum"]["wavelengths"],
                    intensities=state_data["excitation_spectrum"]["intensities"],
                )

            if "emission_spectrum" in state_data:
                emission_spectrum = SpectralData(
                    wavelengths=state_data["emission_spectrum"]["wavelengths"],
                    intensities=state_data["emission_spectrum"]["intensities"],
                )

            if "extinction_coefficient" in state_data:
                extinction_coefficient = state_data["extinction_coefficient"]

            if "quantum_yield" in state_data:
                quantum_yield = state_data["quantum_yield"]
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
            )
            states[state.name] = state

        # Build transitions
        transitions = {}
        for _, trans_data in fluor_config.get("transitions", {}).items():
            if trans_data.get("photon_dependent", False):
                transition = StateTransition(
                    from_state=trans_data["from_state"],
                    to_state=trans_data["to_state"],
                    activation_spectrum=SpectralData(
                        wavelengths=trans_data.get("activation_spectrum")[
                            "wavelengths"
                        ],
                        intensities=trans_data.get("activation_spectrum")[
                            "intensities"
                        ],
                    ),
                    activation_extinction_coefficient_lambda_val=trans_data.get(
                        "activation_spectrum"
                    )["activation_extinction_coefficient"],
                    activation_extinction_coefficient=None,
                    activation_cross_section=None,
                    base_rate=None,
                )
            else:
                transition = StateTransition(
                    from_state=trans_data["from_state"],
                    to_state=trans_data["to_state"],
                    base_rate=trans_data.get("base_rate", None),
                    activation_spectrum=None,
                    activation_extinction_coefficient_lambda_val=None,
                    activation_extinction_coefficient=None,
                    activation_cross_section=None,
                )
            transitions[transition.from_state + transition.to_state] = transition

        # Create and return fluorophore
        return Fluorophore(
            name=fluor_config["name"], states=states, transitions=transitions
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
                wavelength=wavelength,
                numerical_aperture=float(params_config["numerical_aperture"]),
                pixel_size=pixel_size,
                z_step=float(params_config["z_step"]) if z_step is None else z_step,
                refractive_index=float(params_config.get("refractive_index", 1.0)),
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
        filters_config = config.get("filters", {})
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

    def get_configs(self) -> Dict[str, Any]:
        # base config
        base_config = ConfigList(
            CellParameters=self.cell_params,
            MoleculeParameters=self.molecule_params,
            GlobalParameters=self.global_params,
            CondensateParameters=self.condensate_params,
            OutputParameters=self.output_params,
        )

        # fluorophore config
        fluorophore = self.create_fluorophore_from_config(self.config)
        # psf config
        psf, psf_config = self.create_psf_from_config(self.config)
        # lasers config
        lasers = self.create_lasers_from_config(self.config)
        # filters config
        filters = self.create_filter_set_from_config(self.config)
        # detector config
        detector, qe = self.create_detector_from_config(self.config)

        sim_config_dict = {
            "config": base_config,
            "fluorophore": fluorophore,
            "psf": psf,
            "psf_config": psf_config,
            "lasers": lasers,
            "filters": filters,
            "detector": detector,
            "qe": qe,
        }
        return sim_config_dict
