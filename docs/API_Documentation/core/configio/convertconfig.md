# Virtual Microscope Configuration Parser Documentation

## Overview

This module provides functionality to parse TOML configuration files and set up a virtual microscope simulation environment. It handles all aspects of configuration including global parameters, cell parameters, molecule parameters, condensate parameters, fluorophores, PSF (Point Spread Function), lasers, filters, channels, detectors, and experimental settings.

## Key Components

### Configuration Loading

```python
load_config(config_path: Union[str, Path]) -> Dict[str, Any]
```

Loads and parses a TOML configuration file from the specified path.

### Data Model Schema Population

```python
populate_dataclass_schema(config: Dict[str, Any]) -> Tuple[GlobalParameters, CellParameters, MoleculeParameters, CondensateParameters, OutputParameters]
```

Populates Pydantic schema models from configuration data, returning structured parameter objects.

### Experiment Configuration

```python
create_experiment_from_config(config: Dict[str, Any]) -> Tuple[BaseExpConfig, Callable]
```

Creates an experiment configuration and associated callable function based on the experiment type (time-series or z-stack).

### Fluorophore Configuration

```python
create_fluorophores_from_config(config: Dict[str, Any]) -> List[Fluorophore]
```

Creates a list of fluorophore objects from configuration data, including states and transitions.

### PSF (Point Spread Function) Configuration

```python
create_psf_from_config(config: Dict[str, Any]) -> Tuple[Callable, Dict[str, Any]]
```

Creates a PSF engine function and additional configuration from the config data.

### Laser Configuration

```python
create_lasers_from_config(config: Dict[str, Any]) -> Dict[str, LaserProfile]
```

Creates laser profile instances (Gaussian, Widefield, HiLo) from configuration data.

### Filter Configuration

```python
create_filter_set_from_config(config: Dict[str, Any]) -> FilterSet
```

Creates a filter set (excitation, emission, dichroic) from configuration data.

### Channel Configuration

```python
create_channels(config: Dict[str, Any]) -> Channels
```

Creates channel objects from configuration data.

### Detector Configuration

```python
create_detector_from_config(config: Dict[str, Any]) -> Tuple[Detector, QuantumEfficiency]
```

Creates a detector instance (CMOS or EMCCD) and quantum efficiency from configuration data.

### Cell Creation

```python
create_cell_from_params(cell_params) -> BaseCell
```

Creates a cell object based on cell parameters.

### Sample Plane Creation

```python
create_sample_plane(global_params: GlobalParameters, cell: BaseCell) -> SamplePlane
```

Creates a sample plane object based on global parameters and cell information.

### Condensate Configuration

```python
create_condensates_dict(condensate_params: CondensateParameters, cell: BaseCell) -> List[dict]
```

Creates a list of condensate dictionaries based on condensate parameters.

### Sampling Function Creation

```python
create_sampling_functions(condensate_params, cell) -> List[Callable]
```

Creates sampling functions for initial molecule positions.

### Molecule Position Generation

```python
generate_initial_positions(molecule_params: MoleculeParameters, cell: BaseCell, condensate_params: CondensateParameters, sampling_functions: List[Callable]) -> List
```

Generates initial positions for molecules.

### Track Generation

```python
create_track_generator(global_params: GlobalParameters, cell: BaseCell) -> Track_generator
```

Creates a track generator object for molecule motion.

```python
get_tracks(molecule_params: MoleculeParameters, global_params: GlobalParameters, initial_positions: List, track_generator: Track_generator) -> Tuple[List, List]
```

Generates tracks for molecules based on parameters.

### Sample Creation

```python
add_tracks_to_sample(tracks: List, sample_plane: SamplePlane, fluorophore: List[Fluorophore], ID_counter=0) -> SamplePlane
```

Adds tracks to the sample plane.

### Microscope Setup

```python
setup_microscope(config: Dict[str, Any]) -> dict
```

The main function that orchestrates the entire setup process and returns a dictionary containing all created components including:
- Virtual microscope instance
- Base configuration
- PSF engine and configuration
- Channels
- Lasers
- Sample plane
- Tracks
- Points per time
- Condensate dictionary
- Cell
- Experiment configuration
- Experiment function

## Usage

To use this module, create a TOML configuration file with all necessary parameters and call the `setup_microscope` function:

```python
config = load_config("path/to/config.toml")
microscope_setup = setup_microscope(config)

# Access the virtual microscope instance
vm = microscope_setup["microscope"]

# Run an experiment
experiment_func = microscope_setup["experiment_func"]
experiment_config = microscope_setup["experiment_config"]
results = experiment_func(vm, experiment_config)
```

## Configuration Structure

The TOML configuration file should contain the following sections:
- Global_Parameters: General simulation parameters
- Cell_Parameters: Cell-specific parameters
- Molecule_Parameters: Molecule-specific parameters
- Condensate_Parameters: Condensate-specific parameters
- Output_Parameters: Output and saving parameters
- experiment: Experiment-specific parameters
- fluorophores: Fluorophore definitions
- psf: PSF configuration
- lasers: Laser configurations
- channels: Channel configurations
- camera: Camera and detector configurations
