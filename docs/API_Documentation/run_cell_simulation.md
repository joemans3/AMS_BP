## Overview

This module provides a command-line interface (CLI) for the **SMS_BP** package, which is used for simulating single molecule localization microscopy experiments. The CLI is built using **Typer** and offers two main commands:

1. `config`: Generates a sample configuration file.
2. `runsim`: Runs the cell simulation using a provided configuration file.

The module also utilizes **Rich** for enhanced console output and progress tracking.

---

## Main Components

### 1. `typer_app_asms_bp`

The main Typer application object that defines the CLI interface.

- **Name**: `AMS_BP CLI Tool`
- **Help Text**: 
  ```
  CLI tool to run Advanced Molecule Simulation: ASMS-BP. GitHub: https://github.com/joemans3/AMS_BP.
  [Version: {__version__}]
  ```
- **Short Help**: `CLI tool for AMS_BP.`
- **Rich Markup Mode**: `rich`
- **Pretty Exceptions**: Exceptions are displayed without showing locals.
- **Completion**: Disabled.
- **No Args Help**: Enabled.
- **Context Settings**: Help options are `-h` and `--help`.

---

### 2. `cell_simulation()`

A callback function that displays the version information of the **AMS_BP** package.

- **Output**: 
  ```
  AMS_BP version: [bold]{__version__}[/bold]
  ```

---

### 3. `generate_config()`

A command to generate a sample configuration file for the cell simulation.

- **Options**:
  - `--output_path` or `-o`: Path to the output file. Defaults to the current directory.
  - `--recursive_o` or `-r`: If provided, creates the output directory recursively if it does not exist.

- **Behavior**:
  - Validates the output path.
  - Copies the default configuration file (`sim_config.toml`) to the specified output path.
  - Displays progress using **Rich**.

- **Example Usage**:
  ```bash
  python run_cell_simulation.py config --output_path ./configs --recursive_o
  ```

---

### 4. `run_cell_simulation()`

A command to run the cell simulation using a provided configuration file.

- **Arguments**:
  - `config_file`: Path to the configuration file.

- **Behavior**:
  - Validates the configuration file.
  - Loads the configuration using `ConfigLoader`.
  - Sets up the microscope and experiment configuration.
  - Runs the simulation using the provided configuration.
  - Saves the simulation results using `save_config_frames`.
  - Displays progress and timing information using **Rich**.

- **Example Usage**:
  ```bash
  python run_cell_simulation.py runsim ./configs/sim_config.toml
  ```

---

### 5. `validate_config()`

A helper function to validate the configuration file.

- **Behavior**:
  - Checks for the presence of the `Output_Parameters` section.
  - Ensures that the `output_path` is specified within the `Output_Parameters`.
  - Aborts execution if validation fails, displaying an error message using **Rich**.

---

## Usage

### Generating a Configuration File

To generate a sample configuration file:

```bash
python run_cell_simulation.py config [OPTIONS]
```

### Running a Simulation

To run a simulation using a configuration file:

```bash
python run_cell_simulation.py runsim [CONFIG_FILE]
```

---

## Dependencies

- **Rich**: For enhanced console output and progress tracking.
- **Typer**: For building the CLI.
- **Pathlib**: For handling file paths.
- **Shutil**: For copying files.
- **Time**: For timing operations.
- **Contextlib**: For managing context in the simulation.

---

## GitHub Repository

For more information, visit the [AMS_BP GitHub repository](https://github.com/joemans3/AMS_BP).

---

## Version

The current version of the module is:

```
AMS_BP version: [bold]{__version__}[/bold]
```
