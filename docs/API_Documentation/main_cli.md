# AMS_BP CLI API Reference

## Module: `main_cli.py`

This module provides the command-line interface for the AMS_BP package.

### Dependencies

```python
import shutil
import sys
from pathlib import Path
from typing import Optional

import rich
import typer
from PyQt6.QtWidgets import QApplication
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing_extensions import Annotated

from . import __version__
from .gui.main import MainWindow
from .run_sim_util import run_simulation_from_file
```

## Main Application Object

### `typer_app_asms_bp`

The main Typer application instance that defines the CLI.

```python
typer_app_asms_bp = typer.Typer(
    name="AMS_BP CLI Tool",
    help=cli_help_doc,
    short_help="CLI tool for AMS_BP.",
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    add_completion=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
```

## Functions

### `cell_simulation()`

**Description:**  
Callback function that displays version information when the CLI is invoked without a specific command.

**Usage:**  
Automatically called when running the CLI without commands.

**Returns:**  
None

**Example:**
```bash
python -m main_cli
```

---

### `run_gui() -> None`

**Description:**  
Starts the PyQt GUI for the AMS_BP application.

**Parameters:**  
None

**Returns:**  
None

**Example:**
```bash
python -m main_cli gui
```

---

### `generate_config(output_path: Path = Path("."), output_path_make_recursive: Optional[bool] = None) -> None`

**Description:**  
Generates a sample configuration file for the cell simulation.

**Parameters:**
- `output_path`: Path where the configuration file will be saved.
  - Type: `Path`
  - Default: Current directory (`Path(".")`)
  - CLI Option: `--output_path, -o`
  
- `output_path_make_recursive`: Whether to create the output directory if it doesn't exist.
  - Type: `Optional[bool]`
  - Default: `None`
  - CLI Option: `--recursive_o, -r`

**Returns:**  
None

**Raises:**
- `typer.Abort`: If there's an error creating the configuration file or the output directory.

**Example:**
```bash
python -m main_cli config --output_path /path/to/directory --recursive_o
```

---

### `run_cell_simulation(config_file: Path) -> None`

**Description:**  
Runs the cell simulation using the provided configuration file.

**Parameters:**
- `config_file`: Path to the configuration file.
  - Type: `Path`
  - Required: Yes
  - CLI Argument: First positional argument

**Returns:**  
None

**Example:**
```bash
python -m main_cli runsim path/to/config.toml
```

---

### `validate_config(config: dict) -> None`

**Description:**  
Validates that the configuration dictionary has the required structure.

**Parameters:**
- `config`: Configuration dictionary to validate.
  - Type: `dict`
  - Required: Yes

**Returns:**  
None

**Raises:**
- `typer.Abort`: If the configuration doesn't have the required structure.

**Internal Details:**  
- Checks for the presence of an `Output_Parameters` section
- Checks for the presence of an `output_path` field within the `Output_Parameters` section

---

## CLI Commands

| Command | Description | Arguments | Options |
|---------|-------------|-----------|---------|
| `config` | Generate sample configuration file | - | `--output_path, -o`: Output directory<br>`--recursive_o, -r`: Create directories if they don't exist |
| `runsim` | Run cell simulation | `CONFIG_FILE`: Path to config file | - |
| `gui` | Start the PyQt GUI | - | - |

## Import and Use in Other Code

To use this CLI as a module in other Python code:

```python
from ams_bp.main_cli import typer_app_asms_bp, generate_config, run_cell_simulation, run_gui

# Generate a config file
generate_config(Path("/path/to/output"), True)

# Run a simulation
run_cell_simulation(Path("/path/to/config.toml"))

# Run the GUI
run_gui()
```

## Related Files

- `run_sim_util.py`: Contains the `run_simulation_from_file()` function used by the `runsim` command
- `gui/main.py`: Contains the `MainWindow` class used by the `gui` command
- `sim_config.toml`: Template configuration file that gets copied when using the `config` command
