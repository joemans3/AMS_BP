"""
main_cli.py

This file contains the command-line interface (CLI) for the AMS_BP package.

The CLI is built using Typer and provides two main commands:
1. 'config': Generates a sample configuration file.
2. 'runsim': Runs the cell simulation using a provided configuration file.
3. 'gui': starts the PyQT GUI

Main Components:
- typer_app_asms_bp: The main Typer application object.
- cell_simulation(): Callback function that displays the version information.
- generate_config(): Command to generate a sample configuration file.
- run_cell_simulation(): Command to run the cell simulation using a configuration file.
- run_gui(): runs the GUI

Usage:
- To generate a config file: python main_cli.py config [OPTIONS]
- To run a simulation: python main_cli.py runsim [CONFIG_FILE]

The file uses Rich for enhanced console output and progress tracking.
"""

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
from .core.run_sim_util import run_simulation_from_file
from .gui.main import MainWindow
from .tools.logging.logutil import cleanup_old_logs

cli_help_doc = str(
    """
CLI tool to run [underline]A[/underline]dvanced [underline]M[/underline]olecule [underline]S[/underline]imulation: [underline]AMS[/underline]-BP. GitHub: [green]https://github.com/joemans3/AMS_BP[/green].
[Version: [bold]{0}[/bold]]
""".format(__version__)
)


# create a new CLI function
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


# make a callback function to run the simulation
@typer_app_asms_bp.callback(invoke_without_command=True)
def cell_simulation():
    # print version
    # find version using the __version__ variable in the __init__.py file
    cleanup_old_logs(Path.home() / "AMS_runs", max_age_days=7)
    out_string = f"AMS_BP version: [bold]{__version__}[/bold]"
    rich.print(out_string)


@typer_app_asms_bp.command(name="gui")
def run_gui() -> None:
    """Start the PyQt GUI"""
    # Clean old logs
    log_dir = Path.home() / "AMS_runs"
    cleanup_old_logs(log_dir, max_age_days=7)
    app = QApplication(sys.argv)
    editor = MainWindow()
    editor.show()
    sys.exit(app.exec())


@typer_app_asms_bp.command(name="config")
def generate_config(
    output_path: Annotated[
        Path,
        typer.Option("--output_path", "-o", help="Path to the output file"),
    ] = Path("."),
    output_path_make_recursive: Annotated[
        Optional[bool],
        typer.Option(
            "--recursive_o",
            "-r",
            help="Make the output directory if it does not exist",
        ),
    ] = None,
) -> None:
    """
    Generate a sample configuration file for the cell simulation and save it to the specified output path.
    """

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task_1 = progress.add_task(
            description="Processing request to create a default config file ...",
            total=10,
        )

        # check if the output path is provided and is a valid directory | if not none

        try:
            output_path = Path(output_path)
        except ValueError:
            print("FileNotFoundError: Invalid output path.")
            raise typer.Abort()
        # double check if the output path is a valid directory
        if not output_path.is_dir():
            # if not, make the directory
            if output_path_make_recursive:
                try:
                    output_path.mkdir(parents=True, exist_ok=True)
                except FileExistsError:
                    print(f"FileExistsError: Directory {output_path} already exists.")
            else:
                print(f"FileNotFoundError: {output_path} is not a valid directory.")
                raise typer.Abort()
        # find the parent dir
        project_directory = Path(__file__).parent
        # find the config file
        config_file = (
            project_directory / "resources" / "template_configs" / "sim_config.toml"
        )
        output_path = output_path / "sim_config.toml"
        # copy the config file to the output path

        # complete last progress
        progress.update(task_1, completed=10)

        task_2 = progress.add_task(
            description="Copying the config file to the output path ...", total=10
        )
        try:
            shutil.copy(config_file, output_path)
        except FileNotFoundError:
            rich.print(f"Error: No config file found in {project_directory}.")
            raise typer.Abort()
        progress.update(task_2, completed=10)
        # complete
        rich.print(f"Config file saved to {output_path.resolve()}")


@typer_app_asms_bp.command(name="runsim")
def run_cell_simulation(
    config_file: Annotated[Path, typer.Argument(help="Path to the configuration file")],
) -> None:
    run_simulation_from_file(config_file)


def validate_config(config: dict) -> None:
    if "Output_Parameters" not in config:
        rich.print(
            "ConfigError: 'Output_Parameters' section not found in the configuration file."
        )
        raise typer.Abort()
    output_parameters = config["Output_Parameters"]
    if "output_path" not in output_parameters:
        rich.print("ConfigError: 'output_path' not found in the configuration file.")
        raise typer.Abort()
