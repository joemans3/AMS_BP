"""
run_cell_simulation.py

This file contains the command-line interface (CLI) for the SMS_BP package, which is used for simulating single molecule localization microscopy experiments.

The CLI is built using Typer and provides two main commands:
1. 'config': Generates a sample configuration file.
2. 'runsim': Runs the cell simulation using a provided configuration file.

Main Components:
- typer_app_sms_bp: The main Typer application object.
- cell_simulation(): Callback function that displays the version information.
- generate_config(): Command to generate a sample configuration file.
- run_cell_simulation(): Command to run the cell simulation using a configuration file.

Usage:
- To generate a config file: python run_cell_simulation.py config [OPTIONS]
- To run a simulation: python run_cell_simulation.py runsim [CONFIG_FILE]

The file uses Rich for enhanced console output and progress tracking.
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import rich
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing_extensions import Annotated

from . import __version__
from .simulate_cell import Simulate_cells

cli_help_doc = """
CLI tool to run [underline]S[/underline]ingle [underline]M[/underline]olecule [underline]S[/underline]imulation: [underline]SMS[/underline]-BP. GitHub: [green]https://github.com/joemans3/SMS_BP[/green].
[Version: [bold]{0}[/bold]]
""".format(__version__)


# create a new CLI function
typer_app_sms_bp = typer.Typer(
    name="SMS_BP CLI Tool",
    help=cli_help_doc,
    short_help="CLI tool for SMS_BP.",
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    add_completion=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


# make a callback function to run the simulation
@typer_app_sms_bp.callback(invoke_without_command=True)
def cell_simulation():
    # print version
    # find version using the __version__ variable in the __init__.py file
    out_string = f"SMS_BP version: [bold]{__version__}[/bold]"
    rich.print(out_string)


@typer_app_sms_bp.command(name="config")
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
        config_file = project_directory / "sim_config.json"
        output_path = output_path / "sim_config.json"
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


# second command to run the simulation using the config file path as argument
@typer_app_sms_bp.command(name="runsim")
def run_cell_simulation(
    config_file: Annotated[Path, typer.Argument(help="Path to the configuration file")],
) -> None:
    """
    Run the cell simulation using the configuration file provided.
    """
    from contextlib import contextmanager

    @contextmanager
    def progress_context():
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        )
        try:
            with progress:
                yield progress
        finally:
            progress.stop()

    # Use in functions
    with progress_context() as progress:
        start_task_1 = time.time()
        task_1 = progress.add_task(
            description="Processing request to run the simulation ...", total=10
        )

        # check if the config file is a valid file
        if not os.path.isfile(config_file):
            rich.print("FileNotFoundError: Configuration file not found.")
            raise typer.Abort()
        # check if the config file is a valid json file
        try:
            with open(config_file) as f:
                config = json.load(f)
        except json.JSONDecodeError:
            rich.print("JSONDecodeError: Configuration file is not a valid JSON file.")
            raise typer.Abort()

        validate_config(config)

        output_parameters = config["Output_Parameters"]
        output_path = output_parameters["output_path"]

        # find the version flag in the config file
        if "version" in config:
            version = config["version"]
            rich.print(f"Using config version: [bold]{version}[/bold]")
        # complete last progress
        progress.update(task_1, completed=10)
        rich.print(
            "Prep work done in {:.2f} seconds.".format(time.time() - start_task_1)
        )

        time_task_2 = time.time()
        task_2 = progress.add_task(description="Running the simulation ...", total=None)
        # run the simulation
        sim = Simulate_cells(str(config_file))
        sim.get_and_save_sim(
            cd=output_path,
            img_name=output_parameters.get("output_name"),
            subsegment_type=output_parameters.get("subsegment_type"),
            subsegment_num=int(output_parameters.get("subsegment_number")),
        )

        progress.update(task_2, completed=None)
        rich.print(
            "Simulation completed in {:.2f} seconds.".format(time.time() - time_task_2)
        )


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