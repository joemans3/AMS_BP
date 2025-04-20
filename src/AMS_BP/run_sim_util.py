import os
import time
from contextlib import contextmanager
from pathlib import Path

import rich
from rich.progress import Progress, SpinnerColumn, TextColumn

from .configio.convertconfig import load_config, setup_microscope
from .configio.saving import save_config_frames


def run_simulation_from_file(config_file: Path, show_progress: bool = True) -> None:
    """
    Core simulation logic used by both CLI and GUI.
    """

    @contextmanager
    def progress_context():
        if show_progress:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            )
            with progress:
                yield progress
        else:
            # Dummy context for GUI
            class DummyProgress:
                def add_task(self, description, total=None):
                    return None

                def update(self, task, completed):
                    pass

            yield DummyProgress()

    with progress_context() as progress:
        start_task_1 = time.time()
        task_1 = progress.add_task(
            description="Processing request to run the simulation ...", total=10
        )

        if not os.path.isfile(config_file):
            raise FileNotFoundError("Configuration file not found.")

        loadedconfig = load_config(config_file)

        if "version" in loadedconfig:
            version = loadedconfig["version"]
            rich.print(f"Using config version: [bold]{version}[/bold]")

        setup_config = setup_microscope(loadedconfig)
        microscope = setup_config["microscope"]
        configEXP = setup_config["experiment_config"]
        functionEXP = setup_config["experiment_func"]

        progress.update(task_1, completed=10)
        rich.print(
            "Prep work done in {:.2f} seconds.".format(time.time() - start_task_1)
        )

        time_task_2 = time.time()
        task_2 = progress.add_task(description="Running the simulation ...", total=None)

        frames, metadata = functionEXP(microscope=microscope, config=configEXP)

        save_config_frames(
            metadata, frames, setup_config["base_config"].OutputParameter
        )

        progress.update(task_2, completed=None)
        rich.print(
            "Simulation completed in {:.2f} seconds.".format(time.time() - time_task_2)
        )
