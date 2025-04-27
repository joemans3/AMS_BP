import shutil
from datetime import datetime
from pathlib import Path


def setup_run_directory(config_path: Path, base_dir: Path = None) -> dict:
    """
    Sets up a structured directory for a simulation run.

    Returns a dictionary with paths:
        {
            'run_dir': Path,
            'log_file': Path,
            'copied_config': Path,
        }
    """
    if base_dir is None:
        base_dir = Path.home() / "AMS_runs"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    # Copy config into run directory
    copied_config = run_dir / "config.toml"
    shutil.copy(config_path, copied_config)

    # Define log file path
    log_file = run_dir / "sim.log"

    return {
        "run_dir": run_dir,
        "log_file": log_file,
        "copied_config": copied_config,
    }
