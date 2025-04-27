from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from ..core.configio.convertconfig import load_config, setup_microscope
from ..core.configio.saving import save_config_frames
from ..tools.logging.logutil import LogEmitter


class SimulationWorker(QObject):
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, config_path: Path, emitter: LogEmitter, cancel_callback=None):
        super().__init__()
        self.config_path = config_path
        self.emitter = emitter
        self.cancel_callback = cancel_callback or (lambda: False)
        self.failed = False

    def run(self):
        try:
            self.emitter.message.emit(f"Starting simulation for {self.config_path}")

            loadedconfig = load_config(self.config_path)

            if "version" in loadedconfig:
                version = loadedconfig["version"]
                self.emitter.message.emit(f"Using config version: {version}")

            setup_config = setup_microscope(loadedconfig)
            microscope = setup_config["microscope"]
            configEXP = setup_config["experiment_config"]
            functionEXP = setup_config["experiment_func"]

            if self.cancel_callback():
                self.emitter.message.emit("Simulation canceled before execution.")
                return

            frames, metadata = functionEXP(microscope=microscope, config=configEXP)

            if self.cancel_callback():
                self.emitter.message.emit("Simulation canceled after run.")
                return

            save_config_frames(
                metadata, frames, setup_config["base_config"].OutputParameter
            )

            self.emitter.message.emit("Simulation data saved successfully.")
            self.finished.emit()

        except Exception as e:
            self.failed = True
            self.error_occurred.emit(f"Simulation failed: {e}")
        finally:
            self.finished.emit()
