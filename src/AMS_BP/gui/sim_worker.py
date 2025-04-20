from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from .logging_window import LogWindow


class SimulationWorker(QObject):
    finished = pyqtSignal()
    log_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, config_path: Path, log_window: LogWindow):
        super().__init__()
        self.config_path = config_path
        self.log_window = log_window
        self._cancel_requested = False

    def run(self):
        try:
            import sys

            sys.stdout = self
            sys.stderr = self

            def wrapped_run_sim(config_path):
                # Wrap the real simulation call with cancel checks
                from ..configio.convertconfig import load_config, setup_microscope
                from ..configio.saving import save_config_frames

                loadedconfig = load_config(config_path)

                if "version" in loadedconfig:
                    version = loadedconfig["version"]
                    self.log_message.emit(f"Using config version: {version}")

                setup_config = setup_microscope(loadedconfig)
                microscope = setup_config["microscope"]
                configEXP = setup_config["experiment_config"]
                functionEXP = setup_config["experiment_func"]

                # Long-running operation
                if self.log_window.cancel_requested:
                    self.log_message.emit("Simulation canceled before execution.")
                    return

                frames, metadata = functionEXP(microscope=microscope, config=configEXP)

                if self.log_window.cancel_requested:
                    self.log_message.emit("Simulation canceled after run.")
                    return

                save_config_frames(
                    metadata, frames, setup_config["base_config"].OutputParameter
                )

                self.log_message.emit("Simulation data saved successfully.")

            # Run wrapped simulation
            wrapped_run_sim(self.config_path)

            self.finished.emit()
        except Exception as e:
            self.error_occurred.emit(f"Simulation failed: {e}")
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

    def write(self, text):
        if text.strip():
            self.log_message.emit(text)

    def flush(self):
        pass
