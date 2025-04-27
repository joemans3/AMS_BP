import shutil
import sys

# logutil.py (continued)
import time
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal


def cleanup_old_logs(log_dir: Path, max_age_days: int = 7):
    """
    Deletes entire `run_*` directories in `log_dir` if their `sim.log` is older than `max_age_days`.
    """
    if not log_dir.exists():
        return

    now = time.time()
    max_age = max_age_days * 86400  # seconds in a day

    for run_dir in log_dir.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith("run_"):
            log_file = run_dir / "sim.log"
            try:
                if log_file.exists():
                    file_age = now - log_file.stat().st_mtime
                    if file_age > max_age:
                        shutil.rmtree(run_dir)
                        print(f"Deleted old run directory: {run_dir}")
            except Exception as e:
                print(f"Error while trying to delete {run_dir}: {e}")


class LogEmitter(QObject):
    """Qt signal emitter for log messages."""

    message = pyqtSignal(str)


class LoggerStream:
    """Custom stream that writes to file and emits to GUI."""

    def __init__(self, emitter: LogEmitter, log_file_path: Path):
        self.emitter = emitter
        self.log_file_path = log_file_path
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.log_file_path, "a", encoding="utf-8")

    def write(self, text):
        if text.strip():
            timestamped = f"[{datetime.now().strftime('%H:%M:%S')}] {text.strip()}"
            self.file.write(timestamped + "\n")
            self.file.flush()
            self.emitter.message.emit(timestamped)  # <-- This line is essential

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


class LoggerManager:
    """Manages stream redirection and GUI+file log capture."""

    def __init__(self, log_path: Path):
        self.emitter = LogEmitter()
        self.stream = LoggerStream(self.emitter, log_path)
        self._stdout = sys.stdout
        self._stderr = sys.stderr

    def start(self):
        sys.stdout = self.stream
        sys.stderr = self.stream

    def stop(self):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self.stream.close()

    def get_emitter(self) -> LogEmitter:
        return self.emitter
