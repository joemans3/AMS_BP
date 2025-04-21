from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)


class LogWindow(QDialog):
    cancel_requested = False  # Flag to check cancel request

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Simulation Log")
        self.setMinimumSize(600, 300)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)

        # Progress bar (indeterminate)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # Infinite/indeterminate animation

        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.on_cancel_clicked)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)

        layout = QVBoxLayout()
        layout.addWidget(self.log_output)
        layout.addWidget(self.progress)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def append_text(self, text: str):
        self.log_output.appendPlainText(text)
        QTimer.singleShot(0, self.scroll_to_bottom)

    def scroll_to_bottom(self):
        self.log_output.verticalScrollBar().setValue(
            self.log_output.verticalScrollBar().maximum()
        )

    def on_cancel_clicked(self):
        self.cancel_requested = True
        self.append_text("Cancellation requested by user...")

    def mark_success(self):
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 5px;
                background-color: #2d2d2d;
                text-align: center;
                color: white;
                font-weight: bold;
                font-size: 14px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        self.append_text("Simulation completed successfully.")

    def mark_failure(self):
        self.progress.setRange(0, 1)
        self.progress.setValue(0)  # No fill
        self.progress.setFormat("Failed, share logs with developer.")  # Custom text
        self.progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #800000;
                border-radius: 5px;
                background-color: #2d2d2d;
                color: #e53935;
                font-weight: bold;
                font-size: 13px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: transparent;  /* no fill */
            }
        """)
        self.append_text("Simulation failed. Please send logs to the developer.")
