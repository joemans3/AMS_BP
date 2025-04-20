from pathlib import Path

from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QTextBrowser, QVBoxLayout


class HelpWindow(QDialog):
    def __init__(self, help_path: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help")
        self.resize(500, 400)

        layout = QVBoxLayout()
        browser = QTextBrowser()

        try:
            with open(help_path, "r", encoding="utf-8") as f:
                browser.setMarkdown(f.read())
        except Exception as e:
            browser.setPlainText(f"Failed to load help: {e}")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)

        layout.addWidget(browser)
        layout.addWidget(buttons)
        self.setLayout(layout)
