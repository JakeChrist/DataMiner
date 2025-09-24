"""Entry point for launching the DataMiner desktop application."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication, QLabel

from .logging import setup_logging


def main() -> None:
    """Start the PyQt6 application."""
    logger = setup_logging()
    logger.debug("Starting QApplication")

    app = QApplication(sys.argv)
    app.setApplicationName("DataMiner")

    window = QLabel("DataMiner is running.")
    window.setWindowTitle("DataMiner")
    window.resize(400, 200)
    window.show()

    logger.info("Application started")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
