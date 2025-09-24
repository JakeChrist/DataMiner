# DataMiner Desktop Application Scaffold

This repository provides the initial project structure for the DataMiner desktop
application. The goal is to support fully offline execution, from document
ingestion through retrieval and user interface rendering.

## Prerequisites

* Python 3.11 or later.
* Ability to install Python packages from `requirements.txt` (the listed
  dependencies are suitable for offline use once wheels are available locally).

## Installation

1. (Optional) Create and activate a virtual environment.
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
   ```
2. Install dependencies in a single step.
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

## Launching the Application

Run the PyQt6 interface via the package entry point:

```bash
python -m app
```

The starter UI displays a placeholder window confirming the application is
running. Configuration files and logs are stored in the user-specific
configuration directory (`%APPDATA%/DataMiner` on Windows or
`~/.config/DataMiner` on Linux/macOS). All functionality operates locally; the
application performs no network calls or telemetry by default.

## Project Structure

```
app/
├── __init__.py
├── __main__.py
├── config.py
├── logging.py
├── ingest/
├── retrieval/
├── services/
├── storage/
└── ui/
```

Each subpackage is prepared for future expansion to cover document ingestion,
storage, retrieval, background services, and the PyQt6 user interface.
