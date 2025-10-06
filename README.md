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

## Loading a Corpus of Documents

Use the File → Corpus menu (or the toolbar shortcut) to load sources into the
active project:

1. Choose **Add Folder to Corpus…** to select an entire directory. All
   supported files (`.pdf`, `.docx`, `.txt/.text`, `.md/.markdown/.mkd`,
   `.html/.htm`, `.py/.pyw`, `.m`, `.cpp`) inside the folder are queued for
   background indexing.
2. Choose **Add Files to Corpus…** to index specific files without adding the
   surrounding folder.
3. The status bar reports progress while files are parsed. When indexing
   completes the left-hand *Corpus* panel lists the discovered folders and
   documents. Asking questions remains disabled until at least one document has
   been indexed.
4. After documents change on disk, use **Rescan Indexed Folders** to refresh
   the stored corpus without reselecting directories.

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

### Response Quality Assets

The `Docs/` directory now includes the artifacts produced by the response quality improvement initiative:

- `response_template.md` – canonical structure for long-form answers.
- `research_quick_reference.md` – rapid research checklist and prompts.
- `quality_assurance_workflow.md` – peer review process and QA checklist.
- `knowledge_base/` – seeded repository of reusable insights, evidence logs, and playbooks.

These materials support richer, more consistent responses and should be reviewed before drafting substantive answers.
