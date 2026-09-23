# raspi-monita

## Features

## Installation

### Setting up the required environment and installing required libraries

```
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Installing the piper model files

```
python scripts/download_piper_models.py
```

## Running the program

```
python -m raspi_monita.voice_chat
```

## Running Sytlecheck and Tests

```
ruff check .
pytest
```

## License & Attributions

This project is licensed under the **MIT License** -- see the [LICENSE](LICENSE.txt) for details.

### Third-Party Software Notices
This application imports and relies on the following third-party libraries:
* **...**
* **...**
