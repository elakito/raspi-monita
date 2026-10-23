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
The model files are found in https://huggingface.co/rhasspy/piper-voices/tree/main.

If you want to add additional piper models, add them in [resources/piper/manifest.toml](resources/piper/manifest.toml).


## Running the program

```
python -m raspi_monita.voice_chat
```

Usage information is displayed when you run the program with --help.
```
python -m raspi_monita.voice_chat --help
usage: voice_chat.py [-h] [--model MODEL] [--tts {espeak,piper}]
                     [--config CONFIG]

Voice Chat Assistant

options:
  -h, --help            show this help message and exit
  --model MODEL         Model Name (default: gemini-3.5-flash-lite)
  --tts {espeak,piper}  TTS to use (default: piper)
  --config CONFIG       Configuration Name (default default)
```

When the model or tts options are omitted, they overwrite the values set in the configuration file.

### Configuration files

The configuration files are stored in [resources/config](resources/config). To select one, pass its filename to the `--config` parameter.

The available parameters are shown below.

| Parameter | Description |
|---|---|
| `model_name` |  |
| `tts_name` |  |
| `audio_rate` |  |
| `audio_channels` | |
| `system_instruction` | |
| `prompt_instruction` | |
| `voice_registry` | |

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
