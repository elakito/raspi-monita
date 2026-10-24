# raspi-monita

## Features

## Installation

### Setting up the required environment and installing the required libraries

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

If you want to add additional piper models, add them in [resources/piper/manifest.toml](resources/piper/manifest.toml) using the following format.

```
[[voices]]
id = {locale}-{speaker}-{quality}
language =
locale =
speaker =
quality =
model_sha256 = ""
config_sha256 = ""
```

When the sha256 properties are empty, the actual sha256 value will be printed after the resource is downloaded. Update the sha256 values in the [resources/piper/manifest.toml](resources/piper/manifest.toml) file accordingly.

For example, to add de_DE-ramona-low, add the following entry:

```
[[voices]]
id = "de_DE-ramona-low"
language = "de"
locale = "de_DE"
speaker = "ramona"
quality = "low"
model_sha256 = ""
config_sha256 = ""
```

## Running the program

Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey), then set it as the `GEMINI_API_KEY` environment variable.
```
export GEMINI_API_KEY=...
```
To start the program, run the following command.
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
| `model_name` | the Gemini model name such as gemini-3.5-flash-lite |
| `tts_name` | the TTS engine name such as piper |
| `audio_rate` | the audio sampling rate |
| `audio_channels` | the number of audio channels |
| `system_instruction` | the system instruction for the chat |
| `prompt_instruction` | the prompt instruction submitted with the voice data |
| `voice_registry` | the piper voice registry given as a mapping dictionary |

## Running Sytlecheck and Tests

```
ruff check .
pytest
```

## License & Attributions

This project is licensed under the **MIT License** -- see the [LICENSE](LICENSE.txt) for details.
