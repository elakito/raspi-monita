import argparse
import json
import os
import select
import shlex
import subprocess
import sys
import termios
import time
import tty
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
import yaml
from google import genai
from google.genai import errors, types
from piper import PiperVoice, SynthesisConfig

# --- Constants ---
JSON_RS = "\x1e"      # json record separator

# --- Configuration ---
RATE = 16000          # 16kHz for accurate speech recognition
CHANNELS = 1          # Mono recording
SPEED_CHANGE_UNIT = 1.1
HISTORY_FILE_PREFIX = ".voice_chat_history"

# --- Default configuration values ---
DEFAULT_MODEL_NAME = "gemini-3.5-flash-lite"
DEFAULT_TTS_NAME = "piper"
DEFAULT_CONFIG_NAME = "default"
DEFAULT_VOICE_REGISTRY = {"en":    "en_GB-alba-medium.onnx"}

# --- Global instances ---
client = None
voice_manager = None

# 1. Define the exact JSON shape you want the model to return
chat_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "language_user": types.Schema(
            type=types.Type.STRING, 
            description="The ISO 2-letter language code of the language in the user's request."
        ),
        "language_model": types.Schema(
            type=types.Type.STRING, 
            description="The ISO 2-letter language code of the language used in the reply."
        ),
        "transcription": types.Schema(
            type=types.Type.STRING,
            description="The exact text transcription of what the user said in the audio."
        ),
        "response": types.Schema(
            type=types.Type.STRING, 
            description="Your conversational, witty response addressing the user's request. Do not include markdown bold or brackets."
        ),
    },
    required=["language_user", "language_model", "transcription", "response"],
)

class RealTimeVoiceManager:
    def __init__(self, model_dir, voice_registry):
        self.model_dir = model_dir
        self.loaded_voices = {}
        
        self.voice_registry = voice_registry
    def _get_voice(self, lang_code):
        if lang_code not in self.voice_registry:
            raise ValueError(f"Language '{lang_code}' is not supported.")
            
        if lang_code not in self.loaded_voices:
            model_name = self.voice_registry[lang_code]
            model_path = os.path.join(self.model_dir, model_name)
            config_path = f"{model_path}.json"
            
            raw_print(f"[🔄] Loading {lang_code} model into memory...")
            self.loaded_voices[lang_code] = PiperVoice.load(model_path, config_path=config_path)
            
        return self.loaded_voices[lang_code]

    def speak_live(self, text, lang_code, speed):
        """Synthesizes text and pipes raw audio bytes directly into aplay."""
        try:
            voice = self._get_voice(lang_code)
        except ValueError as e:
            raw_print(f"[⚠️] Error unsupported language detected. {e}", file=sys.stderr)
            text = str(e)
            voice = self._get_voice('en')
        
        # Get the exact sample rate required by the loaded voice model
        sample_rate = voice.config.sample_rate

        # Launch aplay to receive raw PCM data over stdin
        # Flags used:
        # -t raw : Tells aplay to expect raw headerless PCM bytes instead of a WAV file
        # -c 1   : 1 channel (mono)
        # -r ... : Pushes the model's native sampling rate (e.g., 22050)
        # -f S16_LE : Signed 16-bit Little Endian (Piper's exact byte format)
        # -q     : Quiet mode (hides buffer logs in console)
        aplay_command = [
            "aplay", 
            "-t", "raw", 
            "-c", "1", 
            "-r", str(sample_rate), 
            "-f", "S16_LE", 
            "-q"
        ]
        
        # Open the audio subprocess pipeline
        player = subprocess.Popen(aplay_command, stdin=subprocess.PIPE)

        raw_print("[🐵️] Speaking...", flush=True)
        try:
            # synthesize_stream yields raw PCM audio bytes with ultra-low latency
            syn_config = SynthesisConfig(length_scale=speed)
            for audio_chunk in voice.synthesize(text, syn_config=syn_config):
                # Send the generated sound segment directly into aplay's mouth
                player.stdin.write(audio_chunk.audio_int16_bytes)
                player.stdin.flush()
        finally:
            # Safely close stdin and wait for aplay to finish speaking the final syllable
            if player.stdin:
                player.stdin.close()
            player.wait()
        
        raw_print("[🐒️] Done speaking.", flush=True)


class UserAction(Enum):
    SUBMIT = 1
    REPLAY = 2
    CANCEL = 3
    FASTER = 4
    SLOWER = 5
    NORMAL = 6
    INVALID = 7
    TERMINATE = 8


def raw_print(*args, **kwargs) -> None:
    # Join arguments into a string and replace \n with \r\n
    output = " ".join(map(str, args)).replace("\n", "\r\n")
    print(output, end="\r\n", **kwargs)


def is_key_waiting() -> bool:
    """Checks standard input instantly. Returns True if a key was pushed."""
    # Arguments: [inputs to watch], [outputs], [exceptions], timeout (0 = do not block)
    ready_to_read, _, _ = select.select([sys.stdin], [], [], 0.0)
    return len(ready_to_read) > 0


def update_config(key, value, default_value, conf):
    """Updates the configuration parameter if missing or be overriden."""
    if key not in conf:
        conf[key] = default_value
    if value is not None and value != conf[key]:
        conf[key] = value


def read_config(config_dir, args) -> dict[str, Any]:
    """Read the configuration parameters from the file and updates the parameters if missing or being overriden."""
    config_path = config_dir / f"{args.config}.yaml"
    with config_path.open(encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    update_config("model_name", args.model, DEFAULT_MODEL_NAME, conf)
    update_config("tts_name", args.tts, DEFAULT_TTS_NAME, conf)
    update_config("audio_rate", None, str(RATE), conf)
    update_config("audio_channels", None, str(CHANNELS), conf)
    update_config("voice_registry", None, DEFAULT_VOICE_REGISTRY, conf)
    return conf


def load_history(name) -> tuple[list[types.Content], str, str]:
    """Loads saved history logs and transforms them into SDK Content structural objects."""
    history_file = f"{HISTORY_FILE_PREFIX}_{name}.jsons"
    if not os.path.exists(history_file):
        return [], None, None
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            # Extract the raw text block from the model's response
            raw_history = [json.loads(entry)
                           for entry in f.read().split(JSON_RS)
                           if entry.strip()]
            return [types.Content(role=entry["role"], parts=[{"text": entry["text"]}]) for entry in raw_history], *((raw_history[-1]["text"], raw_history[-1]["language"]) if raw_history else (None, None))

    except Exception:  # noqa: BLE001
        raw_print("[⚠️] Error reading history file. Starting fresh.", file=sys.stderr)
        return [], None, None


def append_history(name, chat_session, language_user, language_model) -> None:
    """Extracts the most recent history log entries from the active chat state and appends them to local JSON-Sequence storage."""

    history_file = f"{HISTORY_FILE_PREFIX}_{name}.jsons"
    serializable_history = []
    message_pair = chat_session.get_history()[-2:]
    model_text = "".join([p.text for p in message_pair[1].parts if hasattr(p, 'text')])
    try:
        # Unpack the structured JSON payload from the model turn
        data = json.loads(model_text)
        user_text = data.get("transcription", "[Audio Prompt]")
        model_text = data.get("response", model_text)
    except (json.JSONDecodeError, TypeError):
        # Fallback if the data is not in JSON format
        user_text = "[Audio Prompt]"

    # Append the polished, classic interaction format entries to our list
    serializable_history.append({"role": "user", "text": user_text, "language": language_user})
    serializable_history.append({"role": "model", "text": model_text, "language": language_model})

    with open(history_file, "a", encoding="utf-8") as f:
        for entry in serializable_history:
            f.write(JSON_RS)
            json.dump(entry, f)
            f.write("\n")


def speak_text_via_espeak(text, lang, speed, pitch, gap):
    """Speaks a text using espeak-ng."""
    try:
        # shell=True allows espeak-ng to play audio directly to your output device
        subprocess.run(f"espeak-ng -v {lang} -s {speed} -p {pitch} -g {gap} {shlex.quote(text)}", shell=True, check=True)
    except subprocess.CalledProcessError as e:
        raw_print(f"Error executing espeak-ng: {e}", file=sys.stderr)


def speak_text_via_piper(text, lang, speed):
    """Speaks a text using piper."""
    voice_manager.speak_live(text,lang_code=lang, speed=speed)


def append_recording(proc, frames, terminating=False) -> None:
    while True:
        ready_to_read, _, _ = select.select([proc.stdout], [], [], 0.0)
        if ready_to_read:
            chunk = proc.stdout.read(4096)
            if not chunk:  # EOF
                break
            if frames is not None:
                frames.append(chunk)
        elif not terminating or proc.poll() is not None:
            break
    # drain any leftovers until the process terminates and append them
    if terminating:
        leftovers = None
        try:
            leftovers, _ = proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            leftovers, _ = proc.communicate()
        if frames is not None and leftovers:
            frames.append(leftovers)


def record_audio(is_first_time) -> tuple[UserAction, list]:
    """Records sound via native ALSA utilities, controlled entirely via the Enter key."""
    raw_print("\nPress [SPACE] to start/stop recording, [c] to cancel recording,[r] to replay, [<] to speak slower, [>] faster, [=] normal, [q] to quit")

    process = None
    audio_frames = []
    user_action = UserAction.SUBMIT
    # Continuous lookahead checking for the Enter key press
    while True:
        if is_key_waiting():
            # Code halts here until exactly one key is pushed
            key = sys.stdin.read(1).lower()
            if key == ' ':
                if process:
                    raw_print("\n[🟩️] Recorded.")
                    # Safe teardown of the system process
                    process.terminate()
                    append_recording(process, audio_frames, True)
                    process = None
                    break
                else:
                    raw_print("\n[🔴️] Recording... Press [SPACE] again to stop or [c] to cancel.")
                    # Fire up arecord as a non-root background stream running to stdout (-)
                    process = subprocess.Popen(
                        ['arecord', '-t', 'wav', '-f', 'S16_LE', '-r', str(RATE), '-c', str(CHANNELS), '-'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        restore_signals=True
                    )
            elif key == 'c':
                if process:
                    # Safe teardown of the system process
                    process.terminate()
                    append_recording(process, None, True)
                    process = None
                    audio_frames = None
                    user_action = UserAction.CANCEL
                    break
                else:
                    raw_print("\n[❓️] Not in recording, nothing to cancel.")
                    user_action = UserAction.INVALID
                    break
            elif key == 'q':
                if process:
                    process.terminate()
                    append_recording(process, None, True)
                audio_frames = None
                user_action = UserAction.TERMINATE
                break
            elif key == 'r':
                if is_first_time:
                    raw_print("\n[❓️] Nothing recorded yet, nothing to replay.")
                    user_action = UserAction.INVALID
                    break
                elif process:
                    raw_print("\n[❓️] Recording in progress, no replaying.")
                else:
                    audio_frames = None
                    user_action = UserAction.REPLAY
                    break
            elif key == '<':
                if process:
                    raw_print("\n[❓️] Recording in progress.")
                else:
                    user_action = UserAction.SLOWER
                    break
            elif key == '>':
                if process:
                    raw_print("\n[❓️] Recording in progress.")
                else:
                    user_action = UserAction.FASTER
                    break
            elif key == '=':
                if process:
                    raw_print("\n[❓️] Recording in progress.")
                else:
                    user_action = UserAction.NORMAL
                    break
            else:
                # Safely ignore any unmapped keys the user accidentally hits
                raw_print(f"\n[❓️] Unrecognized key: {key.lower()}.")
                user_action = UserAction.INVALID
                break

        if process:
            append_recording(process, audio_frames)

        time.sleep(0.5)
    return (user_action, audio_frames)

def main(args):
    global client, voice_manager
    print("=== Voice Chat Assistant Initialized ===")

    if not os.environ.get('GEMINI_API_KEY'):
        print("Error: GEMINI_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    # project_root/src/raspi-monita -> project_root/resources
    resource_dir = Path(__file__).resolve().parent.parent.parent / "resources"
    config_dir = resource_dir  / "config"
    model_dir = resource_dir  / "piper"

    # config
    conf = read_config(config_dir, args)

    # gemini client, voice manager
    client = genai.Client()
    voice_manager = RealTimeVoiceManager(model_dir, conf["voice_registry"])

    # 1. Save original terminal settings so we can fix the console on exit
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    # 2. Re-hydrate any chat threads from a past run
    past_history, latest_response, latest_language_model = load_history(args.config)
    if past_history:
        raw_print(f"[🔄️] Resumed past conversation context ({len(past_history)} messages parsed).")

    # 3. Establish our official stateful Chat Session
    chat = client.chats.create(
        model=args.model,
        history=past_history,
        config=types.GenerateContentConfig(
            response_mime_type="application/json", # Tells the API to format as JSON
            response_schema=chat_schema,          # Enforces the specific key-value pairs
            system_instruction=conf["system_instruction"]
        )
    )

    try:
        # Puts the terminal into raw mode so keystrokes bypass line buffering
        tty.setraw(fd)

        latest_language_user = None
        speech_speed = 1.0
        while True:
            # 4. Record voice data and send them to Gemini API
            # Gather voice byte profile natively
            user_action, audio_frames = record_audio(latest_response == None)
            if user_action == UserAction.TERMINATE:
                # terminated
                raw_print("\n[⬛️] Shutting down pipeline.")
                break
            if user_action == UserAction.CANCEL:
                # canceled
                raw_print("\n[❌️] Recording canceled.")
                continue
            if user_action == UserAction.SLOWER:
                speech_speed = speech_speed / SPEED_CHANGE_UNIT
                raw_print(f"\n[⬇️️] Speech speed decreased to to {speech_speed:.2f}.")
                continue
            if user_action == UserAction.FASTER:
                speech_speed = speech_speed * SPEED_CHANGE_UNIT
                raw_print(f"\n[⬆️️] Speech speed increased to {speech_speed:.2f}.")
                continue
            if user_action == UserAction.NORMAL:
                speech_speed = 1.0
                raw_print(f"\n[↩️️️] Speech speed reset to {speech_speed:.2f}.")
                continue
            if user_action == UserAction.INVALID:
                continue
            if user_action == UserAction.SUBMIT:
                audio_bytes = b''.join(audio_frames)
                if not audio_bytes or len(audio_bytes) < 44:  # 44 bytes is a standard blank WAV header size
                    raw_print("\n[⚠️] Audio stream too short. Please try again.")
                    continue

                raw_print("\n[🚀️] Transmitting voice data payload to Gemini API...")
                audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")

                try:
                    # Send message through the stateful chat engine
                    # We guide the model to process the audio payload natively
                    response = chat.send_message([
                        conf["prompt_instruction"],
                        audio_part
                    ])
                except errors.APIError as e:
                    raw_print(f"\n[💩️] Something went wrong: {e.code} {e.message}")
                    speak_text_via_piper(e.message, "en", 1.0)
                    continue
                except httpx.TransportError as e:
                    raw_print(f"\n[💩️] Transport failure: {e}")
                    speak_text_via_piper(str(e), "en", 1.0)
                    continue

                raw_print("\n[🤖️] Gemini Response:")
                raw_print(response.text)
                raw_print("=" * 40)
                response_obj = json.loads(response.text)
                latest_response = response_obj['response']
                latest_language_model = response_obj['language_model']
                latest_language_user = response_obj['language_user']
            elif user_action == UserAction.REPLAY:
                raw_print("\n[🔁️] Replaying the audio.")

            # 5. Generate the audio response using TTS
            tts_name = conf["tts_name"]
            if tts_name == "piper":
                speak_text_via_piper(latest_response, latest_language_model, 1.0/speech_speed)
            elif tts_name == "espeak":
                speak_text_via_espeak(latest_response, latest_language_model, int(140*speech_speed), 60, 3)

            if user_action == UserAction.SUBMIT:
                # 6. Synchronize conversation timeline directly to disk
                append_history(args.config, chat, latest_language_user, latest_language_model)
            time.sleep(0.5)

    except KeyboardInterrupt:
        raw_print("\n[👋️] Session terminated.")
    finally:
        # 7. CRUCIAL: Always restore terminal settings back to normal when exiting
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    raw_print("\n[👋️] Session ended safely. Goodbye!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Chat Assistant")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Model Name (default: %(default)s)")
    parser.add_argument("--tts", default=DEFAULT_TTS_NAME, choices={"piper", "espeak"}, help="TTS to use (default: %(default)s)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_NAME, help="Configuration Name (default %(default)s)")
    args = parser.parse_args()

    main(args)
