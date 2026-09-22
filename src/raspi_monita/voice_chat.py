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
from pathlib import Path

import httpx
from google import genai
from google.genai import errors, types
from piper import PiperVoice

# --- Constants ---
JSON_RS = "\x1e"

# --- Configuration ---
RATE = 16000          # 16kHz for accurate speech recognition
CHANNELS = 1          # Mono recording
HISTORY_FILE = ".voice_chat_history.jsons"
DEFAULT_MODEL_NAME = "gemini-3.5-flash-lite"
DEFAULT_TTS_NAME = "piper"

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
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.loaded_voices = {}
        
        # Mapping of your verified local Piper models
        # voices found in https://huggingface.co/rhasspy/piper-voices/tree/main
        self.voice_registry = {
            "de":    "de_DE-kerstin-low.onnx",
            "en":    "en_GB-alba-medium.onnx",
            "es":    "es_ES-sharvard-medium.onnx",
            "fr":    "fr_FR-siwis-medium.onnx",
            "it":    "it_IT-paola-medium.onnx",
            "ja":    "ja_JA-hi_fi_captain-medium.onnx",
            "pt":    "pt_PT-tug\u00e3o-medium.onnx",
            "zh":    "zh_CN-huayan-medium.onnx"
        }

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

    def speak_live(self, text, lang_code):
        """Synthesizes text and pipes raw audio bytes directly into aplay."""
        voice = self._get_voice(lang_code)
        
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
            for audio_chunk in voice.synthesize(text):
                # Send the generated sound segment directly into aplay's mouth
                player.stdin.write(audio_chunk.audio_int16_bytes)
                player.stdin.flush()
        finally:
            # Safely close stdin and wait for aplay to finish speaking the final syllable
            if player.stdin:
                player.stdin.close()
            player.wait()
        
        raw_print("[🐒️] Done speaking.", flush=True)

def raw_print(*args, **kwargs):
    # Join arguments into a string and replace \n with \r\n
    output = " ".join(map(str, args)).replace("\n", "\r\n")
    print(output, end="\r\n", **kwargs)

def is_key_waiting():
    """Checks standard input instantly. Returns True if a key was pushed."""
    # Arguments: [inputs to watch], [outputs], [exceptions], timeout (0 = do not block)
    ready_to_read, _, _ = select.select([sys.stdin], [], [], 0.0)
    return len(ready_to_read) > 0

def load_history() -> list[types.Content]:
    """Loads saved history logs and transforms them into SDK Content structural objects."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            # Extract the raw text block from the model's response
            return [
                types.Content(**json.loads(entry))
                for entry in f.read().split(JSON_RS)
                if entry.strip()
            ]
    except Exception:  # noqa: BLE001
        raw_print("[⚠️] Error reading history file. Starting fresh.", file=sys.stderr)
        return []

def save_history(chat_session):
    """Extracts history logs from the active chat state and pushes them to local JSON storage."""
    serializable_history = []
    raw_history = chat_session.get_history()
    for i in range(0, len(raw_history), 2):
        # Ensure we have a complete pair to analyze
        if i + 1 < len(raw_history):
            message_pair = raw_history[i:i+2]
            # Ignore the user's request with the audio part
            # Extract the raw text block from the model's response
            model_text = "".join([p.text for p in message_pair[1].parts if hasattr(p, 'text')])
            try:
                # Unpack the structured JSON payload from the model turn
                data = json.loads(model_text)
                user_text = data.get("transcription", "[Audio Prompt]")
                assistant_text = data.get("response", model_text)
            except (json.JSONDecodeError, TypeError):
                # Fallback if the data is not in JSON formattext
                user_text = "[Audio Prompt]"
                assistant_text = model_text

            # Append the polished, classic interaction format entries to our list
            serializable_history.append({"role": "user", "parts": [{"text": user_text}]})
            serializable_history.append({"role": "model", "parts": [{"text": assistant_text}]})

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
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

def speak_text_via_piper(text, lang):
    """Speaks a text using piper."""
    voice_manager.speak_live(text,lang_code=lang)


def record_audio():
    """Records sound via native ALSA utilities, controlled entirely via the Enter key."""
    raw_print("\nPress [Space] to start/stop recording, [c] to cancel recording, [q] to quit")

    process = None
    audio_frames = []    
    # Continuous lookahead checking for the Enter key press
    while True:
        if is_key_waiting():
            # Code halts here until exactly one key is pushed
            key = sys.stdin.read(1)
            if key == ' ':
                if process:
                    raw_print("\n[🟩️] Recorded.")
                    # Safe teardown of the system process
                    process.terminate()
                    process.wait()
                    # Drain any lingering audio fragments from the stream
                    leftovers = process.stdout.read()
                    process = None
                    if leftovers:
                        audio_frames.append(leftovers)
                    break
                else:
                    raw_print("\n[🔴️] Recording... Press [SPACE] again to STOP.")
                    # Fire up arecord as a non-root background stream running to stdout (-)
                    process = subprocess.Popen(
                        ['arecord', '-t', 'wav', '-f', 'S16_LE', '-r', str(RATE), '-c', str(CHANNELS), '-'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        restore_signals=True
                    )
            elif key.lower() == 'c':
                if process:
                    raw_print("\n[❌️] Recording canceled.")
                    # Safe teardown of the system process
                    process.terminate()
                    process.wait()
                    process = None
                    audio_frames = []
                    break
                else:
                    raw_print("\n[❓️] Not in recording.")
            elif key.lower() == 'q':
                raw_print("\n[⬛️] Shutting down pipeline.")
                audio_frames = None
                break
            else:
                # Safely ignore any unmapped keys the user accidentally hits
                pass

        if process:
            # Collect streaming data blocks from the binary pipeline pipe
            chunk = process.stdout.read(1024)
            if chunk:
                audio_frames.append(chunk)

        time.sleep(0.5)
    return audio_frames

def main(args):
    global client, voice_manager
    print("=== Voice Chat Assistant Initialized ===")

    if not os.environ.get('GEMINI_API_KEY'):
        print("Error: GEMINI_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    # project_root/src/raspi-monita -> project_root/resources/piper
    model_dir = Path(__file__).resolve().parent.parent.parent / "resources" / "piper"
    client = genai.Client()
    voice_manager = RealTimeVoiceManager(model_dir)

    # 1. Save original terminal settings so we can fix the console on exit
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    # 2. Re-hydrate any chat threads from a past run
    past_history = load_history()
    if past_history:
        raw_print(f"[🔄️] Resumed past conversation context ({len(past_history)} messages parsed).")

    # 3. Establish our official stateful Chat Session
    chat = client.chats.create(
        model=args.model,
        history=past_history,
        config=types.GenerateContentConfig(
            response_mime_type="application/json", # Tells the API to format as JSON
            response_schema=chat_schema,          # Enforces the specific key-value pairs
            system_instruction=(
                "Your name is Kuraspi. You are a witty, friendly, and highly capable Linux desktop voice assistant. "
                "Reply in the user's detected language. Only change languages if the user explicitly requests it. "
                "Never use robotic text placeholders or brackets like '[Insert Date Here]'. "
                "If the user asks you for real-time information (like the current time, date, or weather) "
                "and your Google Search tool is disabled, reply with a charming, natural human response, "
                "such as admitting you don't have a watch on you or that you can't look out the window right now."
            )
        )
    )
    
    try:
        # Puts the terminal into raw mode so keystrokes bypass line buffering
        tty.setraw(fd)

        while True:
            # 4. Record voice data and send them to Gemini API
            # Gather voice byte profile natively
            audio_frames = record_audio()
            if audio_frames is None:
                # terminated
                break
            if not audio_frames:
                # canceled
                continue
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
                    "Listen to this audio snippet. Transcribe it and address the core request directly.", 
                    audio_part
                ])
            except errors.APIError as e:
                raw_print(f"\n[💩️] Something went wrong: {e.code} {e.message}")
                speak_text_via_piper(e.message, "en")
                continue
            except httpx.TransportError as e:
                raw_print(f"\n[💩️] Transport failure: {e}")
                speak_text_via_piper(str(e), "en")
                continue
            
            raw_print("\n[🤖️] Gemini Response:")
            raw_print(response.text)
            raw_print("=" * 40)
            response_obj = json.loads(response.text)

            # 5. Generate the audio response using TTS
            if args.tts == "piper":
                speak_text_via_piper(response_obj['response'], response_obj['language_model'])
            elif args.tts == "espeak":
                speak_text_via_espeak(response_obj['response'], response_obj['language_model'], 140, 60, 3)

            # 6. Synchronize conversation timeline directly to disk
            save_history(chat)
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
    args = parser.parse_args()

    main(args)
