import hashlib
import os
import tempfile
import tomllib
import urllib.request
from pathlib import Path

CHUNK_SIZE = 1024 * 1024  # 1 MiB

def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def voice_path(voice: dict) -> Path:
    directory = Path(
        voice["language"],
        voice["locale"],
        voice["speaker"],
        voice["quality"],
    )

    return directory / f"{voice['locale']}-{voice['speaker']}-{voice['quality']}.onnx"

def download_verified(
    url: str,
    target_dir: Path,
    expected_sha256: str,
) -> None:

    model_name = url.rsplit('/', 1)[-1]
    destination = target_dir / model_name

    if destination.exists():
        actual = sha256_file(destination)
        if actual.lower() == expected_sha256.lower():
            print(f"Already downloaded: {destination}")
            return

    print(f"Downloading {model_name} from {url} ...")
    with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}",
            suffix=".tmp",
            delete=False) as temporary:
        temporary_path = Path(temporary.name)

        try:
            digest = hashlib.sha256()
            with urllib.request.urlopen(url) as response:
                while chunk := response.read(CHUNK_SIZE):
                    temporary.write(chunk)
                    digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256.lower() != expected_sha256.lower():
                raise ValueError(f"SHA-256 mismatch for {url}")

            temporary.flush()
            os.fsync(temporary.fileno())
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    os.replace(temporary_path, destination)
    print(f"Downloaded: {destination}")

def download_voice(manifest: dict, voice: dict, target_dir: Path) -> None:
    base_url = manifest["repository"]["base_url"].rstrip("/")
    model_path = voice_path(voice)

    model_url = f"{base_url}/{model_path}"
    config_url = f"{base_url}/{model_path}.json"

    download_verified(
        model_url,
        target_dir,
        voice["model_sha256"],
    )

    download_verified(
        config_url,
        target_dir,
        voice["config_sha256"],
    )


project_root = Path(__file__).resolve().parent.parent
manifest_path = project_root / "resources" / "piper" / "manifest.toml"
target_dir = project_root / "resources" / "piper"

with manifest_path.open("rb") as f:
    manifest = tomllib.load(f)

for voice in manifest["voices"]:
    download_voice(manifest, voice, target_dir)
