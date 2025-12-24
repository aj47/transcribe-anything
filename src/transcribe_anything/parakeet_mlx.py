"""
Runs transcription using parakeet-mlx on Apple Silicon.

Uses FluidAudio's Parakeet TDT models via the parakeet-mlx Python library.
- parakeet-tdt-0.6b-v2: English-only, highest recall
- parakeet-tdt-0.6b-v3: Multilingual (25 European languages)
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import webvtt  # type: ignore
from iso_env import IsoEnv, IsoEnvArgs, PyProjectToml  # type: ignore

HERE = Path(__file__).parent

# Model mappings
PARAKEET_MODELS = {
    "parakeet-v2": "mlx-community/parakeet-tdt-0.6b-v2",  # English-only, highest recall
    "parakeet-v3": "mlx-community/parakeet-tdt-0.6b-v3",  # Multilingual
    "parakeet": "mlx-community/parakeet-tdt-0.6b-v3",  # Default to v3 (multilingual)
}


def get_parakeet_cache_dir() -> Path:
    """Get the cache directory for Parakeet models."""
    cache_dir = Path.home() / ".cache" / "whisper" / "parakeet_models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_environment() -> IsoEnv:
    """Returns the environment for parakeet-mlx."""
    venv_dir = HERE / "venv" / "parakeet_mlx"
    content_lines: list[str] = []

    content_lines.append("[build-system]")
    content_lines.append('requires = ["setuptools", "wheel"]')
    content_lines.append('build-backend = "setuptools.build_meta"')
    content_lines.append("")
    content_lines.append("[project]")
    content_lines.append('name = "project"')
    content_lines.append('version = "0.1.0"')
    content_lines.append('requires-python = ">=3.11"')
    content_lines.append("dependencies = [")
    # Pin parakeet-mlx to latest version that should work with Python 3.11+
    # The package removed librosa dependency in newer versions
    content_lines.append('  "parakeet-mlx>=0.4.0",')
    content_lines.append('  "webvtt-py",')
    content_lines.append('  "numpy",')
    content_lines.append("]")
    content_lines.append("")
    # Override dependency resolution to avoid numba/llvmlite issues
    content_lines.append("[tool.uv]")
    content_lines.append("override-dependencies = [")
    content_lines.append('  "numba>=0.60.0",')
    content_lines.append('  "llvmlite>=0.43.0",')
    content_lines.append("]")
    content = "\n".join(content_lines)

    pyproject_toml = PyProjectToml(content)
    args = IsoEnvArgs(venv_dir, build_info=pyproject_toml)
    env = IsoEnv(args)
    return env


def _format_timestamp(seconds: float) -> str:
    """Format seconds into SRT timestamp format."""
    milliseconds = int(seconds * 1000)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    secs = milliseconds // 1_000
    milliseconds -= secs * 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def _json_to_srt(json_data: Dict[str, Any]) -> str:
    """Convert parakeet-mlx JSON output to SRT format."""
    srt_content = ""

    sentences = json_data.get("sentences", [])
    if not sentences:
        # Fallback: create single segment from full text
        if "text" in json_data:
            srt_content = "1\n00:00:00,000 --> 00:01:00,000\n" + json_data["text"] + "\n\n"
        return srt_content

    for i, sentence in enumerate(sentences, start=1):
        start_time = sentence.get("start", 0)
        end_time = sentence.get("end", start_time + 5)
        text = sentence.get("text", "").strip()

        if text:
            srt_content += f"{i}\n"
            srt_content += f"{_format_timestamp(start_time)} --> {_format_timestamp(end_time)}\n"
            srt_content += f"{text}\n\n"

    return srt_content


def _generate_output_files(json_data: Dict[str, Any], output_dir: Path) -> None:
    """Generate all output files from the transcription result."""
    # 1. SRT file
    srt_content = _json_to_srt(json_data)
    srt_file = output_dir / "out.srt"
    with open(srt_file, "w", encoding="utf-8") as f:
        f.write(srt_content)

    # 2. Text file
    txt_file = output_dir / "out.txt"
    text_content = json_data.get("text", "")
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(text_content)

    # 3. JSON file
    json_out_file = output_dir / "out.json"
    with open(json_out_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    # 4. VTT file
    try:
        vtt_file = output_dir / "out.vtt"
        webvtt.from_srt(srt_file).save(vtt_file)
    except Exception as e:
        sys.stderr.write(f"Warning: Failed to convert SRT to VTT: {e}\n")


def run_parakeet_mlx(
    input_wav: Path,
    model: str,
    output_dir: Path,
    language: Optional[str] = None,
    other_args: Optional[list[str]] = None,
) -> None:
    """Runs transcription with parakeet-mlx on Apple Silicon.

    Args:
        input_wav: Path to the input WAV file
        model: Model to use (parakeet, parakeet-v2, parakeet-v3, or HF repo ID)
        output_dir: Directory to save output files
        language: Language code (only affects model selection - v2 for English, v3 for others)
        other_args: Additional arguments (currently unused)
    """
    input_wav_abs = input_wav.resolve()
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Determine the model repo ID
    if model in PARAKEET_MODELS:
        repo_id = PARAKEET_MODELS[model]
    elif model and "/" in model:
        # Assume it's a HuggingFace repo ID
        repo_id = model
    else:
        # Default: use v2 for English, v3 for other languages
        if language and language.lower() in ("en", "english"):
            repo_id = PARAKEET_MODELS["parakeet-v2"]
        else:
            repo_id = PARAKEET_MODELS["parakeet-v3"]

    # Get the environment
    env = get_environment()

    # Create a Python script to run the transcription in the isolated environment
    script_content = f'''
import sys
import json

try:
    from parakeet_mlx import from_pretrained

    # Load the model
    model = from_pretrained("{repo_id}")

    # Transcribe the audio file
    result = model.transcribe(
        "{input_wav_abs}",
        chunk_duration=120.0,  # 2 minute chunks for long audio
        overlap_duration=15.0,  # 15 second overlap
    )

    # Build output JSON with sentences for SRT generation
    output = {{
        "text": result.text,
        "sentences": [
            {{
                "text": s.text,
                "start": s.start,
                "end": s.end,
                "duration": s.duration,
            }}
            for s in result.sentences
        ]
    }}

    # Print the result as JSON
    print(json.dumps(output, ensure_ascii=False))

except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''

    # Write the script to a temporary file
    script_file = output_dir / "transcribe_parakeet_script.py"
    with open(script_file, "w", encoding="utf-8") as f:
        f.write(script_content)

    try:
        # Execute the script in the isolated environment
        sys.stderr.write(f"Running parakeet-mlx transcription on {input_wav_abs}\n")
        sys.stderr.write(f"Using model: {repo_id}\n")

        result = env.run([str(script_file)], shell=False, check=False, capture_output=True, text=True)

        # Check for errors
        if result.returncode != 0:
            error_msg = f"parakeet-mlx script failed with return code {result.returncode}"
            if result.stderr:
                error_msg += f"\nSTDERR: {result.stderr}"
            if result.stdout:
                error_msg += f"\nSTDOUT: {result.stdout}"
            raise RuntimeError(error_msg)

        # Parse the JSON output
        try:
            json_data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse parakeet-mlx output JSON: {e}"
            if result.stderr:
                error_msg += f"\nSTDERR: {result.stderr}"
            if result.stdout:
                error_msg += f"\nSTDOUT: {result.stdout}"
            raise ValueError(error_msg)

        # Generate output files
        _generate_output_files(json_data, output_dir)

    finally:
        # Clean up the temporary script
        if script_file.exists():
            script_file.unlink()

