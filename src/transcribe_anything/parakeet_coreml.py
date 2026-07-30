"""Run the pinned FluidAudio/CoreML Parakeet backend on Apple Silicon."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from transcribe_anything.parakeet_mlx import _generate_output_files

FLUIDAUDIO_IMPLEMENTATION_REVISION = "88d6d8166880dee1ac7c32c80f8e10cd782f8ca8"
COREML_V3_AUDITED_MODEL_REVISION = "aed02740059203c4a87495924f685de3722ae9ce"
DEFAULT_BINARY = Path.home() / ".local" / "lib" / "transcribe-anything" / "FluidAudioCLI"
FILLER_END_CORRECTION_S = {"um": 0.16, "uh": 0.14}
FILLER_FALLBACK_END_CORRECTION_S = 0.15
FILLER_SURFACES = {
    "um": "um",
    "umm": "um",
    "uhm": "um",
    "umh": "um",
    "uh": "uh",
    "ah": "ah",
    "hm": "mmm",
    "hmm": "mmm",
    "mm": "mmm",
    "mmm": "mmm",
}


def resolve_fluidaudio_binary() -> Path:
    """Resolve the audited FluidAudio CLI without downloading executables."""
    configured = os.environ.get("TRANSCRIBE_ANYTHING_FLUIDAUDIO_BINARY")
    candidates = [Path(configured).expanduser()] if configured else []
    for command in ("fluidaudio", "FluidAudioCLI"):
        found = shutil.which(command)
        if found:
            candidates.append(Path(found))
    candidates.append(DEFAULT_BINARY)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise RuntimeError(
        "FluidAudioCLI is required for --device parakeet. Install the audited "
        f"binary at {DEFAULT_BINARY}, put fluidaudio on PATH, or set "
        "TRANSCRIBE_ANYTHING_FLUIDAUDIO_BINARY. Use --device parakeet-mlx "
        "only as an explicit rollback."
    )


def model_version(model: str) -> str:
    """Map an optional Parakeet model selector to FluidAudio's CLI value."""
    normalized = (model or "").lower()
    if normalized in {"parakeet-v2", "v2"}:
        return "v2"
    if normalized in {"parakeet-110m", "parakeet-tdt-ctc-110m", "110m"}:
        return "110m"
    return "v3"


def _display_text(words: list[dict[str, Any]]) -> str:
    return " ".join(str(word["word"]).strip() for word in words).strip()


def _sentences_from_words(raw_words: list[dict[str, Any]], full_text: str) -> list[dict[str, Any]]:
    """Group lexical CoreML words into compact timestamped caption sentences."""
    words = [
        {
            "word": str(word["word"]).strip(),
            "start": float(word["startTime"]),
            "end": float(word["endTime"]),
            "confidence": word.get("confidence"),
        }
        for word in raw_words
        if str(word.get("word", "")).strip()
    ]
    if not words:
        return [{"text": full_text, "start": 0.0, "end": 0.0, "duration": 0.0, "words": []}] if full_text else []

    sentences: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for index, word in enumerate(words):
        current.append(word)
        next_word = words[index + 1] if index + 1 < len(words) else None
        duration = current[-1]["end"] - current[0]["start"]
        gap = next_word["start"] - word["end"] if next_word else 0.0
        terminal = bool(re.search(r"[.!?][\"')\]]*$", word["word"]))
        should_close = next_word is None or terminal or gap >= 0.8 or duration >= 8.0 or len(_display_text(current)) >= 120
        if not should_close:
            continue
        text = _display_text(current)
        start = current[0]["start"]
        end = current[-1]["end"]
        sentences.append({"text": text, "start": start, "end": end, "duration": end - start, "words": current})
        current = []
    return sentences


def _normalize_filler(value: str) -> str | None:
    cleaned = re.sub(r"[^a-z]", "", value.lower())
    return FILLER_SURFACES.get(cleaned)


def _filler_events(sentences: list[dict[str, Any]], duration_s: float) -> list[dict[str, Any]]:
    """Emit audible filler candidates without making an editorial decision."""
    events = []
    for sentence in sentences:
        for word in sentence["words"]:
            surface = _normalize_filler(word["word"])
            if surface is None:
                continue
            raw_end = float(word["end"])
            correction = FILLER_END_CORRECTION_S.get(surface, FILLER_FALLBACK_END_CORRECTION_S)
            events.append(
                {
                    "surface": surface,
                    "raw_surface": word["word"],
                    "start": float(word["start"]),
                    "end": round(min(duration_s, raw_end + correction), 3),
                    "raw_end": raw_end,
                    "end_correction_s": correction,
                    "confidence": word.get("confidence"),
                    "event_type": "literal_filler",
                    "editorial_decision": "unreviewed",
                }
            )
    return events


def convert_fluidaudio_result(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert FluidAudio JSON to transcribe-anything's stable output shape."""
    duration_s = float(raw.get("durationSeconds", 0.0))
    sentences = _sentences_from_words(raw.get("wordTimings", []), str(raw.get("text", "")))
    version = str(raw.get("modelVersion", "v3"))
    return {
        "schema_version": "transcribe_anything.parakeet_coreml.v1",
        "text": str(raw.get("text", "")),
        "sentences": sentences,
        "filler_events": _filler_events(sentences, duration_s),
        "backend": {
            "implementation": "FluidAudio/CoreML",
            "implementation_revision": FLUIDAUDIO_IMPLEMENTATION_REVISION,
            "model": f"parakeet-{version}-int8",
            "model_revision": None,
            "audited_model_revision": COREML_V3_AUDITED_MODEL_REVISION if version == "v3" else None,
            "model_revision_enforced": False,
            "encoder_precision": "int8",
            "filler_boundary_calibration": {
                "um_end_s": FILLER_END_CORRECTION_S["um"],
                "uh_end_s": FILLER_END_CORRECTION_S["uh"],
                "fallback_end_s": FILLER_FALLBACK_END_CORRECTION_S,
                "fit_role": "isolated teacher-development data; detection remains separate from removal",
            },
        },
        "performance": {
            "audio_duration_s": duration_s,
            "inference_s": raw.get("processingTimeSeconds"),
            "rtfx": raw.get("rtfx"),
        },
    }


def run_parakeet_coreml(
    input_wav: Path,
    model: str,
    output_dir: Path,
    language: str | None = None,
    other_args: list[str] | None = None,
) -> None:
    """Transcribe with FluidAudio/CoreML Parakeet and write standard artifacts."""
    del language, other_args
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("FluidAudio/CoreML Parakeet requires an Apple Silicon Mac")
    output_dir.mkdir(parents=True, exist_ok=True)
    binary = resolve_fluidaudio_binary()
    with tempfile.TemporaryDirectory(prefix="transcribe-anything-coreml-") as tmp:
        raw_path = Path(tmp) / "fluidaudio.json"
        command = [
            str(binary),
            "transcribe",
            str(input_wav.resolve()),
            "--model-version",
            model_version(model),
            "--encoder-precision",
            "int8",
            "--metadata",
            "--word-timestamps",
            "--output-json",
            str(raw_path),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode:
            raise RuntimeError(
                f"FluidAudioCLI failed with code {completed.returncode}\n"
                f"STDOUT: {completed.stdout}\nSTDERR: {completed.stderr}"
            )
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    converted = convert_fluidaudio_result(raw)
    _generate_output_files(converted, output_dir)
