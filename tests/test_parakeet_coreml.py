"""Tests for the FluidAudio/CoreML Parakeet adapter."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from transcribe_anything.parakeet_coreml import convert_fluidaudio_result, model_version


class ParakeetCoreMLTest(unittest.TestCase):
    def test_model_selection_defaults_to_v3(self) -> None:
        self.assertEqual(model_version(""), "v3")
        self.assertEqual(model_version("small"), "v3")
        self.assertEqual(model_version("parakeet-v3"), "v3")
        self.assertEqual(model_version("parakeet-v2"), "v2")
        self.assertEqual(model_version("parakeet-110m"), "110m")

    def test_conversion_emits_calibrated_unreviewed_filler_events(self) -> None:
        raw = {
            "durationSeconds": 5.0,
            "modelVersion": "v3",
            "processingTimeSeconds": 0.1,
            "rtfx": 50.0,
            "text": "Okay. Um this works.",
            "wordTimings": [
                {"word": "Okay.", "startTime": 0.2, "endTime": 0.7, "confidence": 0.9},
                {"word": "Um", "startTime": 1.0, "endTime": 1.2, "confidence": 0.8},
                {"word": "this", "startTime": 1.5, "endTime": 1.8, "confidence": 0.95},
                {"word": "works.", "startTime": 1.9, "endTime": 2.4, "confidence": 0.97},
            ],
        }
        converted = convert_fluidaudio_result(raw)
        self.assertEqual(converted["backend"]["model"], "parakeet-v3-int8")
        self.assertIsNone(converted["backend"]["model_revision"])
        self.assertEqual(converted["backend"]["audited_model_revision"], "aed02740059203c4a87495924f685de3722ae9ce")
        self.assertFalse(converted["backend"]["model_revision_enforced"])
        self.assertEqual(len(converted["sentences"]), 2)
        self.assertEqual(converted["filler_events"], [
            {
                "surface": "um",
                "raw_surface": "Um",
                "start": 1.0,
                "end": 1.36,
                "raw_end": 1.2,
                "end_correction_s": 0.16,
                "confidence": 0.8,
                "event_type": "literal_filler",
                "editorial_decision": "unreviewed",
            }
        ])

    @patch("transcribe_anything.parakeet_coreml.platform.machine", return_value="x86_64")
    @patch("transcribe_anything.parakeet_coreml.platform.system", return_value="Darwin")
    def test_rejects_intel_mac(self, _system: object, _machine: object) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from transcribe_anything.parakeet_coreml import run_parakeet_coreml

        with TemporaryDirectory() as tmp, self.assertRaisesRegex(RuntimeError, "Apple Silicon"):
            run_parakeet_coreml(Path("audio.wav"), "", Path(tmp))


if __name__ == "__main__":
    unittest.main()
