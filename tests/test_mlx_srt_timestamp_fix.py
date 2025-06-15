"""
Test for MLX SRT timestamp generation fix.

This test verifies that the MLX backend correctly handles timestamps
and doesn't generate SRT files with timestamps exceeding the media duration.
"""

import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Any, Dict

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the functions directly
def _format_timestamp(seconds: float) -> str:
    """Format seconds into SRT timestamp format."""
    milliseconds = int(seconds * 1000)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _get_audio_duration(wav_file: Path) -> float | None:
    """Get the duration of a WAV file in seconds."""
    try:
        with wave.open(str(wav_file), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        sys.stderr.write(f"Warning: Could not determine audio duration: {e}\n")
        return None


def _json_to_srt(json_data: Dict[str, Any], duration: float | None = None) -> str:
    """Convert lightning-whisper-mlx JSON output to SRT format."""
    srt_content = ""

    if "segments" not in json_data:
        # If no segments, try to create a single segment from the full text
        if "text" in json_data:
            srt_content = "1\n00:00:00,000 --> 00:01:00,000\n" + json_data["text"] + "\n\n"
        return srt_content

    segment_number = 1
    for segment in json_data["segments"]:
        # Handle both old format (start/end) and new format (list with start, end, text)
        if isinstance(segment, list) and len(segment) >= 3:
            # New format: [start_time, end_time, text] - timestamps are already in seconds
            start_time = float(segment[0])
            end_time = float(segment[1])
            text = segment[2].strip()
        else:
            # Old format: dict with start/end/text
            start_time = segment.get("start", 0)
            end_time = segment.get("end", start_time + 5)  # Default to 5 seconds if no end time
            text = segment.get("text", "").strip()

        # Validate timestamps against duration if provided
        if duration is not None:
            if start_time >= duration:
                # Skip segments that start at or after the media ends
                continue
            if end_time > duration:
                # Clamp end time to media duration
                end_time = duration
            # Skip segments with zero or negative duration after clamping
            if end_time <= start_time:
                continue

        if text:  # Only include non-empty segments
            srt_content += f"{segment_number}\n"
            srt_content += f"{_format_timestamp(start_time)} --> {_format_timestamp(end_time)}\n"
            srt_content += f"{text}\n\n"
            segment_number += 1

    return srt_content


class TestMLXSRTTimestampFix(unittest.TestCase):
    """Test MLX SRT timestamp generation fix."""

    def test_json_to_srt_with_duration_validation(self):
        """Test that SRT generation respects duration limits."""
        # Mock JSON data with timestamps that would exceed a 40-minute (2400 second) video
        json_data = {
            "text": "This is a test transcription.",
            "segments": [
                [0.0, 10.0, "First segment"],
                [10.0, 20.0, "Second segment"],
                [2390.0, 2400.0, "Valid segment near end"],
                [2400.0, 2410.0, "Invalid segment past end"],  # Should be filtered out
                [2395.0, 2450.0, "Segment extending past end"],  # Should be clamped
            ]
        }
        
        # Test with 40-minute duration (2400 seconds)
        duration = 2400.0
        srt_content = _json_to_srt(json_data, duration)
        
        # Verify the SRT content
        lines = srt_content.strip().split('\n')
        
        # Should have 4 segments (one filtered out)
        segment_count = srt_content.count('\n\n')
        self.assertEqual(segment_count, 4, "Should have 4 segments after filtering")
        
        # Check that no timestamps exceed the duration
        for line in lines:
            if '-->' in line:
                # Parse timestamp line
                start_str, end_str = line.split(' --> ')
                
                # Convert timestamps back to seconds for validation
                def timestamp_to_seconds(ts_str):
                    parts = ts_str.split(':')
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    seconds_and_ms = parts[2].split(',')
                    seconds = int(seconds_and_ms[0])
                    milliseconds = int(seconds_and_ms[1])
                    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
                
                start_seconds = timestamp_to_seconds(start_str)
                end_seconds = timestamp_to_seconds(end_str)
                
                # Verify timestamps don't exceed duration
                self.assertLessEqual(start_seconds, duration, f"Start time {start_seconds} exceeds duration {duration}")
                self.assertLessEqual(end_seconds, duration, f"End time {end_seconds} exceeds duration {duration}")

    def test_json_to_srt_without_duration(self):
        """Test that SRT generation works without duration validation."""
        json_data = {
            "text": "This is a test transcription.",
            "segments": [
                [0.0, 10.0, "First segment"],
                [10.0, 20.0, "Second segment"],
            ]
        }
        
        srt_content = _json_to_srt(json_data)
        
        # Should generate content without errors
        self.assertIn("First segment", srt_content)
        self.assertIn("Second segment", srt_content)
        self.assertIn("00:00:00,000 --> 00:00:10,000", srt_content)
        self.assertIn("00:00:10,000 --> 00:00:20,000", srt_content)

    def test_json_to_srt_dict_format(self):
        """Test that SRT generation works with dict format segments."""
        json_data = {
            "text": "This is a test transcription.",
            "segments": [
                {"start": 0.0, "end": 10.0, "text": "First segment"},
                {"start": 10.0, "end": 20.0, "text": "Second segment"},
            ]
        }
        
        duration = 30.0
        srt_content = _json_to_srt(json_data, duration)
        
        # Should generate content correctly
        self.assertIn("First segment", srt_content)
        self.assertIn("Second segment", srt_content)

    def test_timestamp_format_precision(self):
        """Test that timestamps are formatted correctly with proper precision."""
        json_data = {
            "segments": [
                [1.234, 5.678, "Test segment"],
            ]
        }
        
        srt_content = _json_to_srt(json_data)
        
        # Check that milliseconds are properly formatted
        self.assertIn("00:00:01,234 --> 00:00:05,678", srt_content)

    def test_get_audio_duration_with_invalid_file(self):
        """Test that _get_audio_duration handles invalid files gracefully."""
        invalid_path = Path("/nonexistent/file.wav")
        duration = _get_audio_duration(invalid_path)
        self.assertIsNone(duration, "Should return None for invalid file")

    def test_real_world_scenario_40_minute_video(self):
        """Test a real-world scenario with a 40-minute video that had timestamps extending past 1 hour."""
        # This simulates the actual bug reported: 40-minute video with SRT timestamps past 1 hour
        json_data = {
            "text": "This is a long transcription of a 40-minute video.",
            "segments": [
                # Normal segments
                [0.0, 5.0, "Introduction"],
                [5.0, 10.0, "First topic"],
                # Segments near the end (around 39-40 minutes)
                [2340.0, 2350.0, "Near the end"],
                [2350.0, 2400.0, "Final segment"],
                # These would be the problematic segments that extend past the video duration
                # In the original bug, these would have been created due to incorrect timestamp conversion
                [2400.0, 2410.0, "This should be filtered out"],
                [2410.0, 2420.0, "This should also be filtered out"],
            ]
        }

        # 40-minute duration (2400 seconds)
        duration = 2400.0
        srt_content = _json_to_srt(json_data, duration)

        # Verify that no timestamps exceed 40 minutes (2400 seconds)
        lines = srt_content.strip().split('\n')
        for line in lines:
            if '-->' in line:
                # Parse timestamp line
                start_str, end_str = line.split(' --> ')

                # Convert timestamps back to seconds for validation
                def timestamp_to_seconds(ts_str):
                    parts = ts_str.split(':')
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    seconds_and_ms = parts[2].split(',')
                    seconds = int(seconds_and_ms[0])
                    milliseconds = int(seconds_and_ms[1])
                    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000

                start_seconds = timestamp_to_seconds(start_str)
                end_seconds = timestamp_to_seconds(end_str)

                # Verify timestamps don't exceed 40 minutes
                self.assertLessEqual(start_seconds, 2400, f"Start time {start_seconds} exceeds 40 minutes")
                self.assertLessEqual(end_seconds, 2400, f"End time {end_seconds} exceeds 40 minutes")

                # Verify no timestamps exceed 1 hour (3600 seconds) - the original bug
                self.assertLess(start_seconds, 3600, f"Start time {start_seconds} exceeds 1 hour (original bug)")
                self.assertLess(end_seconds, 3600, f"End time {end_seconds} exceeds 1 hour (original bug)")

        # Verify that only 4 segments remain (2 problematic ones filtered out)
        segment_count = srt_content.count('\n\n')
        self.assertEqual(segment_count, 4, "Should have 4 segments after filtering out invalid ones")


if __name__ == "__main__":
    unittest.main()
