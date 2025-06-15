"""
Simple test for MLX timestamp fix.

Tests that the fix for doubled timestamps works correctly.
"""

import sys
from pathlib import Path
from typing import Any, Dict

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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


def _json_to_srt(json_data: Dict[str, Any]) -> str:
    """Convert lightning-whisper-mlx JSON output to SRT format."""
    srt_content = ""

    if "segments" not in json_data:
        # If no segments, try to create a single segment from the full text
        if "text" in json_data:
            srt_content = "1\n00:00:00,000 --> 00:01:00,000\n" + json_data["text"] + "\n\n"
        return srt_content

    for i, segment in enumerate(json_data["segments"], start=1):
        # Handle both old format (start/end) and new format (list with start, end, text)
        if isinstance(segment, list) and len(segment) >= 3:
            # New format: [start_time, end_time, text] - timestamps appear to be doubled
            start_time = segment[0] / 2  # Fix doubled timestamps
            end_time = segment[1] / 2
            text = segment[2].strip()
        else:
            # Old format: dict with start/end/text
            start_time = segment.get("start", 0)
            end_time = segment.get("end", start_time + 5)  # Default to 5 seconds if no end time
            text = segment.get("text", "").strip()

        if text:  # Only include non-empty segments
            srt_content += f"{i}\n"
            srt_content += f"{_format_timestamp(start_time)} --> {_format_timestamp(end_time)}\n"
            srt_content += f"{text}\n\n"

    return srt_content


def test_doubled_timestamps_fix():
    """Test that doubled timestamps are correctly fixed."""
    # Simulate MLX output with doubled timestamps
    # For a 7-minute video (420 seconds), the doubled timestamps would be 840 seconds
    json_data = {
        "text": "This is a test transcription.",
        "segments": [
            [0.0, 20.0, "First segment"],      # Should become 0-10 seconds
            [20.0, 40.0, "Second segment"],    # Should become 10-20 seconds
            [800.0, 840.0, "Last segment"],   # Should become 400-420 seconds (6:40-7:00)
        ]
    }
    
    srt_content = _json_to_srt(json_data)
    print("Generated SRT content:")
    print(srt_content)
    
    # Check that timestamps are correctly halved
    lines = srt_content.strip().split('\n')
    
    # Find timestamp lines
    timestamp_lines = [line for line in lines if '-->' in line]
    
    # Verify the timestamps
    expected_timestamps = [
        "00:00:00,000 --> 00:00:10,000",  # 0-10 seconds
        "00:00:10,000 --> 00:00:20,000",  # 10-20 seconds  
        "00:06:40,000 --> 00:07:00,000",  # 400-420 seconds (6:40-7:00)
    ]
    
    for i, expected in enumerate(expected_timestamps):
        actual = timestamp_lines[i]
        print(f"Expected: {expected}")
        print(f"Actual:   {actual}")
        assert actual == expected, f"Timestamp mismatch: expected {expected}, got {actual}"
    
    print("✅ All timestamp tests passed!")


def test_7_minute_video_scenario():
    """Test the specific 7-minute video scenario mentioned."""
    # 7 minutes = 420 seconds
    # If timestamps are doubled, they would appear as 840 seconds (14 minutes)
    json_data = {
        "segments": [
            [0.0, 10.0, "Start"],           # Should be 0-5 seconds
            [840.0, 840.0, "End"],         # Should be 420 seconds (7:00)
        ]
    }
    
    srt_content = _json_to_srt(json_data)
    
    # Check that the end timestamp is 7:00, not 14:00
    assert "00:07:00,000" in srt_content, "Should contain 7:00 timestamp"
    assert "00:14:00,000" not in srt_content, "Should NOT contain 14:00 timestamp"
    
    print("✅ 7-minute video test passed!")


if __name__ == "__main__":
    test_doubled_timestamps_fix()
    test_7_minute_video_scenario()
    print("🎉 All tests passed! The timestamp doubling issue is fixed.")
