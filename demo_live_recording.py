#!/usr/bin/env python3
"""
Demo script showing how to use the new live recording and transcription feature.

This script demonstrates the usage without actually running the audio recording
since we're in a containerized environment without audio hardware.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def demo_command_line_usage():
    """Show command line usage examples."""
    print("=== Live Recording and Transcription Demo ===\n")
    
    print("Command Line Usage Examples:")
    print("=" * 50)
    
    examples = [
        ("Basic live transcription", "transcribe-anything --live"),
        ("Live with custom model", "transcribe-anything --live --device mlx --model medium"),
        ("Live with custom vocabulary", 'transcribe-anything --live --initial_prompt "Meeting notes: project planning, deadlines"'),
        ("List audio devices", "transcribe-anything --list-devices"),
        ("Use specific device", "transcribe-anything --live --device-id 2"),
        ("Custom chunk settings", "transcribe-anything --live --chunk-duration 3.0 --overlap-duration 0.5"),
        ("Include desktop audio", "transcribe-anything --live --include-desktop-audio"),
        ("Custom output file", "transcribe-anything --live --live-output meeting_transcript.txt"),
    ]
    
    for description, command in examples:
        print(f"\n{description}:")
        print(f"  {command}")
    
    print("\n" + "=" * 50)

def demo_python_api():
    """Show Python API usage examples."""
    print("\nPython API Usage Examples:")
    print("=" * 50)
    
    api_examples = """
# Basic live transcription
from transcribe_anything.api import transcribe_live

transcribe_live(
    model="small",
    device="mlx",
    output_file="meeting_notes.txt"
)

# Advanced live transcription
transcribe_live(
    model="medium",
    device="insane",
    language="en",
    initial_prompt="Technical discussion about AI and machine learning",
    chunk_duration=3.0,
    overlap_duration=0.5,
    include_desktop_audio=False,
    output_file="live_transcript.txt"
)

# List available audio devices
from transcribe_anything.live_transcriber import LiveTranscriber
LiveTranscriber.list_audio_devices()
"""
    
    print(api_examples)
    print("=" * 50)

def demo_features():
    """Show key features of the live recording system."""
    print("\nKey Features:")
    print("=" * 50)
    
    features = [
        "Real-time audio recording from microphone",
        "Experimental desktop audio capture",
        "Overlapping audio chunks for continuous transcription",
        "Support for all Whisper backends (CPU, CUDA, MLX, Insane)",
        "Custom vocabulary support via initial_prompt",
        "Configurable chunk duration and overlap",
        "Audio device selection",
        "Live output to file with timestamps",
        "Cross-platform support (Windows, macOS, Linux)",
        "Thread-safe audio processing",
        "Graceful error handling and recovery"
    ]
    
    for i, feature in enumerate(features, 1):
        print(f"{i:2d}. {feature}")
    
    print("\n" + "=" * 50)

def demo_workflow():
    """Show the typical workflow."""
    print("\nTypical Workflow:")
    print("=" * 50)
    
    workflow = """
1. List available audio devices:
   transcribe-anything --list-devices

2. Start live transcription:
   transcribe-anything --live --device mlx --device-id 1

3. Speak into microphone - transcription appears in real-time

4. Press Ctrl+C to stop recording

5. Check output file (live_transcription.txt by default)

Example output file content:
[14:30:15] Hello, this is a test of the live transcription feature.
[14:30:22] The system is working correctly and capturing my speech.
[14:30:28] I can use custom vocabulary by providing an initial prompt.
"""
    
    print(workflow)
    print("=" * 50)

def demo_technical_details():
    """Show technical implementation details."""
    print("\nTechnical Implementation:")
    print("=" * 50)
    
    details = """
Architecture:
- LiveAudioRecorder: Handles microphone and desktop audio capture
- LiveTranscriber: Coordinates recording and transcription
- Threaded design: Separate threads for recording, transcription, and I/O
- Circular buffering: Prevents memory buildup during long sessions

Audio Processing:
- Default: 44.1kHz sample rate, mono audio
- Configurable chunk duration (default: 5 seconds)
- Overlapping chunks (default: 1 second overlap)
- Real-time conversion to WAV format for Whisper

Backend Integration:
- MLX: Apple Silicon optimized (recommended for Mac)
- Insane: GPU accelerated (recommended for Windows/Linux)
- CPU: Universal compatibility
- CUDA: Original OpenAI implementation

Platform-Specific Features:
- Windows: DirectShow for desktop audio
- macOS: AVFoundation for desktop audio
- Linux: PulseAudio for desktop audio
"""
    
    print(details)
    print("=" * 50)

if __name__ == "__main__":
    demo_command_line_usage()
    demo_python_api()
    demo_features()
    demo_workflow()
    demo_technical_details()
    
    print("\n🎉 Live Recording and Transcription Feature Demo Complete!")
    print("\nTo use this feature in a real environment:")
    print("1. Install: pip install transcribe-anything")
    print("2. Run: transcribe-anything --live")
    print("3. Start speaking and watch the live transcription!")
