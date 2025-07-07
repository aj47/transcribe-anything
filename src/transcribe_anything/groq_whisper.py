"""
Groq speech-to-text API integration for transcribe-anything.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from groq import Groq

from transcribe_anything.logger import log_error


def run_groq_whisper(
    input_wav: Path,
    model: str,
    output_dir: Path,
    task: str = "transcribe",
    language: Optional[str] = None,
    api_key: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    other_args: Optional[list[str]] = None,
) -> None:
    """
    Transcribe audio using Groq's speech-to-text API.
    
    Args:
        input_wav: Path to the input audio file
        model: Groq model to use (whisper-large-v3, whisper-large-v3-turbo, distil-whisper-large-v3-en)
        output_dir: Directory to save output files
        task: Task to perform (transcribe or translate)
        language: Language of the audio (ISO-639-1 format, auto-detected if None)
        api_key: Groq API key (if None, will try to get from environment)
        initial_prompt: Optional prompt to guide the model's style
        other_args: Additional arguments (currently unused for Groq)
    """
    # Get API key from parameter or environment
    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError(
            "Groq API key is required. Set GROQ_API_KEY environment variable or pass api_key parameter."
        )
    
    # Initialize Groq client
    client = Groq(api_key=api_key)
    
    # Validate model
    valid_models = [
        "whisper-large-v3",
        "whisper-large-v3-turbo", 
        "distil-whisper-large-v3-en"
    ]
    if model not in valid_models:
        # Try to map common model names to Groq models
        model_mapping = {
            "large": "whisper-large-v3",
            "large-v3": "whisper-large-v3",
            "turbo": "whisper-large-v3-turbo",
            "small": "whisper-large-v3-turbo",  # Use turbo for better performance
            "medium": "whisper-large-v3",
            "tiny": "distil-whisper-large-v3-en",  # Use English-only for tiny
        }
        if model in model_mapping:
            original_model = model
            model = model_mapping[model]
            print(f"Mapping model '{original_model}' to Groq model '{model}'")
        else:
            print(f"Warning: Unknown model '{model}', defaulting to 'whisper-large-v3'")
            model = "whisper-large-v3"
    
    # Check file size and determine if chunking is needed
    file_size = input_wav.stat().st_size
    max_size = 20 * 1024 * 1024  # 20MB to be safe (Groq limit is 25MB, leave buffer for encoding overhead)
    needs_chunking = file_size > max_size
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Transcribing with Groq API using model: {model}")
    print(f"File size: {file_size / 1024 / 1024:.1f}MB")

    if needs_chunking:
        print(f"File exceeds {max_size / 1024 / 1024:.0f}MB safe limit (Groq max is 25MB), will process in chunks...")
        _transcribe_large_file_chunked(
            input_wav, model, output_dir, task, language,
            api_key, initial_prompt, other_args
        )
    else:
        _transcribe_single_file(
            input_wav, model, output_dir, task, language,
            api_key, initial_prompt, other_args
        )


def _transcribe_single_file(
    input_wav: Path,
    model: str,
    output_dir: Path,
    task: str = "transcribe",
    language: Optional[str] = None,
    api_key: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    other_args: Optional[list[str]] = None,
) -> None:
    """Transcribe a single audio file that fits within Groq's size limits."""
    client = Groq(api_key=api_key)

    try:
        # Prepare API call parameters
        transcription_params = {
            "model": model,
            "response_format": "verbose_json",  # Get timestamps and metadata
            "temperature": 0.0,  # Use deterministic output
        }
        
        # Add optional parameters
        if language:
            transcription_params["language"] = language
            
        if initial_prompt:
            # Groq limits prompts to 224 tokens
            if len(initial_prompt) > 1000:  # Rough token estimate
                initial_prompt = initial_prompt[:1000]
                print("Warning: Initial prompt truncated to fit Groq's 224 token limit")
            transcription_params["prompt"] = initial_prompt
        
        # Add timestamp granularities for detailed output
        transcription_params["timestamp_granularities"] = ["word", "segment"]
        
        # Open and transcribe the audio file
        with open(input_wav, "rb") as audio_file:
            if task == "translate":
                # Use translation endpoint
                transcription = client.audio.translations.create(
                    file=audio_file,
                    **{k: v for k, v in transcription_params.items() if k != "timestamp_granularities"}
                )
            else:
                # Use transcription endpoint
                transcription = client.audio.transcriptions.create(
                    file=audio_file,
                    **transcription_params
                )
        
        # Convert response to our standard format
        if hasattr(transcription, 'segments'):
            # Verbose JSON response with segments
            segments = transcription.segments
            words = getattr(transcription, 'words', [])
        else:
            # Simple text response - create a single segment
            segments = [{
                "id": 0,
                "seek": 0,
                "start": 0.0,
                "end": 0.0,
                "text": transcription.text,
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": 0.0,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0
            }]
            words = []
        
        # Create JSON output in the same format as other backends
        json_output = {
            "text": transcription.text,
            "segments": segments,
            "language": getattr(transcription, 'language', language or 'unknown'),
            "words": words
        }
        
        # Save JSON output
        json_file = output_dir / "out.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        
        # Generate other output formats
        _generate_output_formats(json_output, output_dir)
        
        print(f"Groq transcription completed successfully")
        print(f"Output saved to: {output_dir}")
        
    except Exception as e:
        error_msg = f"Groq API error: {str(e)}"
        log_error(error_msg)
        raise RuntimeError(error_msg) from e


def _transcribe_large_file_chunked(
    input_wav: Path,
    model: str,
    output_dir: Path,
    task: str = "transcribe",
    language: Optional[str] = None,
    api_key: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    other_args: Optional[list[str]] = None,
) -> None:
    """Transcribe a large audio file by splitting it into chunks."""
    client = Groq(api_key=api_key)

    # Get audio duration to calculate chunk size
    duration = _get_audio_duration(input_wav)
    if duration is None:
        raise RuntimeError("Could not determine audio duration for chunking")

    # Calculate chunk duration (aim for ~15MB chunks, estimate 1MB per minute for WAV)
    chunk_duration_minutes = 15  # Start with 15 minutes per chunk to stay under 25MB limit
    max_chunk_size_mb = 15

    print(f"Audio duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")

    # Split audio into chunks
    chunks = _split_audio_into_chunks(input_wav, chunk_duration_minutes * 60)

    print(f"Processing {len(chunks)} chunks...")

    all_segments = []
    all_words = []
    full_text_parts = []
    current_time_offset = 0.0

    try:
        for i, chunk_path in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")

            # Check chunk size
            chunk_size = chunk_path.stat().st_size
            if chunk_size > 20 * 1024 * 1024:  # Still too large for Groq's 25MB limit
                print(f"Warning: Chunk {i+1} is still {chunk_size / 1024 / 1024:.1f}MB, may fail (Groq limit is 25MB)")

            # Transcribe this chunk
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_output_dir = Path(temp_dir)
                _transcribe_single_file(
                    chunk_path, model, temp_output_dir, task, language,
                    api_key, initial_prompt, other_args
                )

                # Read the chunk results
                chunk_json_file = temp_output_dir / "out.json"
                if chunk_json_file.exists():
                    with open(chunk_json_file, 'r', encoding='utf-8') as f:
                        chunk_data = json.load(f)

                    # Adjust timestamps and collect segments
                    if 'segments' in chunk_data:
                        for segment in chunk_data['segments']:
                            segment['start'] += current_time_offset
                            segment['end'] += current_time_offset
                            all_segments.append(segment)

                    # Adjust word timestamps if present
                    if 'words' in chunk_data:
                        for word in chunk_data['words']:
                            if 'start' in word:
                                word['start'] += current_time_offset
                            if 'end' in word:
                                word['end'] += current_time_offset
                            all_words.append(word)

                    # Collect text
                    if 'text' in chunk_data:
                        full_text_parts.append(chunk_data['text'].strip())

            # Update time offset for next chunk (with small overlap buffer)
            current_time_offset += chunk_duration_minutes * 60

        # Combine all results
        combined_result = {
            "text": " ".join(full_text_parts),
            "segments": all_segments,
            "language": language or "unknown",
            "words": all_words
        }

        # Save combined JSON output
        json_file = output_dir / "out.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(combined_result, f, indent=2, ensure_ascii=False)

        # Generate other output formats
        _generate_output_formats(combined_result, output_dir)

        print(f"Chunked transcription completed successfully")
        print(f"Total segments: {len(all_segments)}")
        print(f"Output saved to: {output_dir}")

    finally:
        # Clean up chunk files
        for chunk_path in chunks:
            try:
                chunk_path.unlink()
            except Exception:
                pass


def _get_audio_duration(audio_file: Path) -> Optional[float]:
    """Get audio duration in seconds using ffprobe."""
    try:
        # Try ffprobe first
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(audio_file)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except FileNotFoundError:
        print("Warning: ffprobe not found, trying static_ffmpeg...")
        try:
            # Try static_ffmpeg if available
            cmd = [
                "static_ffmpeg", "-i", str(audio_file), "-f", "null", "-"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            # Parse duration from stderr output
            for line in result.stderr.split('\n'):
                if 'Duration:' in line:
                    duration_str = line.split('Duration:')[1].split(',')[0].strip()
                    # Parse HH:MM:SS.mmm format
                    parts = duration_str.split(':')
                    if len(parts) == 3:
                        hours = float(parts[0])
                        minutes = float(parts[1])
                        seconds = float(parts[2])
                        return hours * 3600 + minutes * 60 + seconds
        except Exception as e:
            print(f"Warning: Could not get audio duration with static_ffmpeg: {e}")
    except Exception as e:
        print(f"Warning: Could not get audio duration: {e}")
    return None


def _split_audio_into_chunks(input_file: Path, chunk_duration: float) -> List[Path]:
    """Split audio file into chunks of specified duration."""
    chunks = []
    chunk_index = 0

    # Get total duration
    total_duration = _get_audio_duration(input_file)
    if total_duration is None:
        raise RuntimeError("Cannot split audio: duration unknown")

    current_start = 0.0
    temp_dir = Path(tempfile.mkdtemp())

    while current_start < total_duration:
        chunk_end = min(current_start + chunk_duration, total_duration)
        chunk_file = temp_dir / f"chunk_{chunk_index:03d}.wav"

        # Use ffmpeg to extract chunk (try static_ffmpeg first, then ffmpeg)
        for ffmpeg_cmd in ["static_ffmpeg", "ffmpeg"]:
            cmd = [
                ffmpeg_cmd, "-y", "-i", str(input_file),
                "-ss", str(current_start),
                "-t", str(chunk_end - current_start),
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                str(chunk_file)
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, timeout=300)
                if result.returncode == 0 and chunk_file.exists():
                    chunks.append(chunk_file)
                    print(f"Created chunk {chunk_index}: {current_start:.1f}s - {chunk_end:.1f}s")
                    break
                else:
                    if ffmpeg_cmd == "ffmpeg":  # Last attempt failed
                        print(f"Warning: Failed to create chunk {chunk_index}")
            except FileNotFoundError:
                if ffmpeg_cmd == "ffmpeg":  # Last attempt, command not found
                    raise RuntimeError("Neither static_ffmpeg nor ffmpeg found. Cannot chunk large audio files.")
                continue  # Try next command
            except subprocess.TimeoutExpired:
                print(f"Warning: Timeout creating chunk {chunk_index}")
                break

        current_start = chunk_end
        chunk_index += 1

    return chunks


def _generate_output_formats(json_data: dict, output_dir: Path) -> None:
    """Generate SRT, VTT, and TXT files from JSON transcription data."""
    
    # Generate TXT file
    txt_file = output_dir / "out.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(json_data["text"])
    
    # Generate SRT file
    srt_file = output_dir / "out.srt"
    _generate_srt(json_data["segments"], srt_file)
    
    # Generate VTT file  
    vtt_file = output_dir / "out.vtt"
    _generate_vtt(json_data["segments"], vtt_file)


def _generate_srt(segments: list, output_file: Path) -> None:
    """Generate SRT subtitle file from segments."""
    with open(output_file, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, 1):
            start_time = _format_timestamp_srt(segment.get("start", 0))
            end_time = _format_timestamp_srt(segment.get("end", 0))
            text = segment.get("text", "").strip()
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")


def _generate_vtt(segments: list, output_file: Path) -> None:
    """Generate VTT subtitle file from segments."""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        
        for segment in segments:
            start_time = _format_timestamp_vtt(segment.get("start", 0))
            end_time = _format_timestamp_vtt(segment.get("end", 0))
            text = segment.get("text", "").strip()
            
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")


def _format_timestamp_srt(seconds: float) -> str:
    """Format timestamp for SRT format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format timestamp for VTT format (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"


def get_groq_models() -> list[str]:
    """Get list of available Groq models."""
    return [
        "whisper-large-v3",
        "whisper-large-v3-turbo", 
        "distil-whisper-large-v3-en"
    ]


def validate_groq_api_key(api_key: Optional[str] = None) -> bool:
    """Validate Groq API key by making a test request."""
    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        return False
    
    try:
        client = Groq(api_key=api_key)
        # Try to list models to validate the key
        models = client.models.list()
        return True
    except Exception:
        return False
