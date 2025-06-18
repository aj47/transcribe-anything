"""
Live transcription coordinator that manages recording and transcription.
"""

import json
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import queue

from transcribe_anything.live_recorder import LiveAudioRecorder
from transcribe_anything.whisper import get_computing_device, run_whisper
from transcribe_anything.whisper_mac import run_whisper_mac_mlx
from transcribe_anything.insanely_fast_whisper import run_insanely_fast_whisper
from transcribe_anything.api import Device


class LiveTranscriber:
    """Coordinates live audio recording and transcription."""
    
    def __init__(
        self,
        model: str = "small",
        device: Optional[str] = None,
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
        chunk_duration: float = 5.0,
        overlap_duration: float = 1.0,
        include_desktop_audio: bool = False,
        device_id: Optional[int] = None,
        output_file: Optional[str] = None,
        hugging_face_token: Optional[str] = None,
        other_args: Optional[list] = None
    ):
        """
        Initialize the live transcriber.
        
        Args:
            model: Whisper model to use
            device: Device to use for transcription (cuda, cpu, insane, mlx)
            language: Language of the audio
            task: Task to perform (transcribe or translate)
            initial_prompt: Initial prompt for better recognition
            chunk_duration: Duration of each audio chunk in seconds
            overlap_duration: Overlap between chunks in seconds
            include_desktop_audio: Whether to include desktop audio
            device_id: Specific audio device ID to use
            output_file: File to write live transcriptions to
            hugging_face_token: Token for speaker diarization
            other_args: Additional arguments for whisper backend
        """
        self.model = model
        self.device = device or get_computing_device()
        self.language = language
        self.task = task
        self.initial_prompt = initial_prompt
        self.hugging_face_token = hugging_face_token
        self.other_args = other_args or []
        
        # Set up output file
        if output_file:
            self.output_file = Path(output_file)
        else:
            self.output_file = Path("live_transcription.txt")
        
        # Initialize audio recorder
        self.recorder = LiveAudioRecorder(
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration,
            include_desktop_audio=include_desktop_audio,
            device_id=device_id
        )
        
        # Transcription state
        self.is_transcribing = False
        self.transcription_thread = None
        self.transcription_queue = queue.Queue()
        
        # Results tracking
        self.chunk_counter = 0
        self.total_transcribed_text = ""
        
        # Determine device enum for backend selection
        self.device_enum = self._get_device_enum()
        
        print(f"Live transcriber initialized:")
        print(f"  Model: {self.model}")
        print(f"  Device: {self.device}")
        print(f"  Language: {self.language or 'auto-detect'}")
        print(f"  Output file: {self.output_file}")
        print(f"  Chunk duration: {chunk_duration}s")
        print(f"  Include desktop audio: {include_desktop_audio}")
    
    def _get_device_enum(self) -> Device:
        """Convert device string to Device enum."""
        if self.device == "insane":
            return Device.INSANE
        elif self.device == "mlx":
            return Device.MLX
        elif self.device == "mps":
            return Device.MLX  # mps is backward compatibility for mlx
        elif self.device == "cuda":
            return Device.CUDA
        else:
            return Device.CPU
    
    def _transcribe_chunk(self, audio_chunk, chunk_id: int):
        """Transcribe a single audio chunk."""
        try:
            # Check if we should stop transcribing
            if not self.is_transcribing:
                return

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)

                # Save audio chunk to temporary WAV file
                chunk_wav = tmpdir_path / f"chunk_{chunk_id}.wav"
                self.recorder.save_chunk_to_wav(audio_chunk, chunk_wav)

                # Check again if we should stop transcribing before starting transcription
                if not self.is_transcribing:
                    return

                # Transcribe using appropriate backend
                if self.device_enum == Device.INSANE:
                    run_insanely_fast_whisper(
                        input_wav=chunk_wav,
                        model=self.model,
                        output_dir=tmpdir_path,
                        task=self.task,
                        language=self.language or "auto",
                        hugging_face_token=self.hugging_face_token,
                        other_args=self.other_args,
                    )
                elif self.device_enum == Device.MLX:
                    run_whisper_mac_mlx(
                        input_wav=chunk_wav,
                        model=self.model,
                        output_dir=tmpdir_path,
                        language=self.language,
                        task=self.task,
                        other_args=self.other_args
                    )
                else:
                    run_whisper(
                        input_wav=chunk_wav,
                        device=str(self.device),
                        model=self.model,
                        output_dir=tmpdir_path,
                        task=self.task,
                        language=self.language or "auto",
                        other_args=self.other_args,
                    )
                
                # Read transcription result
                json_file = tmpdir_path / "out.json"
                txt_file = tmpdir_path / "out.txt"
                
                transcription_text = ""
                if json_file.exists():
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            result = json.load(f)
                            if isinstance(result, dict) and 'text' in result:
                                transcription_text = result['text'].strip()
                            elif isinstance(result, list) and len(result) > 0:
                                # Handle segment-based results
                                transcription_text = " ".join([seg.get('text', '') for seg in result]).strip()
                    except Exception as e:
                        print(f"Error reading JSON result: {e}")
                
                if not transcription_text and txt_file.exists():
                    try:
                        transcription_text = txt_file.read_text(encoding='utf-8').strip()
                    except Exception as e:
                        print(f"Error reading TXT result: {e}")
                
                if transcription_text:
                    timestamp = time.strftime("%H:%M:%S")
                    formatted_text = f"[{timestamp}] {transcription_text}"
                    
                    # Write to output file
                    with open(self.output_file, 'a', encoding='utf-8') as f:
                        f.write(formatted_text + "\n")
                    
                    # Print to console
                    print(f"Chunk {chunk_id}: {transcription_text}")
                    
                    # Update total transcribed text
                    self.total_transcribed_text += transcription_text + " "
                else:
                    print(f"Chunk {chunk_id}: [No speech detected]")

        except KeyboardInterrupt:
            # Re-raise KeyboardInterrupt to allow proper shutdown
            raise
        except RuntimeError as e:
            # Handle subprocess interruption more gracefully
            error_msg = str(e)
            if "return code 130" in error_msg or "KeyboardInterrupt" in error_msg:
                # Return code 130 typically indicates SIGINT (Ctrl+C)
                if self.is_transcribing:
                    print(f"Chunk {chunk_id}: Transcription interrupted")
                return
            else:
                print(f"Error transcribing chunk {chunk_id}: {e}")
        except Exception as e:
            if self.is_transcribing:  # Only print error if we're still supposed to be transcribing
                print(f"Error transcribing chunk {chunk_id}: {e}")
    
    def _transcription_worker(self):
        """Worker thread that processes transcription queue."""
        while self.is_transcribing:
            try:
                # Get next chunk from queue (with timeout)
                chunk_data = self.transcription_queue.get(timeout=1.0)
                if chunk_data is None:  # Shutdown signal
                    break

                audio_chunk, chunk_id = chunk_data
                self._transcribe_chunk(audio_chunk, chunk_id)
                self.transcription_queue.task_done()

            except queue.Empty:
                continue
            except KeyboardInterrupt:
                # Handle KeyboardInterrupt gracefully in worker thread
                print("Transcription worker interrupted")
                break
            except Exception as e:
                if self.is_transcribing:  # Only print error if we're still supposed to be transcribing
                    print(f"Error in transcription worker: {e}")
                # Mark task as done even if there was an error
                try:
                    self.transcription_queue.task_done()
                except ValueError:
                    pass  # task_done() called more times than there were items
    
    def start_live_transcription(self):
        """Start live recording and transcription."""
        if self.is_transcribing:
            print("Live transcription is already running")
            return
        
        print("Starting live transcription...")
        
        # Clear output file
        self.output_file.write_text("", encoding='utf-8')
        
        # Start recording
        self.recorder.start_recording()
        
        # Start transcription worker
        self.is_transcribing = True
        self.transcription_thread = threading.Thread(target=self._transcription_worker)
        self.transcription_thread.daemon = True
        self.transcription_thread.start()
        
        # Main processing loop
        try:
            print("Live transcription started. Press Ctrl+C to stop.")
            while self.is_transcribing:
                # Get audio chunk from recorder
                audio_chunk = self.recorder.get_audio_chunk()
                if audio_chunk is not None:
                    self.chunk_counter += 1
                    # Add to transcription queue
                    self.transcription_queue.put((audio_chunk, self.chunk_counter))
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
        except KeyboardInterrupt:
            print("\nReceived interrupt signal, stopping...")
            self.stop_live_transcription()
    
    def stop_live_transcription(self):
        """Stop live recording and transcription."""
        if not self.is_transcribing:
            return

        print("Stopping live transcription...")

        # Stop recording first
        try:
            self.recorder.stop_recording()
        except Exception as e:
            print(f"Error stopping recorder: {e}")

        # Stop transcription
        self.is_transcribing = False

        # Signal transcription worker to stop
        try:
            self.transcription_queue.put(None)
        except Exception as e:
            print(f"Error signaling transcription worker: {e}")

        # Wait for transcription worker to finish
        if self.transcription_thread:
            try:
                self.transcription_thread.join(timeout=3)  # Reduced timeout
                if self.transcription_thread.is_alive():
                    print("Warning: Transcription worker did not stop cleanly")
            except Exception as e:
                print(f"Error joining transcription thread: {e}")

        # Try to wait for remaining transcriptions to complete, but don't block indefinitely
        try:
            # Clear any remaining items in the queue to avoid blocking
            while not self.transcription_queue.empty():
                try:
                    self.transcription_queue.get_nowait()
                    self.transcription_queue.task_done()
                except queue.Empty:
                    break
                except Exception:
                    break
        except Exception as e:
            print(f"Error clearing transcription queue: {e}")

        print(f"Live transcription stopped. Results saved to: {self.output_file}")
        print(f"Total chunks processed: {self.chunk_counter}")
    
    @staticmethod
    def list_audio_devices():
        """List available audio devices."""
        devices = LiveAudioRecorder.list_audio_devices()
        print("Available audio input devices:")
        for device in devices:
            print(f"  {device}")
        return devices
