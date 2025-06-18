"""
Live audio recording functionality for microphone and desktop audio capture.
"""

import platform
import queue
import threading
import time
from pathlib import Path
from typing import Optional, Callable, List
import tempfile
import subprocess
import sys

import numpy as np
import sounddevice as sd


class AudioDevice:
    """Represents an audio input device."""
    
    def __init__(self, device_id: int, name: str, channels: int, sample_rate: float):
        self.device_id = device_id
        self.name = name
        self.channels = channels
        self.sample_rate = sample_rate
    
    def __str__(self):
        return f"Device {self.device_id}: {self.name} ({self.channels} channels, {self.sample_rate} Hz)"


class LiveAudioRecorder:
    """Records live audio from microphone and optionally desktop audio."""
    
    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 1,
        chunk_duration: float = 5.0,
        overlap_duration: float = 1.0,
        include_desktop_audio: bool = False,
        device_id: Optional[int] = None
    ):
        """
        Initialize the live audio recorder.
        
        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1 for mono, 2 for stereo)
            chunk_duration: Duration of each audio chunk in seconds
            overlap_duration: Overlap between chunks in seconds
            include_desktop_audio: Whether to include desktop/system audio
            device_id: Specific device ID to use (None for default)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.include_desktop_audio = include_desktop_audio
        self.device_id = device_id
        
        self.chunk_size = int(sample_rate * chunk_duration)
        self.overlap_size = int(sample_rate * overlap_duration)
        
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = None
        self.desktop_recording_thread = None
        
        # Audio buffers
        self.mic_buffer = np.array([], dtype=np.float32)
        self.desktop_buffer = np.array([], dtype=np.float32)
        self.combined_buffer = np.array([], dtype=np.float32)
        
        # Lock for thread safety
        self.buffer_lock = threading.Lock()
        
        # Validate device
        self._validate_device()
    
    def _validate_device(self):
        """Validate the selected audio device."""
        try:
            devices = sd.query_devices()
            if self.device_id is not None:
                if self.device_id >= len(devices):
                    raise ValueError(f"Device ID {self.device_id} not found")
                device_info = devices[self.device_id]
                if device_info['max_input_channels'] < self.channels:
                    raise ValueError(f"Device {self.device_id} doesn't support {self.channels} input channels")
        except Exception as e:
            print(f"Warning: Could not validate audio device: {e}")
    
    @staticmethod
    def list_audio_devices() -> List[AudioDevice]:
        """List all available audio input devices."""
        devices = []
        try:
            device_list = sd.query_devices()
            for i, device_info in enumerate(device_list):
                if device_info['max_input_channels'] > 0:  # Only input devices
                    devices.append(AudioDevice(
                        device_id=i,
                        name=device_info['name'],
                        channels=device_info['max_input_channels'],
                        sample_rate=device_info['default_samplerate']
                    ))
        except Exception as e:
            print(f"Error listing audio devices: {e}")
        return devices
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback function for audio input stream."""
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert to mono if needed
        if indata.shape[1] > 1 and self.channels == 1:
            audio_data = np.mean(indata, axis=1, dtype=np.float32)
        else:
            audio_data = indata[:, 0].astype(np.float32)
        
        with self.buffer_lock:
            self.mic_buffer = np.append(self.mic_buffer, audio_data)
    
    def _record_microphone(self):
        """Record audio from microphone in a separate thread."""
        try:
            with sd.InputStream(
                device=self.device_id,
                channels=self.channels,
                samplerate=self.sample_rate,
                callback=self._audio_callback,
                blocksize=1024
            ):
                while self.is_recording:
                    time.sleep(0.1)
        except Exception as e:
            print(f"Error recording microphone: {e}")
    
    def _record_desktop_audio(self):
        """Record desktop audio using platform-specific methods."""
        if not self.include_desktop_audio:
            return
            
        system = platform.system()
        try:
            if system == "Windows":
                self._record_desktop_windows()
            elif system == "Darwin":  # macOS
                self._record_desktop_macos()
            elif system == "Linux":
                self._record_desktop_linux()
            else:
                print(f"Desktop audio recording not supported on {system}")
        except Exception as e:
            print(f"Error recording desktop audio: {e}")
    
    def _record_desktop_windows(self):
        """Record desktop audio on Windows using ffmpeg."""
        # Use ffmpeg to capture desktop audio via DirectShow
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        
        cmd = [
            "static_ffmpeg",
            "-f", "dshow",
            "-i", "audio=Stereo Mix",  # Default Windows loopback device
            "-acodec", "pcm_s16le",
            "-ar", str(self.sample_rate),
            "-ac", str(self.channels),
            "-t", "1",  # Record 1 second chunks
            "-y", temp_path
        ]
        
        while self.is_recording:
            try:
                subprocess.run(cmd, capture_output=True, timeout=2)
                # Read the recorded audio and add to buffer
                # This is a simplified implementation
                time.sleep(1)
            except Exception as e:
                print(f"Windows desktop audio error: {e}")
                time.sleep(1)
    
    def _record_desktop_macos(self):
        """Record desktop audio on macOS using ffmpeg."""
        # Use ffmpeg to capture desktop audio via AVFoundation
        while self.is_recording:
            try:
                # This would need proper implementation with AVFoundation
                # For now, just sleep to prevent busy waiting
                time.sleep(1)
            except Exception as e:
                print(f"macOS desktop audio error: {e}")
                time.sleep(1)
    
    def _record_desktop_linux(self):
        """Record desktop audio on Linux using PulseAudio."""
        # Use ffmpeg to capture desktop audio via PulseAudio
        while self.is_recording:
            try:
                # This would need proper implementation with PulseAudio
                # For now, just sleep to prevent busy waiting
                time.sleep(1)
            except Exception as e:
                print(f"Linux desktop audio error: {e}")
                time.sleep(1)
    
    def start_recording(self):
        """Start live audio recording."""
        if self.is_recording:
            print("Recording is already in progress")
            return
        
        print("Starting live audio recording...")
        self.is_recording = True
        
        # Start microphone recording thread
        self.recording_thread = threading.Thread(target=self._record_microphone)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        # Start desktop audio recording thread if enabled
        if self.include_desktop_audio:
            self.desktop_recording_thread = threading.Thread(target=self._record_desktop_audio)
            self.desktop_recording_thread.daemon = True
            self.desktop_recording_thread.start()
        
        print("Live recording started successfully")
    
    def stop_recording(self):
        """Stop live audio recording."""
        if not self.is_recording:
            return
        
        print("Stopping live audio recording...")
        self.is_recording = False
        
        # Wait for threads to finish
        if self.recording_thread:
            self.recording_thread.join(timeout=2)
        if self.desktop_recording_thread:
            self.desktop_recording_thread.join(timeout=2)
        
        print("Live recording stopped")
    
    def get_audio_chunk(self) -> Optional[np.ndarray]:
        """
        Get the next audio chunk for transcription.
        
        Returns:
            Audio chunk as numpy array, or None if not enough data
        """
        with self.buffer_lock:
            if len(self.mic_buffer) >= self.chunk_size:
                # Extract chunk with overlap
                chunk = self.mic_buffer[:self.chunk_size].copy()
                # Keep overlap for next chunk
                self.mic_buffer = self.mic_buffer[self.chunk_size - self.overlap_size:]
                return chunk
        return None
    
    def save_chunk_to_wav(self, audio_chunk: np.ndarray, output_path: Path):
        """Save an audio chunk to a WAV file."""
        import wave
        
        # Convert float32 to int16
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        
        with wave.open(str(output_path), 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
