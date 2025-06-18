"""
Test live recording functionality.
"""

import unittest
import tempfile
import time
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mock dependencies before importing our modules
mock_sd = MagicMock()
mock_np = MagicMock()
mock_np.array = MagicMock(return_value=MagicMock())
mock_np.append = MagicMock(return_value=MagicMock())
mock_np.mean = MagicMock(return_value=MagicMock())
mock_np.float32 = float
mock_np.int16 = int

sys.modules['sounddevice'] = mock_sd
sys.modules['numpy'] = mock_np
sys.modules['static_ffmpeg'] = MagicMock()
sys.modules['yt_dlp'] = MagicMock()
sys.modules['appdirs'] = MagicMock()
sys.modules['disklru'] = MagicMock()
sys.modules['filelock'] = MagicMock()
sys.modules['webvtt'] = MagicMock()

from transcribe_anything.live_recorder import LiveAudioRecorder, AudioDevice
from transcribe_anything.live_transcriber import LiveTranscriber


class TestLiveRecording(unittest.TestCase):
    """Test live recording functionality."""
    
    def test_audio_device_creation(self):
        """Test AudioDevice creation."""
        device = AudioDevice(
            device_id=0,
            name="Test Microphone",
            channels=2,
            sample_rate=44100.0
        )
        
        self.assertEqual(device.device_id, 0)
        self.assertEqual(device.name, "Test Microphone")
        self.assertEqual(device.channels, 2)
        self.assertEqual(device.sample_rate, 44100.0)
        
        # Test string representation
        device_str = str(device)
        self.assertIn("Test Microphone", device_str)
        self.assertIn("44100", device_str)
    
    def test_live_audio_recorder_initialization(self):
        """Test LiveAudioRecorder initialization."""
        recorder = LiveAudioRecorder(
            sample_rate=22050,
            channels=1,
            chunk_duration=3.0,
            overlap_duration=0.5
        )
        
        self.assertEqual(recorder.sample_rate, 22050)
        self.assertEqual(recorder.channels, 1)
        self.assertEqual(recorder.chunk_duration, 3.0)
        self.assertEqual(recorder.overlap_duration, 0.5)
        self.assertEqual(recorder.chunk_size, 22050 * 3)  # 3 seconds
        self.assertEqual(recorder.overlap_size, 22050 * 0.5)  # 0.5 seconds
        self.assertFalse(recorder.is_recording)
    
    @patch('transcribe_anything.live_recorder.sd.query_devices')
    def test_list_audio_devices(self, mock_query_devices):
        """Test listing audio devices."""
        # Mock device list
        mock_devices = [
            {
                'name': 'Microphone 1',
                'max_input_channels': 2,
                'default_samplerate': 44100.0
            },
            {
                'name': 'Microphone 2', 
                'max_input_channels': 1,
                'default_samplerate': 48000.0
            },
            {
                'name': 'Speaker',
                'max_input_channels': 0,  # Output only
                'default_samplerate': 44100.0
            }
        ]
        mock_query_devices.return_value = mock_devices
        
        devices = LiveAudioRecorder.list_audio_devices()
        
        # Should only return input devices (max_input_channels > 0)
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0].name, 'Microphone 1')
        self.assertEqual(devices[1].name, 'Microphone 2')
    
    def test_live_transcriber_initialization(self):
        """Test LiveTranscriber initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.txt"
            
            transcriber = LiveTranscriber(
                model="tiny",
                device="cpu",
                language="en",
                task="transcribe",
                chunk_duration=2.0,
                overlap_duration=0.3,
                output_file=str(output_file)
            )
            
            self.assertEqual(transcriber.model, "tiny")
            self.assertEqual(transcriber.device, "cpu")
            self.assertEqual(transcriber.language, "en")
            self.assertEqual(transcriber.task, "transcribe")
            self.assertEqual(transcriber.output_file, output_file)
            self.assertFalse(transcriber.is_transcribing)
    
    @patch('transcribe_anything.live_recorder.sd.InputStream')
    def test_recorder_start_stop(self, mock_input_stream):
        """Test starting and stopping recording."""
        # Mock the input stream context manager
        mock_stream = MagicMock()
        mock_input_stream.return_value.__enter__.return_value = mock_stream
        mock_input_stream.return_value.__exit__.return_value = None
        
        recorder = LiveAudioRecorder(chunk_duration=1.0)
        
        # Test starting recording
        recorder.start_recording()
        self.assertTrue(recorder.is_recording)
        self.assertIsNotNone(recorder.recording_thread)
        
        # Give it a moment to start
        time.sleep(0.1)
        
        # Test stopping recording
        recorder.stop_recording()
        self.assertFalse(recorder.is_recording)
    
    def test_chunk_saving(self):
        """Test saving audio chunks to WAV files."""
        import numpy as np
        
        recorder = LiveAudioRecorder()
        
        # Create a test audio chunk (1 second of sine wave)
        sample_rate = 44100
        duration = 1.0
        frequency = 440.0  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_chunk = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_chunk.wav"
            
            # Save the chunk
            recorder.save_chunk_to_wav(audio_chunk, output_path)
            
            # Verify file was created
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
    
    def test_device_enum_conversion(self):
        """Test device string to enum conversion."""
        transcriber = LiveTranscriber(device="insane")
        from transcribe_anything.api import Device
        
        self.assertEqual(transcriber._get_device_enum(), Device.INSANE)
        
        transcriber.device = "mlx"
        self.assertEqual(transcriber._get_device_enum(), Device.MLX)
        
        transcriber.device = "cuda"
        self.assertEqual(transcriber._get_device_enum(), Device.CUDA)
        
        transcriber.device = "cpu"
        self.assertEqual(transcriber._get_device_enum(), Device.CPU)


if __name__ == "__main__":
    unittest.main()
