"""Unit tests for live microphone voice capture."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from src.stt.live_capture import list_audio_input_devices, record_live_voice


def test_list_audio_input_devices_empty_or_list():
    devices = list_audio_input_devices()
    assert isinstance(devices, list)
    for dev in devices:
        assert "index" in dev
        assert "name" in dev
        assert "channels" in dev
        assert dev["channels"] > 0


def test_record_live_voice_mocked_sounddevice(tmp_path):
    mock_sd = MagicMock()
    mock_sd.query_devices.return_value = [
        {"name": "Mock Microphone", "max_input_channels": 1, "default_samplerate": 16000.0}
    ]
    # Mock rec to fill with dummy audio
    sample_rate = 16000
    duration = 0.5
    dummy_samples = (np.sin(np.linspace(0, 10, int(sample_rate * duration))) * 10000).astype(np.int16)
    mock_sd.rec.return_value = dummy_samples
    mock_sd.wait.return_value = None

    target_wav = str(tmp_path / "test_out.wav")

    with patch.dict("sys.modules", {"sounddevice": mock_sd}):
        out_file = record_live_voice(
            duration=duration,
            sample_rate=sample_rate,
            output_path=target_wav,
            auto_stop_silence=False,
        )

        assert out_file == target_wav
        assert (tmp_path / "test_out.wav").exists()


def test_record_live_voice_no_devices():
    mock_sd = MagicMock()
    mock_sd.query_devices.return_value = []

    with patch.dict("sys.modules", {"sounddevice": mock_sd}):
        out_file = record_live_voice(duration=1.0)
        assert out_file is None
