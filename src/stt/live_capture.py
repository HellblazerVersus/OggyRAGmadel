"""Live voice capture utility for recording microphone audio input."""

import os
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from src.utils.logging import logger, console


def list_audio_input_devices() -> List[Dict[str, Union[int, str, float]]]:
    """Queries and returns all available audio input devices (microphones)."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devs = []
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                input_devs.append({
                    "index": idx,
                    "name": dev.get("name", f"Device #{idx}"),
                    "channels": dev.get("max_input_channels", 1),
                    "default_samplerate": dev.get("default_samplerate", 16000.0),
                })
        return input_devs
    except Exception as exc:
        logger.warning(f"Could not query audio devices: {exc}")
        return []


def record_live_voice(
    duration: float = 5.0,
    sample_rate: int = 16000,
    device: Optional[Union[int, str]] = None,
    auto_stop_silence: bool = False,
    silence_threshold: float = 0.015,
    silence_duration: float = 1.2,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """Records audio from system microphone.
    
    Args:
        duration: Maximum recording duration in seconds.
        sample_rate: Audio sampling rate (default 16000 Hz for Whisper / Sarvam).
        device: Device index or device name substring.
        auto_stop_silence: If True, automatically stops recording after silence is detected.
        silence_threshold: RMS amplitude threshold below which audio is considered silence.
        silence_duration: Duration of consecutive silence in seconds before auto-stopping.
        output_path: Optional specific path to save the WAV file. If None, uses a NamedTemporaryFile.
        
    Returns:
        Path to the saved WAV file, or None if recording failed.
    """
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        console.print("[bold red]Error:[/bold red] 'sounddevice' or 'soundfile' is not installed.")
        console.print("Please install them or run: [cyan]uv sync[/cyan]")
        return None

    try:
        # Check if any input device is available
        input_devices = list_audio_input_devices()
        if not input_devices:
            console.print("[bold yellow]⚠️ No active microphone input devices found on system.[/bold yellow]")
            console.print("[dim]Use --file <path> to test with an audio file, or --text for text queries.[/dim]")
            return None

        # Resolve device index
        dev_idx = None
        if device is not None:
            if isinstance(device, int):
                dev_idx = device
            elif isinstance(device, str) and device.isdigit():
                dev_idx = int(device)
            else:
                for d in input_devices:
                    if str(device).lower() in str(d["name"]).lower():
                        dev_idx = d["index"]
                        break

        # Output target file
        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            target_wav = tmp.name
            tmp.close()
        else:
            target_wav = output_path
            os.makedirs(os.path.dirname(os.path.abspath(target_wav)), exist_ok=True)

        if not auto_stop_silence:
            # Fixed duration recording with visual status
            with console.status(f"[bold red]🎙️ LIVE RECORDING ({duration:.1f}s)... Speak your command now![/bold red]", spinner="dots"):
                recording = sd.rec(
                    int(duration * sample_rate),
                    samplerate=sample_rate,
                    channels=1,
                    dtype="int16",
                    device=dev_idx,
                )
                sd.wait()

            sf.write(target_wav, recording, sample_rate)
            console.print(f"[bold green]✓ Recorded {duration:.1f}s of audio.[/bold green]")
            return target_wav

        else:
            # Silence-aware dynamic recording
            console.print(f"[bold red]🎙️ LIVE RECORDING (Speaking... max {duration:.1f}s)... Speak your command![/bold red]")
            chunk_duration = 0.1  # 100ms chunks
            chunk_samples = int(sample_rate * chunk_duration)
            recorded_chunks = []
            silence_chunks = 0
            max_silence_chunks = int(silence_duration / chunk_duration)
            max_total_chunks = int(duration / chunk_duration)
            has_spoken = False

            with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", device=dev_idx) as stream:
                for chunk_idx in range(max_total_chunks):
                    chunk, overflowed = stream.read(chunk_samples)
                    recorded_chunks.append(chunk)

                    # Compute RMS energy of chunk
                    rms = float(np.sqrt(np.mean(chunk**2))) if len(chunk) > 0 else 0.0

                    if rms > silence_threshold:
                        has_spoken = True
                        silence_chunks = 0
                    elif has_spoken:
                        silence_chunks += 1
                        if silence_chunks >= max_silence_chunks:
                            # User finished speaking and paused
                            break

            if not recorded_chunks:
                return None

            audio_data = np.concatenate(recorded_chunks, axis=0)
            # Convert float32 [-1, 1] to int16
            audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
            sf.write(target_wav, audio_int16, sample_rate)
            actual_duration = len(audio_int16) / sample_rate
            console.print(f"[bold green]✓ Voice captured ({actual_duration:.1f}s). Processing...[/bold green]")
            return target_wav

    except Exception as exc:
        console.print(f"[bold red]Live audio capture error:[/bold red] {exc}")
        logger.error(f"Live voice capture exception: {exc}")
        return None
