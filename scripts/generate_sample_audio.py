"""Generates a sample test voice audio file."""

import numpy as np
import soundfile as sf
from pathlib import Path

def main():
    Path("data").mkdir(exist_ok=True)
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Synthetic speech-like formant harmonics
    audio = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 880 * t)
    audio = (audio * 32767).astype(np.int16)
    out_path = "data/sample_voice_query.wav"
    sf.write(out_path, audio, sample_rate)
    print(f"Generated sample voice file at {out_path}")

if __name__ == "__main__":
    main()
