"""Speech-To-Text interface with Sarvam AI, ElevenLabs, and faster-whisper backends."""

import io
import os
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from src.pipeline.schemas import STTResult
from src.utils.logging import logger


class BaseSTTTranscriber(ABC):
    """Abstract interface for Speech-to-Text transcribers.
    
    Allows seamless swapping of ASR backends (e.g. Sarvam AI, ElevenLabs,
    faster-whisper, AI4Bharat IndicConformer / IndicWhisper).
    """

    @abstractmethod
    def transcribe(
        self,
        audio_input: Union[str, Path, bytes, np.ndarray],
        language: Optional[str] = "hi",
    ) -> STTResult:
        """Transcribes speech audio into text."""
        pass


class SarvamTranscriber(BaseSTTTranscriber):
    """Speech-to-Text using Sarvam AI's saaras model (optimized for Indic languages)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "saaras:v2",
        default_language: str = "hi",
    ):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        self.model = model
        self.default_language = default_language

        if not self.api_key:
            raise ValueError(
                "SARVAM_API_KEY is required. Set it as an environment variable or pass api_key."
            )

        logger.info(f"Initializing SarvamTranscriber (model='{self.model}')")

    def _get_language_code(self, language: str) -> str:
        """Maps ISO language codes to Sarvam language codes."""
        mapping = {
            "hi": "hi-IN", "en": "en-IN", "ta": "ta-IN", "te": "te-IN",
            "kn": "kn-IN", "ml": "ml-IN", "bn": "bn-IN", "gu": "gu-IN",
            "mr": "mr-IN", "pa": "pa-IN", "or": "od-IN", "ur": "ur-IN",
        }
        return mapping.get(language, "hi-IN")

    def transcribe(
        self,
        audio_input: Union[str, Path, bytes, np.ndarray],
        language: Optional[str] = "hi",
    ) -> STTResult:
        from sarvamai import SarvamAI

        lang = language or self.default_language
        client = SarvamAI(api_subscription_key=self.api_key)

        try:
            # Prepare audio file
            if isinstance(audio_input, (str, Path)):
                audio_path = str(audio_input)
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
                with open(audio_path, "rb") as f:
                    audio_file = f.read()
                file_obj = io.BytesIO(audio_file)
                file_obj.name = os.path.basename(audio_path)
            elif isinstance(audio_input, bytes):
                file_obj = io.BytesIO(audio_input)
                file_obj.name = "audio.wav"
            elif isinstance(audio_input, np.ndarray):
                import soundfile as sf
                buf = io.BytesIO()
                arr = audio_input
                if arr.ndim > 1:
                    arr = arr.mean(axis=-1)
                if arr.dtype == np.int16:
                    arr = arr.astype(np.float32) / 32768.0
                elif arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                sf.write(buf, arr, 16000, format="WAV")
                buf.seek(0)
                buf.name = "audio.wav"
                file_obj = buf
            else:
                raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

            response = client.speech_to_text.transcribe(
                file=file_obj,
                model=self.model,
                language_code=self._get_language_code(lang),
            )

            transcript = getattr(response, "transcript", "") or ""
            if not transcript and hasattr(response, "text"):
                transcript = response.text or ""

            return STTResult(
                transcribed_text=transcript.strip(),
                detected_language=lang,
                duration_seconds=0.0,
                avg_logprob=None,
            )

        except Exception as e:
            logger.error(f"[SarvamTranscriber] Transcription failed: {e}")
            raise


class ElevenLabsTranscriber(BaseSTTTranscriber):
    """Speech-to-Text using ElevenLabs Scribe v2 model."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "scribe_v2",
        default_language: str = "hi",
    ):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY", "")
        self.model_id = model_id
        self.default_language = default_language

        if not self.api_key:
            raise ValueError(
                "ELEVENLABS_API_KEY is required. Set it as an environment variable or pass api_key."
            )

        logger.info(f"Initializing ElevenLabsTranscriber (model='{self.model_id}')")

    def _get_language_code(self, language: str) -> str:
        """Maps ISO language codes to ElevenLabs language codes."""
        mapping = {
            "hi": "hin", "en": "eng", "ta": "tam", "te": "tel",
            "kn": "kan", "ml": "mal", "bn": "ben", "gu": "guj",
            "mr": "mar", "pa": "pan",
        }
        return mapping.get(language, "hin")

    def transcribe(
        self,
        audio_input: Union[str, Path, bytes, np.ndarray],
        language: Optional[str] = "hi",
    ) -> STTResult:
        from elevenlabs.client import ElevenLabs

        lang = language or self.default_language
        client = ElevenLabs(api_key=self.api_key)

        try:
            if isinstance(audio_input, (str, Path)):
                audio_path = str(audio_input)
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
                file_obj = open(audio_path, "rb")
            elif isinstance(audio_input, bytes):
                file_obj = io.BytesIO(audio_input)
            elif isinstance(audio_input, np.ndarray):
                import soundfile as sf
                buf = io.BytesIO()
                arr = audio_input
                if arr.ndim > 1:
                    arr = arr.mean(axis=-1)
                if arr.dtype == np.int16:
                    arr = arr.astype(np.float32) / 32768.0
                elif arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                sf.write(buf, arr, 16000, format="WAV")
                buf.seek(0)
                file_obj = buf
            else:
                raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

            transcription = client.speech_to_text.convert(
                file=file_obj,
                model_id=self.model_id,
                language_code=self._get_language_code(lang),
            )

            transcript = getattr(transcription, "text", "") or ""

            return STTResult(
                transcribed_text=transcript.strip(),
                detected_language=lang,
                duration_seconds=0.0,
                avg_logprob=None,
            )

        except Exception as e:
            logger.error(f"[ElevenLabsTranscriber] Transcription failed: {e}")
            raise
        finally:
            if isinstance(audio_input, (str, Path)) and 'file_obj' in locals():
                file_obj.close()


class FasterWhisperTranscriber(BaseSTTTranscriber):
    """Low-latency ASR using faster-whisper (CTranslate2 backend)."""

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: Optional[str] = None,
        compute_type: str = "float16",
        default_language: str = "hi",
        beam_size: int = 1,
        vad_filter: bool = True,
        cpu_threads: int = 4,
    ):
        self.model_size = model_size
        self.default_language = default_language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.cpu_threads = cpu_threads

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"

        # On CPU, float16 is not supported by CTranslate2, so fallback to int8 or float32
        if self.device == "cpu" and compute_type in ("float16", "int8_float16"):
            self.compute_type = "int8"
        else:
            self.compute_type = compute_type

        logger.info(
            f"Initializing FasterWhisperTranscriber (model='{self.model_size}', device='{self.device}', compute_type='{self.compute_type}', beam_size={self.beam_size})"
        )
        
        from faster_whisper import WhisperModel

        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )
        except Exception as e:
            if self.device != "cpu":
                logger.warning(
                    f"[FasterWhisperTranscriber] Failed to initialize on {self.device} ({e}). Falling back to CPU with int8 quantization."
                )
                self.device = "cpu"
                self.compute_type = "int8"
                self.model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=self.cpu_threads,
                )
            else:
                raise

    def transcribe(
        self,
        audio_input: Union[str, Path, bytes, np.ndarray],
        language: Optional[str] = "hi",
    ) -> STTResult:
        lang = language or self.default_language
        audio_target = None

        try:
            # Handle different input types directly in memory to avoid disk I/O
            if isinstance(audio_input, (str, Path)):
                audio_target = str(audio_input)
                if not os.path.exists(audio_target):
                    raise FileNotFoundError(f"Audio file not found: {audio_target}")
            elif isinstance(audio_input, bytes):
                # Use in-memory buffer directly (zero disk I/O)
                audio_target = io.BytesIO(audio_input)
            elif isinstance(audio_input, np.ndarray):
                # Normalize waveform in-memory (faster-whisper expects 1D float32)
                arr = audio_input
                if arr.ndim > 1:
                    arr = arr.mean(axis=-1)
                if arr.dtype == np.int16:
                    arr = arr.astype(np.float32) / 32768.0
                elif arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                audio_target = arr
            else:
                raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

            # Perform transcription with faster-whisper
            try:
                segments, info = self.model.transcribe(
                    audio_target,
                    language=lang,
                    beam_size=self.beam_size,
                    best_of=1,
                    temperature=0.0,
                    vad_filter=self.vad_filter,
                )
                segment_list = list(segments)
            except Exception as exc:
                if self.device != "cpu":
                    logger.warning(
                        f"[FasterWhisperTranscriber] Runtime error on {self.device} ({exc}). Switching to CPU int8..."
                    )
                    from faster_whisper import WhisperModel
                    self.device = "cpu"
                    self.compute_type = "int8"
                    self.model = WhisperModel(
                        self.model_size,
                        device="cpu",
                        compute_type="int8",
                        cpu_threads=self.cpu_threads,
                    )
                    segments, info = self.model.transcribe(
                        audio_target,
                        language=lang,
                        beam_size=self.beam_size,
                        best_of=1,
                        temperature=0.0,
                        vad_filter=self.vad_filter,
                    )
                    segment_list = list(segments)
                else:
                    raise

            segment_texts = []
            logprobs = []
            for segment in segment_list:
                segment_texts.append(segment.text)
                logprobs.append(segment.avg_logprob)

            transcribed_text = " ".join(segment_texts).strip()
            avg_logprob = float(np.mean(logprobs)) if logprobs else 0.0

            return STTResult(
                transcribed_text=transcribed_text,
                detected_language=info.language if info.language else lang,
                duration_seconds=float(info.duration) if info.duration else 0.0,
                avg_logprob=avg_logprob,
            )

        except Exception:
            raise


class MockTranscriber(BaseSTTTranscriber):
    """Deterministic mock transcriber for testing and synthetic validation."""

    def __init__(self, fixed_text: str = "भारत की राजधानी क्या है?"):
        self.fixed_text = fixed_text

    def transcribe(
        self,
        audio_input: Union[str, Path, bytes, np.ndarray],
        language: Optional[str] = "hi",
    ) -> STTResult:
        return STTResult(
            transcribed_text=self.fixed_text,
            detected_language=language or "hi",
            duration_seconds=1.5,
            avg_logprob=-0.1,
        )


def get_transcriber(
    provider: str = "sarvam",
    model_size: str = "large-v3-turbo",
    device: Optional[str] = None,
    compute_type: str = "float16",
    beam_size: int = 1,
    vad_filter: bool = True,
    cpu_threads: int = 4,
    api_key: Optional[str] = None,
) -> BaseSTTTranscriber:
    """Factory function for STT transcribers.
    
    Supported providers: 'sarvam', 'elevenlabs', 'faster_whisper', 'mock'
    """
    p = provider.lower()

    if p == "sarvam":
        try:
            return SarvamTranscriber(api_key=api_key)
        except (ValueError, ImportError) as e:
            logger.warning(f"Could not initialize Sarvam STT ({e}). Trying fallbacks...")
            # Try ElevenLabs as fallback
            try:
                return ElevenLabsTranscriber()
            except (ValueError, ImportError):
                pass
            # Fall back to faster-whisper
            try:
                return FasterWhisperTranscriber(
                    model_size=model_size, device=device, compute_type=compute_type,
                    beam_size=beam_size, vad_filter=vad_filter, cpu_threads=cpu_threads,
                )
            except Exception:
                pass
            logger.warning("All STT providers failed. Using MockTranscriber.")
            return MockTranscriber()

    elif p == "elevenlabs":
        try:
            return ElevenLabsTranscriber(api_key=api_key)
        except (ValueError, ImportError) as e:
            logger.warning(f"Could not initialize ElevenLabs STT ({e}). Falling back...")
            return MockTranscriber()

    elif p in ("faster_whisper", "whisper"):
        return FasterWhisperTranscriber(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            vad_filter=vad_filter,
            cpu_threads=cpu_threads,
        )

    elif p == "mock":
        return MockTranscriber()

    else:
        raise ValueError(f"Unknown STT provider: {provider}. Supported: sarvam, elevenlabs, faster_whisper, mock")
