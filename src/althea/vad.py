"""Voice Activity Detection module.

After the wake word fires, this module captures audio from the microphone
until approximately 1.5 seconds of silence has been detected (Silero VAD).
The resulting Utterance audio is then passed to a callback for Transcription.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import sounddevice as sd  # only used for type hints

logger = logging.getLogger(__name__)

# Microphone sample rate (must match Silero VAD's expected 16 kHz).
_SAMPLE_RATE: int = 16_000

# Audio chunk size fed to Silero VAD (512 samples = 32 ms at 16 kHz).
# Silero VAD v5 supports 512 or 1024 samples at 16 kHz.
_CHUNK_SAMPLES: int = 512

# Silence duration (in seconds) before the Utterance is considered complete.
_SILENCE_THRESHOLD_SECONDS: float = 1.5

# Silero VAD speech probability threshold (0–1).  Chunks above this are
# considered "speech"; chunks below are "silence".
_SPEECH_PROBABILITY_THRESHOLD: float = 0.5


def _seconds_to_chunks(seconds: float) -> int:
    """Convert a duration in seconds to a number of VAD chunks."""
    return int(seconds * _SAMPLE_RATE / _CHUNK_SAMPLES)


def _int16_to_float32(chunk: np.ndarray) -> np.ndarray:
    """Normalise an int16 PCM array to float32 in the range [-1, 1]."""
    return chunk.astype(np.float32) / 32768.0


class VoiceActivityDetector:
    """Captures a single Utterance from the microphone using Silero VAD.

    After :meth:`start_capture` is called (typically from the wake word
    callback), the detector streams audio, accumulates speech frames, and
    calls *on_utterance* once ~1.5 s of silence is detected.

    VAD inference runs on a dedicated inference thread (not the PortAudio
    callback thread) to avoid real-time audio buffer underflows.

    Usage::

        def handle_utterance(audio: np.ndarray) -> None:
            print("Utterance captured:", audio.shape)

        vad = VoiceActivityDetector(on_utterance=handle_utterance)
        vad.start_capture()   # call this after wake word fires
    """

    def __init__(
        self,
        on_utterance: Callable[[np.ndarray], None],
        *,
        silence_threshold_seconds: float = _SILENCE_THRESHOLD_SECONDS,
        speech_probability_threshold: float = _SPEECH_PROBABILITY_THRESHOLD,
    ) -> None:
        """Initialise the VoiceActivityDetector.

        Args:
            on_utterance: Called with the captured float32 audio array once
                end-of-speech is detected.
            silence_threshold_seconds: Seconds of silence that mark the end
                of an Utterance.
            speech_probability_threshold: Silero VAD probability cutoff above
                which a chunk is classified as speech.
        """
        self._on_utterance = on_utterance
        self._silence_threshold_seconds = silence_threshold_seconds
        self._speech_probability_threshold = speech_probability_threshold
        self._silence_chunks = _seconds_to_chunks(silence_threshold_seconds)

        self._vad_model: object | None = None  # silero_vad model, lazy-loaded
        self._torch: object | None = None  # torch module, cached after first load
        self._capture_thread: threading.Thread | None = None
        self._stop_capture_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_capture(self) -> None:
        """Start capturing a single Utterance in a background thread.

        If a capture is already in progress it is silently ignored — the
        existing capture takes precedence.
        """
        if self._capture_thread is not None and self._capture_thread.is_alive():
            logger.debug("VAD capture already in progress; ignoring start_capture().")
            return

        self._stop_capture_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_utterance,
            name="vad-capture",
            daemon=True,
        )
        self._capture_thread.start()
        logger.info("VAD capture started.")

    def stop(self) -> None:
        """Abort any in-progress capture cleanly."""
        self._stop_capture_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=5)
            self._capture_thread = None
        logger.info("VAD capture stopped.")

    @property
    def is_capturing(self) -> bool:
        """``True`` while a capture thread is running."""
        return self._capture_thread is not None and self._capture_thread.is_alive()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_vad_model(self) -> object:
        """Lazy-load the Silero VAD model via torch.hub."""
        import torch  # type: ignore[import-untyped]

        self._torch = torch
        logger.info("Loading Silero VAD model …")
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        model.eval()
        logger.info("Silero VAD model loaded.")
        return model

    def _predict_speech(self, chunk: np.ndarray) -> float:
        """Return Silero VAD speech probability for *chunk* (int16 input).

        Args:
            chunk: Flat int16 audio array of exactly ``_CHUNK_SAMPLES`` samples.

        Returns:
            Speech probability in [0, 1].
        """
        torch = self._torch
        audio_float = _int16_to_float32(chunk)
        tensor = torch.from_numpy(audio_float).unsqueeze(0)
        prob: float = self._vad_model(tensor, _SAMPLE_RATE).item()
        return prob

    def _capture_utterance(self) -> None:
        """Main capture loop — runs in its own thread.

        VAD inference runs on this thread (not the PortAudio callback thread)
        via a chunk queue, avoiding real-time buffer underflows.
        """
        try:
            import sounddevice as sd  # type: ignore[import-untyped]
        except ImportError as exc:
            logger.error("sounddevice is not installed. Cannot capture audio: %s", exc)
            return

        if self._vad_model is None:
            try:
                self._vad_model = self._load_vad_model()
            except Exception:
                logger.exception("Failed to load Silero VAD model.")
                return

        # Reset Silero VAD's internal RNN state so previous Utterances don't
        # bleed into this capture session.
        if hasattr(self._vad_model, "reset_states"):
            self._vad_model.reset_states()

        chunk_queue: queue.Queue[np.ndarray | None] = queue.Queue()

        def _callback(
            indata: np.ndarray,
            frames: int,  # noqa: ARG001
            time_info: object,  # noqa: ARG001
            status: object,
        ) -> None:
            """Minimal PortAudio callback — just enqueues raw chunks."""
            if status:
                logger.debug("VAD audio status: %s", status)
            if self._stop_capture_event.is_set():
                chunk_queue.put(None)  # sentinel
                raise sd.CallbackStop
            chunk_queue.put(indata[:, 0].copy())

        logger.info("VAD listening for the user's Utterance …")
        try:
            with sd.InputStream(
                samplerate=_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=_CHUNK_SAMPLES,
                callback=_callback,
            ):
                audio_buffer = self._run_vad_loop(chunk_queue)
        except sd.CallbackStop:
            pass  # raised intentionally by _callback to end the stream
        except sd.PortAudioError as exc:
            logger.error("Microphone unavailable during VAD capture: %s", exc)
            return
        except Exception:
            logger.exception("Unexpected error during VAD capture.")
            return

        if not audio_buffer:
            logger.warning("VAD capture ended with no Utterance detected.")
            return

        utterance = np.concatenate([_int16_to_float32(c) for c in audio_buffer])
        logger.info(
            "Utterance captured: %.2f s",
            len(utterance) / _SAMPLE_RATE,
        )
        self._on_utterance(utterance)

    def _run_vad_loop(
        self, chunk_queue: "queue.Queue[np.ndarray | None]"
    ) -> list[np.ndarray]:
        """Consume *chunk_queue* and return buffered speech chunks (int16).

        Runs VAD inference on *this* thread (off the PortAudio callback).
        Accumulates chunks once speech starts; stops after ``_silence_chunks``
        consecutive silent frames following detected speech.
        """
        audio_buffer: list[np.ndarray] = []
        silent_chunks: int = 0
        speech_started: bool = False

        while not self._stop_capture_event.is_set():
            try:
                chunk = chunk_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if chunk is None:  # stop sentinel
                break

            prob = self._predict_speech(chunk)
            is_speech = prob >= self._speech_probability_threshold

            if is_speech:
                speech_started = True
                silent_chunks = 0
                audio_buffer.append(chunk)
            elif speech_started:
                # Buffer trailing silence so we don't clip the end of words.
                audio_buffer.append(chunk)
                silent_chunks += 1
                if silent_chunks >= self._silence_chunks:
                    break
            # Leading silence (before first speech) is discarded.

        return audio_buffer
