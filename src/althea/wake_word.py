"""Wake word detection module.

Streams audio continuously from the default microphone and calls a callback
when the placeholder wake word ("Hey Jarvis") is detected. The custom
"Althea" model will replace this once it is trained (see ADR-0001).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import sounddevice as sd  # only used for type hints

logger = logging.getLogger(__name__)

# Microphone sample rate expected by openWakeWord (16 kHz mono).
_SAMPLE_RATE: int = 16_000

# Number of audio samples per callback chunk (~80 ms, the model's native frame).
_CHUNK_SAMPLES: int = 1_280

# Minimum model score to count as a detection (0–1 scale).
_DETECTION_THRESHOLD: float = 0.5

# Name of the placeholder pre-trained model bundled with openWakeWord.
PLACEHOLDER_WAKE_WORD: str = "hey_jarvis"


class WakeWordDetector:
    """Continuously streams audio and fires a callback on wake word detection.

    Usage::

        def on_wake():
            print("Wake word detected!")

        detector = WakeWordDetector(on_wake_word=on_wake)
        detector.start()
        # … later …
        detector.stop()
    """

    def __init__(
        self,
        on_wake_word: Callable[[], None],
        *,
        wake_word: str = PLACEHOLDER_WAKE_WORD,
        threshold: float = _DETECTION_THRESHOLD,
    ) -> None:
        """Initialise the detector.

        Args:
            on_wake_word: Called (in the audio thread) every time the wake
                word is detected above *threshold*.
            wake_word: The openWakeWord model name to load.  Defaults to the
                bundled ``"hey_jarvis"`` placeholder.
            threshold: Detection score threshold in the range ``[0, 1]``.
        """
        self._on_wake_word = on_wake_word
        self._wake_word = wake_word
        self._threshold = threshold

        self._model: object | None = None  # openwakeword.Model, lazy-loaded
        self._stream: "sd.InputStream | None" = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the audio capture thread.

        Safe to call once per ``WakeWordDetector`` instance.  Raises
        ``RuntimeError`` if already running.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("WakeWordDetector is already running.")

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="wake-word-detector",
            daemon=True,
        )
        self._thread.start()
        logger.info("Wake word detector started (model: %s)", self._wake_word)

    def stop(self) -> None:
        """Stop audio capture and wait for the thread to exit cleanly."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Wake word detector stopped.")

    @property
    def is_running(self) -> bool:
        """``True`` while the background thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> object:
        """Lazy-initialise the openWakeWord model (deferred for testability).

        Downloads the model file on first run if it is not already cached
        locally.  Replace ``PLACEHOLDER_WAKE_WORD`` with the path to the
        trained ``althea.onnx`` model once it has been trained.
        """
        import os

        import openwakeword  # type: ignore[import-untyped]
        from openwakeword.model import Model  # type: ignore[import-untyped]
        from openwakeword.utils import download_models  # type: ignore[import-untyped]

        # Download the placeholder model if it hasn't been fetched yet.
        # TODO: swap wake_word for the path to the trained althea.onnx once
        #       the custom model is ready — no other code changes needed.
        onnx_paths = openwakeword.get_pretrained_model_paths("onnx")
        model_cached = any(
            self._wake_word.replace(" ", "_") in p and os.path.exists(p)
            for p in onnx_paths
        )
        if not model_cached:
            logger.info(
                "Wake word model not found locally — downloading '%s' …",
                self._wake_word,
            )
            download_models([f"{self._wake_word}_v0.1"])

        logger.info("Loading wake word model '%s' …", self._wake_word)
        model = Model(
            wakeword_models=[self._wake_word],
            inference_framework="onnx",
        )
        logger.info("Wake word model loaded.")
        return model



    def _run(self) -> None:
        """Main loop: open a microphone stream and feed chunks to the model."""
        try:
            import sounddevice as sd  # type: ignore[import-untyped]
        except ImportError as exc:
            logger.error(
                "sounddevice is not installed. Cannot capture audio: %s", exc
            )
            return

        try:
            self._model = self._load_model()
        except Exception:
            logger.exception("Failed to load wake word model.")
            return

        logger.info(
            "Listening for wake word '%s' on the default microphone …",
            self._wake_word,
        )

        try:
            with sd.InputStream(
                samplerate=_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=_CHUNK_SAMPLES,
                callback=self._audio_callback,
            ):
                # Block until stop() is called.
                self._stop_event.wait()
        except sd.PortAudioError as exc:
            logger.error("Microphone unavailable: %s", exc)
        except Exception:
            logger.exception("Unexpected error in audio capture loop.")

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,  # noqa: ARG002
        time_info: object,  # noqa: ARG002
        status: object,
    ) -> None:
        """sounddevice callback — called on every audio chunk."""
        if status:
            logger.debug("Audio stream status: %s", status)

        if self._model is None or self._stop_event.is_set():
            return

        # openWakeWord expects a flat int16 array.
        audio_chunk = indata[:, 0]
        prediction: dict[str, float] = self._model.predict(audio_chunk)

        score = prediction.get(self._wake_word, 0.0)
        if score >= self._threshold:
            logger.info(
                "Wake word detected! (model=%s, score=%.3f)",
                self._wake_word,
                score,
            )
            self._on_wake_word()
