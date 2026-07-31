"""Transcription module — converts spoken Utterance audio to text.

Uses Faster-Whisper with the OpenVINO backend to leverage the Intel Arc A530M
GPU.  Falls back to CPU if OpenVINO initialisation fails (per ADR-0001).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from faster_whisper import WhisperModel  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# Default model size — small enough for low latency, accurate enough for most
# commands.  Can be overridden at construction time.
_DEFAULT_MODEL_SIZE: str = "base"

# Faster-Whisper compute type for OpenVINO / CPU.
_COMPUTE_TYPE: str = "int8"

# Sample rate Faster-Whisper expects (16 kHz).
_SAMPLE_RATE: int = 16_000


class Transcriber:
    """Transcribes spoken audio (numpy float32 array) to text.

    Usage::

        transcriber = Transcriber()
        text = transcriber.transcribe(audio_array)
        print(text)

    The underlying Faster-Whisper model is lazy-loaded on the first call to
    :meth:`transcribe` so that imports of this module don't pay the model-load
    cost.
    """

    def __init__(
        self,
        model_size: str = _DEFAULT_MODEL_SIZE,
    ) -> None:
        """Initialise the Transcriber.

        Args:
            model_size: Faster-Whisper model size (e.g. ``"base"``,
                ``"small"``, ``"medium"``).
        """
        self._model_size = model_size
        self._model: "WhisperModel | None" = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe *audio* and return the text.

        Args:
            audio: 1-D float32 array of audio samples at 16 kHz.

        Returns:
            The transcribed text, stripped of leading/trailing whitespace.
            Returns an empty string if transcription yields no segments.
        """
        if self._model is None:
            self._model = self._load_model()

        segments, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=5,
        )
        text = " ".join(seg.text for seg in segments).strip()
        logger.info("Transcription: %r", text)
        return text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> "WhisperModel":
        """Load (and optionally download) the Faster-Whisper model.

        Tries the OpenVINO backend first to leverage the Intel Arc GPU.
        Falls back to CPU if OpenVINO is unavailable or initialisation fails.
        """
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        # Attempt OpenVINO backend (Intel Arc A530M).
        try:
            logger.info(
                "Loading Faster-Whisper '%s' with OpenVINO backend …",
                self._model_size,
            )
            model = WhisperModel(
                self._model_size,
                device="openvino",
                compute_type=_COMPUTE_TYPE,
            )
            logger.info("Faster-Whisper loaded on OpenVINO backend.")
            return model
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "OpenVINO backend unavailable (%s) — falling back to CPU.",
                exc,
            )

        # CPU fallback.
        logger.info(
            "Loading Faster-Whisper '%s' on CPU …",
            self._model_size,
        )
        model = WhisperModel(
            self._model_size,
            device="cpu",
            compute_type=_COMPUTE_TYPE,
        )
        logger.info("Faster-Whisper loaded on CPU.")
        return model
