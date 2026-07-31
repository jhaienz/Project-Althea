"""Transcription module — converts spoken Utterance audio to text.

Uses Faster-Whisper on CPU with int8 quantisation (Intel MKL optimised for
the Intel Arc A530M host CPU).  Note: CTranslate2 (Faster-Whisper's backend)
does not support OpenVINO or Intel discrete GPU acceleration; true OpenVINO
inference requires whisper.cpp with the OpenVINO backend (see ADR-0001 for
future consideration).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from faster_whisper import WhisperModel  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# Default model size — small enough for low latency, accurate enough for most
# Commands.  Can be overridden at construction time.
_DEFAULT_MODEL_SIZE: str = "base"

# int8 quantisation: best latency/accuracy balance on Intel CPUs via MKL.
_COMPUTE_TYPE: str = "int8"

# Sample rate Faster-Whisper expects (16 kHz).
_SAMPLE_RATE: int = 16_000


class Transcriber:
    """Transcribes a spoken Utterance (numpy float32 array) to text.

    Usage::

        transcriber = Transcriber()
        text = transcriber.transcribe(utterance_audio)
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

    def transcribe(self, utterance: np.ndarray) -> str:
        """Transcribe *utterance* and return the Command text.

        Args:
            utterance: 1-D float32 array of audio samples at 16 kHz.

        Returns:
            The transcribed Command text, stripped of leading/trailing
            whitespace.  Returns an empty string if transcription yields no
            segments.
        """
        if self._model is None:
            self._model = self._load_model()

        segments, _ = self._model.transcribe(
            utterance,
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

        Runs on CPU with int8 quantisation, which leverages Intel MKL for
        efficient inference on Intel hardware.  CTranslate2 does not expose
        an OpenVINO device; see the module docstring for alternatives.
        """
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        logger.warning(
            "Faster-Whisper uses CTranslate2, which does not support "
            "OpenVINO/Intel Arc GPU — running on CPU (int8, Intel MKL). "
            "For OpenVINO acceleration, use whisper.cpp with the OpenVINO backend."
        )
        logger.info(
            "Loading Faster-Whisper '%s' on CPU (compute_type=%s) …",
            self._model_size,
            _COMPUTE_TYPE,
        )
        model = WhisperModel(
            self._model_size,
            device="cpu",
            compute_type=_COMPUTE_TYPE,
        )
        logger.info("Faster-Whisper model loaded.")
        return model
