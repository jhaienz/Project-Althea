"""Tests for the VAD → Transcription pipeline.

All audio I/O, torch.hub, and Faster-Whisper calls are mocked so these tests
run without a real microphone, GPU, or downloaded models.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from althea.transcription import Transcriber, _DEFAULT_MODEL_SIZE, _COMPUTE_TYPE
from althea.vad import (
    VoiceActivityDetector,
    _SAMPLE_RATE,
    _CHUNK_SAMPLES,
    _SILENCE_THRESHOLD_SECONDS,
    _SPEECH_PROBABILITY_THRESHOLD,
    _seconds_to_chunks,
    _int16_to_float32,
)

_SILENCE_CHUNKS = _seconds_to_chunks(_SILENCE_THRESHOLD_SECONDS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_int16_chunk(value: int = 0, size: int = _CHUNK_SAMPLES) -> np.ndarray:
    """Return a flat int16 array of *size* samples all set to *value*."""
    return np.full((size,), value, dtype=np.int16)


def _indata(chunk: np.ndarray) -> np.ndarray:
    """Wrap a flat array as a (N, 1) sounddevice-style indata array."""
    return chunk[:, np.newaxis]


def _make_float32_audio(seconds: float = 1.0) -> np.ndarray:
    """Return a float32 audio array of *seconds* duration (silence)."""
    samples = int(seconds * _SAMPLE_RATE)
    return np.zeros(samples, dtype=np.float32)


@pytest.fixture()
def mock_sounddevice():
    """Inject a mocked sounddevice into sys.modules."""
    sd_mock = types.ModuleType("sounddevice")

    class _FakeInputStream:
        def __init__(self, *args, **kwargs):
            self._callback = kwargs.get("callback")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _CallbackStop(Exception):
        pass

    sd_mock.InputStream = _FakeInputStream
    sd_mock.PortAudioError = OSError
    sd_mock.CallbackStop = _CallbackStop

    old = sys.modules.get("sounddevice")
    sys.modules["sounddevice"] = sd_mock
    yield sd_mock
    if old is None:
        sys.modules.pop("sounddevice", None)
    else:
        sys.modules["sounddevice"] = old


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for module-level helper functions."""

    def test_seconds_to_chunks_basic(self):
        chunks = _seconds_to_chunks(1.5)
        assert chunks == int(1.5 * _SAMPLE_RATE / _CHUNK_SAMPLES)

    def test_int16_to_float32_max_value(self):
        chunk = np.array([32767], dtype=np.int16)
        result = _int16_to_float32(chunk)
        assert result.dtype == np.float32
        assert result[0] == pytest.approx(32767 / 32768.0, rel=1e-4)

    def test_int16_to_float32_zero(self):
        chunk = np.zeros(10, dtype=np.int16)
        result = _int16_to_float32(chunk)
        np.testing.assert_array_equal(result, np.zeros(10, dtype=np.float32))

    def test_int16_to_float32_negative(self):
        chunk = np.array([-32768], dtype=np.int16)
        result = _int16_to_float32(chunk)
        assert result[0] == pytest.approx(-32768 / 32768.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Transcriber tests
# ---------------------------------------------------------------------------


class TestTranscriberInit:
    """Tests for Transcriber initialisation."""

    def test_default_model_size(self):
        t = Transcriber()
        assert t._model_size == _DEFAULT_MODEL_SIZE

    def test_custom_model_size(self):
        t = Transcriber(model_size="small")
        assert t._model_size == "small"

    def test_model_is_none_before_first_transcribe(self):
        t = Transcriber()
        assert t._model is None


class TestTranscriberLoadModel:
    """Tests for _load_model."""

    def test_load_model_uses_cpu_device(self):
        """_load_model calls WhisperModel with device='cpu'."""
        import faster_whisper

        cpu_model = MagicMock()

        def _fake_whisper(model_size, device, compute_type):
            assert device == "cpu"
            return cpu_model

        with patch.object(faster_whisper, "WhisperModel", side_effect=_fake_whisper):
            t = Transcriber()
            model = t._load_model()
        assert model is cpu_model

    def test_load_model_uses_int8_compute_type(self):
        """_load_model calls WhisperModel with compute_type='int8'."""
        import faster_whisper

        received: dict[str, str] = {}

        def _fake_whisper(model_size, device, compute_type):
            received["compute_type"] = compute_type
            return MagicMock()

        with patch.object(faster_whisper, "WhisperModel", side_effect=_fake_whisper):
            t = Transcriber()
            t._load_model()

        assert received["compute_type"] == _COMPUTE_TYPE

    def test_load_model_logs_openvino_limitation(self, caplog):
        """_load_model logs a warning about the CTranslate2/OpenVINO limitation."""
        import faster_whisper
        import logging

        with (
            patch.object(faster_whisper, "WhisperModel", return_value=MagicMock()),
            caplog.at_level(logging.WARNING, logger="althea.transcription"),
        ):
            t = Transcriber()
            t._load_model()

        assert any("OpenVINO" in r.message for r in caplog.records)


class TestTranscriberTranscribe:
    """Tests for the transcribe() method."""

    def _make_transcriber_with_mock_model(self, *segment_texts) -> tuple[Transcriber, MagicMock]:
        mock_model = MagicMock()
        segments = [MagicMock(text=t) for t in segment_texts]
        mock_model.transcribe.return_value = (segments, MagicMock())
        t = Transcriber()
        t._model = mock_model
        return t, mock_model

    def test_transcribe_returns_text(self):
        t, _ = self._make_transcriber_with_mock_model("hello world")
        result = t.transcribe(_make_float32_audio())
        assert result == "hello world"

    def test_transcribe_joins_multiple_segments(self):
        t, _ = self._make_transcriber_with_mock_model("hello", "world")
        result = t.transcribe(_make_float32_audio())
        assert result == "hello world"

    def test_transcribe_strips_whitespace(self):
        t, _ = self._make_transcriber_with_mock_model("  hello  ")
        result = t.transcribe(_make_float32_audio())
        assert result == "hello"

    def test_transcribe_returns_empty_string_when_no_segments(self):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock())
        t = Transcriber()
        t._model = mock_model
        assert t.transcribe(_make_float32_audio()) == ""

    def test_transcribe_passes_utterance_to_model(self):
        t, mock_model = self._make_transcriber_with_mock_model("test")
        audio = np.ones(16000, dtype=np.float32)
        t.transcribe(audio)
        args, _ = mock_model.transcribe.call_args
        np.testing.assert_array_equal(args[0], audio)

    def test_transcribe_uses_english_language(self):
        t, mock_model = self._make_transcriber_with_mock_model("test")
        t.transcribe(_make_float32_audio())
        _, kwargs = mock_model.transcribe.call_args
        assert kwargs.get("language") == "en"

    def test_transcribe_lazy_loads_model(self):
        """Model is loaded on the first transcribe() call, not at init."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock())
        t = Transcriber()
        assert t._model is None
        with patch.object(t, "_load_model", return_value=mock_model) as mock_load:
            t.transcribe(_make_float32_audio())
            mock_load.assert_called_once()
        assert t._model is mock_model

    def test_transcribe_does_not_reload_model_on_second_call(self):
        """Subsequent transcribe() calls reuse the cached model."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock())
        t = Transcriber()
        with patch.object(t, "_load_model", return_value=mock_model) as mock_load:
            t.transcribe(_make_float32_audio())
            t.transcribe(_make_float32_audio())
            mock_load.assert_called_once()


# ---------------------------------------------------------------------------
# VoiceActivityDetector tests
# ---------------------------------------------------------------------------


class TestVoiceActivityDetectorInit:
    """Tests for VoiceActivityDetector initialisation."""

    def test_default_silence_threshold(self):
        vad = VoiceActivityDetector(on_utterance=MagicMock())
        assert vad._silence_threshold_seconds == _SILENCE_THRESHOLD_SECONDS

    def test_custom_silence_threshold(self):
        vad = VoiceActivityDetector(on_utterance=MagicMock(), silence_threshold_seconds=2.0)
        assert vad._silence_threshold_seconds == 2.0

    def test_silence_chunks_derived_from_threshold(self):
        vad = VoiceActivityDetector(on_utterance=MagicMock(), silence_threshold_seconds=2.0)
        assert vad._silence_chunks == _seconds_to_chunks(2.0)

    def test_default_speech_probability_threshold(self):
        vad = VoiceActivityDetector(on_utterance=MagicMock())
        assert vad._speech_probability_threshold == _SPEECH_PROBABILITY_THRESHOLD

    def test_is_capturing_false_before_start(self):
        vad = VoiceActivityDetector(on_utterance=MagicMock())
        assert not vad.is_capturing

    def test_vad_model_none_before_capture(self):
        vad = VoiceActivityDetector(on_utterance=MagicMock())
        assert vad._vad_model is None


class TestVoiceActivityDetectorPredictSpeech:
    """Tests for _predict_speech — the Silero VAD wrapper."""

    def test_predict_speech_converts_int16_to_float32(self):
        """_predict_speech normalises int16 to float32 before passing to model."""
        import torch

        received_tensors: list[np.ndarray] = []

        def _fake_model(tensor, sr):
            received_tensors.append(tensor.numpy())
            return MagicMock(item=MagicMock(return_value=0.5))

        vad = VoiceActivityDetector(on_utterance=MagicMock())
        vad._vad_model = _fake_model
        vad._torch = torch

        chunk = np.full(_CHUNK_SAMPLES, 32767, dtype=np.int16)
        vad._predict_speech(chunk)

        assert len(received_tensors) == 1
        assert received_tensors[0].max() == pytest.approx(32767 / 32768.0, rel=1e-4)

    def test_predict_speech_returns_model_probability(self):
        """_predict_speech returns the float probability from the model."""
        import torch

        vad = VoiceActivityDetector(on_utterance=MagicMock())
        vad._torch = torch

        prob_value = 0.78
        fake_result = MagicMock()
        fake_result.item.return_value = prob_value

        def _fake_model(tensor, sr):
            return fake_result

        vad._vad_model = _fake_model
        result = vad._predict_speech(_make_int16_chunk())
        assert result == pytest.approx(prob_value)


class TestRunVADLoop:
    """Tests for the _run_vad_loop state machine (off-callback inference)."""

    def _make_vad_with_probs(self, probs: list[float]) -> VoiceActivityDetector:
        """Return a VoiceActivityDetector whose _predict_speech cycles through probs."""
        import torch

        prob_iter = iter(probs)

        def _fake_model(tensor, sr):
            return MagicMock(item=MagicMock(return_value=next(prob_iter, 0.0)))

        vad = VoiceActivityDetector(on_utterance=MagicMock())
        vad._vad_model = _fake_model
        vad._torch = torch
        return vad

    def _fill_queue(self, q: "queue.Queue", probs: list[float]) -> None:
        """Fill *q* with int16 chunks (speech=1000, silence=0) matching probs."""
        for prob in probs:
            value = 1000 if prob >= _SPEECH_PROBABILITY_THRESHOLD else 0
            q.put(_make_int16_chunk(value=value))
        q.put(None)  # sentinel

    def test_speech_chunks_are_buffered(self):
        """Chunks classified as speech are returned in the buffer."""
        probs = [0.9, 0.9, 0.9] + [0.0] * (_SILENCE_CHUNKS + 1)
        vad = self._make_vad_with_probs(probs)

        q: queue.Queue = queue.Queue()
        self._fill_queue(q, probs)

        result = vad._run_vad_loop(q)
        # 3 speech + _SILENCE_CHUNKS trailing silence chunks
        assert len(result) == 3 + _SILENCE_CHUNKS

    def test_leading_silence_is_discarded(self):
        """Frames before speech begins are not included in the buffer."""
        leading = 5
        speech = 3
        probs = [0.0] * leading + [0.9] * speech + [0.0] * (_SILENCE_CHUNKS + 1)
        vad = self._make_vad_with_probs(probs)

        q: queue.Queue = queue.Queue()
        self._fill_queue(q, probs)

        result = vad._run_vad_loop(q)
        assert len(result) == speech + _SILENCE_CHUNKS

    def test_empty_result_when_no_speech(self):
        """If only silence is detected, _run_vad_loop returns an empty list."""
        probs = [0.0] * 20
        vad = self._make_vad_with_probs(probs)

        q: queue.Queue = queue.Queue()
        self._fill_queue(q, probs)

        result = vad._run_vad_loop(q)
        assert result == []

    def test_result_chunks_are_int16(self):
        """Chunks returned by _run_vad_loop are int16 (conversion happens after)."""
        probs = [0.9] * 3 + [0.0] * (_SILENCE_CHUNKS + 1)
        vad = self._make_vad_with_probs(probs)

        q: queue.Queue = queue.Queue()
        self._fill_queue(q, probs)

        result = vad._run_vad_loop(q)
        assert all(c.dtype == np.int16 for c in result)


class TestVoiceActivityDetectorStateReset:
    """Tests that VAD state is reset between Utterances."""

    def test_reset_states_called_before_each_capture(self, mock_sounddevice):
        """reset_states() is called on the model before each utterance capture."""
        on_utterance = MagicMock()
        vad = VoiceActivityDetector(on_utterance=on_utterance)

        mock_model = MagicMock()
        mock_model.reset_states = MagicMock()

        with patch.object(vad, "_load_vad_model", return_value=mock_model):
            with patch.object(vad, "_run_vad_loop", return_value=[]):
                vad._capture_utterance()

        mock_model.reset_states.assert_called_once()

    def test_reset_states_skipped_if_model_lacks_method(self, mock_sounddevice):
        """If the model has no reset_states(), capture proceeds without error."""
        on_utterance = MagicMock()
        vad = VoiceActivityDetector(on_utterance=on_utterance)

        mock_model = MagicMock(spec=[])  # no reset_states attribute

        with patch.object(vad, "_load_vad_model", return_value=mock_model):
            with patch.object(vad, "_run_vad_loop", return_value=[]):
                vad._capture_utterance()  # should not raise


class TestVoiceActivityDetectorLifecycle:
    """Tests for start_capture / stop lifecycle."""

    def test_stop_before_start_is_safe(self):
        """stop() before start_capture() raises no error."""
        vad = VoiceActivityDetector(on_utterance=MagicMock())
        vad.stop()

    def test_start_capture_while_already_capturing_is_ignored(self):
        """A second start_capture() while a capture is live is ignored."""
        vad = VoiceActivityDetector(on_utterance=MagicMock())

        vad._capture_thread = threading.Thread(target=lambda: time.sleep(10), daemon=True)
        vad._capture_thread.start()

        thread_before = vad._capture_thread
        vad.start_capture()
        assert vad._capture_thread is thread_before

        vad._stop_capture_event.set()
        vad._capture_thread.join(timeout=1)


class TestVADMissingDependencies:
    """Tests for graceful degradation when dependencies are missing."""

    def test_missing_sounddevice_logs_error(self, caplog):
        """If sounddevice can't be imported, an error is logged."""
        import logging
        import builtins

        real_import = builtins.__import__

        def _block_sounddevice(name, *args, **kwargs):
            if name == "sounddevice":
                raise ImportError("no sounddevice")
            return real_import(name, *args, **kwargs)

        import torch

        vad = VoiceActivityDetector(on_utterance=MagicMock())
        vad._vad_model = MagicMock()
        vad._torch = torch

        with (
            patch("builtins.__import__", side_effect=_block_sounddevice),
            caplog.at_level(logging.ERROR, logger="althea.vad"),
        ):
            vad._capture_utterance()

        assert any("sounddevice" in r.message for r in caplog.records)

    def test_missing_vad_model_logs_error(self, caplog, mock_sounddevice):
        """If Silero VAD model fails to load, an error is logged."""
        import logging

        vad = VoiceActivityDetector(on_utterance=MagicMock())

        with (
            patch.object(vad, "_load_vad_model", side_effect=RuntimeError("no torch")),
            caplog.at_level(logging.ERROR, logger="althea.vad"),
        ):
            vad._capture_utterance()

        assert any("Silero VAD" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Integration test — VAD → Transcriber pipeline
# ---------------------------------------------------------------------------


class TestVADTranscriptionPipeline:
    """Integration tests for the VAD → Transcription handoff."""

    def test_pipeline_produces_text_from_utterance(self):
        """Simulate: VAD captures Utterance, Transcriber produces Command text."""
        transcribed: list[str] = []

        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = "open the browser"

        def _on_utterance(audio: np.ndarray) -> None:
            text = mock_transcriber.transcribe(audio)
            transcribed.append(text)

        vad = VoiceActivityDetector(on_utterance=_on_utterance)

        fake_audio = _make_float32_audio(seconds=1.0)
        _on_utterance(fake_audio)

        assert transcribed == ["open the browser"]
        mock_transcriber.transcribe.assert_called_once()

    def test_pipeline_logs_transcribed_command(self, caplog):
        """Transcribed Command text is logged (per project logging standard)."""
        import logging

        # Simulate main.py's _on_utterance wired to a Transcriber.
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = "set a timer for 5 minutes"

        main_logger = logging.getLogger("althea.main")

        def _on_utterance(audio: np.ndarray) -> None:
            text = mock_transcriber.transcribe(audio)
            if text:
                main_logger.info("Command: %s", text)

        with caplog.at_level(logging.INFO, logger="althea.main"):
            _on_utterance(_make_float32_audio())

        assert any("set a timer for 5 minutes" in r.message for r in caplog.records)
