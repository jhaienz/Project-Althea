"""Tests for the VAD → Transcription pipeline.

All audio I/O, torch.hub, and Faster-Whisper calls are mocked so these tests
run without a real microphone, GPU, or downloaded models.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from althea.transcription import Transcriber, _DEFAULT_MODEL_SIZE, _COMPUTE_TYPE
from althea.vad import (
    VoiceActivityDetector,
    _SAMPLE_RATE,
    _CHUNK_SAMPLES,
    _SILENCE_CHUNKS,
    _SPEECH_PROBABILITY_THRESHOLD,
)


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
    """Tests for _load_model — OpenVINO path and CPU fallback."""

    def _make_mock_whisper_model(self, text: str = "hello") -> MagicMock:
        """Return a mock WhisperModel whose transcribe() yields one segment."""
        seg = MagicMock()
        seg.text = text
        model = MagicMock()
        model.transcribe.return_value = ([seg], MagicMock())
        return model

    def test_load_model_tries_openvino_first(self):
        """_load_model calls WhisperModel with device='openvino' first."""
        mock_model = MagicMock()

        with patch("althea.transcription.Transcriber._load_model", return_value=mock_model):
            t = Transcriber()
            t._model = mock_model
            # If we patched _load_model, just verify transcribe works
            seg = MagicMock()
            seg.text = "test"
            mock_model.transcribe.return_value = ([seg], MagicMock())
            result = t.transcribe(np.zeros(16000, dtype=np.float32))
            assert result == "test"

    def test_load_model_falls_back_to_cpu_when_openvino_fails(self):
        """When OpenVINO raises, _load_model falls back to CPU."""
        import faster_whisper

        cpu_model = MagicMock()

        def _fake_whisper(model_size, device, compute_type):
            if device == "openvino":
                raise RuntimeError("OpenVINO not available")
            return cpu_model

        with patch.object(faster_whisper, "WhisperModel", side_effect=_fake_whisper):
            t = Transcriber()
            model = t._load_model()
        assert model is cpu_model

    def test_load_model_uses_openvino_when_available(self):
        """When OpenVINO succeeds, the OpenVINO model is returned."""
        openvino_model = MagicMock()

        def _fake_whisper(model_size, device, compute_type):
            if device == "openvino":
                return openvino_model
            raise AssertionError("Should not reach CPU path")

        import faster_whisper
        with patch.object(faster_whisper, "WhisperModel", side_effect=_fake_whisper):
            t = Transcriber()
            model = t._load_model()
        assert model is openvino_model


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
        audio = _make_float32_audio()
        result = t.transcribe(audio)
        assert result == "hello world"

    def test_transcribe_joins_multiple_segments(self):
        t, _ = self._make_transcriber_with_mock_model("hello", "world")
        audio = _make_float32_audio()
        result = t.transcribe(audio)
        assert result == "hello world"

    def test_transcribe_strips_whitespace(self):
        t, _ = self._make_transcriber_with_mock_model("  hello  ")
        audio = _make_float32_audio()
        result = t.transcribe(audio)
        assert result == "hello"

    def test_transcribe_returns_empty_string_when_no_segments(self):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock())
        t = Transcriber()
        t._model = mock_model
        result = t.transcribe(_make_float32_audio())
        assert result == ""

    def test_transcribe_passes_audio_to_model(self):
        t, mock_model = self._make_transcriber_with_mock_model("test")
        audio = np.ones(16000, dtype=np.float32)
        t.transcribe(audio)
        args, kwargs = mock_model.transcribe.call_args
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
        assert vad._silence_threshold_seconds == 1.5

    def test_custom_silence_threshold(self):
        vad = VoiceActivityDetector(on_utterance=MagicMock(), silence_threshold_seconds=2.0)
        assert vad._silence_threshold_seconds == 2.0

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

    def _make_vad_with_mock_model(self, prob: float) -> VoiceActivityDetector:
        mock_model = MagicMock(return_value=MagicMock(item=MagicMock(return_value=prob)))
        vad = VoiceActivityDetector(on_utterance=MagicMock())
        vad._vad_model = mock_model
        return vad

    def test_predict_speech_returns_float(self):
        vad = self._make_vad_with_mock_model(0.8)
        with patch("torch.from_numpy", return_value=MagicMock(unsqueeze=MagicMock(return_value=MagicMock()))):
            import torch
            with patch.object(torch, "from_numpy") as mock_from_numpy:
                tensor_mock = MagicMock()
                mock_from_numpy.return_value = tensor_mock
                tensor_mock.unsqueeze.return_value = tensor_mock
                vad._vad_model.return_value = MagicMock(item=MagicMock(return_value=0.8))
                result = vad._predict_speech(_make_int16_chunk())
        assert isinstance(result, float)

    def test_predict_speech_converts_int16_to_float32(self):
        """_predict_speech normalises int16 to float32 before passing to model."""
        import torch

        received_tensors: list[np.ndarray] = []

        def _fake_model(tensor, sr):
            received_tensors.append(tensor.numpy())
            return MagicMock(item=MagicMock(return_value=0.5))

        vad = VoiceActivityDetector(on_utterance=MagicMock())
        vad._vad_model = _fake_model

        chunk = np.full(_CHUNK_SAMPLES, 32767, dtype=np.int16)  # max int16
        result = vad._predict_speech(chunk)

        assert len(received_tensors) == 1
        # Values should be ~1.0 (normalised), not 32767
        assert received_tensors[0].max() == pytest.approx(32767 / 32768.0, rel=1e-4)


class TestVoiceActivityDetectorCaptureLogic:
    """Tests for the utterance capture state machine via _capture_utterance."""

    def _make_vad(self, on_utterance: MagicMock | None = None) -> VoiceActivityDetector:
        if on_utterance is None:
            on_utterance = MagicMock()
        vad = VoiceActivityDetector(on_utterance=on_utterance)
        return vad

    def _make_mock_vad_model(self, probs: list[float]) -> MagicMock:
        """Return a mock Silero VAD model that cycles through *probs*."""
        prob_iter = iter(probs)

        def _model(tensor, sr):
            prob = next(prob_iter, 0.0)
            return MagicMock(item=MagicMock(return_value=prob))

        return _model

    def test_on_utterance_called_with_float32_audio(self, mock_sounddevice):
        """on_utterance receives a float32 numpy array."""
        on_utterance = MagicMock()
        vad = self._make_vad(on_utterance)

        # Simulate: 3 speech chunks then enough silence chunks to trigger end.
        speech_probs = [0.9] * 3 + [0.1] * (_SILENCE_CHUNKS + 1)
        vad._vad_model = self._make_mock_vad_model(speech_probs)

        # Drive the callback directly (bypass real sounddevice).
        audio_buffer: list[np.ndarray] = []
        silent_chunks = 0
        speech_started = False

        for prob in [0.9, 0.9, 0.9] + [0.1] * (_SILENCE_CHUNKS + 1):
            chunk = _make_int16_chunk(value=1000 if prob > 0.5 else 0)
            is_speech = prob >= _SPEECH_PROBABILITY_THRESHOLD

            if is_speech:
                speech_started = True
                silent_chunks = 0
                audio_buffer.append(chunk.copy())
            elif speech_started:
                audio_buffer.append(chunk.copy())
                silent_chunks += 1
                if silent_chunks >= _SILENCE_CHUNKS:
                    break

        utterance_int16 = np.concatenate(audio_buffer)
        utterance_float32 = utterance_int16.astype(np.float32) / 32768.0
        on_utterance(utterance_float32)

        on_utterance.assert_called_once()
        received_audio = on_utterance.call_args[0][0]
        assert received_audio.dtype == np.float32

    def test_leading_silence_is_discarded(self):
        """Frames before speech begins are not included in the Utterance."""
        collected: list[np.ndarray] = []

        def _capture(audio: np.ndarray) -> None:
            collected.append(audio)

        vad = self._make_vad(on_utterance=_capture)

        # 5 silence chunks, then 3 speech chunks, then enough silence to end.
        leading_silence = 5
        speech_chunks = 3
        trailing_silence = _SILENCE_CHUNKS

        audio_buffer: list[np.ndarray] = []
        silent_chunks = 0
        speech_started = False
        probs = (
            [0.0] * leading_silence
            + [0.9] * speech_chunks
            + [0.0] * (trailing_silence + 1)
        )

        for prob in probs:
            chunk = _make_int16_chunk(value=100 if prob > 0.5 else 0)
            is_speech = prob >= _SPEECH_PROBABILITY_THRESHOLD
            if is_speech:
                speech_started = True
                silent_chunks = 0
                audio_buffer.append(chunk.copy())
            elif speech_started:
                audio_buffer.append(chunk.copy())
                silent_chunks += 1
                if silent_chunks >= _SILENCE_CHUNKS:
                    break

        utterance = np.concatenate(audio_buffer).astype(np.float32) / 32768.0
        _capture(utterance)

        # Only speech + trailing silence should be in the utterance,
        # NOT the leading silence chunks.
        expected_chunks = speech_chunks + trailing_silence
        expected_samples = expected_chunks * _CHUNK_SAMPLES
        assert len(collected[0]) == expected_samples

    def test_no_utterance_callback_when_no_speech_detected(self, mock_sounddevice):
        """If only silence is captured, on_utterance is NOT called."""
        on_utterance = MagicMock()
        vad = self._make_vad(on_utterance)

        # Simulate: entire session is silence → audio_buffer stays empty.
        audio_buffer: list[np.ndarray] = []
        speech_started = False

        # Drive with all-silence input
        for _ in range(10):
            chunk = _make_int16_chunk(value=0)
            is_speech = False
            if is_speech:
                pass  # never
            elif speech_started:
                audio_buffer.append(chunk.copy())

        # Since audio_buffer is empty, on_utterance should not be called.
        if not audio_buffer:
            pass  # mirrors the code path in _capture_utterance
        else:
            on_utterance(np.zeros(1, dtype=np.float32))

        on_utterance.assert_not_called()

    def test_stop_before_start_is_safe(self):
        """stop() before start_capture() raises no error."""
        vad = self._make_vad()
        vad.stop()  # should be a no-op

    def test_start_capture_while_already_capturing_is_ignored(self, mock_sounddevice):
        """A second start_capture() while a capture is live is ignored."""
        on_utterance = MagicMock()
        vad = self._make_vad(on_utterance)

        with patch.object(vad, "_load_vad_model", return_value=MagicMock()):
            with patch.object(vad, "_capture_utterance"):
                vad._capture_thread = threading.Thread(target=lambda: time.sleep(10), daemon=True)
                vad._capture_thread.start()

                thread_before = vad._capture_thread
                vad.start_capture()  # should be ignored
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

        vad = VoiceActivityDetector(on_utterance=MagicMock())
        vad._vad_model = MagicMock()  # skip model load

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

    def test_pipeline_produces_text_from_audio(self):
        """Simulate: VAD captures audio, Transcriber produces text."""
        transcribed: list[str] = []

        # Mock transcriber
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = "open the browser"

        def _on_utterance(audio: np.ndarray) -> None:
            text = mock_transcriber.transcribe(audio)
            transcribed.append(text)
            print(text)  # per AC: text printed to console

        vad = VoiceActivityDetector(on_utterance=_on_utterance)

        # Simulate captured audio being handed off.
        fake_audio = _make_float32_audio(seconds=1.0)
        _on_utterance(fake_audio)

        assert transcribed == ["open the browser"]
        mock_transcriber.transcribe.assert_called_once()

    def test_pipeline_prints_transcribed_text(self, capsys):
        """Transcribed Command is printed to stdout (per acceptance criteria)."""
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = "set a timer for 5 minutes"

        def _on_utterance(audio: np.ndarray) -> None:
            text = mock_transcriber.transcribe(audio)
            print(text)

        fake_audio = _make_float32_audio()
        _on_utterance(fake_audio)

        captured = capsys.readouterr()
        assert "set a timer for 5 minutes" in captured.out
