"""Tests for WakeWordDetector.

All audio input and openWakeWord calls are mocked so these tests run
without a real microphone or the openwakeword / sounddevice packages.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from typing import Any
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from althea.wake_word import WakeWordDetector, _WAKE_WORD_KEY, _DETECTION_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def _make_chunk(value: int = 0, size: int = 1280) -> np.ndarray:
    """Return a flat int16 array of *size* samples all set to *value*."""
    return np.full((size,), value, dtype=np.int16)


def _indata(chunk: np.ndarray) -> np.ndarray:
    """Wrap a flat array as a (N, 1) sounddevice-style indata array."""
    return chunk[:, np.newaxis]


@pytest.fixture()
def mock_sounddevice():
    """Inject a fully mocked ``sounddevice`` module into sys.modules.

    This prevents the real import (which requires libportaudio) from
    running during lifecycle tests that exercise ``_run``.
    """
    sd_mock = types.ModuleType("sounddevice")

    # Provide a mock InputStream context manager that blocks on stop_event.
    class _FakeInputStream:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    sd_mock.InputStream = _FakeInputStream
    sd_mock.PortAudioError = OSError  # match what real sounddevice exposes

    old = sys.modules.get("sounddevice")
    sys.modules["sounddevice"] = sd_mock
    yield sd_mock
    if old is None:
        sys.modules.pop("sounddevice", None)
    else:
        sys.modules["sounddevice"] = old


# ---------------------------------------------------------------------------
# Unit tests — audio callback
# ---------------------------------------------------------------------------


class TestWakeWordDetectorCallback:
    """Tests focused on the _audio_callback method."""

    def _make_detector(self, callback=None):
        if callback is None:
            callback = MagicMock()
        detector = WakeWordDetector(on_wake_word=callback)
        return detector, callback

    def test_callback_fires_when_score_above_threshold(self):
        """on_wake_word is called when model score ≥ threshold."""
        on_wake = MagicMock()
        detector, _ = self._make_detector(on_wake)

        mock_model = MagicMock()
        mock_model.predict.return_value = {_WAKE_WORD_KEY: 0.9}
        detector._model = mock_model

        chunk = _make_chunk()
        detector._audio_callback(_indata(chunk), 1280, None, None)

        on_wake.assert_called_once()

    def test_callback_not_fired_when_score_below_threshold(self):
        """on_wake_word is NOT called when model score < threshold."""
        on_wake = MagicMock()
        detector, _ = self._make_detector(on_wake)

        mock_model = MagicMock()
        mock_model.predict.return_value = {_WAKE_WORD_KEY: 0.1}
        detector._model = mock_model

        chunk = _make_chunk()
        detector._audio_callback(_indata(chunk), 1280, None, None)

        on_wake.assert_not_called()

    def test_callback_not_fired_when_score_exactly_below_threshold(self):
        """Edge: score just below threshold does not trigger."""
        on_wake = MagicMock()
        detector, _ = self._make_detector(on_wake)

        mock_model = MagicMock()
        below = _DETECTION_THRESHOLD - 0.001
        mock_model.predict.return_value = {_WAKE_WORD_KEY: below}
        detector._model = mock_model

        detector._audio_callback(_indata(_make_chunk()), 1280, None, None)
        on_wake.assert_not_called()

    def test_callback_fires_at_exact_threshold(self):
        """Edge: score exactly at threshold should trigger."""
        on_wake = MagicMock()
        detector, _ = self._make_detector(on_wake)

        mock_model = MagicMock()
        mock_model.predict.return_value = {_WAKE_WORD_KEY: _DETECTION_THRESHOLD}
        detector._model = mock_model

        detector._audio_callback(_indata(_make_chunk()), 1280, None, None)
        on_wake.assert_called_once()

    def test_callback_silenced_when_stop_event_set(self):
        """on_wake_word is NOT called if stop_event is set (shutting down)."""
        on_wake = MagicMock()
        detector, _ = self._make_detector(on_wake)

        mock_model = MagicMock()
        mock_model.predict.return_value = {_WAKE_WORD_KEY: 0.99}
        detector._model = mock_model
        detector._stop_event.set()  # simulate shutdown

        detector._audio_callback(_indata(_make_chunk()), 1280, None, None)
        on_wake.assert_not_called()

    def test_callback_silenced_when_model_is_none(self):
        """on_wake_word is NOT called if the model hasn't loaded yet."""
        on_wake = MagicMock()
        detector, _ = self._make_detector(on_wake)
        # _model is None by default

        detector._audio_callback(_indata(_make_chunk()), 1280, None, None)
        on_wake.assert_not_called()

    def test_audio_data_is_passed_to_model_predict(self):
        """The audio chunk is passed to model.predict as a flat array."""
        on_wake = MagicMock()
        detector, _ = self._make_detector(on_wake)

        mock_model = MagicMock()
        mock_model.predict.return_value = {_WAKE_WORD_KEY: 0.0}
        detector._model = mock_model

        chunk = _make_chunk(value=42)
        detector._audio_callback(_indata(chunk), 1280, None, None)

        # Verify predict was called with the flat chunk (channel 0 extracted).
        args, _ = mock_model.predict.call_args
        np.testing.assert_array_equal(args[0], chunk)

    def test_missing_wake_word_key_in_prediction(self):
        """If model prediction lacks the wake word key, no callback fires."""
        on_wake = MagicMock()
        detector, _ = self._make_detector(on_wake)

        mock_model = MagicMock()
        mock_model.predict.return_value = {}  # no key at all
        detector._model = mock_model

        detector._audio_callback(_indata(_make_chunk()), 1280, None, None)
        on_wake.assert_not_called()


# ---------------------------------------------------------------------------
# Unit tests — lifecycle
# ---------------------------------------------------------------------------


class TestWakeWordDetectorLifecycle:
    """Tests for start/stop thread lifecycle."""

    def test_start_launches_background_thread(self, mock_sounddevice):
        """start() creates a live daemon thread."""
        on_wake = MagicMock()
        detector = WakeWordDetector(on_wake_word=on_wake)

        with patch.object(detector, "_load_model", return_value=MagicMock()):
            # Unblock the _run loop after a short delay.
            def _unblock():
                time.sleep(0.1)
                detector._stop_event.set()

            threading.Thread(target=_unblock, daemon=True).start()
            detector.start()
            assert detector._thread is not None
            detector.stop()

    def test_is_running_false_before_start(self):
        detector = WakeWordDetector(on_wake_word=MagicMock())
        assert not detector.is_running

    def test_double_start_raises(self, mock_sounddevice):
        """Calling start() twice raises RuntimeError."""
        on_wake = MagicMock()
        detector = WakeWordDetector(on_wake_word=on_wake)

        with patch.object(detector, "_load_model", return_value=MagicMock()):
            detector.start()
            time.sleep(0.05)  # let thread reach the wait()

            try:
                with pytest.raises(RuntimeError, match="already running"):
                    detector.start()
            finally:
                detector.stop()

    def test_stop_is_idempotent_before_start(self):
        """stop() before start() does not raise."""
        detector = WakeWordDetector(on_wake_word=MagicMock())
        detector.stop()  # should be a no-op

    def test_run_handles_missing_microphone(self, caplog, mock_sounddevice):
        """If PortAudioError is raised, the thread exits cleanly (logged)."""
        on_wake = MagicMock()
        detector = WakeWordDetector(on_wake_word=on_wake)

        # Override mock's InputStream to raise PortAudioError.
        mock_sounddevice.InputStream = MagicMock(
            side_effect=mock_sounddevice.PortAudioError("No device")
        )

        with patch.object(detector, "_load_model", return_value=MagicMock()):
            import logging
            with caplog.at_level(logging.ERROR, logger="althea.wake_word"):
                detector._run()

        assert any("Microphone unavailable" in r.message for r in caplog.records)

    def test_run_handles_missing_sounddevice(self, caplog):
        """If sounddevice is not importable, the thread logs an error and exits."""
        on_wake = MagicMock()
        detector = WakeWordDetector(on_wake_word=on_wake)

        import builtins
        real_import = builtins.__import__

        def _block_sounddevice(name, *args, **kwargs):
            if name == "sounddevice":
                raise ImportError("Mocked missing sounddevice")
            return real_import(name, *args, **kwargs)

        import logging
        with (
            patch("builtins.__import__", side_effect=_block_sounddevice),
            caplog.at_level(logging.ERROR, logger="althea.wake_word"),
        ):
            detector._run()

        assert any("sounddevice" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Unit tests — __init__ defaults
# ---------------------------------------------------------------------------


class TestWakeWordDetectorInit:
    """Tests for __init__ defaults and custom arguments."""

    def test_default_wake_word(self):
        detector = WakeWordDetector(on_wake_word=MagicMock())
        assert detector._wake_word == _WAKE_WORD_KEY

    def test_custom_wake_word(self):
        detector = WakeWordDetector(on_wake_word=MagicMock(), wake_word="hey_jarvis")
        assert detector._wake_word == "hey_jarvis"

    def test_custom_threshold(self):
        detector = WakeWordDetector(on_wake_word=MagicMock(), threshold=0.8)
        assert detector._threshold == 0.8
