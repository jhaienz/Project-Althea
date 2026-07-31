"""Integration tests for the voice-to-agent main loop wiring."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

import althea.main as main_mod


def test_main_run_executes_full_voice_to_agent_pipeline(caplog) -> None:
    """Wake word -> VAD -> transcription -> agent -> response is fully wired."""
    main_mod._shutdown_event.clear()
    fake_audio = np.zeros(16_000, dtype=np.float32)

    agent = MagicMock()
    agent.run = AsyncMock(return_value="done")
    transcriber = MagicMock()
    transcriber.transcribe.return_value = "open browser"
    voice_responder = MagicMock()
    voice_responder.speak.side_effect = (
        lambda _text, on_complete=None: on_complete() if on_complete else None
    )

    vad = MagicMock()
    detector = MagicMock()

    def _build_vad(*, on_utterance):
        vad.start_capture.side_effect = lambda: on_utterance(fake_audio)
        return vad

    def _build_detector(*, on_wake_word):
        detector.start.side_effect = lambda: (on_wake_word(), main_mod._shutdown_event.set())
        return detector

    with (
        patch("althea.agent.AltheaAgent", return_value=agent),
        patch("althea.transcription.Transcriber", return_value=transcriber),
        patch("althea.tts.VoiceResponder", return_value=voice_responder),
        patch("althea.vad.VoiceActivityDetector", side_effect=_build_vad),
        patch("althea.wake_word.WakeWordDetector", side_effect=_build_detector),
        patch("signal.signal"),
        patch("sys.exit"),
        caplog.at_level(logging.INFO, logger="althea.main"),
    ):
        main_mod.run()

    transcriber.transcribe.assert_called_once()
    agent.run.assert_awaited_once_with("open browser")
    voice_responder.speak.assert_called_once()
    vad.start_capture.assert_called_once()
    vad.stop.assert_called_once()
    detector.start.assert_called_once()
    detector.stop.assert_called_once()

    logs = [r.message for r in caplog.records]
    assert "State transition: idle -> wake-word-detected" in logs
    assert "State transition: wake-word-detected -> listening" in logs
    assert "State transition: listening -> transcribing" in logs
    assert "State transition: transcribing -> reasoning" in logs
    assert "State transition: reasoning -> responding" in logs
    assert "State transition: responding -> idle" in logs
    assert "Command: open browser" in logs
    assert "Althea: done" in logs


def test_main_run_handles_agent_error_and_returns_idle(caplog) -> None:
    """If agent reasoning fails, the loop logs and returns to idle state."""
    main_mod._shutdown_event.clear()
    fake_audio = np.zeros(16_000, dtype=np.float32)

    agent = MagicMock()
    agent.run = AsyncMock(side_effect=RuntimeError("boom"))
    transcriber = MagicMock()
    transcriber.transcribe.return_value = "do thing"
    voice_responder = MagicMock()

    vad = MagicMock()
    detector = MagicMock()

    def _build_vad(*, on_utterance):
        vad.start_capture.side_effect = lambda: on_utterance(fake_audio)
        return vad

    def _build_detector(*, on_wake_word):
        detector.start.side_effect = lambda: (on_wake_word(), main_mod._shutdown_event.set())
        return detector

    with (
        patch("althea.agent.AltheaAgent", return_value=agent),
        patch("althea.transcription.Transcriber", return_value=transcriber),
        patch("althea.tts.VoiceResponder", return_value=voice_responder),
        patch("althea.vad.VoiceActivityDetector", side_effect=_build_vad),
        patch("althea.wake_word.WakeWordDetector", side_effect=_build_detector),
        patch("signal.signal"),
        patch("sys.exit"),
        caplog.at_level(logging.INFO, logger="althea.main"),
    ):
        main_mod.run()

    logs = [r.message for r in caplog.records]
    assert "State transition: transcribing -> reasoning" in logs
    assert "State transition: reasoning -> idle" in logs
    assert any("Agent failed to process command: do thing" in r.message for r in caplog.records)


def test_main_run_handles_transcription_error_and_returns_idle(caplog) -> None:
    """If transcription fails, the loop logs and returns to idle."""
    main_mod._shutdown_event.clear()
    fake_audio = np.zeros(16_000, dtype=np.float32)

    agent = MagicMock()
    agent.run = AsyncMock(return_value="unused")
    transcriber = MagicMock()
    transcriber.transcribe.side_effect = RuntimeError("bad audio")
    voice_responder = MagicMock()

    vad = MagicMock()
    detector = MagicMock()

    def _build_vad(*, on_utterance):
        vad.start_capture.side_effect = lambda: on_utterance(fake_audio)
        return vad

    def _build_detector(*, on_wake_word):
        detector.start.side_effect = lambda: (on_wake_word(), main_mod._shutdown_event.set())
        return detector

    with (
        patch("althea.agent.AltheaAgent", return_value=agent),
        patch("althea.transcription.Transcriber", return_value=transcriber),
        patch("althea.tts.VoiceResponder", return_value=voice_responder),
        patch("althea.vad.VoiceActivityDetector", side_effect=_build_vad),
        patch("althea.wake_word.WakeWordDetector", side_effect=_build_detector),
        patch("signal.signal"),
        patch("sys.exit"),
        caplog.at_level(logging.INFO, logger="althea.main"),
    ):
        main_mod.run()

    logs = [r.message for r in caplog.records]
    assert "State transition: listening -> transcribing" in logs
    assert "State transition: transcribing -> idle" in logs
    assert any("Transcription failed." in r.message for r in caplog.records)
    agent.run.assert_not_called()


def test_main_run_handles_vad_start_error_and_returns_idle(caplog) -> None:
    """If VAD cannot start after wake word, the loop logs and returns to idle."""
    main_mod._shutdown_event.clear()

    agent = MagicMock()
    agent.run = AsyncMock(return_value="unused")
    transcriber = MagicMock()
    voice_responder = MagicMock()

    vad = MagicMock()
    vad.start_capture.side_effect = RuntimeError("no microphone")
    detector = MagicMock()

    def _build_vad(*, on_utterance):  # noqa: ARG001
        return vad

    def _build_detector(*, on_wake_word):
        detector.start.side_effect = lambda: (on_wake_word(), main_mod._shutdown_event.set())
        return detector

    with (
        patch("althea.agent.AltheaAgent", return_value=agent),
        patch("althea.transcription.Transcriber", return_value=transcriber),
        patch("althea.tts.VoiceResponder", return_value=voice_responder),
        patch("althea.vad.VoiceActivityDetector", side_effect=_build_vad),
        patch("althea.wake_word.WakeWordDetector", side_effect=_build_detector),
        patch("signal.signal"),
        patch("sys.exit"),
        caplog.at_level(logging.INFO, logger="althea.main"),
    ):
        main_mod.run()

    logs = [r.message for r in caplog.records]
    assert "State transition: idle -> wake-word-detected" in logs
    assert "State transition: wake-word-detected -> listening" in logs
    assert "State transition: listening -> idle" in logs
    assert any(
        "Failed to start VAD capture after wake word." in r.message
        for r in caplog.records
    )
