"""Integration tests for the voice-to-agent main loop wiring."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

import althea.main as main_mod


def test_wake_word_greeting_finishes_before_command_capture() -> None:
    main_mod._shutdown_event.clear()
    agent = MagicMock()
    transcriber = MagicMock()
    voice_responder = MagicMock()
    vad = MagicMock()
    detector = MagicMock()

    def _speak(_text, on_complete=None):
        vad.start_capture.assert_not_called()
        on_complete()

    voice_responder.speak.side_effect = _speak

    def _build_detector(*, on_wake_word):
        detector.start.side_effect = lambda: (
            on_wake_word(),
            main_mod._shutdown_event.set(),
        )
        return detector

    with (
        patch("althea.agent.AltheaAgent", return_value=agent),
        patch("althea.transcription.Transcriber", return_value=transcriber),
        patch("althea.tts.VoiceResponder", return_value=voice_responder),
        patch("althea.vad.VoiceActivityDetector", return_value=vad),
        patch("althea.wake_word.WakeWordDetector", side_effect=_build_detector),
        patch("althea.tools.browser.browser_stop") as browser_stop,
        patch("althea.main.threading.Timer"),
        patch("signal.signal"),
        patch("sys.exit"),
    ):
        main_mod.run()

    voice_responder.speak.assert_called_once_with(
        "Hi Master Jai, what can I do for you today?",
        on_complete=voice_responder.speak.call_args.kwargs["on_complete"],
    )
    vad.start_capture.assert_called_once()
    browser_stop.assert_called_once_with()


def test_voice_response_finishes_before_follow_up_command_capture(caplog) -> None:
    main_mod._shutdown_event.clear()
    fake_audio = np.zeros(16_000, dtype=np.float32)

    loops = []

    async def _run_agent(_command, *, on_progress=None):
        loops.append(asyncio.get_running_loop())
        return "done"

    async def _stop_browser():
        loops.append(asyncio.get_running_loop())

    agent = MagicMock()
    agent.run = AsyncMock(side_effect=_run_agent)
    transcriber = MagicMock()
    transcriber.transcribe.return_value = "open browser"
    voice_responder = MagicMock()
    voice_responder.speak.side_effect = (
        lambda _text, on_complete=None: on_complete() if on_complete else None
    )

    vad = MagicMock()
    detector = MagicMock()

    def _build_vad(*, on_utterance):
        captures = iter(
            (lambda: on_utterance(fake_audio), lambda: None, lambda: None)
        )
        vad.start_capture.side_effect = lambda: next(captures)()
        return vad

    def _build_detector(*, on_wake_word):
        detector.start.side_effect = lambda: (
            on_wake_word(),
            on_wake_word(),
            main_mod._shutdown_event.set(),
        )
        return detector

    with (
        patch("althea.agent.AltheaAgent", return_value=agent),
        patch("althea.transcription.Transcriber", return_value=transcriber),
        patch("althea.tts.VoiceResponder", return_value=voice_responder),
        patch("althea.vad.VoiceActivityDetector", side_effect=_build_vad),
        patch("althea.wake_word.WakeWordDetector", side_effect=_build_detector),
        patch("althea.tools.browser.browser_stop", side_effect=_stop_browser),
        patch("althea.main.threading.Timer"),
        patch("signal.signal"),
        patch("sys.exit"),
        caplog.at_level(logging.INFO, logger="althea.main"),
    ):
        main_mod.run()

    transcriber.transcribe.assert_called_once()
    agent.run.assert_awaited_once()
    assert agent.run.call_args.args == ("open browser",)
    assert [call.args[0] for call in voice_responder.speak.call_args_list] == [
        "Hi Master Jai, what can I do for you today?",
        "done",
    ]
    assert vad.start_capture.call_count == 2
    vad.stop.assert_called_once()
    detector.start.assert_called_once()
    detector.stop.assert_called_once()
    assert loops[0] is loops[1]

    logs = [r.message for r in caplog.records]
    assert "State transition: idle -> wake-word-detected" in logs
    assert "State transition: wake-word-detected -> responding" in logs
    assert "State transition: responding -> listening" in logs
    assert "State transition: listening -> transcribing" in logs
    assert "State transition: transcribing -> reasoning" in logs
    assert "State transition: reasoning -> responding" in logs
    assert logs.count("State transition: responding -> listening") == 2
    assert "Command: open browser" in logs
    assert "Althea: done" in logs


def test_empty_transcription_prompts_retry_before_listening_again(caplog) -> None:
    main_mod._shutdown_event.clear()
    fake_audio = np.zeros(16_000, dtype=np.float32)
    agent = MagicMock()
    transcriber = MagicMock()
    transcriber.transcribe.return_value = ""
    voice_responder = MagicMock()
    voice_responder.speak.side_effect = (
        lambda _text, on_complete=None: on_complete() if on_complete else None
    )
    vad = MagicMock()
    detector = MagicMock()

    def _build_vad(*, on_utterance):
        captures = iter((lambda: on_utterance(fake_audio), lambda: None))
        vad.start_capture.side_effect = lambda: next(captures)()
        return vad

    def _build_detector(*, on_wake_word):
        detector.start.side_effect = lambda: (
            on_wake_word(),
            main_mod._shutdown_event.set(),
        )
        return detector

    with (
        patch("althea.agent.AltheaAgent", return_value=agent),
        patch("althea.transcription.Transcriber", return_value=transcriber),
        patch("althea.tts.VoiceResponder", return_value=voice_responder),
        patch("althea.vad.VoiceActivityDetector", side_effect=_build_vad),
        patch("althea.wake_word.WakeWordDetector", side_effect=_build_detector),
        patch("althea.main.threading.Timer"),
        patch("signal.signal"),
        patch("sys.exit"),
        caplog.at_level(logging.INFO, logger="althea.main"),
    ):
        main_mod.run()

    assert [call.args[0] for call in voice_responder.speak.call_args_list] == [
        "Hi Master Jai, what can I do for you today?",
        "Sorry, I didn't catch that. Please try again.",
    ]
    assert vad.start_capture.call_count == 2
    agent.run.assert_not_called()
    assert "State transition: transcribing -> responding" in [
        record.message for record in caplog.records
    ]


def test_inactivity_timeout_ends_conversation(caplog) -> None:
    main_mod._shutdown_event.clear()
    agent = MagicMock()
    transcriber = MagicMock()
    voice_responder = MagicMock()
    voice_responder.speak.side_effect = (
        lambda _text, on_complete=None: on_complete() if on_complete else None
    )
    vad = MagicMock()
    detector = MagicMock()
    timer = MagicMock()
    timeout_callback = None

    def _build_timer(_seconds, callback):
        nonlocal timeout_callback
        timeout_callback = callback
        return timer

    def _build_detector(*, on_wake_word):
        def _start():
            on_wake_word()
            timeout_callback()
            main_mod._shutdown_event.set()

        detector.start.side_effect = _start
        return detector

    with (
        patch("althea.agent.AltheaAgent", return_value=agent),
        patch("althea.transcription.Transcriber", return_value=transcriber),
        patch("althea.tts.VoiceResponder", return_value=voice_responder),
        patch("althea.vad.VoiceActivityDetector", return_value=vad),
        patch("althea.wake_word.WakeWordDetector", side_effect=_build_detector),
        patch("althea.main.threading.Timer", side_effect=_build_timer) as timer_class,
        patch("signal.signal"),
        patch("sys.exit"),
        caplog.at_level(logging.INFO, logger="althea.main"),
    ):
        main_mod.run()

    timer_class.assert_called_once_with(30.0, timeout_callback)
    timer.start.assert_called_once()
    agent.reset_session.assert_called_once()
    assert "State transition: listening -> idle" in [
        record.message for record in caplog.records
    ]


def test_exit_phrase_ends_conversation(caplog) -> None:
    main_mod._shutdown_event.clear()
    fake_audio = np.zeros(16_000, dtype=np.float32)
    agent = MagicMock()
    transcriber = MagicMock()
    transcriber.transcribe.return_value = "goodbye"
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
        detector.start.side_effect = lambda: (
            on_wake_word(),
            main_mod._shutdown_event.set(),
        )
        return detector

    with (
        patch("althea.agent.AltheaAgent", return_value=agent),
        patch("althea.transcription.Transcriber", return_value=transcriber),
        patch("althea.tts.VoiceResponder", return_value=voice_responder),
        patch("althea.vad.VoiceActivityDetector", side_effect=_build_vad),
        patch("althea.wake_word.WakeWordDetector", side_effect=_build_detector),
        patch("althea.main.threading.Timer"),
        patch("signal.signal"),
        patch("sys.exit"),
        caplog.at_level(logging.INFO, logger="althea.main"),
    ):
        main_mod.run()

    agent.run.assert_not_called()
    agent.reset_session.assert_called_once()
    voice_responder.speak.assert_called_once()
    assert "State transition: transcribing -> idle" in [
        record.message for record in caplog.records
    ]


def test_agent_error_prompts_retry_before_listening_again(caplog) -> None:
    main_mod._shutdown_event.clear()
    fake_audio = np.zeros(16_000, dtype=np.float32)

    agent = MagicMock()
    agent.run = AsyncMock(side_effect=RuntimeError("boom"))
    transcriber = MagicMock()
    transcriber.transcribe.return_value = "do thing"
    voice_responder = MagicMock()
    voice_responder.speak.side_effect = (
        lambda _text, on_complete=None: on_complete() if on_complete else None
    )

    vad = MagicMock()
    detector = MagicMock()

    def _build_vad(*, on_utterance):
        captures = iter((lambda: on_utterance(fake_audio), lambda: None))
        vad.start_capture.side_effect = lambda: next(captures)()
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
        patch("althea.main.threading.Timer"),
        patch("signal.signal"),
        patch("sys.exit"),
        caplog.at_level(logging.INFO, logger="althea.main"),
    ):
        main_mod.run()

    logs = [r.message for r in caplog.records]
    assert "State transition: transcribing -> reasoning" in logs
    assert "State transition: reasoning -> responding" in logs
    assert "State transition: responding -> listening" in logs
    assert any("Agent failed to process command: do thing" in r.message for r in caplog.records)
    assert [call.args[0] for call in voice_responder.speak.call_args_list] == [
        "Hi Master Jai, what can I do for you today?",
        "Sorry, something went wrong. Please try again.",
    ]
    assert vad.start_capture.call_count == 2


def test_transcription_error_prompts_retry_before_listening_again(caplog) -> None:
    main_mod._shutdown_event.clear()
    fake_audio = np.zeros(16_000, dtype=np.float32)

    agent = MagicMock()
    agent.run = AsyncMock(return_value="unused")
    transcriber = MagicMock()
    transcriber.transcribe.side_effect = RuntimeError("bad audio")
    voice_responder = MagicMock()
    voice_responder.speak.side_effect = (
        lambda _text, on_complete=None: on_complete() if on_complete else None
    )

    vad = MagicMock()
    detector = MagicMock()

    def _build_vad(*, on_utterance):
        captures = iter((lambda: on_utterance(fake_audio), lambda: None))
        vad.start_capture.side_effect = lambda: next(captures)()
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
        patch("althea.main.threading.Timer"),
        patch("signal.signal"),
        patch("sys.exit"),
        caplog.at_level(logging.INFO, logger="althea.main"),
    ):
        main_mod.run()

    logs = [r.message for r in caplog.records]
    assert "State transition: listening -> transcribing" in logs
    assert "State transition: transcribing -> responding" in logs
    assert "State transition: responding -> listening" in logs
    assert any("Transcription failed." in r.message for r in caplog.records)
    assert [call.args[0] for call in voice_responder.speak.call_args_list] == [
        "Hi Master Jai, what can I do for you today?",
        "Sorry, I didn't catch that. Please try again.",
    ]
    assert vad.start_capture.call_count == 2
    agent.run.assert_not_called()


def test_main_run_handles_vad_start_error_and_returns_idle(caplog) -> None:
    """If VAD cannot start after wake word, the loop logs and returns to idle."""
    main_mod._shutdown_event.clear()

    agent = MagicMock()
    agent.run = AsyncMock(return_value="unused")
    transcriber = MagicMock()
    voice_responder = MagicMock()
    voice_responder.speak.side_effect = (
        lambda _text, on_complete=None: on_complete() if on_complete else None
    )

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
        patch("althea.main.threading.Timer"),
        patch("signal.signal"),
        patch("sys.exit"),
        caplog.at_level(logging.INFO, logger="althea.main"),
    ):
        main_mod.run()

    logs = [r.message for r in caplog.records]
    assert "State transition: idle -> wake-word-detected" in logs
    assert "State transition: wake-word-detected -> responding" in logs
    assert "State transition: responding -> listening" in logs
    assert "State transition: listening -> idle" in logs
    assert any(
        "Failed to start VAD capture." in r.message
        for r in caplog.records
    )
