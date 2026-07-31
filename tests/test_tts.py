"""Tests for Piper text-to-speech voice responses."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from althea.tts import VoiceResponder


def test_speak_starts_background_thread() -> None:
    """speak() starts playback on a daemon thread."""
    responder = VoiceResponder(model_path="voice.onnx")

    fake_thread = MagicMock()
    with patch("althea.tts.threading.Thread", return_value=fake_thread) as thread_cls:
        responder.speak("hello")

    thread_cls.assert_called_once()
    _, kwargs = thread_cls.call_args
    assert kwargs["daemon"] is True
    fake_thread.start.assert_called_once()


def test_synthesize_and_play_generates_audio_from_text(tmp_path: Path) -> None:
    """_synthesize_and_play writes Piper output audio and plays it."""
    responder = VoiceResponder(
        model_path="models/en_US-amy-medium.onnx",
        piper_binary="piper",
        player_binary="aplay",
    )

    wav_file = tmp_path / "speech.wav"

    mock_tmp = MagicMock()
    mock_tmp.__enter__.return_value.name = str(wav_file)
    mock_tmp.__exit__.return_value = False

    seen: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        if cmd[0] == "piper":
            seen["input"] = kwargs["input"]
            seen["check"] = kwargs["check"]
            output_file = Path(cmd[cmd.index("--output_file") + 1])
            output_file.write_bytes(b"RIFF....WAVEdata")
        if cmd[0] == "aplay":
            assert Path(cmd[1]).read_bytes() == b"RIFF....WAVEdata"
        return MagicMock()

    with (
        patch("althea.tts.tempfile.NamedTemporaryFile", return_value=mock_tmp),
        patch("althea.tts.subprocess.run", side_effect=_fake_run) as run_mock,
    ):
        responder._synthesize_and_play("hello world")

    assert run_mock.call_count == 2
    piper_call = run_mock.call_args_list[0]
    assert seen["input"] == "hello world"
    assert seen["check"] is True
    assert piper_call.args[0][:3] == [
        "piper",
        "--model",
        "models/en_US-amy-medium.onnx",
    ]
    assert "--output_file" in piper_call.args[0]

    aplay_call = run_mock.call_args_list[1]
    assert aplay_call.args[0] == ["aplay", str(wav_file)]
    assert aplay_call.kwargs["check"] is True
    assert not wav_file.exists()


def test_synthesize_and_play_raises_when_audio_is_empty(tmp_path: Path) -> None:
    """_synthesize_and_play fails if Piper outputs no audio bytes."""
    responder = VoiceResponder(model_path="models/en_US-amy-medium.onnx")
    wav_file = tmp_path / "speech.wav"

    mock_tmp = MagicMock()
    mock_tmp.__enter__.return_value.name = str(wav_file)
    mock_tmp.__exit__.return_value = False

    def _fake_run(cmd, **kwargs):  # noqa: ARG001
        if cmd[0] == "piper":
            Path(cmd[cmd.index("--output_file") + 1]).write_bytes(b"")
        return MagicMock()

    with (
        patch("althea.tts.tempfile.NamedTemporaryFile", return_value=mock_tmp),
        patch("althea.tts.subprocess.run", side_effect=_fake_run),
    ):
        with pytest.raises(RuntimeError, match="no audio bytes were generated"):
            responder._synthesize_and_play("hello world")
