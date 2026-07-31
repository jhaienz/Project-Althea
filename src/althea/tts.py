"""Voice response module powered by local Piper TTS."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "amy.onnx"


class VoiceResponder:
    """Generate and play a Voice Response from text using Piper."""

    def __init__(
        self,
        *,
        model_path: str | Path = _DEFAULT_MODEL_PATH,
        piper_binary: str = "piper",
        player_binary: str = "aplay",
    ) -> None:
        self._model_path = model_path
        self._piper_binary = piper_binary
        self._player_binary = player_binary
        self._enabled = True
        if shutil.which(self._piper_binary) is None:
            logger.error(
                "Piper TTS disabled: '%s' not found. Install Piper.",
                self._piper_binary,
            )
            self._enabled = False
        elif shutil.which(self._player_binary) is None:
            logger.error(
                "Piper TTS disabled: '%s' not found. Install an audio player.",
                self._player_binary,
            )
            self._enabled = False

    def speak(self, text: str, on_complete: Callable[[], None] | None = None) -> None:
        """Start voice playback on a background thread."""
        if not self._enabled:
            if on_complete is not None:
                on_complete()
            return
        thread = threading.Thread(
            target=self._speak_worker,
            args=(text, on_complete),
            name="tts-playback",
            daemon=True,
        )
        thread.start()

    def _speak_worker(self, text: str, on_complete: Callable[[], None] | None) -> None:
        """Synthesize and play text, then fire completion callback."""
        try:
            self._synthesize_and_play(text)
        except FileNotFoundError as exc:
            logger.error("Piper playback dependency missing: %s", exc)
        except subprocess.CalledProcessError:
            logger.exception("Piper failed to synthesize or play voice response.")
        except RuntimeError as exc:
            logger.error("Piper produced invalid audio output: %s", exc)
        finally:
            if on_complete is not None:
                on_complete()

    def _synthesize_and_play(self, text: str) -> None:
        """Synthesize *text* to WAV with Piper, then play through system audio."""
        if not text.strip():
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = Path(tmp.name)

        try:
            subprocess.run(
                [
                    self._piper_binary,
                    "--model",
                    self._model_path,
                    "--output_file",
                    str(output_path),
                ],
                input=text,
                text=True,
                check=True,
            )
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError("no audio bytes were generated")
            subprocess.run(
                [self._player_binary, str(output_path)],
                check=True,
            )
        finally:
            output_path.unlink(missing_ok=True)
