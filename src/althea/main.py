"""Althea main loop — starts the assistant and listens for the wake word."""

import asyncio
import logging
import os
import signal
import sys
import threading

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# Set by the SIGINT handler; the main loop blocks on this event.
_shutdown_event = threading.Event()


def _handle_sigint(sig: int, frame: object) -> None:  # noqa: ARG001
    """Signal stop on Ctrl+C (SIGINT) so the main loop can clean up."""
    logger.info("Althea shutting down.")
    _shutdown_event.set()


def run() -> None:
    """Start the Althea main loop."""
    from althea.agent import AltheaAgent
    from althea.tts import VoiceResponder
    from althea.transcription import Transcriber
    from althea.vad import VoiceActivityDetector
    from althea.wake_word import WakeWordDetector

    signal.signal(signal.SIGINT, _handle_sigint)
    logger.info("Althea is running")

    agent = AltheaAgent()
    transcriber = Transcriber()
    configured_tts_model = os.getenv("ALTHEA_PIPER_MODEL_PATH")
    voice_responder = (
        VoiceResponder(model_path=configured_tts_model)
        if configured_tts_model
        else VoiceResponder()
    )
    state = "idle"
    state_lock = threading.Lock()

    def _transition(next_state: str) -> None:
        nonlocal state
        with state_lock:
            logger.info("State transition: %s -> %s", state, next_state)
            state = next_state

    def _on_utterance(audio: np.ndarray) -> None:
        """Called by VoiceActivityDetector with the captured Utterance audio."""
        _transition("transcribing")
        try:
            text = transcriber.transcribe(audio)
        except Exception:
            logger.exception("Transcription failed.")
            _transition("idle")
            return
        if not text:
            logger.warning("Transcription returned empty text.")
            _transition("idle")
            return
        logger.info("Command: %s", text)
        _transition("reasoning")
        try:
            response = asyncio.run(agent.run(text))
        except Exception:
            logger.exception("Agent failed to process command: %s", text)
            _transition("idle")
            return
        _transition("responding")
        if response:
            logger.info("Althea: %s", response)
            voice_responder.speak(response, on_complete=lambda: _transition("idle"))
            return
        _transition("idle")

    vad = VoiceActivityDetector(on_utterance=_on_utterance)

    def _on_wake_word() -> None:
        """Called by WakeWordDetector when the wake word is detected."""
        _transition("wake-word-detected")
        _transition("listening")
        logger.info("Wake word detected — starting VAD capture.")
        try:
            vad.start_capture()
        except Exception:
            logger.exception("Failed to start VAD capture after wake word.")
            _transition("idle")

    detector = WakeWordDetector(on_wake_word=_on_wake_word)
    detector.start()

    try:
        _shutdown_event.wait()
    finally:
        vad.stop()
        detector.stop()
        sys.exit(0)


def main() -> None:
    """Console-script entry point (installed via pyproject.toml)."""
    run()
