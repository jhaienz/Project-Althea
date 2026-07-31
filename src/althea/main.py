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

_GREETING = "Hi Master Jai, what can I do for you today?"
_RETRY_PROMPT = "Sorry, I didn't catch that. Please try again."
_ERROR_PROMPT = "Sorry, something went wrong. Please try again."
_INACTIVITY_TIMEOUT_SECONDS = 30.0
_EXIT_PHRASES = frozenset(
    {"bye", "exit", "goodbye", "stop listening", "that's all", "that is all"}
)

# Set by the SIGINT handler; the main loop blocks on this event.
_shutdown_event = threading.Event()


def _handle_sigint(sig: int, frame: object) -> None:  # noqa: ARG001
    """Signal stop on Ctrl+C (SIGINT) so the main loop can clean up."""
    logger.info("Althea shutting down.")
    _shutdown_event.set()


def run() -> None:
    """Start the Althea main loop."""
    from althea.agent import AltheaAgent
    from althea.tools.browser import browser_stop
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
    inactivity_timer: threading.Timer | None = None

    def _transition(next_state: str, expected_state: str | None = None) -> bool:
        nonlocal state
        with state_lock:
            if expected_state is not None and state != expected_state:
                return False
            logger.info("State transition: %s -> %s", state, next_state)
            state = next_state
            return True

    def _on_utterance(audio: np.ndarray) -> None:
        """Called by VoiceActivityDetector with the captured Utterance audio."""
        nonlocal inactivity_timer
        if not _transition("transcribing", "listening"):
            return
        if inactivity_timer is not None:
            inactivity_timer.cancel()
            inactivity_timer = None
        try:
            text = transcriber.transcribe(audio)
        except Exception:
            logger.exception("Transcription failed.")
            _respond_then_listen(_RETRY_PROMPT)
            return
        if not text:
            logger.warning("Transcription returned empty text.")
            _respond_then_listen(_RETRY_PROMPT)
            return
        if text.casefold().strip(" .!?") in _EXIT_PHRASES:
            logger.info("Exit phrase detected.")
            agent.reset_session()
            _transition("idle")
            return
        logger.info("Command: %s", text)
        _transition("reasoning")
        try:
            response = asyncio.run(agent.run(text))
        except Exception:
            logger.exception("Agent failed to process command: %s", text)
            _respond_then_listen(_ERROR_PROMPT)
            return
        if response:
            logger.info("Althea: %s", response)
            _respond_then_listen(response)
            return
        _start_listening()

    vad = VoiceActivityDetector(on_utterance=_on_utterance)

    def _respond_then_listen(response: str) -> None:
        _transition("responding")
        voice_responder.speak(response, on_complete=_start_listening)

    def _on_inactivity_timeout() -> None:
        nonlocal inactivity_timer
        if not _transition("idle", "listening"):
            return
        inactivity_timer = None
        logger.info("Active conversation timed out.")
        vad.stop()
        agent.reset_session()

    def _start_listening() -> None:
        nonlocal inactivity_timer
        _transition("listening")
        inactivity_timer = threading.Timer(
            _INACTIVITY_TIMEOUT_SECONDS, _on_inactivity_timeout
        )
        inactivity_timer.daemon = True
        inactivity_timer.start()
        try:
            vad.start_capture()
        except Exception:
            logger.exception("Failed to start VAD capture.")
            inactivity_timer.cancel()
            inactivity_timer = None
            _transition("idle")

    def _on_wake_word() -> None:
        """Called by WakeWordDetector when the wake word is detected."""
        if not _transition("wake-word-detected", "idle"):
            return
        _respond_then_listen(_GREETING)

    detector = WakeWordDetector(on_wake_word=_on_wake_word)
    detector.start()

    try:
        _shutdown_event.wait()
    finally:
        if inactivity_timer is not None:
            inactivity_timer.cancel()
        vad.stop()
        detector.stop()
        asyncio.run(browser_stop())
        sys.exit(0)


def main() -> None:
    """Console-script entry point (installed via pyproject.toml)."""
    run()
