"""Althea main loop — starts the assistant and listens for the wake word."""

import asyncio
import logging
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
    from althea.transcription import Transcriber
    from althea.vad import VoiceActivityDetector
    from althea.wake_word import WakeWordDetector

    signal.signal(signal.SIGINT, _handle_sigint)
    logger.info("Althea is running")

    agent = AltheaAgent()
    transcriber = Transcriber()

    def _on_utterance(audio: np.ndarray) -> None:
        """Called by VoiceActivityDetector with the captured Utterance audio."""
        text = transcriber.transcribe(audio)
        if not text:
            logger.warning("Transcription returned empty text.")
            return
        logger.info("Command: %s", text)
        try:
            response = asyncio.run(agent.run(text))
        except Exception:
            logger.exception("Agent failed to process command: %s", text)
            return
        if response:
            logger.info("Althea: %s", response)

    vad = VoiceActivityDetector(on_utterance=_on_utterance)

    def _on_wake_word() -> None:
        """Called by WakeWordDetector when the wake word is detected."""
        logger.info("Wake word detected — starting VAD capture.")
        vad.start_capture()

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
