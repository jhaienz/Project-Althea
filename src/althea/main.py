"""Althea main loop — starts the assistant and listens for the wake word."""

import logging
import signal
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def _handle_sigint(sig: int, frame: object) -> None:  # noqa: ARG001
    """Exit cleanly on Ctrl+C (SIGINT)."""
    logger.info("Althea shutting down.")
    sys.exit(0)


def _on_wake_word() -> None:
    """Called by WakeWordDetector when the wake word is detected."""
    logger.info("Wake word detected!")
    # Subsequent issues will wire VAD + transcription + Agent here.


def run() -> None:
    """Start the Althea main loop."""
    from althea.wake_word import WakeWordDetector

    signal.signal(signal.SIGINT, _handle_sigint)
    logger.info("Althea is running")

    detector = WakeWordDetector(on_wake_word=_on_wake_word)
    detector.start()

    # Block the main thread; the detector runs in a daemon thread.
    try:
        while True:
            signal.pause()
    finally:
        detector.stop()


def main() -> None:
    """Console-script entry point (installed via pyproject.toml)."""
    run()

