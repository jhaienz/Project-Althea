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


def run() -> None:
    """Start the Althea main loop."""
    signal.signal(signal.SIGINT, _handle_sigint)
    logger.info("Althea is running")

    # Main loop — subsequent issues will wire in the voice pipeline here.
    while True:
        signal.pause()


def main() -> None:
    """Console-script entry point (installed via pyproject.toml)."""
    run()
