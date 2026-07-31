"""Althea entry point — starts the main loop.

Run with: python althea.py
"""

import logging
import signal
import sys
from pathlib import Path

# Allow running directly without installing the package.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv  # noqa: E402

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


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_sigint)
    logger.info("Althea is running")

    # Main loop — subsequent issues will wire in the voice pipeline here.
    while True:
        signal.pause()
