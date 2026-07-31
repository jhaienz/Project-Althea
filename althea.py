"""Althea entry point — starts the main loop.

Run with: python althea.py
"""

import sys
from pathlib import Path

# Allow running directly without installing the package.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from althea.main import run  # noqa: E402

if __name__ == "__main__":
    run()
