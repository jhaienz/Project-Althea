"""Root conftest.py — fixes import priority for the src/ layout.

The top-level ``althea.py`` entry-point script would shadow the
``src/althea`` package if the project root appears first in ``sys.path``.
This conftest ensures ``src/`` is first so ``import althea`` resolves to
the package, not the script.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ package layout takes precedence over the top-level althea.py.
_src = str(Path(__file__).parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
elif sys.path[0] != _src:
    sys.path.remove(_src)
    sys.path.insert(0, _src)
