"""Test-session defaults.

``glace_production.config`` builds its settings at import time and refuses to
import without ``GLACIERS_DATA_DIR``. The tests never touch the
directory it names, so a placeholder lets the modules import; a developer's
real ``.env`` still wins because ``setdefault`` leaves set variables alone.

The pixi environment lets PROJ fetch missing grids from its CDN
(``PROJ_NETWORK``); the suite is offline, so that is switched off before PROJ
first reads it.
"""

from __future__ import annotations

import os

os.environ.setdefault("GLACIERS_DATA_DIR", "/nonexistent/data_dir")
os.environ.pop("PROJ_NETWORK", None)
