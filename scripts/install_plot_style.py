#!/usr/bin/env python3
"""Install the portable Aniket Nord Matplotlib style for the current user."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "dpsae" / "styles" / "aniket-nord.mplstyle"
TARGET = Path(mpl.get_configdir()) / "stylelib" / "aniket-nord.mplstyle"


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE, TARGET)
    print(f"Installed {TARGET}")


if __name__ == "__main__":
    main()
