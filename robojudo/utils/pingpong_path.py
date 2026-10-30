"""Where the `pingpong` package (the table-tennis planner, estimator, policy support: playground/humanoid_pingpong/pingpong-version-0)
lives, and putting it on sys.path. Set PINGPONG_ROOT on a computer where it is somewhere else (the robot's control PC)."""

import os
import sys
from pathlib import Path

from robojudo.config.global_path import ROOT_DIR

_DEFAULT = ROOT_DIR.parent.parent / "playground" / "humanoid_pingpong" / "pingpong-version-0"


def pingpong_root() -> Path:
    root = Path(os.environ.get("PINGPONG_ROOT", _DEFAULT)).expanduser()
    if not (root / "pingpong" / "__init__.py").exists():
        raise FileNotFoundError(
            f"the pingpong package is not at {root} (set PINGPONG_ROOT to the folder that holds pingpong/): "
            "the table-tennis configs need its planner, estimator and swing library"
        )
    return root


def ensure_pingpong() -> Path:
    root = pingpong_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
