import logging

from . import (
    g1_cfg,  # noqa: E402, F401
    g1_custom_cfg,  # noqa: E402, F401
    g1_loco_mimic_cfg,  # noqa: E402, F401
    g1_unified_loco_kick_cfg,  # noqa: E402, F401
)

try:  # table tennis needs the `pingpong` package (PINGPONG_ROOT); every other config works without it
    from . import g1_pingpong_cfg  # noqa: E402, F401
except FileNotFoundError as e:
    logging.getLogger(__name__).warning(f"g1_pingpong configs not registered: {e}")
