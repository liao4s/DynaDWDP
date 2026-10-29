"""DynaDWDP: Dynamic Distributed Weight Data Parallelism.

Extends mainline DWDP (full-layer weight prefetch) with:
  - predictive minimal expert-set prefetch (lookahead head + statistical corrector)
  - dynamic per-token expert activation count with batch-set projection
  - forced / soft / high-precision operating modes
"""

from sglang.srt.layers.moe.dynadwdp.config import DynaDWDPConfig, DynaDWDPMode
from sglang.srt.layers.moe.dynadwdp.manager import (
    DynaDWDManager,
    get_dynadwdp_manager,
    router_only_config,
    set_dynadwdp_enabled,
    set_router_only,
)

__all__ = [
    "DynaDWDPConfig",
    "DynaDWDPMode",
    "DynaDWDManager",
    "get_dynadwdp_manager",
    "set_dynadwdp_enabled",
    "set_router_only",
    "router_only_config",
]
