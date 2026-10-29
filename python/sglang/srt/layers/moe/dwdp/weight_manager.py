# Adapted from NVIDIA TensorRT-LLM (https://github.com/NVIDIA/TensorRT-LLM)
"""Double-buffered async prefetch of peer expert weights into the composite VA."""

from __future__ import annotations

import bisect
import json as _json
import logging
import os as _os
import time as _time
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.moe.dwdp.layout import PeerRanges, lookup_owner
from sglang.srt.layers.moe.dwdp.weight_buffer import WeightBuffer

logger = logging.getLogger(__name__)

# --- Experiment measurement: bare (unhidden) DMA time + bytes of full-layer
# prefetch, enabled with DYNADWDP_MEASURE=1. Same JSON contract as the
# dynadwdp mover so one driver can read both arms.
_MEASURE = _os.environ.get("DYNADWDP_MEASURE", "0") == "1"
_MEAS_STATS = {"us": 0.0, "calls": 0, "bytes": 0}


class DWDPWeightManager:
    def __init__(
        self,
        weight_buffer: WeightBuffer,
        peer_views: Dict[Tuple[int, int, str], torch.Tensor],
        peer_ranges: PeerRanges,
        moe_layer_indices: List[int],
        weight_names: List[str],
        dwdp_rank: int,
        dwdp_size: int,
        transport=None,
    ) -> None:
        self._weight_buffer = weight_buffer
        self._peer_views = peer_views
        self._peer_ranges = peer_ranges
        self._moe_layer_indices = sorted(moe_layer_indices)
        self._moe_layer_set = set(self._moe_layer_indices)
        self._weight_names = list(weight_names)
        self._dwdp_rank = dwdp_rank
        self._dwdp_size = dwdp_size
        # transport handles underpin the VA mappings; must outlive this manager
        self._transport = transport

        device = torch.device("cuda", weight_buffer.device_id)
        # One copy stream per staging slot: with N slots the prefetcher can have
        # up to N layer DMAs in flight, so the copy engine never idles waiting
        # for the previous layer's transfer to retire.
        n_slots = max(2, len(getattr(weight_buffer, "_slot_count", []) or []) or 0)
        try:
            from sglang.srt.layers.moe.dwdp.weight_buffer import NUM_SLOTS as _NS
            n_slots = _NS
        except Exception:
            n_slots = 2
        self._num_slots = n_slots
        self._copy_streams: List[torch.cuda.Stream] = [
            torch.cuda.Stream(device=device) for _ in range(n_slots)
        ]
        self._copy_stream = self._copy_streams[0]

        self._prefetch_events: List[torch.cuda.Event] = [
            torch.cuda.Event() for _ in range(n_slots)
        ]
        self._consume_events: List[torch.cuda.Event] = [
            torch.cuda.Event() for _ in range(n_slots)
        ]

    def stream_for(self, layer_idx: int) -> "torch.cuda.Stream":
        return self._copy_streams[
            self._weight_buffer.buffer_index_for_layer(layer_idx) % len(self._copy_streams)
        ]

        # pre-record consume events so the first prefetch doesn't stall
        current = torch.cuda.current_stream(device)
        for ev in self._consume_events:
            ev.record(current)

        logger.debug(
            f"WeightManager rank={dwdp_rank}/{dwdp_size}, "
            f"{len(moe_layer_indices)} MoE layers, weights={weight_names}"
        )

    @property
    def weight_buffer(self) -> WeightBuffer:
        return self._weight_buffer

    def is_moe_layer(self, layer_idx: int) -> bool:
        return layer_idx in self._moe_layer_set

    def next_moe_layer(self, layer_idx: int) -> Optional[int]:
        pos = bisect.bisect_right(self._moe_layer_indices, layer_idx)
        if pos < len(self._moe_layer_indices):
            return self._moe_layer_indices[pos]
        return None

    def first_moe_layer(self) -> int:
        return self._moe_layer_indices[0]

    def prefetch_layer(self, layer_idx: int) -> None:
        buf_idx = self._weight_buffer.buffer_index_for_layer(layer_idx)
        cs = self._copy_streams[buf_idx % len(self._copy_streams)]
        _t0 = _time.perf_counter() if _MEASURE else 0.0

        with torch.cuda.stream(cs):
            # WAR: wait for compute to finish reading this slot before overwriting
            cs.wait_event(self._consume_events[buf_idx])

            self._prefetch_layer_per_slice(layer_idx)

            self._prefetch_events[buf_idx].record(cs)

        if _MEASURE:
            cs.synchronize()
            _MEAS_STATS["us"] += (_time.perf_counter() - _t0) * 1e6
            _MEAS_STATS["calls"] += 1
            try:
                with open(f"/tmp/exp6_pf_wm_pid{_os.getpid()}.json", "w") as _f:
                    _json.dump(_MEAS_STATS, _f)
            except Exception:
                pass

    def _prefetch_layer_per_slice(self, layer_idx: int) -> None:
        for name in self._weight_names:
            remote_slices = self._weight_buffer.get_remote_slices(layer_idx, name)
            for dst_slice, expert_start, expert_end in remote_slices:
                cursor = expert_start
                dst_offset = 0
                while cursor < expert_end:
                    peer_rank = lookup_owner(cursor, self._peer_ranges)
                    peer_start, peer_end = self._peer_ranges[peer_rank]
                    local_offset = cursor - peer_start
                    chunk_end = min(expert_end, peer_end)
                    n = chunk_end - cursor

                    peer_key = (peer_rank, layer_idx, name)
                    src = self._peer_views[peer_key]
                    dst_slice[dst_offset : dst_offset + n].copy_(
                        src[local_offset : local_offset + n]
                    )
                    if _MEASURE:
                        _MEAS_STATS["bytes"] += (
                            n * dst_slice[0].numel() * dst_slice.element_size()
                        )
                    dst_offset += n
                    cursor = chunk_end

    def wait_prefetch(self, layer_idx: int) -> None:
        buf_idx = self._weight_buffer.buffer_index_for_layer(layer_idx)
        device = torch.device("cuda", self._weight_buffer.device_id)
        compute_stream = torch.cuda.current_stream(device)
        compute_stream.wait_event(self._prefetch_events[buf_idx])

    def record_compute_and_prefetch_next(self, layer_idx: int) -> None:
        buf_idx = self._weight_buffer.buffer_index_for_layer(layer_idx)
        device = torch.device("cuda", self._weight_buffer.device_id)
        compute_stream = torch.cuda.current_stream(device)

        self._consume_events[buf_idx].record(compute_stream)

        # prefetch the layer 2 ahead — it reuses the same buffer slot
        next_layer = self.next_moe_layer(layer_idx)
        if next_layer is not None:
            next_next = self.next_moe_layer(next_layer)
            if next_next is not None:
                self.prefetch_layer(next_next)

    def prefetch_first_layers(self) -> None:
        if len(self._moe_layer_indices) >= 1:
            self.prefetch_layer(self._moe_layer_indices[0])
        if len(self._moe_layer_indices) >= 2:
            self.prefetch_layer(self._moe_layer_indices[1])

    def release(self) -> None:
        if self._weight_buffer is not None:
            self._weight_buffer.release()
            self._weight_buffer = None
        if self._transport is not None:
            self._transport.release()
            self._transport = None
        self._peer_views.clear()
