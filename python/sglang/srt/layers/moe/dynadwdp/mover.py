"""Expert-granular prefetch on top of the mainline DWDP weight manager.

The mainline DWDPWeightManager.prefetch_layer copies ALL remote expert bytes
of a layer. DynaDWDP only needs the predicted minimal expert set:
  prefetch_layer_experts(layer, expert_ids) copies exactly those experts
  (weight-name tensors), in the given priority order, on the same copy stream
  with the same double-buffer event protocol.

ensure_expert_sync() implements the HIGH_PRECISION fallback path: a
synchronous single-expert pull on the current stream, used right before the
MoE kernel consumes it.
"""

from __future__ import annotations

import logging
import os
import time as _time
import json as _json
from typing import Iterable, List

import torch

from sglang.srt.layers.moe.dwdp.layout import lookup_owner
from sglang.srt.layers.moe.dwdp.weight_manager import DWDPWeightManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# cudaMemcpyBatchAsync (CUDA 12.8+): submit ALL per-expert copies in ONE
# runtime call. The previous per-expert copy_ loop cost ~130us of CPU per
# submission (96 x 130us = 12.5ms/layer), which serialized the DMA batch at
# the CPU submission rate and left a ~6.5ms wait_prefetch bubble every layer.
# ---------------------------------------------------------------------------
import ctypes as _ctypes

_batch_rt = None
try:
    _batch_rt = _ctypes.CDLL("libcudart.so")
except OSError:
    try:
        _batch_rt = _ctypes.CDLL("libcudart.so.13")
    except OSError:
        _batch_rt = None

# raw cudaMemcpyAsync via ctypes: identical call to what Tensor.copy_(...,
# non_blocking=True) issues, minus ~130us/call of torch dispatch+slice
# overhead. 96 per-expert copies submitted in ~0.3ms instead of ~12.5ms,
# which used to stretch the DMA batch to the wait_prefetch deadline and
# stall the compute stream ~6.5ms every layer.
_rt = _batch_rt
_HAS_RAW_MEMCPY = _rt is not None and hasattr(_rt, "cudaMemcpyAsync")
if _HAS_RAW_MEMCPY:
    _rt.cudaMemcpyAsync.restype = _ctypes.c_int
    _rt.cudaMemcpyAsync.argtypes = [
        _ctypes.c_void_p,   # dst
        _ctypes.c_void_p,   # src
        _ctypes.c_size_t,   # size (bytes)
        _ctypes.c_int,      # kind (cudaMemcpyDeviceToDevice = 3)
        _ctypes.c_void_p,   # stream
    ]
_MEMCPY_D2D = 3

# --- Experiment-3 measurement: bare DMA time of each prefetch op ---
# When DYNADWDP_MEASURE=1, every prefetch_layer_experts call synchronizes the
# copy stream after issuing and records the bare (unhidden) DMA time + bytes.
_MEASURE = os.environ.get("DYNADWDP_MEASURE", "0") == "1"
_PREFETCH_STATS = {"us": 0.0, "calls": 0, "bytes": 0}
_PER_LAYER = []

_HAS_BATCH_MEMCPY = (
    _batch_rt is not None
    and hasattr(_batch_rt, "cudaMemcpyBatchAsync")
    and __import__("os").environ.get("DYNADWDP_BATCH_MEMCPY", "0") == "1"
)
# NOTE: cudaMemcpyBatchAsync was evaluated and DISABLED by default — in this
# VMM + NVLink setup it shows intermittent catastrophic slowdowns (e.g. 78s
# TTFT spikes, tps 400) and non-reproducible timings, suspect driver-level
# rescheduling of cross-peer copies interacting with the event protocol.
# The plain per-expert copy_ loop on the copy engine is stable and, with the
# async prediction pipeline, already hides DMA behind compute.
_attrs_device = 0  # current device id, resolved lazily


def _resolve_attrs_device() -> None:
    global _attrs_device
    _attrs_device = torch.cuda.current_device()
if _HAS_BATCH_MEMCPY:
    _batch_rt.cudaMemcpyBatchAsync.restype = _ctypes.c_int
    # CUDA 13 signature: (dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, stream)
    _batch_rt.cudaMemcpyBatchAsync.argtypes = [
        _ctypes.POINTER(_ctypes.c_void_p),   # dsts
        _ctypes.POINTER(_ctypes.c_void_p),   # srcs
        _ctypes.POINTER(_ctypes.c_size_t),   # sizes
        _ctypes.c_size_t,                    # count
        _ctypes.c_void_p,                    # attrs
        _ctypes.POINTER(_ctypes.c_size_t),   # attrsIdxs
        _ctypes.c_size_t,                    # numAttrs
        _ctypes.c_void_p,                    # stream
    ]

    # struct cudaMemcpyAttributes { srcAccessOrder(enum4), srcLocHint(8),
    # dstLocHint(8), flags(uint4) } — stream-ordered default attributes
    class _MemcpyAttrs(_ctypes.Structure):
        _fields_ = [
            ("srcAccessOrder", _ctypes.c_int),
            ("srcLocHintType", _ctypes.c_int),
            ("srcLocHintId", _ctypes.c_int),
            ("dstLocHintType", _ctypes.c_int),
            ("dstLocHintId", _ctypes.c_int),
            ("flags", _ctypes.c_uint),
        ]

    _DEFAULT_ATTRS = _MemcpyAttrs(
        0x1,  # cudaMemcpySrcAccessOrderStream
        0x1, 0,  # cudaMemLocationTypeDevice, current device id patched at call
        0x1, 0,
        0,
    )
    _DEFAULT_ATTRS_IDXS = (_ctypes.c_size_t * 1)(0)


def _cuda_memcpy_batch_async(dsts, srcs, sizes, stream_handle: int) -> int:
    """One-call submission of N device-to-device copies on the given stream.

    Returns the CUDA error code (0 = success). All tensors must be contiguous
    device memory; the caller holds the stream context.
    """
    n = len(dsts)
    a_d = (_ctypes.c_void_p * n)(*dsts)
    a_s = (_ctypes.c_void_p * n)(*srcs)
    a_z = (_ctypes.c_size_t * n)(*sizes)
    attrs = _MemcpyAttrs.from_buffer_copy(_DEFAULT_ATTRS)
    attrs.dstLocHintId = _attrs_device
    attrs.srcLocHintId = _attrs_device
    return _batch_rt.cudaMemcpyBatchAsync(
        a_d, a_s, a_z, n,
        _ctypes.byref(attrs), _DEFAULT_ATTRS_IDXS, 1,
        _ctypes.c_void_p(stream_handle),
    )

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

if _HAS_TRITON:

    @triton.jit
    def _gather_copy_kernel(
        ids_ptr,                  # [S] int32 global expert ids
        dst_ptr,                  # full buffer, int32 view, [E, R]
        src0_ptr, src1_ptr, src2_ptr, src3_ptr,
        b0, b1, b2, b3, b4,       # peer boundaries (peer_start, ascending; b4 = E)
        R,                        # int32 units per expert
        S,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        nb = tl.cdiv(R, BLOCK)
        s = pid // nb
        blk = pid % nb
        if s < S:
            e = tl.load(ids_ptr + s).to(tl.int64)
            offs = blk * BLOCK + tl.arange(0, BLOCK)
            m = offs < R
            src_off = (e - 0) * 0  # placeholder to establish ptr type
            if e < b1:
                base = src0_ptr + (e - b0) * R
            elif e < b2:
                base = src1_ptr + (e - b1) * R
            elif e < b3:
                base = src2_ptr + (e - b2) * R
            else:
                base = src3_ptr + (e - b3) * R
            v = tl.load(base + offs, mask=m)
            tl.store(dst_ptr + e * R + offs, v, mask=m)


def _fused_gather_copy(full: torch.Tensor, peer_views: List[torch.Tensor],
                       peer_starts: List[int], ids: torch.Tensor) -> None:
    """full[e] <- peer_views[owner(e)][e - peer_start[owner(e)]] for e in ids.

    One kernel launch per call (vs one copy_ per contiguous run). All tensors
    are viewed as int32 for a vectorized elementwise copy.
    """
    E, R_elem = full.shape[0], full[0].numel()
    assert R_elem % 4 == 0
    dst = full.view(torch.int32)
    R = R_elem // 4
    S = ids.numel()
    npeer = len(peer_views)
    ptrs = []
    for r in range(4):
        ptrs.append(peer_views[min(r, npeer - 1)].view(torch.int32) if npeer else dst)
    bounds = list(peer_starts) + [E] * (4 - len(peer_starts))
    bnd = (bounds + [E, E, E, E])[:5]
    nb = triton.cdiv(R, 4096)
    _gather_copy_kernel[(S * nb,)](
        ids, dst,
        ptrs[0], ptrs[1], ptrs[2], ptrs[3],
        bnd[0], bnd[1], bnd[2], bnd[3], bnd[4],
        R, S,
        BLOCK=4096,
        num_warps=8,
    )


class SubsetPrefetcher(DWDPWeightManager):
    def prefetch_layer_experts(
        self, layer_idx: int, expert_ids: Iterable[int]
    ) -> int:
        """Copy only the given (remote) experts for this layer.

        Returns bytes issued. Uses the standard double-buffer event protocol:
        WAR on consume event, record prefetch event at the end.
        """
        buf_idx = self._weight_buffer.buffer_index_for_layer(layer_idx)
        cs = self._copy_streams[buf_idx % len(self._copy_streams)]
        local_start = self._weight_buffer.local_start
        local_end = self._weight_buffer.local_end
        nbytes = 0

        # contention-experiment knob: skip the actual DMA entirely (events
        # still fire so the protocol stays consistent); compute reads stale
        # buffer contents but memory traffic is unchanged, so kernel timings
        # remain valid
        _t0 = _time.perf_counter() if _MEASURE else 0.0
        if os.environ.get("DYNADWDP_SKIP_PREFETCH", "0") == "1":
            with torch.cuda.stream(cs):
                cs.wait_event(self._consume_events[buf_idx])
                self._prefetch_events[buf_idx].record(cs)
            return 0

        with torch.cuda.stream(cs):
            cs.wait_event(self._consume_events[buf_idx])
            todo = sorted(
                e for e in expert_ids if not (local_start <= e < local_end)
            )
            if todo and _HAS_BATCH_MEMCPY and os.environ.get(
                "DYNADWDP_BATCH_MEMCPY", "0"
            ) == "1":
                # one-call DMA batch submission: per-expert copies with
                # pointer arithmetic done host-side (no sync).
                # NOTE: unit-tested OK on regular allocations, but triggers a
                # hard crash (SIGSEGV, no python traceback) on the VMM-mapped
                # peer windows under sustained multi-forward load — keep OFF
                # until the access-property interaction is root-caused.
                _resolve_attrs_device()
                dsts, srcs, sizes = [], [], []
                for name in self._weight_names:
                    full = self._weight_buffer.get_full_tensor(layer_idx, name)
                    peb = full[0].numel() * full.element_size()
                    fptr = full.data_ptr()
                    for e in todo:
                        pr = lookup_owner(e, self._peer_ranges)
                        ps, _ = self._peer_ranges[pr]
                        sv = self._peer_views[(pr, layer_idx, name)]
                        dsts.append(fptr + e * peb)
                        srcs.append(sv.data_ptr() + (e - ps) * peb)
                        sizes.append(peb)
                    nbytes += len(todo) * peb
                rc = _cuda_memcpy_batch_async(
                    dsts, srcs, sizes, cs.cuda_stream
                )
                if rc != 0:
                    raise RuntimeError(
                        f"cudaMemcpyBatchAsync failed rc={rc}"
                    )
            elif todo:
                # Copy contiguous runs of experts instead of one expert
                # at a time: per-expert copy_ launches cost ~10-30us each
                # and dominate the forward when the set is large.
                runs: List[List[int]] = []
                for e in todo:
                    if runs and e == runs[-1][-1] + 1:
                        runs[-1].append(e)
                    else:
                        runs.append([e])
                for name in self._weight_names:
                    full = self._weight_buffer.get_full_tensor(layer_idx, name)
                    peb = full[0].numel() * full.element_size()
                    if _HAS_RAW_MEMCPY:
                        # raw cudaMemcpyAsync per (merged, peer-clipped) run:
                        # same DMA the copy_ loop issues, ~40x less CPU
                        fbase = full.data_ptr()
                        sh = cs.cuda_stream
                        for run in runs:
                            s0, s1 = run[0], run[-1] + 1
                            cursor = s0
                            while cursor < s1:
                                peer_rank = lookup_owner(cursor, self._peer_ranges)
                                peer_start, peer_end = self._peer_ranges[peer_rank]
                                seg_end = min(s1, peer_end)
                                n = seg_end - cursor
                                src = self._peer_views[(peer_rank, layer_idx, name)]
                                rc = _rt.cudaMemcpyAsync(
                                    fbase + cursor * peb,
                                    src.data_ptr() + (cursor - peer_start) * peb,
                                    n * peb,
                                    _MEMCPY_D2D,
                                    sh,
                                )
                                if rc != 0:
                                    raise RuntimeError(
                                        f"cudaMemcpyAsync failed rc={rc}"
                                    )
                                nbytes += n * peb
                                cursor = seg_end
                    else:
                        for run in runs:
                            s0, s1 = run[0], run[-1] + 1
                            cursor = s0
                            while cursor < s1:
                                peer_rank = lookup_owner(cursor, self._peer_ranges)
                                peer_start, peer_end = self._peer_ranges[peer_rank]
                                seg_end = min(s1, peer_end)
                                n = seg_end - cursor
                                src = self._peer_views[(peer_rank, layer_idx, name)]
                                full[cursor:seg_end].copy_(
                                    src[cursor - peer_start:seg_end - peer_start],
                                    non_blocking=True,
                                )
                                nbytes += n * peb
                                cursor = seg_end
            self._prefetch_events[buf_idx].record(cs)
        if _MEASURE:
            cs.synchronize()
            _PREFETCH_STATS["us"] += (_time.perf_counter() - _t0) * 1e6
            _PREFETCH_STATS["calls"] += 1
            _PREFETCH_STATS["bytes"] += nbytes
            try:
                with open(f"/tmp/exp6_pf_mv_pid{_os.getpid()}.json", "w") as _f:
                    json.dump(_PREFETCH_STATS, _f)
            except Exception:
                pass
        return nbytes

    def ensure_expert_sync(self, layer_idx: int, expert_id: int) -> None:
        """Synchronous single-expert pull on the CURRENT stream (fallback)."""
        local_start = self._weight_buffer.local_start
        local_end = self._weight_buffer.local_end
        if local_start <= expert_id < local_end:
            return
        # must also respect WAR on the buffer slot
        buf_idx = self._weight_buffer.buffer_index_for_layer(layer_idx)
        device = torch.device("cuda", self._weight_buffer.device_id)
        torch.cuda.current_stream(device).wait_event(self._prefetch_events[buf_idx])

        for name in self._weight_names:
            full = self._weight_buffer.get_full_tensor(layer_idx, name)
            peer_rank = lookup_owner(expert_id, self._peer_ranges)
            peer_start, _ = self._peer_ranges[peer_rank]
            key = (peer_rank, layer_idx, name)
            src = self._peer_views[key]
            full[expert_id].copy_(src[expert_id - peer_start])

    def prefetch_layer_experts_dev(self, layer_idx: int, ids_dev, wait_ev=None) -> int:
        """P2: device-side prefetch -- no host ids, no D2H, no tolist().

        Copies the given (remote) experts straight from the peer views with the
        fused gather-copy kernel, on this layer's copy stream. Because nothing
        has to reach the host, the DMA can be issued the moment the prediction
        kernels are enqueued.
        """
        buf_idx = self._weight_buffer.buffer_index_for_layer(layer_idx)
        cs = self._copy_streams[buf_idx % len(self._copy_streams)]
        local_start = self._weight_buffer.local_start
        local_end = self._weight_buffer.local_end
        keep = (ids_dev < local_start) | (ids_dev >= local_end)
        ids = ids_dev[keep].to(torch.int32)
        if ids.numel() == 0:
            with torch.cuda.stream(cs):
                cs.wait_event(self._consume_events[buf_idx])
                self._prefetch_events[buf_idx].record(cs)
            return 0
        nbytes = 0
        peers = [r for r, _ in self._peer_ranges]
        starts = [self._peer_ranges[r][0] for r in peers]
        with torch.cuda.stream(cs):
            cs.wait_event(self._consume_events[buf_idx])
            if wait_ev is not None:
                cs.wait_event(wait_ev)  # order after the prediction kernels
            for name in self._weight_names:
                full = self._weight_buffer.get_full_tensor(layer_idx, name)
                views = [self._peer_views[(r, layer_idx, name)] for r in peers]
                _fused_gather_copy(full, views, starts, ids)
                nbytes += ids.numel() * full[0].numel() * full.element_size()
            self._prefetch_events[buf_idx].record(cs)
        return nbytes


    def prefetch_layer(self, layer_idx: int) -> None:
        """Mainline full-layer prefetch, instrumented when DYNADWDP_MEASURE=1."""
        if not _MEASURE:
            return super().prefetch_layer(layer_idx)
        _t0 = _time.perf_counter()
        super().prefetch_layer(layer_idx)
        self._copy_streams[self._weight_buffer.buffer_index_for_layer(layer_idx) % len(self._copy_streams)].synchronize()
        _dt = (_time.perf_counter() - _t0) * 1e6
        _PREFETCH_STATS["us"] += _dt
        _PREFETCH_STATS["calls"] += 1
        _PER_LAYER.append({"layer": layer_idx, "us": round(_dt, 1)})
        try:
            with open(f"/tmp/exp6_pf_mv_pid{_os.getpid()}.json", "w") as _f:
                _json.dump({"stats": _PREFETCH_STATS, "per_layer": _PER_LAYER[-200:]}, _f)
        except Exception:
            pass

    def prefetch_layer_full(self, layer_idx: int) -> None:
        """Full-layer prefetch (mainline behavior), used for the first layers."""
        self.prefetch_layer(layer_idx)
