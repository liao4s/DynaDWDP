"""Dynamic expert router: per-token variable activation count + set projection.

Replaces fixed top-k with:
  1. cumulative-mass criterion:  k_t = min{ m : sum_{j<=m} p_j >= 1 - epsilon }
  2. gap criterion:              cut at first significant drop p_m - p_{m+1}
  3. clamp to [k_min, k_max]
  4. set projection onto the ready expert set (local + prefetched):
     implemented as masked selection — probabilities of not-ready experts are
     zeroed and the mass renormalized over ready experts, so each token is
     served by its best ready experts.

Output is padded to k_max slots with weight-0 dummy entries so downstream
grouped-GEMM kernels keep their fixed-k interface.

PERFORMANCE CONTRACT: this kernel runs once per MoE layer per forward, so it
must not trigger any host-device synchronization. All telemetry is written
into a GPU-side accumulator tensor and only read at stats-dump time.
"""

from __future__ import annotations

from typing import Tuple

import torch


class DynSelectTelemetry:
    """GPU-side accumulators; synced only when dumped."""

    def __init__(self, device):
        self.count = torch.zeros(1, dtype=torch.float64, device=device)
        self.avg_k_sum = torch.zeros(1, dtype=torch.float32, device=device)
        self.replaced_sum = torch.zeros(1, dtype=torch.float32, device=device)
        self.us_sum = torch.zeros(1, dtype=torch.float64, device=device)
        # lifetime accumulators: never zeroed by the periodic dumper, so a
        # post-hoc reader sees the whole server lifetime, not the last window
        self.lt_count = torch.zeros(1, dtype=torch.float64, device=device)
        self.lt_avg_k_sum = torch.zeros(1, dtype=torch.float32, device=device)


# dyn_select emits -1 (weight-0) for empty slots. The MoE runners drop these
# slots, shrinking grouped-GEMM work with the real assignment count. The flag
# is flipped on by the DynaDWDP manager at setup; mainline topk (never -1) is
# unaffected.
_DROP_SENTINEL_ENABLED = False


def set_drop_sentinel_enabled(v: bool) -> None:
    global _DROP_SENTINEL_ENABLED
    _DROP_SENTINEL_ENABLED = v


def drop_sentinel_enabled() -> bool:
    return _DROP_SENTINEL_ENABLED


_TELEMETRY = {}


def _tele_key(device) -> str:
    d = torch.device(device)
    idx = d.index if d.index is not None else torch.cuda.current_device()
    return f"cuda:{idx}"


def _tele(device):
    key = _tele_key(device)
    if key not in _TELEMETRY:
        _TELEMETRY[key] = DynSelectTelemetry(torch.device(key))
    return _TELEMETRY[key]


_KT_DUMP = __import__("os").environ.get("DYNADWDP_KT_DUMP", "0") == "1"
_FIXED_WIDTH = __import__("os").environ.get("DYNADWDP_FIXED_WIDTH", "0") == "1"
_KT_STATE = {"count": 0, "sum_kt": 0.0, "lt_count": 0, "lt_sum_kt": 0.0}

_KT_ACC = {"sum": None, "count": 0, "eps": None, "kmax": None}


def _kt_acc(kt, eps=None, kmax=None):
    """Accumulate mean k on device (no host sync)."""
    try:
        _KT_ACC["eps"] = eps
        _KT_ACC["kmax"] = kmax
        v = kt.float().mean()
        _KT_ACC["sum"] = v.clone() if _KT_ACC["sum"] is None else _KT_ACC["sum"] + v
        _KT_ACC["count"] += 1
    except Exception:
        pass


def kt_snapshot():
    """Materialise the accumulator (single .item()); call from the stats loop."""
    if not _KT_ACC["count"] or _KT_ACC["sum"] is None:
        return None
    return {
        "avg_kt": round(float(_KT_ACC["sum"] / _KT_ACC["count"]), 4),
        "calls": _KT_ACC["count"],
        "eps": _KT_ACC["eps"],
        "k_max": _KT_ACC["kmax"],
    }


def _kt_dump(kt_mean: float, w_mean_slots=None, eps=None, kmax=None, extra=None):
    import os as _os
    import json as _json
    _KT_STATE["count"] += 1
    _KT_STATE["sum_kt"] += kt_mean
    _KT_STATE["lt_count"] += 1
    _KT_STATE["lt_sum_kt"] += kt_mean
    try:
        d = {"pid": _os.getpid(), "count": _KT_STATE["count"],
             "avg_kt": round(_KT_STATE["sum_kt"] / _KT_STATE["count"], 4),
             "eps": eps, "k_max": kmax, **(extra or {})}
        if w_mean_slots is not None:
            d["w_mean_slot"] = [round(float(x), 5) for x in w_mean_slots]
        with open(f"/tmp/dynadwdp_kt_dump_pid{_os.getpid()}.json", "w") as _f:
            _json.dump(d, _f)
    except Exception as _e:
        import sys as _sys
        print(f"[KT-DUMP-ERR] {type(_e).__name__}: {_e}", flush=True, file=_sys.stderr)

@torch.no_grad()
def dyn_select(
    router_logits: torch.Tensor,
    ready_mask: torch.Tensor,
    k_min: int = 2,
    k_max: int = 8,
    epsilon: float = 0.15,
    tau: float = 0.5,
    renormalize: bool = True,
    dummy_expert_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Select variable-count experts per token from the ready set.

    All tensor math stays on-device; no .item()/.tolist()/bool() syncs.
    `info` contains GPU tensors — call .item() only in the stats dumper.
    """
    import time

    t0 = time.perf_counter()
    T, E = router_logits.shape
    device = router_logits.device

    if _HAS_TRITON and router_logits.is_cuda:
        probs_f32 = router_logits.to(torch.float32)
        if _KT_DUMP and _KT_STATE["count"] < 3:
            import sys as _sys
            print(f"[KT-DEBUG] dyn_select TRITON entered T={router_logits.shape[0]} eps={epsilon} k={k_min}-{k_max}", flush=True, file=_sys.stderr)
        out_w, out_id, out_kt = _dyn_select_fused(
            probs_f32, ready_mask, k_min, k_max, epsilon, tau,
            fixed_width=_FIXED_WIDTH, dummy_expert_id=dummy_expert_id,
        )
        tele = _tele(device)
        tele.count += 1
        tele.avg_k_sum += out_kt.float().mean()
        tele.us_sum += (time.perf_counter() - t0) * 1e6
        tele.lt_count += out_kt.numel()
        # NOTE: use out_kt (computed above) — referencing `info` here was an
        # UnboundLocalError (info is only assigned below), which silently
        # forced every caller onto the torch fallback path via the crash.
        tele.lt_avg_k_sum += out_kt.float().sum()
        info = {"avg_k_t": out_kt, "replaced_t": tele.replaced_sum.squeeze(0)}
        # NOTE: .item() here would force a full device sync every layer; keep it
        # behind the debug flag (avg_k is already accumulated on-GPU in tele).
        if _KT_DUMP:
            # accumulate on-device: float(...) here would force a D2H + sync
            # per layer; the stats loop materialises it every few seconds.
            _kt_acc(out_kt, eps=epsilon, kmax=k_max)
        return out_w, out_id, info

    if _KT_DUMP and _KT_STATE["count"] < 3:
        import sys as _sys
        print(f"[KT-DEBUG] dyn_select fallback entered T={router_logits.shape[0]} E={router_logits.shape[1]} eps={epsilon} k={k_min}-{k_max} triton={_HAS_TRITON}", flush=True, file=_sys.stderr)
    probs = torch.softmax(router_logits.to(torch.float32), dim=-1)  # [T, E]

    # --- set projection: mask to ready experts, renormalize mass ---
    masked = probs * ready_mask.to(torch.float32).unsqueeze(0)
    mass = masked.sum(dim=-1, keepdim=True)
    masked = masked / mass.clamp_min(1e-12)

    # tokens whose original top-1 expert is not ready are counted as replaced
    orig_top1 = probs.argmax(dim=-1)
    replaced = (~ready_mask[orig_top1]).to(torch.float64).sum()

    topv, topi = torch.topk(masked, k_max, dim=-1)  # [T, K] desc over ready

    # --- per-token k_t on the renormalized distribution ---
    cum = torch.cumsum(topv, dim=-1)
    need = (cum < (1.0 - epsilon)).sum(dim=-1) + 1
    if k_max > 1:
        gaps = topv[:, :-1] - topv[:, 1:]
        # tau<=0 semantics: DISABLE the gap criterion (a zero threshold would
        # otherwise flag every ranking step as a "cliff" and pin k to k_min)
        drop_idx = (gaps >= (tau * topv[:, :-1].clamp_min(1e-12))) & (tau > 0)
        ar = torch.arange(1, k_max, device=device).expand(T, -1)
        gap_k = torch.where(drop_idx, ar, torch.full_like(ar, k_max)).min(dim=-1).values
    else:
        gap_k = torch.full((T,), k_max, device=device, dtype=need.dtype)
    k_t = torch.minimum(need, gap_k).clamp(min=k_min, max=k_max)  # [T]

    arK = torch.arange(k_max, device=device).expand(T, -1)
    keep = (arK < k_t.unsqueeze(-1)) & (topv > 0)

    weights = torch.where(keep, topv, torch.zeros_like(topv))
    if renormalize:
        wsum = weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = weights / wsum

    if _FIXED_WIDTH:
        # Fixed-width mode: every slot carries a REAL expert id; slots beyond
        # k_t keep weight 0. Mathematically identical to the pruned form
        # (0 * expert = 0) but dispatch never sees invalid ids, which certain
        # MoE runners mishandle (corrupted outputs observed on Qwen3 fp8).
        ids = topi
    else:
        # empty slots -> -1 drop sentinel (weight 0). The MoE runners drop them:
        #  - triton: moe_align_block_size(ignore_invalid_expert=True) excludes -1
        #    from the sorted layout, so GEMM blocks shrink with the real work;
        #  - deep_gemm: fused_moe_dispatch_index treats expert<0 as invalid
        #    (src2dst=-1, not counted in tokens_per_expert).
        ids = torch.where(keep, topi, torch.full_like(topi, dummy_expert_id))

    tele = _tele(device)
    tele.count += 1
    tele.avg_k_sum += k_t.to(torch.float64).mean()
    tele.replaced_sum += replaced
    tele.us_sum += (time.perf_counter() - t0) * 1e6

    info = {
        "avg_k_t": k_t,            # GPU tensor [T]
        "replaced_t": replaced,    # GPU tensor scalar
    }
    if _KT_DUMP:
        _wmean = weights[:, :8].float().mean(0) if weights.shape[1] >= 1 else None
        _kt_dump(float(k_t.to(torch.float64).mean()), _wmean, eps=epsilon, kmax=k_max,
                 extra={"need_mean": round(float(need.to(torch.float64).mean()), 3),
                        "gapk_mean": round(float(gap_k.to(torch.float64).mean()), 3),
                        "probs_top8_full": [round(float(x), 5) for x in
                                            torch.softmax(router_logits.to(torch.float32), -1)[0].topk(8).values]})
    # mainline TopK contract: topk_weights must be fp32 (FlashInfer grouped
    # GEMM checks token_final_scales dtype == float32)
    return weights.to(torch.float32), ids.to(torch.int32), info


# ---------------- fused triton implementation ----------------
# One program per token: softmax -> ready-mask renorm -> serial dual-criteria
# selection (K argmax rounds over E) -> weights/ids/k_t. Replaces ~25 torch
# elementwise/reduction kernels (launch-bound in eager mode) with a single
# kernel launch. Semantics matches the torch path above exactly.

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = __import__("os").environ.get("DYNADWDP_NO_TRITON", "0") != "1"
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

if _HAS_TRITON:

    @triton.jit
    def _dyn_select_kernel(
        logits_ptr, ready_ptr, out_w_ptr, out_id_ptr, out_kt_ptr,
        replaced_ptr,
        EPS, TAU,
        E: tl.constexpr, K: tl.constexpr, K_MIN: tl.constexpr,
        FIXED_WIDTH: tl.constexpr, DUMMY_ID: tl.constexpr,
    ):
        t = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, E)
        lg = tl.load(logits_ptr + t * E + offs).to(tl.float32)
        rdy = tl.load(ready_ptr + offs)

        mx = tl.max(lg, axis=0)
        p = tl.exp(lg - mx)
        p = p / tl.sum(p, axis=0)

        # tokens whose original top-1 is not ready are "replaced"
        o1 = tl.argmax(p, axis=0)
        r1 = tl.load(ready_ptr + o1)
        if r1 == 0:
            tl.atomic_add(replaced_ptr, 1.0)

        pm = tl.where(rdy != 0, p, 0.0)
        pm = pm / tl.maximum(tl.sum(pm, axis=0), 1e-12)

        slot = tl.arange(0, K)
        ws = tl.zeros((K,), dtype=tl.float32)
        ids = tl.full((K,), -1, dtype=tl.int32)
        pmw = pm
        prev_v = 1e30
        cum = 0.0
        need_k = K
        gap_k = K
        for k in tl.static_range(K):
            v = tl.max(pmw, axis=0)
            idx = tl.argmax(pmw, axis=0)
            gap_hit = (k >= 1) & (TAU > 0) & ((prev_v - v) >= TAU * tl.maximum(prev_v, 1e-12))
            gap_k = tl.minimum(gap_k, tl.where(gap_hit, k, K))
            cum += v
            need_k = tl.minimum(need_k, tl.where(cum >= 1.0 - EPS, k + 1, K))
            ws = tl.where(slot == k, v, ws)
            ids = tl.where(slot == k, idx.to(tl.int32), ids)
            pmw = tl.where(offs == idx, -1.0, pmw)
            prev_v = v

        kt = tl.maximum(tl.minimum(need_k, gap_k), K_MIN)
        kt = tl.minimum(kt, K)
        keep = (slot < kt) & (ws > 0)
        w_out = tl.where(keep, ws, 0.0)
        if FIXED_WIDTH:
            # every slot carries a REAL expert id; slots >= kt keep weight 0
            id_out = ids
        else:
            id_out = tl.where(keep, ids, DUMMY_ID)
        w_out = w_out / tl.maximum(tl.sum(w_out, axis=0), 1e-12)

        tl.store(out_w_ptr + t * K + slot, w_out)
        tl.store(out_id_ptr + t * K + slot, id_out)
        tl.store(out_kt_ptr + t, kt)


def _dyn_select_fused(
    router_logits: torch.Tensor,
    ready_mask: torch.Tensor,
    k_min: int,
    k_max: int,
    epsilon: float,
    tau: float,
    fixed_width: bool = True,
    dummy_expert_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused path. Returns (weights fp32 [T,K], ids int32 [T,K], kt int32 [T])."""
    T, E = router_logits.shape
    device = router_logits.device
    out_w = torch.empty(T, k_max, device=device, dtype=torch.float32)
    out_id = torch.empty(T, k_max, device=device, dtype=torch.int32)
    out_kt = torch.empty(T, device=device, dtype=torch.int32)
    tele = _tele(device)
    _dyn_select_kernel[(T,)](
        router_logits, ready_mask,
        out_w, out_id, out_kt,
        tele.replaced_sum,
        epsilon, tau,
        E=E, K=k_max, K_MIN=k_min,
        FIXED_WIDTH=bool(fixed_width), DUMMY_ID=int(dummy_expert_id),
        num_warps=4,
    )
    tele.avg_k_sum += out_kt.to(torch.float32).mean()
    return out_w, out_id, out_kt


def tele_summary(device) -> dict:
    """Read accumulators (synchronizes); used only by the stats dumper."""
    tele = _TELEMETRY.get(_tele_key(device))
    if tele is None or int(tele.count.item()) == 0:
        return {}
    n = tele.count.item()
    out = {
        "layers_called": n,
        "avg_k": (tele.avg_k_sum / n).item(),
        "replaced_per_layer": (tele.replaced_sum / n).item(),
        "select_us": (tele.us_sum / n).item(),
    }
    tele.count.zero_(); tele.avg_k_sum.zero_(); tele.replaced_sum.zero_(); tele.us_sum.zero_()
    lt_n = tele.lt_count.item()
    if lt_n > 0:
        out["lifetime_layers"] = lt_n
        out["lifetime_avg_k"] = (tele.lt_avg_k_sum / lt_n).item()
    return out
