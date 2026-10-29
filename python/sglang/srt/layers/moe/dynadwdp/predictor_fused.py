"""Fused prediction-pipeline kernels (DynaDWDP).

Replaces ~15 eager torch kernels + one hidden CPU synchronization
(`bool(torch.isfinite(x).all())` inside predict()) per MoE layer with 3
triton launches:

  K1 _pool_mean_kernel   hidden [T,H] -> pooled fp32 [H]
  K2 _head_kernel        [pooled; routing] + W[E,H+E] -> sigmoid(logits) [E]
                         (non-finite input -> uniform 1/E, no host sync)
  K3 _blend_topk_kernel  q_head, q_hist -> blend -> top-N ids int64 [n]

Semantics matches LookaheadHeadBank.predict() + the manager blend exactly.
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

if _HAS_TRITON:

    @triton.jit
    def _pool_mean_kernel(
        h_ptr, out_ptr, T,
        H: tl.constexpr, BLOCK_F: tl.constexpr, BLOCK_T: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_f = pid * BLOCK_F + tl.arange(0, BLOCK_F)
        acc = tl.zeros((BLOCK_F,), dtype=tl.float32)
        for t0 in range(0, T, BLOCK_T):
            offs_t = t0 + tl.arange(0, BLOCK_T)
            m = offs_t < T
            v = tl.load(
                h_ptr + offs_t[:, None] * H + offs_f[None, :],
                mask=m[:, None], other=0.0,
            ).to(tl.float32)
            acc += tl.sum(v, 0)
        tl.store(out_ptr + offs_f, acc / T)

    @triton.jit
    def _head_kernel(
        pooled_ptr, routing_ptr, w_ptr, out_ptr,
        H: tl.constexpr,          # hidden size (any value)
        E: tl.constexpr,          # num experts (128)
        BLOCK_E: tl.constexpr,    # outputs per program
        BLOCK_H: tl.constexpr,    # dot-product chunk
    ):
        pid = tl.program_id(0)
        offs_e = pid * BLOCK_E + tl.arange(0, BLOCK_E)
        offs_h = tl.arange(0, BLOCK_H)
        offs_r = tl.arange(0, 128)
        r = tl.load(routing_ptr + offs_r)
        bad = tl.max(
            tl.where((r != r) | (tl.abs(r) == float("inf")), 1.0, 0.0), 0
        )
        logits = tl.zeros((BLOCK_E,), dtype=tl.float32)
        for h0 in range(0, H, BLOCK_H):
            m = (h0 + offs_h) < H
            xf = tl.load(pooled_ptr + h0 + offs_h, mask=m, other=0.0)
            bad += tl.max(
                tl.where((xf != xf) | (tl.abs(xf) == float("inf")), 1.0, 0.0), 0
            )
            wf = tl.load(
                w_ptr + offs_e[:, None] * (H + 128) + (h0 + offs_h)[None, :],
                mask=m[None, :], other=0.0,
            )
            logits += tl.sum(wf * xf[None, :], 1)
        # tail: routing columns (H..H+127)
        we = tl.load(
            w_ptr + offs_e[:, None] * (H + 128) + (H + offs_r)[None, :]
        )
        logits += tl.sum(we * r[None, :], 1)
        out = tl.sigmoid(logits)
        out = tl.where(bad > 0, 1.0 / E, out)
        tl.store(out_ptr + offs_e, out)

    @triton.jit
    def _blend_topk_kernel(
        qhead_ptr, qhist_ptr, out_ptr,
        lam, n_full,
        MODE: tl.constexpr,       # 0: q=q_head  1: q=q_hist  2: blend
        E: tl.constexpr,
    ):
        offs = tl.arange(0, E)
        qh = tl.load(qhead_ptr + offs)
        if MODE == 0:
            q = qh
        elif MODE == 1:
            q = tl.load(qhist_ptr + offs)
        else:
            sh = tl.sum(qh, 0)
            qh = qh / tl.maximum(sh, 1e-9)
            qh = tl.where(qh != qh, 0.0, qh)  # NaN guard (0/0)
            qhist = tl.load(qhist_ptr + offs)
            q = lam * qh + (1.0 - lam) * qhist
        for i in range(n_full):
            v = tl.max(q, 0)
            idx = tl.argmax(q, 0)
            tl.store(out_ptr + i, idx.to(tl.int64))
            q = tl.where(offs == idx, -1.0, q)


def fused_predict(
    hidden_states: torch.Tensor,          # [T, H] bf16/fp16/fp32
    routing: Optional[torch.Tensor],      # [E] fp32 or None
    weight: torch.Tensor,                 # [E, H+E] fp32 (head.weight)
    q_hist: Optional[torch.Tensor],       # [E] fp32 or None
    head_warmup: bool,                    # True -> pure histogram mode
    lam: float,
    n_full: int,
    scratch: dict,
) -> torch.Tensor:
    """Returns int64 [n_full] top-N expert ids (device)."""
    H = hidden_states.shape[1]
    E = weight.shape[0]
    dev = hidden_states.device
    pooled = scratch.get("pooled")
    if pooled is None or pooled.numel() != H:
        pooled = torch.empty(H, device=dev, dtype=torch.float32)
        scratch["pooled"] = pooled
    T = hidden_states.shape[0]
    _pool_mean_kernel[(H // 64,)](
        hidden_states, pooled, T, H=H, BLOCK_F=64, BLOCK_T=128, num_warps=4
    )
    q_head = scratch.get("q_head")
    if q_head is None:
        q_head = torch.empty(E, device=dev, dtype=torch.float32)
        scratch["q_head"] = q_head
    if routing is None:
        routing = scratch.get("zeros")
        if routing is None:
            routing = torch.zeros(E, device=dev, dtype=torch.float32)
            scratch["zeros"] = routing
    _head_kernel[(E // 64,)](
        pooled, routing, weight, q_head,
        H=H, E=E, BLOCK_E=64, BLOCK_H=1024, num_warps=8,
    )
    if q_hist is None or head_warmup:
        mode = 1 if q_hist is not None else 0
    else:
        mode = 2
    idx = scratch.get("idx")
    if idx is None or idx.numel() != n_full:
        idx = torch.empty(n_full, device=dev, dtype=torch.int64)
        scratch["idx"] = idx
    _blend_topk_kernel[(1,)](
        q_head, q_hist if q_hist is not None else routing,
        idx, lam, n_full,
        MODE=mode, E=E, num_warps=4,
    )
    return idx
