"""EP expert predictor: lookahead heads + statistical corrector + set generator.

Timing contract (matches the double-buffer prefetch architecture):
  at MoE layer l compute time we observe a_l (the layer-l MoE input) and must
  issue the prefetch for layer l+2. The head for target layer l+2 therefore
  maps [pooled(a_l) ; routing_multihot_l] -> expert-activation distribution of
  layer (l+2)'s router, a distance-2 lookahead. routing_multihot_l is the
  layer-l ACTUAL dynamic-k selected-expert set (batch-level multi-hot) — the
  strongest cross-layer locality signal.

Set semantics (v2): the prefetch COUNT is controlled by the
--dynadwdp-predictor-strength parameter (N = round(R * (1 - s)), R = remote
expert count); the prefetch IDS are the top-N experts by predicted
probability. The old delta/mass-coverage generator is retired.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def build_head_input(
    pooled_feat: torch.Tensor,
    routing_multihot: Optional[torch.Tensor],
    num_experts: int,
) -> torch.Tensor:
    """Single source of truth for the head input — used by both predict() and
    submit_sample() so training and inference see identical features.

    pooled_feat: [hidden] fp32; routing_multihot: [E] float in {0,1} or None
    (None -> zero vector: no routing information available).
    Returns [hidden + E] fp32.
    """
    x = pooled_feat.detach().to(torch.float32).flatten()
    if routing_multihot is None:
        r = torch.zeros(num_experts, device=x.device, dtype=x.dtype)
    else:
        r = routing_multihot.detach().to(torch.float32).flatten()
        if r.numel() != num_experts:
            r = torch.zeros(num_experts, device=x.device, dtype=x.dtype)
    return torch.cat([x, r])


class LookaheadHeadBank:
    """One lightweight linear head per predicted MoE layer.

    head_{l+2}: Linear(hidden + E -> E) applied to [pooled(a_l) ; routing_l].
    Trained online (calibrate mode) with BCE against the actual batch-level
    expert multi-hot of layer l+2.
    """

    def __init__(
        self,
        moe_layer_indices: List[int],
        hidden_size: int,
        num_experts: int,
        lr: float = 1e-3,
        device_id: int = 0,
        dp_rank: int = 0,
    ):
        self.moe_layer_indices = list(moe_layer_indices)
        self.pos_of = {li: i for i, li in enumerate(self.moe_layer_indices)}
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.device = torch.device("cuda", device_id)
        in_dim = hidden_size + num_experts
        self.heads = nn.ModuleDict(
            {
                str(li): nn.Linear(in_dim, num_experts, bias=False)
                for li in self.moe_layer_indices
            }
        ).to(self.device, torch.float32)
        self.opt = torch.optim.AdamW(self.heads.parameters(), lr=lr)
        self._train_steps = 0
        self.enabled = True
        self._cache_dir = ""  # set by save()/manager; enables _autosave
        self._rank0 = dp_rank == 0  # only DP rank 0 persists checkpoints

    # ---------- inference ----------
    @torch.no_grad()
    def predict(
        self,
        target_layer: int,
        pooled_feat: torch.Tensor,
        routing_multihot: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns expert activation probability [E] (fp32, on device)."""
        key = str(target_layer)
        if key not in self.heads:
            return torch.full(
                (self.num_experts,), 1.0 / self.num_experts, device=self.device
            )
        head = self.heads[key]
        x = build_head_input(pooled_feat, routing_multihot, self.num_experts).to(
            self.device
        )
        if not bool(torch.isfinite(x).all()):
            return torch.full(
                (self.num_experts,), 1.0 / self.num_experts, device=self.device
            )
        return torch.sigmoid(head(x.unsqueeze(0))).squeeze(0)

    # ---------- training ----------
    def submit_sample(
        self,
        target_layer: int,
        pooled_feat: torch.Tensor,
        routing_multihot: Optional[torch.Tensor],
        label_multihot: torch.Tensor,
    ):
        """Train one (input features, actual-expert-set) pair immediately.

        pooled_feat/routing_multihot: the SAME features predict() saw for this
        target layer; label_multihot: [E] {0,1} actual dynamic-k hit set.
        Uses BCEWithLogits on the raw head output (numerically stable; the
        sigmoid is applied only at inference inside predict()).
        """
        key = str(target_layer)
        if key not in self.heads:
            return
        x = build_head_input(pooled_feat, routing_multihot, self.num_experts).to(
            self.device
        )
        if not bool(torch.isfinite(x).all()):
            return  # skip non-finite features (e.g., warmup dummy batches)
        y = label_multihot.detach().to(self.device, torch.float32).flatten()
        head = self.heads[key]
        # model forward runs under no_grad — re-enable for head training
        with torch.enable_grad():
            logits = head(x.unsqueeze(0))
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, y.unsqueeze(0)
            )
            if bool(torch.isfinite(loss)):
                loss.backward()
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)
                self._train_steps += 1
                if self._train_steps % 1000 == 0 and self._rank0:
                    self._autosave()

    @property
    def train_steps(self) -> int:
        return self._train_steps

    # ---------- persistence ----------
    def _autosave(self):
        if getattr(self, "_cache_dir", ""):
            self.save(self._cache_dir)

    def save(self, cache_dir: str):
        try:
            os.makedirs(cache_dir, exist_ok=True)
            self._cache_dir = cache_dir
            # atomic write: concurrent readers (or a SIGKILL mid-write) must
            # never observe a torn checkpoint
            tmp = os.path.join(cache_dir, "lookahead_heads.pt.tmp")
            torch.save(
                {"state": self.heads.state_dict(), "steps": self._train_steps},
                tmp,
            )
            os.replace(tmp, os.path.join(cache_dir, "lookahead_heads.pt"))
        except Exception as e:
            logger.warning(f"lookahead head save failed: {e}")

    def load(self, cache_dir: str) -> bool:
        path = os.path.join(cache_dir, "lookahead_heads.pt")
        if not os.path.exists(path):
            return False
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.heads.load_state_dict(ckpt["state"])
            self._train_steps = ckpt.get("steps", 0)
            logger.info(
                f"loaded lookahead heads from {path} ({self._train_steps} train steps)"
            )
            return True
        except Exception as e:
            logger.warning(f"lookahead head load failed: {e}")
            return False


class StatisticalCorrector:
    """Per-target-layer EWMA histogram of batch-level expert routing mass."""

    def __init__(self, moe_layer_indices: List[int], num_experts: int, device):
        self.hist: Dict[int, torch.Tensor] = {
            li: torch.full((num_experts,), 1.0 / num_experts, device=device)
            for li in moe_layer_indices
        }
        self.alpha = 0.05
        self.device = device

    def update(self, target_layer: int, mass: torch.Tensor):
        """mass: [E] nonneg (e.g., sum of selected routing weights per expert)."""
        h = self.hist.get(target_layer)
        if h is None:
            return
        m = mass.to(self.device, torch.float32)
        # NOTE: `if s > 0` on a CUDA scalar would force a D2H + stream sync every
        # layer. Do the guard on-device instead (identical semantics: update
        # only when the mass sum is positive, otherwise keep the histogram).
        s = m.sum()
        mn = m / s.clamp_min(1e-12)
        hn = h * (1 - self.alpha) + mn * self.alpha
        # s <= 0 -> keep h unchanged (exactly the old `if s > 0` semantics)
        h.copy_(torch.where(s > 0, hn, h))

    def get(self, target_layer: int) -> Optional[torch.Tensor]:
        return self.hist.get(target_layer)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {str(k): v.clone() for k, v in self.hist.items()}

    def load_state_dict(self, sd: Dict[str, torch.Tensor]):
        for k, v in sd.items():
            li = int(k)
            if li in self.hist:
                self.hist[li].copy_(v.to(self.device))


def top_n_prefetch_set(
    q: torch.Tensor,
    local_start: int,
    local_end: int,
    n: int,
) -> List[int]:
    """Top-N prefetch set by predicted probability, excluding local experts.

    q: [E] predicted hit probabilities (only the ORDER matters). n >= 1.
    """
    n = max(1, int(n))
    order = torch.argsort(q, descending=True).tolist()
    selected: List[int] = []
    for e in order:
        if len(selected) >= n:
            break
        if local_start <= e < local_end:
            continue  # resident, no prefetch needed
        selected.append(e)
    return selected


def generate_prefetch_set(
    q: torch.Tensor,
    local_start: int,
    local_end: int,
    delta: float,
    extra_experts: int,
    max_experts: int = 0,
) -> Tuple[List[int], float]:
    """[RETIRED] Old mass-coverage generator — kept only for A/B comparison.

    The sigmoid-mass-coverage semantics is incorrect (independent sigmoid
    outputs do not form a routing distribution) and was replaced by the
    strength-parameter-driven top-N generator above.
    """
    order = torch.argsort(q, descending=True)
    total = float(q.sum())
    if total <= 0:
        order_list = order.tolist()
        return [e for e in order_list if not (local_start <= e < local_end)], 0.0

    covered = 0.0
    selected: List[int] = []
    for e in order.tolist():
        if covered >= delta and len(selected) >= extra_experts:
            break
        if local_start <= e < local_end:
            continue  # resident, no prefetch needed
        selected.append(e)
        covered += float(q[e])
        if max_experts and len(selected) >= max_experts:
            break
    return selected, covered


def merge_ready_sets(
    predicted: List[int],
    local_start: int,
    local_end: int,
    cached: Optional[set] = None,
) -> set:
    """Full ready set = local resident + predicted prefetch + cached."""
    ready = set(range(local_start, local_end))
    ready.update(predicted)
    if cached:
        ready.update(cached)
    return ready
