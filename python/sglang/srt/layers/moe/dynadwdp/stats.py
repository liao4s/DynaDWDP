"""Runtime statistics for DynaDWDP (experiment data collection)."""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Dict

import torch


class DynaDWDPStats:
    """Accumulates per-forward metrics; dumped periodically to /tmp."""

    def __init__(self, rank: int, dump_every: int = 200, dump_dir: str = "/tmp"):
        self.rank = rank
        self.dump_every = dump_every
        self.dump_dir = dump_dir
        self.counters: Dict[str, float] = defaultdict(float)
        self.n_forwards = 0
        self._last_dump = time.time()
        # per-forward samples kept in memory (bounded)
        self.samples: list = []

    def record_forward(
        self,
        layer_idx: int,
        num_tokens: int,
        avg_k: float,
        n_selected_experts: int,
        n_ready_experts: int,
        n_replaced_tokens: int,
        n_fallback_fetches: int,
        pred_set_size: int,
        pred_hit_frac: float,
        prefetch_bytes: int,
        predict_us: float = 0.0,
        select_us: float = 0.0,
    ):
        self.n_forwards += 1
        c = self.counters
        c["forwards"] += 1
        c["tokens"] += num_tokens
        c["sum_avg_k"] += avg_k
        c["sum_selected"] += n_selected_experts
        c["sum_pred_set"] += pred_set_size
        c["sum_pred_hit"] += pred_hit_frac * max(pred_set_size, 1)
        c["sum_replaced_tokens"] += n_replaced_tokens
        c["sum_fallback"] += n_fallback_fetches
        c["sum_prefetch_bytes"] += prefetch_bytes
        c["sum_predict_us"] += predict_us
        c["sum_select_us"] += select_us
        if len(self.samples) < 20000:
            self.samples.append({
                "layer": layer_idx,
                "tokens": num_tokens,
                "avg_k": round(avg_k, 3),
                "n_sel": n_selected_experts,
                "n_ready": n_ready_experts,
                "replaced_tok": n_replaced_tokens,
                "fallback": n_fallback_fetches,
                "pred_set": pred_set_size,
                "pred_hit": round(pred_hit_frac, 4),
                "prefetch_mb": round(prefetch_bytes / 1e6, 2),
                "predict_us": round(predict_us, 1),
                "select_us": round(select_us, 1),
            })
        if self.n_forwards % self.dump_every == 0:
            self.dump()

    def dump(self):
        try:
            os.makedirs(self.dump_dir, exist_ok=True)
            path = os.path.join(self.dump_dir, f"dynadwdp_stats_rank{self.rank}.json")
            n = max(self.counters["forwards"], 1)
            data = {
                "rank": self.rank,
                "updated": time.time(),
                "n_forwards": self.n_forwards,
                "avg": {
                    "avg_k": self.counters["sum_avg_k"] / n,
                    "selected_experts": self.counters["sum_selected"] / n,
                    "pred_set": self.counters["sum_pred_set"] / n,
                    "pred_hit_frac": self.counters["sum_pred_hit"]
                    / max(self.counters["sum_pred_set"], 1),
                    "replaced_token_frac": self.counters["sum_replaced_tokens"]
                    / max(self.counters["tokens"], 1),
                    "fallback_fetches": self.counters["sum_fallback"] / n,
                    "prefetch_mb_per_fwd": self.counters["sum_prefetch_bytes"]
                    / n
                    / 1e6,
                    "predict_us": self.counters["sum_predict_us"] / n,
                    "select_us": self.counters["sum_select_us"] / n,
                },
                "samples_tail": self.samples[-500:],
            }
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, path)
        except Exception:
            pass


_stats: Dict[int, DynaDWDPStats] = {}


def get_stats(rank: int) -> DynaDWDPStats:
    if rank not in _stats:
        _stats[rank] = DynaDWDPStats(rank)
    return _stats[rank]
