"""DynaDWDManager: orchestrates predictor, dynamic router and subset prefetch.

Lifecycle mirrors the mainline DwdpManager:
  setup(model)            -> build layout/transport/buffer (reusing mainline
                             machinery), attach lookahead heads
  prefetch_first_layers() -> full prefetch of the first two MoE layers
  wait_prefetch(l)        -> mainline event wait
  record_compute_and_prefetch_next(l, hidden_states)
                          -> consume event; predict C_{l+2} from pooled a_l;
                             issue subset prefetch for l+2
  prepare_layer(l, router_logits)
                          -> build ready mask, run DynTopK (dynamic k + set
                             projection + replacement/fallback), return
                             StandardTopKOutput; feed actual routing to the
                             corrector and (in calibrate mode) train heads
"""

from __future__ import annotations

import bisect
import json
import os
import logging
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

import torch

from sglang.srt.layers.moe.dynadwdp.config import DynaDWDPConfig, DynaDWDPMode
from sglang.srt.layers.moe.dynadwdp.dynamic_router import dyn_select
from sglang.srt.layers.moe.dynadwdp.mover import SubsetPrefetcher
from sglang.srt.layers.moe.dynadwdp.predictor_fused import (
    fused_predict,
    _HAS_TRITON as _FUSED_PREDICT_TRITON,
)
import os as _os
_FUSED_PREDICT = _FUSED_PREDICT_TRITON and _os.environ.get(
    "DYNADWDP_FUSED_PREDICT", "1"
) == "1"
from sglang.srt.layers.moe.dynadwdp.predictor import (
    LookaheadHeadBank,
    StatisticalCorrector,
    top_n_prefetch_set,
)
from sglang.srt.layers.moe.dynadwdp.stats import get_stats
from sglang.srt.layers.moe.dwdp.dwdp_manager import DwdpManager
from sglang.srt.layers.moe.dwdp.weight_manager import DWDPWeightManager
from sglang.srt.layers.moe.topk import StandardTopKOutput

logger = logging.getLogger(__name__)

_manager: Optional["DynaDWDManager"] = None
_router_only: bool = False
_router_cfg: Optional[DynaDWDPConfig] = None


def get_dynadwdp_manager() -> Optional["DynaDWDManager"]:
    return _manager


def set_dynadwdp_enabled(mgr: Optional["DynaDWDManager"]) -> None:
    global _manager
    _manager = mgr


def set_router_only(enabled: bool, cfg: Optional[DynaDWDPConfig] = None) -> None:
    """Enable dynamic routing WITHOUT the DWDP weight machinery (used for
    isolated quality evaluation on standard EP deployments)."""
    global _router_only, _router_cfg
    _router_only = enabled
    _router_cfg = cfg


def router_only_config() -> Optional[DynaDWDPConfig]:
    return _router_cfg if _router_only else None


class DynaDWDManager(DwdpManager):
    """Subclasses the mainline manager; replaces the weight manager with the
    SubsetPrefetcher and adds prediction / dynamic routing."""

    def __init__(self, server_args, config: DynaDWDPConfig):
        super().__init__(server_args)
        self.cfg = config
        self.hidden_size: Optional[int] = None
        self.heads: Optional[LookaheadHeadBank] = None
        self.corrector: Optional[StatisticalCorrector] = None
        # target_layer -> prefetch set issued for it (ready set for routing)
        self.pending_ready: Dict[int, set] = {}
        # target_layer -> (pooled feature, routing multihot) captured at record
        # time — the exact prediction-time features, reused as training input
        self._pending_inputs: Dict[int, tuple] = {}
        self._decode_full_done: set = set()
        self._prefetched_experts: Dict[int, List[int]] = {}
        self._last_prefetch_bytes = 0
        self._last_predict_us = 0.0
        self._mask_cache: Dict[int, torch.Tensor] = {}
        # predictor v2: routing features & telemetry
        self._last_routing: Dict[int, torch.Tensor] = {}
        self._decode_mode = False
        self._decode_mask: Optional[torch.Tensor] = None
        self._pred_pending: Dict[int, tuple] = {}
        self._pred_scratch: dict = {}
        self._pinned_pool: Dict[int, torch.Tensor] = {}
        # Dedicated high-priority stream for the tiny prediction-ids D2H. Sharing
        # the weight-copy stream would queue it behind a multi-GB prefetch DMA,
        # so the consumer's event would not be ready and would hard-sync.
        self._id_stream = torch.cuda.Stream(
            device=torch.device("cuda", self.device_id), priority=-1
        )
        dev = torch.device("cuda", self.device_id)
        self._ptel = {
            "n": torch.zeros(1, dtype=torch.float64, device=dev),
            "actual_sum": torch.zeros(1, dtype=torch.float64, device=dev),
            "hit_sum": torch.zeros(1, dtype=torch.float64, device=dev),
        }
        self._btel = {"calls": 0, "slots": 0, "bytes": 0, "full_calls": 0}
        self._layer_pref = {}
        self._stats_err = 0
        self._early = {}
        self._pref_early = 0
        self._pref_late = 0
        self._late_skips = 0
        # P2: device-side prefetch (no host ids / no D2H on the critical path)
        self._dev_prefetch = _os.environ.get("DYNADWDP_DEV_PREFETCH", "0") == "1"

    # ---------- setup ----------
    def setup(self, model) -> None:
        # run the mainline setup but with our weight manager
        if self._weight_manager is not None:
            return
        from sglang.srt.layers.moe.dwdp.dwdp_manager import _EXPERT_WEIGHT_NAMES

        moe_layers = self._collect_moe_layers(model)
        if not moe_layers:
            raise RuntimeError("DynaDWDP enabled but no FusedMoE layers found")
        self._moe_layer_indices = [li for li, _ in moe_layers]

        expert_counts = {e.num_global_routed_experts for _, e in moe_layers}
        assert len(expert_counts) == 1, expert_counts
        num_routed = expert_counts.pop()
        assert num_routed % self.dwdp_size == 0

        # Clamp dynamic-k to the model's trained top-k: slots beyond top_k
        # dilute the routing mix inside kernels that normalize over the slot
        # dimension, which corrupts long-context generation (observed on
        # gpt-oss top_k=4 with k_max=8).
        model_top_k = int(moe_layers[0][1].top_k)
        if self.cfg.k_max > model_top_k:
            logger.info(
                f"DynaDWDP: clamping k_max {self.cfg.k_max} -> {model_top_k} "
                f"(model trained top_k={model_top_k})"
            )
            self.cfg.k_max = model_top_k
        # Routing with fewer experts than trained shifts the distribution and
        # degrades long-context quality (immediate-EOS observed on gpt-oss)
        # unless a consistency-correction calibration is run. Default soft
        # mode therefore keeps the trained top-k; variable k stays opt-in
        # via an explicit --dynadwdp-k-min below top_k.
        if self.cfg.k_min is None:  # user did not override -> static top-k
            self.cfg.k_min = self.cfg.k_max
        self.cfg.k_min = min(self.cfg.k_min, self.cfg.k_max)

        from sglang.srt.layers.moe.dwdp.layout import DwdpExpertLayout

        self.layout = DwdpExpertLayout(
            num_routed_experts=num_routed,
            dwdp_size=self.dwdp_size,
            dwdp_rank=self.dwdp_rank,
        )
        logger.info(
            f"DynaDWDP layout: {num_routed} experts, mode={self.cfg.mode.value}, "
            f"local [{self.layout.local_expert_start}, {self.layout.local_expert_end})"
        )

        local_params = {}
        for li, experts in moe_layers:
            local_params[(li, "w13_weight")] = experts.w13_weight.data
            local_params[(li, "w2_weight")] = experts.w2_weight.data
        from sglang.srt.layers.moe.dwdp.layout import build_layer_weight_specs

        layer_weight_specs = build_layer_weight_specs(
            local_params, self.layout.num_routed_experts
        )

        from sglang.srt.runtime_context import get_parallel

        group = get_parallel().tp_group
        from sglang.srt.layers.moe.dwdp.transport import DWDPTransport

        transport = DWDPTransport.create(
            layer_weight_specs=layer_weight_specs,
            local_params=local_params,
            group=group,
            layout=self.layout,
            device_id=self.device_id,
        )
        from sglang.srt.layers.moe.dwdp.weight_buffer import WeightBuffer

        weight_buffer = WeightBuffer.create(
            layer_weight_specs=layer_weight_specs,
            handles=transport.handle_set,
            local_start=self.layout.local_expert_start,
            local_end=self.layout.local_expert_end,
            dwdp_size=self.dwdp_size,
            device_id=self.device_id,
        )

        self._fill_edge_bytes(weight_buffer, transport.peer_views)

        self._weight_manager = SubsetPrefetcher(
            weight_buffer=weight_buffer,
            peer_views=transport.peer_views,
            peer_ranges=self.layout.peer_ranges,
            moe_layer_indices=self._moe_layer_indices,
            weight_names=list(_EXPERT_WEIGHT_NAMES),
            dwdp_rank=self.dwdp_rank,
            dwdp_size=self.dwdp_size,
            transport=transport,
        )

        for li, experts in moe_layers:
            experts.bind_full_expert_weights(
                {
                    name: weight_buffer.get_full_tensor(li, name)
                    for name in weight_buffer.weight_names(li)
                }
            )
        self._allgather_small_params(moe_layers, group)

        # hidden size from any weight spec: w13 full shape [E, 2*d_f, d]
        any_spec = layer_weight_specs[self._moe_layer_indices[0]]["w13_weight"]
        # hidden size MUST come from the model config: the w13 spec's last dim
        # is the expert FFN intermediate size on some quant layouts (e.g.
        # gpt-oss mxfp4), which would mis-shape the look-ahead heads
        cfg_hs = getattr(getattr(model, "config", None), "hidden_size", None)
        self.hidden_size = int(cfg_hs) if cfg_hs else int(any_spec.full_shape[-1])

        self.heads = LookaheadHeadBank(
            moe_layer_indices=self._moe_layer_indices,
            hidden_size=self.hidden_size,
            num_experts=self.layout.num_routed_experts,
            lr=self.cfg.lr,
            device_id=self.device_id,
            dp_rank=getattr(self, "dwdp_rank", 0),
        )
        # the autosave path needs the cache dir regardless of whether setup's
        # no-clobber save was skipped — otherwise training is never persisted
        self.heads._cache_dir = self.cfg.head_cache_dir or ""
        if not self.cfg.calibrate:
            self.heads.load(self.cfg.head_cache_dir)
        self.corrector = StatisticalCorrector(
            self._moe_layer_indices,
            self.layout.num_routed_experts,
            torch.device("cuda", self.device_id),
        )
        self._load_corrector_snapshot()

        # first two layers: full prefetch (no routing history yet)
        self._weight_manager.prefetch_first_layers()
        for li in self._moe_layer_indices[:2]:
            self.pending_ready[li] = set(range(self.layout.num_routed_experts))

        # arm the autosave path — but NEVER clobber an existing checkpoint:
        # 4 DP ranks start concurrently here and the untrained default would
        # race with (and overwrite) a trained checkpoint saved by a previous
        # calibration run
        if not os.path.exists(
            os.path.join(self.cfg.head_cache_dir or "", "lookahead_heads.pt")
        ):
            self.heads.save(self.cfg.head_cache_dir)

        # enable -1 drop-sentinel handling in the MoE runners
        from sglang.srt.layers.moe.dynadwdp.dynamic_router import (
            set_drop_sentinel_enabled,
        )
        set_drop_sentinel_enabled(True)

        self._stop_stats = False
        self._stats_thread = threading.Thread(
            target=self._stats_loop, daemon=True
        )
        self._stats_thread.start()
        logger.info("DynaDWDP setup complete.")

    def _stats_loop(self):
        """Periodically dump GPU-side router/predictor telemetry (syncs in
        background thread only)."""
        from sglang.srt.layers.moe.dynadwdp.dynamic_router import tele_summary

        device = torch.device("cuda", self.device_id)
        while not self._stop_stats:
            time.sleep(5)
            try:
                s = tele_summary(device)
                pt = self._ptel
                n = int(pt["n"].item())
                # prefetch telemetry is ALWAYS dumped (independent of predictor
                # samples): per-layer prefetched expert count + bytes + totals
                s_pred = {
                    "predictor_strength": self.cfg.predictor_strength,
                    "prefetch_calls": self._btel["calls"],
                    "prefetch_expert_slots": self._btel["slots"],
                    "prefetch_bytes": self._btel["bytes"],
                    "full_prefetch_calls": self._btel["full_calls"],
                    "per_layer_experts": {str(k): v[0] for k, v in self._layer_pref.items()},
                    "prefetch_early": self._pref_early,
                    "prefetch_late": self._pref_late,
                    "prefetch_late_skips": self._late_skips,
                }
                self._btel = {"calls": 0, "slots": 0, "bytes": 0, "full_calls": 0}
                s.update(s_pred)
                if n > 0:
                    s.update({
                        "layers_sampled": n,
                        "avg_actual_experts": (pt["actual_sum"] / n).item(),
                        "avg_hit_experts": (pt["hit_sum"] / n).item(),
                        "avg_hit_frac": ((pt["hit_sum"] / pt["actual_sum"].clamp_min(1.0))).item(),
                    })
                    pt["n"].zero_(); pt["actual_sum"].zero_(); pt["hit_sum"].zero_()
                    # periodic corrector snapshot (crash-safe autosave)
                    self._save_corrector_snapshot()
                s["ts"] = time.time()
                with open(
                    f"/tmp/dynadwdp_tele_rank{self.dwdp_rank}.json", "w"
                ) as f:
                    json.dump(s, f)
                # k-trace dump lives here too: one .item() per stats tick
                # instead of one per MoE layer (that per-layer sync dominated
                # the CPU time in front of fused_moe).
                try:
                    if os.environ.get("DYNADWDP_KT_DUMP", "0") == "1":
                        from sglang.srt.layers.moe.dynadwdp.dynamic_router import (
                            kt_snapshot,
                        )
                        snap = kt_snapshot()
                        if snap:
                            with open(f"/tmp/dynadwdp_kt_dump_pid{os.getpid()}.json", "w") as kf:
                                json.dump(snap, kf)
                except Exception:
                    pass
            except Exception as _e:
                self._stats_err = getattr(self, "_stats_err", 0) + 1
                if self._stats_err <= 3:
                    logger.warning(f"dynadwdp stats loop error: {_e!r}")

    # ---------- helpers ----------
    def _pooled_feature(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.detach().to(torch.float32).mean(dim=0)

    def _predict_ready(
        self,
        target_layer: int,
        pooled: torch.Tensor,
        routing_multihot: Optional[torch.Tensor],
    ) -> List[int]:
        """Predicted prefetch set (remote-only, priority ordered).

        Count semantics: N = round(R * (1 - predictor_strength)) where R is
        the remote expert count — s=0 reproduces DWDP-Full's full prefetch.
        Id semantics: top-N by blended predicted probability (head over
        [pooled ; routing_multihot] features, blended with the statistical
        corrector). Unhit tokens are served by local replacement downstream.
        """
        cfg = self.cfg
        local_w = self.layout.local_expert_end - self.layout.local_expert_start
        R = self.layout.num_routed_experts - local_w
        n = max(1, int(round(R * (1.0 - cfg.predictor_strength))))

        q_head = self.heads.predict(target_layer, pooled, routing_multihot)
        q_hist = self.corrector.get(target_layer) if self.corrector else None
        if q_hist is None or self.heads.train_steps < cfg.head_warmup:
            q = q_hist if q_hist is not None else q_head
        else:
            # normalize both signals to sum 1 (only ORDER matters for top-N,
            # but the blend must compare like with like)
            qh = q_head / q_head.sum().clamp_min(1e-9)
            lam = cfg.corrector_lambda
            q = lam * qh + (1 - lam) * q_hist
        return top_n_prefetch_set(
            q, self.layout.local_expert_start, self.layout.local_expert_end, n
        )

    # ---------- prefetch hooks (called from FusedMoE.forward_impl) ----------
    def record_compute_and_prefetch_next(
        self, layer_idx: int, hidden_states: Optional[torch.Tensor] = None
    ) -> None:
        wm = self._weight_manager
        if wm is None:
            return
        device = torch.device("cuda", self._weight_buffer_device())
        wm._consume_events[wm._weight_buffer.buffer_index_for_layer(layer_idx)].record(
            torch.cuda.current_stream(device)
        )
        next_layer = wm.next_moe_layer(layer_idx)
        if next_layer is None:
            return
        next_next = wm.next_moe_layer(next_layer)
        if next_next is None:
            return

        t0 = time.perf_counter()
        if hidden_states is not None and hidden_states.shape[0] > 0 and not self._decode_mode:
            routing = self._last_routing.get(layer_idx)
            pooled = self._pooled_feature(hidden_states)
            head = (
                self.heads.heads[str(next_next)]
                if str(next_next) in self.heads.heads
                else None
            )
            if (
                _FUSED_PREDICT
                and head is not None
                and hidden_states.shape[1] == self.heads.hidden_size
            ):
                # fused path: 3 triton kernels (pool-mean, head+sigmoid,
                # blend+topN) replacing ~15 eager kernels + a hidden host
                # sync (bool(isfinite)); ids land as int64 [n_full]
                q_hist = self.corrector.get(next_next) if self.corrector else None
                top_idx = fused_predict(
                    hidden_states, routing, head.weight.detach(),
                    q_hist,
                    self.heads.train_steps < self.cfg.head_warmup,
                    self.cfg.corrector_lambda,
                    min(
                        max(1, int(round(
                            (self.layout.num_routed_experts
                             - (self.layout.local_expert_end
                                - self.layout.local_expert_start))
                            * (1.0 - self.cfg.predictor_strength)))
                        ) + (
                            self.layout.local_expert_end
                            - self.layout.local_expert_start),
                        self.layout.num_routed_experts,
                    ),
                    self._pred_scratch,
                )
                pinned = self._get_pinned(top_idx.numel(), next_next)
                idx_dev = top_idx.clone()  # device-side ids for the mask (no H2D)
                ev_pred = torch.cuda.Event()
                ev_pred.record(torch.cuda.current_stream(device))
                cs = self._id_stream
                cs.wait_event(ev_pred)
                with torch.cuda.stream(cs):
                    pinned.copy_(top_idx, non_blocking=True)
                    ev = torch.cuda.Event()
                    ev.record(cs)
                self._decode_full_done.discard(next_next)  # buffer will be rewritten
                n_remote = max(1, top_idx.numel() - (
                    self.layout.local_expert_end
                    - self.layout.local_expert_start))
                self._pred_pending[next_next] = (ev, pinned, n_remote, idx_dev)
                if self._dev_prefetch:
                    # P2: issue the DMA right here, driven by the device ids;
                    # nothing has to come back to the host.
                    try:
                        self._weight_manager.prefetch_layer_experts_dev(
                            next_next, idx_dev, wait_ev=ev_pred)
                        self._pred_pending.pop(next_next, None)
                        self._early[next_next] = idx_dev
                        self._pref_early += 1
                    except Exception as _e:  # noqa: BLE001
                        logger.warning(f"dev prefetch failed, falling back: {_e}")
                self._pending_inputs[next_next] = (pooled, routing)
                self._drain_early()  # plan A: issue ready prefetches immediately
                self._trndbg_store = getattr(self, "_trndbg_store", 0) + 1
                if self._trndbg_store % 400 == 1:
                    logger.info(f"[trndbg] rec l={layer_idx} nn={next_next} stored fused-pending")
                self._last_predict_us = (time.perf_counter() - t0) * 1e6
                return

            # layer-l's own dynamic-k routing, just computed by prepare_layer(l)
            # GPU-side prediction; ids are copied ASYNCHRONOUSLY into a pinned
            # buffer — no host sync here (a sync would drain the GPU pipeline
            # every layer). The prefetch + ready-set are consumed later, in
            # _consume_prediction at the target layer's prepare time, where
            # the wait overlaps the previous layer's GPU work.
            q_head = self.heads.predict(next_next, pooled, routing)
            q_hist = self.corrector.get(next_next) if self.corrector else None
            if q_hist is None or self.heads.train_steps < self.cfg.head_warmup:
                q = q_hist if q_hist is not None else q_head
            else:
                qh = q_head / q_head.sum().clamp_min(1e-9)
                lam = self.cfg.corrector_lambda
                q = lam * qh + (1 - lam) * q_hist
            local_w = self.layout.local_expert_end - self.layout.local_expert_start
            R = self.layout.num_routed_experts - local_w
            n = max(1, int(round(R * (1.0 - self.cfg.predictor_strength))))
            n_full = min(n + local_w, self.layout.num_routed_experts)
            top_idx = torch.topk(q, n_full).indices.to(torch.int64)
            # D2H on the COPY stream, gated ONLY by the prediction kernels
            # (not the whole compute stream): the ids land in pinned memory
            # right after the head forward, so the consumer's query() is
            # satisfied ~2 layers of GPU time before the copies are needed —
            # a compute-stream D2H would complete only at this layer's MoE
            # end, forcing a blocking synchronize every layer.
            ev_pred = torch.cuda.Event()
            ev_pred.record(torch.cuda.current_stream(device))
            cs = self._id_stream
            cs.wait_event(ev_pred)
            pinned = self._get_pinned(n_full, next_next)
            with torch.cuda.stream(cs):
                pinned.copy_(top_idx, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(cs)
            self._pred_pending[next_next] = (ev, pinned, n, top_idx.clone())
            self._pending_inputs[next_next] = (pooled, routing)
            dt_us = (time.perf_counter() - t0) * 1e6
        else:
            # no tokens (or decode mode): the layer's full remote set must be
            # resident, but its buffer content is STABLE across decode steps —
            # re-issuing the full 1.81 GB DMA every step for every layer was a
            # ~512x traffic amplification that stalled generation. Issue once
            # per layer per decode phase; prefill clears the set (its own
            # fetches rewrite the buffers).
            # REVERTED decode dedup: under concurrent prefill+decode the
            # per-layer "already fetched" skip let decode read buffers that a
            # concurrent request's subset prefetch had rewritten (garbage
            # weights -> corrupted output). The v5 behavior (full prefetch
            # every step) is slower but always correct.
            wm.prefetch_layer(next_next)
            self._pred_pending[next_next] = None
            predicted = [
                e
                for e in range(self.layout.num_routed_experts)
                if not (
                    self.layout.local_expert_start
                    <= e
                    < self.layout.local_expert_end
                )
            ]
            self._prefetched_experts[next_next] = predicted
            self.pending_ready[next_next] = set(range(self.layout.num_routed_experts))
            self._layer_pref[next_next] = (len(predicted), 0)
            self._mask_cache.pop(next_next, None)
            self._last_prefetch_bytes = 0
            self._btel["full_calls"] += 1
            return

    def _get_pinned(self, topk_n: int, layer_idx: Optional[int] = None) -> torch.Tensor:
        """Pinned id buffer.

        With layer_idx, a per-layer slot is used (P1 ring): the previous
        single-slot-per-size buffer forced the consumer to wait for the D2H to
        land before the next prediction could reuse it, which is what produced
        the per-layer cudaStreamSynchronize.
        """
        key = topk_n if layer_idx is None else (layer_idx, topk_n)
        buf = self._pinned_pool.get(key)
        if buf is None:
            buf = torch.empty(topk_n, dtype=torch.int64, pin_memory=True)
            self._pinned_pool[key] = buf
        return buf

    def _next_index(self, layer_idx: int) -> Optional[int]:
        """The MoE layer after layer_idx (None if layer_idx is the last)."""
        idx = self._moe_layer_indices
        pos = bisect.bisect_right(idx, layer_idx)
        return idx[pos] if pos < len(idx) else None

    def _update_mask(self, layer_idx: int, idx_dev: torch.Tensor) -> None:
        """Project the ready set onto the device-side routing mask."""
        ls, le = self.layout.local_expert_start, self.layout.local_expert_end
        mask = self._mask_cache.get(layer_idx)
        if mask is None:
            mask = torch.zeros(
                self.layout.num_routed_experts, dtype=torch.bool,
                device=torch.device("cuda", self._weight_buffer_device()))
            self._mask_cache[layer_idx] = mask
        mask.zero_()
        mask[ls:le] = True
        mask.index_fill_(0, idx_dev, True)

    def _issue_prefetch(self, layer_idx: int, pinned, n: int) -> None:
        """Build the DMA list on the host and issue the subset prefetch."""
        ids_all = pinned.tolist()
        ls, le = self.layout.local_expert_start, self.layout.local_expert_end
        predicted = [e for e in ids_all if not (ls <= e < le)][:n]
        cap = self.cfg.max_prefetch_experts
        if cap and len(predicted) > cap:
            predicted = predicted[:cap]
        wm = self._weight_manager
        nbytes = wm.prefetch_layer_experts(layer_idx, predicted)
        ready = set(range(ls, le))
        ready.update(predicted)
        self._prefetched_experts[layer_idx] = predicted
        self.pending_ready[layer_idx] = ready
        self._last_prefetch_bytes = nbytes
        # telemetry: subset vs full-prefetch fallbacks counted separately
        bt = self._btel
        if nbytes > 0:
            bt["calls"] += 1
            bt["slots"] += len(predicted)
            bt["bytes"] += int(nbytes)
        else:
            bt["full_calls"] += 1
        self._layer_pref[layer_idx] = (len(predicted), int(nbytes))

    def _drain_early(self) -> None:
        """Issue pending prefetches as early as possible (plan A).

        Called at the end of every MoE layer: for any pending prediction whose
        ids have already landed in the pinned buffer, issue the DMA NOW — one
        full layer before the target layer's router. The DMA then overlaps the
        next layer's attention/gate/MoE instead of sitting directly in front of
        fused_moe (which is what the previous consume-at-target-layer order did).
        The check is non-blocking (ev.query()); anything not ready stays queued
        and is handled by _consume_prediction at the target layer.
        """
        for li in list(self._pred_pending):
            item = self._pred_pending.get(li)
            if item is None:
                continue
            ev, pinned, n, idx_dev = item
            if not ev.query():
                continue
            self._pred_pending.pop(li, None)
            self._issue_prefetch(li, pinned, n)
            self._early[li] = idx_dev
            self._pref_early += 1

    def _consume_prediction(self, layer_idx: int) -> None:
        """Apply the prediction for layer_idx to the routing mask.

        If the prefetch was already issued early (_drain_early), only the mask
        is updated here. Otherwise fall back to the original order: wait for the
        ids, issue the DMA, then update the mask.
        """
        idx_dev = self._early.pop(layer_idx, None)
        if idx_dev is not None:
            self._update_mask(layer_idx, idx_dev)
            return
        item = self._pred_pending.pop(layer_idx, None)
        if item is None:
            return
        ev, pinned, n, idx_dev = item
        if not ev.query():
            # P1: never block. The ids are not on the host yet, so skip this
            # layer's prefetch and keep the mask from the previous forward (the
            # staging slot still holds that set -- consistent, just not updated).
            # Rare: it only happens while the CPU runs ahead of the GPU.
            self._pref_late += 1
            self._late_skips += 1
            self._pred_pending.pop(layer_idx, None)
            return
        self._issue_prefetch(layer_idx, pinned, n)
        self._update_mask(layer_idx, idx_dev)

    def _weight_buffer_device(self) -> int:
        return self._weight_manager._weight_buffer.device_id

    def prefetch_first_layers(self) -> None:
        if self._weight_manager is not None:
            self._weight_manager.prefetch_first_layers()
            for li in self._moe_layer_indices[:2]:
                self.pending_ready[li] = set(range(self.layout.num_routed_experts))

    # ---------- dynamic routing (called from the MoE block) ----------
    def prepare_layer(
        self, layer_idx: int, router_logits: torch.Tensor
    ) -> StandardTopKOutput:
        """Zero-synchronization dynamic routing for one MoE layer.

        Falls back to per-mode synchronous handling only in HIGH_PRECISION
        mode when fallback fetches are actually needed.
        """
        cfg = self.cfg
        E = self.layout.num_routed_experts
        device = router_logits.device

        # Issue any ready prefetch NOW, i.e. before this layer's fused_moe
        # kernel is launched, so the DMA overlaps this layer's MoE compute
        # instead of starting after it.
        if self._weight_manager is not None:
            self._drain_early()

        # consume the prediction targeting THIS layer (issued at record(l-2)):
        # the event is >=2 layers old so the sync is free, and the subset
        # prefetch gets this layer's attention+router window (~7ms) to land
        # before wait_prefetch(l) — with the previous "consume(l+1)" timing
        # the window collapsed to sub-millisecond and every layer stalled
        # ~7ms in wait_prefetch.
        self._consume_prediction(layer_idx)

        # decode-mode detection: predictor features are prefill-shaped, so on
        # small (decode) batches fall back to the full ready set — DWDP-Full
        # semantics, zero replacement, no quality loss.
        self._decode_mode = router_logits.shape[0] < cfg.decode_threshold

        if self._decode_mode:
            mask = self._decode_mask
            if mask is None or mask.device != device:
                self._decode_mask = torch.ones(
                    E, dtype=torch.bool, device=device
                )
                mask = self._decode_mask
        else:
            mask = self._mask_cache.get(layer_idx)
            if mask is None:
                # first touch of this layer: no prediction consumed yet ->
                # conservative full-ready (no H2D, no host-side set sorting)
                mask = torch.ones(E, dtype=torch.bool, device=device)
                self._mask_cache[layer_idx] = mask

        weights, ids, info = dyn_select(
            router_logits,
            mask,
            k_min=cfg.k_min,
            k_max=cfg.k_max,
            epsilon=cfg.epsilon,
            tau=cfg.tau,
            dummy_expert_id=-1,
        )

        # corrector EWMA update (pure GPU ops, no sync) — prefill only:
        # decode-time mass must not pollute the prefill routing features
        sel_mask = weights > 0
        if not self._decode_mode:
            mass = torch.zeros(E, device=device)
            flat_ids = ids.long().flatten()
            # NOTE: boolean indexing here (flat_ids[valid]) lowered to
            # aten::nonzero, which forces a D2H + cudaStreamSynchronize twice
            # per layer. Instead clamp the index and zero the weight of the
            # dropped slots: scatter_add_ then adds 0.0 to expert 0.
            valid = flat_ids >= 0
            safe_ids = torch.where(valid, flat_ids, torch.zeros_like(flat_ids))
            safe_w = torch.where(
                valid, sel_mask.to(torch.float32).flatten(),
                torch.zeros_like(flat_ids, dtype=torch.float32))
            mass.scatter_add_(0, safe_ids, safe_w)
            if self.corrector is not None:
                self.corrector.update(layer_idx, mass)

            routing_multihot = (mass > 0).to(torch.float32)
            self._last_routing[layer_idx] = routing_multihot

            pending = self._pending_inputs.pop(layer_idx, None)
            self._trndbg_n = getattr(self, "_trndbg_n", 0) + 1
            if self._trndbg_n % 400 == 1:
                logger.info(
                    f"[trndbg] prep l={layer_idx} pending={pending is not None} "
                    f"cal={self.cfg.calibrate} steps={self.heads.train_steps if self.heads else -1}"
                )
            if pending is not None and self.cfg.calibrate and self.heads is not None:
                p_pool, p_routing = pending
                self.heads.submit_sample(layer_idx, p_pool, p_routing, routing_multihot)

            # GPU-side predictor telemetry (no sync; dumped by the stats thread)
            pt = self._ptel
            pt["n"] += 1
            pt["actual_sum"] += routing_multihot.sum()
            pt["hit_sum"] += (mask & (mass > 0)).to(torch.float32).sum()

        if cfg.mode == DynaDWDPMode.HIGH_PRECISION:
            self._maybe_fallback(layer_idx, router_logits, mask, weights, ids)

        return StandardTopKOutput(
            topk_weights=weights,
            topk_ids=ids,
            router_logits=router_logits,
        )

    def _maybe_fallback(
        self, layer_idx, router_logits, mask, weights, ids
    ) -> None:
        """HIGH_PRECISION: synchronously fetch missing experts when the
        missing-token fraction exceeds eta (re-selects afterwards)."""
        cfg = self.cfg
        hit = mask[ids.long()]  # [T, K]
        miss_per_tok = (~hit).to(torch.float64).sum(-1)
        T = float(router_logits.shape[0])
        # single sync point, only in this mode
        if float(miss_per_tok.mean()) <= cfg.eta * cfg.k_max:
            return
        missing = torch.unique(ids[(~hit) & (weights > 0)].long()).tolist()
        fetches = 0
        for e in missing:
            if fetches >= cfg.max_sync_fetches:
                break
            self._weight_manager.ensure_expert_sync(layer_idx, e)
            self.pending_ready.setdefault(layer_idx, set()).add(e)
            fetches += 1
        if fetches:
            E = self.layout.num_routed_experts
            device = router_logits.device
            m = torch.zeros(E, dtype=torch.bool, device=device)
            idx = torch.tensor(
                sorted(self.pending_ready[layer_idx]), dtype=torch.long, device=device
            )
            m[idx] = True
            self._mask_cache[layer_idx] = m
            weights2, ids2, _info = dyn_select(
                router_logits, m, k_min=cfg.k_min, k_max=cfg.k_max,
                epsilon=cfg.epsilon, tau=cfg.tau,
                dummy_expert_id=-1,
            )
            weights.copy_(weights2)
            ids.copy_(ids2)

    def cleanup(self) -> None:
        self.finalize()
        super().cleanup()

    def finalize(self):
        if self.cfg.calibrate and self.heads is not None:
            self.heads.save(self.cfg.head_cache_dir)
            self._save_corrector_snapshot()
            logger.info(
                f"saved lookahead heads ({self.heads.train_steps} steps) "
                f"to {self.cfg.head_cache_dir}"
            )

    def _save_corrector_snapshot(self):
        try:
            import os

            os.makedirs(self.cfg.head_cache_dir, exist_ok=True)
            torch.save(
                self.corrector.state_dict(),
                os.path.join(self.cfg.head_cache_dir, "corrector.pt"),
            )
        except Exception as e:
            logger.warning(f"corrector snapshot save failed: {e}")

    def _load_corrector_snapshot(self):
        import os

        path = os.path.join(self.cfg.head_cache_dir, "corrector.pt")
        if not os.path.exists(path):
            return False
        try:
            sd = torch.load(path, map_location="cpu", weights_only=False)
            self.corrector.load_state_dict(sd)
            logger.info(f"loaded corrector snapshot from {path}")
            return True
        except Exception as e:
            logger.warning(f"corrector snapshot load failed: {e}")
            return False
