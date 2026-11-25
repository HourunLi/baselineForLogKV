"""Diagnostic decomposition for LogKV slot attention — score/value oracle grid.

For ANY bucketing of the tokens into slots ``s`` (each slot covering a token
set ``S_s``), attention factors EXACTLY:

    L_s  = logsumexp_{j in S_s}(z_j)              # slot log-mass, z_j = scale·q·k_j
    V_s  = sum_{j in S_s} softmax_{j in S_s}(z_j)·v_j
    out  = softmax_s(L_s)·V_s   ==  dense attention          (algebraic identity)

The old first-order LogKV path replaces both sides with:

    L_s ≈ scale·q·mean_j(k_j) + λ·log(w_s)        (score side)
    V_s ≈ mean_j(v_j)                              (value side)

This module keeps that old path as ``baseline_1st_order`` and swaps the two
sides in and out independently to build an attribution grid, so a benchmark drop
can be split into "scoring error" vs "read-out error":

    mode          L_s (score)              V_s (value)              answers
    ----------    ---------------------    ---------------------    -------------------
    baseline      production               production               control (== prod)
    baseline_1st_order
                  mean/log-w approx        mean(v)                  old LogKV baseline
    s_oracle      exact logsumexp          mean(v)                  how much is scoring
    v_oracle      mean/log-w approx        exact softmax·v          how much is read-out
    gamma_only    exact logsumexp          mean(v) + scale·Γ_s·q    1st-order value fix
    exact         exact                    exact                    self-check (== dense)
    dense         no bucketing — softmax over every raw token       independent oracle

``exact`` and ``dense`` must agree (two independent routes to the same identity);
``baseline`` must reproduce the current production path bit-for-bit (it delegates
to it). ``baseline_1st_order`` keeps the old mean/log-w + mean(v) path as a
stable control after persisted second-order stats are wired into production.

``gamma_only`` probes whether a CHEAP, on-the-fly linear value correction closes
most of ``s_oracle``'s gap without the full ``v_oracle`` within-slot softmax
read-out: ``V_s(q) ≈ mean(v) + scale·Γ_s·q`` where
``Γ_s = mean_j[(v_j − mean(v))(k_j − mean(k))^T]`` is the within-slot key/value
cross-covariance (recomputed fresh per query from ``k_prefix``/``v_prefix``, NOT
persisted anywhere — this is the delta-method expansion of the exact within-slot
softmax around uniform weights; see the score-side ``Σ_s`` analogue in the design
notes). A large gap between ``s_oracle`` and ``gamma_only`` (gamma_only much less
negative / closer to baseline) says the linear term captures most of what
``s_oracle`` was missing, which is the case FOR implementing the persisted
``Γ_s`` correction in production; a small gap says the interaction needs the
full within-slot distribution, not just its first moment.

Two more instruments beyond the mode grid, both opt-in via ``diag_mode(...)``:

  * ``exact_from_layer``: layers ``>= exact_from_layer`` are forced to ``exact``
    regardless of the active ``mode``, in the SAME real forward pass (errors from
    earlier baseline layers propagate in normally). Sweeping this — see
    ``diag_mode(mode, exact_from_layer=28-N)`` for ``N`` in {0,1,2,4,7,14,21,28} —
    answers whether a benchmark's gap is dominated by cumulative drift from early
    layers (fixing only the tail doesn't help until N is large) or by the last
    layers' own local distortion (a small N recovers most of it). ``by_layer_output``
    is still collected at every layer including the forced-exact tail, so
    ``err_baseline`` at the LAST layer, read across the N sweep, is the signal: it
    is a counterfactual (what baseline WOULD have given at that layer) evaluated on
    whatever hidden state this specific N actually produced, so it shrinks with N
    only through less upstream drift, not because the layer's own behavior changed.
  * ``peak_window_from_end``: restricts D5 peakiness collection (``add_peak``) to
    query positions within the last ``peak_window_from_end`` tokens of the
    sequence (needs ``seq_len`` passed to ``diag_block_attention``). Unfiltered
    peakiness pools every query in the block-prefill indiscriminately, which
    dilutes a benchmark's few truly retrieval-critical query positions (e.g. a
    NIAH question near the prompt tail) against a much larger population of
    ordinary reading positions. ``None`` (default) keeps the old unfiltered
    behavior. Whenever this is set, ``add_peak`` ALSO buckets samples by
    relative position from the sequence end (``add_peak_by_pos`` /
    ``peakiness_by_position`` in ``summary()``) instead of only pooling them
    into one number — this is what makes D5 re-binning legible: a real
    retrieval spike shows up as a sharp curve peaked at (or near) the
    question/answer boundary, whereas a flat curve across the window means the
    window itself is still too wide (or the theory's prediction doesn't hold).
    NOTE: this only reaches query positions inside a fresh PREFILL block (see
    the module-level precondition below) — autoregressive decode is not
    instrumented (the raw un-compacted tokens a decode query would score
    against are already gone by then), so "generation-phase" queries are
    approximated by the prompt's tail window, not literal decode steps. This is
    a reasonable proxy for NIAH (the question tokens themselves are exactly
    where the model must locate the needle) but a weak one for LongBench
    summarization prompts (no natural trailing "question" span) — see the
    research log for the decode-hook extension this would need to close.
  * ``second_order_max_width`` / ``second_order_max_layer``: D1 shows the
    persisted rank-1 Σ_s/Γ_s stats are essentially exact at ``slot_width == 2``
    (a 2-point covariance is exactly rank 1) but degrade sharply and
    monotonically from ``width == 4`` upward, as every ``compact()`` carry
    re-truncates an already-approximate summary back to rank 1. Separately,
    the layer-ablation D2 sweep shows the model's SENSITIVITY to any given
    error is itself layer-dependent (e.g. LongBench's baseline-vs-1st-order
    ROI flips from favorable to net-harmful specifically in the tail layers) —
    and this is NOT explained by width alone, since every layer sees the same
    slot_width distribution for a given sequence (each layer's LogKV cache is
    an independent Fenwick hierarchy over the same token positions). So width
    unreliability and layer sensitivity are two distinct, independently-acting
    axes. ``second_order_scale`` is currently one coupled scalar applied
    uniformly to every slot regardless of either. These two instruments test
    the coarsest possible fix on each axis — hard cutoffs — without touching
    production or CPT training: a slot keeps the normal ``second_order_scale``
    correction only if BOTH ``slot_width <= second_order_max_width`` AND
    ``layer <= second_order_max_layer``; failing either falls back to the
    1st-order (mean/log-w, mean(v)) path, as if ``second_order_scale`` were 0
    just for that slot. Each defaults to ``None`` (no cutoff on that axis);
    setting only one tests that axis alone, setting both tests the joint
    (AND) gate. Neither changes what ``baseline`` returns/propagates — they
    only add one more grid corner, ``err_width_gated`` in ``by_layer_output``,
    computed on the SAME hidden states as ``err_baseline`` and
    ``err_baseline_1st_order`` in the same run (the name is kept from the
    width-only instrument for continuity with already-collected width-sweep
    dumps; it now reflects whichever cutoff(s) are active). A sweep over
    these — width alone, layer alone, then jointly once both show independent
    effect — says how much of the D1/D2 unreliability actually translates
    into output harm, before committing to a finer gate or touching CPT.

``summary()`` reduces the accumulated stats to three tables, labeled with the
mode they were collected under:
  * ``by_level`` — per (layer, slot_width): score side ``logit_mae/logit_rmse``
    (approx vs exact slot logit) + ``intra_slot_logit_var``, and value side
    ``value_mae/value_rel`` (mean-pooled vs exact within-slot read-out). When
    persisted rank-1 stats are supplied it also includes D1 projection errors:
    ``sigma_q_*``, ``score2_*``, ``gamma_q_*`` and ``value2_*``. These are
    measurements of the CURRENT cache state, taken identically under every mode;
    across-mode dumps differ only through hidden-state drift caused by the
    substituted outputs.
  * ``by_layer_output`` — per layer: relative L2 error of the baseline /
    baseline_1st_order / s_oracle / v_oracle / gamma_only outputs vs
    ``exact``, all corners evaluated on the SAME hidden states, so one run
    yields a drift-free D2 attribution.
  * ``peakiness`` — per layer: running max + p99.9 of |scale·q·k| over a
    deterministic sample buffer, with the pool size ``n``.

Scope / preconditions (enforced; also see ``diag_block_attention``):
  * Diagnostics run on a FRESH prefill only. The slot→token span mapping assumes
    every slot covers a contiguous, time-ordered run of the exact prefix tokens
    (``get_attention_state`` invariant). A decode continuation / pending tail
    breaks ``token_count == len(prefix)`` and is rejected.
  * ``mode == "off"`` (default) is fully inert: ``diag_block_attention`` is never
    reached from ``model.py`` unless a ``diag_mode(...)`` context is active, so
    production training/eval keeps its exact numerics and performance.
"""

from __future__ import annotations

import contextlib

import torch

from litgpt.log_kv_cache import CacheAttentionState, append_exact_tokens, log_kv_slot_attention

_MODES = ("off", "baseline", "baseline_1st_order", "s_oracle", "v_oracle", "gamma_only", "exact", "dense")
# Per-layer cap on the deterministic |scale·q·k| sample buffer backing the
# summary()-time p99.9 (see DiagState.add_peak). ~1 MB fp32 per layer.
_PEAK_BUF_CAP = 262_144
# Per-(layer, relative-position) cap for the D5 re-binning buffer (see
# DiagState.add_peak_by_pos). Only active when peak_window_from_end is set, so
# the position axis is bounded by the window width, not the sequence length —
# a much smaller cap than _PEAK_BUF_CAP is enough (worst case
# n_layers * window * cap * 4 bytes; 28 * 128 * 8192 * 4 ≈ 117 MB).
_PEAK_POS_BUF_CAP = 8_192
# Modes whose slot logit is the exact within-slot logsumexp (score oracle).
_L_EXACT = ("s_oracle", "gamma_only", "exact")
# Modes whose read-out is the exact within-slot softmax over v (value oracle).
_V_EXACT = ("v_oracle", "exact")
# Mode whose read-out is the 1st-order linear correction mean(v) + scale·Γ_s·q.
_V_GAMMA = ("gamma_only",)


# ======================================================================
# Global diagnostic state
# ======================================================================


class DiagState:
    """Global singleton holding the active diagnostic mode + accumulated stats.

    ``mode`` gates everything: ``enabled`` is ``mode != "off"`` and the model
    only routes through the diagnostic branch when ``enabled`` is true. Stats
    accumulate across chunks / blocks / layers / samples into ``stats`` (keyed by
    ``(layer, slot_width)``) and ``peak`` (keyed by ``layer``); ``summary()``
    reduces them to a plot-ready dict. Use ``diag_mode(...)`` to flip the mode.
    """

    def __init__(self) -> None:
        self.mode: str = "off"
        self.q_chunk: int = 64
        self.collect: bool = True
        # Layer-ablation sweep (see module docstring): layers >= this are forced
        # to "exact" regardless of ``mode``. None = no override (legacy).
        self.exact_from_layer: int | None = None
        # D5 re-binning (see module docstring): restrict add_peak to the last N
        # tokens of the sequence. None = unfiltered (legacy).
        self.peak_window_from_end: int | None = None
        # Width-gated 2nd-order ablation (see module docstring): slots with
        # slot_width > this use the 1st-order path regardless of mode. None =
        # no cutoff (legacy — every slot uses second_order_scale unconditionally).
        self.second_order_max_width: int | None = None
        # Layer-gated 2nd-order ablation (see module docstring): layers >
        # this use the 1st-order path regardless of mode, ANDed with the
        # width cutoff above. None = no cutoff on this axis.
        self.second_order_max_layer: int | None = None
        # Mode the accumulated stats were collected under. Unlike ``mode`` this is
        # NOT restored when a ``diag_mode`` context exits (mirroring the stats
        # themselves), so a ``summary()`` dump written after the ``with`` block —
        # eval.py writes there — is labeled with the collecting mode, not "off".
        self.stats_mode: str = "off"
        # (layer, slot_width) -> running logit-error / intra-slot-variance /
        # value-readout-error sums.
        self.stats: dict[tuple[int, int], dict[str, float]] = {}
        # layer -> peak |scale·q·k| running max + deterministic sample buffer.
        self.peak: dict[int, dict] = {}
        # (layer, rel_pos_from_end) -> same running max + sample buffer as
        # ``peak``, but bucketed by query position instead of pooled (D5
        # re-binning; only populated while ``peak_window_from_end`` is set).
        self.peak_pos: dict[tuple[int, int], dict] = {}
        # layer -> running squared output error of each grid corner vs ``exact``.
        self.out_stats: dict[int, dict[str, float]] = {}

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def reset_stats(self) -> None:
        self.stats = {}
        self.peak = {}
        self.peak_pos = {}
        self.out_stats = {}

    def add_slot_stats(
        self, layer: int, width: int, err: torch.Tensor, intra_var: torch.Tensor
    ) -> None:
        """Accumulate approx-vs-exact logit error + intra-slot logit variance.

        ``err = L_approx - L_exact`` and ``intra_var = var_{j in slot}(z_j)`` are
        both (B, G, rf, T_c, n_slots); everything is summed so D3's rate-distortion
        curve (variance vs slot width) is exact per (layer, width).
        """
        acc = self._level_acc(layer, width)
        acc["count"] += float(err.numel())
        acc["sum_abs"] += err.abs().sum().item()
        acc["sum_sq"] += err.double().square().sum().item()
        acc["var_sum"] += intra_var.sum().item()

    def add_value_stats(
        self, layer: int, width: int, v_err: torch.Tensor, v_ref: torch.Tensor
    ) -> None:
        """Accumulate read-out error ``‖mean(v) − V_exact(q, slot)‖`` per (query, slot).

        ``v_err`` / ``v_ref`` are (B, G, rf, T_c, n) per-dim RMS norms (L2 ÷ √Dv) of
        the mean-vs-exact read-out gap and of the exact read-out itself. Lands in
        the same (layer, width) cells as ``add_slot_stats``, so D2's score and
        value sides share one rate-distortion table: ``value_mae`` is absolute
        (per-dim RMS units), ``value_rel`` is relative to ``‖V_exact‖``.
        """
        acc = self._level_acc(layer, width)
        acc["v_count"] += float(v_err.numel())
        acc["v_abs"] += v_err.sum().item()
        acc["v_rel"] += (v_err / v_ref.clamp_min(1e-8)).sum().item()

    def add_projection_stats(
        self,
        layer: int,
        width: int,
        sigma_q_err: torch.Tensor,
        sigma_q_ref: torch.Tensor,
        score2_err: torch.Tensor,
        score2_ref: torch.Tensor,
        gamma_q_err: torch.Tensor,
        gamma_q_ref: torch.Tensor,
        value2_err: torch.Tensor,
        value2_ref: torch.Tensor,
    ) -> None:
        """D1: persisted rank-1 stats vs exact projected Sigma/Gamma quantities.

        Matrix Frobenius error is useful but indirect; these are the quantities
        the actual attention formula consumes on the real query distribution.
        """
        acc = self._level_acc(layer, width)
        acc["sigma_q_count"] += float(sigma_q_err.numel())
        acc["sigma_q_abs"] += sigma_q_err.abs().sum().item()
        acc["sigma_q_rel"] += (sigma_q_err.abs() / sigma_q_ref.abs().clamp_min(1e-8)).sum().item()
        acc["score2_abs"] += score2_err.abs().sum().item()
        acc["score2_rel"] += (score2_err.abs() / score2_ref.abs().clamp_min(1e-8)).sum().item()

        acc["gamma_q_count"] += float(gamma_q_err.numel())
        acc["gamma_q_abs"] += gamma_q_err.sum().item()
        acc["gamma_q_rel"] += (gamma_q_err / gamma_q_ref.clamp_min(1e-8)).sum().item()
        acc["value2_abs"] += value2_err.sum().item()
        acc["value2_rel"] += (value2_err / value2_ref.clamp_min(1e-8)).sum().item()

    def add_out_stats(
        self,
        layer: int,
        err_base: float,
        err_base_1st: float,
        err_s: float,
        err_v: float,
        err_gamma: float,
        err_width_gated: float,
        ref: float,
        n_queries: int,
    ) -> None:
        """Accumulate per-layer squared output error of each grid corner vs ``exact``.

        One diagnostic run therefore yields the full D2 attribution per layer
        (``by_layer_output`` in ``summary()``) with all corners evaluated on the
        SAME hidden states — no cross-run drift confound.
        """
        acc = self.out_stats.setdefault(
            int(layer), {
                "base": 0.0, "base_1st": 0.0, "s": 0.0, "v": 0.0,
                "gamma": 0.0, "width_gated": 0.0, "ref": 0.0, "n": 0,
            }
        )
        acc["base"] += err_base
        acc["base_1st"] += err_base_1st
        acc["s"] += err_s
        acc["v"] += err_v
        acc["gamma"] += err_gamma
        acc["width_gated"] += err_width_gated
        acc["ref"] += ref
        acc["n"] += n_queries

    def _level_acc(self, layer: int, width: int) -> dict[str, float]:
        return self.stats.setdefault(
            (int(layer), int(width)),
            {
                "count": 0.0, "sum_abs": 0.0, "sum_sq": 0.0, "var_sum": 0.0,
                "v_count": 0.0, "v_abs": 0.0, "v_rel": 0.0,
                "sigma_q_count": 0.0, "sigma_q_abs": 0.0, "sigma_q_rel": 0.0,
                "score2_abs": 0.0, "score2_rel": 0.0,
                "gamma_q_count": 0.0, "gamma_q_abs": 0.0, "gamma_q_rel": 0.0,
                "value2_abs": 0.0, "value2_rel": 0.0,
            },
        )

    def add_peak(self, layer: int, abs_z: torch.Tensor) -> None:
        """Record D5 peakiness: running max + a sample buffer of |scale·q·k|.

        The buffer holds a deterministic strided subsample per layer (capped at
        ``_PEAK_BUF_CAP``; thinned 2× on overflow), and ``summary()`` takes ONE
        quantile over the whole run. The previous max-of-per-call-p99.9 proxy was
        not comparable across modes — slot paths call once per width-run, dense
        once per prefix chunk, so identical layer-0 populations still produced
        different p99.9 (the spurious exact-vs-dense gaps in early dumps).
        """
        if abs_z.numel() == 0:
            return
        flat = abs_z.detach().reshape(-1).float()
        acc = self.peak.setdefault(
            int(layer), {"r2_max": 0.0, "buf": [], "buf_n": 0, "stride": 1, "n_seen": 0}
        )
        acc["r2_max"] = max(acc["r2_max"], flat.max().item())
        acc["n_seen"] += flat.numel()
        take = flat[:: acc["stride"]]
        if take.numel() > _PEAK_BUF_CAP:  # one huge call: pre-thin it directly
            take = take[:: -(-take.numel() // _PEAK_BUF_CAP)]
        acc["buf"].append(take.cpu())
        acc["buf_n"] += take.numel()
        while acc["buf_n"] > _PEAK_BUF_CAP:
            thinned = torch.cat(acc["buf"])[::2]
            acc["buf"] = [thinned]
            acc["buf_n"] = thinned.numel()
            acc["stride"] *= 2

    def add_peak_by_pos(self, layer: int, rel_pos: torch.Tensor, abs_z: torch.Tensor) -> None:
        """D5 re-binning: record peakiness keyed by relative position from the
        sequence end, instead of pooling the whole window into one number.

        ``abs_z`` has the query axis at dim 3 (matches every ``add_peak`` call
        site: (B,G,rf,t_c,...) from both the slot and dense routes). ``rel_pos``
        is a (t_c,) int tensor, one entry per query row along that axis — 0 is
        the LAST token of the sequence, 1 the second-to-last, etc. Only called
        while ``DIAG.peak_window_from_end`` is set (see module docstring), so
        the number of distinct keys is bounded by the window width, not the
        sequence length. Mirrors ``add_peak``'s running-max + strided sample
        buffer per (layer, rel_pos) cell so ``summary()`` can plot a peakiness
        CURVE over position — the signal D5 actually needs: a real retrieval
        spike is sharp and localized, a bug/dilution artifact is flat.
        """
        if abs_z.numel() == 0:
            return
        t_c = abs_z.size(3)
        for i in range(t_c):
            flat = abs_z.select(3, i).detach().reshape(-1).float()
            if flat.numel() == 0:
                continue
            key = (int(layer), int(rel_pos[i].item()))
            acc = self.peak_pos.setdefault(
                key, {"r2_max": 0.0, "buf": [], "buf_n": 0, "stride": 1, "n_seen": 0}
            )
            acc["r2_max"] = max(acc["r2_max"], flat.max().item())
            acc["n_seen"] += flat.numel()
            take = flat[:: acc["stride"]]
            if take.numel() > _PEAK_POS_BUF_CAP:
                take = take[:: -(-take.numel() // _PEAK_POS_BUF_CAP)]
            acc["buf"].append(take.cpu())
            acc["buf_n"] += take.numel()
            while acc["buf_n"] > _PEAK_POS_BUF_CAP:
                thinned = torch.cat(acc["buf"])[::2]
                acc["buf"] = [thinned]
                acc["buf_n"] = thinned.numel()
                acc["stride"] *= 2

    def summary(self) -> dict:
        """Plot-ready reduction of the accumulated stats (see module docstring)."""
        by_level = []
        for (layer, width), acc in sorted(self.stats.items()):
            n = max(acc["count"], 1.0)
            nv = max(acc.get("v_count", 0.0), 1.0)
            ns = max(acc.get("sigma_q_count", 0.0), 1.0)
            ng = max(acc.get("gamma_q_count", 0.0), 1.0)
            by_level.append(
                {
                    "layer": layer,
                    "slot_width": width,
                    "logit_mae": acc["sum_abs"] / n,
                    "logit_rmse": (acc["sum_sq"] / n) ** 0.5,
                    "intra_slot_logit_var": acc["var_sum"] / n,
                    "value_mae": acc.get("v_abs", 0.0) / nv,
                    "value_rel": acc.get("v_rel", 0.0) / nv,
                    "sigma_q_mae": acc.get("sigma_q_abs", 0.0) / ns,
                    "sigma_q_rel": acc.get("sigma_q_rel", 0.0) / ns,
                    "score2_mae": acc.get("score2_abs", 0.0) / ns,
                    "score2_rel": acc.get("score2_rel", 0.0) / ns,
                    "gamma_q_mae": acc.get("gamma_q_abs", 0.0) / ng,
                    "gamma_q_rel": acc.get("gamma_q_rel", 0.0) / ng,
                    "value2_mae": acc.get("value2_abs", 0.0) / ng,
                    "value2_rel": acc.get("value2_rel", 0.0) / ng,
                }
            )
        by_layer_output = []
        for layer, acc in sorted(self.out_stats.items()):
            ref = max(acc["ref"], 1e-30)
            by_layer_output.append(
                {
                    "layer": layer,
                    "err_baseline": (acc["base"] / ref) ** 0.5,
                    "err_baseline_1st_order": (acc.get("base_1st", 0.0) / ref) ** 0.5,
                    "err_s_oracle": (acc["s"] / ref) ** 0.5,
                    "err_v_oracle": (acc["v"] / ref) ** 0.5,
                    "err_gamma_only": (acc.get("gamma", 0.0) / ref) ** 0.5,
                    "err_width_gated": (acc.get("width_gated", 0.0) / ref) ** 0.5,
                    "n_queries": acc["n"],
                }
            )
        peakiness = []
        for layer, acc in sorted(self.peak.items()):
            buf = torch.cat(acc["buf"]) if acc["buf"] else torch.zeros(1)
            peakiness.append(
                {
                    "layer": layer,
                    "r2_max": acc["r2_max"],
                    "r2_p999": torch.quantile(buf, 0.999).item(),
                    "n": acc["n_seen"],
                }
            )
        peakiness_by_position = []
        for (layer, rel_pos), acc in sorted(self.peak_pos.items()):
            buf = torch.cat(acc["buf"]) if acc["buf"] else torch.zeros(1)
            peakiness_by_position.append(
                {
                    "layer": layer,
                    "rel_pos_from_end": rel_pos,
                    "r2_max": acc["r2_max"],
                    "r2_p999": torch.quantile(buf, 0.999).item(),
                    "r2_mean": buf.mean().item(),
                    "n": acc["n_seen"],
                }
            )
        return {
            "mode": self.stats_mode,
            "by_level": by_level,
            "by_layer_output": by_layer_output,
            "peakiness_by_position": peakiness_by_position,
            "peakiness": peakiness,
        }


DIAG = DiagState()


@contextlib.contextmanager
def diag_mode(
    mode: str,
    q_chunk: int = 64,
    collect: bool = True,
    reset: bool = True,
    exact_from_layer: int | None = None,
    peak_window_from_end: int | None = None,
    second_order_max_width: int | None = None,
    second_order_max_layer: int | None = None,
):
    """Temporarily switch the global diagnostic mode; restore on exit.

    Args:
        mode: one of ``off / baseline / baseline_1st_order / s_oracle /
            v_oracle / gamma_only / exact / dense``.
        q_chunk: query-axis chunk for the memory-heavy oracle paths (they
            materialize a per-token fp32 score; see ``diag_block_attention``).
        collect: gather per-level stats + peakiness while running.
        reset: clear previously accumulated stats on entry (default). Stats are
            deliberately NOT cleared on exit, so ``DIAG.summary()`` can be read
            after the ``with`` block.
        exact_from_layer: layer-ablation sweep — layers ``>= exact_from_layer``
            are forced to ``exact`` regardless of ``mode``. ``None`` disables the
            override (every layer uses ``mode``, the old behavior).
        peak_window_from_end: restrict D5 peakiness (``add_peak``) to query
            positions within the last N tokens of the sequence. ``None`` keeps
            the old unfiltered pooling.
        second_order_max_width: width-gated 2nd-order ablation — slots with
            ``slot_width > second_order_max_width`` use the 1st-order path
            instead of ``second_order_scale``'s correction. ``None`` disables
            the cutoff on this axis (the old behavior).
        second_order_max_layer: layer-gated 2nd-order ablation — layers
            ``> second_order_max_layer`` use the 1st-order path instead of
            ``second_order_scale``'s correction. ANDed with
            ``second_order_max_width`` when both are set. ``None`` disables
            the cutoff on this axis. Either/both add ``err_width_gated`` to
            ``by_layer_output``.
    """
    if mode not in _MODES:
        raise ValueError(f"unknown diag mode {mode!r}; expected one of {_MODES}")
    prev = (
        DIAG.mode, DIAG.q_chunk, DIAG.collect,
        DIAG.exact_from_layer, DIAG.peak_window_from_end,
        DIAG.second_order_max_width, DIAG.second_order_max_layer,
    )
    if reset:
        DIAG.reset_stats()
    DIAG.mode, DIAG.q_chunk, DIAG.collect = mode, q_chunk, collect
    DIAG.exact_from_layer = exact_from_layer
    DIAG.peak_window_from_end = peak_window_from_end
    DIAG.second_order_max_width = second_order_max_width
    DIAG.second_order_max_layer = second_order_max_layer
    if mode != "off":
        # Not restored on exit (mirrors the stats themselves): eval.py dumps
        # summary() after this context closes, and the label must name the mode
        # the stats were collected under — restoring it produced mode="off" in
        # every early dump.
        DIAG.stats_mode = mode
    try:
        yield DIAG
    finally:
        (
            DIAG.mode, DIAG.q_chunk, DIAG.collect,
            DIAG.exact_from_layer, DIAG.peak_window_from_end,
            DIAG.second_order_max_width, DIAG.second_order_max_layer,
        ) = prev


# ======================================================================
# Slot -> token-span mapping
# ======================================================================


def slot_runs(slot_w: torch.Tensor, slot_valid: torch.Tensor | None = None) -> tuple[list[tuple[int, int, int]], int]:
    """Group time-ordered slots into runs of equal width.

    LogKV produces slots in level order (every slot in a hierarchy level shares a
    width — a power of two — and the recent window is all width 1), so the valid
    slot sequence collapses into a few maximal runs of constant width. Each run
    of ``n`` slots of ``width`` covers a contiguous block of ``n * width`` exact
    tokens, so ``reshape(..., n, width)`` recovers the per-slot token spans without
    any scatter. ``slot_valid`` filters the fixed pooled-prefix capacity used by
    the GPU layout; exact suffix slots remain valid.

    Requires ``slot_w`` identical across batch/group; otherwise there is no
    single contiguous slot→token mapping shared by the diagnostic batch.

    Returns:
        runs: list of ``(slot_offset, n_slots, width)`` in time order.
        total_tokens: sum of ``n_slots * width`` over all runs.
    """
    if not torch.all(slot_w == slot_w[0, 0]):
        raise ValueError(
            "slot widths differ across batch/group — diagnostics require one "
            "contiguous slot->token span mapping shared by the diagnostic batch"
        )

    if slot_valid is not None:
        if not torch.all(slot_valid == slot_valid[0, 0]):
            raise ValueError("slot validity differs across batch/group — diagnostics require aligned slots")
        pooled = slot_valid.size(-1)
        valid0 = slot_valid[0, 0]
        slot_w = torch.cat([slot_w[:, :, :pooled][:, :, valid0], slot_w[:, :, pooled:]], dim=2)

    widths_tensor = slot_w[0, 0].round().to(torch.long)
    if widths_tensor.numel() == 0:
        return [], 0

    widths = widths_tensor.tolist()
    runs: list[tuple[int, int, int]] = []
    total = 0
    i, s = 0, len(widths)
    while i < s:
        w = widths[i]
        j = i
        while j < s and widths[j] == w:
            j += 1
        n = j - i
        runs.append((i, n, w))
        total += n * w
        i = j
    return runs, total


def _filter_valid_slot_prefix(
    slot_valid: torch.Tensor | None,
    *tensors: torch.Tensor | None,
) -> tuple[torch.Tensor | None, ...]:
    """Drop invalid pooled-prefix capacity before oracle span math."""
    if slot_valid is None:
        return tensors
    if not torch.all(slot_valid == slot_valid[0, 0]):
        raise ValueError("slot validity differs across batch/group — diagnostics require aligned slots")

    pooled = slot_valid.size(-1)
    valid0 = slot_valid[0, 0]

    def keep(t: torch.Tensor | None) -> torch.Tensor | None:
        if t is None:
            return None
        if t.dim() == 4:
            return torch.cat([t[:, :, :pooled, :][:, :, valid0, :], t[:, :, pooled:, :]], dim=2)
        return torch.cat([t[:, :, :pooled][:, :, valid0], t[:, :, pooled:]], dim=2)

    return tuple(keep(t) for t in tensors)


# ======================================================================
# Attention decompositions
# ======================================================================


def _production_block(
    q: torch.Tensor,
    slot_k: torch.Tensor,
    slot_v: torch.Tensor,
    slot_w: torch.Tensor,
    k_tail: torch.Tensor,
    v_tail: torch.Tensor,
    scale: float,
    lam: float,
    slot_sigma_u: torch.Tensor | None = None,
    slot_sigma2: torch.Tensor | None = None,
    slot_gamma_a: torch.Tensor | None = None,
    slot_gamma_b: torch.Tensor | None = None,
    slot_gamma: torch.Tensor | None = None,
    slot_valid: torch.Tensor | None = None,
    M_s: torch.Tensor | None = None,
    second_order_scale: float = 1.0,
) -> torch.Tensor:
    """Exactly the production block-prefill call — the ``baseline`` output.

    Delegates to ``append_exact_tokens`` + ``log_kv_slot_attention`` with the same
    argument order the model uses, so ``baseline`` reproduces production bit-for-bit
    (the zero-intrusion guarantee).
    """
    state = append_exact_tokens(
        CacheAttentionState(
            slot_k=slot_k,
            slot_v=slot_v,
            slot_w=slot_w,
            slot_valid=slot_valid,
            M_s=M_s,
            slot_sigma_u=slot_sigma_u,
            slot_sigma2=slot_sigma2,
            slot_gamma_a=slot_gamma_a,
            slot_gamma_b=slot_gamma_b,
            slot_gamma=slot_gamma,
        ),
        k_tail,
        v_tail,
    )
    if slot_sigma_u is None:
        return log_kv_slot_attention(
            q, state.slot_k, state.slot_v, state.slot_w,
            scale=scale, lam=lam, causal_tail=k_tail.size(2),
            slot_M=state.M_s,
            slot_valid=state.slot_valid,
            second_order_scale=second_order_scale,
        )
    return log_kv_slot_attention(
        q,
        state.slot_k,
        state.slot_v,
        state.slot_w,
        scale=scale,
        lam=lam,
        causal_tail=k_tail.size(2),
        slot_M=state.M_s,
        slot_valid=state.slot_valid,
        slot_sigma_u=state.slot_sigma_u,
        slot_sigma2=state.slot_sigma2,
        slot_gamma_a=state.slot_gamma_a,
        slot_gamma_b=state.slot_gamma_b,
        slot_gamma=state.slot_gamma,
        second_order_scale=second_order_scale,
    )


def _peak_local_slice(t_c: int, global_c0: int, seq_len: int | None) -> slice | None:
    """Local [lo, t_c) slice restricting a query chunk to ``DIAG.peak_window_from_end``.

    ``global_c0`` is the sequence position of local index 0 in this chunk
    (``q_offset + c0``); ``seq_len`` is the CURRENT example's total prefill
    length (varies per call, so it is threaded through as an argument, not
    stored on ``DIAG`` — see ``diag_block_attention``'s ``seq_len`` param).
    Returns ``None`` if the chunk has no overlap with the window (skip
    ``add_peak`` entirely) or ``slice(0, t_c)`` unfiltered when no window /
    no ``seq_len`` is given (legacy behavior).
    """
    window = DIAG.peak_window_from_end
    if window is None or seq_len is None:
        return slice(0, t_c)
    lo_global = seq_len - window
    lo = max(0, lo_global - global_c0)
    if lo >= t_c:
        return None
    return slice(lo, t_c)


def _peak_rel_pos(peak_sl: slice, global_c0: int, seq_len: int) -> torch.Tensor:
    """Relative-from-end position (0 = last token) for each query kept by ``peak_sl``.

    ``peak_sl`` is the local ``[lo, t_c)`` slice ``_peak_local_slice`` returned
    for this chunk; ``global_c0`` is this chunk's sequence offset
    (``q_offset + c0``). Only called when both ``DIAG.peak_window_from_end``
    and ``seq_len`` are set (see the call sites), so the result never needs to
    handle the unfiltered-legacy case.
    """
    local_idx = torch.arange(peak_sl.start, peak_sl.stop)
    global_pos = global_c0 + local_idx
    return seq_len - 1 - global_pos


def _causal_tail_mask(c0: int, t_c: int, blk: int, device: torch.device) -> torch.Tensor:
    """(T_c, blk) bool, True where query at block-offset c0+i must NOT see tail j.

    The in-flight block is causal against itself: query i sees tail token j iff
    ``j <= i`` (query i == tail token i).
    """
    q_pos = torch.arange(c0, c0 + t_c, device=device)
    j_pos = torch.arange(blk, device=device)
    return j_pos[None, :] > q_pos[:, None]


def _diag_slot_core(
    mode: str,
    q: torch.Tensor,          # (B, nh, T_q, D)
    k_prefix: torch.Tensor,   # (B, G, N, D) exact tokens the slots cover
    v_prefix: torch.Tensor,   # (B, G, N, Dv)
    slot_k: torch.Tensor,     # (B, G, S, D) frozen slot state
    slot_v: torch.Tensor,     # (B, G, S, Dv)
    slot_w: torch.Tensor,     # (B, G, S)
    k_tail: torch.Tensor,     # (B, G, blk, D) causal in-flight block
    v_tail: torch.Tensor,     # (B, G, blk, Dv)
    runs: list[tuple[int, int, int]],
    scale: float,
    lam: float,
    layer: int,
    q_offset: int = 0,
    seq_len: int | None = None,
    slot_sigma_u: torch.Tensor | None = None,
    slot_sigma2: torch.Tensor | None = None,
    slot_gamma_a: torch.Tensor | None = None,
    slot_gamma_b: torch.Tensor | None = None,
    slot_gamma: torch.Tensor | None = None,
    second_order_scale: float = 1.0,
) -> torch.Tensor:
    """Slot-bucketed attention with per-side (score/value) oracle swaps + stats.

    Handles ``baseline`` (current production), ``baseline_1st_order`` (old
    mean/log-w path), ``s_oracle``, ``v_oracle``, ``gamma_only`` and ``exact``.
    Everything runs in fp32; the
    query axis is chunked (``DIAG.q_chunk``) because the oracle sides
    materialize a per-token (B, G, rf, T_c, N) score, unlike the O(S) production
    path. ``q_offset``/``seq_len`` are only used to restrict D5 peakiness to
    ``DIAG.peak_window_from_end`` (see module docstring); they do not affect the
    returned output.

    With ``DIAG.collect`` the within-slot softmax + token values are retained per
    chunk and the whole diagnostic grid is evaluated (value-side stats +
    drift-free per-layer output attribution), so every collecting run —
    including ``baseline`` — carries the value-oracle memory envelope. The
    returned output for the active mode is built from the same expressions as the
    stats-off path, so enabling ``collect`` never changes what the model sees.
    """
    B, nh, T_q, D = q.shape
    G = slot_k.size(1)
    rf = nh // G
    Dv = slot_v.size(-1)
    S = slot_k.size(2)
    blk = k_tail.size(2)
    collect = DIAG.collect
    l_exact = mode in _L_EXACT
    v_exact = mode in _V_EXACT
    v_gamma = mode in _V_GAMMA
    has_rank1_stats = slot_sigma_u is not None
    if has_rank1_stats and any(x is None for x in (slot_sigma2, slot_gamma_a, slot_gamma_b, slot_gamma)):
        raise ValueError("_diag_slot_core() requires either all rank-1 stats or none")
    use_rank1_stats = has_rank1_stats and second_order_scale != 0.0
    # Layer half of the joint width x layer gate (see module docstring); the
    # width half is checked per-run below since it varies within one call.
    layer_ok = DIAG.second_order_max_layer is None or layer <= DIAG.second_order_max_layer

    qf = q.float()
    kp = k_prefix.float()
    vp = v_prefix.float()
    skf = slot_k.float()
    svf = slot_v.float()
    kt = k_tail.float()
    vt = v_tail.float()
    logw = lam * slot_w.float().log()  # (B, G, S); w=1 slots contribute 0
    ssuf = slot_sigma_u.float() if has_rank1_stats else None
    ss2f = slot_sigma2.float() if has_rank1_stats else None
    sgaf = slot_gamma_a.float() if has_rank1_stats else None
    sgbf = slot_gamma_b.float() if has_rank1_stats else None
    sgf = slot_gamma.float() if has_rank1_stats else None

    outs: list[torch.Tensor] = []
    q_chunk = max(1, DIAG.q_chunk)
    for c0 in range(0, T_q, q_chunk):
        c1 = min(T_q, c0 + q_chunk)
        t_c = c1 - c0
        qg = qf[:, :, c0:c1, :].reshape(B, G, rf, t_c, D)
        peak_sl = _peak_local_slice(t_c, q_offset + c0, seq_len)

        # Per-run slot logits, both sides. Under ``collect`` both sides plus the
        # within-slot softmax/values are always materialized (the D2 grid needs
        # all grid corners); with stats off, only what the active mode returns,
        # so a stats-off oracle run stays as lean as before.
        l_ex_parts: list[torch.Tensor | None] = []
        l_ap_parts: list[torch.Tensor | None] = []
        l_prod_parts: list[torch.Tensor | None] = []
        l_width_gated_parts: list[torch.Tensor | None] = []  # per-run width-gated score
        w_keep: list[torch.Tensor | None] = []        # per-run within-slot softmax(z)
        v_keep: list[torch.Tensor | None] = []         # per-run token values
        v_gamma_keep: list[torch.Tensor | None] = []   # per-run mean(v)+scale·Γ_s·q
        v_prod_keep: list[torch.Tensor | None] = []    # per-run persisted-Gamma read-out
        v_width_gated_keep: list[torch.Tensor | None] = []  # per-run width-gated read-out
        keep_tokens = v_exact or v_gamma or collect     # exact-V / gamma-V need raw v
        tok = 0
        for soff, n, width in runs:
            # D1/D2-motivated joint ablation (see module docstring): a slot
            # keeps second_order_scale only if BOTH its width and this call's
            # layer pass their respective cutoffs; either None => that axis
            # imposes no cutoff, so with both None eff_scale ==
            # second_order_scale everywhere and err_width_gated == err_baseline.
            width_ok = DIAG.second_order_max_width is None or width <= DIAG.second_order_max_width
            eff_scale = second_order_scale if (layer_ok and width_ok) else 0.0
            kr = kp[:, :, tok:tok + n * width, :].reshape(B, G, n, width, D)
            # z is the per-token score; every reachable mode needs it (exact-L
            # logsumexp, exact-V within-softmax, Γ_s, or stats), so unconditional.
            z = scale * torch.einsum("bgrtd,bgnwd->bgrtnw", qg, kr)  # (B,G,rf,t_c,n,width)
            l_exact_run = torch.logsumexp(z, dim=-1) if (l_exact or collect) else None
            if not l_exact or collect:
                skr = skf[:, :, soff:soff + n, :]                   # (B,G,n,D)
                l_approx_run = (
                    scale * torch.einsum("bgrtd,bgnd->bgrtn", qg, skr)
                    + logw[:, :, None, None, soff:soff + n]
                )
            else:
                l_approx_run = None
            l_ex_parts.append(l_exact_run)
            l_ap_parts.append(l_approx_run)
            sigma_q_persist = score2_persist = None
            if has_rank1_stats and l_approx_run is not None:
                sigma_u_run = ssuf[:, :, soff:soff + n, :]
                sigma2_run = ss2f[:, :, soff:soff + n]
                sigma_dot = torch.einsum("bgrtd,bgnd->bgrtn", qg, sigma_u_run)
                sigma_q_persist = sigma_dot.square() * sigma2_run[:, :, None, None, :]
                score2_persist = 0.5 * scale * scale * sigma_q_persist
                l_prod_parts.append(
                    l_approx_run + second_order_scale * score2_persist
                    if use_rank1_stats
                    else l_approx_run
                )
                l_width_gated_parts.append(
                    l_approx_run + eff_scale * score2_persist
                    if eff_scale != 0.0
                    else l_approx_run
                )
            else:
                l_prod_parts.append(l_approx_run)
                l_width_gated_parts.append(l_approx_run)

            within = torch.softmax(z, dim=-1) if keep_tokens else None
            vr = (
                vp[:, :, tok:tok + n * width, :].reshape(B, G, n, width, Dv)
                if keep_tokens
                else None
            )
            w_keep.append(within)
            v_keep.append(vr)

            if v_gamma or collect:
                # Γ_s = mean_j[(v_j-v̄_s)(k_j-k̄_s)^T], recomputed fresh from the raw
                # run tokens (NOT persisted) — the delta-method / 1st-order Taylor
                # expansion of the exact within-slot softmax read-out around
                # uniform weights (see module docstring). skr/svr are the STORED
                # slot means (== mean_j k_j / mean_j v_j by construction of
                # compact()), so no separate mean pass over kr/vr is needed.
                skr_g = skf[:, :, soff:soff + n, :]                   # (B,G,n,D)
                svr_g = svf[:, :, soff:soff + n, :]                   # (B,G,n,Dv)
                k_c = kr - skr_g.unsqueeze(3)                         # (B,G,n,width,D)
                v_c = vr - svr_g.unsqueeze(3)                         # (B,G,n,width,Dv)
                gamma = torch.einsum("bgnwc,bgnwd->bgncd", v_c, k_c) / width  # (B,G,n,Dv,D)
                corr = scale * torch.einsum("bgncd,bgrtd->bgrtnc", gamma, qg)  # (B,G,rf,t_c,n,Dv)
                v_gamma_run = svr_g[:, :, None, None, :, :] + corr    # (B,G,rf,t_c,n,Dv)
            else:
                v_gamma_run = None
            v_gamma_keep.append(v_gamma_run)
            if has_rank1_stats and (v_gamma or collect):
                gamma_a_run = sgaf[:, :, soff:soff + n, :]
                gamma_b_run = sgbf[:, :, soff:soff + n, :]
                gamma_run = sgf[:, :, soff:soff + n]
                gamma_dot = torch.einsum("bgrtd,bgnd->bgrtn", qg, gamma_a_run)
                gamma_q_persist = (
                    gamma_dot.unsqueeze(-1)
                    * gamma_run[:, :, None, None, :, None]
                    * gamma_b_run[:, :, None, None, :, :]
                )
                corr_persist = scale * gamma_q_persist
                v_prod_keep.append(
                    svr_g[:, :, None, None, :, :] + second_order_scale * corr_persist
                    if use_rank1_stats
                    else svr_g[:, :, None, None, :, :]
                )
                v_width_gated_keep.append(
                    svr_g[:, :, None, None, :, :] + eff_scale * corr_persist
                    if eff_scale != 0.0
                    else svr_g[:, :, None, None, :, :]
                )
            else:
                gamma_q_persist = corr_persist = None
                v_prod_keep.append(None)
                v_width_gated_keep.append(None)

            if collect:
                qk_centered = torch.einsum("bgrtd,bgnwd->bgrtnw", qg, k_c)
                sigma_q_exact = qk_centered.square().mean(dim=-1)
                score2_exact = 0.5 * scale * scale * sigma_q_exact
                DIAG.add_slot_stats(
                    layer, width, l_approx_run - l_exact_run, z.var(dim=-1, unbiased=False)
                )
                if peak_sl is not None:
                    z_abs = z[:, :, :, peak_sl].abs()
                    DIAG.add_peak(layer, z_abs)
                    if DIAG.peak_window_from_end is not None and seq_len is not None:
                        DIAG.add_peak_by_pos(
                            layer, _peak_rel_pos(peak_sl, q_offset + c0, seq_len), z_abs
                        )
                # Value side of the D2 grid: what a width-w mean-pooled read-out
                # actually loses, ‖mean(v) − softmax_within(z)·v‖ per (query, slot).
                v_ex_run = torch.einsum("bgrtnw,bgnwc->bgrtnc", within, vr)
                diff = v_ex_run - svf[:, :, None, None, soff:soff + n, :]
                dv_sqrt = Dv ** 0.5
                DIAG.add_value_stats(
                    layer, width, diff.norm(dim=-1) / dv_sqrt, v_ex_run.norm(dim=-1) / dv_sqrt
                )
                if has_rank1_stats:
                    gamma_q_exact = torch.einsum("bgrtnw,bgnwc->bgrtnc", qk_centered, v_c) / width
                    gamma_q_err = (gamma_q_persist - gamma_q_exact).norm(dim=-1) / dv_sqrt
                    gamma_q_ref = gamma_q_exact.norm(dim=-1) / dv_sqrt
                    value2_err = (corr_persist - corr).norm(dim=-1) / dv_sqrt
                    value2_ref = corr.norm(dim=-1) / dv_sqrt
                    DIAG.add_projection_stats(
                        layer,
                        width,
                        sigma_q_persist - sigma_q_exact,
                        sigma_q_exact,
                        score2_persist - score2_exact,
                        score2_exact,
                        gamma_q_err,
                        gamma_q_ref,
                        value2_err,
                        value2_ref,
                    )
            tok += n * width

        l_tail = scale * torch.einsum("bgrtd,bgsd->bgrts", qg, kt)  # (B,G,rf,t_c,blk)
        l_tail = l_tail.masked_fill(
            _causal_tail_mask(c0, t_c, blk, q.device).view(1, 1, 1, t_c, blk), float("-inf")
        )

        def _mix(l_parts: list[torch.Tensor], v_source: str) -> torch.Tensor:
            """Softmax over [slot logits | causal tail] + the chosen read-out.

            ``v_source``: "mean" (old first-order, p·slot_v) / "exact"
            (within-slot softmax over raw v — needs w_keep/v_keep) / "gamma"
            (exact on-the-fly Γ correction — needs v_gamma_keep) / "persisted"
            (current production persisted Γ correction — needs v_prod_keep).
            """
            p = torch.softmax(torch.cat(l_parts + [l_tail], dim=-1), dim=-1)
            p_prefix = p[..., :S]
            p_tail = p[..., S:]
            if v_source == "exact":
                o = qg.new_zeros(B, G, rf, t_c, Dv)
                for (soff, n, _width), within, vr in zip(runs, w_keep, v_keep):
                    tw = p_prefix[..., soff:soff + n].unsqueeze(-1) * within
                    o = o + torch.einsum("bgrtnw,bgnwc->bgrtc", tw, vr)
            elif v_source == "gamma":
                o = qg.new_zeros(B, G, rf, t_c, Dv)
                for (soff, n, _width), vg in zip(runs, v_gamma_keep):
                    o = o + torch.einsum(
                        "bgrts,bgrtsc->bgrtc", p_prefix[..., soff:soff + n], vg
                    )
            elif v_source == "persisted":
                o = qg.new_zeros(B, G, rf, t_c, Dv)
                for (soff, n, _width), vg in zip(runs, v_prod_keep):
                    o = o + torch.einsum(
                        "bgrts,bgrtsc->bgrtc", p_prefix[..., soff:soff + n], vg
                    )
            elif v_source == "persisted_gated":
                o = qg.new_zeros(B, G, rf, t_c, Dv)
                for (soff, n, _width), vg in zip(runs, v_width_gated_keep):
                    o = o + torch.einsum(
                        "bgrts,bgrtsc->bgrtc", p_prefix[..., soff:soff + n], vg
                    )
            else:
                o = torch.einsum("bgrts,bgsc->bgrtc", p_prefix, svf)  # p·mean(v) == p·slot_v
            return o + torch.einsum("bgrts,bgsc->bgrtc", p_tail, vt)

        if collect:
            # All grid corners on the SAME hidden states — drift-free per-layer
            # D2 attribution from a single run (see add_out_stats).
            out_exact_c = _mix(l_ex_parts, "exact")
            out_base_1st_c = _mix(l_ap_parts, "mean")
            out_base_c = _mix(l_prod_parts, "persisted") if use_rank1_stats else out_base_1st_c
            out_s_c = _mix(l_ex_parts, "mean")
            out_v_c = _mix(l_ap_parts, "exact")
            out_gamma_c = _mix(l_ex_parts, "gamma")
            out_width_gated_c = (
                _mix(l_width_gated_parts, "persisted_gated") if use_rank1_stats else out_base_1st_c
            )
            DIAG.add_out_stats(
                layer,
                (out_base_c - out_exact_c).square().sum().item(),
                (out_base_1st_c - out_exact_c).square().sum().item(),
                (out_s_c - out_exact_c).square().sum().item(),
                (out_v_c - out_exact_c).square().sum().item(),
                (out_gamma_c - out_exact_c).square().sum().item(),
                (out_width_gated_c - out_exact_c).square().sum().item(),
                out_exact_c.square().sum().item(),
                B * nh * t_c,
            )
            out = (
                out_exact_c if (l_exact and v_exact)
                else out_gamma_c if v_gamma
                else out_s_c if l_exact
                else out_v_c if v_exact
                else out_base_1st_c if mode == "baseline_1st_order"
                else out_base_c
            )
        else:
            v_src = "gamma" if v_gamma else "exact" if v_exact else "mean"
            if mode == "baseline":
                out = _mix(l_prod_parts, "persisted") if use_rank1_stats else _mix(l_ap_parts, "mean")
            elif mode == "baseline_1st_order":
                out = _mix(l_ap_parts, "mean")
            else:
                out = _mix(l_ex_parts if l_exact else l_ap_parts, v_src)
        outs.append(out.reshape(B, nh, t_c, Dv))

    return torch.cat(outs, dim=2).to(q.dtype)


def _diag_dense(
    q: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    k_tail: torch.Tensor,
    v_tail: torch.Tensor,
    scale: float,
    layer: int,
    q_offset: int = 0,
    seq_len: int | None = None,
) -> torch.Tensor:
    """Dense causal attention over every raw token — independent of any bucketing.

    Prefix fully visible (every prefix token precedes the block), in-flight block
    causal against itself. This is the ground-truth oracle that ``exact`` must
    match, computed by a completely separate route (flat softmax, no slots).
    ``q_offset``/``seq_len`` only restrict D5 peakiness (``DIAG.peak_window_from_end``,
    see module docstring); the returned output is unaffected.
    """
    B, nh, T_q, D = q.shape
    G = k_prefix.size(1)
    rf = nh // G
    Dv = v_prefix.size(-1)
    N = k_prefix.size(2)
    blk = k_tail.size(2)
    collect = DIAG.collect

    qf = q.float()
    kp = k_prefix.float()
    vp = v_prefix.float()
    kt = k_tail.float()
    vt = v_tail.float()

    outs: list[torch.Tensor] = []
    q_chunk = max(1, DIAG.q_chunk)
    for c0 in range(0, T_q, q_chunk):
        c1 = min(T_q, c0 + q_chunk)
        t_c = c1 - c0
        qg = qf[:, :, c0:c1, :].reshape(B, G, rf, t_c, D)
        peak_sl = _peak_local_slice(t_c, q_offset + c0, seq_len)

        z_pre = scale * torch.einsum("bgrtd,bgnd->bgrtn", qg, kp)   # (B,G,rf,t_c,N)
        z_tail = scale * torch.einsum("bgrtd,bgsd->bgrts", qg, kt)  # (B,G,rf,t_c,blk)
        z_tail = z_tail.masked_fill(
            _causal_tail_mask(c0, t_c, blk, q.device).view(1, 1, 1, t_c, blk), float("-inf")
        )
        if collect and peak_sl is not None:
            z_abs = z_pre[:, :, :, peak_sl].abs()
            DIAG.add_peak(layer, z_abs)
            if DIAG.peak_window_from_end is not None and seq_len is not None:
                DIAG.add_peak_by_pos(
                    layer, _peak_rel_pos(peak_sl, q_offset + c0, seq_len), z_abs
                )

        p = torch.softmax(torch.cat([z_pre, z_tail], dim=-1), dim=-1)
        out = torch.einsum("bgrtn,bgnc->bgrtc", p[..., :N], vp)
        out = out + torch.einsum("bgrts,bgsc->bgrtc", p[..., N:], vt)
        outs.append(out.reshape(B, nh, t_c, Dv))

    return torch.cat(outs, dim=2).to(q.dtype)


def diag_block_attention(
    q: torch.Tensor,          # (B, nh, T_q, D) current-block queries, post-RoPE
    k_prefix: torch.Tensor,   # (B, G, N, D) exact tokens the frozen slots cover
    v_prefix: torch.Tensor,   # (B, G, N, Dv)
    slot_k: torch.Tensor,     # (B, G, S, D) frozen slot state (get_attention_state)
    slot_v: torch.Tensor,     # (B, G, S, Dv)
    slot_w: torch.Tensor,     # (B, G, S)
    k_tail: torch.Tensor,     # (B, G, T_q, D) in-flight block, causal vs queries
    v_tail: torch.Tensor,     # (B, G, T_q, Dv)
    scale: float,
    lam: float = 1.0,
    layer: int = 0,
    q_offset: int = 0,
    seq_len: int | None = None,
    slot_sigma_u: torch.Tensor | None = None,
    slot_sigma2: torch.Tensor | None = None,
    slot_gamma_a: torch.Tensor | None = None,
    slot_gamma_b: torch.Tensor | None = None,
    slot_gamma: torch.Tensor | None = None,
    slot_valid: torch.Tensor | None = None,
    M_s: torch.Tensor | None = None,
    second_order_scale: float = 1.0,
) -> torch.Tensor:
    """Diagnostic replacement for one production block-prefill attention step.

    Parallels ``log_kv_slot_attention`` but additionally receives the EXACT prefix
    tokens (``k_prefix``/``v_prefix``) the frozen slots cover, so the score and
    value sides can each be swapped between the old first-order approximation,
    the current production path and the exact within-slot computation (see
    module docstring for the mode grid).

    ``q_offset`` (this block's starting position in the sequence) and ``seq_len``
    (the sequence's total prefill length) are passed straight through to
    ``DIAG.peak_window_from_end`` filtering — see module docstring — and are not
    otherwise used; callers that never set that field can leave them at their
    defaults.

    Preconditions (the caller in ``model.py`` guards them; do NOT relax):
      * fresh prefill only — ``k_prefix`` must equal the tokens the slots cover,
        i.e. ``cache.token_count == start`` with no decode continuation / pending
        tail. Not valid during decode.

    Returns (B, nh, T_q, Dv), matching the production output shape.
    """
    mode = DIAG.mode
    # Layer-ablation sweep (see module docstring): tail layers forced to exact
    # regardless of ``mode``, in the same real forward pass as everything below.
    if DIAG.exact_from_layer is not None and layer >= DIAG.exact_from_layer:
        mode = "exact"

    has_rank1_stats = slot_sigma_u is not None
    if has_rank1_stats and any(x is None for x in (slot_sigma2, slot_gamma_a, slot_gamma_b, slot_gamma)):
        raise ValueError("diag_block_attention() requires either all rank-1 stats or none")

    if mode in ("off", "baseline", "baseline_1st_order"):
        # off: defensive (never reached with a live diag context). baseline:
        # bit-exact production, optionally with a stats-only oracle pass.
        prod_second_order_scale = 0.0 if mode == "baseline_1st_order" else second_order_scale
        out = _production_block(
            q=q,
            slot_k=slot_k,
            slot_v=slot_v,
            slot_w=slot_w,
            k_tail=k_tail,
            v_tail=v_tail,
            scale=scale,
            lam=lam,
            slot_sigma_u=slot_sigma_u,
            slot_sigma2=slot_sigma2,
            slot_gamma_a=slot_gamma_a,
            slot_gamma_b=slot_gamma_b,
            slot_gamma=slot_gamma,
            slot_valid=slot_valid,
            M_s=M_s,
            second_order_scale=prod_second_order_scale,
        )
        if mode in ("baseline", "baseline_1st_order") and DIAG.collect:
            (
                diag_slot_k,
                diag_slot_v,
                diag_slot_w,
                diag_slot_sigma_u,
                diag_slot_sigma2,
                diag_slot_gamma_a,
                diag_slot_gamma_b,
                diag_slot_gamma,
            ) = _filter_valid_slot_prefix(
                slot_valid,
                slot_k,
                slot_v,
                slot_w,
                slot_sigma_u,
                slot_sigma2,
                slot_gamma_a,
                slot_gamma_b,
                slot_gamma,
            )
            runs, total = slot_runs(diag_slot_w)
            if total != k_prefix.size(2):
                raise ValueError(
                    f"slot span ({total}) != prefix length ({k_prefix.size(2)}): "
                    "diagnostics require a fresh prefill with contiguous slots"
                )
            _diag_slot_core(
                mode=mode,
                q=q,
                k_prefix=k_prefix,
                v_prefix=v_prefix,
                slot_k=diag_slot_k,
                slot_v=diag_slot_v,
                slot_w=diag_slot_w,
                k_tail=k_tail,
                v_tail=v_tail,
                runs=runs,
                scale=scale,
                lam=lam,
                layer=layer,
                q_offset=q_offset,
                seq_len=seq_len,
                slot_sigma_u=diag_slot_sigma_u,
                slot_sigma2=diag_slot_sigma2,
                slot_gamma_a=diag_slot_gamma_a,
                slot_gamma_b=diag_slot_gamma_b,
                slot_gamma=diag_slot_gamma,
                second_order_scale=second_order_scale,
            )
        return out

    if mode == "dense":
        return _diag_dense(q, k_prefix, v_prefix, k_tail, v_tail, scale, layer, q_offset, seq_len)

    (
        diag_slot_k,
        diag_slot_v,
        diag_slot_w,
        diag_slot_sigma_u,
        diag_slot_sigma2,
        diag_slot_gamma_a,
        diag_slot_gamma_b,
        diag_slot_gamma,
    ) = _filter_valid_slot_prefix(
        slot_valid,
        slot_k,
        slot_v,
        slot_w,
        slot_sigma_u,
        slot_sigma2,
        slot_gamma_a,
        slot_gamma_b,
        slot_gamma,
    )
    runs, total = slot_runs(diag_slot_w)
    if total != k_prefix.size(2):
        raise ValueError(
            f"slot span ({total}) != prefix length ({k_prefix.size(2)}): "
            "diagnostics require a fresh prefill with contiguous slots"
        )
    return _diag_slot_core(
        mode=mode,
        q=q,
        k_prefix=k_prefix,
        v_prefix=v_prefix,
        slot_k=diag_slot_k,
        slot_v=diag_slot_v,
        slot_w=diag_slot_w,
        k_tail=k_tail,
        v_tail=v_tail,
        runs=runs,
        scale=scale,
        lam=lam,
        layer=layer,
        q_offset=q_offset,
        seq_len=seq_len,
        slot_sigma_u=diag_slot_sigma_u,
        slot_sigma2=diag_slot_sigma2,
        slot_gamma_a=diag_slot_gamma_a,
        slot_gamma_b=diag_slot_gamma_b,
        slot_gamma=diag_slot_gamma,
        second_order_scale=second_order_scale,
    )


# ======================================================================
# D4 (offline, standalone): adjacency grouping vs oracle clustering
# ======================================================================


def _kmeans(x: torch.Tensor, k: int, n_iters: int, seed: int) -> torch.Tensor:
    """Plain Lloyd's-algorithm k-means. Returns (k,) int64 cluster assignment.

    ``x`` is (N, D). Uses randomized init (k-means++-lite: k random distinct
    points) — fine here since this is read-only offline analysis, never on the
    backward-replay path that requires determinism (see log_kv_cache.py).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    N = x.size(0)
    init_idx = torch.randperm(N, generator=g)[:k]
    centers = x[init_idx].clone()
    assign = torch.zeros(N, dtype=torch.long)
    for _ in range(n_iters):
        d2 = (x.unsqueeze(1) - centers.unsqueeze(0)).square().sum(-1)  # (N, k)
        new_assign = d2.argmin(dim=1)
        if torch.equal(new_assign, assign) and _ > 0:
            break
        assign = new_assign
        for c in range(k):
            members = x[assign == c]
            if members.numel() > 0:
                centers[c] = members.mean(dim=0)
    return assign


def oracle_cluster_gap(
    k_prefix: torch.Tensor,   # (G, N, D) raw prefix keys, ONE batch item
    v_prefix: torch.Tensor,   # (G, N, Dv)
    q: torch.Tensor,          # (G, rf, T_q, D) queries (already GQA-folded)
    n_slots: int,             # budget: cluster count, should match the current
                               # adjacency S for this prefix so the comparison is
                               # apples-to-apples on the SAME compression ratio
    scale: float,
    n_iters: int = 25,
    seed: int = 0,
) -> dict:
    """D4: same-budget position-agnostic k-means clustering vs strict time
    adjacency, measured with the SAME estimator (mean-pool + logsumexp-vs-approx
    logit error) on both sides — isolates whether the ACCURACY cost comes from
    the adjacency CONSTRAINT (grouping only contiguous-in-time tokens) or from
    the mean-pooling estimator itself (see module docstring's Σ_s/Γ_s notes and
    the research log's D4 design).

    Standalone / offline only — NOT wired into ``diag_mode``'s live mode grid,
    run it from a small script on a handful of cached (k_prefix, v_prefix, q)
    triples (e.g. captured via a one-off hook, or from a ``diag_mode("exact")``
    run's inputs). Cost is O(N·k) per k-means iteration — fine for a handful of
    examples, not for a full benchmark sweep.

    Returns a dict with matched-budget error stats for both groupings:
    ``{"adjacency": {...}, "oracle_cluster": {...}}``, each with ``logit_mae``
    and ``value_rel`` pooled over all clusters/slots and all queries — directly
    comparable to the corresponding aggregate of ``by_level`` for this layer.
    """
    G, N, _D = k_prefix.shape
    Dv = v_prefix.size(-1)
    rf, T_q = q.size(1), q.size(2)
    kp, vp, qf = k_prefix.float(), v_prefix.float(), q.float()

    def _adjacency_grouping(n: int) -> torch.Tensor:
        # Contiguous runs of near-equal size, matching how compact() would
        # partition N tokens into n slots — same spirit as slot_runs, but built
        # from a flat token count rather than an existing cache's slot_w. Shared
        # across every group (time order doesn't depend on G), unlike the
        # per-group k-means assignment below.
        base, rem = N // n, N % n
        sizes = [base + 1] * rem + [base] * (n - rem)
        assign = torch.empty(N, dtype=torch.long)
        i = 0
        for c, sz in enumerate(sizes):
            assign[i:i + sz] = c
            i += sz
        return assign

    def _grouping_error(assigns: torch.Tensor, n: int) -> dict:
        """Mean logit_mae / value_rel over all (group, cluster, query-head) cells.

        ``assigns`` is (N,) — same grouping for every G — or (G, N) — a
        per-group grouping (k-means clusters independently per KV group).
        """
        mae_num = mae_den = vrel_num = vrel_den = 0.0
        for g in range(G):
            a = assigns[g] if assigns.dim() == 2 else assigns
            for c in range(n):
                idx = (a == c).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                kc, vc = kp[g, idx, :], vp[g, idx, :]  # (w, D) / (w, Dv)
                w = idx.numel()
                k_bar, v_bar = kc.mean(dim=0), vc.mean(dim=0)
                for r in range(rf):
                    qr = qf[g, r]                                      # (T_q, D)
                    z = scale * (qr @ kc.T)                            # (T_q, w)
                    l_exact = torch.logsumexp(z, dim=-1)                # (T_q,)
                    l_approx = scale * (qr @ k_bar) + torch.log(torch.tensor(float(w)))
                    mae_num += (l_approx - l_exact).abs().sum().item()
                    mae_den += T_q
                    within = torch.softmax(z, dim=-1)                   # (T_q, w)
                    v_exact = within @ vc                                # (T_q, Dv)
                    diff = (v_exact - v_bar).norm(dim=-1) / (Dv ** 0.5)
                    ref = v_exact.norm(dim=-1) / (Dv ** 0.5)
                    vrel_num += (diff / ref.clamp_min(1e-8)).sum().item()
                    vrel_den += T_q
        return {
            "logit_mae": mae_num / max(mae_den, 1),
            "value_rel": vrel_num / max(vrel_den, 1),
        }

    adj_assign = _adjacency_grouping(n_slots)
    oracle_assign = torch.stack(
        [_kmeans(kp[g], n_slots, n_iters, seed) for g in range(G)]
    )  # (G, N) — clustered independently per KV group

    return {
        "adjacency": _grouping_error(adj_assign, n_slots),
        "oracle_cluster": _grouping_error(oracle_assign, n_slots),
    }
