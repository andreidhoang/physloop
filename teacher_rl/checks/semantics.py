"""Reward/evaluation semantics over captured dense metrics (E10-E14; build-spec A9).

Serves S4.1/S4.3 [C] (G9, G7). Agent-authored verification that compares three views
of one rollout without choosing between them:

1. **Online aggregate** - what the AlpaGym reward reads. Training AlpaSim
   (``alpasim@46b401b``) computes ``aggregated_metrics = {m.name: m.aggregate()}``
   (``src/eval/src/eval/scenario_evaluator.py:101``) with ``MetricReturn.aggregate``
   (``data.py:2017-2036``): valid entries only, over the *whole* timeline, NaN when
   nothing is valid. ``eval_relevant`` is appended only afterwards (:124-132), so the
   online value includes pre-engagement (warm-up) timesteps.
2. **Dense timestep metrics** - ``RolloutReturn.timestep_metrics`` (runtime.proto
   :86-101; engine.py:39-54), which AlpaGym discards (``dense={}``); captured by H4.
3. **Offline evaluation** - ``aggregation/processing.py`` ``DEFAULT_MODIFIERS``
   (:38-66) applied in order: drop rows before the first ``eval_relevant > 0``; add
   ``offroad_or_collision``, ``collision_at_fault``, ``offroad_or_collision_at_fault``,
   ``duration_frac_20s``; drop a whole trajectory with any ``img_is_black > 0``; keep
   rows up to and including the first ``offroad_or_collision > 0``; then aggregate
   with Polars (nulls skipped by mean/median/max/min; ``last`` keeps a null). The
   offline pivot reads only ``values``: the ``valid`` flags are ignored. A missing
   ``offroad`` column is zero-filled upstream (:744-750); this module never adopts
   that fill and reports the offline view as unknown instead.

Verification level per function:

- ``online_aggregate`` - L1, cross-checked against the vendored pinned
  ``MetricReturn.aggregate`` executed in the tests.
- ``offline_view`` / ``reconcile_metric_domains`` - L1 re-derivation of the pinned
  modifier chain from source text (Polars is not a dependency); literal known
  answers. Actual dense captures are L3.
- ``contact_predicates`` (E11), ``check_reference_alignment`` (E12, raise paths of
  ``alpagym@972d160:.../rewards/distance_to_gt.py``), ``diagnose_trajectory`` (E13),
  ``summarize_validity`` (E14) - L1; actual captures L3.

Value convention: captured numbers are finite int/float; a non-finite capture is the
token ``"nan"``, ``"inf"`` or ``"-inf"`` (JSON-safe); ``None`` means missing. Missing
or unknown is never zero.

Owner inputs (required, no defaults): E12 comparison tolerance when a captured report
is compared; E13 thresholds; E14 required metric names (from the registered reward
arm). Missing -> ``blocked`` naming the field, facts still computed.

This module never computes a reward for training, never changes a predicate, mask or
termination rule, never selects a retry and never imports torch.
"""

import math
from itertools import pairwise

import numpy as np

from teacher_rl.checks import Issues, artifact, check_scope, is_count, is_real, is_text, require

PINNED_ALPASIM = "alpasim@46b401beaec27977cbb86826df8e39cec2d49e49:src/eval/src/eval"
PINNED_REWARD = (
    "alpagym@972d160eed0e23d388497851504a3a233fec5879:packages/runtime/src/alpagym_runtime/rewards"
)
AGGREGATIONS = ("mean", "median", "max", "min", "last")
TOKENS = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}
REQUIRED_OFFLINE = (
    "eval_relevant",
    "collision_any",
    "collision_front",
    "collision_lateral",
    "img_is_black",
)
COMBINED = {
    "offroad_or_collision": (("collision_any", "offroad"), "max"),
    "collision_at_fault": (("collision_front", "collision_lateral"), "max"),
    "offroad_or_collision_at_fault": (("collision_at_fault", "offroad"), "max"),
}
ENVELOPE_M = 4.0
SILENT_RETRY = "silent_retry_new_session"
THRESHOLD_KEYS = (
    "stationary_max_path_m",
    "oscillation_min_heading_sign_changes",
    "emergency_min_deceleration_mps2",
    "detour_min_length_ratio",
)

_METRIC_KEYS = {"name", "timestamps_us", "values", "valid", "time_aggregation"}


def _decode(value, path):
    if value is None:
        return None
    if type(value) is str:
        require(value in TOKENS, f"{path}: unknown token {value!r}")
        return TOKENS[value]
    require(is_real(value), f"{path}: values must be finite int/float, a token or None")
    return float(value)


def _encode(value):
    if value is None:
        return None
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def _check_metric(metric, path):
    require(
        type(metric) is dict and set(metric) == _METRIC_KEYS,
        f"{path} needs exactly {sorted(_METRIC_KEYS)}",
    )
    require(is_text(metric["name"]), f"{path}.name must be text")
    n = len(metric["timestamps_us"]) if type(metric["timestamps_us"]) is list else -1
    require(
        n >= 0 and all(is_count(t) for t in metric["timestamps_us"]),
        f"{path}.timestamps_us must be ints >= 0",
    )
    require(
        type(metric["values"]) is list and len(metric["values"]) == n, f"{path}.values must align"
    )
    require(
        type(metric["valid"]) is list
        and len(metric["valid"]) == n
        and all(type(v) is bool for v in metric["valid"]),
        f"{path}.valid must be aligned booleans",
    )
    require(
        metric["time_aggregation"] in AGGREGATIONS,
        f"{path}.time_aggregation must be one of {AGGREGATIONS}",
    )
    for i, value in enumerate(metric["values"]):
        require(value is not None, f"{path}.values[{i}] is missing (dense values are never null)")
        _decode(value, f"{path}.values[{i}]")


def _check_rollout(rollout):
    keys = {"session_uuid", "timestep_metrics", "aggregated_metrics"}
    require(type(rollout) is dict and set(rollout) == keys, f"rollout needs exactly {sorted(keys)}")
    require(is_text(rollout["session_uuid"]), "rollout.session_uuid must be text")
    metrics = rollout["timestep_metrics"]
    require(type(metrics) is list, "timestep_metrics must be a list")
    for i, metric in enumerate(metrics):
        _check_metric(metric, f"timestep_metrics[{i}]")
    names = [m["name"] for m in metrics]
    require(len(set(names)) == len(names), "timestep_metrics repeats a metric name")
    aggregated = rollout["aggregated_metrics"]
    require(
        aggregated is None or type(aggregated) is dict, "aggregated_metrics must be a dict or None"
    )
    for name, value in (aggregated or {}).items():
        require(is_text(name), "aggregated_metrics keys must be text")
        _decode(value, f"aggregated_metrics.{name}")


def online_aggregate(metric):
    """Replicate the pinned ``MetricReturn.aggregate`` (L1): valid values, whole series.

    Returns a finite float or a token (``"nan"`` when no value is valid, as upstream).
    """
    _check_metric(metric, "metric")
    values = np.array([_decode(v, "values") for v in metric["values"]], dtype=np.float64)
    valid = np.array(metric["valid"], dtype=bool)
    chosen = values[valid]
    if len(chosen) == 0:
        return "nan"
    kind = metric["time_aggregation"]
    func = {"mean": np.mean, "median": np.median, "max": np.max, "min": np.min}.get(kind)
    result = float(chosen[-1]) if kind == "last" else float(func(chosen))
    return _encode(result)


def _gt0(value):
    """Kleene ``value > 0.0`` on a decoded float (None -> None)."""
    return None if value is None else value > 0.0


def _kleene_or(a, b):
    if a is True or b is True:
        return True
    if a is None or b is None:
        return None
    return False


def _polars_aggregate(column, kind):
    present = [v for v in column if v is not None]
    if kind == "last":
        return column[-1] if column else None
    if not present:
        return None
    arr = np.array(present, dtype=np.float64)
    return float({"mean": np.mean, "median": np.median, "max": np.max, "min": np.min}[kind](arr))


def offline_view(rollout, *, stop_after=None):
    """Re-derive the pinned offline modifier chain for one trajectory (L1).

    Returns ``{"state": "computed" | "removed" | "unknown", "reason", "values",
    "rows_kept", "stages"}``. ``stop_after`` (``"pivot"``, ``"pre_engagement"`` or
    ``None`` for the full chain) supports attribution of online/offline differences.
    """
    _check_rollout(rollout)
    require(
        stop_after in (None, "pivot", "pre_engagement"),
        "stop_after must be None, 'pivot' or 'pre_engagement'",
    )
    metrics = {m["name"]: m for m in rollout["timestep_metrics"]}
    aggregation = {name: m["time_aggregation"] for name, m in metrics.items()}
    for name, m in metrics.items():
        if len(set(m["timestamps_us"])) != len(m["timestamps_us"]):
            return {"state": "unknown", "reason": f"duplicate_timestamp:{name}", "values": None}
    if any(type(v) is str for m in metrics.values() for v in m["values"]):
        # Polars NaN/inf semantics (comparison order, NaN-skipping max vs propagating
        # mean) are not modelled here, so the offline view is not derived.
        return {
            "state": "unknown",
            "reason": "nonfinite_value_semantics_unmodelled",
            "values": None,
        }
    if "offroad" not in metrics:
        return {"state": "unknown", "reason": "offroad_missing_upstream_zero_fills", "values": None}
    missing = [name for name in REQUIRED_OFFLINE if name not in metrics]
    if missing:
        return {
            "state": "unknown",
            "reason": "offline_column_missing:" + ",".join(missing),
            "values": None,
        }
    timestamps = sorted({t for m in metrics.values() for t in m["timestamps_us"]})
    table = {
        name: {t: _decode(v, "v") for t, v in zip(m["timestamps_us"], m["values"])}
        for name, m in metrics.items()
    }
    rows = [
        {name: table[name].get(t) for name in metrics} | {"timestamps_us": t} for t in timestamps
    ]
    stages = {"pivot_rows": len(rows)}
    if stop_after != "pivot":
        first = next((r["timestamps_us"] for r in rows if _gt0(r["eval_relevant"]) is True), None)
        if first is not None:
            rows = [r for r in rows if r["timestamps_us"] >= first]
        stages["after_pre_engagement_rows"] = len(rows)
    if stop_after is None:
        for name, (parts, kind) in COMBINED.items():
            for r in rows:
                value = _kleene_or(_gt0(r[parts[0]]), _gt0(r[parts[1]]))
                r[name] = None if value is None else float(value)
            aggregation[name] = kind
        start = min((r["timestamps_us"] for r in rows), default=None)
        for r in rows:
            r["duration_frac_20s"] = (r["timestamps_us"] - start) / 20e6
        aggregation["duration_frac_20s"] = "last"
        if any(_gt0(r["img_is_black"]) is True for r in rows):
            return {
                "state": "removed",
                "reason": "black_image_trajectory_removed",
                "values": None,
                "stages": stages,
            }
        incident = next(
            (r["timestamps_us"] for r in rows if _gt0(r["offroad_or_collision"]) is True), None
        )
        if incident is not None:
            rows = [r for r in rows if r["timestamps_us"] <= incident]
        stages["after_incident_rows"] = len(rows)
        if "progress_rel_to_total" in metrics and "progress_clipped_rel" not in metrics:
            for r in rows:
                r["progress_clipped_rel"] = r["progress_rel_to_total"]
            aggregation["progress_clipped_rel"] = "last"
    values = {
        name: _encode(_polars_aggregate([r[name] for r in rows], kind))
        for name, kind in sorted(aggregation.items())
    }
    return {
        "state": "computed",
        "reason": None,
        "values": values,
        "rows_kept": len(rows),
        "stages": stages,
    }


def _same(a, b):
    return type(a) is type(b) and a == b


def reconcile_metric_domains(rollout, *, scope):
    """E10: captured online aggregate vs dense recomputation vs offline evaluation.

    For every dense metric: recompute the online aggregate (must equal the captured
    one exactly, else ``online_aggregate_mismatch``), compute the offline value, and
    attribute any online/offline difference to the first pinned stage that changes
    it: ``invalid_values_included`` (offline ignores ``valid``),
    ``pre_engagement_removed``, ``post_incident_removed`` (or the trajectory is
    ``black_image_trajectory_removed``). Differences are facts (``warning``), never
    reconciled away; Huy registers which domain an objective uses.

    Upstream aggregates every dense metric except ``eval_relevant`` (appended later),
    so a dense metric without a captured aggregate (``online_aggregate_missing``) or
    an aggregate without a dense series (``dense_metric_missing``) means the capture
    is incomplete: the captured-vs-recomputed comparison did not happen for it and
    the status is ``unverified``, never ``passed``.
    """
    check_scope(scope)
    _check_rollout(rollout)
    issues = Issues()
    sid = rollout["session_uuid"]
    aggregated = rollout["aggregated_metrics"]
    if aggregated is None:
        issues.add(
            "aggregated_metrics_none",
            f"session:{sid}",
            "metrics=None: a metric reward term raises ValueError upstream "
            "(compute.py:33-34) -> silent retry.",
        )
        aggregated = {}
    full = offline_view(rollout)
    pivot = offline_view(rollout, stop_after="pivot")
    engaged = offline_view(rollout, stop_after="pre_engagement")
    if full["state"] == "unknown":
        issues.add(
            "offline_view_unknown",
            f"session:{sid}",
            f"Offline view not derivable: {full['reason']}.",
            severity="warning",
        )
    rows = []
    for metric in sorted(rollout["timestep_metrics"], key=lambda m: m["name"]):
        name = metric["name"]
        path = f"session:{sid}/metric:{name}"
        online = online_aggregate(metric)
        row = {
            "metric": name,
            "time_aggregation": metric["time_aggregation"],
            "online_recomputed": online,
            "captured_online": None,
            "online_matches_capture": None,
            "invalid_rows": metric["valid"].count(False),
            "offline": None,
            "offline_state": full["state"],
            "differs": None,
            "first_differing_stage": None,
        }
        if name != "eval_relevant":
            if name in aggregated:
                row["captured_online"] = aggregated[name]
                row["online_matches_capture"] = _same(
                    _encode(_decode(aggregated[name], "c")), online
                )
                if not row["online_matches_capture"]:
                    issues.add(
                        "online_aggregate_mismatch",
                        path,
                        "Captured aggregate differs from dense recomputation.",
                    )
            else:
                issues.add(
                    "online_aggregate_missing",
                    path,
                    "Dense metric has no captured online aggregate.",
                    severity="warning",
                )
        if full["state"] == "computed":
            row["offline"] = full["values"].get(name)
            row["differs"] = not _same(row["offline"], online)
            if row["differs"]:
                stage_values = [
                    ("invalid_values_included", pivot),
                    ("pre_engagement_removed", engaged),
                    ("post_incident_removed", full),
                ]
                for stage, view in stage_values:
                    if view["state"] == "computed" and not _same(view["values"].get(name), online):
                        row["first_differing_stage"] = stage
                        break
                issues.add(
                    "online_offline_domain_differs",
                    path,
                    f"online {online!r} vs offline {row['offline']!r} "
                    f"({row['first_differing_stage']}).",
                    severity="warning",
                )
        elif full["state"] == "removed":
            row["differs"] = True
            row["first_differing_stage"] = full["reason"]
        rows.append(row)
    dense_names = {m["name"] for m in rollout["timestep_metrics"]}
    for name in sorted(set(aggregated) - dense_names):
        issues.add(
            "dense_metric_missing",
            f"session:{sid}/metric:{name}",
            "Aggregate without dense series.",
            severity="warning",
        )
    incomplete = {"online_aggregate_missing", "dense_metric_missing"} & set(issues.codes())
    if issues.has("error"):
        status = "blocked"
    elif full["state"] == "unknown" or incomplete:
        status = "unverified"
    else:
        status = "passed"
    gaps = (["capture incomplete (see issues)"] if incomplete else []) + (
        ["offline view not derivable"] if full["state"] == "unknown" else []
    )
    meaning = {
        "passed": "reconciliation complete; differences are facts, not failures",
        "unverified": "reconciliation incomplete: " + "; ".join(gaps),
        "blocked": "captured aggregates are absent or contradict the dense recomputation",
    }[status]
    return artifact(
        "s4_e10_metric_domain_reconciliation",
        module=__name__,
        module_file=__file__,
        scope=scope,
        inputs={"rollout": rollout},
        units={"metrics": "as emitted by the pinned AlpaSim scorers", "timestamps": "us"},
        status=status,
        issues=issues,
        body={
            "ticket": "E10",
            "pinned_source": PINNED_ALPASIM,
            "status_meaning": meaning,
            "session_uuid": sid,
            "offline_state": full["state"],
            "offline_reason": full["reason"],
            "offline_values": full["values"],
            "metrics": rows,
        },
    )


def contact_predicates(rollout, *, scope):
    """E11: any-contact vs additive front+lateral over the online (valid) domain.

    ``any_contact = max(collision_any)`` while ``additive = max(front) + max(lateral)``
    can be 2 for one episode; the two are different objectives. Per-timestep
    consistency ``any == front or rear or lateral`` is checked. Source limitation: the
    pinned ``CollisionScorer`` (``collision.py:44-55``) keeps only the last
    intersecting actor's class per timestep, which dense values cannot reveal.
    """
    check_scope(scope)
    _check_rollout(rollout)
    metrics = {m["name"]: m for m in rollout["timestep_metrics"]}
    issues = Issues()
    sid = rollout["session_uuid"]
    names = ("collision_any", "collision_front", "collision_rear", "collision_lateral")
    missing = [n for n in names if n not in metrics]
    body = {
        "ticket": "E11",
        "session_uuid": sid,
        "pinned_source": PINNED_ALPASIM + "/scorers/collision.py",
    }
    if missing:
        issues.add(
            "collision_metric_missing", f"session:{sid}", f"Missing dense series: {missing}."
        )
        body.update(any_contact=None, additive_front_lateral=None)
    else:
        maxima = {n: online_aggregate(metrics[n] | {"time_aggregation": "max"}) for n in names}
        if any(type(v) is str for v in maxima.values()):
            issues.add(
                "collision_metric_nonfinite_or_empty",
                f"session:{sid}",
                "A collision maximum is not finite.",
            )
            body.update(any_contact=None, additive_front_lateral=None, maxima=maxima)
        else:
            series = {
                n: dict(
                    zip(
                        metrics[n]["timestamps_us"], (_decode(v, "v") for v in metrics[n]["values"])
                    )
                )
                for n in names
            }
            inconsistent = sorted(
                t
                for t in series["collision_any"]
                if all(t in series[n] for n in names[1:])
                and (series["collision_any"][t] > 0) != any(series[n][t] > 0 for n in names[1:])
            )
            if inconsistent:
                issues.add(
                    "collision_any_inconsistent",
                    f"session:{sid}",
                    f"any != front|rear|lateral at {inconsistent}.",
                )
            additive = maxima["collision_front"] + maxima["collision_lateral"]
            body.update(
                maxima=maxima,
                any_contact=maxima["collision_any"],
                additive_front_lateral=additive,
                additive_differs_from_any=additive != maxima["collision_any"],
                inconsistent_timestamps_us=inconsistent,
            )
    return artifact(
        "s4_e11_contact_predicates",
        module=__name__,
        module_file=__file__,
        scope=scope,
        inputs={"rollout": rollout},
        units={"contact": "indicator (0/1) maxima over valid timesteps"},
        status="blocked" if issues.has("error") else "passed",
        issues=issues,
        body=body,
    )


def _interp(x, xs, ys):
    """Linear interpolation on sorted ``xs`` (clamped like ``np.interp``)."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    if xs[hi] == xs[lo]:
        return ys[hi]
    w = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + w * (ys[hi] - ys[lo])


def _check_poses(poses, path, with_yaw):
    keys = {"timestamp_us", "x", "y", "yaw"} if with_yaw else {"timestamp_us", "x", "y"}
    require(type(poses) is list, f"{path} must be a list")
    for i, pose in enumerate(poses):
        require(
            type(pose) is dict and set(pose) == keys, f"{path}[{i}] needs exactly {sorted(keys)}"
        )
        require(type(pose["timestamp_us"]) is int, f"{path}[{i}].timestamp_us must be int")
        require(
            all(is_real(pose[k]) for k in keys - {"timestamp_us"}),
            f"{path}[{i}] coordinates must be finite",
        )


def reference_rmse(executed, ground_truth):
    """Independent oracle for the pinned ``distance_to_gt`` RMSE (L1).

    Returns ``(outcome, value)``: ``("raises", code)`` mirroring each upstream raise
    path in source order; ``("not_reproduced", "gt_duplicate_timestamps")`` when GT
    timestamps repeat (no raise path is taken, but ``np.interp`` on repeated ``xp``
    is ill-defined and this oracle does not reproduce which duplicate it uses); else
    ``("computed", {"rmse_xy", "num_paired"})``.
    """
    if ground_truth is None:
        return "raises", "ground_truth_missing"
    if len(executed) < 2:
        return "raises", "too_few_executed_poses"
    gt = ground_truth["poses"]
    if len(gt) < 2:
        return "raises", "too_few_gt_poses"
    ts = [p["timestamp_us"] for p in executed]
    if any(b <= a for a, b in pairwise(ts)):
        return "raises", "executed_timestamps_not_increasing"
    order = sorted(range(len(gt)), key=lambda i: gt[i]["timestamp_us"])  # stable
    gts = [gt[i]["timestamp_us"] for i in order]
    gxy = [(gt[i]["x"], gt[i]["y"]) for i in order]
    anchor = ground_truth["anchor_timestamp_us"]
    if anchor < ts[0] or anchor > ts[-1]:
        return "raises", "gt_anchor_outside_executed_range"
    ax = _interp(anchor, ts, [p["x"] for p in executed])
    ay = _interp(anchor, ts, [p["y"] for p in executed])
    yaw = math.atan2(
        _interp(anchor, ts, [math.sin(p["yaw"]) for p in executed]),
        _interp(anchor, ts, [math.cos(p["yaw"]) for p in executed]),
    )
    c, s = math.cos(yaw), math.sin(yaw)
    local = [(c * x - s * y + ax, s * x + c * y + ay) for x, y in gxy]
    paired = [p for p in executed if gts[0] <= p["timestamp_us"] <= gts[-1]]
    if not paired:
        return "raises", "no_executed_points_in_gt_span"
    if len(set(gts)) != len(gts):
        return "not_reproduced", "gt_duplicate_timestamps"
    errors = []
    for p in paired:
        gx = _interp(p["timestamp_us"], gts, [q[0] for q in local])
        gy = _interp(p["timestamp_us"], gts, [q[1] for q in local])
        errors.append((p["x"] - gx) ** 2 + (p["y"] - gy) ** 2)
    return "computed", {"rmse_xy": math.sqrt(sum(errors) / len(errors)), "num_paired": len(errors)}


def check_reference_alignment(executed, ground_truth, *, captured=None, tolerance=None, scope):
    """E12: timestamp interpolation and the upstream raise paths (L1).

    ``executed``: ``[{timestamp_us, x, y, yaw}]`` (executed poses as the reward sees
    them); ``ground_truth``: ``{anchor_timestamp_us, poses: [{timestamp_us, x, y}]}``
    (rig-frame XY at the anchor) or ``None``. Every upstream ``ValueError`` here
    becomes a silent retry with a new session (``streaming_worker.py:255-256``,
    budget 3 per payload), so a raise is reported with that consequence, never as a
    driving outcome. ``captured`` (``{gt_rmse_xy, gt_num_paired}``) is compared only
    under an owner ``tolerance`` (``{"absolute": v}``); missing -> ``blocked``.
    Duplicate GT timestamps give ``upstream_outcome`` ``not_reproduced``, ``rmse_xy``
    ``None`` and status ``unverified`` (the pinned ``np.interp`` choice is not reproduced).
    """
    check_scope(scope)
    _check_poses(executed, "executed", True)
    if ground_truth is not None:
        require(
            type(ground_truth) is dict and set(ground_truth) == {"anchor_timestamp_us", "poses"},
            "ground_truth needs exactly anchor_timestamp_us and poses",
        )
        require(type(ground_truth["anchor_timestamp_us"]) is int, "anchor_timestamp_us must be int")
        _check_poses(ground_truth["poses"], "ground_truth.poses", False)
    require(
        captured is None
        or (type(captured) is dict and set(captured) == {"gt_rmse_xy", "gt_num_paired"}),
        "captured must be None or {gt_rmse_xy, gt_num_paired}",
    )
    require(
        tolerance is None
        or (
            type(tolerance) is dict
            and set(tolerance) == {"absolute"}
            and is_real(tolerance["absolute"])
            and tolerance["absolute"] >= 0
        ),
        "tolerance must be None or {'absolute': finite >= 0}",
    )
    issues = Issues()
    outcome, value = reference_rmse(executed, ground_truth)
    body = {
        "ticket": "E12",
        "pinned_source": PINNED_REWARD + "/distance_to_gt.py",
        "upstream_outcome": outcome,
    }
    if outcome == "not_reproduced":
        body.update(raise_code=None, consequence=None, rmse_xy=None, num_paired=None)
        issues.add(
            value,
            "ground_truth",
            "Duplicate GT timestamps make np.interp ill-defined; the upstream RMSE is "
            "not reproduced, so no value is reported.",
            severity="warning",
        )
    elif outcome == "raises":
        body.update(raise_code=value, consequence=SILENT_RETRY, rmse_xy=None, num_paired=None)
        issues.add(
            value,
            "executed",
            "Upstream raises ValueError here; the attempt is silently retried.",
            severity="warning",
        )
    else:
        body.update(raise_code=None, consequence=None, **value)
    if captured is not None:
        if tolerance is None:
            issues.add(
                "tolerance_unregistered",
                "tolerance",
                "Comparing a captured RMSE needs an owner tolerance.",
            )
        elif outcome == "computed":
            ok = (
                is_real(captured["gt_rmse_xy"])
                and abs(captured["gt_rmse_xy"] - value["rmse_xy"]) <= tolerance["absolute"]
                and captured["gt_num_paired"] == value["num_paired"]
            )
            body["captured_matches"] = ok
            if not ok:
                issues.add(
                    "captured_reference_mismatch",
                    "captured",
                    "Captured RMSE/pairs differ from the oracle.",
                )
        elif outcome == "not_reproduced":
            body["captured_matches"] = None
        else:
            issues.add(
                "captured_report_for_raising_input",
                "captured",
                "Upstream would have raised; a report exists.",
            )
    if issues.has("error"):
        status = "blocked"
    elif outcome == "not_reproduced":
        status = "unverified"
    else:
        status = "passed"
    return artifact(
        "s4_e12_reference_alignment",
        module=__name__,
        module_file=__file__,
        scope=scope,
        inputs={
            "executed": executed,
            "ground_truth": ground_truth,
            "captured": captured,
            "tolerance": tolerance,
        },
        units={"rmse_xy": "m", "timestamps": "us"},
        status=status,
        issues=issues,
        body=body,
    )


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def diagnose_trajectory(executed, *, reference_xy=None, thresholds, scope):
    """E13: threshold-free trajectory facts plus owner-threshold flags (L1).

    Facts: duration, path length, net displacement, segment speeds, maximum
    deceleration, heading-rate sign changes and (with ``reference_xy``, a list of
    ``[x, y]``) the path-length ratio to the reference. ``thresholds`` (owner input,
    keys ``THRESHOLD_KEYS``) turn facts into ``stationary``, ``oscillation``,
    ``hard_stop`` and ``long_detour`` flags; ``None`` blocks. Whether a hard stop was
    a *legitimate* emergency stop needs hazard context and stays ``unverified``.
    Progress is reported by the caller separately; it never stands in for these.
    """
    check_scope(scope)
    _check_poses(executed, "executed", True)
    require(len(executed) >= 2, "executed needs at least two poses")
    ts = [p["timestamp_us"] for p in executed]
    require(all(b > a for a, b in pairwise(ts)), "executed timestamps must increase strictly")
    if reference_xy is not None:
        require(
            type(reference_xy) is list
            and len(reference_xy) >= 2
            and all(
                type(p) is list and len(p) == 2 and all(is_real(c) for c in p) for p in reference_xy
            ),
            "reference_xy must be a list of >= 2 [x, y] pairs",
        )
    if thresholds is not None:
        require(
            type(thresholds) is dict
            and set(thresholds) == set(THRESHOLD_KEYS)
            and all(is_real(v) for v in thresholds.values()),
            f"thresholds must have exactly {THRESHOLD_KEYS} with finite values",
        )
    issues = Issues()
    if thresholds is None:
        issues.add(
            "thresholds_unregistered", "thresholds", "E13 diagnostic thresholds are owner inputs."
        )
    seg = [
        (
            math.hypot(b["x"] - a["x"], b["y"] - a["y"]),
            (b["timestamp_us"] - a["timestamp_us"]) / 1e6,
        )
        for a, b in pairwise(executed)
    ]
    speeds = [d / dt for d, dt in seg]
    mids = [(a["timestamp_us"] + b["timestamp_us"]) / 2e6 for a, b in pairwise(executed)]
    decels = [-(v2 - v1) / (t2 - t1) for v1, v2, t1, t2 in zip(speeds, speeds[1:], mids, mids[1:])]
    rates = [_wrap(b["yaw"] - a["yaw"]) for a, b in pairwise(executed)]
    signs = [(r > 0) - (r < 0) for r in rates if r != 0]
    path_length = sum(d for d, _ in seg)
    facts = {
        "duration_s": (ts[-1] - ts[0]) / 1e6,
        "path_length_m": path_length,
        "net_displacement_m": math.hypot(
            executed[-1]["x"] - executed[0]["x"], executed[-1]["y"] - executed[0]["y"]
        ),
        "max_speed_mps": max(speeds),
        "final_speed_mps": speeds[-1],
        "max_deceleration_mps2": max(decels) if decels else None,
        "heading_sign_changes": sum(1 for a, b in pairwise(signs) if a != b),
        "reference_length_m": None,
        "length_ratio": None,
    }
    if reference_xy is not None:
        ref = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in pairwise(reference_xy))
        facts["reference_length_m"] = ref
        facts["length_ratio"] = path_length / ref if ref > 0 else None
    flags = None
    if thresholds is not None:
        decel = facts["max_deceleration_mps2"]
        flags = {
            "stationary": path_length <= thresholds["stationary_max_path_m"],
            "oscillation": facts["heading_sign_changes"]
            >= thresholds["oscillation_min_heading_sign_changes"],
            "hard_stop": None
            if decel is None
            else decel >= thresholds["emergency_min_deceleration_mps2"],
            "long_detour": None
            if facts["length_ratio"] is None
            else facts["length_ratio"] >= thresholds["detour_min_length_ratio"],
            "hard_stop_legitimacy": "unverified",
        }
    return artifact(
        "s4_e13_trajectory_diagnostics",
        module=__name__,
        module_file=__file__,
        scope=scope,
        inputs={"executed": executed, "reference_xy": reference_xy, "thresholds": thresholds},
        units={
            "duration": "s",
            "length": "m",
            "speed": "m/s (segment average)",
            "deceleration": "m/s^2 between segment midpoints",
            "heading_sign_changes": "count",
        },
        status="blocked" if issues.has("error") else "passed",
        issues=issues,
        body={"ticket": "E13", "facts": facts, "flags": flags},
    )


_ATTEMPT_KEYS = {
    "attempt_id",
    "payload_id",
    "session_uuid",
    "outcome",
    "failure_cause",
    "aggregated_metrics",
    "img_is_black_max",
    "max_dist_to_gt_trajectory_m",
    "old_logprobs_nonfinite",
    "nonfinite_resamples",
}
INDICATORS = (
    "black_frame",
    "out_of_envelope",
    "missing_metric",
    "nonfinite_metric",
    "nonfinite_log_density",
    "nonfinite_resampled",
)
CAUSE_PREFIXES = (
    ("KeyError", "missing_metric"),
    ("ValueError: metric reward term requires episode.metrics", "metrics_none"),
    ("ValueError: compute_distance_to_gt_reward", "reference_alignment"),
    ("ValueError: GT anchor timestamp", "reference_alignment"),
    ("FloatingPointError", "nonfinite"),
)


def _cause_category(cause):
    if cause is None:
        return "unknown_cause"
    for prefix, category in CAUSE_PREFIXES:
        if cause.startswith(prefix):
            return category
    return "other"


def summarize_validity(attempts, *, required_metrics, scope):
    """E14: per-attempt validity/infrastructure indicators and silent-retry causes.

    Each indicator is ``True``/``False``/``None`` (unknown, never zero). Required
    metric names (from the registered reward arm; owner input, no default) decide
    ``missing_metric`` and ``nonfinite_metric``; a missing metric raises ``KeyError``
    upstream (``compute.py:38``) and ``metrics=None`` raises ``ValueError`` (:33-34),
    both silently retried. ``out_of_envelope`` (max deviation >= 4 m) is an offline
    measure reported separately; ``nonfinite_log_density`` raises in the trainer
    (``replay_objective.py:32-39``); ``nonfinite_resampled`` counts NaN re-samples
    hidden by ``retry_on_nonfinite``. Unknown counts are separate from false counts.

    A ``null`` aggregated value counts as missing (module value convention). A
    ``succeeded`` attempt lacking a required metric (or with ``metrics=None``)
    contradicts the pinned path, which would have raised and retried:
    ``succeeded_attempt_missing_metric`` (error). ``passed`` means the summary is
    complete, not that the attempts are valid (``status_meaning``).
    """
    check_scope(scope)
    require(type(attempts) is list and attempts, "attempts must be a nonempty list")
    if required_metrics is not None:
        require(
            type(required_metrics) is list
            and required_metrics
            and all(is_text(n) for n in required_metrics)
            and len(set(required_metrics)) == len(required_metrics),
            "required_metrics must be a nonempty list of distinct names",
        )
    ids = set()
    for i, a in enumerate(attempts):
        path = f"attempts[{i}]"
        require(
            type(a) is dict and set(a) == _ATTEMPT_KEYS,
            f"{path} needs exactly {sorted(_ATTEMPT_KEYS)}",
        )
        require(
            is_text(a["attempt_id"]) and a["attempt_id"] not in ids,
            f"{path}.attempt_id must be unique text",
        )
        ids.add(a["attempt_id"])
        require(is_text(a["payload_id"]) and is_text(a["session_uuid"]), f"{path} ids must be text")
        require(a["outcome"] in ("succeeded", "failed"), f"{path}.outcome must be succeeded|failed")
        require(
            a["failure_cause"] is None or type(a["failure_cause"]) is str,
            f"{path}.failure_cause must be text or None",
        )
        require(
            a["aggregated_metrics"] is None or type(a["aggregated_metrics"]) is dict,
            f"{path}.aggregated_metrics must be dict or None",
        )
        for name, value in (a["aggregated_metrics"] or {}).items():
            _decode(value, f"{path}.aggregated_metrics.{name}")
        for key in ("img_is_black_max", "max_dist_to_gt_trajectory_m"):
            require(a[key] is None or is_real(a[key]), f"{path}.{key} must be finite or None")
        require(
            a["old_logprobs_nonfinite"] in (None, True, False),
            f"{path}.old_logprobs_nonfinite must be bool or None",
        )
        require(
            a["nonfinite_resamples"] is None or is_count(a["nonfinite_resamples"]),
            f"{path}.nonfinite_resamples must be int >= 0 or None",
        )
    issues = Issues()
    if required_metrics is None:
        issues.add(
            "required_metrics_unregistered",
            "required_metrics",
            "Reward-arm metric names are an owner input.",
        )
    rows, counts, causes, per_payload = (
        [],
        {k: {"true": 0, "false": 0, "unknown": 0} for k in INDICATORS},
        {},
        {},
    )
    for a in sorted(attempts, key=lambda x: x["attempt_id"]):
        metrics = a["aggregated_metrics"]
        if required_metrics is None:
            missing = nonfinite_metric = None
        elif metrics is None:
            missing, nonfinite_metric = True, None
        else:
            present = [_decode(metrics.get(n), "m") for n in required_metrics]
            missing = any(v is None for v in present)
            nonfinite_metric = any(v is not None and not math.isfinite(v) for v in present)
        if a["outcome"] == "succeeded" and missing is True:
            issues.add(
                "succeeded_attempt_missing_metric",
                f"attempt:{a['attempt_id']}",
                "A required metric is missing, so the pinned reward raises and retries "
                "(compute.py:33-38); a succeeded outcome contradicts the capture.",
            )
        black = None if a["img_is_black_max"] is None else a["img_is_black_max"] > 0
        envelope = (
            None
            if a["max_dist_to_gt_trajectory_m"] is None
            else a["max_dist_to_gt_trajectory_m"] >= ENVELOPE_M
        )
        resampled = None if a["nonfinite_resamples"] is None else a["nonfinite_resamples"] > 0
        indicators = {
            "black_frame": black,
            "out_of_envelope": envelope,
            "missing_metric": missing,
            "nonfinite_metric": nonfinite_metric,
            "nonfinite_log_density": a["old_logprobs_nonfinite"],
            "nonfinite_resampled": resampled,
        }
        for key, value in indicators.items():
            counts[key]["unknown" if value is None else ("true" if value else "false")] += 1
        cause = _cause_category(a["failure_cause"]) if a["outcome"] == "failed" else None
        if cause is not None:
            causes[cause] = causes.get(cause, 0) + 1
            issues.add(
                "silent_retry_attempt",
                f"attempt:{a['attempt_id']}",
                f"Failed attempt ({cause}) is retried upstream with a new session.",
                severity="warning",
            )
        per_payload.setdefault(a["payload_id"], []).append(a["attempt_id"])
        rows.append(
            {
                "attempt_id": a["attempt_id"],
                "payload_id": a["payload_id"],
                "outcome": a["outcome"],
                "retry_cause": cause,
                "indicators": indicators,
            }
        )
    if issues.has("error"):
        status = "blocked"
    elif any(c["unknown"] for c in counts.values()):
        status = "unverified"
    else:
        status = "passed"
    return artifact(
        "s4_e14_validity_summary",
        module=__name__,
        module_file=__file__,
        scope=scope,
        inputs={"attempts": attempts, "required_metrics": required_metrics},
        units={"counts": "attempts", "out_of_envelope": "max deviation >= 4 m (offline measure)"},
        status=status,
        issues=issues,
        body={
            "ticket": "E14",
            "status_meaning": (
                "summary complete; indicators are facts, not a validity pass "
                "(true counts are invalid or infrastructure events)"
            ),
            "indicator_counts": counts,
            "retry_causes": dict(sorted(causes.items())),
            "attempts_per_payload": {k: len(v) for k, v in sorted(per_payload.items())},
            "attempts": rows,
            "upstream_retry_budget_per_payload": 3,
        },
    )
