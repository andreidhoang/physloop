"""S4 training diagnostics from protocol-design section 6 (build-spec A9).

Serves S4.3/S4.4 [C] (G9, G7). Agent-authored descriptive diagnostics over captured
training records, each with toy known answers in ``tests/test_s4_diagnostics.py``:
group reward variance; tied/all-fail/mixed groups; the component that decided each
group's order; flip share on re-simulation pairs; action diversity; replay rows after
the first hard event; gradient-norm statistics; changed-parameter norms per module;
clip fraction; KL only where computed; rollout lag; useful samples per paid GPU-hour.

These describe what the shipped trainer did. They are not falsifier verdicts: Huy
fills the quantitative effects, tolerances and decision rules (protocol-design
section 6), and a nonzero gradient is never treated as proof of useful credit.

Verification level: every function is L1 (pure, literal known answers); applying
them to an actual run needs H4 captures on an admitted host (L3).

Agent technical defaults (D-018), stated so Huy sees what he inherits:

- *tied* means all rewards equal as given (float64); ``tied_float32`` is reported too.
- *deciding component*: between the highest- and lowest-total members (first index on
  equal totals), the component(s) with the largest absolute contribution difference;
  ``None`` for an all-tied group.
- *flip*: any member pair whose strict order (``<``, ``=``, ``>``) differs between the
  original and re-simulated rewards.
- *action diversity*: mean pairwise Euclidean distance between members' flattened
  action arrays.
- *rows after the first hard event*: replay rows with timestamp strictly greater than
  the first hard-event timestamp.
- *clip fraction*: the pinned trainer's definition (``trainer.py:474-477``): valid
  rows with ratio ``< 1 - clip_low`` or ``> 1 + clip_high``; with no valid row the
  trainer prints 0.0, here it is ``not_estimable``.

Owner input (required, no default): ``failure_components`` - the names of the
per-member reward *contributions* (scaled terms, as summed into the reward) whose
nonzero value marks a hard event; missing -> all-fail classification ``blocked``
while tied/mixed counts are still reported.

Not estimable is never zero. This module never imports torch, never changes a
reward, advantage or gradient, and never emits an admission or claim.
"""

import math

from teacher_rl.checks import Issues, artifact, check_scope, is_count, is_real, is_text, require
from teacher_rl.checks.audit import kl_status
from teacher_rl.checks.signal import float32_values
from teacher_rl.checks.steps import SOURCES, parse_grad_norm

NOT_ESTIMABLE = "not_estimable"
SECTIONS = (
    "groups",
    "resimulation_pairs",
    "group_actions",
    "replay_rows",
    "grad_norms",
    "parameter_deltas",
    "clip",
    "kl",
    "lag",
    "throughput",
)


def _reals(values, name):
    require(
        type(values) is list and all(is_real(v) for v in values),
        f"{name} must be a list of finite numbers",
    )
    return values


def group_reward_stats(rewards):
    """Population mean/variance of one group (L1); fewer than two members -> not estimable."""
    _reals(rewards, "rewards")
    require(len(rewards) >= 1, "rewards must be nonempty")
    if len(rewards) < 2:
        return {
            "group_size": 1,
            "mean": rewards[0],
            "variance": NOT_ESTIMABLE,
            "tied": NOT_ESTIMABLE,
            "tied_float32": NOT_ESTIMABLE,
        }
    mean = math.fsum(rewards) / len(rewards)
    variance = math.fsum((r - mean) ** 2 for r in rewards) / len(rewards)
    return {
        "group_size": len(rewards),
        "mean": mean,
        "variance": variance,
        "tied": len(set(rewards)) == 1,
        "tied_float32": len(set(float32_values(rewards))) == 1,
    }


def _check_components(components, n, name):
    require(type(components) is list and len(components) == n, f"{name} must align with rewards")
    keys = None
    for member in components:
        require(
            type(member) is dict
            and member
            and all(is_text(k) and is_real(v) for k, v in member.items()),
            f"{name} entries must map names to finite numbers",
        )
        keys = set(member) if keys is None else keys
        require(set(member) == keys, f"{name} members must share component names")


def deciding_component(components, totals):
    """Component(s) with the largest |difference| between top and bottom members (L1)."""
    _reals(totals, "totals")
    _check_components(components, len(totals), "components")
    if len(set(totals)) <= 1:
        return None
    top = max(range(len(totals)), key=lambda i: (totals[i], -i))
    bottom = min(range(len(totals)), key=lambda i: (totals[i], i))
    deltas = {k: abs(components[top][k] - components[bottom][k]) for k in components[top]}
    best = max(deltas.values())
    return sorted(k for k, v in deltas.items() if v == best)


def classify_groups(groups, *, failure_components):
    """Tied / all-fail / mixed counts per protocol-design section 6 (L1)."""
    require(type(groups) is list and groups, "groups must be a nonempty list")
    if failure_components is not None:
        require(
            type(failure_components) is list
            and failure_components
            and all(is_text(c) for c in failure_components),
            "failure_components must be a nonempty list of names or None",
        )
    ids = set()
    rows = []
    for i, group in enumerate(groups):
        require(
            type(group) is dict and set(group) == {"group_id", "rewards", "components"},
            f"groups[{i}] needs group_id, rewards, components",
        )
        require(
            is_text(group["group_id"]) and group["group_id"] not in ids,
            f"groups[{i}].group_id must be unique text",
        )
        ids.add(group["group_id"])
        _reals(group["rewards"], f"groups[{i}].rewards")
        _check_components(group["components"], len(group["rewards"]), f"groups[{i}].components")
        stats = group_reward_stats(group["rewards"])
        all_fail = None
        if failure_components is not None:
            names = set(group["components"][0])
            require(set(failure_components) <= names, f"groups[{i}] lacks a failure component")
            all_fail = all(any(m[c] != 0 for c in failure_components) for m in group["components"])
        rows.append(
            {
                "group_id": group["group_id"],
                **stats,
                "all_fail": all_fail,
                "deciding_component": deciding_component(group["components"], group["rewards"]),
            }
        )
    rows.sort(key=lambda r: r["group_id"])
    estimable = [r for r in rows if r["tied"] != NOT_ESTIMABLE]
    return {
        "groups": rows,
        "n_groups": len(rows),
        "tied_groups": sum(r["tied"] is True for r in estimable),
        "mixed_groups": sum(r["tied"] is False for r in estimable),
        "all_fail_groups": None if failure_components is None else sum(r["all_fail"] for r in rows),
        "single_member_groups": len(rows) - len(estimable),
    }


def _order(values):
    return [
        (values[i] > values[j]) - (values[i] < values[j])
        for i in range(len(values))
        for j in range(i + 1, len(values))
    ]


def flip_share(pairs):
    """Share of groups whose member order changes on re-simulation (L1)."""
    require(type(pairs) is list, "pairs must be a list")
    flipped = []
    for i, pair in enumerate(pairs):
        require(
            type(pair) is dict and set(pair) == {"group_id", "original", "resimulated"},
            f"pairs[{i}] needs group_id, original, resimulated",
        )
        _reals(pair["original"], "original")
        _reals(pair["resimulated"], "resimulated")
        require(
            len(pair["original"]) == len(pair["resimulated"]) >= 2,
            "paired groups must align with >= 2 members",
        )
        if _order(pair["original"]) != _order(pair["resimulated"]):
            flipped.append(pair["group_id"])
    if not pairs:
        return {"n_pairs": 0, "flipped": [], "share": NOT_ESTIMABLE}
    return {"n_pairs": len(pairs), "flipped": sorted(flipped), "share": len(flipped) / len(pairs)}


def _flatten(value):
    if type(value) is list:
        return [x for item in value for x in _flatten(item)]
    require(is_real(value), "actions must be nested lists of finite numbers")
    return [float(value)]


def action_diversity(group_actions):
    """Mean pairwise Euclidean distance between members' flattened actions (L1)."""
    require(type(group_actions) is list, "group_actions must be a list")
    rows = []
    for i, group in enumerate(group_actions):
        require(
            type(group) is dict and set(group) == {"group_id", "actions"},
            f"group_actions[{i}] needs group_id, actions",
        )
        flat = [_flatten(a) for a in group["actions"]]
        require(len({len(f) for f in flat}) <= 1, "members' actions must share a shape")
        if len(flat) < 2:
            rows.append({"group_id": group["group_id"], "mean_pairwise_distance": NOT_ESTIMABLE})
            continue
        dists = [
            math.sqrt(math.fsum((x - y) ** 2 for x, y in zip(flat[a], flat[b])))
            for a in range(len(flat))
            for b in range(a + 1, len(flat))
        ]
        rows.append(
            {"group_id": group["group_id"], "mean_pairwise_distance": math.fsum(dists) / len(dists)}
        )
    return {"groups": sorted(rows, key=lambda r: r["group_id"])}


def rows_after_first_hard_event(rollouts):
    """Replay rows strictly after each rollout's first hard event (L1)."""
    require(type(rollouts) is list, "rollouts must be a list")
    total = after = unknown = 0
    for i, r in enumerate(rollouts):
        require(
            type(r) is dict
            and set(r) == {"rollout_id", "row_timestamps_us", "first_hard_event_us"},
            f"rollouts[{i}] needs rollout_id, row_timestamps_us, first_hard_event_us",
        )
        rows = r["row_timestamps_us"]
        require(
            type(rows) is list and all(type(t) is int for t in rows),
            "row_timestamps_us must be ints",
        )
        event = r["first_hard_event_us"]
        require(
            event is None or event == "unknown" or type(event) is int,
            "first_hard_event_us must be int, None or 'unknown'",
        )
        total += len(rows)
        if event == "unknown":
            unknown += len(rows)
        elif event is not None:
            after += sum(t > event for t in rows)
    known = total - unknown
    return {
        "rows_total": total,
        "rows_after_first_hard_event": after,
        "rows_unknown": unknown,
        "share": after / known if known else NOT_ESTIMABLE,
    }


def grad_norm_stats(values, *, source):
    """Count, zeros, non-finite tokens and finite min/mean/max (L1).

    ``source`` (required): ``in_process_capture`` (exact floats or ``nan``/``inf``/
    ``-inf`` tokens) or ``log_inferred`` (the trainer's printed ``%.6f`` text). A
    printed ``0.000000`` is a norm below 5e-7 that may still have stepped, so it is
    counted as ``printed_zero`` and ``exact_zero`` is ``None`` (unverifiable).
    """
    require(source in SOURCES, f"source must be one of {SOURCES}")
    require(type(values) is list, "values must be a list")
    finite = []
    nonfinite = printed_zero = 0
    for v in values:
        kind, number = parse_grad_norm(v, source)
        if kind == "nonfinite":
            nonfinite += 1
            continue
        printed_zero += kind == "logged_zero"
        finite.append(number)
    logged = source == "log_inferred"
    return {
        "count": len(values),
        "source": source,
        "exact_zero": None if logged else sum(v == 0.0 for v in finite),
        "printed_zero": printed_zero if logged else None,
        "nonfinite": nonfinite,
        "min": min(finite) if finite else NOT_ESTIMABLE,
        "max": max(finite) if finite else NOT_ESTIMABLE,
        "mean": math.fsum(finite) / len(finite) if finite else NOT_ESTIMABLE,
    }


def changed_parameter_norms(deltas, *, prefixes):
    """Per-module L2 norm of parameter change: ``sqrt(sum of squared tensor norms)`` (L1)."""
    require(
        type(deltas) is dict
        and all(is_text(k) and is_real(v) and v >= 0 for k, v in deltas.items()),
        "deltas must map names to finite norms >= 0",
    )
    require(
        type(prefixes) is list and prefixes and all(is_text(p) for p in prefixes),
        "prefixes must be a nonempty list of names",
    )
    require(len(set(prefixes)) == len(prefixes), "prefixes repeat")
    sums = {p: 0.0 for p in prefixes}
    counts = {p: 0 for p in prefixes}
    unmatched = []
    for name, norm in sorted(deltas.items()):
        matches = [p for p in prefixes if name.startswith(p)]
        require(len(matches) <= 1, f"{name} matches several prefixes")
        if not matches:
            unmatched.append(name)
            continue
        sums[matches[0]] += norm * norm
        counts[matches[0]] += 1
    return {
        "modules": {
            p: {
                "tensors": counts[p],
                # no captured tensor is missing evidence, never "unchanged"
                "l2_change": math.sqrt(sums[p]) if counts[p] else NOT_ESTIMABLE,
            }
            for p in prefixes
        },
        "unmatched_tensors": unmatched,
    }


def clip_fraction(ratios, is_padding, *, clip_low, clip_high):
    """Pinned trainer clip fraction over valid rows (L1)."""
    _reals(ratios, "ratios")
    require(
        type(is_padding) is list
        and len(is_padding) == len(ratios)
        and all(type(p) is bool for p in is_padding),
        "is_padding must align booleans",
    )
    require(
        is_real(clip_low) and is_real(clip_high) and clip_low >= 0 and clip_high >= 0,
        "clip bounds must be finite >= 0",
    )
    valid = [r for r, pad in zip(ratios, is_padding) if not pad]
    if not valid:
        return {"valid_rows": 0, "clipped_rows": 0, "fraction": NOT_ESTIMABLE}
    clipped = sum(r < 1.0 - clip_low or r > 1.0 + clip_high for r in valid)
    return {"valid_rows": len(valid), "clipped_rows": clipped, "fraction": clipped / len(valid)}


def rollout_lag(records):
    """Policy lag ``trainer_step - behaviour_version`` per consumed rollout (L1).

    A behaviour version newer than the trainer step (negative lag) raises
    ``ValueError``, as E05 treats it as ``behaviour_version_from_future``.
    """
    require(type(records) is list, "records must be a list")
    lags, unknown = [], 0
    for i, r in enumerate(records):
        require(
            type(r) is dict and set(r) == {"trainer_step", "behaviour_version"},
            f"records[{i}] needs trainer_step, behaviour_version",
        )
        require(is_count(r["trainer_step"]), "trainer_step must be an int >= 0")
        if r["behaviour_version"] is None:
            unknown += 1
            continue
        require(is_count(r["behaviour_version"]), "behaviour_version must be an int >= 0 or None")
        require(
            r["behaviour_version"] <= r["trainer_step"],
            f"records[{i}]: behaviour_version from the future (exceeds trainer_step)",
        )
        lags.append(r["trainer_step"] - r["behaviour_version"])
    return {
        "n": len(records),
        "unknown": unknown,
        "max_lag": max(lags) if lags else NOT_ESTIMABLE,
        "mean_lag": sum(lags) / len(lags) if lags else NOT_ESTIMABLE,
        "histogram": {str(k): lags.count(k) for k in sorted(set(lags))},
    }


def useful_samples_per_gpu_hour(valid_samples, paid_seconds, gpu_count):
    """Valid samples per paid GPU-hour: ``samples / (seconds * gpus / 3600)`` (L1)."""
    require(is_count(valid_samples), "valid_samples must be an int >= 0")
    require(type(gpu_count) is int and gpu_count >= 1, "gpu_count must be a positive int")
    require(
        paid_seconds is None or (is_real(paid_seconds) and paid_seconds >= 0),
        "paid_seconds must be finite >= 0 or None",
    )
    if paid_seconds is None or paid_seconds == 0:
        return {"gpu_hours": None if paid_seconds is None else 0.0, "value": NOT_ESTIMABLE}
    gpu_hours = paid_seconds * gpu_count / 3600
    return {"gpu_hours": gpu_hours, "value": valid_samples / gpu_hours}


def summarize_training_diagnostics(sections, *, scope):
    """Assemble the section-6 diagnostics artifact; absent sections are ``not_run``.

    ``sections`` keys (all optional): ``groups`` (``{groups, failure_components}``),
    ``resimulation_pairs``, ``group_actions``, ``replay_rows``, ``grad_norms``,
    ``parameter_deltas`` (``{deltas, prefixes}``), ``clip`` (``{ratios, is_padding,
    clip_low, clip_high}``), ``kl`` (``{kl_beta, logged, source}``), ``grad_norms``
    (``{values, source}``), ``lag``, ``throughput`` (``{valid_samples, paid_seconds,
    gpu_count}``).

    Section status is ``computed``, ``not_run`` or, for ``kl``, the ``kl_status``.
    Artifact status: ``blocked`` on an error issue, ``not_run`` when no section ran,
    ``unverified`` when a section is ``unverified``, else ``passed`` (descriptive only).
    """
    check_scope(scope)
    require(
        type(sections) is dict and set(sections) <= set(SECTIONS),
        f"sections keys must be within {SECTIONS}",
    )
    issues = Issues()
    results = {}
    for name in SECTIONS:
        if name not in sections:
            results[name] = {"status": "not_run"}
            continue
        value = sections[name]
        if name == "groups":
            require(
                type(value) is dict and set(value) == {"groups", "failure_components"},
                "groups section needs groups, failure_components",
            )
            result = classify_groups(
                value["groups"], failure_components=value["failure_components"]
            )
            if value["failure_components"] is None:
                issues.add(
                    "failure_components_unregistered",
                    "groups.failure_components",
                    "Hard-event components are an owner input.",
                )
        elif name == "resimulation_pairs":
            result = flip_share(value)
        elif name == "group_actions":
            result = action_diversity(value)
        elif name == "replay_rows":
            result = rows_after_first_hard_event(value)
        elif name == "grad_norms":
            require(
                type(value) is dict and set(value) == {"values", "source"},
                "grad_norms needs values, source",
            )
            result = grad_norm_stats(value["values"], source=value["source"])
        elif name == "parameter_deltas":
            require(
                type(value) is dict and set(value) == {"deltas", "prefixes"},
                "parameter_deltas needs deltas, prefixes",
            )
            result = changed_parameter_norms(value["deltas"], prefixes=value["prefixes"])
        elif name == "clip":
            require(
                type(value) is dict
                and set(value) == {"ratios", "is_padding", "clip_low", "clip_high"},
                "clip needs ratios, is_padding, clip_low, clip_high",
            )
            result = clip_fraction(
                value["ratios"],
                value["is_padding"],
                clip_low=value["clip_low"],
                clip_high=value["clip_high"],
            )
        elif name == "kl":
            require(
                type(value) is dict and set(value) == {"kl_beta", "logged", "source"},
                "kl needs kl_beta, logged, source",
            )
            result = kl_status(value["kl_beta"], value["logged"], source=value["source"])
            if result["kl_status"] == "nonfinite":
                issues.add(
                    "kl_nonfinite_logged",
                    "kl.logged",
                    "A logged kl_avg is not finite.",
                    severity="warning",
                )
            results[name] = {"status": result["kl_status"], **result}
            continue
        elif name == "lag":
            result = rollout_lag(value)
        else:
            require(
                type(value) is dict
                and set(value) == {"valid_samples", "paid_seconds", "gpu_count"},
                "throughput needs valid_samples, paid_seconds, gpu_count",
            )
            result = useful_samples_per_gpu_hour(
                value["valid_samples"], value["paid_seconds"], value["gpu_count"]
            )
        results[name] = {"status": "computed", **result}
    section_states = {r["status"] for r in results.values()}
    if issues.has("error"):
        status = "blocked"
    elif section_states == {"not_run"}:
        status = "not_run"
    elif "unverified" in section_states:
        status = "unverified"
    else:
        status = "passed"
    return artifact(
        "s4_training_diagnostics",
        module=__name__,
        module_file=__file__,
        scope=scope,
        inputs={"sections": sections},
        units={
            "rewards": "reward units of the registered arm",
            "distances": "action units (as captured)",
            "grad_norm": "reduced L2 gradient norm",
            "l2_change": "parameter units",
            "lag": "policy steps",
            "throughput": "valid samples per paid GPU-hour",
        },
        status=status,
        issues=issues,
        body={"diagnostic_only": True, "sections": results},
    )
