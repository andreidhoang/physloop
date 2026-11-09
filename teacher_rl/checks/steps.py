"""Optimizer-step accounting, zero-gradient skip semantics and the pinned LR lambda.

E20 and the torch-free part of E07 (build-spec A9); serves S4.2/S4.3 [C] (G9).
Agent-authored verification of the shipped AlpaGym replay trainer
(``alpagym@972d160:packages/runtime/src/alpagym_runtime/cosmos/trainer.py``):

- ``_run_training_loop`` (:321-360): ``mini_batch_size = min(mini_batch, rows)``,
  ``ceil(rows / mini_batch_size)`` minibatches per iteration, ``iterations`` passes
  with an unseeded ``torch.randperm``; ``train/num_batches`` counts *minibatches*.
- ``all_reduce_states`` (:541-547): ``optimizer.step()`` runs only when the reduced
  gradient norm is not exactly ``0.0``; a non-finite reduced gradient raises.
- ``step_training`` (:211): ``lr_schedulers.step()`` once per trainer step, always,
  even when every minibatch was skipped.
- Cosmos ``build_lr_schedulers`` / ``warmup_stable_decay``
  (``cosmos-rl@d2a2c57:cosmos_rl/policy/trainer/optm/__init__.py:500-620``): linear
  warmup from ``optm_warmup_start_factor`` (Cosmos default 0.0, absent from the
  AlpaGym schema) so warmup >= 2 starts at lr 0; warmup == 1 gives factor 1.0.

Verification level per function:

- ``minibatch_count``, ``expected_optimizer_steps`` - L1 (literal and exhaustive
  agreement with the pinned loop arithmetic).
- ``reconcile_optimizer_steps`` - L1 over captured records; exact counts need the
  in-process capture (H4) on a host (L3). A count inferred from ``grad_norm=%.6f``
  log lines is ``unverified`` whenever a ``0.000000`` is printed: a norm below
  5e-7 prints as zero yet still steps.
- ``check_skip_semantics`` - L1 over parameter/optimizer snapshot digests (the
  digest producer is ``teacher_rl.params``, A10); actual snapshots L3.
- ``resolve_lr_schedule``, ``lr_factor``, ``lr_factors`` - L1 re-derivation checked
  against the pinned source executed in the tests.

Snapshot schema consumed by ``check_skip_semantics`` (exact keys)::

    {"parameters": {name: sha256-hex}, "optimizer_state": {name: {key: sha256-hex}},
     "step_counters": {name: int}}

``parameters`` must be nonempty; ``optimizer_state`` and ``step_counters`` must name
the same nonempty set of optimizer parameters (an entry with no state yet maps to
``{}`` and counter ``0``), all of which appear in ``parameters``. Frozen parameters
outside the optimizer may appear in ``parameters`` only. Otherwise the check is
``blocked`` (``snapshot_empty`` / ``snapshot_keys_mismatch``), never a vacuous pass.

Owner inputs: every learning-rate setting (training steps, warmup, decay ratio and
type, min factor, warmup start factor) is a required argument with no default; the
values here only reproduce what upstream would do with them.

This module never imports torch, never runs or patches a trainer, and never decides
whether a skipped or zero-lr step is acceptable for a scientific claim.
"""

import math
import re

from teacher_rl.checks import Issues, artifact, check_scope, is_count, is_real, require

PINNED_TRAINER = (
    "alpagym@972d160eed0e23d388497851504a3a233fec5879:"
    "packages/runtime/src/alpagym_runtime/cosmos/trainer.py"
)
PINNED_LR = (
    "cosmos-rl@d2a2c57c4bd6496482bc42da19a59b4432705eda:cosmos_rl/policy/trainer/optm/__init__.py"
)
SOURCES = ("in_process_capture", "log_inferred")
DECAY_TYPES = ("linear", "sqrt", "cosine", "none", None)
_LOGGED_NORM = re.compile(r"\d+\.\d{6}|nan|inf")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_STEP_KEYS = {
    "trainer_step",
    "rows",
    "mini_batch",
    "iterations",
    "source",
    "minibatches",
    "reported_num_batches",
}
_SNAPSHOT_KEYS = {"parameters", "optimizer_state", "step_counters"}


def _positive(value, name):
    require(type(value) is int and value >= 1, f"{name} must be a positive int")


def minibatch_count(rows, mini_batch, iterations):
    """Minibatches the pinned loop runs: ``iterations * ceil(rows / min(mb, rows))`` (L1)."""
    _positive(rows, "rows (upstream raises on zero trainable samples)")
    _positive(mini_batch, "mini_batch")
    _positive(iterations, "iterations")
    size = min(mini_batch, rows)
    return iterations * ((rows + size - 1) // size)


def expected_optimizer_steps(rows, mini_batch, iterations, zero_norm_minibatches):
    """E20: ``iterations * ceil(rows / mini_batch) - zero_norm_minibatches`` (L1).

    Unit: ``optimizer.step()`` calls in one trainer step.
    """
    total = minibatch_count(rows, mini_batch, iterations)
    require(
        is_count(zero_norm_minibatches) and zero_norm_minibatches <= total,
        "zero_norm_minibatches must be an int in [0, minibatch count]",
    )
    return total - zero_norm_minibatches


def parse_grad_norm(value, source):
    """Return (kind, number) where kind is exact|nonfinite|logged_zero|logged_nonzero."""
    if source == "in_process_capture":
        if value in ("nan", "inf", "-inf"):
            return "nonfinite", None
        require(is_real(value) and value >= 0, "captured grad_norm must be a finite number >= 0")
        return "exact", float(value)
    require(
        type(value) is str and _LOGGED_NORM.fullmatch(value) is not None,
        "log_inferred grad_norm must be the '%.6f' text the trainer prints",
    )
    if value in ("nan", "inf"):
        return "nonfinite", None
    return ("logged_zero" if float(value) == 0.0 else "logged_nonzero"), float(value)


def reconcile_optimizer_steps(trainer_steps, *, scope):
    """E20: reconcile captured per-minibatch records with the pinned formula.

    Each record: ``trainer_step`` (int), ``rows``, ``mini_batch``, ``iterations``,
    ``source`` (``in_process_capture`` | ``log_inferred``), ``minibatches`` (list of
    ``{grad_norm, optimizer_step_called}``; ``log_inferred`` carries the printed text
    and ``None``), ``reported_num_batches`` (``train/num_batches``, int or None).
    """
    check_scope(scope)
    require(type(trainer_steps) is list and trainer_steps, "trainer_steps must be a nonempty list")
    issues = Issues()
    rows_out = []
    seen = set()
    for index, record in enumerate(trainer_steps):
        require(
            type(record) is dict and set(record) == _STEP_KEYS,
            f"trainer_steps[{index}] needs exactly {sorted(_STEP_KEYS)}",
        )
        require(is_count(record["trainer_step"]), "trainer_step must be an int >= 0")
        require(record["source"] in SOURCES, f"source must be one of {SOURCES}")
        require(type(record["minibatches"]) is list, "minibatches must be a list")
        require(
            record["reported_num_batches"] is None or is_count(record["reported_num_batches"]),
            "reported_num_batches must be an int >= 0 or None",
        )
        step = record["trainer_step"]
        path = f"trainer_step:{step}"
        require(step not in seen, f"trainer_step {step} repeated")
        seen.add(step)
        for name in ("mini_batch", "iterations"):
            _positive(record[name], name)
        require(is_count(record["rows"]), "rows must be an int >= 0")
        row = {"trainer_step": step, "source": record["source"]}
        if record["rows"] == 0:
            issues.add(
                "no_trainable_samples",
                path,
                "Zero trainable rows: the pinned step_training raises before the loop (:196-200).",
            )
            rows_out.append(row)
            continue
        expected_batches = minibatch_count(
            record["rows"], record["mini_batch"], record["iterations"]
        )
        row["expected_minibatches"] = expected_batches
        row["observed_minibatches"] = len(record["minibatches"])
        if len(record["minibatches"]) != expected_batches:
            issues.add(
                "minibatch_count_mismatch",
                path,
                f"{len(record['minibatches'])} minibatches captured, "
                f"pinned formula gives {expected_batches}.",
            )
        if (
            record["reported_num_batches"] is not None
            and record["reported_num_batches"] != expected_batches
        ):
            issues.add(
                "num_batches_mismatch", path, "train/num_batches disagrees with the formula."
            )
        kinds = []
        for position, minibatch in enumerate(record["minibatches"]):
            require(
                type(minibatch) is dict
                and set(minibatch) == {"grad_norm", "optimizer_step_called"},
                f"{path} minibatch {position} needs grad_norm and optimizer_step_called",
            )
            kind, _ = parse_grad_norm(minibatch["grad_norm"], record["source"])
            called = minibatch["optimizer_step_called"]
            if record["source"] == "in_process_capture":
                require(type(called) is bool, "in-process optimizer_step_called must be bool")
            else:
                require(called is None, "log_inferred records cannot know optimizer_step_called")
            if kind == "nonfinite":
                issues.add(
                    "nonfinite_gradient_upstream_raises",
                    f"{path}/minibatch:{position}",
                    "A non-finite reduced gradient raises FloatingPointError upstream (:530-533).",
                )
            elif kind == "exact" and called != (minibatch["grad_norm"] != 0.0):
                issues.add(
                    "skip_rule_violated",
                    f"{path}/minibatch:{position}",
                    "optimizer.step must run iff the exact reduced grad norm != 0.0 (:545-546).",
                )
            kinds.append(kind)
        zero_exact = sum(
            1
            for kind, mb in zip(kinds, record["minibatches"])
            if kind == "exact" and mb["grad_norm"] == 0.0
        )
        logged_zero = kinds.count("logged_zero")
        row["zero_norm_minibatches"] = (
            zero_exact if record["source"] == "in_process_capture" else None
        )
        if record["source"] == "in_process_capture":
            called = sum(mb["optimizer_step_called"] is True for mb in record["minibatches"])
            row["counted_optimizer_steps"] = called
            if len(kinds) == expected_batches and "nonfinite" not in kinds:
                row["expected_optimizer_steps"] = expected_optimizer_steps(
                    record["rows"], record["mini_batch"], record["iterations"], zero_exact
                )
                if row["expected_optimizer_steps"] != called:
                    issues.add(
                        "optimizer_step_count_mismatch", path, "Counted steps differ from formula."
                    )
        else:
            certain = kinds.count("logged_nonzero")
            if logged_zero:
                row["counted_optimizer_steps"] = None
                row["optimizer_steps_bounds"] = [certain, certain + logged_zero]
                issues.add(
                    "unverifiable_log_precision",
                    path,
                    f"{logged_zero} minibatch(es) printed grad_norm=0.000000; a norm < 5e-7 "
                    "prints as zero but still steps, so the exact count needs in-process capture.",
                    severity="warning",
                )
            else:
                row["counted_optimizer_steps"] = certain
        rows_out.append(row)
    if issues.has("error"):
        status = "blocked"
    elif "unverifiable_log_precision" in issues.codes():
        status = "unverified"
    else:
        status = "passed"
    return artifact(
        "s4_e20_optimizer_step_reconciliation",
        module=__name__,
        module_file=__file__,
        scope=scope,
        inputs={"trainer_steps": trainer_steps},
        units={
            "optimizer_steps": "optimizer.step() calls per trainer step",
            "minibatches": "replay minibatches per trainer step (train/num_batches)",
            "grad_norm": "reduced L2 gradient norm (dimensionless)",
        },
        status=status,
        issues=issues,
        body={
            "ticket": "E20",
            "pinned_source": PINNED_TRAINER,
            "trainer_steps": sorted(rows_out, key=lambda r: r["trainer_step"]),
        },
    )


def _check_snapshot(snapshot, path):
    require(
        type(snapshot) is dict and set(snapshot) == _SNAPSHOT_KEYS,
        f"{path} needs exactly {sorted(_SNAPSHOT_KEYS)}",
    )
    params, state, counters = (
        snapshot[k] for k in ("parameters", "optimizer_state", "step_counters")
    )
    require(
        type(params) is dict
        and all(type(v) is str and _HEX64.fullmatch(v) for v in params.values()),
        f"{path}.parameters must map names to sha256 hex",
    )
    require(
        type(state) is dict
        and all(
            type(v) is dict and all(type(d) is str and _HEX64.fullmatch(d) for d in v.values())
            for v in state.values()
        ),
        f"{path}.optimizer_state must map names to {{key: sha256 hex}}",
    )
    require(
        type(counters) is dict and all(is_count(v) for v in counters.values()),
        f"{path}.step_counters must map names to ints >= 0",
    )


def _snapshot_coverage(snapshot, path, issues):
    """Add ``snapshot_empty`` / ``snapshot_keys_mismatch`` for a snapshot that cannot
    show anything (empty maps) or whose optimizer entries do not name parameters."""
    params, state, counters = (
        snapshot[k] for k in ("parameters", "optimizer_state", "step_counters")
    )
    empty = [k for k, v in (("parameters", params), ("optimizer_state", state)) if not v]
    empty += ["step_counters"] if not counters else []
    if empty:
        issues.add(
            "snapshot_empty",
            path,
            f"Empty {empty}: an unchanged empty map shows nothing about the skip.",
        )
        return
    if set(state) != set(counters) or not set(state) <= set(params):
        issues.add(
            "snapshot_keys_mismatch",
            path,
            "optimizer_state and step_counters must name the same parameters, "
            "all present in parameters.",
        )


def _check_scheduler(state, path):
    require(
        type(state) is dict
        and set(state) == {"last_epoch", "lrs"}
        and type(state["last_epoch"]) is int
        and type(state["lrs"]) is list
        and all(is_real(v) for v in state["lrs"]),
        f"{path} needs last_epoch (int) and lrs (list of finite numbers)",
    )


def check_skip_semantics(trainer_step, *, scope):
    """E07 (torch-free): skipped minibatches leave parameters, optimizer moments and
    per-parameter step counters unchanged; the scheduler still advances once.

    ``trainer_step``: ``{trainer_step, scheduler_before, scheduler_after,
    minibatches: [{index, grad_norm, optimizer_step_called, before, after}]}`` with
    snapshots in the schema of the module docstring, captured around each minibatch.
    A stepped minibatch must advance at least one step counter by exactly one and
    none backwards; its weights need not change (lr may be 0 at warmup start).
    """
    check_scope(scope)
    keys = {"trainer_step", "scheduler_before", "scheduler_after", "minibatches"}
    require(
        type(trainer_step) is dict and set(trainer_step) == keys,
        f"trainer_step needs exactly {sorted(keys)}",
    )
    require(is_count(trainer_step["trainer_step"]), "trainer_step must be an int >= 0")
    _check_scheduler(trainer_step["scheduler_before"], "scheduler_before")
    _check_scheduler(trainer_step["scheduler_after"], "scheduler_after")
    minibatches = trainer_step["minibatches"]
    require(type(minibatches) is list and minibatches, "minibatches must be a nonempty list")
    issues = Issues()
    rows = []
    previous_after = None
    for position, mb in enumerate(minibatches):
        mb_keys = {"index", "grad_norm", "optimizer_step_called", "before", "after"}
        require(
            type(mb) is dict and set(mb) == mb_keys,
            f"minibatches[{position}] needs exactly {sorted(mb_keys)}",
        )
        require(mb["index"] == position, "minibatch indices must be 0..n-1 in order")
        require(is_real(mb["grad_norm"]) and mb["grad_norm"] >= 0, "grad_norm must be finite >= 0")
        require(type(mb["optimizer_step_called"]) is bool, "optimizer_step_called must be bool")
        _check_snapshot(mb["before"], f"minibatches[{position}].before")
        _check_snapshot(mb["after"], f"minibatches[{position}].after")
        path = f"minibatch:{position}"
        for side in ("before", "after"):
            _snapshot_coverage(mb[side], f"{path}/{side}", issues)
        before, after = mb["before"], mb["after"]
        if previous_after is not None and before != previous_after:
            issues.add(
                "snapshot_discontinuity",
                path,
                "State changed between consecutive minibatch snapshots.",
            )
        previous_after = after
        if any(set(before[k]) != set(after[k]) for k in _SNAPSHOT_KEYS):
            issues.add(
                "snapshot_keys_changed", path, "Parameter/optimizer/counter key sets differ."
            )
        skip = mb["grad_norm"] == 0.0
        params_changed = before["parameters"] != after["parameters"]
        state_changed = before["optimizer_state"] != after["optimizer_state"]
        counters_changed = before["step_counters"] != after["step_counters"]
        if mb["optimizer_step_called"] == skip:
            issues.add("skip_rule_violated", path, "optimizer.step must run iff grad_norm != 0.0.")
        if skip:
            if params_changed:
                issues.add(
                    "parameters_changed_at_skip", path, "Parameters changed at a skipped step."
                )
            if state_changed:
                issues.add(
                    "optimizer_state_changed_at_skip",
                    path,
                    "Optimizer moments changed at a skipped step.",
                )
            if counters_changed:
                issues.add(
                    "step_counter_changed_at_skip", path, "Step counters changed at a skipped step."
                )
        else:
            shared = set(before["step_counters"]) & set(after["step_counters"])
            deltas = [after["step_counters"][k] - before["step_counters"][k] for k in shared]
            if not deltas or any(d not in (0, 1) for d in deltas) or 1 not in deltas:
                issues.add(
                    "step_counter_not_advanced",
                    path,
                    "A real step must advance a counter by exactly one.",
                )
        rows.append(
            {
                "index": position,
                "expected": "skip" if skip else "step",
                "optimizer_step_called": mb["optimizer_step_called"],
                "parameters_changed": params_changed,
                "optimizer_state_changed": state_changed,
                "step_counters_changed": counters_changed,
            }
        )
    before_epoch = trainer_step["scheduler_before"]["last_epoch"]
    after_epoch = trainer_step["scheduler_after"]["last_epoch"]
    advanced = after_epoch == before_epoch + 1
    if not advanced:
        issues.add(
            "scheduler_not_advanced",
            "scheduler",
            "lr_schedulers.step() runs exactly once per trainer step "
            "even when every minibatch skipped.",
        )
    return artifact(
        "s4_e07_skip_semantics",
        module=__name__,
        module_file=__file__,
        scope=scope,
        inputs={"trainer_step": trainer_step},
        units={"grad_norm": "reduced L2 gradient norm", "lrs": "learning rate per param group"},
        status="blocked" if issues.has("error") else "passed",
        issues=issues,
        body={
            "ticket": "E07",
            "pinned_source": PINNED_TRAINER,
            "trainer_step": trainer_step["trainer_step"],
            "scheduler_advanced": advanced,
            "lrs_after": trainer_step["scheduler_after"]["lrs"],
            "minibatches": rows,
        },
    )


def resolve_lr_schedule(
    *, training_steps, warmup_steps, decay_ratio, decay_type, min_lr_factor, warmup_start_factor
):
    """Resolve warmup/stable/decay step counts exactly as the pinned builder does (L1).

    ``warmup_steps``: an int, or a float (a float <= 1.0 is a *fraction* of
    ``training_steps``, so ``1.0`` means the whole run while ``1`` means one step).
    Warmup longer than the run is clipped to it; ``decay_ratio`` ``None`` decays
    over the remaining steps, otherwise ``round(training_steps * ratio)`` (Python
    half-even) clipped to fit; ``stable_steps = training + 1 - warmup - decay``.
    """
    _positive(training_steps, "training_steps")
    require(
        type(warmup_steps) in (int, float) and is_real(warmup_steps) and warmup_steps >= 0,
        "warmup_steps must be a finite int/float >= 0",
    )
    require(
        decay_ratio is None or (is_real(decay_ratio) and decay_ratio >= 0),
        "decay_ratio must be None or finite >= 0",
    )
    require(decay_type in DECAY_TYPES, f"decay_type must be one of {DECAY_TYPES}")
    require(is_real(min_lr_factor), "min_lr_factor must be a finite number (owner input)")
    require(
        is_real(warmup_start_factor), "warmup_start_factor must be a finite number (owner input)"
    )
    notes = []
    if type(warmup_steps) is float and warmup_steps <= 1.0:
        warmup = int(warmup_steps * training_steps)
        notes.append("float warmup <= 1.0 interpreted as a fraction of training_steps")
    else:
        warmup = int(warmup_steps)
    if warmup > training_steps:
        notes.append(f"warmup {warmup} clipped to training_steps {training_steps}")
        warmup = training_steps
    if decay_ratio is not None:
        decay = round(training_steps * decay_ratio)
        if warmup + decay > training_steps:
            notes.append("decay clipped so warmup + decay fits the run")
            decay = training_steps - warmup
    else:
        decay = training_steps - warmup
    return {
        "training_steps": training_steps,
        "warmup_steps": warmup,
        "stable_steps": training_steps + 1 - warmup - decay,
        "decay_steps": decay,
        "decay_type": decay_type,
        "min_lr_factor": min_lr_factor,
        "warmup_start_factor": warmup_start_factor,
        "notes": notes,
        "pinned_source": PINNED_LR,
    }


_SCHEDULE_KEYS = {
    "training_steps",
    "warmup_steps",
    "stable_steps",
    "decay_steps",
    "decay_type",
    "min_lr_factor",
    "warmup_start_factor",
    "notes",
    "pinned_source",
}


def _check_schedule(schedule):
    """Fail closed on a schedule ``resolve_lr_schedule`` could not have produced."""
    require(
        type(schedule) is dict and set(schedule) == _SCHEDULE_KEYS,
        f"schedule needs exactly {sorted(_SCHEDULE_KEYS)} (use resolve_lr_schedule)",
    )
    _positive(schedule["training_steps"], "schedule.training_steps")
    for key in ("warmup_steps", "stable_steps", "decay_steps"):
        require(is_count(schedule[key]), f"schedule.{key} must be an int >= 0")
    require(
        schedule["stable_steps"]
        == schedule["training_steps"] + 1 - schedule["warmup_steps"] - schedule["decay_steps"],
        "schedule.stable_steps must equal training + 1 - warmup - decay",
    )
    require(
        schedule["decay_type"] in DECAY_TYPES,
        f"schedule.decay_type must be one of {DECAY_TYPES} (upstream raises otherwise)",
    )
    for key in ("min_lr_factor", "warmup_start_factor"):
        require(is_real(schedule[key]), f"schedule.{key} must be a finite number")
    require(schedule["pinned_source"] == PINNED_LR, "schedule.pinned_source must be PINNED_LR")


def lr_factor(step, schedule):
    """Pinned ``warmup_stable_decay`` multiplier for 0-indexed scheduler ``step`` (L1).

    The lr applied during trainer step ``k`` (after ``k`` scheduler steps) is
    ``base_lr * lr_factor(k, schedule)``. ``schedule`` must be a
    ``resolve_lr_schedule`` result; an unknown decay type or inconsistent step counts
    raise ``ValueError`` (the pinned builder raises on an unknown decay type).
    """
    require(is_count(step), "step must be an int >= 0")
    _check_schedule(schedule)
    warmup, stable, decay = (
        schedule["warmup_steps"],
        schedule["stable_steps"],
        schedule["decay_steps"],
    )
    start = schedule["warmup_start_factor"]
    if step < warmup:
        if warmup == 1:
            return 1.0
        progress = float(step) / float(warmup - 1)
        return start + (1.0 - start) * progress
    if step < warmup + stable:
        return 1.0
    require(decay != 0, "decay_steps is zero at a decay step (upstream asserts)")
    progress = float(step + 1 - (warmup + stable)) / decay
    kind = schedule["decay_type"]
    if kind == "linear":
        adjustment = 1 - progress
    elif kind == "sqrt":
        adjustment = 1 - math.sqrt(progress)
    elif kind == "cosine":
        adjustment = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:  # "none" or None (validated above)
        adjustment = 1.0
    floor = schedule["min_lr_factor"]
    return floor + (1 - floor) * adjustment


def lr_factors(schedule, count):
    """Return ``[lr_factor(k) for k in range(count)]``."""
    require(is_count(count), "count must be an int >= 0")
    return [lr_factor(k, schedule) for k in range(count)]
