"""E04 old/new log-density replay pairing and E01 two-run repeatability (G3; protocol-design §4).

Serves S4.2-S4.3 [C] (G9, G7) under D-024. Agent-authored evaluator engineering (D-018). Both
checks read capture-derived artifacts (``teacher_rl.records`` / ``teacher_rl.adapters``); they
compare, they never tune, select or widen anything.

**E04** (``check_logprob_replay``): pair, row by row, the behaviour log-density the rollout
stored (``minibatch_started.old_logprobs``; pinned packer ``old_logprob.reshape(1)`` float32,
``alpagym@972d160 cosmos/packer.py:203-214``) with the trainer's replayed log-density
(``replay_forward.log_probs``; ``trainer.py:401-404, 436-445``) while the trainer still holds
the behaviour weights. Preconditions, each fail-closed:

- ``trainer_records``: an ``s4_trainer_step_records`` v1 artifact whose status is ``passed`` or
  ``not_estimable`` (row lists do not need the E07/E20 inputs); ``blocked``/``not_run`` refuse.
- mu = 1: the stream's captured ``trainer_configured.iterations`` must be exactly 1
  (``mu_not_one``; ``None`` is ``mu_not_captured``). E04's injected configuration (lane-r3 §9).
- Before the first optimizer step: per stream, minibatches are taken in capture order up to
  and including the first one whose ``optimizer.step`` was ``taken`` (its forward precedes its
  own update); later ones are excluded. A minibatch whose step outcome was not observed ends
  the window after it (``optimizer_step_unobserved``): later rows cannot be shown to use the
  behaviour weights.
- Equal behaviour version: ``trace_integrity`` must be a passed ``s4_trace_integrity`` of the
  ``strict_parity`` cohort (every trainer-received rollout generated at ``expected_version``;
  ``records.check_trace_integrity``, which reads the version at generation, never the packer's
  constant ``weight_version`` 0), bound to the same capture streams (every trainer-records
  stream with its chain head appears among its inputs), and ``trainer_weight_version`` - the
  version of the weights the trainer holds before its first optimizer step, a required run
  input (no default) - must equal it (``behaviour_version_mismatch``). The declaration is
  cross-checked against the captured ``trainer_step`` of each stream's first in-window step
  (``trainer_weight_version_check``): pinned AlpaGym reads ``current_step`` in
  ``weight_version`` units (``rollout_filter.py:36-38, 61-63``) while pinned Cosmos-RL
  increments it before dispatching the step's data (``status.py:1346, 1397``), so the held
  version is ``current_step`` or ``current_step - 1``. A declaration matching neither blocks
  (``trainer_weight_version_contradicted``); one matching either is a warning
  (``trainer_weight_version_unverified``) naming the readings; which one holds is L3.
- ``tolerance``: the registered owner tolerance in ``teacher_rl.parity.validate_tolerance``
  form, applied with parity's documented conjunction rule (``|n - o| <= atol + rtol * |o|``
  when either is given; float32 ``ulp(n, o) <= max_ulp`` when given). ``None`` is
  ``tolerance_unregistered`` and **blocks**; every per-row fact is still computed.

Rows: old entries must be one-element lists (``old_logprob_shape`` otherwise); padding rows
(``is_padding == [True]``) carry the packer's placeholder ``0.0`` (``packer.py:229-236``) and
are excluded and counted, never paired. A non-finite token blocks (upstream raises
``FloatingPointError``, ``replay_objective.py:32-39``). Captured values come from float32
tensors, so a value that is not exactly representable in float32 blocks
(``value_not_float32``). Per row: difference ``n - o``, absolute and relative error
(``None`` where ``o == 0``), float32 ULP distance (sign-magnitude order, ``+0``/``-0`` are 0
apart), ratio ``exp(n - o)`` and a verdict ``identical`` (bitwise) |
``within_registered_tolerance`` | ``exceeds`` | ``tolerance_unregistered``. The overall
verdict is ``exceeds`` if any row exceeds, else ``tolerance_unregistered`` if any non-identical
row has no tolerance, else ``within_registered_tolerance`` if any row differs, else
``identical``. The verdict is data beside the status. In-window minibatches whose rows could
not be read (``replay_rows_unavailable``: a log-prob blob over budget or not supplied, or
misaligned lists) are counted (``unavailable_minibatches``, ``unavailable_rows``; ``None``
when the row count itself is unknown) and make the status ``unverified``; the verdict is
then withheld (``None``) unless a paired row already ``exceeds``. ``not_estimable`` is a
window with no unreadable minibatch and no non-padding row. Status precedence: ``blocked`` >
``unverified`` > ``not_estimable`` > ``passed``.

**E01** (``check_repeatability``): two ``s4_repeatability_run`` artifacts (built by
``teacher_rl.adapters.repeatability_run``) declared identical in weights, inputs and noise
seeds. The declaration ``{weights, inputs, noise_seeds, rollout_force_determinism}`` is
required: the three identity flags must be ``True`` (otherwise it is not an E01 comparison)
and ``rollout_force_determinism`` (bool, no default) is the declared mode. Each declared
identity is checked against capture: ``base_weights_hash`` equal (weights); the
``effective_config_hash``, ``uv_lock_sha256`` and ``capture_tree_hash`` equal (inputs); the
same unit keys (``teacher_rl.adapters.repeatability_run``: scene, prompt, payload instance,
retry number and start order; session UUIDs and stream labels differ per run and are never a
key); and per unit
both ``session_started.random_seed`` captured and equal (noise seeds). A contradicted
declaration blocks with no verdict. Units are then compared field by field (outcome, reward
total and components, and the decoded simulator reply: success, error, dense series,
aggregates; never the random session or rollout UUIDs, nor completion files, whose content
names the session) by canonical-JSON equality. A field is compared only when both runs
record it as observed (the unit's ``observed`` map; the simulator reply is observed only when
decoded or raised, never when its bytes were not captured, stored or supplied); otherwise it
is listed in ``not_compared_fields`` and never counted as identical (``outputs_not_compared``
warning). Verdict ``not_repeatable`` when any compared field differs (a fact even if other
fields were not compared); ``repeatable`` only when every unit was compared in every field
and agrees; otherwise ``None``. Status: ``blocked`` on any error; ``not_estimable`` when no
unit was compared; ``unverified`` when any field was not compared; else ``passed``.
``expectation`` is
``bitwise_identical`` under ``rollout_force_determinism: true`` and ``none_declared``
otherwise (unseeded rollout RNG: a difference is a fact, not a defect). No threshold exists.
At the training pin AlpaSim session seeds are random and unsettable (``registration
SIMULATOR_SEED_MODE``), so real runs block on ``declared_noise_seeds_differ`` until a seed
injection exists - that is the honest result.

Verification level per function: ``float32_ulp``, ``check_logprob_replay``,
``check_repeatability``: **L1** (literal known answers and negative controls in
``tests/test_s4_replay_checks.py``). Whether actual replays agree is **L3**
(``qualify_on_host``).

Agent technical defaults (D-018): the window rule above (include the minibatch whose own step
is the first taken; stop after an unobserved step outcome), the ``value_not_float32`` check,
excluding padding rows, the overall-verdict and status precedence, withholding a verdict over
an incomplete window, the two-reading trainer-version cross-check, and E01's observed-field
rule. None of them is a tolerance.

This module never imports torch, never runs a model, never chooses or widens a tolerance and
never interprets a verdict as a learning or validation claim.
"""

import math
import re
import struct

from gate.s2_identity import canonical_json_bytes
from teacher_rl.checks import Issues, artifact, check_scope, is_count, is_real
from teacher_rl.parity import validate_tolerance

TOKENS = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}
VERDICTS = ("identical", "within_registered_tolerance", "exceeds", "tolerance_unregistered")
RECORDS_KIND = "s4_trainer_step_records"
TRACE_KIND = "s4_trace_integrity"
RUN_KIND = "s4_repeatability_run"
LAUNCH_HASH_FIELDS = ("effective_config_hash", "uv_lock_sha256", "base_weights_hash",
                      "capture_tree_hash")
INPUT_HASH_FIELDS = ("effective_config_hash", "uv_lock_sha256", "capture_tree_hash")
DECLARATION_KEYS = ("weights", "inputs", "noise_seeds", "rollout_force_determinism")
UNIT_OUTPUT_FIELDS = ("outcome", "reward_total", "reward_components", "simulator")
PINNED = "alpagym@972d160eed0e23d388497851504a3a233fec5879:packages/runtime/src/alpagym_runtime"
_HEX64 = re.compile(r"[0-9a-f]{64}")


def float32_ulp(a, b):
    """ULP distance of two float32-representable finite values (L1)."""

    def ordered(x):
        bits = struct.unpack("<i", struct.pack("<f", x))[0]
        return bits if bits >= 0 else -(bits & 0x7FFFFFFF)

    return abs(ordered(a) - ordered(b))


def _is_f32(x):
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0] == x
    except OverflowError:
        return False


def _value(value):
    """(float, problem) for a captured scalar: real, token, or a problem string."""
    if type(value) is str and value in TOKENS:
        return TOKENS[value], "nonfinite"
    if is_real(value):
        return float(value), None
    return None, "malformed"


def _within(new, old, tol):
    diff = abs(new - old)
    checks = []
    if tol["atol"] is not None or tol["rtol"] is not None:
        checks.append(diff <= (tol["atol"] or 0.0) + (tol["rtol"] or 0.0) * abs(old))
    if tol["max_ulp"] is not None:
        checks.append(float32_ulp(new, old) <= tol["max_ulp"])
    return all(checks)


def _row(new, old, tol):
    diff = new - old
    identical = struct.pack("<d", new) == struct.pack("<d", old)
    try:
        ratio = math.exp(diff)
    except OverflowError:
        ratio = "inf"
    row = {"old": old, "new": new, "difference": diff, "abs_error": abs(diff),
           "rel_error": None if old == 0 else abs(diff) / abs(old),
           "ulp": float32_ulp(new, old), "ratio": ratio}
    if identical:
        row["verdict"] = "identical"
    elif tol is None:
        row["verdict"] = "tolerance_unregistered"
    else:
        row["verdict"] = "within_registered_tolerance" if _within(new, old, tol) else "exceeds"
    return row


def _overall(rows):
    verdicts = {r["verdict"] for r in rows}
    for verdict in ("exceeds", "tolerance_unregistered", "within_registered_tolerance"):
        if verdict in verdicts:
            return verdict
    return "identical" if rows else None


def _stream_inputs(value):
    inputs = value.get("inputs") if type(value) is dict else None
    if type(inputs) is not list:
        return None
    return {(i.get("stream_id"), i.get("chain_head")) for i in inputs
            if type(i) is dict and i.get("artifact_kind") == "s4_capture_stream"}


def _preconditions(trainer_records, trace_integrity, trainer_weight_version, issues):
    """Return the behaviour version when every precondition holds, else ``None``."""
    usable = (type(trainer_records) is dict
              and trainer_records.get("artifact_kind") == RECORDS_KIND
              and trainer_records.get("schema_version") == 1
              and trainer_records.get("status") in ("passed", "not_estimable")
              and type(trainer_records.get("streams")) is list)
    if not usable:
        issues.add("trainer_records_not_usable", "trainer_records",
                   "a passed or not_estimable s4_trainer_step_records v1 is required")
    version = None
    if (type(trace_integrity) is not dict or trace_integrity.get("artifact_kind") != TRACE_KIND
            or trace_integrity.get("status") != "passed"
            or trace_integrity.get("cohort") != "strict_parity"
            or type(trace_integrity.get("expected_version")) is not int):
        issues.add("behaviour_version_unestablished", "trace_integrity",
                   "a passed strict_parity s4_trace_integrity with an integer expected_version "
                   "is required (behaviour version at generation)")
    else:
        version = trace_integrity["expected_version"]
    if usable and version is not None:
        own, bound = _stream_inputs(trainer_records), _stream_inputs(trace_integrity)
        if not own or bound is None or not own <= bound:
            issues.add("capture_binding_mismatch", "trace_integrity",
                       "the trace integrity was not computed over the trainer-records streams "
                       "(stream id and chain head)")
            version = None
    if trainer_weight_version is None:
        issues.add("trainer_weight_version_unregistered", "trainer_weight_version",
                   "the version of the weights the trainer holds before its first optimizer "
                   "step is a required run input (no default)")
        return None
    if not is_count(trainer_weight_version):
        issues.add("trainer_weight_version_invalid", "trainer_weight_version",
                   "must be an int >= 0")
        return None
    if version is not None and trainer_weight_version != version:
        issues.add("behaviour_version_mismatch", "trainer_weight_version",
                   f"trainer weights at version {trainer_weight_version}, rollouts generated "
                   f"at {version}")
        return None
    return version if usable else None


def _version_check(declared, first_steps, issues):
    """Cross-check the declared trainer weight version against captured trainer steps.

    Two pinned readings of ``trainer_step_started.current_step`` exist: AlpaGym calls it the
    trainer's policy step in ``weight_version`` units (``rollout_filter.py:36-38, 61-63``:
    held version = ``current_step``), while Cosmos-RL increments ``current_step`` before it
    dispatches the step's data (``cosmos_rl status.py:1346, 1397``; weights synced after step
    ``N`` carry ``weight_step=N``, ``:1222-1250``: held version = ``current_step - 1``). A
    declaration outside both readings for any stream, or one that needs different readings
    in different streams, contradicts the capture and blocks; otherwise it stays
    ``unverified`` (L3) with the reading(s) common to every stream named (L2).
    """
    if not is_count(declared) or not first_steps:
        return None
    readings, contradicted = None, []
    for sid in sorted(first_steps):
        step = first_steps[sid]
        if not is_count(step):
            contradicted.append(sid)
            continue
        match = {name for name, held in (("current_step", step),
                                         ("current_step_minus_1", step - 1))
                 if held == declared}
        if not match:
            contradicted.append(sid)
        readings = match if readings is None else readings & match
    readings = readings or set()
    if contradicted or not readings:
        issues.add("trainer_weight_version_contradicted", "trainer_weight_version",
                   f"declared {declared} is neither current_step nor current_step - 1 of the "
                   f"first in-window trainer step in {contradicted}" if contradicted else
                   f"declared {declared} needs different readings in different streams; no "
                   "single reading explains every stream")
        status = "contradicted"
    else:
        issues.add("trainer_weight_version_unverified", "trainer_weight_version",
                   f"declared {declared} agrees with reading(s) {sorted(readings)} of the "
                   "captured trainer step; which reading holds at the pin is qualify_on_host",
                   severity="warning")
        status = "unverified"
    return {"declared": declared, "captured_first_trainer_step": dict(sorted(
        first_steps.items())), "status": status, "consistent_readings": sorted(readings)}


def _minibatch_rows(sid, step, position, mb, tol, issues, out):
    path = f"{sid}:trainer_step:{step['trainer_step']}/minibatch:{position}"
    rows = mb.get("rows")
    lists = [mb.get(n) for n in ("old_logprobs", "new_log_probs", "is_padding")]
    if not is_count(rows) or any(type(v) is not list or len(v) != rows for v in lists):
        issues.add("replay_rows_unavailable", path,
                   "old/new log-probs or padding are missing or misaligned (e.g. a blob was "
                   "not supplied); these rows are not paired", severity="warning")
        return 0, 0, rows if is_count(rows) else None
    old_rows, new_rows, pads = lists
    indices = mb.get("row_indices") if type(mb.get("row_indices")) is list else [None] * rows
    padding = 0
    for r in range(rows):
        where = f"{path}/row:{r}"
        pad = pads[r]
        if type(pad) is not list or len(pad) != 1 or type(pad[0]) is not bool:
            issues.add("padding_shape", where, "is_padding must be a one-element bool list")
            continue
        if pad[0]:
            padding += 1
            continue
        old = old_rows[r]
        if type(old) is not list or len(old) != 1:
            issues.add("old_logprob_shape", where,
                       "the packer stores one old log-prob per row (reshape(1))")
            continue
        o, o_bad = _value(old[0])
        n, n_bad = _value(new_rows[r])
        if "malformed" in (o_bad, n_bad):
            issues.add("log_density_malformed", where, "log-density is not a number or token")
            continue
        if "nonfinite" in (o_bad, n_bad):
            issues.add("nonfinite_log_density", where,
                       "a non-finite log-density raises upstream (replay_objective.py:32-39)")
            continue
        if not (_is_f32(o) and _is_f32(n)):
            issues.add("value_not_float32", where,
                       "captured log-densities come from float32 tensors")
            continue
        out.append({"stream_id": sid, "trainer_step": step["trainer_step"],
                    "minibatch": position, "row": r,
                    "pool_row": indices[r] if r < len(indices) else None, **_row(n, o, tol)})
    return rows - padding, padding, 0


def check_logprob_replay(trainer_records, *, trace_integrity, trainer_weight_version,
                         tolerance, scope):
    """E04 per-row old/new log-density pairing (module docstring; L1)."""
    check_scope(scope)
    issues = Issues()
    tol = None
    if tolerance is None:
        issues.add("tolerance_unregistered", "tolerance",
                   "the E04 log-density tolerance is a required owner input (no default)")
    else:
        try:
            tol = validate_tolerance(tolerance)
        except ValueError as exc:
            issues.add("tolerance_invalid", "tolerance", str(exc))
    version = _preconditions(trainer_records, trace_integrity, trainer_weight_version, issues)
    rows, windows, padding_rows, first_steps = [], {}, 0, {}
    unavailable_mbs, unavailable_rows = 0, 0
    streams = trainer_records.get("streams") if type(trainer_records) is dict else None
    for stream in streams if type(streams) is list else []:
        sid = stream.get("stream_id")
        iterations = (stream.get("configured") or {}).get("iterations")
        if iterations is None:
            issues.add("mu_not_captured", f"{sid}", "trainer_configured iterations not captured")
            continue
        if iterations != 1:
            issues.add("mu_not_one", f"{sid}", f"E04 needs mu = 1, the run has {iterations}")
            continue
        window = {"minibatches": 0, "unavailable_minibatches": 0, "ended_by": "end_of_capture",
                  "first_step_at": None}
        done = False
        for step in stream.get("steps") or []:
            for position, mb in enumerate(step.get("minibatches") or []):
                if done:
                    break
                if window["minibatches"] == 0:
                    first_steps[sid] = step.get("trainer_step")
                window["minibatches"] += 1
                _paired, padded, gone = _minibatch_rows(sid, step, position, mb, tol, issues,
                                                        rows)
                padding_rows += padded
                if gone != 0:
                    window["unavailable_minibatches"] += 1
                    unavailable_mbs += 1
                    unavailable_rows = None if gone is None or unavailable_rows is None else (
                        unavailable_rows + gone)
                outcome = mb.get("optimizer_step")
                if outcome == "taken":
                    window.update(ended_by="first_optimizer_step",
                                  first_step_at={"trainer_step": step["trainer_step"],
                                                 "minibatch": position})
                    done = True
                elif outcome != "skipped":
                    issues.add("optimizer_step_unobserved",
                               f"{sid}:trainer_step:{step['trainer_step']}/minibatch:{position}",
                               "the step outcome was not observed; the window ends here",
                               severity="warning")
                    window["ended_by"] = "optimizer_step_unobserved"
                    done = True
            if done:
                break
        windows[sid] = window
    version_check = _version_check(trainer_weight_version, first_steps, issues)
    verdict = _overall(rows)
    if unavailable_mbs and verdict != "exceeds":
        verdict = None  # an incomplete window cannot show identity or tolerance
    if issues.has("error"):
        status = "blocked"
    elif unavailable_mbs:
        status = "unverified"
    elif not rows:
        status = "not_estimable"
    else:
        status = "passed"
    return artifact(
        "s4_e04_logprob_replay", module=__name__, module_file=__file__, scope=scope,
        inputs={"trainer_records": trainer_records, "trace_integrity": trace_integrity,
                "trainer_weight_version": trainer_weight_version, "tolerance": tolerance},
        units={"log_density": "natural-log density as stored (float32 values)",
               "ulp": "float32 units in the last place", "ratio": "exp(new - old)"},
        status=status, issues=issues,
        body={"ticket": "E04", "pinned_source": PINNED, "behaviour_version": version,
              "tolerance": tol, "verdict": verdict, "windows": windows,
              "trainer_weight_version_check": version_check,
              "counts": {"paired_rows": len(rows), "padding_rows_excluded": padding_rows,
                         "unavailable_minibatches": unavailable_mbs,
                         "unavailable_rows": unavailable_rows,
                         "identical": sum(r["verdict"] == "identical" for r in rows),
                         "exceeds": sum(r["verdict"] == "exceeds" for r in rows)},
              "max_abs_error": max((r["abs_error"] for r in rows), default=None),
              "max_ulp": max((r["ulp"] for r in rows), default=None),
              "rows": rows})


# ---------------------------------------------------------------- E01


def _run_ok(run, label, scope, issues):
    if (type(run) is not dict or run.get("artifact_kind") != RUN_KIND
            or run.get("schema_version") != 1 or run.get("status") != "passed"
            or type(run.get("units")) is not dict or type(run.get("launch")) is not dict):
        issues.add("run_not_usable", label, f"a passed {RUN_KIND} v1 is required")
        return False
    if run.get("scope") != scope:
        issues.add("scope_mismatch", label, f"run scope {run.get('scope')!r} != {scope!r}")
        return False
    bad = [f for f in LAUNCH_HASH_FIELDS
           if type(run["launch"].get(f)) is not str or not _HEX64.fullmatch(run["launch"][f])]
    if bad:
        issues.add("launch_identity_missing", label, f"missing/malformed launch hashes {bad}")
        return False
    return True


def _same(a, b):
    return canonical_json_bytes(a) == canonical_json_bytes(b)


def _observed(unit, field):
    """True only when the run recorded ``field`` as observed (no record: not observed)."""
    observed = unit.get("observed")
    return type(observed) is dict and observed.get(field) is True


def check_repeatability(run_a, run_b, *, declaration, scope):
    """E01 two-run repeatability under the declared mode (module docstring; L1)."""
    check_scope(scope)
    issues = Issues()
    mode = None
    if type(declaration) is not dict or set(declaration) != set(DECLARATION_KEYS):
        issues.add("declaration_invalid", "declaration",
                   f"declaration needs exactly {list(DECLARATION_KEYS)} (no defaults)")
    else:
        undeclared = [k for k in ("weights", "inputs", "noise_seeds") if declaration[k] is not True]
        if undeclared:
            issues.add("not_declared_identical", "declaration",
                       f"E01 compares runs declared identical; {undeclared} are not")
        if type(declaration["rollout_force_determinism"]) is not bool:
            issues.add("declaration_invalid", "declaration.rollout_force_determinism",
                       "the declared mode must be a bool")
        else:
            mode = declaration["rollout_force_determinism"]
    ok = _run_ok(run_a, "run_a", scope, issues) & _run_ok(run_b, "run_b", scope, issues)
    units, verdict, differing, not_compared = [], None, [], []
    if ok:
        if run_a.get("run_id") == run_b.get("run_id"):
            issues.add("same_run_twice", "run_b", "E01 needs two distinct runs")
        la, lb = run_a["launch"], run_b["launch"]
        if la["base_weights_hash"] != lb["base_weights_hash"]:
            issues.add("declared_weights_differ", "launch.base_weights_hash",
                       "the runs loaded different base weights")
        changed = [f for f in INPUT_HASH_FIELDS if la[f] != lb[f]]
        if changed:
            issues.add("declared_inputs_differ", "launch", f"input identities differ: {changed}")
        keys_a, keys_b = set(run_a["units"]), set(run_b["units"])
        if keys_a != keys_b:
            issues.add("unit_sets_differ", "units",
                       f"only in run_a {sorted(keys_a - keys_b)}, only in run_b "
                       f"{sorted(keys_b - keys_a)}")
        for key in sorted(keys_a & keys_b):
            ua, ub = run_a["units"][key], run_b["units"][key]
            seeds = (ua.get("random_seed"), ub.get("random_seed"))
            if any(type(s) is not int for s in seeds):
                issues.add("noise_seed_uncaptured", f"unit:{key}",
                           "a session_started random_seed is missing")
            elif seeds[0] != seeds[1]:
                issues.add("declared_noise_seeds_differ", f"unit:{key}",
                           f"session seeds {seeds[0]} != {seeds[1]}")
            compared = [f for f in UNIT_OUTPUT_FIELDS if _observed(ua, f) and _observed(ub, f)]
            missing = [f for f in UNIT_OUTPUT_FIELDS if f not in compared]
            fields = [f for f in compared if not _same(ua.get(f), ub.get(f))]
            units.append({"unit": key, "identical": None if missing and not fields else not fields,
                          "differing_fields": fields, "not_compared_fields": missing})
            if fields:
                differing.append(key)
            if missing:
                not_compared.append(key)
    if ok and not_compared:
        issues.add("outputs_not_compared", "units",
                   f"{len(not_compared)} unit(s) have output fields not observed in both runs "
                   f"({sorted(not_compared)}); absence is never identity", severity="warning")
    if issues.has("error"):
        status = "blocked"
    elif not units:
        status = "not_estimable"
    else:
        status = "unverified" if not_compared else "passed"
        if differing:
            verdict = "not_repeatable"
        elif not not_compared:
            verdict = "repeatable"
    return artifact(
        "s4_e01_repeatability", module=__name__, module_file=__file__, scope=scope,
        inputs={"run_a": run_a, "run_b": run_b, "declaration": declaration},
        units={"units": "attempts keyed by adapters.repeatability_run"},
        status=status, issues=issues,
        body={"ticket": "E01", "declared_mode": None if mode is None else {
                  "rollout_force_determinism": mode},
              "expectation": None if mode is None else (
                  "bitwise_identical" if mode else "none_declared"),
              "verdict": verdict, "compared_units": len(units),
              "differing_units": differing, "units_not_fully_compared": not_compared,
              "units_detail": units})
