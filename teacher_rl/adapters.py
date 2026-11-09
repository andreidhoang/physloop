"""Adapters between S4 artifacts and checks: E07 digests, E16 pin and capture-to-checks (G3).

Tickets: E07 (skip semantics need parameter, optimizer-moment and step-counter digests captured
around each minibatch), E16 (train/gate adapter audit against the pinned gate driver
``alpamayo1_5`` commit); build-spec B3 item (6); G3 (completeness-audit high gap "S4.2-S4.4
capture to checks": captured trainer and simulator evidence had no consumer) for E01, E03,
E07, E10-E14 and E20. Serves S4.1-S4.4/S4.6 [C] (G9, G7) under D-024. Agent-authored evaluator
engineering (D-018).

``teacher_rl.params`` (A10) emits ``s4_param_snapshot`` and ``s4_optimizer_state_snapshot``
artifacts; ``teacher_rl.checks.steps.check_skip_semantics`` (A9) consumes the digest schema::

    {"parameters": {name: sha256}, "optimizer_state": {name: {key: sha256}},
     "step_counters": {name: int}}

Mapping rules (fail closed; ``AdapterError`` names the defect):

- ``parameters``: every snapshot entry of kind ``parameter`` (buffers are excluded: the
  schema names parameters). A ``state_dict`` snapshot (kind ``state``) cannot tell a
  parameter from a buffer and is refused, never guessed.
- ``optimizer_state`` / ``step_counters``: the name set is every member of every optimizer
  ``param_groups[*].params`` - not only the names with state, because
  ``optimizer_state_snapshot`` drops parameters whose state is still empty while
  ``check_skip_semantics`` requires both maps to name the same optimizer parameters. A name
  without state maps to ``{}`` and counter ``0`` (the documented convention). A state entry
  that is a tensor identity contributes its ``sha256``; a JSON scalar contributes the SHA-256
  of its canonical JSON bytes. ``step`` must be an integral, finite, non-negative number
  (a one-element tensor's ``value`` or a plain int); anything else is refused.
- Both snapshots must be ``passed`` artifacts of the same scope.

Driver pin: ``teacher_rl.pins`` does not carry the gate driver's ``alpamayo1_5`` commit; the
engineering pin lives in ``gate.s2_runtime.ALPAMAYO15_SHA`` (the S2 gate path). If a later
``teacher_rl.pins`` revision adds an ``alpamayo1_5`` entry, ``driver_commit_pin`` requires the
two to agree and refuses otherwise (one pin, two homes, never silently diverging).

Capture to checks (G3). Inputs are capture streams as ``teacher_rl.records.streams_from_reports``
shapes ``read_stream`` reports (each with its reader verdict; a stream the reader did not pass
blocks) and the capture blob store as ``{sha256: bytes}`` (each blob re-hashed at consumption).

- ``decode_simulate_calls``: every ``simulate_call`` ``reply_blob`` is decoded by
  ``teacher_rl.simreturn`` and cross-checked against the event's own reply summary (sessions,
  success, rollout UUID, metric names, error text allowing the capture's ``...[+N]``
  truncation) and the request's sessions; a contradiction blocks. A reply the capture did not
  keep (``capture_simulator_bytes: false``, a blob over budget or not supplied) is a warning
  and that session has no decoded rollout: missing is never empty. ``simulate_raised`` calls
  are listed. A session simulated twice blocks. The pinned reader uses only
  ``rollout_returns[0]`` (``streaming_worker.py:232``); further returns are a warning.
- ``semantics_from_capture``: E10 ``reconcile_metric_domains`` and E11 ``contact_predicates``
  per decoded successful rollout; E12 and E13 are ``not_run`` because a ``SimulationReturn``
  carries no executed or ground-truth poses; E14 ``summarize_validity`` over the ledger
  attempts that have a decoded reply, with ``failure_cause = "<cause_type>: <cause_message>"``
  (the form ``CAUSE_PREFIXES`` reads), the reply's aggregates (``None`` when the map is empty,
  as the pinned reader), ``img_is_black_max`` = the maximum of the dense ``img_is_black``
  values (the pinned offline modifier reads values, not ``valid``) and
  ``max_dist_to_gt_trajectory_m`` = the maximum of the dense series named by the owner's
  ``deviation_metric`` (the deviation convention is Huy's; ``None`` leaves it unknown).
  Log-density and re-sample indicators are not in the reply and stay unknown.
- ``step_checks_from_records``: F5's ``e20_trainer_steps`` (one ``reconcile_optimizer_steps``
  per stream, never merged) and ``e07_trainer_steps`` (``check_skip_semantics`` per step).
  A step whose E07 input could not be assembled because its snapshots are unhashed
  (``parameter_hashing: metadata_only``) blocks (``snapshot_unhashed``): never a pass.
- ``e03_from_capture``: the pre/post ``parameter_snapshot`` events of one step of one stream
  -> ``records.param_snapshot_artifact`` (full-identity hashes only; ``local_shard`` or
  unhashed refuse with the records code) -> ``params.diff`` -> ``params.check_trainable_set``
  with the trainable/frozen prefix lists as required inputs (no default).
- ``repeatability_run``: one run's E01 record (``teacher_rl.checks.replay``): the launch
  identity hashes, and per attempt its session seed (``session_started``), ledger outcome and
  reward, and decoded simulator reply, keyed
  ``<scene>|prompt:<p>|instance:<i>|attempt:<a>|start:<k>`` where ``k`` is the start order
  among attempts with the same payload instance and retry number (siblings share
  ``attempt``). Neither the session UUID nor the stream label is a key: both differ on every
  run (the capture labels streams ``<role>-<pid>-<uuid12>``, ``capture/entrypoint.py``
  ``main``); the stream is kept in the unit as data. Two rollout streams binding the same
  payload instance collide and block (``unit_key_repeated``) rather than being paired by a
  random label. Sibling start order is observed thread scheduling, not controlled: a swapped
  order pairs different siblings and shows up as differing seeds or outputs, never as a false
  identity (agent technical defaults, D-018). Each unit records ``simulator_reply``
  (``decoded`` | ``raised`` | ``reply_unavailable`` | ``undecodable`` | ``conflicting_calls``
  | ``no_simulate_record``) and ``observed`` per compared field: the simulator field is
  observed only for ``decoded`` or ``raised``, and outcome/reward only when a terminal record
  exists. The decoder's warnings (``reply_bytes_not_captured``, ``reply_blob_not_stored``,
  ``reply_blob_not_supplied``) are copied into the run's issues with
  ``simulator_reply_unobserved``; missing is never empty, and an unobserved field is never
  counted as identical.

Verification level per function:

- ``skip_snapshot``, ``minibatch_record``, ``trainer_step_input``, ``scheduler_state``,
  ``check_skip``, ``driver_commit_pin``: **L1** (pure mapping with literal known answers and
  negative controls in ``tests/test_s4_adapters.py``).
- ``audit_train_gate_paths``: **L2** (delegates to ``checks.audit.audit_adapter_paths``,
  whose path values are captured from pinned configs); host resolution is L3.
- ``decode_simulate_calls``, ``semantics_from_capture``, ``step_checks_from_records``,
  ``e03_from_capture``, ``repeatability_run``: **L1** plumbing over fake captures written
  through the real ``EventSink``/``read_stream`` (``tests/test_s4_adapters.py``), **L2**
  where they decode pinned wire bytes; behaviour on a real capture is **L3**
  (``qualify_on_host``).

This module never imports torch, never runs or patches a trainer, never reads files (other
than hashing its own source for producer identity) or the network, never fills an owner input
(E14 required metrics, the deviation convention, E03 prefix lists stay required), and never
decides whether a skipped step, a metric difference, a repeatability difference or an adapter
difference is acceptable.
"""

import hashlib
import math
import re
from pathlib import Path

from gate.s2_identity import canonical_json_bytes, identity_hash
from gate.s2_runtime import ALPAMAYO15_SHA
from teacher_rl import params, pins, records, simreturn
from teacher_rl.checks import audit, semantics, steps

SNAPSHOT_KIND = "s4_param_snapshot"
OPTIMIZER_KIND = "s4_optimizer_state_snapshot"
SCOPES = ("synthetic", "public_source_fixture", "actual")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX40 = re.compile(r"[0-9a-f]{40}")


class AdapterError(ValueError):
    """An input artifact cannot be adapted; ``code`` is a stable snake_case reason."""

    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code = code


def _passed(artifact, kind, label):
    if (type(artifact) is not dict or artifact.get("artifact_kind") != kind
            or artifact.get("schema_version") != 1 or artifact.get("status") != "passed"
            or artifact.get("scope") not in SCOPES):
        raise AdapterError("input_invalid", f"{label} is not a passed {kind} v1 artifact")
    return artifact


def _digest(value, path):
    """sha256 of a state entry: a tensor identity's own hash, else canonical JSON bytes."""
    if type(value) is dict:
        sha = value.get("sha256")
        if type(sha) is not str or _HEX64.fullmatch(sha) is None:
            raise AdapterError("state_entry_invalid", f"{path} is an object without sha256")
        return sha
    if value is None or type(value) in (bool, int, str, float, list):
        try:
            return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
        except ValueError as exc:
            raise AdapterError("state_entry_invalid", f"{path}: {exc}") from None
    raise AdapterError("state_entry_invalid", f"{path} has type {type(value).__name__}")


def _step_count(value, path):
    raw = value.get("value") if type(value) is dict else value
    if type(value) is dict and "value" not in value:
        raise AdapterError("step_counter_unreadable",
                           f"{path} is a multi-element tensor; no step count")
    if type(raw) is bool or type(raw) not in (int, float):
        raise AdapterError("step_counter_unreadable", f"{path} is not a number")
    if not math.isfinite(raw) or raw < 0 or raw != int(raw):
        raise AdapterError("step_counter_unreadable",
                           f"{path} must be a finite non-negative integral count, got {raw!r}")
    return int(raw)


def skip_snapshot(param_snapshot, optimizer_snapshot):
    """One ``check_skip_semantics`` snapshot from A10 parameter + optimizer artifacts (L1)."""
    _passed(param_snapshot, SNAPSHOT_KIND, "param_snapshot")
    _passed(optimizer_snapshot, OPTIMIZER_KIND, "optimizer_snapshot")
    if param_snapshot["scope"] != optimizer_snapshot["scope"]:
        raise AdapterError("scope_mismatch", "snapshots come from different scopes")
    if param_snapshot.get("source") != "module":
        raise AdapterError("parameter_kind_unknown",
                           "a state_dict snapshot cannot separate parameters from buffers")
    tensors = param_snapshot.get("tensors")
    if type(tensors) is not dict or not tensors:
        raise AdapterError("input_invalid", "param_snapshot has no tensors")
    parameters = {}
    for name, entry in tensors.items():
        if type(entry) is not dict or entry.get("kind") not in ("parameter", "buffer"):
            raise AdapterError("input_invalid", f"tensor {name!r} has no parameter/buffer kind")
        if entry["kind"] == "parameter":
            sha = entry.get("sha256")
            if type(sha) is not str or _HEX64.fullmatch(sha) is None:
                raise AdapterError("input_invalid", f"tensor {name!r} has no sha256")
            parameters[name] = sha
    groups = optimizer_snapshot.get("param_groups")
    state = optimizer_snapshot.get("state")
    if type(groups) is not list or type(state) is not dict:
        raise AdapterError("input_invalid", "optimizer_snapshot lacks param_groups/state")
    members = []
    for index, group in enumerate(groups):
        params = group.get("params") if type(group) is dict else None
        if type(params) is not list or not all(type(n) is str and n for n in params):
            raise AdapterError("input_invalid", f"param_groups[{index}].params is malformed")
        members.extend(params)
    if len(set(members)) != len(members):
        raise AdapterError("optimizer_param_repeated", "a parameter is in two param groups")
    unknown_state = sorted(set(state) - set(members))
    if unknown_state:
        raise AdapterError("optimizer_state_unowned",
                           f"state for names outside param_groups: {unknown_state}")
    optimizer_state, counters = {}, {}
    for name in sorted(members):
        entry = state.get(name, {})
        if type(entry) is not dict:
            raise AdapterError("input_invalid", f"state[{name!r}] is not an object")
        optimizer_state[name] = {key: _digest(value, f"state/{name}/{key}")
                                 for key, value in sorted(entry.items())}
        counters[name] = _step_count(entry["step"], f"state/{name}/step") if (
            "step" in entry) else 0
    return {"parameters": dict(sorted(parameters.items())),
            "optimizer_state": optimizer_state, "step_counters": counters}


def scheduler_state(scheduler):
    """``{last_epoch, lrs}`` from a torch LR scheduler object (duck-typed; L1)."""
    last_epoch = getattr(scheduler, "last_epoch", None)
    getter = getattr(scheduler, "get_last_lr", None)
    if type(last_epoch) is not int or not callable(getter):
        raise AdapterError("scheduler_unreadable", "scheduler needs int last_epoch and "
                                                   "get_last_lr()")
    lrs = getter()
    if type(lrs) not in (list, tuple) or not all(
            type(v) in (int, float) and type(v) is not bool and math.isfinite(v) for v in lrs):
        raise AdapterError("scheduler_unreadable", "get_last_lr() must return finite numbers")
    return {"last_epoch": last_epoch, "lrs": [float(v) for v in lrs]}


def minibatch_record(index, *, grad_norm, optimizer_step_called, before, after):
    """One ``minibatches[]`` row; ``before``/``after`` are ``(param, optimizer)`` pairs."""
    pair_before, pair_after = tuple(before), tuple(after)
    if len(pair_before) != 2 or len(pair_after) != 2:
        raise AdapterError("input_invalid", "before/after are (param, optimizer) pairs")
    return {"index": index, "grad_norm": grad_norm,
            "optimizer_step_called": optimizer_step_called,
            "before": skip_snapshot(*pair_before), "after": skip_snapshot(*pair_after)}


def trainer_step_input(trainer_step, *, scheduler_before, scheduler_after, minibatches):
    """The exact ``check_skip_semantics`` input from adapted minibatch rows (L1)."""
    return {"trainer_step": trainer_step, "scheduler_before": scheduler_before,
            "scheduler_after": scheduler_after, "minibatches": list(minibatches)}


def check_skip(trainer_step, *, scheduler_before, scheduler_after, minibatches, scope):
    """Adapt and run ``checks.steps.check_skip_semantics`` (E07; L1 over the digests)."""
    return steps.check_skip_semantics(
        trainer_step_input(trainer_step, scheduler_before=scheduler_before,
                           scheduler_after=scheduler_after, minibatches=minibatches),
        scope=scope)


def driver_commit_pin():
    """The gate driver's pinned ``alpamayo1_5`` commit (one value; refuses divergence)."""
    pinned = getattr(pins, "UPSTREAM_PINS", {}).get("alpamayo1_5")
    if pinned is not None:
        commit = pinned.get("commit") if type(pinned) is dict else None
        if commit != ALPAMAYO15_SHA:
            raise AdapterError("driver_pin_divergent",
                               f"teacher_rl.pins alpamayo1_5 {commit!r} != gate.s2_runtime "
                               f"{ALPAMAYO15_SHA}")
    if _HEX40.fullmatch(ALPAMAYO15_SHA) is None:
        raise AdapterError("driver_pin_invalid", "the alpamayo1_5 pin is not 40-hex")
    return ALPAMAYO15_SHA


def audit_train_gate_paths(train_path, gate_path, *, scope):
    """E16 audit with ``expected_driver_commit`` wired to the engineering pin (L2)."""
    return audit.audit_adapter_paths(train_path, gate_path,
                                     expected_driver_commit=driver_commit_pin(), scope=scope)


# ---------------------------------------------------------------- capture to checks (G3)

DECODES_KIND = "s4_simulate_call_decodes"
SEMANTICS_KIND = "s4_capture_semantics"
STEP_CHECKS_KIND = "s4_capture_step_checks"
E03_KIND = "s4_capture_e03"
RUN_KIND = "s4_repeatability_run"
G3_SCHEMA_VERSION = 1
LAUNCH_HASH_FIELDS = records.LAUNCH_HASH_FIELDS
_TRUNCATED = re.compile(r"(?s)(.*)\.\.\.\[\+(\d+)\]")
E12_E13_NOT_RUN = {
    "status": "not_run", "reason": "poses_absent_from_simulation_return",
    "message": "SimulationReturn carries metrics only; E12 needs executed and ground-truth "
               "poses and E13 the executed trajectory (driver session record), which the "
               "simulate capture does not hold"}


def _g3_issue(code, path, message, severity="blocking"):
    return {"code": code, "path": path, "severity": severity, "evidence_refs": [],
            "message": message}


def _g3_producer():
    return {"module": __name__,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _check_g3_scope(scope):
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES} (required, no default)")


def _blocking(issues):
    return any(i["severity"] in ("blocking", "error") for i in issues)


def _stream_inputs(streams, issues):
    """Reader verdict of every stream (a stream the reader did not pass blocks)."""
    if type(streams) is not list or not streams:
        issues.append(_g3_issue("no_capture_streams", "$.streams", "no capture stream supplied"))
        return [], []
    inputs, usable = [], []
    for index, stream in enumerate(streams):
        sid = stream.get("stream_id") if type(stream) is dict else None
        if type(sid) is not str or not sid or type(stream.get("events")) is not list:
            issues.append(_g3_issue("malformed_stream", f"$.streams[{index}]",
                                    "a stream needs a stream_id and an events list"))
            continue
        status = stream.get("reader_status")
        if status != "passed":
            issues.append(_g3_issue("stream_not_verified", sid,
                                    f"read_stream verdict is {status!r}, not passed"))
        inputs.append({"artifact_kind": "s4_capture_stream", "stream_id": sid,
                       "chain_head": stream.get("chain_head"), "reader_status": status})
        usable.append(stream)
    ids = [s["stream_id"] for s in usable]
    if len(set(ids)) != len(ids):
        issues.append(_g3_issue("duplicate_stream", "$.streams", "stream IDs must be unique"))
    return sorted(inputs, key=lambda i: i["stream_id"]), sorted(usable,
                                                                 key=lambda s: s["stream_id"])


def _events(stream, issues):
    events = stream["events"]
    seqs = [e.get("seq") if type(e) is dict else None for e in events]
    if any(type(q) is not int for q in seqs) or len(set(seqs)) != len(seqs):
        issues.append(_g3_issue("malformed_event", stream["stream_id"],
                                "every event needs a distinct integer seq"))
        return []
    return sorted(events, key=lambda e: e["seq"])


def _blob_map(blobs, issues):
    if blobs is None:
        return {}
    if type(blobs) is not dict or not all(
            type(k) is str and _HEX64.fullmatch(k) and type(v) is bytes for k, v in blobs.items()):
        issues.append(_g3_issue("blobs_invalid", "$.blobs", "blobs must map sha256 hex to bytes"))
        return {}
    return blobs


def _reply_bytes(info, blobs, where, issues):
    """The captured reply bytes (hash-verified at consumption) or ``None`` with an issue."""
    if info is None:
        issues.append(_g3_issue("reply_bytes_not_captured", where,
                                "no reply_blob (capture_simulator_bytes was false); the reply "
                                "is unavailable", "warning"))
        return None
    if type(info) is not dict or type(info.get("sha256")) is not str or not _HEX64.fullmatch(
            info["sha256"]):
        issues.append(_g3_issue("reply_blob_invalid", where, "reply_blob is not a blob record"))
        return None
    if info.get("stored") is not True:
        issues.append(_g3_issue("reply_blob_not_stored", where,
                                "the sink did not store the reply (over budget)", "warning"))
        return None
    data = blobs.get(info["sha256"])
    if data is None:
        issues.append(_g3_issue("reply_blob_not_supplied", where,
                                f"blob {info['sha256']} was not supplied", "warning"))
        return None
    if hashlib.sha256(data).hexdigest() != info["sha256"] or (
            type(info.get("bytes")) is int and info["bytes"] != len(data)):
        issues.append(_g3_issue("reply_blob_hash_mismatch", where,
                                "the supplied blob does not match its record"))
        return None
    return data


def _text_matches(summary, full):
    if summary == full:
        return True
    match = _TRUNCATED.fullmatch(summary) if type(summary) is str else None
    return bool(match) and full.startswith(match[1]) and len(full) == len(match[1]) + int(
        match[2])


def _summary_problems(event, returns):
    """Differences between the event's own reply summary and the decoded reply."""
    summary = event.get("returns")
    if type(summary) is not list:
        return ["the event carries no reply summary"]
    if len(summary) != len(returns):
        return [f"summary has {len(summary)} returns, the reply {len(returns)}"]
    problems = []
    for i, (s, r) in enumerate(zip(summary, returns, strict=True)):
        if type(s) is not dict:
            problems.append(f"returns[{i}] is not an object")
            continue
        names = sorted(m["name"] for m in r["timestep_metrics"])
        for key, mine, theirs in (("sessions", s.get("sessions"), r["session_uuids"]),
                                  ("success", s.get("success"), r["success"]),
                                  ("rollout_uuid", s.get("rollout_uuid"), r["rollout_uuid"]),
                                  ("timestep_metrics", s.get("timestep_metrics"), names)):
            if mine != theirs:
                problems.append(f"returns[{i}].{key} {mine!r} != decoded {theirs!r}")
        if not _text_matches(s.get("error"), r["error"]):
            problems.append(f"returns[{i}].error differs from the decoded error")
    requested = event.get("sessions")
    decoded = [u for r in returns for u in r["session_uuids"]]
    if requested is not None and requested != decoded:
        problems.append(f"request sessions {requested!r} != reply sessions {decoded!r}")
    return problems


def _g3_status(issues, *, empty=False):
    if _blocking(issues):
        return "blocked"
    return "not_estimable" if empty else "passed"


def decode_simulate_calls(streams, *, blobs, scope):
    """Decode every captured ``simulate_call`` reply (module docstring; L1/L2)."""
    _check_g3_scope(scope)
    issues = []
    inputs, usable = _stream_inputs(streams, issues)
    blob_map = _blob_map(blobs, issues)
    calls, rollouts = [], {}
    for stream in usable:
        sid = stream["stream_id"]
        for event in _events(stream, issues):
            kind, where = event.get("kind"), f"{sid}#{event['seq']}"
            if kind == "simulate_raised":
                calls.append({"stream_id": sid, "seq": event["seq"], "call": event.get("call"),
                              "sessions": event.get("sessions"), "outcome": "raised",
                              "error_type": event.get("error_type"), "decode": None})
                continue
            if kind != "simulate_call":
                continue
            entry = {"stream_id": sid, "seq": event["seq"], "call": event.get("call"),
                     "sessions": event.get("sessions"), "outcome": None, "error_type": None,
                     "decode": None}
            calls.append(entry)
            data = _reply_bytes(event.get("reply_blob"), blob_map, where, issues)
            if data is None:
                entry["outcome"] = "reply_unavailable"
                continue
            decoded = simreturn.decode_simulation_return(data, scope=scope)
            entry["decode"] = {"status": decoded["status"], "input_hashes": decoded[
                "input_hashes"], "versions": decoded["versions"],
                               "codes": sorted({i["code"] for i in decoded["issues"]})}
            for item in decoded["issues"]:
                issues.append({**item, "path": f"{where}/{item['path']}"})
            if decoded["rollout_returns"] is None:
                entry["outcome"] = "undecodable"
                continue
            entry["outcome"] = "decoded"
            problems = _summary_problems(event, decoded["rollout_returns"])
            for problem in problems:
                issues.append(_g3_issue("reply_summary_contradiction", where, problem))
            if len(decoded["rollout_returns"]) > 1:
                issues.append(_g3_issue("extra_rollout_returns", where,
                                        "the pinned reader uses only rollout_returns[0]",
                                        "warning"))
            for ret in decoded["rollout_returns"]:
                if len(ret["session_uuids"]) != 1:
                    issues.append(_g3_issue("rollout_session_count", f"{where}/{ret['index']}",
                                            f"{len(ret['session_uuids'])} session UUIDs; one "
                                            "is required to key the rollout"))
                    continue
                session = ret["session_uuids"][0]
                if session in rollouts:
                    issues.append(_g3_issue("session_simulated_twice", where,
                                            f"session {session} has two decoded replies"))
                    continue
                try:
                    semantic, refusal = simreturn.to_semantics_input(decoded, ret["index"]), None
                except ValueError as exc:
                    semantic, refusal = None, str(exc)
                rollouts[session] = {"stream_id": sid, "seq": event["seq"],
                                     "call": event.get("call"), "index": ret["index"],
                                     "rollout": ret, "semantics_input": semantic,
                                     "semantics_refusal": refusal}
    counts = {"calls": len(calls),
              **{name: sum(c["outcome"] == name for c in calls)
                 for name in ("decoded", "reply_unavailable", "undecodable", "raised")},
              "rollouts": len(rollouts)}
    if usable and not calls:
        issues.append(_g3_issue("no_simulate_calls", "$.streams",
                                "the streams hold no simulate_call record", "warning"))
    body = {"calls": calls, "rollouts": {k: rollouts[k] for k in sorted(rollouts)},
            "counts": counts}
    return {"artifact_kind": DECODES_KIND, "schema_version": G3_SCHEMA_VERSION, "scope": scope,
            "producer": _g3_producer(), "inputs": inputs,
            "units": {"timestamps_us": "microseconds", "seq": "record index within a stream"},
            "status": _g3_status(issues, empty=not rollouts), "issues": issues, **body,
            "decodes_hash": identity_hash(body)}


def _dense_max(rollout, name):
    """Maximum of a dense series' values (``valid`` ignored), ``None`` if absent/non-finite."""
    metric = next((m for m in rollout["timestep_metrics"] if m["name"] == name), None)
    if metric is None or not metric["values"] or any(
            type(v) is str for v in metric["values"]):
        return None
    return max(metric["values"])


def _passed_kind(value, kind, statuses=("passed",)):
    return (type(value) is dict and value.get("artifact_kind") == kind
            and value.get("schema_version") == 1 and value.get("status") in statuses)


def semantics_from_capture(decodes, ledger, *, required_metrics, deviation_metric, scope):
    """E10-E14 over decoded replies and the attempt ledger (module docstring; L1/L2).

    ``required_metrics`` (E14, from the registered reward arm) and ``deviation_metric`` (the
    dense metric whose maximum is the deviation measure) are owner inputs: required keywords
    with no default; ``None`` is passed through and reported, never filled.
    """
    _check_g3_scope(scope)
    issues = []
    if not _passed_kind(decodes, DECODES_KIND, ("passed", "not_estimable")):
        issues.append(_g3_issue("decodes_not_usable", "decodes",
                                "a passed or not_estimable s4_simulate_call_decodes is required"))
    if not _passed_kind(ledger, "s4_attempt_ledger"):
        issues.append(_g3_issue("ledger_not_passed", "ledger", "a passed s4_attempt_ledger is "
                                "required"))
    for name, value in (("decodes", decodes), ("ledger", ledger)):
        if type(value) is dict and value.get("scope") not in (None, scope):
            issues.append(_g3_issue("scope_mismatch", name, f"{value.get('scope')!r} != "
                                                            f"{scope!r}"))
    if deviation_metric is None:
        issues.append(_g3_issue("deviation_convention_unregistered", "deviation_metric",
                                "the deviation measure is an owner input; E14 out_of_envelope "
                                "stays unknown", "warning"))
    elif type(deviation_metric) is not str or not deviation_metric:
        issues.append(_g3_issue("deviation_metric_invalid", "deviation_metric",
                                "must be a dense metric name or None"))
    usable = not _blocking(issues)
    sessions, checks, e14, without_reply = {}, {}, None, []
    if usable:
        outcomes = {a["session_uuid"]: a for a in ledger["attempts"]}
        for session, entry in decodes["rollouts"].items():
            row = {"attempt_outcome": (outcomes.get(session) or {}).get("outcome"),
                   "e10": None, "e11": None, "e12": dict(E12_E13_NOT_RUN),
                   "e13": dict(E12_E13_NOT_RUN), "semantics_refusal": entry["semantics_refusal"]}
            if entry["semantics_input"] is not None:
                try:
                    e10 = semantics.reconcile_metric_domains(entry["semantics_input"],
                                                             scope=scope)
                    e11 = semantics.contact_predicates(entry["semantics_input"], scope=scope)
                except ValueError as exc:
                    issues.append(_g3_issue("semantics_input_rejected", f"session:{session}",
                                            str(exc)))
                else:
                    checks[session] = {"e10": e10, "e11": e11}
                    row["e10"] = {"status": e10["status"],
                                  "codes": sorted({i["code"] for i in e10["issues"]})}
                    row["e11"] = {"status": e11["status"],
                                  "any_contact": e11.get("any_contact"),
                                  "additive_front_lateral": e11.get("additive_front_lateral"),
                                  "codes": sorted({i["code"] for i in e11["issues"]})}
            sessions[session] = row
        attempts = []
        for attempt in sorted(ledger["attempts"], key=lambda a: a["session_uuid"]):
            session = attempt["session_uuid"]
            decoded = decodes["rollouts"].get(session)
            if decoded is None or attempt["outcome"] is None:
                without_reply.append(session)
                continue
            rollout = decoded["rollout"]
            cause = None
            if attempt["outcome"] != "succeeded" and attempt.get("cause_type") is not None:
                cause = f"{attempt['cause_type']}: {attempt.get('cause_message')}"
            attempts.append({
                "attempt_id": session, "payload_id": attempt["payload_id"],
                "session_uuid": session,
                "outcome": "succeeded" if attempt["outcome"] == "succeeded" else "failed",
                "failure_cause": cause,
                "aggregated_metrics": dict(rollout["aggregated_metrics"]) or None,
                "img_is_black_max": _dense_max(rollout, "img_is_black"),
                "max_dist_to_gt_trajectory_m": None if type(deviation_metric) is not str
                else _dense_max(rollout, deviation_metric),
                "old_logprobs_nonfinite": None, "nonfinite_resamples": None})
        if without_reply:
            issues.append(_g3_issue("attempts_without_decoded_reply", "ledger",
                                    f"{len(without_reply)} attempt(s) have no decoded reply or "
                                    "no terminal record; E14 omits them", "warning"))
        if attempts:
            try:
                e14 = semantics.summarize_validity(attempts, required_metrics=required_metrics,
                                                   scope=scope)
            except ValueError as exc:
                issues.append(_g3_issue("e14_input_rejected", "attempts", str(exc)))
    children = [c["e10"]["status"] for c in checks.values()] + [
        c["e11"]["status"] for c in checks.values()] + ([e14["status"]] if e14 else [])
    if _blocking(issues) or "blocked" in children:
        status = "blocked"
    elif not checks and e14 is None:
        status = "not_run"
    elif "unverified" in children or without_reply or e14 is None:
        status = "unverified"
    else:
        status = "passed"
    return {"artifact_kind": SEMANTICS_KIND, "schema_version": G3_SCHEMA_VERSION,
            "scope": scope, "producer": _g3_producer(),
            "inputs": [{"artifact_kind": DECODES_KIND,
                        "identity_hash": decodes.get("decodes_hash")
                        if type(decodes) is dict else None},
                       {"artifact_kind": "s4_attempt_ledger",
                        "identity_hash": ledger.get("ledger_hash")
                        if type(ledger) is dict else None}],
            "units": {"metrics": "as emitted by the pinned AlpaSim scorers"},
            "status": status, "issues": issues,
            "owner_inputs": {"required_metrics": required_metrics,
                             "deviation_metric": deviation_metric},
            "sessions": sessions, "checks": checks, "e14": e14,
            "attempts_without_decoded_reply": without_reply}


def step_checks_from_records(trainer_records, *, scope):
    """E20 and E07 over F5's assembled trainer-step inputs (module docstring; L1)."""
    _check_g3_scope(scope)
    issues, e20, e07, unassembled = [], {}, [], []
    if not _passed_kind(trainer_records, records.TRAINER_RECORDS_KIND,
                        ("passed", "not_estimable", "not_run", "blocked")):
        issues.append(_g3_issue("trainer_records_invalid", "trainer_records",
                                "an s4_trainer_step_records v1 artifact is required"))
    elif trainer_records["status"] == "blocked":
        issues.append(_g3_issue("trainer_records_blocked", "trainer_records",
                                "the trainer records are blocked; no check runs on them"))
    elif trainer_records.get("scope") != scope:
        issues.append(_g3_issue("scope_mismatch", "trainer_records", "scope differs"))
    usable = not _blocking(issues)
    if usable:
        for sid in sorted(trainer_records["e20_trainer_steps"]):
            e20[sid] = steps.reconcile_optimizer_steps(
                trainer_records["e20_trainer_steps"][sid], scope=scope)
        for entry in trainer_records["e07_trainer_steps"]:
            e07.append({"stream_id": entry["stream_id"], "trainer_step": entry["trainer_step"],
                        "hash_comparability": entry["hash_comparability"],
                        "report": steps.check_skip_semantics(entry["input"], scope=scope)})
        for stream in trainer_records["streams"]:
            for step in stream["steps"]:
                for check in ("e20", "e07"):
                    result = step[check]
                    if result["assembly"] == "passed":
                        continue
                    where = f"{stream['stream_id']}:trainer_step:{step['trainer_step']}/{check}"
                    unassembled.append({"stream_id": stream["stream_id"],
                                        "trainer_step": step["trainer_step"], "check": check,
                                        "assembly": result["assembly"],
                                        "reasons": result["reasons"]})
                    if check == "e07" and "snapshot_unhashed" in result["reasons"]:
                        issues.append(_g3_issue(
                            "snapshot_unhashed", where,
                            "parameter_hashing metadata_only: sha256, hash_scope and optimizer "
                            "digests are null, so E07 cannot run (never passed)"))
                    elif result["assembly"] == "blocked":
                        issues.append(_g3_issue(f"{check}_assembly_blocked", where,
                                                f"reasons {result['reasons']}"))
                    else:
                        issues.append(_g3_issue(f"{check}_not_assembled", where,
                                                f"reasons {result['reasons']}", "warning"))
    children = [r["status"] for r in e20.values()] + [r["report"]["status"] for r in e07]
    if _blocking(issues) or "blocked" in children:
        status = "blocked"
    elif not children:
        status = "not_run"
    elif unassembled or "unverified" in children:
        status = "unverified"
    else:
        status = "passed"
    return {"artifact_kind": STEP_CHECKS_KIND, "schema_version": G3_SCHEMA_VERSION,
            "scope": scope, "producer": _g3_producer(),
            "inputs": [{"artifact_kind": records.TRAINER_RECORDS_KIND,
                        "identity_hash": trainer_records.get("records_hash")
                        if type(trainer_records) is dict else None}],
            "units": {"optimizer_steps": "optimizer.step() calls per trainer step"},
            "status": status, "issues": issues, "e20": e20, "e07": e07,
            "unassembled": unassembled}


def _snapshot_events(stream, trainer_step, issues):
    found = {"pre_step": [], "post_step": []}
    for event in _events(stream, issues):
        if (event.get("kind") == "parameter_snapshot" and event.get("phase") in found
                and event.get("trainer_step") == trainer_step):
            found[event["phase"]].append(event)
    return found


def e03_from_capture(streams, *, stream_id, trainer_step, trainable_prefixes, frozen_prefixes,
                     scope):
    """E03 from one step's captured pre/post snapshots (module docstring; L1)."""
    _check_g3_scope(scope)
    issues = []
    inputs, usable = _stream_inputs(streams, issues)
    stream = next((s for s in usable if s["stream_id"] == stream_id), None)
    before = after = diff_report = trainable = None
    if stream is None:
        issues.append(_g3_issue("stream_missing", "stream_id", f"no stream {stream_id!r}"))
    if trainable_prefixes is None or frozen_prefixes is None:
        issues.append(_g3_issue("prefixes_unregistered", "trainable_prefixes",
                                "trainable and frozen prefix lists are required inputs "
                                "(pinned freeze sites; no default)"))
    if stream is not None and not _blocking(issues):
        found = _snapshot_events(stream, trainer_step, issues)
        for phase, events in found.items():
            if len(events) != 1:
                code = "snapshot_missing" if not events else "duplicate_snapshots"
                issues.append(_g3_issue(code, f"{stream_id}:trainer_step:{trainer_step}",
                                        f"{len(events)} {phase} snapshots"))
        if not _blocking(issues):
            pre, post = found["pre_step"][0], found["post_step"][0]
            if pre["seq"] > post["seq"]:
                issues.append(_g3_issue("snapshot_order", stream_id,
                                        "post_step snapshot precedes pre_step"))
            try:
                before = records.param_snapshot_artifact(pre, scope=scope)
                after = records.param_snapshot_artifact(post, scope=scope)
            except records.RecordsError as exc:
                issues.append(_g3_issue(exc.code, f"{stream_id}:trainer_step:{trainer_step}",
                                        str(exc)))
    if before is not None and after is not None and not _blocking(issues):
        diff_report = params.diff(before, after)
        try:
            trainable = params.check_trainable_set(before, trainable_prefixes, frozen_prefixes,
                                                   diff=diff_report)
        except ValueError as exc:
            issues.append(_g3_issue("prefixes_invalid", "trainable_prefixes", str(exc)))
    children = [r["status"] for r in (diff_report, trainable) if r is not None]
    if _blocking(issues) or "blocked" in children:
        status = "blocked"
    else:
        status = "passed" if trainable is not None else "not_run"
    return {"artifact_kind": E03_KIND, "schema_version": G3_SCHEMA_VERSION, "scope": scope,
            "producer": _g3_producer(), "inputs": inputs, "units": {},
            "status": status, "issues": issues, "stream_id": stream_id,
            "trainer_step": trainer_step, "before": before, "after": after,
            "diff": diff_report, "trainable_set": trainable}


_OBSERVED_REPLIES = ("decoded", "raised")


def _session_replies(decodes):
    """``{session: reply outcome}`` from the decoded calls (one outcome per session).

    ``decoded`` and ``raised`` are observations; ``reply_unavailable`` (bytes not captured, not
    stored or not supplied) and ``undecodable`` are not. A session named by calls with
    different outcomes is ``conflicting_calls`` (not an observation).
    """
    seen = {}
    for call in decodes["calls"]:
        for session in _call_sessions(call) or []:
            seen.setdefault(session, set()).add(call["outcome"])
    return {s: next(iter(o)) if len(o) == 1 else "conflicting_calls" for s, o in seen.items()}


def _call_sessions(call):
    """The call's ``sessions`` when absent (``None``) or a list of nonempty strings, else ``False``.

    The capture writes ``[str(uuid), ...]`` (``capture.entrypoint._SimulateProxy``) or omits the
    field when extraction failed; anything else is malformed and never used as a key.
    """
    sessions = call.get("sessions")
    if sessions is None or (type(sessions) is list
                            and all(type(s) is str and s for s in sessions)):
        return sessions
    return False


def _unit_key(attempt):
    """Run-invariant unit key: never the stream label nor the session UUID (both per run)."""
    _sid, _prompt, instance = attempt["payload_id"].rsplit(":", 2)  # "<sid>:<prompt>:<i>"
    return (f"{attempt.get('scene_id')}|prompt:{attempt.get('prompt_idx')}|instance:"
            f"{instance}|attempt:{attempt.get('attempt')}")


def repeatability_run(streams, *, blobs, launch, run_id, scope):
    """One run's E01 record for ``checks.replay.check_repeatability`` (module docstring)."""
    _check_g3_scope(scope)
    issues = []
    decodes = decode_simulate_calls(streams, blobs=blobs, scope=scope)
    ledger = records.build_attempt_ledger(streams if type(streams) is list else [], scope=scope)
    if decodes["status"] == "blocked":
        issues.append(_g3_issue("decodes_blocked", "decodes", "simulator replies are blocked"))
    for item in decodes["issues"]:
        if item["severity"] not in ("blocking", "error"):
            issues.append({**item, "path": f"decodes/{item['path']}"})
    if ledger["status"] != "passed":
        issues.append(_g3_issue("ledger_not_passed", "ledger",
                                f"the attempt ledger is {ledger['status']!r}"))
    if type(run_id) is not str or not run_id:
        issues.append(_g3_issue("run_id_invalid", "run_id", "run_id must be text"))
    hashes = None
    if type(launch) is not dict or any(
            type(launch.get(f)) is not str or not _HEX64.fullmatch(launch[f])
            for f in LAUNCH_HASH_FIELDS) or launch.get("run_id") != run_id:
        issues.append(_g3_issue("launch_invalid", "launch",
                                f"the run's launch record needs {list(LAUNCH_HASH_FIELDS)} and "
                                "its run_id"))
    else:
        hashes = {f: launch[f] for f in LAUNCH_HASH_FIELDS}
    seeds = {}
    _inputs, usable = _stream_inputs(streams, [])
    for stream in usable:
        for event in _events(stream, issues):
            if event.get("kind") != "session_started":
                continue
            session, seed = event.get("session_uuid"), event.get("random_seed")
            if type(seed) is not int or (session in seeds and seeds[session] != seed):
                issues.append(_g3_issue("session_seed_conflict",
                                        f"{stream['stream_id']}#{event['seq']}",
                                        f"session {session} seed {seed!r} is malformed or "
                                        "conflicts"))
                continue
            seeds[session] = seed
    for call in decodes["calls"]:
        if _call_sessions(call) is False:
            issues.append(_g3_issue("simulate_sessions_invalid",
                                    f"decodes/{call['stream_id']}#{call['seq']}",
                                    "a simulate record's sessions must be a list of session "
                                    "UUID strings"))
    replies = _session_replies(decodes)
    raised = {s: c.get("error_type") for c in decodes["calls"] if c["outcome"] == "raised"
              for s in _call_sessions(c) or []}
    units, order, unobserved = {}, {}, []
    for attempt in sorted(ledger["attempts"], key=lambda a: (a["payload_id"], a["started_seq"])):
        slot = (attempt["payload_id"], attempt.get("attempt"))
        order[slot] = order.get(slot, -1) + 1
        key = f"{_unit_key(attempt)}|start:{order[slot]}"
        if key in units:
            issues.append(_g3_issue("unit_key_repeated", key, "two attempts share a unit key "
                                    "(e.g. two rollout streams bind the same payload instance)"))
            continue
        session = attempt["session_uuid"]
        reply = replies.get(session, "no_simulate_record")
        decoded = decodes["rollouts"].get(session)
        if reply == "decoded" and decoded is not None:
            simulator = {k: decoded["rollout"][k] for k in (
                "scenario_id", "success", "error", "timestep_metrics", "aggregated_metrics")}
        elif reply == "raised":
            simulator = {"raised": raised.get(session)}
        else:
            simulator, reply = None, reply if reply != "decoded" else "not_decoded"
        terminal = attempt["outcome"] is not None
        observed = {"outcome": terminal, "reward_total": terminal, "reward_components": terminal,
                    "simulator": reply in _OBSERVED_REPLIES}
        if not observed["simulator"]:
            unobserved.append(key)
        units[key] = {"stream_id": attempt["payload_id"].rsplit(":", 2)[0],
                      "scene_id": attempt.get("scene_id"),
                      "prompt_idx": attempt.get("prompt_idx"), "attempt": attempt.get("attempt"),
                      "session_uuid": session, "random_seed": seeds.get(session),
                      "outcome": attempt["outcome"], "reward_total": attempt.get("reward_total"),
                      "reward_components": attempt.get("reward_components"),
                      "simulator": simulator, "simulator_reply": reply, "observed": observed}
    if unobserved:
        issues.append(_g3_issue("simulator_reply_unobserved", "units",
                                f"{len(unobserved)} unit(s) have no observed simulator reply "
                                f"({sorted(unobserved)}); E01 does not compare that field",
                                "warning"))
    return {"artifact_kind": RUN_KIND, "schema_version": G3_SCHEMA_VERSION, "scope": scope,
            "producer": _g3_producer(),
            "inputs": [{"artifact_kind": DECODES_KIND, "identity_hash": decodes["decodes_hash"]},
                       {"artifact_kind": "s4_attempt_ledger",
                        "identity_hash": ledger["ledger_hash"]}],
            "units_of_measure": {"random_seed": "AlpaSim session seed (driver SessionRequest)"},
            "status": "blocked" if _blocking(issues) else "passed", "issues": issues,
            "run_id": run_id, "launch": hashes, "units": {k: units[k] for k in sorted(units)}}
