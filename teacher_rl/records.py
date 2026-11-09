"""S4 records ledger: payloads, attempts, silent retries, groups, trace integrity and run identity.

Tickets: S4.3 (capture every attempted group, retry and exhausted/dropped payload before upstream
cleanup), E06 (group reconstruction invariant under permutation), E09 (duplicate/omitted
records, misreported or mixed behaviour versions; strict parity cohort), E15 (run identity at
start and resume; duplicate launch), E17 (actual randomness sources per run), lane-r3 finding 6
(reward errors become silent retries with a new session UUID). Agent-authored under D-018/D-024.

Inputs are capture streams: ``{"stream_id", "events": [{"seq", "kind", ...}], "chain_head",
"reader_status", "reader_issues"}`` as ``streams_from_reports`` shapes the reports of
``teacher_rl.capture.sink.read_stream``. Every stream must carry its reader verdict: a stream the
reader blocked (or one with no verdict) blocks every artifact built over it, and the reader's
issues are folded in. Order is taken only from each stream's ``seq``; input list order never
matters. Empty input (no stream, no payload, no trainer record, no launch) is ``not_run``, never
``passed``.

Trainer events (F5; tickets E03, E04 inputs, E07, E20): ``build_trainer_step_records`` turns
``trainer_configured``, ``trainer_step_started``/``trainer_step_finished`` (or ``call_raised`` on
``step_training``), ``replay_pool_prepared``, ``minibatch_started`` (row indices, old log-probs,
padding, advantages), ``replay_forward``, ``optimizer_reduce``/``optimizer_reduce_raised``
(exact grad norm, ``optimizer.step`` taken/skipped, applied lr), ``scheduler_state``,
``parameter_snapshot`` (schema 2, ``teacher_rl.capture.entrypoint`` docstring) and the
checkpoint export events into one record per trainer step per stream, plus:

- ``e20_trainer_steps``: ``{stream_id: [record]}`` in exactly the
  ``teacher_rl.checks.steps.reconcile_optimizer_steps`` record shape (one list per stream:
  ranks are never merged, and that function rejects a repeated ``trainer_step``);
- ``e07_trainer_steps``: ``[{stream_id, trainer_step, hash_comparability, input}]`` where
  ``input`` is exactly ``teacher_rl.adapters.trainer_step_input`` output, i.e. the
  ``checks.steps.check_skip_semantics`` input, with snapshot digests in the
  ``adapters.skip_snapshot`` shape ``{parameters, optimizer_state, step_counters}``.

Artifact status: ``blocked`` on any blocking issue, ``not_run`` when no trainer step was
captured, ``not_estimable`` when steps exist but neither an E20 nor an E07 input could be
built, otherwise ``passed`` (record integrity and at least one assembled input; never a check
verdict).

Assembly rules (each step reports ``e20``/``e07`` ``assembly`` = ``passed`` (input built; not
a check result) | ``not_estimable`` | ``blocked`` with reason codes; nothing is filled):

- E20 needs a finished step, ``trainer_configured`` ``mini_batch`` and ``iterations``, the
  replay pool ``rows``, and for every minibatch a returned reduce whose
  ``optimizer_step_calls`` is 0 (skipped) or 1 (taken). ``reported_num_batches`` is
  ``metrics["train/num_batches"]`` when it is an int, else ``None``; a value that is present
  but not a count is the warning ``num_batches_unreadable`` (the E20 cross-check cannot run).
  A reduce ``lr`` that is neither ``null`` nor a list of finite numbers blocks
  (``applied_lr_invalid``) and is recorded as ``applied_lr: null``.
- E07 additionally needs exactly one minibatch in the step (pre/post step snapshots bracket
  only that one; per-minibatch snapshots are not captured, so more minibatches are
  ``per_minibatch_snapshots_unavailable``), a finite grad norm, both ``scheduler_state``
  records and both snapshots hashed (``metadata_only`` -> ``snapshot_unhashed``). A snapshot
  with any ``local_shard`` entry is comparable only shard-to-shard inside the same stream:
  ``hash_comparability`` says ``shard_to_shard_same_stream`` and any change of shard layout
  between the two snapshots blocks (``shard_layout_changed``).
- Order is checked, not assumed: the ``pre_step`` snapshot and scheduler record must precede
  the step's first update event (replay pool, minibatch, forward, reduce) and a scheduler
  record must carry the step's call id, else ``trainer_step_evidence_misordered`` blocks and
  the step's E07 is ``blocked``; ``post_step`` records attach only after the step ended
  (otherwise ``orphan_step_event``). A fresh call id (``_OPENING_KINDS``) repeated within a
  stream blocks (``call_reused``).
- Each step's ``row_checks`` is ``{status: run|not_run, reasons}``: ``not_run`` names the
  E04 row checks that could not execute (``row_indices_unavailable`` e.g. an unsupplied blob,
  ``rows_not_captured``, ``row_index_unresolved``, ``minibatch_rows_missing``,
  ``iteration_permutation_not_checked``), so a passed artifact is never read as covering them.
- ``param_snapshot_artifact`` builds an A10 ``s4_param_snapshot`` v1 (for ``params.diff`` and
  ``params.check_trainable_set``, E03) only when every tensor has ``hash_scope: full``; a
  local-shard v1 artifact would silently compare wrong against a full manifest. The capture
  kind ``param`` becomes A10's ``parameter`` here and nowhere else (``CAPTURE_TENSOR_KINDS``).
  An ``s4_optimizer_state_snapshot`` cannot be built from the event (param-group membership is
  not part of the v2 schema); E07 uses the digest above instead. Consumers (G3) read
  ``e07_trainer_steps`` / ``snapshot_digest``, never ``adapters.skip_snapshot`` or
  ``adapters.minibatch_record``, which need A10 artifacts the capture cannot supply.
- Every attempt is preserved: a raised or unterminated step stays in the record with its
  outcome; a ``capture_extract_error`` from a trainer target and an oversize trainer event are
  blocking; bounded lists stored as blobs are resolved only from the supplied ``blobs``
  (``{sha256: bytes}``, hash-verified), otherwise the value is ``None`` with a warning.

Verification ceiling:
- every function here: **L1** (pure, deterministic; literal known answers, permutation and
  negative controls in ``tests/test_s4_records.py``; E07/E20 shape compatibility is proved by
  running ``checks.steps``/``adapters`` on the output).
- That the capture wrappers emit these events with these semantics on a host is **L2/L3**
  (``teacher_rl/capture/entrypoint.py``).

Never: computes advantages (A9's ``signal.py`` owns the float32 oracle), chooses the strict
cohort's expected version, decides whether a dropped payload invalidates a run, treats a
missing record as a pass or a missing seed as zero, runs the E07/E20 checks itself, or
fabricates a per-minibatch snapshot, a grad norm, a step decision or a scheduler epoch.
"""

import hashlib
import math
import re
from pathlib import Path

from gate.s2_identity import canonical_json_bytes, identity_hash, read_json_bytes
from gate.tensor_manifest import DTYPE_ITEMSIZE, HASH_CONVENTION

SCHEMA_VERSION = 1
COHORTS = ("strict_parity", "async_disclosed")
SCOPES = ("synthetic", "public_source_fixture", "actual")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_COMPLETION = re.compile(r"(?P<scene>.+)_(?P<session>[0-9a-f]{32})_(?P<nonce>[0-9a-f]{32})\.json")
LAUNCH_HASH_FIELDS = ("effective_config_hash", "uv_lock_sha256", "base_weights_hash",
                      "capture_tree_hash")
# Blocking codes that mean "nothing to evaluate" rather than "evaluated and failed".
EMPTY_CODES = frozenset({"no_capture_streams", "no_payloads_captured", "no_trainer_records",
                         "no_launch_records", "no_randomness_records", "no_trainer_steps"})
RANDOMNESS_KEYS = {
    "cosmos_train_seed": "train.seed",
    "dataloader_shuffle_seed": "train.train_policy.dataloader_seed",
    "rollout_force_determinism": "policy.inference.sampling.force_determinism",
}
FAILURE_OUTCOMES = ("retry", "dropped", "ignored_after_drop")


def _issue(code, path, message, severity="blocking", refs=()):
    return {"code": code, "path": path, "severity": severity, "evidence_refs": list(refs),
            "message": message}


def _producer():
    return {"module": "teacher_rl.records",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _status(issues):
    blocking = {i["code"] for i in issues if i["severity"] == "blocking"}
    if not blocking:
        return "passed"
    return "not_run" if blocking <= EMPTY_CODES else "blocked"


def _check_scope(scope, issues):
    if scope not in SCOPES:
        issues.append(_issue("unsupported_scope", "$.scope", f"scope must be one of {SCOPES}"))


def streams_from_reports(reports):
    """Turn ``read_stream`` reports into ledger input carrying each reader verdict.

    Returns ``(streams, issues)``; ``issues`` repeats the reader issues for callers that report
    them directly. The ledger and trace-integrity checks fold the per-stream verdict in
    themselves (``_reader_verdicts``), so a blocked report blocks them.
    """
    streams, issues = [], []
    for report in reports:
        issues.extend(report.get("issues", []))
        streams.append({"stream_id": report.get("stream_label"), "events": report["events"],
                        "chain_head": report.get("chain_head"),
                        "reader_status": report.get("status"),
                        "reader_issues": list(report.get("issues", []))})
    return streams, issues


def _reader_verdicts(streams, issues):
    """Fold each stream's reader verdict into ``issues``; return the input records."""
    inputs = []
    for stream in sorted(streams, key=lambda item: str(item.get("stream_id"))):
        sid = stream.get("stream_id")
        status = stream.get("reader_status")
        reader_issues = stream.get("reader_issues")
        if status not in ("passed", "blocked") or type(reader_issues) is not list:
            issues.append(_issue("stream_unverified", f"{sid}",
                                 "stream carries no read_stream verdict (use "
                                 "streams_from_reports over read_stream reports)"))
        else:
            for item in reader_issues:
                code = item.get("code") if type(item) is dict else None
                severity = item.get("severity") if type(item) is dict else None
                issues.append(_issue(str(code), f"{sid}:{item.get('path') if code else ''}",
                                     str(item.get("message")) if code else "malformed issue",
                                     severity="warning" if severity == "warning" else "blocking",
                                     refs=item.get("evidence_refs") or () if code else ()))
            if status == "blocked" and not any(
                    type(i) is dict and i.get("severity") == "blocking" for i in reader_issues):
                issues.append(_issue("stream_reader_blocked", f"{sid}",
                                     "read_stream blocked this stream"))
        inputs.append({"artifact_kind": "s4_capture_stream", "stream_id": sid,
                       "chain_head": stream.get("chain_head"), "reader_status": status})
    return inputs


def _ordered(streams, issues):
    """Yield (stream_id, event) in per-stream seq order with streams sorted by id."""
    ids = [s.get("stream_id") for s in streams]
    for sid in ids:
        if type(sid) is not str or not sid:
            issues.append(_issue("malformed_stream", "$.stream_id", "stream_id must be text"))
    if len(set(ids)) != len(ids):
        issues.append(_issue("duplicate_stream", "$.stream_id", "stream IDs must be unique"))
    ordered = []
    for stream in sorted(streams, key=lambda s: str(s.get("stream_id"))):
        sid = stream.get("stream_id")
        events = stream.get("events")
        if type(events) is not list:
            issues.append(_issue("malformed_stream", f"{sid}", "events must be a list"))
            continue
        seqs = [e.get("seq") if type(e) is dict else None for e in events]
        if any(type(q) is not int for q in seqs):
            issues.append(_issue("malformed_event", f"{sid}", "every event needs an integer seq"))
            continue
        if len(set(seqs)) != len(seqs):
            issues.append(_issue("duplicate_seq", f"{sid}", "two events share a seq"))
            continue
        if sorted(seqs) != list(range(len(seqs))):
            issues.append(_issue("sequence_gap", f"{sid}",
                                 "event seq values are not contiguous from 0 (omitted records)"))
        for event in sorted(events, key=lambda e: e["seq"]):
            ordered.append((sid, event))
    return ordered


def build_attempt_ledger(streams, *, scope):
    """Reconstruct payloads, attempts, retries, drops and groups from capture streams.

    A payload instance is declared once by ``payload_bound`` (capture tags the upstream
    ``SharedPayloadState``; submissions and attempts reference that ``payload_instance``), so
    reconstruction never depends on thread interleaving. Group members are the successful
    attempts whose ``collected_index`` < ``n_target`` (upstream keeps the first ``n_target``
    collected episodes). Every failed attempt is kept with its cause.
    """
    issues = []
    _check_scope(scope, issues)
    if not streams:
        issues.append(_issue("no_capture_streams", "$.streams",
                             "no capture stream was supplied (capture may not have run)"))
    inputs = _reader_verdicts(streams, issues)
    payloads, attempts, sessions = {}, {}, {}
    for sid, event in _ordered(streams, issues):
        kind, where = event.get("kind"), f"{sid}#{event.get('seq')}"
        instance = event.get("payload_instance")
        pid = (sid, instance)
        if kind == "payload_bound":
            if type(instance) is not int or pid in payloads:
                issues.append(_issue("malformed_payload_instance", where,
                                     f"payload_instance {instance!r} missing or bound twice"))
                continue
            payloads[pid] = {"payload_id": f"{sid}:{event.get('prompt_idx')}:{instance}",
                             "stream_id": sid, "prompt_idx": event.get("prompt_idx"),
                             "instance": instance, "n_target": event.get("n_target"),
                             "retry_budget": event.get("retries_left"),
                             "submitted_seq": event["seq"], "submissions": 0, "attempts": []}
        elif kind == "payload_submitted":
            if pid not in payloads:
                issues.append(_issue("submission_without_payload", where,
                                     f"payload_instance {instance!r} was never bound"))
                continue
            payloads[pid]["submissions"] += 1
        elif kind == "attempt_started":
            session = event.get("session_uuid")
            if pid not in payloads or payloads[pid]["prompt_idx"] != event.get("prompt_idx"):
                issues.append(_issue("attempt_without_payload", where,
                                     f"attempt {session} has no bound payload instance"))
                continue
            if session in sessions:
                issues.append(_issue("session_reused", where,
                                     f"session {session} started more than once"))
                continue
            sessions[session] = pid
            attempts[session] = {"session_uuid": session, "payload_id": payloads[pid]["payload_id"],
                                 "prompt_idx": event.get("prompt_idx"),
                                 "scene_id": event.get("scene_id"), "attempt": event.get("attempt"),
                                 "started_seq": event["seq"], "outcome": None}
            payloads[pid]["attempts"].append(session)
        elif kind in ("attempt_succeeded", "attempt_failed"):
            session = event.get("session_uuid")
            record = attempts.get(session)
            if record is None or sessions.get(session) != pid:
                issues.append(_issue("terminal_without_start", where,
                                     f"{kind} for unknown session {session}"))
                continue
            if record["outcome"] is not None:
                issues.append(_issue("duplicate_terminal", where,
                                     f"session {session} has two terminal events"))
                continue
            record["terminal_seq"] = event["seq"]
            if kind == "attempt_succeeded":
                record.update(outcome="succeeded", collected_index=event.get("collected_index"),
                              reward_total=event.get("reward_total"),
                              reward_components=event.get("reward_components"))
            else:
                outcome = event.get("outcome")
                if outcome not in FAILURE_OUTCOMES:
                    issues.append(_issue("malformed_event", where, "unknown failure outcome"))
                record.update(outcome=f"failed_{outcome}", cause_type=event.get("cause_type"),
                              cause_message=event.get("cause_message"))
    groups, counts = [], {"payloads": len(payloads), "attempts": len(attempts),
                          "succeeded": 0, "silent_retries": 0, "dropped_payloads": 0,
                          "unterminated_attempts": 0, "never_started_siblings": 0}
    causes = {}
    for pid in sorted(payloads, key=lambda k: (k[0], k[1])):
        payload = payloads[pid]
        records = [attempts[s] for s in payload["attempts"]]
        n_target = payload["n_target"]
        if type(n_target) is not int or n_target < 1:
            issues.append(_issue("malformed_event", payload["payload_id"], "n_target missing"))
            continue
        retries = [r for r in records if r["outcome"] == "failed_retry"]
        dropped = any(r["outcome"] == "failed_dropped" for r in records)
        succeeded = [r for r in records if r["outcome"] == "succeeded"]
        for r in records:
            if r["outcome"] is None:
                counts["unterminated_attempts"] += 1
                issues.append(_issue("attempt_unterminated", payload["payload_id"],
                                     f"session {r['session_uuid']} has no terminal event"))
            if r["outcome"] and r["outcome"].startswith("failed"):
                name = str(r.get("cause_type"))
                causes[name] = causes.get(name, 0) + 1
        counts["succeeded"] += len(succeeded)
        counts["silent_retries"] += len(retries)
        members = sorted((r for r in succeeded if type(r.get("collected_index")) is int
                          and r["collected_index"] < n_target),
                         key=lambda r: r["collected_index"])
        indexes = [r["collected_index"] for r in members]
        if len(set(indexes)) != len(indexes):
            issues.append(_issue("duplicate_collected_index", payload["payload_id"],
                                 "two successes claim the same collected position"))
        enqueued = n_target + len(retries)
        never_started = max(enqueued - len(records), 0) if dropped else 0
        counts["never_started_siblings"] += never_started
        complete = len(members) == n_target and not dropped
        if dropped:
            counts["dropped_payloads"] += 1
            issues.append(_issue("payload_dropped", payload["payload_id"],
                                 f"prompt_idx {payload['prompt_idx']} exhausted its retries; "
                                 f"{len(members)}/{n_target} collected", severity="warning"))
        elif len(members) != n_target and not any(r["outcome"] is None for r in records):
            issues.append(_issue("group_incomplete", payload["payload_id"],
                                 f"{len(members)}/{n_target} members without a drop"))
        groups.append({
            "group_id": payload["payload_id"], "stream_id": payload["stream_id"],
            "prompt_idx": payload["prompt_idx"], "submitted_seq": payload["submitted_seq"],
            "submissions": payload["submissions"],
            "n_target": n_target, "complete": complete, "dropped": dropped,
            "members": [r["session_uuid"] for r in members],
            "member_rewards": [r.get("reward_total") for r in members],
            "member_is_retry": [isinstance(r.get("attempt"), int) and r["attempt"] > 0
                                for r in members],
            "failed_attempts": [{"session_uuid": r["session_uuid"], "attempt": r.get("attempt"),
                                 "cause_type": r.get("cause_type"), "outcome": r["outcome"]}
                                for r in records if r["outcome"]
                                and r["outcome"].startswith("failed")],
            "never_started_siblings": never_started,
        })
    if streams and not payloads:
        issues.append(_issue("no_payloads_captured", "$.streams",
                             "the streams hold no payload_bound record; nothing was attempted"))
    body = {"groups": groups, "attempts": [attempts[s] for s in sorted(attempts)],
            "counts": counts, "failure_causes": dict(sorted(causes.items()))}
    return {"artifact_kind": "s4_attempt_ledger", "schema_version": SCHEMA_VERSION,
            "scope": scope, "producer": _producer(), "inputs": inputs,
            "units": {"seq": "record index within one capture stream",
                      "collected_index": "position in the upstream collected list"},
            "status": _status(issues), "issues": issues, **body,
            "ledger_hash": identity_hash(body)}


def trace_versions(ledger, streams, issues=None):
    """Behaviour version per group from ``generation_entered``/``generation_exited`` events.

    The captured binding sees only the version passed at call entry. Exits are matched to their
    entry by the event's ``call`` id; a ``call_raised`` record for ``rollout_generation`` closes
    its call at that record (upstream raises on an exhausted payload). A group's version is
    ``entry_version`` when its payload was submitted inside one generation call (between its
    entry and exit records); calls with different versions overlapping the payload's lifetime
    give ``mixed_versions``; a payload submitted outside any call (prefetch) or without an exit
    record is ``undetermined``. Undetermined is never treated as a match. Entries without an
    integer call id, reused call ids and exits that match no open call are appended to
    ``issues`` (blocking) when a list is given.
    """
    issues = [] if issues is None else issues
    calls, open_calls = {}, {}
    for sid, event in _ordered(streams, issues):
        kind, call_id, where = event.get("kind"), event.get("call"), f"{sid}#{event.get('seq')}"
        if kind == "generation_entered":
            if type(call_id) is not int or (sid, call_id) in open_calls:
                issues.append(_issue("malformed_generation_call", where,
                                     f"generation_entered needs a fresh integer call id, got "
                                     f"{call_id!r}"))
                continue
            call = {"enter": event["seq"], "exit": None, "raised": False,
                    "version": event.get("current_weight_version"),
                    "prompts": list(event.get("prompt_idxs") or [])}
            calls.setdefault(sid, []).append(call)
            open_calls[(sid, call_id)] = call
        elif kind == "generation_exited" or (
                kind == "call_raised" and event.get("target") == "rollout_generation"):
            call = open_calls.pop((sid, call_id), None)
            if call is None:
                issues.append(_issue("unmatched_generation_exit", where,
                                     f"{kind} for call {call_id!r} matches no open "
                                     "generation call"))
                continue
            call["exit"] = event["seq"]
            call["raised"] = kind == "call_raised"
    by_session = {a["session_uuid"]: a for a in ledger["attempts"]}
    result = {}
    for group in ledger["groups"]:
        sessions = group["members"] + [f["session_uuid"] for f in group["failed_attempts"]]
        last = max((by_session[s].get("terminal_seq") or by_session[s]["started_seq"]
                    for s in sessions), default=None)
        sub, prompt = group["submitted_seq"], group["prompt_idx"]
        covering = [c for c in calls.get(group["stream_id"], []) if prompt in c["prompts"]]
        if last is None:
            result[group["group_id"]] = {"version": None, "reason": "undetermined"}
            continue
        overlapping = [c for c in covering
                       if c["enter"] <= last and (c["exit"] is None or c["exit"] >= sub)]
        versions = {c["version"] for c in overlapping}
        inside = [c for c in overlapping if c["enter"] < sub and c["exit"] is not None
                  and last < c["exit"]]
        if len(versions) > 1:
            result[group["group_id"]] = {"version": None, "reason": "mixed_versions",
                                         "versions": sorted(versions, key=str)}
        elif inside and type(inside[0]["version"]) is int:
            result[group["group_id"]] = {"version": inside[0]["version"],
                                         "reason": "entry_version"}
        else:
            result[group["group_id"]] = {"version": None, "reason": "undetermined"}
    return result


def check_trace_integrity(ledger, streams, *, cohort, expected_version=None):
    """E09: trainer-received rollouts against the ledger and behaviour versions.

    Rejects duplicate completions or sessions, omitted group members, records that are not
    group members (silent best-of-retry), prompt mismatches, misreported versions and mixed
    versions. ``strict_parity`` additionally requires every version to equal
    ``expected_version`` (a required input for that cohort; no default).
    """
    issues = []
    if cohort not in COHORTS:
        issues.append(_issue("unsupported_cohort", "$.cohort", f"must be one of {COHORTS}"))
    if cohort == "strict_parity" and type(expected_version) is not int:
        issues.append(_issue("expected_version_unregistered", "$.expected_version",
                             "strict parity needs the registered expected version"))
    if ledger.get("status") != "passed":
        issues.append(_issue("ledger_blocked", "$.ledger",
                             f"the attempt ledger is {ledger.get('status')!r}, not passed"))
    if not streams:
        issues.append(_issue("no_capture_streams", "$.streams", "no capture stream supplied"))
    stream_inputs = _reader_verdicts(streams, issues)
    versions = trace_versions(ledger, streams, issues)
    member_of = {}
    for group in ledger["groups"]:
        for session in group["members"]:
            member_of[session] = group
    seen_paths, seen_sessions, received = set(), set(), []
    for sid, event in _ordered(streams, []):  # ordering issues were recorded above
        if event.get("kind") != "trainer_rollouts_received":
            continue
        for index, rollout in enumerate(event.get("rollouts") or []):
            where = f"{sid}#{event['seq']}.rollouts[{index}]"
            path = rollout.get("completion")
            match = _COMPLETION.fullmatch(Path(str(path)).name) if path else None
            if match is None:
                issues.append(_issue("unparseable_completion", where,
                                     "completion is not <scene>_<session>_<nonce>.json"))
                continue
            session = match["session"]
            if path in seen_paths or session in seen_sessions:
                issues.append(_issue("duplicate_record", where,
                                     f"session {session} received more than once"))
                continue
            seen_paths.add(path)
            seen_sessions.add(session)
            received.append((where, session, rollout, event))
    groups_seen = set()
    for where, session, rollout, _event in received:
        group = member_of.get(session)
        if group is None:
            issues.append(_issue("not_a_group_member", where,
                                 f"session {session} is not a collected group member"))
            continue
        groups_seen.add(group["group_id"])
        if rollout.get("prompt_idx") != group["prompt_idx"]:
            issues.append(_issue("prompt_idx_mismatch", where,
                                 f"{rollout.get('prompt_idx')} != {group['prompt_idx']}"))
        info = versions.get(group["group_id"], {"version": None, "reason": "undetermined"})
        reported = rollout.get("weight_version")
        if info["reason"] == "mixed_versions":
            issues.append(_issue("mixed_versions_in_trace", where,
                                 f"group spans versions {info['versions']}"))
        elif info["version"] is None:
            issues.append(_issue("version_undetermined", where,
                                 "behaviour version cannot be established from capture",
                                 severity="blocking" if cohort == "strict_parity" else "warning"))
        elif reported != info["version"]:
            issues.append(_issue("misreported_version", where,
                                 f"stamped {reported} != generated at {info['version']}"))
        if cohort == "strict_parity" and type(expected_version) is int and (
                reported != expected_version or info["version"] != expected_version):
            issues.append(_issue("strict_cohort_version_mismatch", where,
                                 f"version must equal the registered {expected_version}"))
    for group in ledger["groups"]:
        if not group["complete"]:
            continue
        for session in group["members"]:
            if session not in seen_sessions:
                issues.append(_issue("omitted_record", group["group_id"],
                                     f"member {session} never reached the trainer"))
    if not received:
        issues.append(_issue("no_trainer_records", "$.streams",
                             "no trainer_rollouts_received record; nothing reached the trainer"))
    body = {"cohort": cohort, "expected_version": expected_version,
            "received": len(received), "groups_seen": sorted(groups_seen),
            "versions": {k: versions[k] for k in sorted(versions)}}
    return {"artifact_kind": "s4_trace_integrity", "schema_version": SCHEMA_VERSION,
            "scope": ledger.get("scope"), "producer": _producer(),
            "units": {"weight_version": "Cosmos policy step whose weights generated the trace"},
            "status": _status(issues), "issues": issues, **body,
            "inputs": [{"artifact_kind": "s4_attempt_ledger",
                        "identity_hash": ledger.get("ledger_hash")}, *stream_inputs]}


def check_run_identity(launches, checkpoints, *, scope):
    """E15: every launch of a run binds the same configuration, lock, base weights and capture.

    ``launches``: ``{run_id, launch_id, ordinal, phase ("start"|"resume"), <hash fields>,
    resume_from_step?, resume_weights_hash?}``. ``checkpoints``: ``{run_id, step,
    weights_hash}`` recorded at save time. A second start, a reused launch ID or ordinal, a
    resume before its start, a changed hash or a resume whose weights differ from the recorded
    checkpoint is blocking.
    """
    issues = []
    _check_scope(scope, issues)
    if not launches:
        issues.append(_issue("no_launch_records", "$.launches",
                             "no launch record supplied; run identity cannot be established"))
    saved = {}
    for index, ckpt in enumerate(checkpoints):
        key = (ckpt.get("run_id"), ckpt.get("step"))
        if type(key[1]) is not int or not _HEX64.fullmatch(str(ckpt.get("weights_hash"))):
            issues.append(_issue("malformed_checkpoint", f"$.checkpoints[{index}]",
                                 "needs an integer step and a sha256 weights_hash"))
        elif key in saved and saved[key] != ckpt["weights_hash"]:
            issues.append(_issue("checkpoint_conflict", f"$.checkpoints[{index}]",
                                 f"step {key[1]} recorded with two different hashes"))
        else:
            saved[key] = ckpt["weights_hash"]
    starts, launch_ids, ordinals = {}, set(), set()
    for launch in sorted(launches, key=lambda item: (str(item.get("run_id")),
                                                     item.get("ordinal") if type(
                                                         item.get("ordinal")) is int else -1)):
        where = f"{launch.get('run_id')}/{launch.get('launch_id')}"
        if type(launch.get("ordinal")) is not int:
            issues.append(_issue("malformed_launch", where, "ordinal must be an integer"))
            continue
        bad = [f for f in LAUNCH_HASH_FIELDS if not _HEX64.fullmatch(str(launch.get(f)))]
        if bad:
            issues.append(_issue("missing_identity_field", where, f"missing/malformed {bad}"))
            continue
        if launch.get("launch_id") in launch_ids:
            issues.append(_issue("duplicate_launch", where, "launch_id seen before"))
            continue
        launch_ids.add(launch.get("launch_id"))
        if (launch.get("run_id"), launch["ordinal"]) in ordinals:
            issues.append(_issue("duplicate_launch", where, "two launches share an ordinal"))
            continue
        ordinals.add((launch.get("run_id"), launch["ordinal"]))
        run = launch.get("run_id")
        if launch.get("phase") == "start":
            if run in starts:
                issues.append(_issue("duplicate_launch", where, f"run {run} started twice"))
                continue
            starts[run] = launch
        elif launch.get("phase") == "resume":
            start = starts.get(run)
            if start is None:
                issues.append(_issue("resume_without_start", where, "no prior start"))
                continue
            for field in LAUNCH_HASH_FIELDS:
                if launch[field] != start[field]:
                    issues.append(_issue("resume_identity_mismatch", where,
                                         f"{field} differs from the start"))
            step = launch.get("resume_from_step")
            if saved.get((run, step)) is None:
                issues.append(_issue("resume_checkpoint_unrecorded", where,
                                     f"no recorded checkpoint at step {step}"))
            elif launch.get("resume_weights_hash") != saved[(run, step)]:
                issues.append(_issue("resume_weight_mismatch", where,
                                     f"weights differ from the checkpoint saved at step {step}"))
        else:
            issues.append(_issue("malformed_launch", where, "phase must be start or resume"))
    try:  # canonical order: the same records in any input order give the same hash
        inputs = [{"artifact_kind": "s4_launch_records", "identity_hash": identity_hash(
            {"launches": sorted(launches, key=canonical_json_bytes),
             "checkpoints": sorted(checkpoints, key=canonical_json_bytes)})}]
    except (TypeError, ValueError):
        inputs = []
        issues.append(_issue("not_json_native", "$", "launch records are not strict JSON"))
    return {"artifact_kind": "s4_run_identity", "schema_version": SCHEMA_VERSION,
            "scope": scope, "producer": _producer(), "inputs": inputs,
            "units": {"step": "trainer step", "ordinal": "launch order within a run"},
            "status": _status(issues), "issues": issues,
            "runs": sorted(str(r) for r in starts)}


def _effective_config_issues(effective):
    """Validate the compiled effective configuration; return (issues, {key: value})."""
    issues, values = [], {}
    if type(effective) is not dict:
        return [_issue("effective_config_invalid", "$.effective_config", "not an object")], values
    if effective.get("artifact_kind") != "s4_effective_config" or type(
            effective.get("schema_version")) is not int or effective["schema_version"] != 1:
        issues.append(_issue("effective_config_invalid", "$.effective_config",
                             "expected artifact_kind 's4_effective_config' schema_version 1"))
    if effective.get("status") != "passed":
        issues.append(_issue("effective_config_blocked", "$.effective_config.status",
                             "randomness needs a compiled (passed) effective configuration"))
    claimed = effective.get("effective_config_hash")
    if not _HEX64.fullmatch(str(claimed)):
        issues.append(_issue("effective_config_invalid", "$.effective_config_hash",
                             "missing or malformed effective_config_hash"))
    else:
        try:
            recomputed = identity_hash(effective.get("effective_config"))
        except (TypeError, ValueError):
            recomputed = None
        if recomputed != claimed:
            issues.append(_issue("effective_config_hash_mismatch", "$.effective_config_hash",
                                 "effective_config_hash does not bind effective_config"))
    mapping = effective.get("mapping")
    entries = [e for e in mapping if type(e) is dict] if type(mapping) is list else []
    for name, key in RANDOMNESS_KEYS.items():
        found = [e for e in entries if e.get("key") == key]
        if len(found) != 1 or "value" not in found[0]:
            issues.append(_issue("randomness_source_unmapped", f"$.mapping[{key}]",
                                 f"{key} must be mapped exactly once (no default is assumed)"))
            continue
        value = found[0]["value"]
        values[name] = "unseeded" if key == "train.seed" and value is None else value
    return issues, values


def randomness_record(run_id, effective_config, observed):
    """E17: the actual randomness sources of one run, configured versus observed.

    ``effective_config`` must be a passed ``s4_effective_config`` artifact whose mapping holds
    each of ``RANDOMNESS_KEYS`` exactly once (nothing is defaulted). ``observed``:
    ``{"train_seed": int | "unseeded" | None, "dataloader_seed": int | None,
    "force_determinism": bool | None, "session_seeds": [{"session_uuid", "seed"}] | None}``
    (``None`` = not observed). ``train.seed = None`` is recorded as ``"unseeded"``, never 0.
    Status: ``blocked`` on an invalid configuration or a mismatch; ``unverified`` while any
    source is unobserved (absence is never a pass); ``passed`` only when every configured
    source and the session seeds were observed and consistent.
    """
    issues, configured = _effective_config_issues(effective_config)
    if type(observed) is not dict:
        issues.append(_issue("observed_invalid", "$.observed", "observed must be an object"))
    if any(i["severity"] == "blocking" for i in issues):
        return {**_randomness_header(run_id, effective_config), "status": "blocked",
                "issues": issues, "sources": []}
    obs_map = {"cosmos_train_seed": "train_seed", "dataloader_shuffle_seed": "dataloader_seed",
               "rollout_force_determinism": "force_determinism"}
    sources, unverified = [], False
    for name in RANDOMNESS_KEYS:
        value = configured[name]
        seen = observed.get(obs_map[name])
        if seen is None:
            status, unverified = "unverified", True
        elif seen == value and type(seen) is type(value):
            status = "consistent"
        else:
            status = "mismatch"
            issues.append(_issue("randomness_mismatch", f"$.{name}",
                                 f"configured {value!r} but observed {seen!r}"))
        sources.append({"source": name, "configured": value, "observed": seen, "status": status})
    sources.append({"source": "rollout_global_rng", "configured": (
        "seeded_42" if configured["rollout_force_determinism"] is True else "unseeded"),
        "observed": None, "status": "derived",
        "basis": "alpagym policies/determinism.py:22-35 (force_determinism seeds 42)"})
    sources.append({"source": "minibatch_order", "configured": "unseeded_randperm",
                    "observed": None, "status": "fixed_by_upstream",
                    "basis": "alpagym cosmos/trainer.py:333-334 torch.randperm without generator"})
    seeds = observed.get("session_seeds")
    if seeds is None:
        unverified = True
        sources.append({"source": "simulator_session_seeds",
                        "configured": "upstream_random_unsettable", "observed": None,
                        "status": "unverified"})
    else:
        values = [s.get("seed") if type(s) is dict else None for s in seeds] if type(
            seeds) is list else [None]
        if any(type(v) is not int for v in values):
            issues.append(_issue("malformed_session_seed", "$.session_seeds",
                                 "every observed session seed must be an integer"))
        sources.append({"source": "simulator_session_seeds",
                        "configured": "upstream_random_unsettable",
                        "observed": {"sessions": len(values),
                                     "distinct_seeds": len({v for v in values if type(v) is int})},
                        "status": "observed"})
    status = _status(issues)
    if status == "passed" and unverified:
        status = "unverified"
    return {**_randomness_header(run_id, effective_config), "status": status,
            "issues": issues, "sources": sources}


def _randomness_header(run_id, effective_config):
    effective = effective_config if type(effective_config) is dict else {}
    return {"artifact_kind": "s4_randomness_record", "schema_version": SCHEMA_VERSION,
            "scope": effective.get("scope"), "producer": _producer(),
            "inputs": [{"artifact_kind": "s4_effective_config",
                        "identity_hash": effective.get("effective_config_hash")}],
            "units": {"seed": "integer seed or the string 'unseeded'"}, "run_id": run_id}


def describe_run_randomness(records):
    """Across runs: which configured sources are identical (shared, not independent).

    A blocked or malformed record blocks the summary; an unverified record makes it
    ``unverified``; no record is ``not_run``.
    """
    issues, unverified = [], False
    if not records:
        issues.append(_issue("no_randomness_records", "$.records", "no randomness record"))
    run_ids = [r.get("run_id") if type(r) is dict else None for r in records]
    if len(set(map(str, run_ids))) != len(run_ids):
        issues.append(_issue("duplicate_run", "$.records", "run IDs must be unique"))
    usable = []
    for index, record in enumerate(records):
        if type(record) is not dict or record.get("artifact_kind") != "s4_randomness_record" or (
                record.get("schema_version") != SCHEMA_VERSION
                or type(record.get("sources")) is not list):
            issues.append(_issue("malformed_record", f"$.records[{index}]",
                                 "expected an s4_randomness_record schema_version 1"))
            continue
        if record.get("status") == "unverified":
            unverified = True
            issues.append(_issue("randomness_record_unverified", f"$.records[{index}]",
                                 f"run {record.get('run_id')} has unobserved sources",
                                 severity="warning"))
        elif record.get("status") != "passed":
            issues.append(_issue("randomness_record_blocked", f"$.records[{index}]",
                                 f"run {record.get('run_id')} record is "
                                 f"{record.get('status')!r}"))
        usable.append(record)
    shared = {}
    for name in RANDOMNESS_KEYS:
        values = {}
        for record in usable:
            for source in record["sources"]:
                if type(source) is dict and source.get("source") == name:
                    values[str(record.get("run_id"))] = source.get("configured")
        distinct = {repr(v) for v in values.values()}
        shared[name] = {"runs": len(values), "identical_across_runs": len(distinct) == 1
                        and len(values) > 1, "values": {k: values[k] for k in sorted(values)}}
    scopes = sorted({str(r.get("scope")) for r in records if type(r) is dict})
    if len(scopes) > 1:
        issues.append(_issue("mixed_scope", "$.records", f"records mix scopes {scopes}"))
    status = _status(issues)
    if status == "passed" and unverified:
        status = "unverified"
    return {"artifact_kind": "s4_run_randomness", "schema_version": SCHEMA_VERSION,
            "scope": scopes[0] if len(scopes) == 1 else None, "producer": _producer(),
            "inputs": [{"artifact_kind": "s4_randomness_record", "run_id": r.get("run_id")}
                       for r in sorted(usable, key=lambda r: str(r.get("run_id")))],
            "units": {}, "status": status, "issues": issues,
            "runs": sorted(str(r.get("run_id")) for r in usable), "shared_sources": shared,
            "note": "identical configured sources do not vary across runs; independence comes "
                    "only from the remaining unseeded sources"}


# ---------------------------------------------------------------- trainer steps (F5)

TRAINER_RECORDS_KIND = "s4_trainer_step_records"
PARAM_SNAPSHOT_KIND = "s4_param_snapshot"
SNAPSHOT_EVENT_SCHEMA = 2
SNAPSHOT_PHASES = ("init", "pre_step", "post_step")
STEP_PHASES = ("pre_step", "post_step")
# Capture tensor kind -> A10 ``s4_param_snapshot`` kind: the single translation point.
CAPTURE_TENSOR_KINDS = {"param": "parameter", "buffer": "buffer"}
HASH_SCOPES = ("full", "local_shard")
NONFINITE_NORMS = ("nan", "inf", "-inf")
TRAINER_TARGETS = frozenset({"trainer_init", "step_training", "prepare_training_data",
                             "forward_with_reference", "train_minibatch", "all_reduce_states",
                             "save_checkpoint"})
TRAINER_EVENT_KINDS = frozenset({
    "trainer_configured", "parameter_snapshot", "scheduler_state", "trainer_step_started",
    "trainer_step_finished", "replay_pool_prepared", "minibatch_started", "replay_forward",
    "minibatch_finished", "optimizer_reduce", "optimizer_reduce_raised", "checkpoint_save",
    "checkpoint_export_settled"})
_SNAPSHOT_EVENT_KEYS = {"kind", "seq", "schema_version", "phase", "trainer_step", "tensors",
                        "training_flags", "optimizer_state", "step_counters", "applied_lr"}
_TENSOR_KEYS = {"kind", "dtype", "shape", "numel", "requires_grad", "sha256", "hash_scope",
                "shard"}
_SHARD_KEYS = {"global_shape", "offsets", "placement"}
_SCHEDULER_KEYS = {"kind", "seq", "call", "phase", "trainer_step", "last_epoch", "lrs"}
_CONFIGURED = ("mini_batch", "iterations", "group_size", "allowed_outdated_steps")
_MINIBATCH_LISTS = ("row_indices", "advantages", "old_logprobs", "is_padding", "new_log_probs")
# Kinds whose ``call`` is a fresh ``next_call()`` id; the capture never reuses one in a stream.
# (``trainer_step_finished``, ``scheduler_state``, ``minibatch_finished``, ``call_raised`` and
# ``checkpoint_save`` repeat the id of the call they belong to and are not listed.)
_OPENING_KINDS = frozenset({"trainer_configured", "trainer_step_started", "replay_pool_prepared",
                            "minibatch_started", "replay_forward", "optimizer_reduce",
                            "optimizer_reduce_raised"})
# Kinds that begin the update inside a step: pre_step evidence must precede the first of them.
_UPDATE_KINDS = frozenset({"replay_pool_prepared", "minibatch_started", "replay_forward",
                           "optimizer_reduce", "optimizer_reduce_raised"})


class RecordsError(ValueError):
    """A capture event cannot be turned into a consumer input; ``code`` is snake_case."""

    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code = code


def _count(value):
    return type(value) is int and value >= 0


def _real(value):
    return type(value) in (int, float) and math.isfinite(value)


def _tensor_problem(name, entry):
    if type(name) is not str or not name:
        return "tensor names must be nonempty text"
    if type(entry) is not dict or set(entry) != _TENSOR_KEYS:
        return f"tensor {name!r} needs exactly {sorted(_TENSOR_KEYS)}"
    if entry["kind"] not in CAPTURE_TENSOR_KINDS:
        return f"tensor {name!r} kind must be one of {sorted(CAPTURE_TENSOR_KINDS)}"
    if entry["dtype"] not in DTYPE_ITEMSIZE:
        return f"tensor {name!r} dtype {entry['dtype']!r} is not a rule-11 name"
    shape = entry["shape"]
    if type(shape) is not list or not all(_count(d) for d in shape):
        return f"tensor {name!r} shape must be a list of ints >= 0"
    if type(entry["numel"]) is not int or entry["numel"] != math.prod(shape):
        return f"tensor {name!r} numel must equal prod(shape)"
    if type(entry["requires_grad"]) is not bool:
        return f"tensor {name!r} requires_grad must be a bool"
    sha, scope, shard = entry["sha256"], entry["hash_scope"], entry["shard"]
    if sha is None and scope is not None:
        return f"tensor {name!r} has a hash_scope but no sha256"
    if sha is not None and (type(sha) is not str or _HEX64.fullmatch(sha) is None):
        return f"tensor {name!r} sha256 must be 64 lowercase hex"
    if sha is not None and scope not in HASH_SCOPES:
        return f"tensor {name!r} hash_scope must be one of {HASH_SCOPES}"
    if shard is not None:
        if type(shard) is not dict or set(shard) != _SHARD_KEYS:
            return f"tensor {name!r} shard needs exactly {sorted(_SHARD_KEYS)}"
        total, offsets = shard["global_shape"], shard["offsets"]
        if (type(total) is not list or type(offsets) is not list or len(total) != len(shape)
                or len(offsets) != len(shape) or not all(_count(v) for v in total + offsets)):
            return f"tensor {name!r} shard global_shape/offsets must match the local rank"
        if any(o + s > g for o, s, g in zip(offsets, shape, total, strict=True)):
            return f"tensor {name!r} local shard exceeds its global shape"
        if type(shard["placement"]) is not str or not shard["placement"]:
            return f"tensor {name!r} shard placement must be nonempty text"
    if sha is not None and (scope == "local_shard") != (shard is not None):
        return f"tensor {name!r}: hash_scope local_shard exactly when a shard is recorded"
    return None


def _snapshot_problem(event):
    """``None`` or the first defect of a v2 ``parameter_snapshot`` event (fail closed)."""
    if type(event) is not dict or set(event) != _SNAPSHOT_EVENT_KEYS:
        return f"parameter_snapshot needs exactly {sorted(_SNAPSHOT_EVENT_KEYS)}"
    if type(event["schema_version"]) is not int or event["schema_version"] != (
            SNAPSHOT_EVENT_SCHEMA):
        return f"schema_version must be {SNAPSHOT_EVENT_SCHEMA}"
    phase, step = event["phase"], event["trainer_step"]
    if phase not in SNAPSHOT_PHASES:
        return f"phase must be one of {SNAPSHOT_PHASES}"
    if (phase == "init" and step is not None) or (phase != "init" and not _count(step)):
        return "trainer_step is null exactly at init and otherwise an int >= 0"
    tensors = event["tensors"]
    if type(tensors) is not dict or not tensors:
        return "tensors must be a nonempty object"
    for name, entry in tensors.items():
        problem = _tensor_problem(name, entry)
        if problem:
            return problem
    hashed = {entry["sha256"] is not None for entry in tensors.values()}
    if len(hashed) != 1:
        return "every tensor is hashed or none is"
    hashed = hashed.pop()
    flags = event["training_flags"]
    if type(flags) is not dict or not all(type(v) is bool for v in flags.values()):
        return "training_flags must map module names to bools"
    params = {name for name, entry in tensors.items() if entry["kind"] == "param"}
    state, counters = event["optimizer_state"], event["step_counters"]
    if type(state) is not dict or type(counters) is not dict:
        return "optimizer_state and step_counters must be objects"
    if set(state) != set(counters) or not set(state) <= params:
        return "optimizer_state and step_counters must name the same params, all in tensors"
    for name, entries in state.items():
        if type(entries) is not dict or not all(type(k) is str and k for k in entries):
            return f"optimizer_state[{name!r}] must map state keys to digests"
        for digest in entries.values():
            if hashed and (type(digest) is not str or _HEX64.fullmatch(digest) is None):
                return f"optimizer_state[{name!r}] digests must be sha256 hex"
            if not hashed and digest is not None:
                return f"optimizer_state[{name!r}] of an unhashed snapshot must be null"
    if not all(_count(v) for v in counters.values()):
        return "step_counters must be ints >= 0"
    if type(event["applied_lr"]) is not list or not all(_real(v) for v in event["applied_lr"]):
        return "applied_lr must be a list of finite numbers"
    return None


def _hashed(event):
    return next(iter(event["tensors"].values()))["sha256"] is not None


def _checked_snapshot(event):
    problem = _snapshot_problem(event)
    if problem:
        raise RecordsError("snapshot_invalid", problem)
    return event


def hash_comparability(event):
    """``full`` when every tensor was hashed whole, else ``shard_to_shard_same_stream`` (L1)."""
    _checked_snapshot(event)
    if not _hashed(event):
        raise RecordsError("snapshot_unhashed", "a metadata_only snapshot has no hashes")
    scopes = {entry["hash_scope"] for entry in event["tensors"].values()}
    return "full" if scopes == {"full"} else "shard_to_shard_same_stream"


def snapshot_digest(event):
    """E07 digest ``{parameters, optimizer_state, step_counters}`` of a hashed v2 snapshot (L1).

    The exact ``adapters.skip_snapshot`` output shape: parameters only (buffers excluded),
    every optimizer member with ``{}``/``0`` when it has no state yet.
    """
    _checked_snapshot(event)
    if not _hashed(event):
        raise RecordsError("snapshot_unhashed", "a metadata_only snapshot has no hashes")
    tensors = event["tensors"]
    return {"parameters": {name: tensors[name]["sha256"] for name in sorted(tensors)
                           if tensors[name]["kind"] == "param"},
            "optimizer_state": {name: dict(sorted(event["optimizer_state"][name].items()))
                                for name in sorted(event["optimizer_state"])},
            "step_counters": {name: event["step_counters"][name]
                              for name in sorted(event["step_counters"])}}


def param_snapshot_artifact(event, *, scope):
    """An A10 ``s4_param_snapshot`` v1 from a full-identity v2 snapshot event (E03; L1).

    Refuses (``RecordsError``) an invalid or unhashed event and any ``local_shard`` entry.
    ``byte_length`` is ``numel * DTYPE_ITEMSIZE[dtype]``; ``persistent`` is ``None`` (the
    capture does not record buffer persistence; unknown is never ``False``).
    """
    if scope not in SCOPES:
        raise RecordsError("unsupported_scope", f"scope must be one of {SCOPES}")
    if hash_comparability(event) != "full":
        local = sorted(n for n, e in event["tensors"].items() if e["hash_scope"] != "full")
        raise RecordsError("local_shard_not_full_identity",
                           f"{len(local)} tensor(s) hashed over local shards (first {local[0]!r}); "
                           "they compare only shard-to-shard, never as full identities")
    tensors = {}
    for name in sorted(event["tensors"]):
        entry = event["tensors"][name]
        tensors[name] = {"kind": CAPTURE_TENSOR_KINDS[entry["kind"]], "dtype": entry["dtype"],
                         "shape": list(entry["shape"]), "numel": entry["numel"],
                         "byte_length": entry["numel"] * DTYPE_ITEMSIZE[entry["dtype"]],
                         "sha256": entry["sha256"], "requires_grad": entry["requires_grad"],
                         "persistent": None}
    kinds = [entry["kind"] for entry in tensors.values()]
    return {
        "artifact_kind": PARAM_SNAPSHOT_KIND, "schema_version": 1, "scope": scope,
        "producer": _producer(),
        "input_hashes": {"parameter_snapshot_event": identity_hash(event)},
        "units": {"numel": "elements", "byte_length": "bytes",
                  "shape": "elements per dimension, row-major"},
        "hash_convention": HASH_CONVENTION, "status": "passed", "source": "module",
        "capture": {"phase": event["phase"], "trainer_step": event["trainer_step"],
                    "seq": event["seq"]},
        "tensors": tensors, "modules": dict(sorted(event["training_flags"].items())),
        "tensor_set_sha256": identity_hash(
            {n: {k: e[k] for k in ("dtype", "shape", "byte_length", "sha256")}
             for n, e in tensors.items()}),
        "totals": {"tensor_count": len(tensors), "parameter_count": kinds.count("parameter"),
                   "buffer_count": kinds.count("buffer"),
                   "numel": sum(e["numel"] for e in tensors.values()),
                   "byte_length": sum(e["byte_length"] for e in tensors.values())},
        "issues": []}


def _blob_map(blobs, issues):
    if blobs is None:
        return {}
    if type(blobs) is not dict or not all(
            type(k) is str and _HEX64.fullmatch(k) and type(v) is bytes for k, v in blobs.items()):
        issues.append(_issue("blobs_invalid", "$.blobs", "blobs must map sha256 hex to bytes"))
        return {}
    return blobs


def _resolve(value, blobs, where, issues):
    """A captured bounded list (inline values or a content-addressed blob) or ``None``."""
    if value is None:
        return None
    if type(value) is dict and set(value) == {"count", "values"} and _count(value["count"]) and (
            type(value["values"]) is list):
        if len(value["values"]) != value["count"]:
            issues.append(_issue("bounded_count_mismatch", where, "count != len(values)"))
            return None
        return value["values"]
    if type(value) is dict and set(value) == {"count", "blob"} and _count(value["count"]) and (
            type(value["blob"]) is dict):
        blob = value["blob"]
        sha = blob.get("sha256")
        if blob.get("stored") is not True:
            issues.append(_issue("blob_not_stored", where, "the sink did not store this list "
                                 "(over budget); its values are unavailable", "warning"))
            return None
        if type(sha) is not str or sha not in blobs:
            issues.append(_issue("blob_not_supplied", where, f"blob {sha} was not supplied",
                                 "warning"))
            return None
        data = blobs[sha]
        if hashlib.sha256(data).hexdigest() != sha:
            issues.append(_issue("blob_hash_mismatch", where, f"blob bytes do not hash to {sha}"))
            return None
        try:
            values = read_json_bytes(data)
        except ValueError as exc:
            issues.append(_issue("bounded_value_invalid", where, f"blob is not strict JSON: {exc}"))
            return None
        if type(values) is not list or len(values) != value["count"]:
            issues.append(_issue("bounded_count_mismatch", where, "blob list length != count"))
            return None
        return values
    issues.append(_issue("bounded_value_invalid", where, "not a captured bounded list"))
    return None


def _new_step(trainer_step, call, seq):
    return {"trainer_step": trainer_step, "call": call, "outcome": None, "error_type": None,
            "started_seq": seq, "ended_seq": None, "rows": None, "pool_call": None,
            "reported_num_batches": None, "minibatches": [], "orphan_reduces": 0,
            "scheduler": {"pre_step": None, "post_step": None},
            "snapshots": {"pre_step": None, "post_step": None}, "oversize": [],
            "extract_errors": [], "raised_calls": [], "export_calls": [],
            "first_update_seq": None, "misordered": [], "row_checks": None}


def _minibatch(step, call):
    if step is None or type(call) is not int:
        return None
    return next((mb for mb in step["minibatches"] if mb["call"] == call), None)


def _attach(state, slot, phase, trainer_step, value, where, issues, call=None):
    """Attach a pre_step record to the open step, a post_step record to the last closed one.

    Order is checked, never assumed from the producer: a pre_step record arriving after the
    step's first update event (``_UPDATE_KINDS``), or a scheduler record naming another call,
    is ``trainer_step_evidence_misordered`` and is not attached. A post_step record can only
    attach after the step's end (inside an open step it is an ``orphan_step_event``).
    """
    if phase == "pre_step":
        target = state["open"]
    else:
        target = state["steps"][-1] if state["open"] is None and state["steps"] else None
    if target is None:
        issues.append(_issue("orphan_step_event", where, f"{phase} {slot} outside any step"))
    elif target["trainer_step"] != trainer_step:
        issues.append(_issue("trainer_step_mismatch", where,
                             f"{phase} {slot} names step {trainer_step}, "
                             f"the step is {target['trainer_step']}"))
    elif (phase == "pre_step" and target["first_update_seq"] is not None) or (
            slot == "scheduler" and call != target["call"]):
        reason = (f"pre_step {slot} after the update began (seq {target['first_update_seq']})"
                  if slot != "scheduler" or call == target["call"]
                  else f"{phase} scheduler names call {call}, the step is {target['call']}")
        issues.append(_issue("trainer_step_evidence_misordered", where, reason))
        target["misordered"].append(where)
    elif target[slot][phase] is not None:
        issues.append(_issue(f"duplicate_{slot}", where, f"second {phase} {slot}"))
    else:
        target[slot][phase] = value


def _export_fields(export):
    if type(export) is not dict or type(export.get("status")) is not str:
        return {"status": "malformed", "path": None, "synchronized": None,
                "manifest_sha256": None}
    hashed = export["status"] == "hashed"
    return {"status": export["status"], "path": export.get("path"),
            "synchronized": export.get("synchronized"),
            "manifest_sha256": export.get("manifest_sha256") if hashed else None}


def _applied_lr(value, where, issues):
    """``None`` (not reported) or a list of finite numbers; anything else is an issue."""
    if value is None or (type(value) is list and all(_real(v) for v in value)):
        return value
    issues.append(_issue("applied_lr_invalid", where,
                         "optimizer_reduce lr must be null or a list of finite numbers"))
    return None


def _consume(state, event, issues, blobs):
    sid, kind, seq = state["stream_id"], event.get("kind"), event["seq"]
    where, step = f"{sid}#{seq}", state["open"]
    if kind in _OPENING_KINDS and type(event.get("call")) is int:
        if event["call"] in state["calls"]:
            issues.append(_issue("call_reused", where,
                                 f"{kind} reuses call id {event['call']} within this stream"))
        state["calls"].add(event["call"])
    if kind in _UPDATE_KINDS and step is not None and step["first_update_seq"] is None:
        step["first_update_seq"] = seq
    if kind == "trainer_configured":
        if state["configured"] is not None:
            issues.append(_issue("trainer_configured_repeated", where, "second trainer init"))
            return
        settings = {name: event.get(name) for name in _CONFIGURED}
        bad = [n for n, v in settings.items() if v is not None and not _count(v)]
        bad += [n for n in ("mini_batch", "iterations") if settings[n] == 0]
        beta = event.get("kl_beta")
        if bad or not (beta is None or _real(beta)):
            issues.append(_issue("trainer_configured_invalid", where,
                                 f"invalid trainer settings {sorted(bad) or ['kl_beta']}"))
            return
        state["configured"] = {**settings, "kl_beta": beta, "seq": seq}
    elif kind == "trainer_step_started":
        call, current = event.get("call"), event.get("current_step")
        if type(call) is not int or not _count(current):
            issues.append(_issue("malformed_trainer_event", where, "step start needs call/step"))
            return
        if step is not None:
            issues.append(_issue("nested_trainer_step", where, "a step started inside a step"))
            _close_open(state, issues)
        if current in state["seen_steps"]:
            issues.append(_issue("trainer_step_repeated", where,
                                 f"trainer_step {current} started twice in this stream"))
        state["seen_steps"].add(current)
        state["open"] = _new_step(current, call, seq)
        state["steps"].append(state["open"])
    elif kind == "trainer_step_finished" or (
            kind == "call_raised" and event.get("target") == "step_training"):
        if step is None or event.get("call") != step["call"]:
            issues.append(_issue("unmatched_step_end", where, f"{kind} matches no open step"))
            return
        step["ended_seq"] = seq
        if kind == "trainer_step_finished":
            if event.get("current_step") != step["trainer_step"]:
                issues.append(_issue("trainer_step_mismatch", where, "end names another step"))
            metrics = event.get("metrics")
            batches = metrics.get("train/num_batches") if type(metrics) is dict else None
            if type(metrics) is dict and "train/num_batches" in metrics and not _count(batches):
                issues.append(_issue("num_batches_unreadable", where,
                                     f"train/num_batches {batches!r} is not a count; the E20 "
                                     "num_batches cross-check cannot run", "warning"))
            step.update(outcome="finished",
                        reported_num_batches=batches if _count(batches) else None)
        else:
            step.update(outcome="raised", error_type=event.get("error_type"))
        state["open"] = None
    elif kind == "call_raised":
        target = event.get("target")
        if target == "train_minibatch" and _minibatch(step, event.get("call")) is not None:
            _minibatch(step, event.get("call"))["raised"] = event.get("error_type")
        elif target in TRAINER_TARGETS and step is not None:
            step["raised_calls"].append({"target": target, "call": event.get("call"),
                                         "error_type": event.get("error_type")})
    elif kind == "parameter_snapshot":
        problem = _snapshot_problem(event)
        if problem:
            issues.append(_issue("snapshot_invalid", where, problem))
        elif event["phase"] == "init":
            if state["init_snapshot"] is not None:
                issues.append(_issue("duplicate_snapshots", where, "second init snapshot"))
            else:
                state["init_snapshot"] = event
        else:
            _attach(state, "snapshots", event["phase"], event["trainer_step"], event, where,
                    issues)
    elif kind == "scheduler_state":
        if (set(event) != _SCHEDULER_KEYS or event.get("phase") not in STEP_PHASES
                or not _count(event.get("trainer_step"))
                or type(event.get("last_epoch")) is not int or type(event.get("lrs")) is not list
                or not all(_real(v) for v in event["lrs"])):
            issues.append(_issue("scheduler_state_invalid", where, "malformed scheduler_state"))
            return
        _attach(state, "scheduler", event["phase"], event["trainer_step"],
                {"last_epoch": event["last_epoch"], "lrs": list(event["lrs"])}, where, issues,
                call=event.get("call"))
    elif kind in ("replay_pool_prepared", "minibatch_started", "checkpoint_save") and (
            step is None):
        if kind != "checkpoint_save" or event.get("phase") == "returned":
            issues.append(_issue("trainer_event_outside_step", where, f"{kind} outside a step"))
    elif kind == "replay_pool_prepared":
        if step["pool_call"] is not None or not _count(event.get("rows")):
            issues.append(_issue("replay_pool_invalid", where, "repeated or malformed pool"))
            return
        step.update(rows=event["rows"], pool_call=event.get("call"))
    elif kind == "minibatch_started":
        call = event.get("call")
        if type(call) is not int or _minibatch(step, call) is not None:
            issues.append(_issue("malformed_trainer_event", where, "minibatch call reused"))
            return
        record = {"call": call, "rows": event.get("rows"), "pool_call": event.get("pool_call"),
                  "new_log_probs": None, "kl_present": None, "forward_seq": None,
                  "finished": False, "raised": None, "reduce": None}
        for name in ("row_indices", "advantages", "old_logprobs", "is_padding"):
            record[name] = _resolve(event.get(name), blobs, f"{where}.{name}", issues)
        step["minibatches"].append(record)
    elif kind == "replay_forward":
        record = _minibatch(step, event.get("minibatch_call"))
        if record is None or record["forward_seq"] is not None:
            issues.append(_issue("forward_without_minibatch", where,
                                 "replay_forward pairs with no open minibatch, or twice"))
            return
        record.update(forward_seq=seq, kl_present=event.get("kl_present"),
                      new_log_probs=_resolve(event.get("log_probs"), blobs, f"{where}.log_probs",
                                             issues))
    elif kind == "minibatch_finished":
        record = _minibatch(step, event.get("call"))
        if record is None:
            issues.append(_issue("unmatched_minibatch_end", where, "no such open minibatch"))
            return
        record["finished"] = True
    elif kind in ("optimizer_reduce", "optimizer_reduce_raised"):
        record = _minibatch(step, event.get("minibatch_call"))
        if record is None:
            issues.append(_issue("reduce_without_minibatch", where,
                                 "an optimizer reduce outside any captured minibatch"))
            if step is not None:
                step["orphan_reduces"] += 1
            return
        if record["reduce"] is not None:
            issues.append(_issue("reduce_repeated", where, "second reduce in one minibatch"))
            return
        calls = event.get("optimizer_step_calls")
        record["reduce"] = {"status": "returned" if kind == "optimizer_reduce" else "raised",
                            "grad_norm": event.get("grad_norm"), "optimizer_step_calls": calls,
                            "applied_lr": _applied_lr(event.get("lr"), where, issues),
                            "error_type": event.get("error_type")}
        if calls is not None and not _count(calls):
            issues.append(_issue("malformed_trainer_event", where, "bad optimizer_step_calls"))
        elif _count(calls) and calls > 1:
            issues.append(_issue("optimizer_step_called_repeatedly", where,
                                 f"optimizer.step() ran {calls} times in one reduce"))
    elif kind == "checkpoint_save":
        if event.get("phase") != "returned":
            return
        call = event.get("call")
        if type(call) is not int or call in state["exports"]:
            issues.append(_issue("malformed_trainer_event", where, "checkpoint call reused"))
            return
        state["exports"][call] = {"stream_id": sid, "current_step": event.get("current_step"),
                                  "save_call": call, "returned_seq": seq, "settled_seq": None,
                                  **_export_fields(event.get("export"))}
        step["export_calls"].append(call)
    elif kind == "checkpoint_export_settled":
        record = state["exports"].get(event.get("save_call"))
        if record is None or record["status"] != "pending_async_write":
            issues.append(_issue("settle_without_pending_export", where,
                                 "settled export matches no pending save"))
            return
        record.update(_export_fields(event.get("export")), settled_seq=seq)
    elif kind == "checkpoint_export_settle_error":
        issues.append(_issue("export_settle_error", where,
                             f"re-hashing step {event.get('current_step')} failed "
                             f"({event.get('error_type')}); the export stays pending", "warning"))
    elif kind == "event_oversize" and event.get("original_kind") in TRAINER_EVENT_KINDS:
        original = event["original_kind"]
        code = "snapshot_oversize" if original == "parameter_snapshot" else (
            "trainer_event_oversize")
        issues.append(_issue(code, where, f"a {original} event exceeded max_event_bytes; only "
                             "its hash and size were kept"))
        target = step if step is not None else (state["steps"][-1] if state["steps"] else None)
        if target is not None:
            target["oversize"].append(original)
    elif kind == "capture_extract_error" and event.get("target") in TRAINER_TARGETS:
        issues.append(_issue("capture_extract_error", where,
                             f"{event.get('target')}/{event.get('phase')}: "
                             f"{event.get('error_type')}; that evidence is missing"))
        if step is not None:
            step["extract_errors"].append({"target": event.get("target"),
                                           "phase": event.get("phase"),
                                           "error_type": event.get("error_type")})


def _close_open(state, issues):
    step = state["open"]
    if step is not None:
        step["outcome"] = "unterminated"
        issues.append(_issue("trainer_step_unterminated",
                             f"{state['stream_id']}:trainer_step:{step['trainer_step']}",
                             "the step has no finish or raise record"))
        state["open"] = None


def _check_minibatches(sid, state, step, issues):
    """Row-level consistency of one step's minibatches (blocking when contradicted)."""
    base = f"{sid}:trainer_step:{step['trainer_step']}"
    valid_rows, unchecked = True, set()
    for position, mb in enumerate(step["minibatches"]):
        where = f"{base}/minibatch:{position}"
        if step["pool_call"] is not None and mb["pool_call"] != step["pool_call"]:
            issues.append(_issue("minibatch_pool_mismatch", where, "drawn from another pool"))
        if not _count(mb["rows"]):
            issues.append(_issue("malformed_trainer_event", where, "minibatch rows missing"))
            valid_rows = False
            unchecked.add("minibatch_rows_missing")
            continue
        for name in _MINIBATCH_LISTS:
            if mb[name] is not None and len(mb[name]) != mb["rows"]:
                issues.append(_issue("minibatch_rows_mismatch", where,
                                     f"{name} has {len(mb[name])} entries for {mb['rows']} rows"))
        indices = mb["row_indices"]
        if indices is None:
            valid_rows = False
            unchecked.add("row_indices_unavailable")
        elif any(index is None for index in indices):
            issues.append(_issue("row_index_unresolved", where,
                                 "a minibatch sample is not in the captured replay pool"))
            valid_rows = False
            unchecked.add("row_index_unresolved")
        elif step["rows"] is None:
            valid_rows = False  # the pool was not captured: the range cannot be checked
            unchecked.add("rows_not_captured")
        elif any(not _count(i) or i >= step["rows"] for i in indices):
            issues.append(_issue("row_index_out_of_pool", where, "row index outside the pool"))
            valid_rows = False
        pads, advantages = mb["is_padding"], mb["advantages"]
        if pads is not None and advantages is not None and len(pads) == len(advantages):
            for row, (pad, advantage) in enumerate(zip(pads, advantages, strict=True)):
                if type(pad) is list and pad and all(v is True for v in pad) and advantage != 0.0:
                    issues.append(_issue("padding_advantage_nonzero", f"{where}/row:{row}",
                                         "pinned _prepare_training_data gives padding 0.0"))
        if step["outcome"] == "finished" and not mb["finished"]:
            issues.append(_issue("minibatch_unterminated", where, "no minibatch_finished"))
        if mb["finished"] and mb["reduce"] is None:
            issues.append(_issue("optimizer_reduce_missing", where, "a finished minibatch has no "
                                 "reduce record (pinned _train_minibatch always reduces)"))
    configured = state["configured"] or {}
    size, iterations, rows = (configured.get("mini_batch"), configured.get("iterations"),
                              step["rows"])
    permuted = False
    if (valid_rows and step["outcome"] == "finished" and _count(rows) and rows > 0
            and type(size) is int and type(iterations) is int and size > 0):
        per = -(-rows // min(size, rows))
        if len(step["minibatches"]) == iterations * per:
            permuted = True
            for iteration in range(iterations):
                chunk = step["minibatches"][iteration * per:(iteration + 1) * per]
                seen = sorted(i for mb in chunk for i in mb["row_indices"])
                if seen != list(range(rows)):
                    issues.append(_issue("iteration_rows_not_a_permutation",
                                         f"{base}/iteration:{iteration}",
                                         "one iteration's minibatches must cover every pool row "
                                         "exactly once (pinned randperm split)"))
    if not permuted:
        unchecked.add("iteration_permutation_not_checked")
    step["row_checks"] = {"status": "not_run" if unchecked else "run",
                          "reasons": sorted(unchecked)}


def _reduce_reasons(mb):
    """(reasons, blocking reasons, row) for one minibatch's reduce."""
    reduce = mb["reduce"]
    if reduce is None:
        return ["optimizer_reduce_missing"], [], None
    if reduce["status"] == "raised":
        return ["optimizer_reduce_raised"], [], None
    calls, norm = reduce["optimizer_step_calls"], reduce["grad_norm"]
    if calls is None:
        return ["optimizer_step_unobserved"], [], None
    if not _count(calls) or calls > 1:
        return [], ["optimizer_step_called_repeatedly"], None
    if not ((_real(norm) and norm >= 0) or norm in NONFINITE_NORMS):
        return [], ["grad_norm_invalid"], None
    return [], [], {"grad_norm": norm, "optimizer_step_called": calls == 1}


def _assembly(reasons, blocking):
    status = "blocked" if blocking else "not_estimable" if reasons else "passed"
    return {"assembly": status, "reasons": sorted(set(reasons) | set(blocking))}


def _assemble_e20(state, step):
    reasons, blocking, rows = [], [], []
    if step["outcome"] != "finished":
        reasons.append(f"trainer_step_{step['outcome']}")
    configured = state["configured"] or {}
    if type(configured.get("mini_batch")) is not int or type(
            configured.get("iterations")) is not int:
        reasons.append("step_geometry_not_captured")
    if step["rows"] is None:
        reasons.append("rows_not_captured")
    if step["orphan_reduces"]:
        blocking.append("reduce_without_minibatch")
    for mb in step["minibatches"]:
        more, block, row = _reduce_reasons(mb)
        reasons += more
        blocking += block
        rows.append(row)
    result = _assembly(reasons, blocking)
    if result["assembly"] == "passed":
        result["input"] = {"trainer_step": step["trainer_step"], "rows": step["rows"],
                           "mini_batch": configured["mini_batch"],
                           "iterations": configured["iterations"],
                           "source": "in_process_capture", "minibatches": rows,
                           "reported_num_batches": step["reported_num_batches"]}
    return result


def _layout_changed(pre, post):
    shared = set(pre["tensors"]) & set(post["tensors"])
    return sorted(n for n in shared if (pre["tensors"][n]["hash_scope"], pre["tensors"][n]["shard"])
                  != (post["tensors"][n]["hash_scope"], post["tensors"][n]["shard"]))


def _parameter_changes(pre, post):
    """Per-step tensor changes (E03 evidence) between two hashed snapshots of one stream."""
    a, b = pre["tensors"], post["tensors"]
    shared = sorted(set(a) & set(b))
    fields = ("kind", "dtype", "shape", "sha256")
    changed = [n for n in shared if any(a[n][k] != b[n][k] for k in fields)]
    return {"hash_comparability": ("full" if hash_comparability(pre) == "full"
                                   and hash_comparability(post) == "full"
                                   else "shard_to_shard_same_stream"),
            "changed": changed, "unchanged": [n for n in shared if n not in changed],
            "only_in_pre_step": sorted(set(a) - set(b)),
            "only_in_post_step": sorted(set(b) - set(a))}


def _assemble_e07(step):
    reasons, blocking = [], []
    if step["outcome"] != "finished":
        reasons.append(f"trainer_step_{step['outcome']}")
    row = None
    if len(step["minibatches"]) != 1:
        reasons.append("per_minibatch_snapshots_unavailable")
    else:
        more, block, row = _reduce_reasons(step["minibatches"][0])
        reasons += more
        blocking += block
        if row is not None and row["grad_norm"] in NONFINITE_NORMS:
            reasons.append("grad_norm_nonfinite")
    if step["orphan_reduces"]:
        blocking.append("reduce_without_minibatch")
    if step["misordered"]:
        blocking.append("trainer_step_evidence_misordered")
    if None in step["scheduler"].values():
        reasons.append("scheduler_state_missing")
    if "parameter_snapshot" in step["oversize"]:
        reasons.append("snapshot_oversize")
    pre, post = step["snapshots"]["pre_step"], step["snapshots"]["post_step"]
    comparability, changes = None, None
    if pre is None or post is None:
        reasons.append("snapshot_missing")
    elif not (_hashed(pre) and _hashed(post)):
        reasons.append("snapshot_unhashed")
    elif _layout_changed(pre, post):
        blocking.append("shard_layout_changed")
    else:
        changes = _parameter_changes(pre, post)
        comparability = changes["hash_comparability"]
    result = {**_assembly(reasons, blocking), "hash_comparability": comparability}
    if result["assembly"] == "passed":
        result["input"] = {
            "trainer_step": step["trainer_step"],
            "scheduler_before": step["scheduler"]["pre_step"],
            "scheduler_after": step["scheduler"]["post_step"],
            "minibatches": [{"index": 0, "grad_norm": row["grad_norm"],
                             "optimizer_step_called": row["optimizer_step_called"],
                             "before": snapshot_digest(pre), "after": snapshot_digest(post)}]}
    return result, changes


def _snapshot_summary(event):
    if event is None:
        return None
    hashed = _hashed(event)
    return {"seq": event["seq"], "hashed": hashed,
            "hash_comparability": hash_comparability(event) if hashed else None,
            "tensors": len(event["tensors"]),
            "optimizer_params": len(event["optimizer_state"])}


def _step_out(step, e20, e07, changes):
    minibatches = []
    for mb in step["minibatches"]:
        reduce = mb["reduce"] or {}
        calls = reduce.get("optimizer_step_calls")
        minibatches.append({
            **{name: mb[name] for name in ("call", "rows", "pool_call", *_MINIBATCH_LISTS,
                                          "kl_present", "finished", "raised")},
            "reduce_status": reduce.get("status", "missing"), "grad_norm": reduce.get("grad_norm"),
            "optimizer_step_calls": calls,
            "optimizer_step": {0: "skipped", 1: "taken"}.get(calls) if _count(calls) else None,
            "applied_lr": reduce.get("applied_lr")})
    return {**{k: step[k] for k in ("trainer_step", "call", "outcome", "error_type",
                                    "started_seq", "ended_seq", "rows", "pool_call",
                                    "reported_num_batches", "orphan_reduces", "oversize",
                                    "extract_errors", "raised_calls", "export_calls",
                                    "misordered", "row_checks")},
            "minibatches": minibatches,
            "scheduler": dict(step["scheduler"]),
            "snapshots": {p: _snapshot_summary(step["snapshots"][p]) for p in STEP_PHASES},
            "parameter_changes": changes,
            "e20": {k: v for k, v in e20.items() if k != "input"},
            "e07": {k: v for k, v in e07.items() if k != "input"}}


def build_trainer_step_records(streams, *, scope, blobs=None):
    """Per-step trainer records and the exact E20/E07 consumer inputs (module docstring; L1).

    ``streams``: ``streams_from_reports`` output (one stream per process; never merged).
    ``blobs``: optional ``{sha256: bytes}`` from the capture blob store for bounded lists.
    """
    issues = []
    _check_scope(scope, issues)
    if not streams:
        issues.append(_issue("no_capture_streams", "$.streams", "no capture stream supplied"))
    inputs = _reader_verdicts(streams, issues)
    blob_map = _blob_map(blobs, issues)
    states = {}
    for sid, event in _ordered(streams, issues):
        state = states.setdefault(sid, {"stream_id": sid, "configured": None,
                                        "init_snapshot": None, "steps": [], "open": None,
                                        "seen_steps": set(), "exports": {}, "calls": set()})
        _consume(state, event, issues, blob_map)
    out_streams, e20_steps, e07_steps, exports = [], {}, [], []
    for sid in sorted(states):
        state = states[sid]
        _close_open(state, issues)
        if state["configured"] is None and not state["steps"] and (
                state["init_snapshot"] is None):
            continue  # a rollout or controller stream: no trainer evidence
        steps_out = []
        for step in state["steps"]:
            _check_minibatches(sid, state, step, issues)
            e20 = _assemble_e20(state, step)
            e07, changes = _assemble_e07(step)
            if "input" in e20:
                e20_steps.setdefault(sid, []).append(e20["input"])
            if "input" in e07:
                e07_steps.append({"stream_id": sid, "trainer_step": step["trainer_step"],
                                  "hash_comparability": e07["hash_comparability"],
                                  "input": e07["input"]})
            steps_out.append(_step_out(step, e20, e07, changes))
        for call in sorted(state["exports"]):
            record = state["exports"][call]
            if record["status"] == "pending_async_write":
                issues.append(_issue("export_pending_unsettled",
                                     f"{sid}:checkpoint:{record['current_step']}",
                                     "the export writer never confirmed completion; the export "
                                     "is not hashed", "warning"))
            exports.append(record)
        out_streams.append({"stream_id": sid, "configured": state["configured"],
                            "init_snapshot": _snapshot_summary(state["init_snapshot"]),
                            "steps": steps_out})
    total = sum(len(s["steps"]) for s in out_streams)
    if streams and not total:
        issues.append(_issue("no_trainer_steps", "$.streams",
                             "no trainer_step_started record; no trainer step was captured"))
    body = {"streams": out_streams, "e20_trainer_steps": e20_steps,
            "e07_trainer_steps": e07_steps, "exports": exports,
            "counts": {"trainer_steps": total,
                       "minibatches": sum(len(t["minibatches"]) for s in out_streams
                                          for t in s["steps"]),
                       "e20_assembled": sum(len(v) for v in e20_steps.values()),
                       "e07_assembled": len(e07_steps)}}
    status = _status(issues)
    if status == "passed" and not (body["counts"]["e20_assembled"]
                                   or body["counts"]["e07_assembled"]):
        status = "not_estimable"  # steps were captured, but no consumer input could be built
    return {"artifact_kind": TRAINER_RECORDS_KIND, "schema_version": SCHEMA_VERSION,
            "scope": scope, "producer": _producer(), "inputs": inputs,
            "units": {"seq": "record index within one capture stream",
                      "grad_norm": "reduced L2 gradient norm (dimensionless)",
                      "rows": "replay pool rows", "lrs": "learning rate per param group"},
            "status": status, "issues": issues, **body,
            "records_hash": identity_hash(body)}
