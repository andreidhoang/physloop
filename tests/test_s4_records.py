"""Known answers for the S4 records ledger (E06, E09, E15, E17; R3 finding 6). L1.

Streams are hand-written synthetic capture events; expected ledgers are derived by hand from the
upstream retry/collect semantics (streaming_worker.py:132-319 at the AlpaGym pin): a failure
re-enqueues a sibling with a fresh session UUID while budget remains, a payload whose budget is
exhausted resolves with ``[]`` and its pending siblings are skipped, and the group is the first
``n_target`` collected episodes.
"""

import copy
import hashlib
import math
import random

import pytest
from test_s4_registration import valid

from gate.s2_identity import canonical_json_bytes, identity_hash
from teacher_rl import adapters, params, records
from teacher_rl.checks import steps as step_checks
from teacher_rl.export_audit import tensor_table
from teacher_rl.records import (
    build_attempt_ledger,
    check_run_identity,
    check_trace_integrity,
    describe_run_randomness,
    randomness_record,
)
from teacher_rl.registration import compile_overrides

SCENE = "clipgt-a309e228-26e1-423e-a44c-cb00aa7378cb"
S1, S2, S3, S4, S5 = ("1" * 32, "2" * 32, "3" * 32, "4" * 32, "5" * 32)
H = {name: c * 64 for name, c in (("cfg", "a"), ("lock", "b"), ("base", "c"), ("tree", "d"),
                                   ("w10", "e"), ("w20", "f"))}


def stream(sid, events, reader_status="passed", reader_issues=()):
    """A synthetic stream as ``streams_from_reports`` shapes it (reader verdict included)."""
    return {"stream_id": sid, "events": [{"seq": i, **e} for i, e in enumerate(events)],
            "chain_head": None, "reader_status": reader_status,
            "reader_issues": list(reader_issues)}


def completion(session, nonce="9" * 32):
    return f"/run/artifacts/{SCENE}_{session}_{nonce}.json"


def bound(prompt, instance=1, n_target=2, retries_left=3):
    return {"kind": "payload_bound", "payload_instance": instance, "prompt_idx": prompt,
            "n_target": n_target, "retries_left": retries_left}


def submitted(prompt, instance=1):
    return {"kind": "payload_submitted", "payload_instance": instance, "prompt_idx": prompt}


def att(kind, prompt, session, instance=1, **fields):
    event = {"kind": kind, "payload_instance": instance, "prompt_idx": prompt,
             "session_uuid": session, **fields}
    if kind == "attempt_started":
        event.setdefault("scene_id", SCENE)
    return event


def rollout_events(prefetch=False, versions=(4,)):
    """Prompt 7, n_target 2, budget 3: s1 fails (reward KeyError) -> retry s3; s2, s3 succeed."""
    gen = [{"kind": "generation_entered", "call": 100 + i, "prompt_idxs": [7],
            "current_weight_version": v, "is_validation": False}
           for i, v in enumerate(versions)]
    body = [
        att("attempt_started", 7, S1, attempt=0),
        att("attempt_started", 7, S2, attempt=0),
        att("attempt_failed", 7, S1, attempt=0, cause_type="KeyError",
            cause_message="'collision_any'", outcome="retry", retries_left_after=2),
        att("attempt_started", 7, S3, attempt=1),
        att("attempt_succeeded", 7, S2, attempt=0, collected_index=0, reward_total=1.0),
        att("attempt_succeeded", 7, S3, attempt=1, collected_index=1, reward_total=-9.0),
    ]
    opened = {"kind": "stream_opened"}
    exited = {"kind": "generation_exited", "call": 100 + len(versions) - 1, "results": 1}
    if prefetch:  # enqueue_prefetch_payloads submits before the generation call
        return [opened, bound(7), submitted(7), *gen, submitted(7), *body, exited]
    return [opened, gen[0], bound(7), submitted(7), *body, exited]


def find(events, kind, session):
    return next(i for i, e in enumerate(events) if e["kind"] == kind
                and e.get("session_uuid") == session)


def ledger_for(events):
    return build_attempt_ledger([stream("rollout-1", events)], scope="synthetic")


def codes(report, severity=None):
    return {i["code"] for i in report["issues"] if severity is None or i["severity"] == severity}


# ---------------------------------------------------------------- attempt ledger


def test_retry_group_known_answer():
    ledger = ledger_for(rollout_events())
    assert ledger["status"] == "passed", ledger["issues"]
    assert ledger["counts"] == {"payloads": 1, "attempts": 3, "succeeded": 2,
                                "silent_retries": 1, "dropped_payloads": 0,
                                "unterminated_attempts": 0, "never_started_siblings": 0}
    assert ledger["failure_causes"] == {"KeyError": 1}
    (group,) = ledger["groups"]
    assert group["group_id"] == "rollout-1:7:1"
    assert group["submissions"] == 1
    assert group["members"] == [S2, S3]
    assert group["member_rewards"] == [1.0, -9.0]
    assert group["member_is_retry"] == [False, True]
    assert group["complete"] is True and group["dropped"] is False
    assert group["failed_attempts"] == [{"session_uuid": S1, "attempt": 0,
                                         "cause_type": "KeyError", "outcome": "failed_retry"}]


def test_exhausted_payload_is_kept_with_never_started_sibling():
    events = [
        bound(9, retries_left=0),
        submitted(9),
        att("attempt_started", 9, S4, attempt=0),
        att("attempt_failed", 9, S4, attempt=0, cause_type="ValueError",
            cause_message="timestamps", outcome="dropped", retries_left_after=-1),
    ]
    ledger = ledger_for(events)
    assert ledger["status"] == "passed"
    assert codes(ledger) == {"payload_dropped"}
    (group,) = ledger["groups"]
    assert group["dropped"] is True and group["complete"] is False
    assert group["members"] == [] and group["never_started_siblings"] == 1
    assert ledger["counts"]["dropped_payloads"] == 1


def test_resubmission_after_resolution_is_a_new_payload_instance():
    first = rollout_events()[1:-1]
    second = [
        bound(7, instance=2),
        att("attempt_started", 7, S4, instance=2, attempt=0),
        submitted(7, instance=2),  # threads may start siblings before submission returns
        att("attempt_started", 7, S5, instance=2, attempt=0),
        att("attempt_succeeded", 7, S5, instance=2, attempt=0, collected_index=0,
            reward_total=0.5),
        att("attempt_succeeded", 7, S4, instance=2, attempt=0, collected_index=1,
            reward_total=0.25),
    ]
    ledger = ledger_for(first + second)
    assert ledger["status"] == "passed", ledger["issues"]
    assert [g["group_id"] for g in ledger["groups"]] == ["rollout-1:7:1", "rollout-1:7:2"]
    assert ledger["groups"][1]["members"] == [S5, S4]  # collected order, not start order


def test_e06_reconstruction_is_invariant_under_permutation():
    other = [
        bound(3),
        submitted(3),
        att("attempt_started", 3, S4, attempt=0),
        att("attempt_started", 3, S5, attempt=0),
        att("attempt_succeeded", 3, S4, attempt=0, collected_index=0, reward_total=2.0),
        att("attempt_succeeded", 3, S5, attempt=0, collected_index=1, reward_total=2.0),
    ]
    streams = [stream("rollout-a", rollout_events()), stream("rollout-b", other)]
    reference = build_attempt_ledger(streams, scope="synthetic")
    rng = random.Random(20261109)
    for _ in range(25):
        shuffled = copy.deepcopy(streams)
        rng.shuffle(shuffled)
        for item in shuffled:
            rng.shuffle(item["events"])
        assert build_attempt_ledger(shuffled, scope="synthetic")["ledger_hash"] == reference["ledger_hash"]
    assert [g["group_id"] for g in reference["groups"]] == ["rollout-a:7:1", "rollout-b:3:1"]


@pytest.mark.parametrize(("mutate", "code"), [
    (lambda ev: ev.append(att("attempt_started", 7, S2, attempt=0)), "session_reused"),
    (lambda ev: ev.append(att("attempt_started", 99, S5, instance=9, attempt=0)),
     "attempt_without_payload"),
    (lambda ev: ev.insert(1, att("attempt_started", 7, S5, attempt=0)),
     "attempt_without_payload"),  # an attempt before its payload_bound record
    (lambda ev: ev.append(att("attempt_started", 8, S5, attempt=0)),
     "attempt_without_payload"),  # prompt differs from the bound instance
    (lambda ev: ev.append(att("attempt_succeeded", 7, S5, collected_index=0)),
     "terminal_without_start"),
    (lambda ev: ev.append(att("attempt_succeeded", 7, S2, collected_index=0)),
     "duplicate_terminal"),
    (lambda ev: ev.pop(find(ev, "attempt_succeeded", S3)), "attempt_unterminated"),
    (lambda ev: ev.append(bound(7)), "malformed_payload_instance"),
    (lambda ev: ev.append(submitted(7, instance=5)), "submission_without_payload"),
])
def test_ledger_negative_controls(mutate, code):
    events = rollout_events()
    mutate(events)
    ledger = ledger_for(events)
    assert ledger["status"] == "blocked"
    assert code in codes(ledger, "blocking")


def test_omitted_and_duplicated_sequence_numbers_block():
    gap = stream("rollout-1", rollout_events())
    del gap["events"][3]
    assert "sequence_gap" in codes(build_attempt_ledger([gap], scope="synthetic"), "blocking")
    dup = stream("rollout-1", rollout_events())
    dup["events"][2]["seq"] = 1
    assert "duplicate_seq" in codes(build_attempt_ledger([dup], scope="synthetic"), "blocking")
    twins = [stream("same", rollout_events()), stream("same", rollout_events())]
    assert "duplicate_stream" in codes(build_attempt_ledger(twins, scope="synthetic"), "blocking")


def test_missing_member_without_drop_is_incomplete():
    events = rollout_events()
    index = find(events, "attempt_succeeded", S3)
    events[index] = {**events[index], "collected_index": None}  # not collected, no drop
    assert "group_incomplete" in codes(ledger_for(events), "blocking")


# ---------------------------------------------------------------- E09 trace integrity


def trainer_stream(rows):
    return stream("policy-1", [{"kind": "stream_opened"},
                               {"kind": "trainer_rollouts_received", "current_step": 5,
                                "rollouts": rows}])


def row(session, version=4, prompt=7):
    return {"prompt_idx": prompt, "completion": completion(session), "weight_version": version}


def integrity(rows, events=None, **kwargs):
    rollout = stream("rollout-1", events or rollout_events())
    ledger = build_attempt_ledger([rollout], scope="synthetic")
    return check_trace_integrity(ledger, [rollout, trainer_stream(rows)], **kwargs)


def test_strict_parity_cohort_passes_with_uniform_matching_versions():
    report = integrity([row(S2), row(S3)], cohort="strict_parity", expected_version=4)
    assert report["status"] == "passed", report["issues"]
    assert report["versions"] == {"rollout-1:7:1": {"version": 4, "reason": "entry_version"}}
    assert report["received"] == 2


@pytest.mark.parametrize(("rows", "code"), [
    ([row(S2), row(S2), row(S3)], "duplicate_record"),
    ([row(S2)], "omitted_record"),
    ([row(S2), row(S3), row(S1)], "not_a_group_member"),
    ([row(S2), row(S3, version=3)], "misreported_version"),
    ([row(S2), row(S3, prompt=8)], "prompt_idx_mismatch"),
    ([row(S2), {"prompt_idx": 7, "completion": "/run/artifacts/odd.json",
                "weight_version": 4}, row(S3)], "unparseable_completion"),
])
def test_e09_negative_controls(rows, code):
    report = integrity(rows, cohort="async_disclosed")
    assert report["status"] == "blocked"
    assert code in codes(report, "blocking")


def test_strict_cohort_needs_the_registered_version():
    assert "expected_version_unregistered" in codes(
        integrity([row(S2), row(S3)], cohort="strict_parity"), "blocking")
    mismatch = integrity([row(S2), row(S3)], cohort="strict_parity", expected_version=5)
    assert codes(mismatch, "blocking") == {"strict_cohort_version_mismatch"}
    assert "unsupported_cohort" in codes(integrity([row(S2), row(S3)], cohort="any"))


def test_prefetched_payload_version_is_undetermined_not_assumed():
    events = rollout_events(prefetch=True)
    strict = integrity([row(S2), row(S3)], events, cohort="strict_parity", expected_version=4)
    assert "version_undetermined" in codes(strict, "blocking")
    relaxed = integrity([row(S2), row(S3)], events, cohort="async_disclosed")
    assert relaxed["status"] == "passed"
    assert codes(relaxed, "warning") == {"version_undetermined"}


def test_trace_spanning_two_versions_is_rejected_as_mixed():
    events = rollout_events(prefetch=True, versions=(4, 5))
    report = integrity([row(S2), row(S3)], events, cohort="async_disclosed")
    assert "mixed_versions_in_trace" in codes(report, "blocking")
    assert report["versions"]["rollout-1:7:1"] == {"version": None, "reason": "mixed_versions",
                                                    "versions": [4, 5]}


# ---------------------------------------------------------------- E15 run identity


def launch(launch_id, ordinal, phase, **overrides):
    record = {"run_id": "run-a", "launch_id": launch_id, "ordinal": ordinal, "phase": phase,
              "effective_config_hash": H["cfg"], "uv_lock_sha256": H["lock"],
              "base_weights_hash": H["base"], "capture_tree_hash": H["tree"]}
    if phase == "resume":
        record.update(resume_from_step=10, resume_weights_hash=H["w10"])
    record.update(overrides)
    return record


CKPTS = [{"run_id": "run-a", "step": 10, "weights_hash": H["w10"]},
         {"run_id": "run-a", "step": 20, "weights_hash": H["w20"]}]


def test_start_and_matching_resume_pass_in_any_input_order():
    launches = [launch("L2", 1, "resume"), launch("L1", 0, "start")]
    assert check_run_identity(launches, CKPTS, scope="synthetic")["status"] == "passed"
    assert check_run_identity(list(reversed(launches)), CKPTS, scope="synthetic")["status"] == "passed"


@pytest.mark.parametrize(("launches", "code"), [
    ([launch("L1", 0, "start"), launch("L2", 1, "start")], "duplicate_launch"),
    ([launch("L1", 0, "start"), launch("L1", 1, "resume")], "duplicate_launch"),
    ([launch("L1", 0, "start"), launch("L2", 0, "resume")], "duplicate_launch"),
    ([launch("L2", 1, "resume")], "resume_without_start"),
    ([launch("L1", 0, "start"), launch("L2", 1, "resume", effective_config_hash=H["lock"])],
     "resume_identity_mismatch"),
    ([launch("L1", 0, "start"), launch("L2", 1, "resume", capture_tree_hash=H["cfg"])],
     "resume_identity_mismatch"),
    ([launch("L1", 0, "start"), launch("L2", 1, "resume", resume_weights_hash=H["w20"])],
     "resume_weight_mismatch"),
    ([launch("L1", 0, "start"), launch("L2", 1, "resume", resume_from_step=15)],
     "resume_checkpoint_unrecorded"),
    ([launch("L1", 0, "start", uv_lock_sha256=None)], "missing_identity_field"),
    ([launch("L1", "0", "start")], "malformed_launch"),
])
def test_e15_negative_controls(launches, code):
    report = check_run_identity(launches, CKPTS, scope="synthetic")
    assert report["status"] == "blocked"
    assert code in codes(report)


def test_conflicting_checkpoint_records_block():
    conflict = [*CKPTS, {"run_id": "run-a", "step": 10, "weights_hash": H["w20"]}]
    assert "checkpoint_conflict" in codes(check_run_identity([launch("L1", 0, "start")],
                                                             conflict, scope="synthetic"))


# ---------------------------------------------------------------- E17 randomness


def test_randomness_sources_record_unseeded_never_zero():
    effective = compile_overrides(valid())
    observed = {"train_seed": "unseeded", "dataloader_seed": 0, "force_determinism": True,
                "session_seeds": [{"session_uuid": S1, "seed": 11},
                                  {"session_uuid": S2, "seed": 11},
                                  {"session_uuid": S3, "seed": 12}]}
    record = randomness_record("run-a", effective, observed)
    assert record["status"] == "passed"
    by = {s["source"]: s for s in record["sources"]}
    assert by["cosmos_train_seed"] == {"source": "cosmos_train_seed", "configured": "unseeded",
                                       "observed": "unseeded", "status": "consistent"}
    assert by["dataloader_shuffle_seed"]["configured"] == 0
    assert by["rollout_global_rng"]["configured"] == "seeded_42"
    assert by["minibatch_order"]["status"] == "fixed_by_upstream"
    assert by["simulator_session_seeds"]["observed"] == {"sessions": 3, "distinct_seeds": 2}
    zero = randomness_record("run-a", effective, {**observed, "train_seed": 0})
    assert codes(zero) == {"randomness_mismatch"}
    missing = randomness_record("run-a", effective, {})
    assert missing["status"] == "unverified"  # absence is never a pass (A8-R-03)
    assert {s["status"] for s in missing["sources"][:3]} == {"unverified"}
    assert missing["sources"][-1]["status"] == "unverified"


def test_randomness_requires_a_compiled_configuration():
    reg = valid()
    del reg["owner"]["seed_modes"]
    assert randomness_record("run-a", compile_overrides(reg), {})["status"] == "blocked"


def test_runs_sharing_configured_seeds_are_reported_as_shared():
    effective = compile_overrides(valid())
    runs = [randomness_record(rid, effective, {}) for rid in ("run-b", "run-a", "run-c")]
    summary = describe_run_randomness(runs)
    assert summary["runs"] == ["run-a", "run-b", "run-c"]
    assert summary["shared_sources"]["dataloader_shuffle_seed"]["identical_across_runs"] is True
    assert summary["shared_sources"]["cosmos_train_seed"]["values"]["run-a"] == "unseeded"


def test_streams_from_reports_carries_reader_issues():
    report = {"stream_label": "x", "events": [], "issues": [
        {"code": "chain_broken", "severity": "blocking"}]}
    streams, issues = records.streams_from_reports([report])
    assert streams == [{"stream_id": "x", "events": [], "chain_head": None,
                        "reader_status": None, "reader_issues": report["issues"]}]
    assert issues[0]["code"] == "chain_broken"


def test_only_the_first_n_target_collected_successes_are_members():
    events = rollout_events()
    events.insert(-1, att("attempt_started", 7, S4, attempt=0))
    events.insert(-1, att("attempt_succeeded", 7, S4, attempt=0, collected_index=2,
                          reward_total=5.0))
    (group,) = ledger_for(events)["groups"]
    assert group["members"] == [S2, S3]
    assert group["member_rewards"] == [1.0, -9.0]


CONTRACT = {"artifact_kind", "schema_version", "scope", "producer", "inputs", "units", "issues"}


def test_every_records_artifact_carries_the_shared_contract():
    ledger = ledger_for(rollout_events())
    rollout = stream("rollout-1", rollout_events())
    integrity = check_trace_integrity(ledger, [rollout, trainer_stream([row(S2), row(S3)])],
                                      cohort="async_disclosed")
    identity = check_run_identity([launch("L1", 0, "start")], CKPTS, scope="synthetic")
    randomness = randomness_record("run-a", compile_overrides(valid()), {})
    summary = describe_run_randomness([randomness])
    for artifact in (ledger, integrity, identity, randomness, summary):
        assert set(artifact) >= CONTRACT, artifact["artifact_kind"]
        assert artifact["scope"] == "synthetic"
    assert ledger["inputs"] == [{"artifact_kind": "s4_capture_stream", "stream_id": "rollout-1",
                                 "chain_head": None, "reader_status": "passed"}]
    assert randomness["inputs"][0]["identity_hash"] == compile_overrides(valid())[
        "effective_config_hash"]


def test_unknown_scope_fails_closed():
    assert "unsupported_scope" in codes(build_attempt_ledger([], scope="real"), "blocking")
    assert "unsupported_scope" in codes(check_run_identity([], [], scope=None), "blocking")
    effective = compile_overrides(valid())
    full = {"train_seed": "unseeded", "dataloader_seed": 0, "force_determinism": True,
            "session_seeds": []}
    mixed = [randomness_record("a", effective, full),
             {**randomness_record("b", effective, full), "scope": "actual"}]
    assert codes(describe_run_randomness(mixed)) == {"mixed_scope"}


# ---------------------------------------------------------------- review regressions (A8-R-*)


def _sink_stream(tmp_path, label, events):
    from teacher_rl.capture.sink import EventSink

    limits = {"max_event_bytes": 65536, "max_stream_bytes": 4_000_000,
              "max_blob_bytes": 1000, "max_total_blob_bytes": 1000}
    sink = EventSink(tmp_path / "physloop_capture", stream_label=label, limits=limits)
    for event in events:
        sink.emit(event["kind"], {k: v for k, v in event.items() if k != "kind"})
    sink.close()
    return sink.path


def test_reader_blocked_stream_blocks_the_ledger_and_integrity(tmp_path):
    """A8-R-01: a tampered stream (reader: hash_mismatch) never yields a passed ledger."""
    from teacher_rl.capture.sink import read_stream

    path = _sink_stream(tmp_path, "rollout-1", rollout_events()[1:])
    lines = path.read_bytes().splitlines()
    lines[-2] = lines[-2].replace(b'"results":1', b'"results":9')
    path.write_bytes(b"\n".join(lines) + b"\n")
    report = read_stream(path, max_bytes=10**6)
    assert report["status"] == "blocked"
    streams, _ = records.streams_from_reports([report])
    ledger = build_attempt_ledger(streams, scope="synthetic")
    assert ledger["status"] == "blocked"
    assert "hash_mismatch" in codes(ledger, "blocking")
    assert ledger["inputs"][0]["reader_status"] == "blocked"
    clean = stream("rollout-1", rollout_events())
    good_ledger = build_attempt_ledger([clean], scope="synthetic")
    trainer = trainer_stream([row(S2), row(S3)])
    bad_trainer = {**trainer, "reader_status": "blocked", "reader_issues": [
        {"code": "chain_broken", "path": "x", "severity": "blocking", "evidence_refs": [],
         "message": "synthetic"}]}
    report = check_trace_integrity(good_ledger, [clean, bad_trainer], cohort="async_disclosed")
    assert report["status"] == "blocked" and "chain_broken" in codes(report, "blocking")


def test_stream_without_a_reader_verdict_is_unverified_input():
    raw = {"stream_id": "rollout-1",
           "events": [{"seq": i, **e} for i, e in enumerate(rollout_events())]}
    ledger = build_attempt_ledger([raw], scope="synthetic")
    assert ledger["status"] == "blocked" and "stream_unverified" in codes(ledger, "blocking")


def test_empty_inputs_never_pass():
    """A8-R-02: no streams, no payloads, no trainer records or no launches are not a pass."""
    empty = build_attempt_ledger([], scope="synthetic")
    assert empty["status"] == "not_run" and "no_capture_streams" in codes(empty, "blocking")
    probe_only = build_attempt_ledger([stream("policy-1", [{"kind": "stream_opened"}])],
                                      scope="synthetic")
    assert probe_only["status"] == "not_run"
    assert "no_payloads_captured" in codes(probe_only, "blocking")
    integrity_report = check_trace_integrity(empty, [], cohort="async_disclosed")
    assert integrity_report["status"] != "passed"
    nothing_received = integrity([], cohort="async_disclosed")
    assert nothing_received["status"] != "passed"
    assert "no_trainer_records" in codes(nothing_received, "blocking")
    identity = check_run_identity([], [], scope="synthetic")
    assert identity["status"] == "not_run" and "no_launch_records" in codes(identity, "blocking")


def test_randomness_rejects_forged_config_and_blocks_summary():
    """A8-R-03: the effective config is validated, nothing defaults, blocked records propagate."""
    forged = randomness_record("run-b", {"status": "passed"}, {"dataloader_seed": 0})
    assert forged["status"] == "blocked"
    assert "effective_config_invalid" in codes(forged, "blocking")
    effective = compile_overrides(valid())
    incomplete = copy.deepcopy(effective)
    incomplete["mapping"] = [e for e in incomplete["mapping"]
                             if e["key"] != "train.train_policy.dataloader_seed"]
    report = randomness_record("run-c", incomplete, {"dataloader_seed": 0})
    assert report["status"] == "blocked"
    assert "randomness_source_unmapped" in codes(report, "blocking")
    assert not any(s.get("status") == "consistent" for s in report["sources"])
    blocked = randomness_record("run-d", {"status": "blocked"}, {})
    summary = describe_run_randomness([blocked])
    assert summary["status"] == "blocked"
    assert "randomness_record_blocked" in codes(summary, "blocking")
    assert describe_run_randomness([])["status"] == "not_run"
    unverified = describe_run_randomness([randomness_record("run-a", effective, {})])
    assert unverified["status"] == "unverified"


def test_raised_generation_call_is_closed_and_exits_match_by_call():
    """A8-R-04: a raised call does not stay open and make a later group look mixed."""
    events = [
        {"kind": "stream_opened"},
        {"kind": "generation_entered", "call": 1, "prompt_idxs": [9],
         "current_weight_version": 4},
        {"kind": "call_raised", "target": "rollout_generation", "call": 1,
         "error_type": "RuntimeError", "message": "exhausted retries"},
        {"kind": "generation_entered", "call": 2, "prompt_idxs": [9],
         "current_weight_version": 5},
        bound(9),
        att("attempt_started", 9, S1, attempt=0),
        att("attempt_started", 9, S2, attempt=0),
        att("attempt_succeeded", 9, S1, attempt=0, collected_index=0, reward_total=1.0),
        att("attempt_succeeded", 9, S2, attempt=0, collected_index=1, reward_total=1.0),
        {"kind": "generation_exited", "call": 2, "results": 1},
    ]
    report = integrity([row(S1, version=5, prompt=9), row(S2, version=5, prompt=9)], events,
                       cohort="strict_parity", expected_version=5)
    assert report["status"] == "passed", report["issues"]
    assert report["versions"] == {"rollout-1:9:1": {"version": 5, "reason": "entry_version"}}
    stray = [*events[:-1], {"kind": "generation_exited", "call": 7, "results": 1}]
    report = integrity([row(S1, version=5, prompt=9), row(S2, version=5, prompt=9)], stray,
                       cohort="async_disclosed")
    assert "unmatched_generation_exit" in codes(report, "blocking")


def test_run_identity_hash_is_independent_of_input_order():
    """A8-R-05: same records in another order give the same inputs identity hash."""
    launches = [launch("L1", 0, "start"), launch("L2", 1, "resume")]
    a = check_run_identity(launches, CKPTS, scope="synthetic")
    b = check_run_identity(list(reversed(launches)), list(reversed(CKPTS)), scope="synthetic")
    assert a["status"] == b["status"] == "passed"
    assert a["inputs"] == b["inputs"]


# ---------------------------------------------------------------- trainer steps (F5)
# Hand-written v2 capture events (teacher_rl.capture.entrypoint docstring). Expected E20/E07
# inputs are literal; shape compatibility is proved by running the real consumers on them.

D = {c: c * 64 for c in "0123456789abcdef"}


def tensor_entry(kind, digest, shape=(2,), dtype="F32", requires_grad=None, shard=None):
    scope = None if digest is None else ("local_shard" if shard else "full")
    return {"kind": kind, "dtype": dtype, "shape": list(shape), "numel": math.prod(shape),
            "requires_grad": (kind == "param") if requires_grad is None else requires_grad,
            "sha256": digest, "hash_scope": scope, "shard": shard}


def snapshot(phase, step, *, w=D["a"], state=None, counter=0, frozen=D["e"], buf=D["f"],
             shard=None, hashed=True):
    pick = (lambda d: d) if hashed else (lambda d: None)
    state = {} if state is None else state
    return {"kind": "parameter_snapshot", "schema_version": 2, "phase": phase,
            "trainer_step": step,
            "tensors": {"expert.w": tensor_entry("param", pick(w), shard=shard),
                        "vlm.lm_head.weight": tensor_entry("param", pick(frozen), dtype="BF16",
                                                           requires_grad=False),
                        "action_in_proj.freqs": tensor_entry("buffer", pick(buf))},
            "training_flags": {"": True, "vlm": False},
            "optimizer_state": {"expert.w": {k: pick(v) for k, v in state.items()}},
            "step_counters": {"expert.w": counter}, "applied_lr": [0.0001]}


def bounded(values):
    return {"count": len(values), "values": list(values)}


def minibatch(call, rows=(1, 0), advantages=(0.0, 0.5), pads=(True, False), pool=11):
    return [{"kind": "minibatch_started", "call": call, "rows": len(rows), "pool_call": pool,
             "row_indices": bounded(rows), "advantages": bounded(advantages),
             "old_logprobs": bounded([[-1.0 - p] for p in pads]),
             "is_padding": bounded([[p] for p in pads])},
            {"kind": "replay_forward", "call": call + 1, "minibatch_call": call,
             "log_probs": bounded([-1.5] * len(rows)), "kl_present": False}]


def reduce(call, minibatch_call, grad=0.0, calls=0):
    return {"kind": "optimizer_reduce", "call": call, "minibatch_call": minibatch_call,
            "grad_norm": grad, "optimizer_step_calls": calls, "lr": [0.0001]}


def finished(call):
    return {"kind": "minibatch_finished", "call": call, "metrics": [0.1]}


def configured(mini_batch=2, iterations=1):
    return {"kind": "trainer_configured", "call": 1, "mini_batch": mini_batch,
            "iterations": iterations, "group_size": 2, "allowed_outdated_steps": 0,
            "kl_beta": 0.0}


def scheduler(phase, step, epoch, call=10):
    return {"kind": "scheduler_state", "call": call, "phase": phase, "trainer_step": step,
            "last_epoch": epoch, "lrs": [0.0001]}


def trainer_step(step=5, *, grad=0.0, calls=0, pre=None, post=None, body=None, rows=2,
                 batches=1, end="finished", epochs=(4, 5), base=0):
    """One step's events; ``base`` shifts every call id (the capture's ids never repeat)."""
    pre = snapshot("pre_step", step) if pre is None else pre
    post = snapshot("post_step", step) if post is None else post
    if body is None:
        body = [*minibatch(12 + base, pool=11 + base), reduce(14 + base, 12 + base, grad, calls),
                finished(12 + base)]
    head = [{"kind": "trainer_step_started", "call": 10 + base, "current_step": step,
             "total_steps": 10, "received_rollouts": 1, "do_save_checkpoint": False},
            scheduler("pre_step", step, epochs[0], 10 + base), pre,
            {"kind": "replay_pool_prepared", "call": 11 + base, "rows": rows,
             "rollouts": ["/a.json"], "rows_per_rollout": rows,
             "advantages": bounded([0.5, 0.0])}]
    if end == "finished":
        tail = [{"kind": "trainer_step_finished", "call": 10 + base, "current_step": step,
                 "metrics": {"train/num_batches": batches, "train/loss_avg": 0.25}},
                scheduler("post_step", step, epochs[1], 10 + base), post]
    elif end == "raised":
        tail = [{"kind": "call_raised", "target": "step_training", "call": 10 + base,
                 "error_type": "FloatingPointError", "message": "non-finite"}]
    else:
        tail = []
    return [e for e in head + body + tail if e is not None]


def trainer_records(events, sid="policy-1", **kwargs):
    return records.build_trainer_step_records([stream(sid, events)], scope="synthetic", **kwargs)


def digest(w=D["a"], state=None, counter=0, frozen=D["e"]):
    return {"parameters": {"expert.w": w, "vlm.lm_head.weight": frozen},
            "optimizer_state": {"expert.w": {} if state is None else dict(sorted(state.items()))},
            "step_counters": {"expert.w": counter}}


def test_skipped_step_known_answer_feeds_e20_and_e07():
    built = trainer_records([configured(), *trainer_step()])
    assert built["status"] == "passed", built["issues"]
    assert built["e20_trainer_steps"] == {"policy-1": [{
        "trainer_step": 5, "rows": 2, "mini_batch": 2, "iterations": 1,
        "source": "in_process_capture",
        "minibatches": [{"grad_norm": 0.0, "optimizer_step_called": False}],
        "reported_num_batches": 1}]}
    assert built["e07_trainer_steps"] == [{
        "stream_id": "policy-1", "trainer_step": 5, "hash_comparability": "full", "input": {
            "trainer_step": 5, "scheduler_before": {"last_epoch": 4, "lrs": [0.0001]},
            "scheduler_after": {"last_epoch": 5, "lrs": [0.0001]},
            "minibatches": [{"index": 0, "grad_norm": 0.0, "optimizer_step_called": False,
                             "before": digest(), "after": digest()}]}}]
    e20 = step_checks.reconcile_optimizer_steps(built["e20_trainer_steps"]["policy-1"],
                                                scope="synthetic")
    assert e20["status"] == "passed" and e20["trainer_steps"][0]["counted_optimizer_steps"] == 0
    given = built["e07_trainer_steps"][0]["input"]
    assert step_checks.check_skip_semantics(given, scope="synthetic")["status"] == "passed"
    assert adapters.check_skip(5, scheduler_before=given["scheduler_before"],
                               scheduler_after=given["scheduler_after"],
                               minibatches=given["minibatches"],
                               scope="synthetic")["status"] == "passed"
    (step,) = built["streams"][0]["steps"]
    assert step["minibatches"][0] | {} == {
        "call": 12, "rows": 2, "pool_call": 11, "row_indices": [1, 0],
        "advantages": [0.0, 0.5], "old_logprobs": [[-2.0], [-1.0]],
        "is_padding": [[True], [False]], "new_log_probs": [-1.5, -1.5], "kl_present": False,
        "finished": True, "raised": None, "reduce_status": "returned", "grad_norm": 0.0,
        "optimizer_step_calls": 0, "optimizer_step": "skipped", "applied_lr": [0.0001]}
    assert step["e20"] == {"assembly": "passed", "reasons": []}
    assert step["e07"] == {"assembly": "passed", "reasons": [], "hash_comparability": "full"}
    assert built["records_hash"] == identity_hash({k: built[k] for k in (
        "streams", "e20_trainer_steps", "e07_trainer_steps", "exports", "counts")})


def test_changed_parameter_at_a_skip_is_caught_by_the_e07_consumer():
    built = trainer_records([configured(), *trainer_step(post=snapshot("post_step", 5, w=D["b"]))])
    assert built["status"] == "passed"
    verdict = step_checks.check_skip_semantics(built["e07_trainer_steps"][0]["input"],
                                               scope="synthetic")
    assert verdict["status"] == "blocked"
    assert {i["code"] for i in verdict["issues"]} == {"parameters_changed_at_skip"}
    assert built["streams"][0]["steps"][0]["parameter_changes"]["changed"] == ["expert.w"]


def test_taken_step_advances_counters_and_passes():
    post = snapshot("post_step", 5, w=D["b"], state={"exp_avg": D["c"], "step": D["d"]},
                    counter=1)
    built = trainer_records([configured(), *trainer_step(grad=2.0, calls=1, post=post)])
    given = built["e07_trainer_steps"][0]["input"]
    assert given["minibatches"][0]["after"] == digest(
        w=D["b"], state={"exp_avg": D["c"], "step": D["d"]}, counter=1)
    assert step_checks.check_skip_semantics(given, scope="synthetic")["status"] == "passed"
    assert built["streams"][0]["steps"][0]["minibatches"][0]["optimizer_step"] == "taken"


def test_two_minibatches_give_e20_but_not_e07():
    body = [*minibatch(12, rows=(1,), advantages=(0.0,), pads=(True,)), reduce(14, 12),
            finished(12), *minibatch(20, rows=(0,), advantages=(0.5,), pads=(False,)),
            reduce(22, 20, 3.0, 1), finished(20)]
    built = trainer_records([configured(mini_batch=1), *trainer_step(body=body, batches=2)])
    assert built["status"] == "passed", built["issues"]
    (step,) = built["streams"][0]["steps"]
    assert step["e07"]["assembly"] == "not_estimable"
    assert step["e07"]["reasons"] == ["per_minibatch_snapshots_unavailable"]
    (e20_input,) = built["e20_trainer_steps"]["policy-1"]
    assert e20_input["minibatches"] == [{"grad_norm": 0.0, "optimizer_step_called": False},
                                        {"grad_norm": 3.0, "optimizer_step_called": True}]
    assert step_checks.reconcile_optimizer_steps([e20_input],
                                                 scope="synthetic")["status"] == "passed"


@pytest.mark.parametrize(("change", "e20", "e07", "blocking"), [
    ("no_reduce", ["optimizer_reduce_missing"], ["optimizer_reduce_missing"],
     {"optimizer_reduce_missing"}),
    ("unobserved", ["optimizer_step_unobserved"], ["optimizer_step_unobserved"], set()),
    ("no_config", ["step_geometry_not_captured"], [], set()),
    ("no_pool", ["rows_not_captured"], [], set()),
    ("metadata_only", [], ["snapshot_unhashed"], set()),
    ("no_post", [], ["snapshot_missing"], set()),
    ("no_scheduler", [], ["scheduler_state_missing"], set()),
    ("raised", ["trainer_step_raised"], ["scheduler_state_missing", "snapshot_missing",
                                        "trainer_step_raised"], set()),
])
def test_missing_evidence_is_not_estimable_never_filled(change, e20, e07, blocking):
    events = [configured(), *trainer_step()]
    if change == "no_reduce":
        events = [e for e in events if e["kind"] != "optimizer_reduce"]
    elif change == "unobserved":
        next(e for e in events if e["kind"] == "optimizer_reduce")["optimizer_step_calls"] = None
    elif change == "no_config":
        events = events[1:]
    elif change == "no_pool":
        events = [e for e in events if e["kind"] != "replay_pool_prepared"]
        for e in events:
            e.pop("pool_call", None) if e["kind"] == "minibatch_started" else None
    elif change == "metadata_only":
        events = [configured(), *trainer_step(pre=snapshot("pre_step", 5, hashed=False),
                                              post=snapshot("post_step", 5, hashed=False))]
    elif change == "no_post":
        events = events[:-1]
    elif change == "no_scheduler":
        events = [e for e in events if e["kind"] != "scheduler_state"]
    elif change == "raised":
        events = [configured(), *trainer_step(end="raised")]
    built = trainer_records(events)
    (step,) = built["streams"][0]["steps"]
    assert step["e20"]["reasons"] == e20 and step["e07"]["reasons"] == e07
    assert step["e20"]["assembly"] == ("not_estimable" if e20 else "passed")
    assert step["e07"]["assembly"] == ("not_estimable" if e07 else "passed")
    assert ("policy-1" in built["e20_trainer_steps"]) is (not e20)
    assert bool(built["e07_trainer_steps"]) is (not e07)
    assert codes(built, "blocking") == blocking  # the raised attempt itself is kept, not blocked
    # no assembled input at all (both lists of reasons nonempty) is never a pass
    assert built["status"] == ("blocked" if blocking else "not_estimable" if e20 and e07
                               else "passed")
    assert step["outcome"] == ("raised" if change == "raised" else "finished")


@pytest.mark.parametrize(("mutate", "code"), [
    (lambda ev: ev.__setitem__(-1, {**ev[-1], "schema_version": 1}), "snapshot_invalid"),
    (lambda ev: ev[-1]["tensors"]["expert.w"].update(dtype="torch.float32"), "snapshot_invalid"),
    (lambda ev: ev[-1]["tensors"]["expert.w"].update(kind="parameter"), "snapshot_invalid"),
    (lambda ev: ev[-1].update(call=99), "snapshot_invalid"),
    (lambda ev: ev[-1]["tensors"]["expert.w"].update(numel=3), "snapshot_invalid"),
    (lambda ev: ev[-1]["step_counters"].update({"vlm.lm_head.weight": 0}), "snapshot_invalid"),
    (lambda ev: ev[-1]["tensors"]["expert.w"].update(hash_scope="local_shard"),
     "snapshot_invalid"),
    (lambda ev: ev.__setitem__(-1, {"kind": "event_oversize", "original_kind":
                                    "parameter_snapshot", "bytes": 10**7, "sha256": D["9"]}),
     "snapshot_oversize"),
    (lambda ev: ev.append({"kind": "capture_extract_error", "target": "step_training",
                           "phase": "post_step_snapshot", "error_type": "ValueError",
                           "message": "x"}), "capture_extract_error"),
    (lambda ev: next(e for e in ev if e["kind"] == "optimizer_reduce").update(
        optimizer_step_calls=2), "optimizer_step_called_repeatedly"),
    (lambda ev: next(e for e in ev if e["kind"] == "optimizer_reduce").update(
        minibatch_call=None), "reduce_without_minibatch"),
    (lambda ev: next(e for e in ev if e["kind"] == "minibatch_started")["row_indices"].update(
        values=[1, None]), "row_index_unresolved"),
    (lambda ev: next(e for e in ev if e["kind"] == "minibatch_started")["row_indices"].update(
        values=[1, 1]), "iteration_rows_not_a_permutation"),
    (lambda ev: next(e for e in ev if e["kind"] == "minibatch_started")["row_indices"].update(
        values=[1, 2]), "row_index_out_of_pool"),
    (lambda ev: next(e for e in ev if e["kind"] == "minibatch_started")["advantages"].update(
        values=[0.5, 0.5]), "padding_advantage_nonzero"),
    (lambda ev: next(e for e in ev if e["kind"] == "replay_forward").update(
        log_probs={"count": 3, "values": [1.0, 1.0, 1.0]}), "minibatch_rows_mismatch"),
    (lambda ev: ev.extend(trainer_step()), "trainer_step_repeated"),
    (lambda ev: ev.__delitem__(slice(-3, None)), "trainer_step_unterminated"),
    (lambda ev: ev.insert(2, configured()), "trainer_configured_repeated"),
])
def test_trainer_record_negative_controls_block(mutate, code):
    events = [configured(), *trainer_step()]
    mutate(events)
    built = trainer_records(events)
    assert built["status"] == "blocked"
    assert code in codes(built, "blocking"), built["issues"]


def test_local_shard_snapshots_compare_only_shard_to_shard():
    shard = {"global_shape": [5], "offsets": [3], "placement": "[Shard(dim=0)]"}
    pre = snapshot("pre_step", 5, shard=shard)
    built = trainer_records([configured(), *trainer_step(
        pre=pre, post=snapshot("post_step", 5, shard=shard))])
    assert built["status"] == "passed", built["issues"]
    assert built["e07_trainer_steps"][0]["hash_comparability"] == "shard_to_shard_same_stream"
    moved = {**shard, "offsets": [0]}
    built = trainer_records([configured(), *trainer_step(
        pre=pre, post=snapshot("post_step", 5, shard=moved))])
    (step,) = built["streams"][0]["steps"]
    assert step["e07"] == {"assembly": "blocked", "reasons": ["shard_layout_changed"],
                           "hash_comparability": None}
    assert not built["e07_trainer_steps"]
    with pytest.raises(records.RecordsError) as refused:
        records.param_snapshot_artifact({**pre, "seq": 0}, scope="synthetic")
    assert refused.value.code == "local_shard_not_full_identity"


def test_param_snapshot_artifact_feeds_the_e03_consumers():
    pre = {**snapshot("pre_step", 5), "seq": 3}
    post = {**snapshot("post_step", 5, w=D["b"]), "seq": 9}
    before = records.param_snapshot_artifact(pre, scope="synthetic")
    after = records.param_snapshot_artifact(post, scope="synthetic")
    assert before["tensors"]["expert.w"] == {
        "kind": "parameter", "dtype": "F32", "shape": [2], "numel": 2, "byte_length": 8,
        "sha256": D["a"], "requires_grad": True, "persistent": None}
    assert before["tensors"]["vlm.lm_head.weight"]["byte_length"] == 4  # BF16 x 2
    assert before["tensors"]["action_in_proj.freqs"]["kind"] == "buffer"
    assert before["tensor_set_sha256"] == hashlib.sha256(canonical_json_bytes({
        "action_in_proj.freqs": {"dtype": "F32", "shape": [2], "byte_length": 8,
                                 "sha256": D["f"]},
        "expert.w": {"dtype": "F32", "shape": [2], "byte_length": 8, "sha256": D["a"]},
        "vlm.lm_head.weight": {"dtype": "BF16", "shape": [2], "byte_length": 4,
                               "sha256": D["e"]}})).hexdigest()
    assert tensor_table(before)[1] is None
    change = params.diff(before, after)
    assert change["status"] == "passed" and change["changed"] == ["expert.w"]
    check = params.check_trainable_set(after, ["expert."], ["vlm.", "action_in_proj."],
                                       diff=change)
    assert check["status"] == "passed", check["issues"]
    for bad, code in (({**pre, "schema_version": 1}, "snapshot_invalid"),
                      ({**snapshot("init", None, hashed=False), "seq": 0}, "snapshot_unhashed")):
        with pytest.raises(records.RecordsError) as refused:
            records.param_snapshot_artifact(bad, scope="synthetic")
        assert refused.value.code == code
    with pytest.raises(records.RecordsError):
        records.param_snapshot_artifact(pre, scope="real")


def test_bounded_lists_resolve_only_from_verified_blobs():
    row_bytes = canonical_json_bytes([1, 0])
    row_sha = hashlib.sha256(row_bytes).hexdigest()
    events = [configured(), *trainer_step()]
    started = next(e for e in events if e["kind"] == "minibatch_started")
    started["row_indices"] = {"count": 2, "blob": {"sha256": row_sha, "bytes": len(row_bytes),
                                                   "stored": True, "path": "blobs/x"}}
    built = trainer_records(copy.deepcopy(events), blobs={row_sha: row_bytes})
    assert built["status"] == "passed"
    assert built["streams"][0]["steps"][0]["minibatches"][0]["row_indices"] == [1, 0]
    missing = trainer_records(copy.deepcopy(events))
    assert missing["status"] == "passed" and codes(missing) == {"blob_not_supplied"}
    assert missing["streams"][0]["steps"][0]["minibatches"][0]["row_indices"] is None
    tampered = trainer_records(copy.deepcopy(events), blobs={row_sha: b"[0,1]"})
    assert "blob_hash_mismatch" in codes(tampered, "blocking")
    assert "blobs_invalid" in codes(trainer_records(events, blobs={"x": b""}), "blocking")


def test_checkpoint_exports_keep_pending_until_settled():
    save = {"kind": "checkpoint_save", "call": 30, "phase": "returned", "current_step": 5,
            "export": {"path": "/o/safetensors/step_5", "status": "pending_async_write",
                       "reason": "writer_thread_alive_after_join",
                       "writer_thread": "save_and_upload_safetensors", "files": None}}
    events = [configured(), *trainer_step()]
    events.insert(-3, save)
    pending = trainer_records(copy.deepcopy(events))
    assert codes(pending) == {"export_pending_unsettled"}
    assert pending["exports"][0]["status"] == "pending_async_write"
    assert pending["exports"][0]["manifest_sha256"] is None
    events.append({"kind": "checkpoint_export_settled", "save_call": 30, "current_step": 5,
                   "trigger": "shutdown", "final": True,
                   "export": {"path": "/o/safetensors/step_5", "synchronized": True,
                              "status": "hashed", "manifest_sha256": D["7"], "files": None}})
    settled = trainer_records(events)
    assert settled["status"] == "passed", settled["issues"]
    assert {k: settled["exports"][0][k] for k in ("current_step", "status", "manifest_sha256",
                                                  "synchronized", "save_call")} == {
        "current_step": 5, "status": "hashed", "manifest_sha256": D["7"],
        "synchronized": True, "save_call": 30}
    assert settled["streams"][0]["steps"][0]["export_calls"] == [30]
    events.append({**events[-1]})
    assert "settle_without_pending_export" in codes(trainer_records(events), "blocking")


def test_streams_are_never_merged_and_order_is_irrelevant():
    one = stream("policy-0", [configured(), *trainer_step()])
    two = stream("policy-1", [configured(), *trainer_step(grad=2.0, calls=1, post=snapshot(
        "post_step", 5, w=D["b"], state={"exp_avg": D["c"]}, counter=1))])
    rollout = stream("rollout-1", rollout_events())
    a = records.build_trainer_step_records([one, two, rollout], scope="synthetic")
    b = records.build_trainer_step_records([rollout, two, one], scope="synthetic")
    assert a == b and a["status"] == "passed", a["issues"]
    assert sorted(a["e20_trainer_steps"]) == ["policy-0", "policy-1"]
    assert [s["stream_id"] for s in a["streams"]] == ["policy-0", "policy-1"]
    assert a["counts"] == {"trainer_steps": 2, "minibatches": 2, "e20_assembled": 2,
                           "e07_assembled": 2}


def test_trainer_records_empty_inputs_never_pass():
    assert records.build_trainer_step_records([], scope="synthetic")["status"] == "not_run"
    only_rollouts = records.build_trainer_step_records([stream("rollout-1", rollout_events())],
                                                       scope="synthetic")
    assert only_rollouts["status"] == "not_run"
    assert codes(only_rollouts) == {"no_trainer_steps"}
    assert "unsupported_scope" in codes(records.build_trainer_step_records(
        [], scope="real"), "blocking")
    blocked = records.build_trainer_step_records(
        [stream("policy-1", [configured(), *trainer_step()], reader_status="blocked",
                reader_issues=[{"code": "chain_broken", "path": "p", "severity": "blocking",
                                "evidence_refs": [], "message": "m"}])], scope="synthetic")
    assert blocked["status"] == "blocked" and "chain_broken" in codes(blocked)
    built = trainer_records([configured(), *trainer_step()])
    assert set(built) >= CONTRACT and built["scope"] == "synthetic"


# ---- review F5 regressions: evidence order, call reuse, malformed metrics, row coverage ----

def _kinds(events):
    return [e["kind"] if e["kind"] not in ("parameter_snapshot", "scheduler_state")
            else f"{e['kind']}:{e['phase']}" for e in events]


def _move_after(events, moved, anchor_kind):
    events.remove(moved)
    index = next(i for i, e in enumerate(events) if e["kind"] == anchor_kind)
    events.insert(index + 1, moved)
    return events


@pytest.mark.parametrize("kind", ["parameter_snapshot", "scheduler_state"])
@pytest.mark.parametrize("anchor", ["replay_pool_prepared", "minibatch_finished"])
def test_pre_step_evidence_after_the_update_blocks(kind, anchor):
    # a skip step whose pre snapshot was taken after the parameters moved (repro (c)): the
    # bracket is contradicted, so E07 is blocked, never assembled from the post-update record
    events = [configured(), *trainer_step(pre=snapshot("pre_step", 5, w=D["b"]),
                                          post=snapshot("post_step", 5, w=D["b"]))]
    moved = next(e for e in events if e["kind"] == kind and e["phase"] == "pre_step")
    _move_after(events, moved, anchor)
    assert _kinds(events).index(f"{kind}:pre_step") > _kinds(events).index(anchor)
    built = trainer_records(events)
    assert built["status"] == "blocked"
    assert "trainer_step_evidence_misordered" in codes(built, "blocking")
    (step,) = built["streams"][0]["steps"]
    assert step["e07"]["assembly"] == "blocked"
    assert "trainer_step_evidence_misordered" in step["e07"]["reasons"]
    assert not built["e07_trainer_steps"]


def test_scheduler_state_of_another_call_is_misordered():
    events = [configured(), *trainer_step()]
    next(e for e in events if e["kind"] == "scheduler_state")["call"] = 99
    built = trainer_records(events)
    assert "trainer_step_evidence_misordered" in codes(built, "blocking")
    assert not built["e07_trainer_steps"]


def test_post_step_evidence_inside_the_open_step_is_orphaned():
    events = [configured(), *trainer_step()]
    post = events[-1]
    events.remove(post)
    events.insert(next(i for i, e in enumerate(events) if e["kind"] == "trainer_step_finished"),
                  post)
    built = trainer_records(events)
    assert "orphan_step_event" in codes(built, "blocking")
    assert not built["e07_trainer_steps"]


def test_call_ids_reused_across_steps_block():
    distinct = trainer_records([configured(), *trainer_step(5), *trainer_step(6, base=100)])
    assert distinct["status"] == "passed", distinct["issues"]
    assert [r["trainer_step"] for r in distinct["e20_trainer_steps"]["policy-1"]] == [5, 6]
    assert distinct["counts"]["e07_assembled"] == 2
    reused = trainer_records([configured(), *trainer_step(5), *trainer_step(6)])
    assert reused["status"] == "blocked"
    assert "call_reused" in codes(reused, "blocking")
    # a reduce naming an earlier step's minibatch never pairs across steps
    events = [configured(), *trainer_step(5), *trainer_step(6, base=100)]
    [e for e in events if e["kind"] == "optimizer_reduce"][1]["minibatch_call"] = 12
    assert "reduce_without_minibatch" in codes(trainer_records(events), "blocking")


def test_unreadable_num_batches_is_reported_never_silently_dropped():
    events = [configured(), *trainer_step()]
    next(e for e in events if e["kind"] == "trainer_step_finished")["metrics"][
        "train/num_batches"] = 2.0
    built = trainer_records(events)
    assert codes(built) == {"num_batches_unreadable"}
    assert {i["severity"] for i in built["issues"]} == {"warning"}
    assert built["e20_trainer_steps"]["policy-1"][0]["reported_num_batches"] is None
    absent = [configured(), *trainer_step()]
    next(e for e in absent if e["kind"] == "trainer_step_finished")["metrics"].pop(
        "train/num_batches")
    assert trainer_records(absent)["issues"] == []  # absent is not unreadable


@pytest.mark.parametrize("lr", [{"not": "a list"}, ["nan"], [float("inf")], [True], "0.1"])
def test_malformed_applied_lr_blocks_and_is_not_copied(lr):
    events = [configured(), *trainer_step()]
    next(e for e in events if e["kind"] == "optimizer_reduce")["lr"] = lr
    built = trainer_records(events)
    assert built["status"] == "blocked"
    assert "applied_lr_invalid" in codes(built, "blocking")
    assert built["streams"][0]["steps"][0]["minibatches"][0]["applied_lr"] is None


def test_unreported_applied_lr_is_kept_as_null():
    events = [configured(), *trainer_step()]
    next(e for e in events if e["kind"] == "optimizer_reduce")["lr"] = None
    built = trainer_records(events)
    assert built["status"] == "passed", built["issues"]
    assert built["streams"][0]["steps"][0]["minibatches"][0]["applied_lr"] is None


def test_row_checks_record_what_was_not_run():
    assert trainer_records([configured(), *trainer_step()])["streams"][0]["steps"][0][
        "row_checks"] == {"status": "run", "reasons": []}
    events = [configured(), *trainer_step()]
    next(e for e in events if e["kind"] == "minibatch_started")["row_indices"] = {
        "count": 2, "blob": {"sha256": D["3"], "bytes": 5, "stored": True, "path": "blobs/x"}}
    missing = trainer_records(events)
    assert missing["status"] == "passed" and codes(missing) == {"blob_not_supplied"}
    assert missing["streams"][0]["steps"][0]["row_checks"] == {
        "status": "not_run",
        "reasons": ["iteration_permutation_not_checked", "row_indices_unavailable"]}
    no_pool = [e for e in [configured(), *trainer_step()] if e["kind"] != "replay_pool_prepared"]
    for e in no_pool:
        e.pop("pool_call", None) if e["kind"] == "minibatch_started" else None
    assert trainer_records(no_pool)["streams"][0]["steps"][0]["row_checks"] == {
        "status": "not_run", "reasons": ["iteration_permutation_not_checked", "rows_not_captured"]}
    unfinished = trainer_records([configured(), *trainer_step(end="raised")])
    assert unfinished["streams"][0]["steps"][0]["row_checks"] == {
        "status": "not_run", "reasons": ["iteration_permutation_not_checked"]}


def test_capture_digest_equals_the_a10_adapter_digest():
    # F5-08: the capture event has no param-group membership, so G3 consumes
    # records.snapshot_digest; on the same tensors it equals adapters.skip_snapshot's shape
    event = {**snapshot("post_step", 5, w=D["b"], state={"exp_avg": D["c"]}, counter=1),
             "seq": 4}
    assert records.snapshot_digest(event) == digest(w=D["b"], state={"exp_avg": D["c"]},
                                                    counter=1)
