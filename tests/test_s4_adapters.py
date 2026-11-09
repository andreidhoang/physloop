"""Known answers and negative controls for teacher_rl.adapters (B3 item 6; E07, E16; G3).

Inputs are hand-written artifacts in the exact shapes ``teacher_rl.params`` emits; expected
digests are literal or derived from literal bytes, never from the module under test.
"""

import copy
import hashlib

import pytest

from scripts import rehearse_s4 as rh
from teacher_rl import adapters, pins, records

H = {c: c * 64 for c in "abcdef0123"}
PIN_15 = "36aeb4c5938cbc2eb2aed33b22434773da4ab639"


def tensor(kind, sha, shape=(2,)):
    return {"kind": kind, "dtype": "F32", "shape": list(shape), "numel": 2, "byte_length": 8,
            "sha256": sha, "requires_grad": kind == "parameter",
            "persistent": True if kind == "buffer" else None}


def param_snapshot(w=H["a"], b=H["b"], scope="synthetic"):
    return {"artifact_kind": "s4_param_snapshot", "schema_version": 1, "scope": scope,
            "producer": {}, "input_hashes": {}, "units": {}, "hash_convention": "",
            "status": "passed", "source": "module",
            "tensors": {"expert.w": tensor("parameter", w),
                        "expert.b": tensor("parameter", b),
                        "action_in_proj.freqs": tensor("buffer", H["c"]),
                        "vlm.lm_head.weight": tensor("parameter", H["d"])},
            "modules": {}, "tensor_set_sha256": H["e"], "totals": {}, "issues": []}


def optimizer_snapshot(step=1.0, exp_avg=H["0"], scope="synthetic"):
    state = {"expert.w": {
        "exp_avg": {"dtype": "F32", "shape": [2], "byte_length": 8, "sha256": exp_avg},
        "exp_avg_sq": {"dtype": "F32", "shape": [2], "byte_length": 8, "sha256": H["1"]},
        "step": {"dtype": "F32", "shape": [], "byte_length": 4, "sha256": H["2"],
                 "value": step}}}
    return {"artifact_kind": "s4_optimizer_state_snapshot", "schema_version": 1,
            "scope": scope, "producer": {}, "input_hashes": {}, "units": {},
            "status": "passed", "optimizer_class": "AdamW",
            "param_groups": [{"index": 0, "params": ["expert.w", "expert.b"],
                              "hyperparameters": {"lr": 0.0001}}],
            "state": state, "issues": []}


def test_skip_snapshot_known_answer():
    snap = adapters.skip_snapshot(param_snapshot(), optimizer_snapshot())
    assert snap == {
        # buffers excluded; frozen vlm.lm_head.weight stays (outside the optimizer)
        "parameters": {"expert.b": H["b"], "expert.w": H["a"], "vlm.lm_head.weight": H["d"]},
        # expert.b has no optimizer state yet (dropped by optimizer_state_snapshot): {} and 0
        "optimizer_state": {"expert.b": {},
                            "expert.w": {"exp_avg": H["0"], "exp_avg_sq": H["1"],
                                         "step": H["2"]}},
        "step_counters": {"expert.b": 0, "expert.w": 1},
    }


def test_json_scalar_state_is_hashed_from_canonical_bytes():
    opt = optimizer_snapshot()
    opt["state"]["expert.w"]["step"] = 3
    snap = adapters.skip_snapshot(param_snapshot(), opt)
    assert snap["optimizer_state"]["expert.w"]["step"] == hashlib.sha256(b"3").hexdigest()
    assert snap["step_counters"]["expert.w"] == 3


@pytest.mark.parametrize("mutate, code", [
    (lambda p, o: p.update(status="blocked"), "input_invalid"),
    (lambda p, o: p.update(source="state_dict"), "parameter_kind_unknown"),
    (lambda p, o: o.update(scope="actual"), "scope_mismatch"),
    (lambda p, o: o["state"]["expert.w"]["step"].update(value=1.5), "step_counter_unreadable"),
    (lambda p, o: o["state"]["expert.w"]["step"].update(value=-1.0), "step_counter_unreadable"),
    (lambda p, o: o["state"]["expert.w"]["step"].pop("value"), "step_counter_unreadable"),
    (lambda p, o: o["state"]["expert.w"].update(step=True), "step_counter_unreadable"),
    (lambda p, o: o["state"].update({"ghost": {}}), "optimizer_state_unowned"),
    (lambda p, o: o["param_groups"].append({"index": 1, "params": ["expert.w"]}),
     "optimizer_param_repeated"),
    (lambda p, o: o["state"]["expert.w"]["exp_avg"].pop("sha256"), "state_entry_invalid"),
    (lambda p, o: p["tensors"]["expert.w"].update(kind="state"), "input_invalid"),
])
def test_skip_snapshot_negative_controls(mutate, code):
    p, o = param_snapshot(), optimizer_snapshot()
    mutate(p, o)
    with pytest.raises(adapters.AdapterError) as err:
        adapters.skip_snapshot(p, o)
    assert err.value.code == code


SCHED_BEFORE = {"last_epoch": 4, "lrs": [0.0001]}
SCHED_AFTER = {"last_epoch": 5, "lrs": [0.0001]}


def test_check_skip_passes_for_a_true_skip_then_a_step():
    p0, o0 = param_snapshot(), optimizer_snapshot()
    p1, o1 = param_snapshot(w=H["f"]), optimizer_snapshot(step=2.0, exp_avg=H["3"])
    rows = [
        adapters.minibatch_record(0, grad_norm=0.0, optimizer_step_called=False,
                                  before=(p0, o0), after=(p0, o0)),
        adapters.minibatch_record(1, grad_norm=0.5, optimizer_step_called=True,
                                  before=(p0, o0), after=(p1, o1)),
    ]
    report = adapters.check_skip(7, scheduler_before=SCHED_BEFORE, scheduler_after=SCHED_AFTER,
                                 minibatches=rows, scope="synthetic")
    assert report["status"] == "passed", report["issues"]
    assert [r["expected"] for r in report["minibatches"]] == ["skip", "step"]


def test_check_skip_negative_control_moment_changed_at_skip():
    p0, o0 = param_snapshot(), optimizer_snapshot()
    moved = optimizer_snapshot(exp_avg=H["3"])
    rows = [adapters.minibatch_record(0, grad_norm=0.0, optimizer_step_called=False,
                                      before=(p0, o0), after=(p0, moved))]
    report = adapters.check_skip(7, scheduler_before=SCHED_BEFORE, scheduler_after=SCHED_AFTER,
                                 minibatches=rows, scope="synthetic")
    assert report["status"] == "blocked"
    assert "optimizer_state_changed_at_skip" in {i["code"] for i in report["issues"]}


class _Sched:
    def __init__(self, epoch, lrs):
        self.last_epoch, self._lrs = epoch, lrs

    def get_last_lr(self):
        return self._lrs


def test_scheduler_state_and_refusals():
    assert adapters.scheduler_state(_Sched(3, [1e-4, 0])) == {"last_epoch": 3,
                                                               "lrs": [0.0001, 0.0]}
    for bad in (_Sched(3.0, [1e-4]), _Sched(3, [float("nan")]), object()):
        with pytest.raises(adapters.AdapterError):
            adapters.scheduler_state(bad)


def path(commit):
    return {"path_id": "p", "image_size": [320, 512], "coc": False, "rope_offsets": "x",
            "sampler": {"kind": "sde", "steps": 5}, "control_rate_hz": 5.0,
            "controller": "linear", "model_package_commit": commit,
            "vlm_name_or_path": "nvidia/Cosmos-Reason2-8B", "vlm_revision": "r"}


def test_driver_pin_is_the_alpamayo15_literal_and_wired_into_the_audit():
    assert adapters.driver_commit_pin() == PIN_15
    report = adapters.audit_train_gate_paths(path(PIN_15), path(PIN_15), scope="synthetic")
    assert report["status"] == "passed"
    assert report["input_hashes"]["expected_driver_commit"] is not None
    other = adapters.audit_train_gate_paths(path(PIN_15), path("1" * 40), scope="synthetic")
    assert "driver_commit_not_pinned" in {i["code"] for i in other["issues"]}


def test_driver_pin_refuses_a_divergent_second_home(monkeypatch):
    patched = copy.deepcopy(pins.UPSTREAM_PINS)
    patched["alpamayo1_5"] = {"commit": "2" * 40}
    monkeypatch.setattr(pins, "UPSTREAM_PINS", patched)
    with pytest.raises(adapters.AdapterError) as err:
        adapters.driver_commit_pin()
    assert err.value.code == "driver_pin_divergent"
    patched["alpamayo1_5"] = {"commit": PIN_15}
    assert adapters.driver_commit_pin() == PIN_15


# ---------------------------------------------------------------- G3 capture to checks
# Fake captures go through the real EventSink and read_stream
# (scripts.rehearse_s4.write_synthetic_capture); literals are derived in the comments.

A, B, C = rh.SESSIONS_A


def g3_capture(tmp_path, name="cap", **kwargs):
    return rh.write_synthetic_capture(tmp_path / name, **kwargs)


def g3_codes(report, severity=None):
    return {i["code"] for i in report["issues"] if severity in (None, i["severity"])}


def test_decode_simulate_calls_known_answer(tmp_path):
    streams, blobs = g3_capture(tmp_path)
    report = adapters.decode_simulate_calls(streams, blobs=blobs, scope="synthetic")
    assert report["status"] == "passed", report["issues"]
    assert report["counts"] == {"calls": 3, "decoded": 3, "reply_unavailable": 0,
                                "undecodable": 0, "raised": 0, "rollouts": 3}
    assert [c["call"] for c in report["calls"]] == [301, 302, 303]
    assert sorted(report["rollouts"]) == [A, B, C]
    c = report["rollouts"][C]
    assert c["semantics_refusal"] is None
    sem = c["semantics_input"]
    assert sem["session_uuid"] == C
    # aggregates: every dense metric but eval_relevant, the online maximum (all valid)
    assert sem["aggregated_metrics"] == {"collision_any": 1.0, "collision_front": 1.0,
                                         "collision_lateral": 1.0, "collision_rear": 0.0,
                                         "img_is_black": 0.0, "offroad": 0.0}
    hit = next(m for m in sem["timestep_metrics"] if m["name"] == "collision_any")
    assert hit == {"name": "collision_any", "timestamps_us": [0, 100_000, 200_000],
                   "values": [0.0, 0.0, 1.0], "valid": [True, True, True],
                   "time_aggregation": "max"}
    # the KeyError attempt's reply has no collision series at all
    assert sorted(report["rollouts"][A]["semantics_input"]["aggregated_metrics"]) == [
        "collision_rear", "img_is_black", "offroad"]


def test_unavailable_replies_are_warnings_never_empty_rollouts(tmp_path):
    streams, _ = g3_capture(tmp_path)
    none = adapters.decode_simulate_calls(streams, blobs={}, scope="synthetic")
    assert none["status"] == "not_estimable" and none["rollouts"] == {}
    assert g3_codes(none) == {"reply_blob_not_supplied"}
    off_streams, _ = g3_capture(tmp_path, "off", bytes_captured=False)
    off = adapters.decode_simulate_calls(off_streams, blobs={}, scope="synthetic")
    assert g3_codes(off) == {"reply_bytes_not_captured"}
    assert off["counts"]["reply_unavailable"] == 3


def _event(streams, call):
    return next(e for s in streams for e in s["events"]
                if e["kind"] == "simulate_call" and e["call"] == call)


@pytest.mark.parametrize(("mutate", "code"), [
    (lambda s, b: _event(s, 302)["returns"][0].update(success=False),
     "reply_summary_contradiction"),
    (lambda s, b: _event(s, 302)["returns"][0].update(timestep_metrics=["offroad"]),
     "reply_summary_contradiction"),
    (lambda s, b: _event(s, 302).update(sessions=[C]), "reply_summary_contradiction"),
    (lambda s, b: b.update({_event(s, 302)["reply_blob"]["sha256"]: b"\x00"}),
     "reply_blob_hash_mismatch"),
    (lambda s, b: s[0].update(reader_status="blocked"), "stream_not_verified"),
    (lambda s, b: s[0]["events"].append({**copy.deepcopy(_event(s, 303)), "seq": 999}),
     "session_simulated_twice"),
    (lambda s, b: _event(s, 302).update(reply_blob={"sha256": "zz"}), "reply_blob_invalid"),
])
def test_decode_negative_controls(tmp_path, mutate, code):
    streams, blobs = g3_capture(tmp_path)
    streams, blobs = copy.deepcopy(streams), dict(blobs)
    mutate(streams, blobs)
    report = adapters.decode_simulate_calls(streams, blobs=blobs, scope="synthetic")
    assert report["status"] == "blocked" and code in g3_codes(report, "blocking")


def test_truncated_error_text_matches_only_its_own_prefix_and_length():
    assert adapters._text_matches("ab...[+2]", "abcd")
    assert not adapters._text_matches("ab...[+3]", "abcd")
    assert not adapters._text_matches("ax...[+2]", "abcd")
    assert adapters._text_matches("", "")


def semantics_for(tmp_path, **kwargs):
    streams, blobs = g3_capture(tmp_path)
    decodes = adapters.decode_simulate_calls(streams, blobs=blobs, scope="synthetic")
    ledger = records.build_attempt_ledger(streams, scope="synthetic")
    args = {"required_metrics": ["collision_any"], "deviation_metric": None, **kwargs}
    return adapters.semantics_from_capture(decodes, ledger, scope="synthetic", **args)


def test_semantics_from_capture_known_answer(tmp_path):
    report = semantics_for(tmp_path)
    rows = report["sessions"]
    assert {s: (r["attempt_outcome"], r["e10"]["status"], r["e11"]["status"])
            for s, r in rows.items()} == {A: ("failed_retry", "unverified", "blocked"),
                                          B: ("succeeded", "passed", "passed"),
                                          C: ("succeeded", "passed", "passed")}
    assert rows[A]["e11"]["codes"] == ["collision_metric_missing"]
    assert rows[A]["e10"]["codes"] == ["offline_view_unknown"]
    # c: any-contact max 1 vs additive front + lateral 1 + 1 = 2 (E11)
    assert (rows[C]["e11"]["any_contact"], rows[C]["e11"]["additive_front_lateral"]) == (1.0,
                                                                                         2.0)
    assert rows[B]["e12"]["status"] == rows[B]["e13"]["status"] == "not_run"
    assert report["status"] == "blocked"  # E11 blocked on the KeyError attempt's reply
    e14 = report["e14"]
    assert e14["status"] == "unverified"
    assert e14["retry_causes"] == {"missing_metric": 1}
    counts = e14["indicator_counts"]
    assert counts["missing_metric"] == {"true": 1, "false": 2, "unknown": 0}
    assert counts["black_frame"] == {"true": 0, "false": 3, "unknown": 0}
    assert counts["out_of_envelope"] == {"true": 0, "false": 0, "unknown": 3}
    assert counts["nonfinite_log_density"]["unknown"] == 3
    assert "deviation_convention_unregistered" in g3_codes(report, "warning")


def test_semantics_owner_inputs_pass_through_without_defaults(tmp_path):
    unregistered = semantics_for(tmp_path, required_metrics=None)
    assert unregistered["e14"]["status"] == "blocked"
    assert "required_metrics_unregistered" in {i["code"] for i in unregistered["e14"]["issues"]}
    assert unregistered["owner_inputs"] == {"required_metrics": None, "deviation_metric": None}
    deviation = semantics_for(tmp_path / "dev", deviation_metric="offroad")
    assert deviation["e14"]["indicator_counts"]["out_of_envelope"] == {
        "true": 0, "false": 3, "unknown": 0}  # max offroad 0.0 < 4 m (schema exercise)
    with pytest.raises(ValueError, match="scope"):
        adapters.semantics_from_capture({}, {}, required_metrics=None, deviation_metric=None,
                                        scope="nope")


def test_semantics_refuse_unpassed_inputs(tmp_path):
    report = adapters.semantics_from_capture({"artifact_kind": "x"}, {"status": "blocked"},
                                             required_metrics=["collision_any"],
                                             deviation_metric=None, scope="synthetic")
    assert report["status"] == "blocked"
    assert {"decodes_not_usable", "ledger_not_passed"} <= g3_codes(report)


def step_checks(tmp_path, **kwargs):
    streams, blobs = g3_capture(tmp_path, **kwargs)
    trainer = records.build_trainer_step_records(streams, scope="synthetic", blobs=blobs)
    return trainer, adapters.step_checks_from_records(trainer, scope="synthetic")


def test_step_checks_from_records_known_answer(tmp_path):
    trainer, report = step_checks(tmp_path)
    assert trainer["status"] == "passed"
    assert report["status"] == "passed", report["issues"]
    e20 = report["e20"]["trainer_0"]["trainer_steps"]
    # 2 rows, mini_batch 2, mu 1 -> ceil(2/2) = 1 minibatch; grad 0.5 != 0 -> 1 step
    assert e20 == [{"trainer_step": 0, "source": "in_process_capture",
                    "expected_minibatches": 1, "observed_minibatches": 1,
                    "zero_norm_minibatches": 0, "counted_optimizer_steps": 1,
                    "expected_optimizer_steps": 1}]
    (e07,) = report["e07"]
    assert e07["hash_comparability"] == "full"
    assert e07["report"]["status"] == "passed"
    assert e07["report"]["minibatches"] == [{
        "index": 0, "expected": "step", "optimizer_step_called": True,
        "parameters_changed": True, "optimizer_state_changed": True,
        "step_counters_changed": True}]


def test_metadata_only_snapshots_block_e07_never_pass(tmp_path):
    trainer, report = step_checks(tmp_path, trainer_hashed=False)
    assert report["status"] == "blocked" and "snapshot_unhashed" in g3_codes(report)
    assert report["e07"] == []
    assert report["unassembled"] == [{"stream_id": "trainer_0", "trainer_step": 0,
                                      "check": "e07", "assembly": "not_estimable",
                                      "reasons": ["snapshot_unhashed"]}]
    assert report["e20"]["trainer_0"]["status"] == "passed"  # E20 needs no hashes
    refused = adapters.step_checks_from_records(dict(trainer, status="blocked"),
                                                scope="synthetic")
    assert refused["status"] == "blocked" and "trainer_records_blocked" in g3_codes(refused)


def e03(tmp_path, *, trainable=("expert.",), frozen=("vlm.",), **kwargs):
    streams, _ = g3_capture(tmp_path, **kwargs)
    return adapters.e03_from_capture(
        streams, stream_id="trainer_0", trainer_step=0,
        trainable_prefixes=None if trainable is None else list(trainable),
        frozen_prefixes=None if frozen is None else list(frozen), scope="synthetic")


def test_e03_from_capture_known_answer(tmp_path):
    report = e03(tmp_path)
    assert report["status"] == "passed", report["issues"]
    assert report["before"]["tensors"]["expert.w"]["kind"] == "parameter"  # 'param' -> A10
    assert report["diff"]["changed"] == ["expert.w"]
    trainable = report["trainable_set"]
    assert (trainable["trainable"], trainable["frozen"], trainable["buffers"]) == (
        ["expert.w"], ["vlm.lm_head.weight"], ["action_in_proj.freqs"])
    assert trainable["changed_frozen"] == [] and trainable["changed_buffers"] == []
    assert trainable["explicit"]["vlm.lm_head.weight"]["changed"] is False


def test_e03_negative_controls(tmp_path):
    assert "prefixes_unregistered" in g3_codes(e03(tmp_path / "1", trainable=None))
    unhashed = e03(tmp_path / "2", trainer_hashed=False)
    assert unhashed["status"] == "blocked" and "snapshot_unhashed" in g3_codes(unhashed)
    swapped = e03(tmp_path / "3", trainable=("vlm.",), frozen=("expert.",))
    assert swapped["status"] == "blocked"
    assert {"frozen_parameter_changed", "frozen_parameter_requires_grad"} <= {
        i["code"] for i in swapped["trainable_set"]["issues"]}
    streams, _ = g3_capture(tmp_path / "4")
    missing = adapters.e03_from_capture(streams, stream_id="trainer_0", trainer_step=5,
                                        trainable_prefixes=["expert."],
                                        frozen_prefixes=["vlm."], scope="synthetic")
    assert "snapshot_missing" in g3_codes(missing)


def test_repeatability_run_refuses_a_bad_launch_or_seed(tmp_path):
    streams, blobs = g3_capture(tmp_path)
    bad = adapters.repeatability_run(streams, blobs=blobs,
                                     launch=rh.synthetic_launch("other", "7" * 64),
                                     run_id="e01-a", scope="synthetic")
    assert bad["status"] == "blocked" and "launch_invalid" in g3_codes(bad)
    forged = copy.deepcopy(streams)
    forged[0]["events"].append({"seq": 999, "kind": "session_started", "call": 9,
                                "session_uuid": A, "random_seed": 7})
    conflict = adapters.repeatability_run(forged, blobs=blobs,
                                          launch=rh.synthetic_launch("e01-a", "7" * 64),
                                          run_id="e01-a", scope="synthetic")
    assert "session_seed_conflict" in g3_codes(conflict)


# ---------------------------------------------------------------- G3R residual regressions


@pytest.mark.parametrize("sessions", [[["nested"]], [{"session": A}], [7], [""], "not-a-list"])
@pytest.mark.parametrize("kind", ["simulate_call", "simulate_raised"])
def test_repeatability_run_blocks_malformed_simulate_sessions_never_raises(
        tmp_path, kind, sessions):
    """G3R (1): a non-string session would be a dict key (TypeError for a list or dict)."""
    streams, blobs = g3_capture(tmp_path)
    streams = copy.deepcopy(streams)
    if kind == "simulate_call":
        _event(streams, 302)["sessions"] = sessions
    else:
        streams[0]["events"].append({"seq": 999, "kind": "simulate_raised", "call": 399,
                                     "sessions": sessions, "error_type": "RuntimeError"})
    run = adapters.repeatability_run(streams, blobs=blobs,
                                     launch=rh.synthetic_launch("e01-a", "7" * 64),
                                     run_id="e01-a", scope="synthetic")
    assert run["status"] == "blocked"
    assert "simulate_sessions_invalid" in g3_codes(run, "blocking")


def test_repeatability_run_control_well_formed_sessions_pass(tmp_path):
    streams, blobs = g3_capture(tmp_path)
    run = adapters.repeatability_run(streams, blobs=blobs,
                                     launch=rh.synthetic_launch("e01-a", "7" * 64),
                                     run_id="e01-a", scope="synthetic")
    assert run["status"] == "passed", run["issues"]
    assert "simulate_sessions_invalid" not in g3_codes(run)


class _CodeSink:
    def __init__(self):
        self.codes = []

    def add(self, code, *_args, **_kwargs):
        self.codes.append(code)


def test_g3_06_version_fence_same_version_cohorts_never_misfire():
    """G3R (4): independent re-check of checks.replay._version_check (read-only module).

    Hand-derived: declared d, captured first step s. "current_step" holds when s == d,
    "current_step_minus_1" when s - 1 == d; every stream must share one reading.
    """
    from teacher_rl.checks import replay

    for declared, steps, reading in (
            (0, {"t0": 0, "t1": 0, "t2": 0}, ["current_step"]),
            (0, {"t0": 1, "t1": 1}, ["current_step_minus_1"]),
            (4, {"t0": 4, "t1": 4}, ["current_step"]),
            (4, {"t0": 5, "t1": 5}, ["current_step_minus_1"])):
        sink = _CodeSink()
        report = replay._version_check(declared, steps, sink)
        assert report["status"] == "unverified" and report["consistent_readings"] == reading
        assert sink.codes == ["trainer_weight_version_unverified"]
    # negative controls: contradicted values, a non-count captured step, mixed readings
    for declared, steps in ((0, {"t0": 2}), (3, {"t0": 0, "t1": 0}), (0, {"t0": None}),
                            (0, {"t0": True}), (1, {"t0": -1}), (1, {"t0": 1, "t1": 2})):
        sink = _CodeSink()
        report = replay._version_check(declared, steps, sink)
        assert report["status"] == "contradicted", (declared, steps)
        assert sink.codes == ["trainer_weight_version_contradicted"]
    # no declaration or no captured step: nothing to cross-check (other issues block)
    assert replay._version_check(None, {"t0": 0}, _CodeSink()) is None
    assert replay._version_check(0, {}, _CodeSink()) is None


def test_g3_06_version_fence_end_to_end_two_stream_same_version_cohort(tmp_path):
    """G3R (4): two trainer streams at the same first step pass E04; a moved one blocks."""
    from teacher_rl.checks import replay

    streams, blobs = g3_capture(tmp_path)
    ledger = records.build_attempt_ledger(streams, scope="synthetic")
    trainer = records.build_trainer_step_records(streams, scope="synthetic", blobs=blobs)
    trace = records.check_trace_integrity(ledger, streams, cohort="strict_parity",
                                          expected_version=0)
    first = next(s for s in trainer["streams"] if s["stream_id"] == "trainer_0")
    bound = next(i for i in trace["inputs"] if i.get("artifact_kind") == "s4_capture_stream"
                 and i.get("stream_id") == "trainer_0")
    two, trace2 = copy.deepcopy(trainer), copy.deepcopy(trace)
    two["streams"].append({**copy.deepcopy(first), "stream_id": "trainer_1"})
    trace2["inputs"].append({**bound, "stream_id": "trainer_1"})

    def e04(records_artifact):
        return replay.check_logprob_replay(records_artifact, trace_integrity=trace2,
                                           trainer_weight_version=0,
                                           tolerance=dict(rh.SYNTHETIC_E04_TOLERANCE),
                                           scope="synthetic")

    ok = e04(two)
    assert ok["status"] == "passed", ok["issues"]
    assert ok["trainer_weight_version_check"] == {
        "declared": 0, "captured_first_trainer_step": {"trainer_0": 0, "trainer_1": 0},
        "status": "unverified", "consistent_readings": ["current_step"]}
    moved = copy.deepcopy(two)
    moved["streams"][-1]["steps"][0]["trainer_step"] = 1  # needs current_step - 1: mixed
    bad = e04(moved)
    assert bad["status"] == "blocked"
    assert "trainer_weight_version_contradicted" in {i["code"] for i in bad["issues"]}
