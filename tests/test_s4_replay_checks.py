"""Known answers and negative controls for teacher_rl.checks.replay (G3; E04, E01).

Inputs are fake captures written through the real ``EventSink`` and read back with
``read_stream`` (``scripts.rehearse_s4.write_synthetic_capture``), then the real
``teacher_rl.records`` builders. Expected numbers are derived by hand from IEEE-754 float32:
in [1, 2) the spacing is 2**-23, so -1.25 vs -1.25 + 2**-20 is 2**3 = 8 ULP apart.
"""

import copy
import math

import pytest

from scripts import rehearse_s4 as rh
from teacher_rl import adapters, records
from teacher_rl.checks import replay

TOL = dict(rh.SYNTHETIC_E04_TOLERANCE)  # max_ulp 8 (schema exercise, not an owner value)
DECLARED = {"weights": True, "inputs": True, "noise_seeds": True,
            "rollout_force_determinism": True}


def capture(tmp_path, name="a", **kwargs):
    return rh.write_synthetic_capture(tmp_path / name, **kwargs)


def e04_inputs(tmp_path, **kwargs):
    streams, blobs = capture(tmp_path, **kwargs)
    ledger = records.build_attempt_ledger(streams, scope="synthetic")
    trainer = records.build_trainer_step_records(streams, scope="synthetic", blobs=blobs)
    trace = records.check_trace_integrity(ledger, streams, cohort="strict_parity",
                                          expected_version=0)
    return trainer, trace


def e04(trainer, trace, *, version=0, tolerance=TOL):
    return replay.check_logprob_replay(trainer, trace_integrity=trace,
                                       trainer_weight_version=version, tolerance=tolerance,
                                       scope="synthetic")


def codes(report):
    return {i["code"] for i in report["issues"]}


# ---------------------------------------------------------------- float32 ULP


@pytest.mark.parametrize(("a", "b", "ulp"), [
    (-1.25, -1.25 + 2.0 ** -20, 8),          # spacing 2**-23 in [1, 2)
    (0.0, -0.0, 0),                          # signed zeros are 0 apart
    (1.0, 1.0 + 2.0 ** -23, 1),
    (2.0 ** -149, -(2.0 ** -149), 2),        # smallest subnormals straddling zero
    (0.5, 1.0, 2 ** 23),                     # one binade of float32 mantissas
])
def test_float32_ulp_known_answers(a, b, ulp):
    assert replay.float32_ulp(a, b) == ulp == replay.float32_ulp(b, a)


# ---------------------------------------------------------------- E04


def test_e04_known_answer_on_a_fake_capture(tmp_path):
    trainer, trace = e04_inputs(tmp_path)
    assert trainer["status"] == "passed" and trace["status"] == "passed"
    report = e04(trainer, trace)
    assert report["status"] == "passed", report["issues"]
    assert report["behaviour_version"] == 0
    assert report["verdict"] == "within_registered_tolerance"
    assert report["windows"] == {"trainer_0": {
        "minibatches": 1, "unavailable_minibatches": 0, "ended_by": "first_optimizer_step",
        "first_step_at": {"trainer_step": 0, "minibatch": 0}}}
    assert report["counts"] == {"paired_rows": 2, "padding_rows_excluded": 0,
                                "unavailable_minibatches": 0, "unavailable_rows": 0,
                                "identical": 1, "exceeds": 0}
    row0, row1 = report["rows"]
    assert (row0["verdict"], row0["difference"], row0["ulp"], row0["ratio"]) == (
        "identical", 0.0, 0, 1.0)
    assert row1["old"] == -1.25 and row1["new"] == -1.25 + 2.0 ** -20
    assert row1["difference"] == 2.0 ** -20 == 9.5367431640625e-07
    assert row1["rel_error"] == 2.0 ** -20 / 1.25 and row1["ulp"] == 8
    assert row1["ratio"] == math.exp(2.0 ** -20)
    assert row1["verdict"] == "within_registered_tolerance"
    assert (row1["stream_id"], row1["trainer_step"], row1["minibatch"], row1["row"],
            row1["pool_row"]) == ("trainer_0", 0, 0, 1, 1)
    assert report["max_ulp"] == 8 and report["max_abs_error"] == 2.0 ** -20


@pytest.mark.parametrize(("tolerance", "verdict"), [
    ({**TOL, "max_ulp": 7}, "exceeds"),
    ({**TOL, "max_ulp": None, "atol": 2.0 ** -20}, "within_registered_tolerance"),
    ({**TOL, "max_ulp": None, "atol": 2.0 ** -21}, "exceeds"),
    # rtol bound: 2**-20 <= rtol * 1.25 needs rtol >= 2**-20 / 1.25 = 7.62939453125e-07
    ({**TOL, "max_ulp": None, "rtol": 7.62939453125e-07}, "within_registered_tolerance"),
    ({**TOL, "max_ulp": None, "rtol": 7.6e-07}, "exceeds"),
    # conjunction: atol passes, max_ulp 7 fails -> exceeds
    ({**TOL, "atol": 1.0, "max_ulp": 7}, "exceeds"),
])
def test_e04_tolerance_semantics(tmp_path, tolerance, verdict):
    trainer, trace = e04_inputs(tmp_path)
    report = e04(trainer, trace, tolerance=tolerance)
    assert report["status"] == "passed" and report["verdict"] == verdict


def test_e04_unregistered_tolerance_blocks_but_keeps_the_facts(tmp_path):
    trainer, trace = e04_inputs(tmp_path)
    report = e04(trainer, trace, tolerance=None)
    assert report["status"] == "blocked" and "tolerance_unregistered" in codes(report)
    assert report["verdict"] == "tolerance_unregistered" and report["max_ulp"] == 8
    bad = e04(trainer, trace, tolerance={**TOL, "registration_ref": "experiments/tbd.md#x"})
    assert bad["status"] == "blocked" and "tolerance_invalid" in codes(bad)


@pytest.mark.parametrize(("version", "code"), [
    (None, "trainer_weight_version_unregistered"), (1, "behaviour_version_mismatch"),
    (-1, "trainer_weight_version_invalid"), (True, "trainer_weight_version_invalid"),
])
def test_e04_trainer_weight_version_is_required_and_must_match(tmp_path, version, code):
    trainer, trace = e04_inputs(tmp_path)
    report = e04(trainer, trace, version=version)
    assert report["status"] == "blocked" and code in codes(report)


def test_e04_needs_a_bound_strict_parity_trace(tmp_path):
    trainer, trace = e04_inputs(tmp_path)
    streams, _ = capture(tmp_path, "other")
    ledger = records.build_attempt_ledger(streams, scope="synthetic")
    disclosed = records.check_trace_integrity(ledger, streams, cohort="async_disclosed")
    assert "behaviour_version_unestablished" in codes(e04(trainer, disclosed))
    foreign = copy.deepcopy(trace)
    for item in foreign["inputs"]:
        if item.get("artifact_kind") == "s4_capture_stream":
            item["chain_head"] = "0" * 64
    report = e04(trainer, foreign)
    assert report["status"] == "blocked" and "capture_binding_mismatch" in codes(report)


def test_e04_requires_mu_one(tmp_path):
    trainer, trace = e04_inputs(tmp_path, iterations=2)
    report = e04(trainer, trace)
    assert report["status"] == "blocked" and "mu_not_one" in codes(report)
    assert report["counts"]["paired_rows"] == 0
    missing = copy.deepcopy(trainer)
    missing["streams"][0]["configured"] = None
    assert "mu_not_captured" in codes(e04(missing, trace))


def _second_minibatch(trainer, first_step, second_step):
    out = copy.deepcopy(trainer)
    mbs = out["streams"][0]["steps"][0]["minibatches"]
    mbs[0]["optimizer_step"] = first_step
    mbs.append({**copy.deepcopy(mbs[0]), "call": 99, "optimizer_step": second_step})
    return out


def test_e04_window_stops_after_the_first_taken_step(tmp_path):
    trainer, trace = e04_inputs(tmp_path)
    taken = e04(_second_minibatch(trainer, "taken", "taken"), trace)
    assert taken["counts"]["paired_rows"] == 2  # the second minibatch saw updated weights
    skipped = e04(_second_minibatch(trainer, "skipped", "taken"), trace)
    assert skipped["counts"]["paired_rows"] == 4  # a skip leaves the behaviour weights
    assert skipped["windows"]["trainer_0"]["first_step_at"] == {"trainer_step": 0,
                                                               "minibatch": 1}
    unobserved = e04(_second_minibatch(trainer, None, "taken"), trace)
    assert unobserved["counts"]["paired_rows"] == 2
    assert unobserved["windows"]["trainer_0"]["ended_by"] == "optimizer_step_unobserved"
    assert "optimizer_step_unobserved" in codes(unobserved)


def _mutated_rows(trainer, **lists):
    out = copy.deepcopy(trainer)
    mb = out["streams"][0]["steps"][0]["minibatches"][0]
    mb.update(lists)
    return out


@pytest.mark.parametrize(("lists", "code"), [
    ({"old_logprobs": [[-0.5, 0.0], [-1.25]]}, "old_logprob_shape"),
    ({"new_log_probs": ["nan", -1.25]}, "nonfinite_log_density"),
    ({"new_log_probs": [-0.1, -1.25]}, "value_not_float32"),
    ({"new_log_probs": [None, -1.25]}, "log_density_malformed"),
    ({"is_padding": [[False], ["no"]]}, "padding_shape"),
])
def test_e04_row_negative_controls(tmp_path, lists, code):
    trainer, trace = e04_inputs(tmp_path)
    report = e04(_mutated_rows(trainer, **lists), trace)
    assert report["status"] == "blocked" and code in codes(report)


def test_e04_padding_rows_are_excluded_and_missing_rows_never_pair(tmp_path):
    trainer, trace = e04_inputs(tmp_path)
    padded = e04(_mutated_rows(trainer, is_padding=[[False], [True]]), trace)
    assert padded["counts"]["paired_rows"] == 1 and padded["counts"][
        "padding_rows_excluded"] == 1
    assert padded["verdict"] == "identical"
    # rows that exist but cannot be read are a capture gap: unverified (G3-03), not
    # not_estimable (reserved for a window with no non-padding row at all)
    gone = e04(_mutated_rows(trainer, new_log_probs=None), trace)
    assert gone["status"] == "unverified" and "replay_rows_unavailable" in codes(gone)
    assert gone["verdict"] is None and gone["counts"]["unavailable_rows"] == 2
    all_padding = e04(_mutated_rows(trainer, is_padding=[[True], [True]]), trace)
    assert all_padding["status"] == "not_estimable" and all_padding["verdict"] is None
    blocked = dict(trainer, status="blocked")
    assert "trainer_records_not_usable" in codes(e04(blocked, trace))


# ---------------------------------------------------------------- E01


def runs(tmp_path, *, seeds_b=None):
    streams_a, blobs_a = capture(tmp_path, "a")
    if seeds_b is not None:
        original = rh.SESSION_SEEDS
        rh.SESSION_SEEDS = seeds_b
    try:
        streams_b, blobs_b = capture(tmp_path, "b", sessions=rh.SESSIONS_B, trainer=False)
    finally:
        if seeds_b is not None:
            rh.SESSION_SEEDS = original
    run_a = adapters.repeatability_run(streams_a, blobs=blobs_a,
                                       launch=rh.synthetic_launch("e01-a", "7" * 64),
                                       run_id="e01-a", scope="synthetic")
    run_b = adapters.repeatability_run(streams_b, blobs=blobs_b,
                                       launch=rh.synthetic_launch("e01-b", "7" * 64),
                                       run_id="e01-b", scope="synthetic")
    return run_a, run_b


def e01(run_a, run_b, declaration=DECLARED):
    return replay.check_repeatability(run_a, run_b, declaration=declaration, scope="synthetic")


def test_e01_identical_runs_are_repeatable(tmp_path):
    run_a, run_b = runs(tmp_path)
    assert run_a["status"] == run_b["status"] == "passed"
    # no stream label in the key: real labels differ per run (G3-02)
    key = "clipgt-a309e228-26e1-423e-a44c-cb00aa7378cb|prompt:7|instance:1|attempt:0|start:0"
    assert sorted(run_a["units"]) == [key, key.replace("start:0", "start:1"),
                                      key.replace("attempt:0", "attempt:1")]
    assert run_a["units"][key]["random_seed"] == 101
    assert run_a["units"][key]["stream_id"] == "rollout_0"
    assert run_a["units"][key]["simulator_reply"] == "decoded"
    assert run_a["units"][key]["observed"] == {"outcome": True, "reward_total": True,
                                               "reward_components": True, "simulator": True}
    assert run_a["units"][key]["session_uuid"] != run_b["units"][key]["session_uuid"]
    report = e01(run_a, run_b)
    assert report["status"] == "passed", report["issues"]
    assert report["verdict"] == "repeatable" and report["compared_units"] == 3
    assert report["expectation"] == "bitwise_identical"


def test_e01_output_difference_is_a_fact_under_either_mode(tmp_path):
    run_a, run_b = runs(tmp_path)
    key = min(run_b["units"])
    run_b["units"][key]["simulator"]["aggregated_metrics"]["offroad"] = 0.5
    run_b["units"][key]["reward_total"] = 2.0
    report = e01(run_a, run_b)
    assert report["status"] == "passed" and report["verdict"] == "not_repeatable"
    assert report["differing_units"] == [key]
    detail = next(u for u in report["units_detail"] if u["unit"] == key)
    assert detail["differing_fields"] == ["reward_total", "simulator"]
    unseeded = e01(run_a, run_b, {**DECLARED, "rollout_force_determinism": False})
    assert unseeded["verdict"] == "not_repeatable"
    assert unseeded["expectation"] == "none_declared"


def test_e01_contradicted_seed_declaration_blocks(tmp_path):
    run_a, run_b = runs(tmp_path, seeds_b=(101, 999, 103))
    report = e01(run_a, run_b)
    assert report["status"] == "blocked" and report["verdict"] is None
    assert codes(report) == {"declared_noise_seeds_differ"}
    run_a, run_b = runs(tmp_path / "again")
    run_b["units"][min(run_b["units"])]["random_seed"] = None
    assert codes(e01(run_a, run_b)) == {"noise_seed_uncaptured"}


@pytest.mark.parametrize(("mutate", "code"), [
    (lambda a, b: b["launch"].update(base_weights_hash="e" * 64), "declared_weights_differ"),
    (lambda a, b: b["launch"].update(uv_lock_sha256="e" * 64), "declared_inputs_differ"),
    (lambda a, b: b["units"].pop(min(b["units"])), "unit_sets_differ"),
    (lambda a, b: b.update(run_id="e01-a"), "same_run_twice"),
    (lambda a, b: b.update(status="blocked"), "run_not_usable"),
    (lambda a, b: b.update(scope="actual"), "scope_mismatch"),
    (lambda a, b: b["launch"].update(capture_tree_hash="x"), "launch_identity_missing"),
])
def test_e01_negative_controls(tmp_path, mutate, code):
    run_a, run_b = runs(tmp_path)
    mutate(run_a, run_b)
    report = e01(run_a, run_b)
    assert report["status"] == "blocked" and code in codes(report)
    assert report["verdict"] is None


@pytest.mark.parametrize(("declaration", "code"), [
    ({k: v for k, v in DECLARED.items() if k != "rollout_force_determinism"},
     "declaration_invalid"),
    ({**DECLARED, "weights": False}, "not_declared_identical"),
    ({**DECLARED, "rollout_force_determinism": "yes"}, "declaration_invalid"),
    (None, "declaration_invalid"),
])
def test_e01_declaration_is_required(tmp_path, declaration, code):
    run_a, run_b = runs(tmp_path)
    report = e01(run_a, run_b, declaration)
    assert report["status"] == "blocked" and code in codes(report)


# ---------------------------------------------------------------- review regressions (G3)


def _shifted_reply(original):
    """Same reply, but offroad = 0.5 at the last step (aggregate max 0.5)."""
    from teacher_rl import simreturn

    def reply(session, *, collision):
        message = simreturn.message_class().FromString(original(session, collision=collision))
        ret = message.rollout_returns[0]
        for metric in ret.timestep_metrics:
            if metric.name == "offroad":
                metric.values[2] = 0.5
        ret.aggregated_metrics["offroad"] = 0.5
        return message.SerializeToString(deterministic=True)

    return reply


def _run(streams, blobs, name):
    return adapters.repeatability_run(streams, blobs=blobs,
                                      launch=rh.synthetic_launch(name, "7" * 64),
                                      run_id=name, scope="synthetic")


def test_e01_unobserved_replies_are_never_identical(tmp_path, monkeypatch):
    # G3-01: run b's replies really differ (offroad 0.5 vs 0.0); outcomes and rewards agree
    streams_a, blobs_a = capture(tmp_path, "a", trainer=False)
    monkeypatch.setattr(rh, "synthetic_reply_bytes", _shifted_reply(rh.synthetic_reply_bytes))
    streams_b, blobs_b = capture(tmp_path, "b", sessions=rh.SESSIONS_B, trainer=False)
    seen = e01(_run(streams_a, blobs_a, "e01-a"), _run(streams_b, blobs_b, "e01-b"))
    assert seen["status"] == "passed" and seen["verdict"] == "not_repeatable"
    assert [u["differing_fields"] for u in seen["units_detail"]] == [["simulator"]] * 3
    run_a, run_b = _run(streams_a, None, "e01-a"), _run(streams_b, None, "e01-b")
    assert run_a["status"] == "passed"  # the run is usable; its gaps are recorded facts
    assert {"reply_blob_not_supplied", "simulator_reply_unobserved"} <= codes(run_a)
    assert {u["simulator_reply"] for u in run_a["units"].values()} == {"reply_unavailable"}
    assert {u["observed"]["simulator"] for u in run_a["units"].values()} == {False}
    report = e01(run_a, run_b)
    assert report["status"] == "unverified" and report["verdict"] is None
    assert "outputs_not_compared" in codes(report)
    assert [u["not_compared_fields"] for u in report["units_detail"]] == [["simulator"]] * 3
    assert [u["identical"] for u in report["units_detail"]] == [None] * 3
    # one side unobserved is still not a comparison
    half = e01(_run(streams_a, blobs_a, "e01-a"), run_b)
    assert half["status"] == "unverified" and half["verdict"] is None


def test_e01_uncaptured_reply_bytes_are_never_identical(tmp_path):
    streams_a, blobs_a = capture(tmp_path, "a", trainer=False, bytes_captured=False)
    streams_b, blobs_b = capture(tmp_path, "b", sessions=rh.SESSIONS_B, trainer=False,
                                 bytes_captured=False)
    run_a = _run(streams_a, blobs_a, "e01-a")
    assert "reply_bytes_not_captured" in codes(run_a)
    report = e01(run_a, _run(streams_b, blobs_b, "e01-b"))
    assert report["status"] == "unverified" and report["verdict"] is None


def test_e01_observed_difference_stands_beside_unobserved_fields(tmp_path):
    streams_a, _ = capture(tmp_path, "a", trainer=False)
    streams_b, _ = capture(tmp_path, "b", sessions=rh.SESSIONS_B, trainer=False)
    run_a, run_b = _run(streams_a, None, "e01-a"), _run(streams_b, None, "e01-b")
    key = min(run_b["units"])
    run_b["units"][key]["reward_total"] = 2.0
    report = e01(run_a, run_b)
    assert report["status"] == "unverified" and report["verdict"] == "not_repeatable"
    assert report["differing_units"] == [key]


def test_e01_unit_keys_survive_real_stream_labels(tmp_path):
    # G3-02: entrypoint.py labels streams f"{role}-{pid}-{uuid12}", different on every run
    streams_a, blobs_a = capture(tmp_path, "a", trainer=False,
                                 rollout_label="rollout-4242-000000000001")
    streams_b, blobs_b = capture(tmp_path, "b", sessions=rh.SESSIONS_B, trainer=False,
                                 rollout_label="rollout-4243-000000000002")
    assert streams_a[0]["stream_id"] != streams_b[0]["stream_id"]
    run_a, run_b = _run(streams_a, blobs_a, "e01-a"), _run(streams_b, blobs_b, "e01-b")
    assert sorted(run_a["units"]) == sorted(run_b["units"])
    assert all("rollout-424" not in key for key in run_a["units"])
    report = e01(run_a, run_b)
    assert report["status"] == "passed", report["issues"]
    assert report["verdict"] == "repeatable" and report["compared_units"] == 3


def test_e01_zero_compared_units_is_not_estimable(tmp_path):
    # G3-04
    run_a, run_b = runs(tmp_path)
    run_a["units"], run_b["units"] = {}, {}
    report = e01(run_a, run_b)
    assert report["status"] == "not_estimable" and report["verdict"] is None
    assert report["compared_units"] == 0


def test_e01_unit_without_observation_record_is_not_compared(tmp_path):
    run_a, run_b = runs(tmp_path)
    for unit in run_b["units"].values():
        del unit["observed"]
    report = e01(run_a, run_b)
    assert report["status"] == "unverified" and report["verdict"] is None


def _partial_window(trainer, *, second_new=None):
    """Two minibatches: mb0 skipped (window continues) with unavailable replay, mb1 taken."""
    out = _second_minibatch(trainer, "skipped", "taken")
    mbs = out["streams"][0]["steps"][0]["minibatches"]
    mbs[0]["new_log_probs"] = None  # e.g. its log_probs blob was over budget / not supplied
    if second_new is not None:
        mbs[1]["new_log_probs"] = second_new
    return out


def test_e04_unavailable_rows_in_the_window_are_unverified_never_passed(tmp_path):
    # G3-03
    trainer, trace = e04_inputs(tmp_path)
    report = e04(_partial_window(trainer), trace)
    assert report["status"] == "unverified" and report["verdict"] is None
    assert report["counts"]["paired_rows"] == 2
    assert report["counts"]["unavailable_rows"] == 2
    assert report["counts"]["unavailable_minibatches"] == 1
    assert report["windows"]["trainer_0"]["unavailable_minibatches"] == 1
    # an established exceedance is a fact even when the window is incomplete
    exceeding = e04(_partial_window(trainer, second_new=[-0.5, -1.0]), trace)
    assert exceeding["status"] == "unverified" and exceeding["verdict"] == "exceeds"


def test_e04_declared_trainer_version_is_cross_checked_against_the_captured_step(tmp_path):
    # G3-06: pinned cosmos-rl dispatches step N on weights N-1 (status.py:1346, 1397) while
    # AlpaGym calls current_step and weight_version the same unit (rollout_filter.py:36-38);
    # a declaration outside {step - 1, step} contradicts both readings
    trainer, trace = e04_inputs(tmp_path)
    ok = e04(trainer, trace)
    assert ok["trainer_weight_version_check"] == {
        "declared": 0, "captured_first_trainer_step": {"trainer_0": 0},
        "status": "unverified", "consistent_readings": ["current_step"]}
    assert "trainer_weight_version_unverified" in codes(ok)
    moved = copy.deepcopy(trainer)
    moved["streams"][0]["steps"][0]["trainer_step"] = 5
    report = e04(moved, trace)
    assert report["status"] == "blocked"
    assert "trainer_weight_version_contradicted" in codes(report)
    lagged = copy.deepcopy(trainer)
    lagged["streams"][0]["steps"][0]["trainer_step"] = 1
    lag = e04(lagged, trace)
    assert lag["status"] == "passed"
    assert lag["trainer_weight_version_check"]["consistent_readings"] == ["current_step_minus_1"]


def test_e04_version_readings_must_agree_across_streams():
    # declared 5: stream A at step 5 needs "current_step", stream B at step 6 needs
    # "current_step_minus_1"; no single reading explains both -> contradicted
    checks = []
    report = replay._version_check(5, {"trainer_0": 5, "trainer_1": 6}, _Recorder(checks))
    assert report["status"] == "contradicted" and report["consistent_readings"] == []
    assert checks == ["trainer_weight_version_contradicted"]
    three = replay._version_check(5, {"a": 5, "b": 6, "c": 5}, _Recorder([]))
    assert three["status"] == "contradicted"
    agree = replay._version_check(5, {"a": 6, "b": 6}, _Recorder([]))
    assert agree["status"] == "unverified"
    assert agree["consistent_readings"] == ["current_step_minus_1"]


class _Recorder:
    def __init__(self, sink):
        self.sink = sink

    def add(self, code, *_args, **_kwargs):
        self.sink.append(code)
