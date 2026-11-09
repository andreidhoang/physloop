"""Known answers and negative controls for teacher_rl.evaluate (S4.6 evaluation handoff).

The inference checkpoint of each synthetic run is the converted export of
``scripts.rehearse_s4.synthetic_export_chain`` (the hand-written release/export/inference
shapes of tests/test_s4_export_audit.py; the expert weight differs per run). The literal
``tiny.safetensors`` of tests/test_tensor_manifest.py (tensor "a" F32 [1.0, -2.0], "b" U8
scalar 7; per-tensor hashes computed there with ``shasum -a 256``) is the negative control
for the export-audit binding. Registrations and registries are synthetic schema exercises,
never owner choices.
"""

import copy
import struct

import pytest

from gate import superiority
from gate.s2_gate import REGISTRATION_FIELDS, evaluate_registered
from gate.s2_identity import identity_hash
from gate.tensor_manifest import build_manifest
from scripts.rehearse_s4 import (
    SYNTHETIC_GUARD,
    _s2_guard_registration,
    synthetic_export_chain,
    synthetic_records,
    synthetic_registration,
    synthetic_run_registry,
)
from teacher_rl import evaluate

_HEADER = (b'{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},'
           b'"b":{"dtype":"U8","shape":[],"data_offsets":[8,9]}}')
TINY = struct.pack("<Q", 105) + _HEADER + bytes.fromhex("0000803f000000c0") + b"\x07"
TINY_SHA = {"a": "ee4ac73c2bd27756ab82780f27c73a7bc4d3f0bb6acb37e008bc27eccd7e588b",
            "b": "ca358758f6d27e6cf45272937977a748fd88391db679ceda7dc7bf1f005ee879"}
RUNS = ["run-a", "run-b", "run-c"]
METRIC = {"name": "collision_any", "kind": "binary", "higher_is_worse": True,
          "unit": "indicator"}


GUARD_METRIC = dict(SYNTHETIC_GUARD["metric"])
GUARD = SYNTHETIC_GUARD["name"]


def s4_registration(decision_rule=None, **aggregation):
    reg = synthetic_registration()
    # the F4 superiority slots; synthetic stand-ins, not owner choices
    reg["owner"]["improvement_decision_rule"] = {
        "rule_id": superiority.RULE_ID, "registration_ref": "synthetic-ref",
        "guards": [dict(SYNTHETIC_GUARD)], "guard_strata": "primary_strata",
        "gating_strata": "all_guard_strata", "look": {"kind": "fixed"},
        **(decision_rule or {})}
    # the owner slot (B3 re-verification); synthetic values, not owner choices
    reg["owner"]["run_aggregation"] = {"rule": "all", "k": None, "family_alpha": 0.05,
                                       **aggregation}
    return reg


def registry(reg=None, **aggregation):
    return synthetic_run_registry(reg or s4_registration(), **aggregation)


def s2_registration(candidate):
    return {
        "schema_version": 1, "registration_id": "synthetic-s2-eval",
        "reference": "teacher:nvidia/Alpamayo-1.5-10B@7aba8293c09993f2e125c6819df05d7fa3e873ea",
        "candidate": candidate,
        "protocol": {"identity": "synthetic-protocol", "config_hash": "c" * 64,
                     "evaluator": "gate.s2_gate"},
        "seed_mode": "synthetic_seeded", "score_definition": "synthetic score",
        "deviation_convention": "synthetic convention",
        "estimand": REGISTRATION_FIELDS["estimand"], "required_strata": ["urban"],
        "margin": 0.1, "alpha": 0.05, "metric": dict(METRIC),
        "analysis_mode": "fixed_sample",
    }


def label(m):
    return "tensor_set_sha256:" + m["tensor_set_sha256"]


def request_kwargs(run_id="run-a", variant=0, step=3):
    audit_inputs, m = synthetic_export_chain(variant)
    return {"inference_manifest": m, "s4_registration": s4_registration(),
            "run_registry": registry(), "run_id": run_id, "checkpoint_step": step,
            "s2_registration": s2_registration(label(m)), "scope": "synthetic",
            "export_audit_inputs": audit_inputs}


def request(run_id="run-a", variant=0, step=3, **overrides):
    kwargs = request_kwargs(run_id, variant, step)
    kwargs.update(overrides)
    return evaluate.build_evaluation_request(**kwargs)


def test_request_binds_the_tensor_identity_the_export_audit_and_the_s2_protocol():
    kw = request_kwargs()
    req = evaluate.build_evaluation_request(**kw)
    assert req["status"] == "passed", req["issues"]
    m = kw["inference_manifest"]
    assert req["candidate"]["label"] == label(m)
    assert req["candidate"]["tensor_set_sha256"] == m["tensor_set_sha256"]
    assert req["candidate"]["tensor_count"] == 6  # six exported parameters, renamed
    assert req["run_id"] == "run-a" and req["checkpoint_step"] == 3
    assert req["gate_entry"] == "gate.s2_gate.evaluate_registered"
    assert req["s2_registration_id"] == "synthetic-s2-eval"
    assert req["s2_registration"] == kw["s2_registration"]
    assert req["run_registry_hash"] == identity_hash(kw["run_registry"])
    assert req["export_audit"]["input_hashes"]["inference"] == identity_hash(m)
    assert req["execution_status"] == "qualify_on_host"
    assert req["request_hash"] == request()["request_hash"]


def test_exported_steps_follow_trainer_steps_save_freq_and_export_safetensors():
    reg = s4_registration()  # trainer_steps 3, save_freq 1, export_safetensors true
    assert evaluate.exported_steps(reg) == [1, 2, 3]
    reg["owner"]["checkpoint_retention"]["save_freq"] = 2
    assert evaluate.exported_steps(reg) == [2, 3]
    reg["owner"]["checkpoint_retention"]["export_safetensors"] = False
    assert evaluate.exported_steps(reg) == [3]  # the final step always exports


@pytest.mark.parametrize("change, code", [
    (lambda kw: kw.update(run_id="run-z"), "run_id_unregistered"),
    (lambda kw: kw.update(checkpoint_step=-1), "invalid_checkpoint_step"),
    # review B3-11: steps the registered run never exports
    (lambda kw: kw.update(checkpoint_step=10**9), "checkpoint_step_not_exported"),
    (lambda kw: kw.update(checkpoint_step=4), "checkpoint_step_not_exported"),
    (lambda kw: kw.update(checkpoint_step=0), "checkpoint_step_not_exported"),
    (lambda kw: kw.update(s2_registration=s2_registration("tensor_set_sha256:" + "0" * 64)),
     "candidate_identity_mismatch"),
    (lambda kw: kw["s2_registration"].pop("margin"), "s2_registration_invalid"),
    (lambda kw: kw["inference_manifest"].update(tensor_set_sha256="0" * 64),
     "manifest_invalid"),
    (lambda kw: kw.update(inference_manifest=build_manifest({"x.safetensors": b"\x00" * 8},
                                                            scope="synthetic")),
     "manifest_invalid"),
    (lambda kw: kw.update(run_registry=dict(registry(), run_ids=RUNS[:2])),
     "run_registry_invalid"),
    (lambda kw: kw["run_registry"].update(registration_hash="0" * 64),
     "run_registry_invalid"),
    (lambda kw: kw["s4_registration"]["owner"].pop("group_size"), "s4_registration_blocked"),
    (lambda kw: kw.update(scope="actual"), "scope_mismatch"),
    # review B3-11: the run's export audit must pass for exactly this inference manifest
    (lambda kw: kw.update(export_audit_inputs=None), "export_audit_missing"),
    (lambda kw: kw.update(export_audit_inputs=synthetic_export_chain(1)[0]),
     "export_audit_blocked"),
    (lambda kw: kw["export_audit_inputs"].update(release=None), "export_audit_blocked"),
])
def test_request_negative_controls(change, code):
    kw = request_kwargs()
    change(kw)
    req = evaluate.build_evaluation_request(**kw)
    assert req["status"] == "blocked"
    assert code in {i["code"] for i in req["issues"]}, req["issues"]
    assert req["request_hash"] is None


def test_a_checkpoint_outside_the_audited_export_chain_is_refused():
    m = build_manifest({"model.safetensors": TINY}, scope="synthetic")
    assert {n: e["sha256"] for n, e in m["tensors"].items()} == TINY_SHA
    kw = request_kwargs()
    kw.update(inference_manifest=m, s2_registration=s2_registration(label(m)))
    req = evaluate.build_evaluation_request(**kw)
    assert "export_audit_blocked" in {i["code"] for i in req["issues"]}


def requests():
    return [request(run_id=r, variant=v) for r, v in zip(RUNS, (0, 1, 2))]


def spec_for(reqs=None, **kw):
    args = {"requests": requests() if reqs is None else reqs,
            "s4_registration": s4_registration(), "run_registry": registry(),
            "scope": "synthetic"}
    args.update(kw)
    return evaluate.aggregation_spec(**args)


def codes(spec):
    return {i["code"] for i in spec["issues"]}


def test_aggregation_spec_reads_the_registered_runs_rule_and_s2_values():
    spec = spec_for()
    assert spec["status"] == "passed", spec["issues"]
    assert spec["registered_run_ids"] == RUNS
    assert spec["rule"] == "all" and spec["k"] is None and spec["family_alpha"] == 0.05
    assert spec["delta_min"] == 0.03  # smallest_improvement_of_interest.value (synthetic)
    assert spec["rule_id"] == superiority.RULE_ID
    assert spec["s2_registered"] == {"registration_id": "synthetic-s2-eval",
                                     "required_strata": ["urban"], "metric": METRIC,
                                     "margin": 0.1, "alpha": 0.05,
                                     "analysis_mode": "fixed_sample"}
    assert spec["run_registry_hash"] == identity_hash(registry())
    # F4: the registered superiority slots and their resolution against the S2 strata
    assert spec["decision_rule"] == {
        "guards": [SYNTHETIC_GUARD], "guard_strata": "primary_strata",
        "gating_strata": "all_guard_strata", "look": {"kind": "fixed"},
        "resolved_guard_strata": ["urban"], "resolved_gating_strata": ["urban"]}
    reqs = requests()
    assert spec["run_identities"] == {
        r["run_id"]: {"candidate": r["candidate"]["label"], "reference": r["reference"],
                      "s2_registration_hash": r["s2_registration_hash"]} for r in reqs}


@pytest.mark.parametrize(("rule", "code"), [
    ({"look": {"kind": "sequential", "look_index": 2}}, "look_not_evaluable"),
    ({"gating_strata": ["highway"]}, "gating_strata_not_subset"),
])
def test_spec_refuses_a_rule_the_glue_cannot_evaluate(rule, code):
    reg = s4_registration(rule)
    rr = registry(reg)
    reqs = [request(run_id=r, variant=v, s4_registration=reg, run_registry=rr)
            for r, v in zip(RUNS, (0, 1, 2))]
    assert all(r["status"] == "passed" for r in reqs)  # the registration itself is valid
    spec = spec_for(reqs, s4_registration=reg, run_registry=rr)
    assert spec["status"] == "blocked" and code in codes(spec)
    assert spec["decision_rule"] is None and spec["spec_hash"] is None


@pytest.mark.parametrize("aggregation, code", [
    ({"rule": None}, "aggregation_rule_unregistered"),
    ({"rule": "any"}, "aggregation_rule_unregistered"),
    ({"rule": "at_least_k", "k": None}, "aggregation_k_invalid"),
    ({"rule": "at_least_k", "k": 4}, "aggregation_k_invalid"),
    ({"rule": "all", "k": 2}, "aggregation_k_invalid"),
    ({"family_alpha": None}, "family_alpha_invalid"),
    ({"family_alpha": 1.0}, "family_alpha_invalid"),
])
def test_registry_aggregation_inputs_have_no_defaults(aggregation, code):
    bad = registry(**aggregation)
    spec = spec_for(run_registry=bad)
    assert spec["status"] == "blocked"
    assert code in codes(spec)
    req = request(run_registry=bad)
    assert req["status"] == "blocked" and code in codes(req)


def test_review_b3_20_rule_is_fixed_by_the_registry_the_requests_were_built_under():
    v1 = registry()
    v1["schema_version"] = 1
    del v1["aggregation"]
    assert "run_registry_invalid" in codes(spec_for(run_registry=v1))
    # The same requests cannot be re-aggregated under a more favourable rule: a registry
    # with another rule is another registry, and every request names its registry hash.
    favourable = registry(rule="at_least_k", k=1)
    spec = spec_for(run_registry=favourable)
    assert spec["status"] == "blocked"
    assert codes(spec) == {"aggregation_mismatch", "request_registry_mismatch"}


def test_aggregation_inputs_are_read_from_the_registration_owner_slot():
    reg = s4_registration(rule="at_least_k", k=2, family_alpha=0.01)
    rr = registry(reg, rule="at_least_k", k=2, family_alpha=0.01)
    reqs = [request(run_id=r, variant=v, s4_registration=reg, run_registry=rr)
            for r, v in zip(RUNS, (0, 1, 2))]
    assert all(r["status"] == "passed" for r in reqs), [r["issues"] for r in reqs]
    spec = spec_for(reqs, s4_registration=reg, run_registry=rr)
    assert spec["status"] == "passed", spec["issues"]
    assert (spec["rule"], spec["k"], spec["family_alpha"]) == ("at_least_k", 2, 0.01)
    # a registry whose copy differs from the registered slot is refused, never preferred
    stale = registry(reg)  # rule all, alpha 0.05
    assert "aggregation_mismatch" in codes(spec_for(reqs, s4_registration=reg,
                                                    run_registry=stale))
    assert "aggregation_mismatch" in codes(request(s4_registration=reg, run_registry=stale))
    # no slot, no default: the registration itself is blocked
    bare = s4_registration()
    del bare["owner"]["run_aggregation"]
    assert "s4_registration_blocked" in codes(spec_for(s4_registration=bare,
                                                       run_registry=registry(bare)))


def test_aggregation_spec_negative_controls():
    reqs = requests()
    assert "run_missing_request" in codes(spec_for(reqs[:2]))
    dup = reqs[:2] + [copy.deepcopy(reqs[1])]
    assert "run_duplicated" in codes(spec_for(dup))
    same = [request(run_id=r) for r in RUNS]  # one checkpoint claimed by three runs
    assert "candidate_reused" in codes(spec_for(same))
    other_rule = s4_registration()
    other_rule["owner"]["improvement_decision_rule"]["rule_id"] = "synthetic-rule"
    spec = spec_for(reqs, s4_registration=other_rule, run_registry=registry(other_rule))
    assert "decision_rule_mismatch" in codes(spec)
    kw = request_kwargs("run-c", 2)
    kw["s2_registration"]["margin"] = 0.2
    drifted = evaluate.build_evaluation_request(**kw)
    assert "protocol_mismatch" in codes(spec_for([*reqs[:2], drifted]))
    blocked = dict(reqs[0], status="blocked")
    assert "request_not_passed" in codes(spec_for([blocked, *reqs[1:]]))


# --- review B3-03: requests are recomputed, never trusted from their status ------------------


def test_edited_request_with_a_stale_hash_is_refused():
    reqs = requests()
    forged = copy.deepcopy(reqs[2])
    forged["candidate"]["tensor_set_sha256"] = "f" * 64
    forged["checkpoint_step"] = 999
    spec = spec_for([reqs[0], reqs[1], forged])
    assert spec["status"] == "blocked"
    assert {(i["code"], i["path"]) for i in spec["issues"]} >= {
        ("request_invalid", "requests[2]")}


def test_self_consistent_forgery_with_a_bad_s2_binding_is_refused():
    reqs = requests()
    forged = copy.deepcopy(reqs[2])
    forged["s2_registration"]["margin"] = 0.5  # body edited, hash recomputed by the forger
    body = {key: forged[key] for key in evaluate.REQUEST_BODY_KEYS}
    forged["request_hash"] = identity_hash(body)
    spec = spec_for([reqs[0], reqs[1], forged])
    assert ("request_invalid", "requests[2]") in {(i["code"], i["path"])
                                                  for i in spec["issues"]}


def test_hand_written_request_dict_is_refused_not_a_key_error():
    reqs = requests()
    minimal = {"artifact_kind": "s4_evaluation_request", "status": "passed", "run_id": "run-c",
               "s4_registration_hash": reqs[0]["s4_registration_hash"],
               "s2_protocol_hash": reqs[0]["s2_protocol_hash"],
               "candidate": {"tensor_set_sha256": "e" * 64}, "request_hash": "not-a-hash"}
    spec = spec_for([reqs[0], reqs[1], minimal])
    assert spec["status"] == "blocked" and spec["request_hashes"] is None
    assert "requests[2]" in {i["path"] for i in spec["issues"]}


def test_request_from_another_scope_is_refused():
    reqs = requests()
    other = copy.deepcopy(reqs[2])
    other["scope"] = "actual"
    assert "scope_mismatch" in codes(spec_for([reqs[0], reqs[1], other]))


# --- aggregate ------------------------------------------------------------------------------


def gate_response(lower, upper, state, strata="urban", metric=METRIC, alpha=0.05, margin=0.1,
                  roads=30):
    """The gate.api response shape (what gate.s2_gate returns in ``gate_response``)."""
    names = [strata] if type(strata) is str else list(strata)
    share = roads // len(names)
    verdict = {"overall": state, "strata": {name: state for name in names},
               "details": {name: {"lower": lower, "upper": upper, "n_roads": share,
                                  "n_scenes": share, "n_replicates": share} for name in names},
               "alpha": alpha, "method": superiority.FIXED_METHOD, "margin": margin}
    return {"schema_version": 1, "analysis_mode": "fixed_sample", "verdict": verdict,
            "comparison": {"metric": metric}}


def guard_response(state="GO", **kw):
    args = {"metric": GUARD_METRIC, "alpha": 0.05, "margin": 0.1, **kw}
    return gate_response(-0.05, 0.05, state, **args)


def decision(lower, upper, state="GO", delta_min=0.03, strata="urban", guards=None, **kw):
    """decide() as the registered rule requires (one synthetic guard on the primary strata)."""
    guards = {GUARD: guard_response(strata=strata)} if guards is None else guards
    return superiority.decide(gate_response(lower, upper, state, strata, **kw),
                              required_strata=[strata], guards=guards, delta_min=delta_min)


def bound(spec, make):
    return {r: {"request_hash": spec["request_hashes"][r], "decision": make()} for r in RUNS}


def test_aggregate_calls_the_superiority_rule_with_registered_inputs():
    spec = spec_for()
    decisions = bound(spec, lambda: decision(-0.20, -0.05))
    result = evaluate.aggregate(decisions, spec)
    assert result["status"] == "passed"
    assert result["aggregate"]["outcome"] == superiority.SUPPORTED
    decisions["run-b"]["decision"] = decision(-0.10, 0.05, "GO")
    assert evaluate.aggregate(decisions, spec)["aggregate"]["outcome"] != superiority.SUPPORTED
    with pytest.raises(ValueError):
        evaluate.aggregate({r: decisions[r] for r in RUNS[:2]}, spec)
    with pytest.raises(ValueError):
        evaluate.aggregate(decisions, dict(spec, status="blocked"))


def test_review_b3_02_registered_delta_min_known_answer():
    spec = spec_for()
    out = evaluate.aggregate(bound(spec, lambda: decision(-0.02, -0.005)), spec)
    # upper -0.005 excludes 0; lower -0.02 > -0.03 rules out a 0.03 improvement
    assert out["aggregate"]["outcome"] == superiority.IMPROVED_BELOW_MIN


@pytest.mark.parametrize("make, message", [
    (lambda: decision(-0.02, -0.005, delta_min=0.01), "delta_min"),
    (lambda: decision(-0.20, -0.05, strata="highway"), "strata"),
    (lambda: decision(-0.20, -0.05, metric=None), "metric"),
    (lambda: decision(-0.20, -0.05, metric=dict(METRIC, name="hard_brake")), "metric"),
    # 0.01, not 0.1: aggregate_runs itself refuses a level above family_alpha 0.05, so only
    # a level below the registered alpha exercises the binding
    (lambda: decision(-0.20, -0.05, alpha=0.01), "alpha"),
    (lambda: superiority.decide(gate_response(-0.20, -0.05, "GO")["verdict"],
                                required_strata=["urban"],
                                guards={GUARD: guard_response()}, delta_min=0.03),
     "metric"),  # a bare verdict carries no metric
    # F4 (completeness audit "S4.6 superiority binding"): guards, strata and look are bound
    (lambda: decision(-0.20, -0.05, guards={}), "guards"),  # decide(guards={}) is refused
    (lambda: decision(-0.20, -0.05, guards={"other_guard": guard_response()}), "guards"),
    (lambda: decision(-0.20, -0.05, guards={GUARD: guard_response(margin=0.2)}), "margin"),
    (lambda: decision(-0.20, -0.05, guards={GUARD: guard_response(alpha=0.01)}), "alpha"),
    (lambda: decision(-0.20, -0.05, guards={GUARD: guard_response(
        metric=dict(GUARD_METRIC, higher_is_worse=False))}), "metric"),
    (lambda: superiority.decide(gate_response(-0.20, -0.05, "GO"), required_strata=["urban"],
                                guards={GUARD: guard_response(strata="highway")},
                                guard_strata=["highway"], delta_min=0.03), "guard_strata"),
])
def test_review_b3_02_decisions_off_the_registration_raise(make, message):
    """Every run decided off the registration, uniformly (review B3-02 re-verification).

    A uniform deviation is internally consistent, so ``gate.superiority.aggregate_runs``
    accepts it; only ``_bound_decisions`` can refuse it (a pass-through mutant fails every
    case). Limitation: the primary margin cannot be checked on the decision because
    ``superiority.decide`` does not return it; it is bound through the request's S2
    registration instead.
    """
    spec = spec_for()
    with pytest.raises(ValueError, match=message):
        evaluate.aggregate(bound(spec, make), spec)


def _sequential_guard_decision():
    d = decision(-0.20, -0.05)
    d["guards"][GUARD]["method"] = superiority.SEQUENTIAL_METHOD
    return d


@pytest.mark.parametrize("make, message", [
    # review F4-05: a guard decided by another gate method than the registered fixed analysis
    (_sequential_guard_decision, "gate method"),
    # review F4-05: a fixed-look registration refuses terminal=True even with look None
    (lambda: {**decision(-0.20, -0.05), "terminal": True}, "terminal"),
])
def test_review_f4_05_guard_method_and_fixed_look_terminal_are_bound(make, message):
    spec = spec_for()
    assert make()["look"] is None
    with pytest.raises(ValueError, match=message):
        evaluate.aggregate(bound(spec, make), spec)


def test_review_b3_02_decisions_are_bound_to_their_request_hashes():
    spec = spec_for()
    decisions = bound(spec, lambda: decision(-0.20, -0.05))
    decisions["run-a"]["request_hash"] = spec["request_hashes"]["run-b"]
    with pytest.raises(ValueError, match="not bound"):
        evaluate.aggregate(decisions, spec)
    with pytest.raises(ValueError):
        evaluate.aggregate({r: decision(-0.20, -0.05) for r in RUNS}, spec)  # unbound


def test_review_f4_gating_strata_and_look_are_bound():
    """A refined guard registration: guards on urban/highway, only urban gating."""
    reg = s4_registration({"guard_strata": ["highway", "urban"], "gating_strata": ["urban"]})
    rr = registry(reg)
    reqs = [request(run_id=r, variant=v, s4_registration=reg, run_registry=rr)
            for r, v in zip(RUNS, (0, 1, 2))]
    spec = spec_for(reqs, s4_registration=reg, run_registry=rr)
    assert spec["status"] == "passed", spec["issues"]
    assert spec["decision_rule"]["resolved_gating_strata"] == ["urban"]
    refined = {GUARD: guard_response(strata=("highway", "urban"))}

    def make(gating):
        return superiority.decide(gate_response(-0.20, -0.05, "GO"), required_strata=["urban"],
                                  guards=refined, guard_strata=["highway", "urban"],
                                  gating_strata=gating, delta_min=0.03)

    out = evaluate.aggregate(bound(spec, lambda: make(["urban"])), spec)
    assert out["aggregate"]["outcome"] == superiority.SUPPORTED
    with pytest.raises(ValueError, match="gating_strata"):
        evaluate.aggregate(bound(spec, lambda: make(None)), spec)


def test_review_f4_decision_at_a_look_is_refused_under_a_fixed_registration():
    spec = spec_for()

    def make():
        d = decision(-0.20, -0.05)
        return {**d, "look": 1, "terminal": True}

    with pytest.raises(ValueError, match="look"):
        evaluate.aggregate(bound(spec, make), spec)


# --- F4 glue: superiority.decide on gate.s2_gate.evaluate_registered output -----------------
#
# Records (synthetic, binary, higher is worse) on 40 urban roads; the protected bounds give:
# candidate never fails, reference always fails -> road differences -1, bounds (-1, -0.824):
# improved, and lower -1 <= -delta_min 0.03 (not ruled out) -> IMPROVEMENT_SUPPORTED. A guard
# with identical outputs has upper zero_difference_upper(40, 0.05) = 0.0881 < margin 0.1 -> GO.


ALL_ROADS = range(40)


def evaluations(spec, *, guard_candidate_fails=(), primary=None, guard=None):
    out = {}
    for run in RUNS:
        identity = spec["run_identities"][run]
        s2 = s2_registration(identity["candidate"])
        g2 = _s2_guard_registration(identity["candidate"])
        primary_records = synthetic_records(reference_fail_roads=ALL_ROADS)
        guard_records = synthetic_records(candidate_fail_roads=guard_candidate_fails)
        out[run] = {
            "primary": primary or evaluate_registered(primary_records, s2, scope="synthetic",
                                                      adapter_metric=s2["metric"]),
            "guards": {GUARD: guard or evaluate_registered(
                guard_records, g2, scope="synthetic", adapter_metric=g2["metric"])},
            "guard_registrations": {GUARD: g2},
        }
    return out


def test_glue_decides_each_run_from_real_gate_output_and_aggregates():
    spec = spec_for()
    result = evaluate.decide_and_aggregate(evaluations(spec), spec)
    assert result["status"] == "passed", result["issues"]
    assert result["outcomes"] == dict.fromkeys(RUNS, superiority.SUPPORTED)
    assert result["aggregate"]["outcome"] == superiority.SUPPORTED
    assert result["aggregate"]["run_rule"] == "all" and result["aggregate"]["m"] == 3
    d = result["decisions"]["run-a"]
    assert d["strata"]["urban"]["upper"] == pytest.approx(-0.8238053942403952)
    assert d["guards"][GUARD]["state"] == "GO"
    assert d["guards"][GUARD]["strata"]["urban"]["upper"] == pytest.approx(0.0880973028798024)
    assert d["claims"]["improvement"]["level"] == 0.05


def test_glue_guard_regression_blocks_the_claim():
    spec = spec_for()
    evals = evaluations(spec, guard_candidate_fails=range(30))
    assert evals["run-a"]["guards"][GUARD]["verdict"] == "NO-GO"  # the protected gate's own
    result = evaluate.decide_and_aggregate(evals, spec)
    assert result["outcomes"] == dict.fromkeys(RUNS, superiority.GUARD_REGRESSION_FLAGGED)
    assert result["aggregate"]["outcome"] == superiority.GUARD_REGRESSION_FLAGGED


def test_glue_uncertified_guard_is_not_an_improvement():
    spec = spec_for()
    evals = evaluations(spec, guard_candidate_fails=range(3))
    assert evals["run-a"]["guards"][GUARD]["verdict"] == "INSUFFICIENT"
    result = evaluate.decide_and_aggregate(evals, spec)
    assert result["outcomes"] == dict.fromkeys(RUNS, superiority.GUARD_NOT_CERTIFIED)
    # neither claim holds in any run, so aggregate_runs reads INSUFFICIENT
    assert result["aggregate"]["outcome"] == superiority.INSUFFICIENT
    assert result["aggregate"]["n_improvement_claims"] == 0


def _first(spec):
    return spec["run_identities"]["run-a"]["candidate"]


def _forged(evaluation, change):
    forged = copy.deepcopy(evaluation)
    change(forged["gate_response"]["comparison"])
    return forged


def _edited_guard(spec):
    return {**_s2_guard_registration(_first(spec)),
            "score_definition": "edited, laxer hard-brake score"}


def _edited_guard_evaluation(spec):
    return evaluate_registered(synthetic_records(), _edited_guard(spec), scope="synthetic",
                               adapter_metric=GUARD_METRIC)


@pytest.mark.parametrize(("mutate", "message"), [
    # an evaluation of another run's registration (other candidate)
    (lambda e, spec: e["run-a"].update(primary=e["run-b"]["primary"]), "candidate"),
    (lambda e, spec: e["run-a"]["guards"].update({GUARD: e["run-b"]["guards"][GUARD]}),
     "candidate"),
    (lambda e, spec: e["run-a"].update(guards={}), "guard evaluations"),
    (lambda e, spec: e["run-a"]["guards"].update(extra=e["run-a"]["guards"][GUARD]),
     "guard evaluations"),
    (lambda e, spec: e["run-a"].update(primary=dict(e["run-a"]["primary"], status="blocked")),
     "passed"),
    (lambda e, spec: e["run-a"].update(primary=dict(e["run-a"]["primary"], scope="actual")),
     "scope"),
    # the guard's evaluation used as the primary (wrong registration id)
    (lambda e, spec: e["run-a"].update(primary=e["run-a"]["guards"][GUARD]), "evaluated under"),
    # a guard evaluated under a registration with another margin or protocol
    (lambda e, spec: e["run-a"]["guards"].update({GUARD: evaluate_registered(
        synthetic_records(), {**_s2_guard_registration(_first(spec)), "margin": 0.2},
        scope="synthetic", adapter_metric=GUARD_METRIC)}), "margin"),
    (lambda e, spec: e["run-a"]["guards"].update({GUARD: evaluate_registered(
        synthetic_records(), {**_s2_guard_registration(_first(spec)),
                              "deviation_convention": "another convention"},
        scope="synthetic", adapter_metric=GUARD_METRIC)}), "protocol"),
    # the primary evaluated under an edited registration of the same id
    (lambda e, spec: e["run-a"].update(primary=evaluate_registered(
        synthetic_records(reference_fail_roads=ALL_ROADS),
        {**s2_registration(_first(spec)), "score_definition": "edited score"},
        scope="synthetic", adapter_metric=METRIC)), "registered S2 registration"),
    # a guard covering other observations than the primary (decide's own coverage check)
    (lambda e, spec: e["run-a"]["guards"].update({GUARD: evaluate_registered(
        synthetic_records(roads=39), _s2_guard_registration(_first(spec)),
        scope="synthetic", adapter_metric=GUARD_METRIC)}), "paired observations"),
    (lambda e, spec: e.pop("run-c"), "cover exactly"),
    # review F4-05: a forged comparison margin on the primary (its hash still matches)
    (lambda e, spec: e["run-a"].update(primary=_forged(
        e["run-a"]["primary"], lambda c: c.update(margin=0.2))), "comparison margin"),
    # review F4-04: an evaluation without input_hashes blocks instead of raising
    (lambda e, spec: e["run-a"].update(primary={**e["run-a"]["primary"],
                                                "input_hashes": None}), "input_hashes"),
    (lambda e, spec: e["run-a"]["guards"].update({GUARD: {
        **e["run-a"]["guards"][GUARD], "input_hashes": {"records": None}}}), "input_hashes"),
    # review F4-03: guard registrations are required and bound by bytes
    (lambda e, spec: e["run-a"].pop("guard_registrations"), "guard_registrations"),
    (lambda e, spec: e["run-a"].update(guard_registrations={}), "guard registrations"),
    (lambda e, spec: e["run-a"]["guard_registrations"].update(
        {GUARD: e["run-b"]["guard_registrations"][GUARD]}), "candidate and reference"),
    (lambda e, spec: e["run-a"]["guard_registrations"].update(
        {GUARD: {**e["run-a"]["guard_registrations"][GUARD], "margin": "x"}}),
     "valid S2 registration"),
    # the registered guard registration, but the evaluation ran under an edited same-id one
    (lambda e, spec: e["run-a"]["guards"].update({GUARD: _edited_guard_evaluation(spec)}),
     "not evaluated under the guard's registered S2 registration"),
    # the edited same-id registration supplied with its evaluation (consistent pair)
    (lambda e, spec: e["run-a"].update(guards={GUARD: _edited_guard_evaluation(spec)},
                                       guard_registrations={GUARD: _edited_guard(spec)}),
     "s2_protocol_hash"),
])
def test_glue_negative_controls_block_with_the_reason(mutate, message):
    spec = spec_for()
    evals = evaluations(spec)
    mutate(evals, spec)
    result = evaluate.decide_and_aggregate(evals, spec)
    assert result["status"] == "blocked"
    assert result["aggregate"] is None and result["decisions"] is None
    [issue] = result["issues"]
    assert issue["code"] == "decision_binding_failed"
    assert message in issue["message"], issue["message"]


def test_glue_refuses_an_edited_spec_and_decide_run_returns_the_bound_entry():
    spec = spec_for()
    evals = evaluations(spec)
    entry = evaluate.decide_run(spec=spec, run_id="run-a", primary=evals["run-a"]["primary"],
                                guards=evals["run-a"]["guards"],
                                guard_registrations=evals["run-a"]["guard_registrations"])
    assert entry["request_hash"] == spec["request_hashes"]["run-a"]
    assert entry["decision"]["outcome"] == superiority.SUPPORTED
    edited = copy.deepcopy(spec)
    edited["decision_rule"]["guards"] = []
    assert evaluate.decide_and_aggregate(evals, edited)["status"] == "blocked"
    with pytest.raises(ValueError, match="registered run"):
        evaluate.decide_run(spec=spec, run_id="run-z", primary=evals["run-a"]["primary"],
                            guards=evals["run-a"]["guards"],
                            guard_registrations=evals["run-a"]["guard_registrations"])


def test_review_f4_03_synthetic_guard_hash_is_the_registration_without_candidate():
    """The registered hash is one value for every run: only the candidate differs."""
    spec = spec_for()
    hashes = {evaluate._s2_protocol_hash(_s2_guard_registration(identity["candidate"]))
              for identity in spec["run_identities"].values()}
    assert hashes == {SYNTHETIC_GUARD["s2_protocol_hash"]}
    full = {identity_hash(_s2_guard_registration(identity["candidate"]))
            for identity in spec["run_identities"].values()}
    assert len(full) == 3  # why a full-registration hash could not be registered once


def test_review_f4_03_guard_under_an_edited_same_id_registration_is_refused():
    """Probe p9(3): before F4-03 every run aggregated to IMPROVEMENT_SUPPORTED."""
    spec = spec_for()
    evals = evaluations(spec)
    for run in RUNS:
        candidate = spec["run_identities"][run]["candidate"]
        edited = {**_s2_guard_registration(candidate), "score_definition": "edited score"}
        evals[run]["guards"][GUARD] = evaluate_registered(
            synthetic_records(), edited, scope="synthetic", adapter_metric=edited["metric"])
    result = evaluate.decide_and_aggregate(evals, spec)
    assert result["status"] == "blocked" and result["aggregate"] is None
    assert [i["code"] for i in result["issues"]] == ["decision_binding_failed"]


def _strata_records(strata_roads, candidate_fail=(), reference_fail=(), far=()):
    """Paired binary rows per (stratum, road); ``far`` rows deviate 4.0 m (>= the envelope)."""
    rows, index = [], 0
    for stratum, count in strata_roads:
        for road in range(count):
            key = (stratum, road)
            for arm, failed in (("A", key in reference_fail), ("B", key in candidate_fail)):
                rows.append({"arm": arm, "scene_id": f"{stratum}/road-{road:03d}/scene-0",
                             "road_id": f"{stratum}/road-{road:03d}", "stratum": stratum,
                             "replicate_id": 0, "value": 1 if failed else 0, "completed": True,
                             "deviation_m": 4.0 if (key, arm) in far else 0.5, "seed": index})
            index += 1
    return rows


def _refined():
    """Primary on one pooled 80-road stratum; the guard on highway/urban (40 roads each),
    registered guard_strata [highway, urban] and gating_strata [urban] (synthetic slots)."""
    reg = s4_registration({"guard_strata": ["highway", "urban"], "gating_strata": ["urban"]})
    endpoint = {k: v for k, v in _s2_guard_registration(None).items() if k != "candidate"}
    reg["owner"]["improvement_decision_rule"]["guards"] = [{
        **SYNTHETIC_GUARD, "s2_protocol_hash": identity_hash(
            {**endpoint, "required_strata": ["highway", "urban"]})}]
    rr = registry(reg)
    reqs = []
    for index, run in enumerate(RUNS):
        audit, manifest = synthetic_export_chain(index)
        s2 = {**s2_registration(label(manifest)), "required_strata": ["pooled"]}
        reqs.append(evaluate.build_evaluation_request(
            inference_manifest=manifest, s4_registration=reg, run_registry=rr, run_id=run,
            checkpoint_step=3, s2_registration=s2, scope="synthetic",
            export_audit_inputs=audit))
    spec = spec_for(reqs, s4_registration=reg, run_registry=rr)
    assert spec["status"] == "passed", spec["issues"]
    return spec


def _refined_evaluations(spec, *, guard_candidate_fail=(), far=()):
    out = {}
    for run in RUNS:
        candidate = spec["run_identities"][run]["candidate"]
        primary = {**s2_registration(candidate), "required_strata": ["pooled"]}
        guard = {**_s2_guard_registration(candidate), "required_strata": ["highway", "urban"]}
        out[run] = {
            "primary": evaluate_registered(
                _strata_records([("pooled", 80)],
                                reference_fail={("pooled", r) for r in range(80)}, far=far),
                primary, scope="synthetic", adapter_metric=primary["metric"]),
            "guards": {GUARD: evaluate_registered(
                _strata_records([("highway", 40), ("urban", 40)],
                                candidate_fail=guard_candidate_fail), guard,
                scope="synthetic", adapter_metric=guard["metric"])},
            "guard_registrations": {GUARD: guard},
        }
    return out


@pytest.mark.parametrize(("harmed", "highway", "outcome"), [
    # identical guard outputs: upper zero_difference_upper(40) = 0.0881 < 0.1 on both strata
    ((), "GO", superiority.SUPPORTED),
    # 3 highway roads harmed: INSUFFICIENT on the non-gating stratum only (the uncertified
    # case of test_glue_uncertified_guard_is_not_an_improvement); urban alone gates
    (range(3), "INSUFFICIENT", superiority.SUPPORTED),
    # 30 highway roads harmed: NO-GO in a non-gating stratum still flags the regression
    (range(30), "NO-GO", superiority.GUARD_REGRESSION_FLAGGED),
])
def test_review_f4_05_glue_with_listed_guard_and_gating_strata(harmed, highway, outcome):
    """Primary: reference fails on all 80 pooled roads, candidate never (differences -1):
    upper < 0, lower -1 <= -0.03 -> improved, not ruled out."""
    spec = _refined()
    assert spec["decision_rule"]["resolved_guard_strata"] == ["highway", "urban"]
    assert spec["decision_rule"]["resolved_gating_strata"] == ["urban"]
    evals = _refined_evaluations(spec, guard_candidate_fail={("highway", r) for r in harmed})
    result = evaluate.decide_and_aggregate(evals, spec)
    assert result["status"] == "passed", result["issues"]
    guard = result["decisions"]["run-a"]["guards"][GUARD]
    assert {s: v["state"] for s, v in guard["strata"].items()} == {"highway": highway,
                                                                   "urban": "GO"}
    assert result["decisions"]["run-a"]["gating_strata"] == ["urban"]
    assert result["outcomes"] == dict.fromkeys(RUNS, outcome)
    assert result["aggregate"]["outcome"] == outcome


def test_review_f4_08_decision_reports_out_of_envelope_rollouts_separately():
    """Two primary rollouts at 4.0 m (one per arm, road 5) and one candidate rollout at
    4.0 m (road 7): the gate scores them; the decision artifact reports them per run."""
    spec = _refined()
    far = {(("pooled", 5), "A"), (("pooled", 5), "B"), (("pooled", 7), "B")}
    evals = _refined_evaluations(spec, far=far)
    result = evaluate.decide_and_aggregate(evals, spec)
    assert result["status"] == "passed", result["issues"]
    assert set(result["evaluations"]) == set(RUNS)
    run = result["evaluations"]["run-a"]
    envelope = run["primary"]["out_of_envelope"]
    assert envelope["count"] == 3 and envelope["by_arm"] == {"A": 1, "B": 2}
    assert envelope["by_stratum"] == {"pooled": {"A": 1, "B": 2}}
    assert envelope["gate_counts_agree"] is True and envelope["threshold_m"] == 4.0
    assert envelope == evals["run-a"]["primary"]["out_of_envelope"]
    assert run["guards"][GUARD]["out_of_envelope"]["count"] == 0
    assert run["primary"]["request_hash"] == evals["run-a"]["primary"]["request_hash"]
    assert run["guards"][GUARD]["registration_hash"] == identity_hash(
        evals["run-a"]["guard_registrations"][GUARD])
    blocked = evaluate.decide_and_aggregate({**evals, "run-c": {}}, spec)
    assert blocked["status"] == "blocked" and blocked["evaluations"] is None


@pytest.mark.parametrize("endpoint", ["primary", "guard"])
@pytest.mark.parametrize("key", ["request_hash", "out_of_envelope"])
def test_g3_passed_evaluation_missing_a_carried_key_blocks_not_keyerror(endpoint, key):
    """G3 item 5: before the fix ``_carried`` indexed both keys unchecked, so each of these
    four variants raised KeyError out of decide_and_aggregate (only ValueError/TypeError were
    caught). Now each returns the blocked decision_binding_failed artifact naming the run."""
    spec = _refined()
    evals = _refined_evaluations(spec)
    target = (evals["run-b"]["primary"] if endpoint == "primary"
              else evals["run-b"]["guards"][GUARD])
    del target[key]
    result = evaluate.decide_and_aggregate(evals, spec)
    assert result["status"] == "blocked"
    assert [i["code"] for i in result["issues"]] == ["decision_binding_failed"]
    message = result["issues"][0]["message"]
    assert "run 'run-b'" in message and key in message
    assert (endpoint == "guard") == (f"guard {GUARD!r}" in message)
    assert result["evaluations"] is None and result["decisions"] is None


@pytest.mark.parametrize(("key", "value"), [("request_hash", None), ("request_hash", "A" * 64),
                                            ("out_of_envelope", None), ("out_of_envelope", [])])
def test_g3_malformed_carried_values_block(key, value):
    spec = _refined()
    evals = _refined_evaluations(spec)
    evals["run-a"]["primary"][key] = value
    result = evaluate.decide_and_aggregate(evals, spec)
    assert result["status"] == "blocked"
    assert result["issues"][0]["code"] == "decision_binding_failed"
