"""Known answers and negative controls for teacher_rl.forecast (protocol-design section 8).

Literal arithmetic (quote: $1.50 per GPU-hour, $0.10 storage per instance-hour, 60 s billing,
transfer $0.02/GB down and $0.05/GB up, fees bound $1.00; every job on 2 GPUs, so one billed
second costs 2 x 1.50 / 3600 = $1/1200):

- setup: 1 job + 1 rerun x 1800 s -> 2 x 1800/1200 = $3.00
- training: 3 runs x 1 replica + 1 failed rerun = 4 jobs x 5430 s, billed 5460 s (91 min)
  -> 4 x 5460/1200 = 4 x 4.55 = $18.20
- development selection: 3 runs x 2 checkpoints = 6 jobs x 600 s -> 6 x 0.5 = $3.00
- locked confirmation: 3 fresh runs x 5400 s -> 3 x 4.5 = $13.50
- storage: 1 instance x 36000 s x 0.10/3600 = $1.00
- transfer: 100 GB x 0.02 + 20 GB x 0.05 = $3.00
- fees $1.00, failure reserve $5.00 -> total $47.70; usage $41.70; liabilities $6.00
"""

import copy

import pytest

from teacher_rl import forecast

QUOTE = {
    "artifact_kind": "s2_cost_quote", "schema_version": 1, "scope": "synthetic",
    "currency": "USD", "provider": "vast", "rate_basis": "per_gpu",
    "compute_usd_per_hour": "1.50", "storage_usd_per_hour": "0.10",
    "transfer_usd_per_gb": {"download": "0.02", "upload": "0.05"},
    "billing_granularity_seconds": 60,
    "fees_and_tax": {"bound_usd": "1.00", "basis": "synthetic bound"},
}


def line(line_id, jobs, wall, replicas=1, reruns=0, gpus=2):
    return {"line_id": line_id, "jobs": jobs, "replicas": replicas, "rerun_jobs": reruns,
            "wall_seconds_per_job": wall, "gpu_count": gpus, "basis": "synthetic"}


def actual_events(cash="50.00", snapshot="60.00"):
    return [
        {"event_id": "e1", "kind": "opening_credit", "amount_usd": "10.00", "as_of_epoch": 0},
        {"event_id": "e2", "kind": "cash_deposit", "amount_usd": cash, "at_epoch": 10},
        {"event_id": "e3", "kind": "credit_snapshot", "credit_usd": snapshot, "at_epoch": 100},
        {"event_id": "e4", "kind": "prior_usage", "amount_usd": "0.60",
         "decision_ref": "D-020", "counts_toward": "pilot"},
    ]


def inputs():
    return {
        "artifact_kind": "s4_cost_forecast_input", "schema_version": 1, "scope": "synthetic",
        "quote": copy.deepcopy(QUOTE),
        "host_gpu_count": 2,
        "components": {
            "setup": [line("bootstrap", 1, 1800, reruns=1)],
            "training": [line("baseline_runs", 3, 5430, reruns=1)],
            "development_selection": [line("checkpoint_evals", 6, 600)],
            "locked_confirmation": [line("fresh_runs", 3, 5400)],
        },
        "storage": [{"line_id": "host_disk", "instances": 1, "lifetime_seconds": 36000,
                     "basis": "synthetic"}],
        "transfer": {"download_gb": "100", "upload_gb": "20", "basis": "synthetic"},
        "failure_reserve": {"amount_usd": "5.00", "basis": "synthetic"},
        "ledger_scope": {"tier": "stage", "as_of_epoch": 200,
                         "caps_usd": {"pilot": "100", "stage": "1200", "project": "2500"},
                         "storage_retention_until_epoch": None},
        "actual_events": actual_events(),
    }


def test_formula_known_answer():
    report = forecast.forecast(inputs())
    assert report["status"] == "passed", report["issues"]
    assert report["components_usd"] == {
        "setup": "3", "training": "18.2", "development_selection": "3",
        "locked_confirmation": "13.5", "storage": "1", "transfer": "3", "fees": "1",
        "failure_reserve": "5"}
    assert report["total_usd"] == "47.7"
    assert report["usage_usd"] == "41.7"
    assert report["lines"]["training"][0]["billed_jobs"] == 4
    assert report["lines"]["training"][0]["billed_seconds_per_job"] == 5460
    assert report["lines"]["training"][0]["usd"] == "18.2"
    assert report["billed_jobs_total"] == 2 + 4 + 6 + 3


def test_cash_usage_credit_and_liabilities_are_separate():
    report = forecast.forecast(inputs())
    money = report["money"]
    assert money["current"] == {"cash_paid_usd": "50", "accrued_usd": "0",
                                "credit_remaining_usd": "60", "liabilities_usd": "1"}
    # fees are already a current liability (counted once); the reserve is added
    assert money["projected"] == {"accrued_usd": "41.7", "credit_remaining_usd": "18.3",
                                  "liabilities_usd": "6"}
    assert money["additional_cash_needed_usd"] == "0"
    tiers = report["tiers"]
    assert tiers["stage"] == {"cap_usd": "1200", "projected_used_usd": "42.3",
                              "projected_liability_usd": "6", "projected_headroom_usd": "1151.7",
                              "fits": True}
    assert tiers["project"]["projected_headroom_usd"] == "2451.7"
    assert tiers["pilot"] == {"cap_usd": "100", "projected_used_usd": "0.6",
                              "projected_liability_usd": "0", "projected_headroom_usd": "99.4",
                              "fits": True}


def test_prepaid_credit_short_needs_new_cash():
    data = inputs()
    data["actual_events"] = actual_events(cash="20.00", snapshot="30.00")
    report = forecast.forecast(data)
    assert report["money"]["projected"]["credit_remaining_usd"] == "-11.7"
    assert report["money"]["additional_cash_needed_usd"] == "17.7"


def test_forecast_over_a_cap_blocks():
    data = inputs()
    data["ledger_scope"]["caps_usd"] = {"pilot": "30", "stage": "40", "project": "2500"}
    report = forecast.forecast(data)
    assert report["status"] == "blocked"
    assert report["tiers"]["stage"]["fits"] is False
    assert "forecast_exceeds_stage_cap" in {i["code"] for i in report["issues"]}


def test_failed_reruns_and_replicas_are_counted():
    data = inputs()
    data["components"]["training"] = [line("baseline_runs", 3, 5400, replicas=2, reruns=2)]
    report = forecast.forecast(data)
    # (3 x 2 + 2) x 5400/1200 = 8 x 4.5 = 36
    assert report["components_usd"]["training"] == "36"


@pytest.mark.parametrize("path, value, code", [
    (("components", "training", 0, "wall_seconds_per_job"), None, "measured_input_missing"),
    (("failure_reserve", "amount_usd"), None, "failure_reserve_missing"),
    (("quote", "fees_and_tax", "bound_usd"), None, "fees_and_tax_unbounded"),
    (("transfer", "download_gb"), None, "measured_input_missing"),
])
def test_missing_is_never_zero(path, value, code):
    data = inputs()
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    report = forecast.forecast(data)
    assert report["status"] == "not_estimable"
    assert report["total_usd"] is None
    assert code in {i["code"] for i in report["issues"]}


def test_one_missing_measurement_never_hides_the_other_components():
    data = inputs()
    data["components"]["training"][0]["wall_seconds_per_job"] = None
    report = forecast.forecast(data)
    assert report["components_usd"]["training"] is None
    assert report["components_usd"]["setup"] == "3"
    assert report["components_usd"]["development_selection"] == "3"
    assert report["components_usd"]["locked_confirmation"] == "13.5"
    assert report["total_usd"] is None


@pytest.mark.parametrize("mutate, code", [
    (lambda d: d["components"].pop("locked_confirmation"), "invalid_input"),
    (lambda d: d["components"].update(setup=[]), "component_empty"),
    (lambda d: d["components"]["training"][0].update(gpu_count=1), "gpu_count_mismatch"),
    (lambda d: d["components"]["training"][0].update(jobs=-1), "invalid_input"),
    (lambda d: d["components"]["training"][0].update(wall_seconds_per_job=1.5),
     "invalid_input"),
    (lambda d: d["failure_reserve"].update(amount_usd=5.0), "invalid_input"),
    (lambda d: d["transfer"].update(upload_gb="-1"), "invalid_input"),
    (lambda d: d.update(schema_version=2), "invalid_input"),
    (lambda d: d["components"]["training"].append(line("baseline_runs", 1, 60)),
     "duplicate_line_id"),
    (lambda d: d["actual_events"].append({"event_id": "x", "kind": "nope"}),
     "current_ledger_blocked"),
])
def test_negative_controls_block(mutate, code):
    data = inputs()
    mutate(data)
    report = forecast.forecast(data)
    assert report["status"] == "blocked"
    assert code in {i["code"] for i in report["issues"]}
    assert report["total_usd"] is None


def test_deterministic_and_input_bound():
    a, b = forecast.forecast(inputs()), forecast.forecast(inputs())
    assert a == b
    assert a["input_hashes"]["inputs"] is not None
    assert a["cloud_launch_authorized"] is False


def rental(gpu_count=2, rate="3.50"):
    return {
        "artifact_kind": "s2_action_record", "schema_version": 1, "scope": "synthetic",
        "provider": "vast", "offer": {"offer_id": 1, "sku": "synthetic x2"}, "region": "XX",
        "resources": {"gpu_count": gpu_count, "gpu_edition": "RTX PRO 6000 WS", "disk_gb": 400,
                      "image": "docker.io/vastai/kvm:ubuntu", "volumes": []},
        "max_hourly_usd": rate,
        "window": {"start_utc": "2027-01-15T08:00:00+00:00",
                   "end_utc": "2027-01-15T18:00:00+00:00"},
        "lifecycle_limits": {"startup_limit_seconds": 1200, "arm_limit_seconds": 120,
                             "export_attempt_limit": 2},
        "retention_after_stop": {"mode": "destroy_after_stop", "authority_ref": "D-999",
                                 "retain_until_utc": None},
        "export_destination": "evidence/s4-synthetic-export",
        "cap_headroom_snapshot": {
            "as_of_utc": "2027-01-15T07:00:00+00:00", "tier": "stage",
            "caps_usd": {"pilot": "100", "stage": "1200", "project": "2500"},
            "used_usd": {"pilot": "0.6", "stage": "0.6", "project": "0.6"},
            "liability_usd": {"pilot": "0", "stage": "0", "project": "0"}},
        "approval": {"decision_id": "D-999", "quoted_instruction": "Synthetic only."},
        "license_attestation_refs": [], "scientific_registration_ref": None,
        "engineering_only": True,
    }


def test_rental_crosscheck_uses_the_action_record_gpu_count_and_rate():
    report = forecast.forecast(inputs())
    ok = forecast.crosscheck_rental(report, rental())
    assert ok["status"] == "passed", ok["issues"]
    assert ok["gpu_count"] == 2
    assert ok["quoted_instance_usd_per_hour"] == "3.1"  # 2 x 1.50 + 0.10 storage
    bad = forecast.crosscheck_rental(report, rental(gpu_count=1))
    assert "gpu_count_mismatch" in {i["code"] for i in bad["issues"]}
    pricey = forecast.crosscheck_rental(report, rental(rate="3.00"))
    assert "quote_exceeds_max_hourly" in {i["code"] for i in pricey["issues"]}
    unadmitted = rental()
    unadmitted["approval"]["decision_id"] = "nope"
    assert "rental_record_not_admitted" in {
        i["code"] for i in forecast.crosscheck_rental(report, unadmitted)["issues"]}


# --- review B3-07 / B3-13 / B3-14 ----------------------------------------------------------


def test_review_b3_07_scope_must_match_the_quote():
    data = inputs()
    data["scope"] = "actual"  # priced from a synthetic quote and ledger
    report = forecast.forecast(data)
    assert report["status"] == "blocked" and report["total_usd"] is None
    assert "scope_mismatch" in {i["code"] for i in report["issues"]}


@pytest.mark.parametrize("gpus", [1, 4])
def test_review_b3_13_host_gpu_count_is_the_two_gpu_topology(gpus):
    data = inputs()
    data["host_gpu_count"] = gpus
    for lines in data["components"].values():
        for row in lines:
            row["gpu_count"] = gpus
    report = forecast.forecast(data)
    assert report["status"] == "blocked" and report["total_usd"] is None
    assert "host_gpu_count_mismatch" in {i["code"] for i in report["issues"]}


def test_review_b3_14_a_lower_tier_exactly_at_its_cap_does_not_block():
    data = inputs()
    data["actual_events"][3]["amount_usd"] = "100.00"  # prior usage fills the pilot cap
    report = forecast.forecast(data)
    assert report["status"] == "passed", report["issues"]
    assert report["tiers"]["pilot"] == {"cap_usd": "100", "projected_used_usd": "100",
                                        "projected_liability_usd": "0",
                                        "projected_headroom_usd": "0", "fits": True}
    # the forecast's own tier still needs strict headroom: 99.4 + 41.7 + 1 + 5 used/owed
    data["ledger_scope"]["caps_usd"]["stage"] = "147.7"
    blocked = forecast.forecast(data)
    assert "forecast_exceeds_stage_cap" in {i["code"] for i in blocked["issues"]}


# --- F4: a per-line host and GPU count (completeness audit "S4 forecast") -------------------
#
# Schema v2 adds ``evaluation_host`` (null or a separate host with its own quote and GPU count)
# and a ``host`` on every compute line, storage entry and transfer entry. Known answer: the v1
# fixture with the locked confirmation moved to a 1-GPU S2 host quoted at $0.90 per GPU-hour,
# $0.05 storage per instance-hour, 60 s billing, transfer $0.01/GB down and $0.04/GB up, fees
# bound $0.50:
#
# - locked confirmation: 3 jobs x 5400 s x 1 GPU x 0.90/3600 = 3 x 1.35 = $4.05
# - evaluation storage 1 x 18000 s x 0.05/3600 = $0.25; transfer 60 x 0.01 + 1 x 0.04 = $0.64
# - training host unchanged: setup 3, training 18.2, development 3, storage 1, transfer 3
# - usage 33.14 (training host 28.2, evaluation host 4.94); fees 1.00 + 0.50; reserve 5
#   -> total $39.64
# - credit 60 absorbs only the training host: max(0, 28.2 + 1 + 5 - 60) = 0, plus the
#   evaluation host's 4.94 + 0.50 paid separately -> additional cash $5.44
# - stage tier: used 0.6 + 33.14 = 33.74, liability 1 + 5 + 0.5 = 6.5 -> headroom 1159.76

EVAL_QUOTE = {
    "artifact_kind": "s2_cost_quote", "schema_version": 1, "scope": "synthetic",
    "currency": "USD", "provider": "vast", "rate_basis": "per_gpu",
    "compute_usd_per_hour": "0.90", "storage_usd_per_hour": "0.05",
    "transfer_usd_per_gb": {"download": "0.01", "upload": "0.04"},
    "billing_granularity_seconds": 60,
    "fees_and_tax": {"bound_usd": "0.50", "basis": "synthetic bound"},
}


def v2_line(line_id, jobs, wall, *, host="training", gpus=None, **kw):
    value = line(line_id, jobs, wall, gpus=(2 if host == "training" else 1) if gpus is None
                 else gpus, **kw)
    value["host"] = host
    return value


def v2_inputs():
    data = inputs()
    data["schema_version"] = 2
    data["evaluation_host"] = {"gpu_count": 1, "quote": copy.deepcopy(EVAL_QUOTE),
                               "basis": "synthetic separate S2 host"}
    data["components"] = {
        "setup": [v2_line("bootstrap", 1, 1800, reruns=1)],
        "training": [v2_line("baseline_runs", 3, 5430, reruns=1)],
        "development_selection": [v2_line("checkpoint_evals", 6, 600)],
        "locked_confirmation": [v2_line("fresh_runs", 3, 5400, host="evaluation")],
    }
    data["storage"] = [
        {"line_id": "host_disk", "host": "training", "instances": 1, "lifetime_seconds": 36000,
         "basis": "synthetic"},
        {"line_id": "eval_disk", "host": "evaluation", "instances": 1,
         "lifetime_seconds": 18000, "basis": "synthetic"}]
    data["transfer"] = [
        {"line_id": "train_io", "host": "training", "download_gb": "100", "upload_gb": "20",
         "basis": "synthetic"},
        {"line_id": "eval_io", "host": "evaluation", "download_gb": "60", "upload_gb": "1",
         "basis": "synthetic"}]
    return data


def test_v2_separate_one_gpu_evaluation_host_known_answer():
    report = forecast.forecast(v2_inputs())
    assert report["status"] == "passed", report["issues"]
    assert report["components_usd"] == {
        "setup": "3", "training": "18.2", "development_selection": "3",
        "locked_confirmation": "4.05", "storage": "1.25", "transfer": "3.64", "fees": "1.5",
        "failure_reserve": "5"}
    assert report["total_usd"] == "39.64" and report["usage_usd"] == "33.14"
    row = report["lines"]["locked_confirmation"][0]
    assert (row["host"], row["gpu_count"], row["usd"], row["gpu_hours"]) == (
        "evaluation", 1, "4.05", "4.5")
    assert report["lines"]["training"][0]["host"] == "training"
    assert report["usage_by_host_usd"] == {"training": "28.2", "evaluation": "4.94"}
    assert report["hosts"] == {
        "training": {"gpu_count": 2, "provider": "vast", "rate_basis": "per_gpu",
                     "compute_usd_per_hour": "1.5", "storage_usd_per_hour": "0.1"},
        "evaluation": {"gpu_count": 1, "provider": "vast", "rate_basis": "per_gpu",
                       "compute_usd_per_hour": "0.9", "storage_usd_per_hour": "0.05"}}
    money = report["money"]
    assert money["projected"] == {"accrued_usd": "28.2", "credit_remaining_usd": "31.8",
                                  "liabilities_usd": "6.5"}
    assert money["separate_host"]["usage_usd"] == "4.94"
    assert money["separate_host"]["fees_bound_usd"] == "0.5"
    assert money["separate_host"]["cash_needed_usd"] == "5.44"
    assert money["additional_cash_needed_usd"] == "5.44"
    assert report["tiers"]["stage"] == {
        "cap_usd": "1200", "projected_used_usd": "33.74", "projected_liability_usd": "6.5",
        "projected_headroom_usd": "1159.76", "fits": True}
    assert report["billed_jobs_total"] == 15
    # 2 (setup) + 12.1333... (training) + 2 (development) + 4.5 (confirmation) GPU-hours
    assert report["gpu_hours_total"].startswith("20.6333")


def test_v1_input_keeps_its_answer_and_names_the_training_host():
    report = forecast.forecast(inputs())
    assert report["total_usd"] == "47.7"
    assert {row["host"] for rows in report["lines"].values() for row in rows} == {"training"}
    assert report["usage_by_host_usd"] == {"training": "41.7"}
    assert report["money"]["separate_host"] is None


def test_v2_without_an_evaluation_host_equals_the_v1_answer():
    data = v2_inputs()
    data["evaluation_host"] = None
    data["components"]["locked_confirmation"] = [v2_line("fresh_runs", 3, 5400)]
    data["storage"] = data["storage"][:1]
    data["transfer"] = data["transfer"][:1]
    report = forecast.forecast(data)
    assert report["status"] == "passed", report["issues"]
    assert report["total_usd"] == "47.7"
    assert report["money"]["additional_cash_needed_usd"] == "0"


@pytest.mark.parametrize("mutate, code", [
    (lambda d: d.update(evaluation_host=None), "evaluation_host_unregistered"),
    (lambda d: d["components"]["training"][0].update(host="evaluation", gpu_count=1),
     "training_host_required"),
    (lambda d: d["components"]["setup"][0].update(host="evaluation", gpu_count=1),
     "training_host_required"),
    (lambda d: d["components"]["locked_confirmation"][0].update(gpu_count=2),
     "gpu_count_mismatch"),
    (lambda d: d["components"]["locked_confirmation"][0].update(host="cloud"), "invalid_input"),
    (lambda d: d["components"]["locked_confirmation"][0].pop("host"), "invalid_input"),
    (lambda d: d["evaluation_host"].update(gpu_count=0), "invalid_input"),
    (lambda d: d["evaluation_host"]["quote"].update(scope="actual"), "scope_mismatch"),
    (lambda d: d["evaluation_host"]["quote"].update(currency="EUR"),
     "invalid_evaluation_quote"),
    (lambda d: d["evaluation_host"].pop("basis"), "invalid_input"),
    (lambda d: d.update(storage=d["storage"][:1]), "storage_missing_for_host"),
    (lambda d: d.update(transfer=d["transfer"][:1]), "transfer_missing_for_host"),
    (lambda d: d["storage"][1].update(host="nowhere"), "invalid_input"),
    (lambda d: d.update(transfer={"download_gb": "1", "upload_gb": "1", "basis": "x"}),
     "invalid_input"),
    (lambda d: d.update(schema_version=3), "invalid_input"),
])
def test_v2_negative_controls_block(mutate, code):
    data = v2_inputs()
    mutate(data)
    report = forecast.forecast(data)
    assert report["status"] == "blocked"
    assert code in {i["code"] for i in report["issues"]}, report["issues"]
    assert report["total_usd"] is None


def test_v2_unbounded_evaluation_fees_are_not_estimable():
    data = v2_inputs()
    data["evaluation_host"]["quote"]["fees_and_tax"]["bound_usd"] = None
    report = forecast.forecast(data)
    assert report["status"] == "not_estimable" and report["total_usd"] is None
    assert ("fees_and_tax_unbounded", "evaluation_host.quote.fees_and_tax.bound_usd") in {
        (i["code"], i["path"]) for i in report["issues"]}


def test_v2_rental_crosscheck_checks_one_host_against_one_record():
    report = forecast.forecast(v2_inputs())
    training = forecast.crosscheck_rental(report, rental())
    assert training["status"] == "passed", training["issues"]
    assert training["quoted_instance_usd_per_hour"] == "3.1"
    evaluation = forecast.crosscheck_rental(report, rental(gpu_count=1, rate="1.00"),
                                            host="evaluation")
    assert evaluation["status"] == "passed", evaluation["issues"]
    assert evaluation["gpu_count"] == 1
    assert evaluation["quoted_instance_usd_per_hour"] == "0.95"  # 1 x 0.90 + 0.05 storage
    wrong = forecast.crosscheck_rental(report, rental(gpu_count=2), host="evaluation")
    assert "gpu_count_mismatch" in {i["code"] for i in wrong["issues"]}
    pricey = forecast.crosscheck_rental(report, rental(gpu_count=1, rate="0.90"),
                                        host="evaluation")
    assert "quote_exceeds_max_hourly" in {i["code"] for i in pricey["issues"]}
    absent = forecast.crosscheck_rental(forecast.forecast(inputs()), rental(gpu_count=1),
                                        host="evaluation")
    assert "host_not_in_forecast" in {i["code"] for i in absent["issues"]}
