"""Known answers and negative controls for gate.s2_cost (CPU-08; build-spec A3).

Every literal below is hand-derived in the comment beside it; none is produced by
calling the implementation under test. The D-019/D-020 fixture values are copied
from evidence/s2-vast-host-20261101-v1/ (plan.json, account-after-delete.json,
charges-after-delete.json) and re-derived with Decimal here.
"""

import json
from decimal import Decimal
from pathlib import Path

import pytest

from gate import s2_cost

EVIDENCE = Path(__file__).resolve().parents[1] / "evidence" / "s2-vast-host-20261101-v1"
CAPS = {"pilot": "100", "stage": "1200", "project": "2500"}


def quote(**overrides):
    value = {
        "artifact_kind": "s2_cost_quote",
        "schema_version": 1,
        "scope": "synthetic",
        "currency": "USD",
        "provider": "vast",
        "rate_basis": "per_gpu",
        "compute_usd_per_hour": "2",
        "storage_usd_per_hour": "0",
        "transfer_usd_per_gb": {"download": "0", "upload": "0"},
        "billing_granularity_seconds": 1,
        "fees_and_tax": {"bound_usd": "0", "basis": "synthetic fixture: no fees"},
    }
    value.update(overrides)
    return value


def scope(**overrides):
    value = {
        "tier": "pilot",
        "as_of_epoch": 10_000,
        "caps_usd": dict(CAPS),
        "storage_retention_until_epoch": None,
    }
    value.update(overrides)
    return value


def codes(report):
    return [issue["code"] for issue in report["issues"]]


# ---------------------------------------------------------------- billing units


@pytest.mark.parametrize(
    "seconds,granularity,expected",
    [(61, 60, 120), (60, 60, 60), (0, 60, 0), (1, 60, 60), (61, 1, 61), (0.5, 1, 1)],
)
def test_billed_seconds_rounds_up_to_granularity(seconds, granularity, expected):
    # 61 s at per-minute billing -> ceil(61/60) = 2 billed minutes = 120 s.
    assert s2_cost.billed_seconds(seconds, granularity) == expected


@pytest.mark.parametrize(
    "seconds,granularity", [(-1, 60), (float("nan"), 60), (10, 0), (10, 1.5), (True, 60)]
)
def test_billed_seconds_rejects_invalid_inputs(seconds, granularity):
    with pytest.raises(ValueError):
        s2_cost.billed_seconds(seconds, granularity)


def test_gpu_hours_is_devices_times_billed_hours():
    # 2 devices x 1800 s = 2 x 0.5 h = 1 GPU-hour.
    assert s2_cost.gpu_hours(2, 1800) == Decimal(1)


def test_two_gpus_half_hour_at_two_dollars_per_gpu_hour_is_two_dollars():
    events = [{"event_id": "c1", "kind": "compute_interval", "resource_id": "i-1",
               "start_epoch": 0, "end_epoch": 1800, "gpu_count": 2}]
    ledger = s2_cost.reconcile_cost(events, quote(), scope())
    # 2 GPUs x 0.5 h x $2/GPU-h = $2; GPU-h = 1.
    assert ledger["compute"]["usd"] == "2"
    assert ledger["compute"]["gpu_hours"] == "1"
    assert ledger["compute"]["billed_seconds"] == 1800


def test_per_instance_rate_is_not_multiplied_by_gpu_count():
    events = [{"event_id": "c1", "kind": "compute_interval", "resource_id": "i-1",
               "start_epoch": 0, "end_epoch": 1800, "gpu_count": 2}]
    ledger = s2_cost.reconcile_cost(events, quote(rate_basis="per_instance"), scope())
    # $2/instance-hour x 0.5 h = $1, while GPU-hours remain 2 x 0.5 = 1.
    assert ledger["compute"]["usd"] == "1"
    assert ledger["compute"]["gpu_hours"] == "1"


def test_sixty_one_seconds_at_per_minute_billing_charges_two_minutes():
    events = [{"event_id": "c1", "kind": "compute_interval", "resource_id": "i-1",
               "start_epoch": 0, "end_epoch": 61, "gpu_count": 1}]
    ledger = s2_cost.reconcile_cost(
        events, quote(compute_usd_per_hour="60", billing_granularity_seconds=60), scope()
    )
    # 120 billed seconds at $60/h = $2 (not $1.0166...).
    assert ledger["compute"]["billed_seconds"] == 120
    assert ledger["compute"]["usd"] == "2"


# ---------------------------------------------------------------- cash vs usage


def test_deposit_then_usage_is_one_cash_payment_and_remaining_credit():
    events = [
        {"event_id": "o", "kind": "opening_credit", "amount_usd": "0", "as_of_epoch": 0},
        {"event_id": "d", "kind": "cash_deposit", "amount_usd": "10", "at_epoch": 1},
        {"event_id": "s", "kind": "provider_charge_snapshot", "resource_id": "i-1",
         "at_epoch": 2, "amount_usd": "3"},
    ]
    ledger = s2_cost.reconcile_cost(events, quote(), scope())
    # $10 deposit + $3 usage: cash paid $10 (not $13), usage $3, credit $7.
    assert ledger["cash_paid_usd"] == "10"
    assert ledger["usage"]["accrued_usd"] == "3"
    assert ledger["credit_remaining_usd"] == "7"


def test_missing_opening_credit_makes_credit_not_estimable_not_zero():
    events = [{"event_id": "d", "kind": "cash_deposit", "amount_usd": "10", "at_epoch": 1}]
    ledger = s2_cost.reconcile_cost(events, quote(), scope())
    assert ledger["credit_remaining_usd"] is None
    assert ledger["credit_status"] == "not_estimable"


def test_d020_prior_usage_counts_toward_caps_but_is_not_new_cash():
    events = [{"event_id": "d020", "kind": "prior_usage", "amount_usd": "0.60",
               "decision_ref": "D-020", "counts_toward": "pilot"}]
    ledger = s2_cost.reconcile_cost(events, quote(), scope())
    assert ledger["cash_paid_usd"] == "0"
    assert ledger["prior_usage_usd"] == "0.6"
    # Nested: pilot usage also counts toward stage and project.
    assert {tier: ledger["tiers"][tier]["used_usd"] for tier in s2_cost.TIERS} == {
        "pilot": "0.6", "stage": "0.6", "project": "0.6"
    }


def test_stage_prior_usage_does_not_count_toward_the_pilot_tier():
    events = [{"event_id": "p", "kind": "prior_usage", "amount_usd": "5",
               "decision_ref": "D-099", "counts_toward": "stage"}]
    ledger = s2_cost.reconcile_cost(events, quote(), scope())
    assert ledger["tiers"]["pilot"]["used_usd"] == "0"
    assert ledger["tiers"]["stage"]["used_usd"] == "5"


def test_d019_fixture_credit_delta_exposes_provider_rounding_gap():
    plan = json.loads((EVIDENCE / "plan.json").read_text())
    after = json.loads((EVIDENCE / "account-after-delete.json").read_text())
    charges = json.loads((EVIDENCE / "charges-after-delete.json").read_text())
    events = [
        {"event_id": "open", "kind": "opening_credit",
         "amount_usd": s2_cost.provider_amount(plan["initial_account_credit_usd"]),
         "as_of_epoch": 0},
        {"event_id": "snap", "kind": "provider_charge_snapshot", "resource_id": "58628722",
         "at_epoch": 100, "amount_usd": s2_cost.provider_amount(charges["task_rows"][0]["amount"])},
        {"event_id": "credit", "kind": "credit_snapshot", "at_epoch": 101,
         "credit_usd": s2_cost.provider_amount(after["credit"])},
    ]
    ledger = s2_cost.reconcile_cost(events, quote(scope="public_source_fixture"), scope())
    # 22.793515596363648 - 22.19326022146356 = 0.600255374900088 consumed from credit,
    # versus a displayed provider amount of 0.6: a 0.000255374900088 unexplained gap.
    assert ledger["usage"]["observed_credit_consumption_usd"] == "0.600255374900088"
    assert ledger["usage"]["provider_recorded_usd"] == "0.6"
    assert ledger["usage"]["accrued_usd"] == "0.600255374900088"
    assert ledger["usage"]["credit_vs_recorded_gap_usd"] == "0.000255374900088"
    assert "provider_rounding_gap" in codes(ledger)
    assert ledger["cash_paid_usd"] == "0"
    assert ledger["settlement_status"] == "pending"


def test_provider_amount_rejects_bool_nan_and_negative():
    for bad in (True, float("nan"), float("inf"), -0.1, "abc", None):
        with pytest.raises(ValueError):
            s2_cost.provider_amount(bad)


# ---------------------------------------------------------------- storage


def test_storage_liability_accrues_after_stop_until_destroy():
    events = [
        {"event_id": "c", "kind": "compute_interval", "resource_id": "i-1",
         "start_epoch": 0, "end_epoch": 3600, "gpu_count": 1},
        {"event_id": "s", "kind": "storage_interval", "resource_id": "i-1",
         "start_epoch": 0, "end_epoch": 7200},
    ]
    ledger = s2_cost.reconcile_cost(
        events, quote(compute_usd_per_hour="1", storage_usd_per_hour="0.5"), scope()
    )
    # Compute billed 1 h = $1; storage billed create->destroy 2 h x $0.5 = $1.
    assert ledger["compute"]["usd"] == "1"
    assert ledger["storage"]["usd"] == "1"
    assert ledger["usage"]["computed_usd"] == "2"
    assert ledger["liabilities_usd"] == "0"


def test_open_storage_is_a_forward_liability_until_the_retention_deadline():
    events = [{"event_id": "s", "kind": "storage_interval", "resource_id": "i-1",
               "start_epoch": 0, "end_epoch": None}]
    ledger = s2_cost.reconcile_cost(
        events, quote(storage_usd_per_hour="1"),
        scope(as_of_epoch=3600, storage_retention_until_epoch=3600 + 7200),
    )
    # Accrued 1 h x $1 = $1 by as_of; liability 2 more hours x $1 = $2.
    assert ledger["storage"]["usd"] == "1"
    assert ledger["liabilities_usd"] == "2"
    assert ledger["storage"]["open_resources"] == ["i-1"]


def test_open_storage_without_retention_deadline_is_unknown_exposure():
    events = [{"event_id": "s", "kind": "storage_interval", "resource_id": "i-1",
               "start_epoch": 0, "end_epoch": None}]
    ledger = s2_cost.reconcile_cost(events, quote(storage_usd_per_hour="1"),
                                    scope(as_of_epoch=3600))
    assert ledger["liabilities_usd"] is None
    assert ledger["liability_status"] == "not_estimable"
    assert "open_storage_without_retention_bound" in codes(ledger)


# ---------------------------------------------------------------- liabilities


def test_unknown_liability_blocks_and_provider_error_cannot_clear_it():
    events = [
        {"event_id": "l1", "kind": "liability", "description": "unsettled tax",
         "amount_usd": "1"},
        {"event_id": "e1", "kind": "provider_error", "operation": "charges"},
    ]
    ledger = s2_cost.reconcile_cost(events, quote(), scope())
    assert ledger["liabilities_usd"] == "1"
    events.append({"event_id": "l2", "kind": "liability", "description": "unknown fee",
                   "amount_usd": None})
    ledger = s2_cost.reconcile_cost(events, quote(), scope())
    assert ledger["liabilities_usd"] is None
    assert ledger["status"] == "not_estimable"


def test_liability_cleared_only_with_evidence():
    events = [
        {"event_id": "l1", "kind": "liability", "description": "tax", "amount_usd": "1"},
        {"event_id": "x", "kind": "liability_cleared", "liability_event_id": "l1",
         "evidence_ref": "evidence/receipt.json"},
    ]
    assert s2_cost.reconcile_cost(events, quote(), scope())["liabilities_usd"] == "0"
    events[1]["evidence_ref"] = ""
    ledger = s2_cost.reconcile_cost(events, quote(), scope())
    assert ledger["status"] == "blocked"
    assert "invalid_event" in codes(ledger)


def test_unknown_fee_bound_is_not_estimable():
    ledger = s2_cost.reconcile_cost(
        [], quote(fees_and_tax={"bound_usd": None, "basis": "not quoted"}), scope()
    )
    assert ledger["liabilities_usd"] is None
    assert "fees_and_tax_unbounded" in codes(ledger)


def test_settlement_is_pending_unless_final_evidence_recorded():
    events = [{"event_id": "f", "kind": "settlement_final", "at_epoch": 5,
               "amount_usd": "3", "evidence_ref": "evidence/invoice.json"}]
    assert s2_cost.reconcile_cost([], quote(), scope())["settlement_status"] == "pending"
    assert s2_cost.reconcile_cost(events, quote(), scope())["settlement_status"] == "final"


def test_provider_snapshot_decrease_keeps_the_maximum_and_flags():
    events = [
        {"event_id": "a", "kind": "provider_charge_snapshot", "resource_id": "i-1",
         "at_epoch": 1, "amount_usd": "0.258"},
        {"event_id": "b", "kind": "provider_charge_snapshot", "resource_id": "i-1",
         "at_epoch": 2, "amount_usd": "0.2"},
    ]
    ledger = s2_cost.reconcile_cost(events, quote(), scope())
    assert ledger["usage"]["provider_recorded_usd"] == "0.258"
    assert "provider_snapshot_decreased" in codes(ledger)


def test_cumulative_snapshots_are_not_double_counted():
    # charges-live 0.258 then charges-after-delete 0.6 for the same instance -> 0.6.
    events = [
        {"event_id": "a", "kind": "provider_charge_snapshot", "resource_id": "58628722",
         "at_epoch": 1, "amount_usd": "0.258"},
        {"event_id": "b", "kind": "provider_charge_snapshot", "resource_id": "58628722",
         "at_epoch": 2, "amount_usd": "0.6"},
    ]
    assert s2_cost.reconcile_cost(events, quote(), scope())["usage"]["provider_recorded_usd"] \
        == "0.6"


# ---------------------------------------------------------------- caps


def test_nested_caps_are_checked_and_exceedance_blocks():
    ledger = s2_cost.reconcile_cost(
        [], quote(), scope(caps_usd={"pilot": "1300", "stage": "1200", "project": "2500"})
    )
    assert "caps_not_nested" in codes(ledger)
    assert ledger["status"] == "blocked"
    events = [{"event_id": "s", "kind": "provider_charge_snapshot", "resource_id": "i",
               "at_epoch": 1, "amount_usd": "100.01"}]
    ledger = s2_cost.reconcile_cost(events, quote(), scope())
    assert "cap_exceeded" in codes(ledger)


def test_cap_exceedance_is_detected_even_when_liability_is_unknown():
    events = [
        {"event_id": "s", "kind": "provider_charge_snapshot", "resource_id": "i",
         "at_epoch": 1, "amount_usd": "100.5"},
        {"event_id": "l", "kind": "liability", "description": "x", "amount_usd": None},
    ]
    ledger = s2_cost.reconcile_cost(events, quote(), scope())
    assert "cap_exceeded" in codes(ledger)
    assert ledger["status"] == "blocked"


def test_cap_exceedance_counts_liability():
    events = [
        {"event_id": "s", "kind": "provider_charge_snapshot", "resource_id": "i",
         "at_epoch": 1, "amount_usd": "99"},
        {"event_id": "l", "kind": "liability", "description": "x", "amount_usd": "2"},
    ]
    ledger = s2_cost.reconcile_cost(events, quote(), scope())
    assert ledger["tiers"]["pilot"]["headroom_usd"] == "-1"
    assert "cap_exceeded" in codes(ledger)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e, q, s: e.append({"event_id": "x", "kind": "mystery"}),
        lambda e, q, s: e.append({"event_id": "d", "kind": "cash_deposit", "amount_usd": 10.0,
                                  "at_epoch": 1}),
        lambda e, q, s: e.append({"event_id": "d", "kind": "cash_deposit", "amount_usd": "-1",
                                  "at_epoch": 1}),
        lambda e, q, s: e.extend([
            {"event_id": "dup", "kind": "cash_deposit", "amount_usd": "1", "at_epoch": 1},
            {"event_id": "dup", "kind": "cash_deposit", "amount_usd": "1", "at_epoch": 1}]),
        lambda e, q, s: e.append({"event_id": "c", "kind": "compute_interval",
                                  "resource_id": "i", "start_epoch": 10, "end_epoch": 5,
                                  "gpu_count": 1}),
        lambda e, q, s: e.append({"event_id": "p", "kind": "prior_usage", "amount_usd": "1",
                                  "decision_ref": "D-020", "counts_toward": "galaxy"}),
        lambda e, q, s: e.append({"event_id": "d", "kind": "cash_deposit", "amount_usd": "1",
                                  "at_epoch": 1, "extra": True}),
    ],
)
def test_invalid_events_fail_closed(mutate):
    events, q, s = [], quote(), scope()
    mutate(events, q, s)
    ledger = s2_cost.reconcile_cost(events, q, s)
    assert ledger["status"] == "blocked"
    assert "invalid_event" in codes(ledger)


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": 2},
        {"currency": "EUR"},
        {"rate_basis": "per_node_or_gpu"},
        {"billing_granularity_seconds": 0},
        {"compute_usd_per_hour": 2.0},
        {"scope": "real"},
    ],
)
def test_invalid_quote_fails_closed(overrides):
    ledger = s2_cost.reconcile_cost([], quote(**overrides), scope())
    assert ledger["status"] == "blocked"
    assert "invalid_quote" in codes(ledger)


def test_missing_quote_key_fails_closed():
    q = quote()
    del q["fees_and_tax"]
    assert "invalid_quote" in codes(s2_cost.reconcile_cost([], q, scope()))


def test_ledger_artifact_contract_and_determinism():
    events = [{"event_id": "d", "kind": "cash_deposit", "amount_usd": "10", "at_epoch": 1},
              {"event_id": "o", "kind": "opening_credit", "amount_usd": "0", "as_of_epoch": 0}]
    first = s2_cost.reconcile_cost(events, quote(), scope())
    second = s2_cost.reconcile_cost(list(reversed(events)), quote(), scope())
    assert first == second
    for key in ("artifact_kind", "schema_version", "scope", "producer", "input_hashes",
                "units", "issues", "status"):
        assert key in first
    assert first["artifact_kind"] == "s2_cost_ledger"
    assert first["scope"] == "synthetic"
    assert first["cloud_launch_authorized"] is False


# ---------------------------------------------------------------- next unit


def limits(**overrides):
    value = {
        "task_allowance_usd": None,
        "reserves_usd": {"cleanup": "0"},
        "deadline_epoch": 100_000,
    }
    value.update(overrides)
    return value


def unit(**overrides):
    value = {"upper_bound_usd": "2", "max_seconds": 0, "cleanup_seconds": 0, "gpu_count": 1}
    value.update(overrides)
    return value


def accrued_six_liability_one():
    events = [
        {"event_id": "s", "kind": "provider_charge_snapshot", "resource_id": "i",
         "at_epoch": 1, "amount_usd": "6"},
        {"event_id": "l", "kind": "liability", "description": "unsettled", "amount_usd": "1"},
    ]
    return s2_cost.reconcile_cost(events, quote(compute_usd_per_hour="0"), scope())


def test_equality_boundary_is_rejected_under_strict_headroom():
    # Allowance $10, accrued $6, liability $1, teardown reserve $1 -> remaining $2.
    # A $2 next unit is rejected (2 < 2 is false); $1.99 is accepted.
    ledger = accrued_six_liability_one()
    lim = limits(task_allowance_usd="10", reserves_usd={"cleanup": "1"})
    decision = s2_cost.can_start_next_unit(ledger, lim, unit(), now_epoch=0)
    assert decision["can_start_next_unit"] is False
    assert "task_allowance_headroom_exhausted" in codes(decision)
    assert decision["remaining_usd"]["task_allowance"] == "2"
    ok = s2_cost.can_start_next_unit(ledger, lim, unit(upper_bound_usd="1.99"), now_epoch=0)
    assert ok["can_start_next_unit"] is True
    assert ok["cloud_launch_authorized"] is False


def test_exposure_is_max_of_declared_bound_and_quote_for_unit_plus_cleanup():
    ledger = s2_cost.reconcile_cost([], quote(compute_usd_per_hour="1",
                                              storage_usd_per_hour="0.5"), scope())
    decision = s2_cost.can_start_next_unit(
        ledger, limits(), unit(upper_bound_usd="0.10", max_seconds=3000, cleanup_seconds=600),
        now_epoch=0,
    )
    # Quote exposure: (1 GPU x $1 + $0.5 storage) x 3600 s / 3600 = $1.5 > $0.10.
    assert decision["exposure_usd"] == "1.5"
    two = s2_cost.can_start_next_unit(
        ledger, limits(), unit(upper_bound_usd="0", max_seconds=3600, gpu_count=2), now_epoch=0)
    # per_gpu: (2 x $1 + $0.5) x 1 h = $2.5.
    assert two["exposure_usd"] == "2.5"


def test_at_deadline_blocks_and_duration_must_fit_strictly():
    ledger = s2_cost.reconcile_cost([], quote(compute_usd_per_hour="0"), scope())
    at = s2_cost.can_start_next_unit(ledger, limits(deadline_epoch=50), unit(upper_bound_usd="0"),
                                     now_epoch=50)
    assert "deadline_reached" in codes(at)
    fits = s2_cost.can_start_next_unit(
        ledger, limits(deadline_epoch=100), unit(upper_bound_usd="0", max_seconds=60,
                                                 cleanup_seconds=40), now_epoch=0)
    # 0 + 60 + 40 = 100 is not strictly before the deadline.
    assert "deadline_headroom_exhausted" in codes(fits)


def test_unknown_exposure_or_blocked_ledger_blocks_continuation():
    events = [{"event_id": "l", "kind": "liability", "description": "x", "amount_usd": None}]
    ledger = s2_cost.reconcile_cost(events, quote(), scope())
    decision = s2_cost.can_start_next_unit(ledger, limits(), unit(), now_epoch=0)
    assert "unknown_charge_exposure" in codes(decision)
    assert decision["can_start_next_unit"] is False


def test_reserves_include_independent_evaluation_and_require_cleanup():
    ledger = s2_cost.reconcile_cost([], quote(compute_usd_per_hour="0"), scope())
    decision = s2_cost.can_start_next_unit(
        ledger, limits(reserves_usd={"cleanup": "1", "independent_evaluation": "98"}),
        unit(upper_bound_usd="1"), now_epoch=0,
    )
    # Pilot $100 - $1 - $98 = $1 remaining; a $1 unit is rejected at equality.
    assert "pilot_headroom_exhausted" in codes(decision)
    missing = s2_cost.can_start_next_unit(ledger, limits(reserves_usd={}), unit(), now_epoch=0)
    assert missing["can_start_next_unit"] is False
    assert "invalid_next_unit_input" in codes(missing)
    for bad in (limits(task_allowance_usd=10), {"reserves_usd": {"cleanup": "0"}}):
        assert "invalid_next_unit_input" in codes(
            s2_cost.can_start_next_unit(ledger, bad, unit(), now_epoch=0))
    assert "invalid_next_unit_input" in codes(
        s2_cost.can_start_next_unit(ledger, limits(), unit(), now_epoch=float("nan")))


# ---------------------------------------------------------------- review regressions (A3)


def test_a3_03_final_settlement_above_accrued_raises_usage_and_blocks():
    events = [
        {"event_id": "s", "kind": "provider_charge_snapshot", "resource_id": "i",
         "at_epoch": 1, "amount_usd": "3"},
        {"event_id": "f", "kind": "settlement_final", "at_epoch": 5, "amount_usd": "150",
         "evidence_ref": "evidence/invoice.json"},
    ]
    ledger = s2_cost.reconcile_cost(events, quote(), scope())
    # A $150 final invoice counts in full: pilot used $150 > $100 cap.
    assert ledger["usage"]["settled_usd"] == "150"
    assert ledger["usage"]["accrued_usd"] == "150"
    assert ledger["tiers"]["pilot"]["used_usd"] == "150"
    assert "settlement_exceeds_accrued" in codes(ledger)
    assert "cap_exceeded" in codes(ledger)
    assert ledger["status"] == "blocked"
    decision = s2_cost.can_start_next_unit(ledger, limits(), unit(upper_bound_usd="50"),
                                           now_epoch=0)
    assert decision["can_start_next_unit"] is False


def test_a3_03_settlement_below_accrued_keeps_accrued():
    events = [
        {"event_id": "s", "kind": "provider_charge_snapshot", "resource_id": "i",
         "at_epoch": 1, "amount_usd": "3"},
        {"event_id": "f", "kind": "settlement_final", "at_epoch": 5, "amount_usd": "2.5",
         "evidence_ref": "evidence/invoice.json"},
    ]
    ledger = s2_cost.reconcile_cost(events, quote(), scope())
    assert ledger["usage"]["accrued_usd"] == "3"
    assert "settlement_exceeds_accrued" not in codes(ledger)
    assert ledger["status"] == "passed"


@pytest.mark.parametrize("mutate,code", [
    (lambda x: x.update(status="draft"), "ledger_not_passed"),
    (lambda x: x.update(status="qualify_on_host"), "ledger_not_passed"),
    (lambda x: x.update(schema_version=99), "invalid_ledger"),
    (lambda x: x.update(schema_version=True), "invalid_ledger"),
    (lambda x: x.pop("rates"), "invalid_ledger"),
    (lambda x: x.pop("tiers"), "invalid_ledger"),
    (lambda x: x["tiers"].pop("project"), "invalid_ledger"),
    (lambda x: x["rates"].update(compute_usd_per_hour=2.0), "invalid_ledger"),
])
def test_a3_08_next_unit_requires_a_complete_passed_ledger(mutate, code):
    ledger = s2_cost.reconcile_cost([], quote(), scope())
    assert s2_cost.can_start_next_unit(ledger, limits(), unit(), now_epoch=0)[
        "can_start_next_unit"] is True
    mutate(ledger)
    decision = s2_cost.can_start_next_unit(ledger, limits(), unit(), now_epoch=0)
    assert decision["can_start_next_unit"] is False
    assert code in codes(decision)
    partial = {"artifact_kind": "s2_cost_ledger", "schema_version": 1, "status": "passed",
               "liabilities_usd": "0"}
    assert "invalid_ledger" in codes(
        s2_cost.can_start_next_unit(partial, limits(), unit(), now_epoch=0))


@pytest.mark.parametrize("version", [True, 1.0])
def test_a3_08_quote_schema_version_must_be_the_integer_one(version):
    ledger = s2_cost.reconcile_cost([], quote(schema_version=version), scope())
    assert ledger["status"] == "blocked"
    assert "invalid_quote" in codes(ledger)


def test_a3_09_open_storage_past_its_retention_deadline_blocks():
    events = [{"event_id": "s", "kind": "storage_interval", "resource_id": "i-1",
               "start_epoch": 0, "end_epoch": None}]
    ledger = s2_cost.reconcile_cost(events, quote(storage_usd_per_hour="1"),
                                    scope(as_of_epoch=10_000,
                                          storage_retention_until_epoch=5_000))
    assert ledger["liabilities_usd"] is None
    assert ledger["liability_status"] == "not_estimable"
    assert "open_storage_past_retention" in codes(ledger)
    assert ledger["status"] == "blocked"
    at = s2_cost.reconcile_cost(events, quote(storage_usd_per_hour="1"),
                                scope(as_of_epoch=5_000, storage_retention_until_epoch=5_000))
    assert "open_storage_past_retention" in codes(at)  # equality is overdue too


def test_a3_10_accrued_sums_per_resource_maxima():
    events = [
        # Resource A: quote-derived 9000 s x $2/GPU-h = $5, no provider row yet.
        {"event_id": "a", "kind": "compute_interval", "resource_id": "A",
         "start_epoch": 0, "end_epoch": 9000, "gpu_count": 1},
        # Resource B (a label-matched duplicate): provider snapshot $5 only.
        {"event_id": "b", "kind": "provider_charge_snapshot", "resource_id": "B",
         "at_epoch": 9000, "amount_usd": "5"},
        # Resource C: computed $2 (3600 s) vs provider $3 -> $3.
        {"event_id": "c", "kind": "compute_interval", "resource_id": "C",
         "start_epoch": 0, "end_epoch": 3600, "gpu_count": 1},
        {"event_id": "d", "kind": "provider_charge_snapshot", "resource_id": "C",
         "at_epoch": 3600, "amount_usd": "3"},
    ]
    ledger = s2_cost.reconcile_cost(events, quote(), scope(as_of_epoch=9000))
    # max(5, 0) + max(0, 5) + max(2, 3) = 13; the old max of totals gave max(7, 8) = 8.
    assert ledger["usage"]["computed_usd"] == "7"
    assert ledger["usage"]["provider_recorded_usd"] == "8"
    assert ledger["usage"]["accrued_usd"] == "13"
