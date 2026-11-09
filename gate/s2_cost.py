"""S2 cost reconciliation and next-unit continuation check (CPU-08; build-spec A3).

Serves S2.2/S2.7 [C] (G7) under D-018/D-024. Agent-authored evaluator engineering.

Purpose: keep cash paid, usage accrued, remaining credit and liabilities separate
(D-018: a prepaid credit deposit is one cash outlay; consumption from it is never a
second payment), apply provider billing granularity, and decide conservatively
whether the next bounded unit fits inside every nested cap (P-005 under D-018:
pilot <= S2 stage <= project), a task allowance, registered reserves and the
original absolute deadline.

Verification ceiling (build-spec section 1):

- ``billed_seconds``, ``gpu_hours``, ``provider_amount`` - L1, literal known answers.
- ``reconcile_cost`` - L1 arithmetic over supplied events. The meaning of a real
  provider's charge rows and credit fields is L3 (``qualify_on_host``): Vast's
  ``show invoices-v1 --charges`` amounts are rounded provider snapshots, never an
  assurance of final settlement (D-020).
- ``can_start_next_unit`` - L1 strict-headroom rule. It is an engineering stop rule,
  never a budget approval or rental authority (``cloud_launch_authorized`` is
  always ``False``).

Rules:

- Amounts are exact ``Decimal`` values supplied as decimal strings; binary floats
  are rejected, except at the provider boundary through ``provider_amount`` which
  converts the shortest float repr (``Decimal(str(x))``).
- Usage accrued is the maximum of (a) the per-resource sum of
  ``max(quote-derived, latest cumulative provider snapshot)`` plus transfer, keyed by
  the shared ``resource_id`` (the bound instance ID; a resource seen by only one
  source counts in full), (b) the observed credit consumption (opening credit +
  deposits - latest credit snapshot) and (c) the largest ``settlement_final`` amount.
  A gap between observed credit consumption and the displayed provider amount is
  reported as ``provider_rounding_gap`` (fixture: D-019 credit delta
  0.600255374900088 versus displayed $0.60); a final settlement above the other
  figures is reported as ``settlement_exceeds_accrued`` and raises usage.
- Storage is billed from create to destroy (it continues after stop). Open storage
  accrues to ``as_of_epoch`` and is a forward liability up to the recorded
  retention deadline; without that deadline the exposure is unknown, and open
  storage at or past that deadline is overdue (blocking
  ``open_storage_past_retention``, liability not estimable).
- Missing is never zero: unknown liabilities, unbounded fees/tax or missing opening
  credit give ``not_estimable`` and block continuation. Provider errors are recorded
  and never clear a liability.
- Prior usage (for example D-020's $0.60, drawn from existing credit) counts toward
  the nested caps from its ``counts_toward`` tier upward; it is never new cash.

Owner inputs (required, never defaulted here): the cap values (D-018 records
$100/$1,200/$2,500), any task allowance, the reserve amounts (cleanup and any
registered independent-evaluation reserve), the retention deadline and the absolute
deadline. The strict ``exposure < remaining`` boundary is an
``agent technical default (D-018)`` from readiness CPU-08.

This module never calls a provider, reads credentials, approves spending, funds an
account, or changes caps.
"""

import math
from decimal import ROUND_CEILING, Context, Decimal, InvalidOperation, localcontext

from gate.s2_identity import identity_hash

SCHEMA_VERSION = 1
TIERS = ("pilot", "stage", "project")
SCOPES = ("synthetic", "public_source_fixture", "actual")
_CONTEXT = Context(prec=60, rounding=ROUND_CEILING)

_QUOTE_KEYS = {
    "artifact_kind", "schema_version", "scope", "currency", "provider", "rate_basis",
    "compute_usd_per_hour", "storage_usd_per_hour", "transfer_usd_per_gb",
    "billing_granularity_seconds", "fees_and_tax",
}
_EVENT_KEYS = {
    "cash_deposit": {"amount_usd", "at_epoch"},
    "opening_credit": {"amount_usd", "as_of_epoch"},
    "credit_snapshot": {"credit_usd", "at_epoch"},
    "prior_usage": {"amount_usd", "decision_ref", "counts_toward"},
    "compute_interval": {"resource_id", "start_epoch", "end_epoch", "gpu_count"},
    "storage_interval": {"resource_id", "start_epoch", "end_epoch"},
    "transfer": {"direction", "gb"},
    "provider_charge_snapshot": {"resource_id", "at_epoch", "amount_usd"},
    "liability": {"description", "amount_usd"},
    "liability_cleared": {"liability_event_id", "evidence_ref"},
    "provider_error": {"operation"},
    "settlement_final": {"at_epoch", "amount_usd", "evidence_ref"},
}


class _Invalid(ValueError):
    """Structural input error; converted into a blocking issue."""


def _issue(code, path, message, severity="blocking", evidence_refs=()):
    return {
        "code": code,
        "path": path,
        "severity": severity,
        "evidence_refs": list(evidence_refs),
        "message": message,
    }


def _fmt(value):
    if value is None:
        return None
    value = value.normalize(_CONTEXT)
    if value == 0:
        return "0"
    return format(value, "f")


def _usd(value, *, allow_negative=False):
    """Strict decimal string -> Decimal; ints, floats and bools are rejected."""
    if type(value) is str:
        try:
            amount = Decimal(value)
        except InvalidOperation:
            raise _Invalid("invalid decimal amount") from None
    else:
        raise _Invalid("amounts must be decimal strings")
    if not amount.is_finite() or (amount < 0 and not allow_negative):
        raise _Invalid("amounts must be finite and nonnegative")
    return amount


def provider_amount(value):
    """Convert one provider JSON number (float/int) to a nonnegative decimal string.

    L1. Uses the shortest repr (``str(float)``), never the binary expansion.
    """
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("provider amounts must be finite nonnegative numbers")
    return _fmt(Decimal(str(value)))


def _epoch(value, *, allow_none=False):
    if value is None and allow_none:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise _Invalid("epochs must be finite nonnegative numbers")
    return value


def billed_seconds(seconds, granularity_seconds):
    """Round a duration up to whole billing units (61 s at 60 s -> 120 s). L1."""
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("duration must be a finite nonnegative number")
    if type(granularity_seconds) is not int or granularity_seconds <= 0:
        raise ValueError("billing granularity must be a positive integer of seconds")
    return math.ceil(Decimal(str(seconds)) / granularity_seconds) * granularity_seconds


def gpu_hours(devices, billed):
    """GPU-hours = devices x billed hours (2 x 1800 s = 1). L1."""
    if type(devices) is not int or devices < 1 or type(billed) is not int or billed < 0:
        raise ValueError("devices must be >= 1 and billed seconds a nonnegative integer")
    with localcontext(_CONTEXT):
        return Decimal(devices) * Decimal(billed) / Decimal(3600)


def _validate_quote(quote):
    if type(quote) is not dict or set(quote) != _QUOTE_KEYS:
        raise _Invalid("quote keys must match the schema exactly")
    version = quote["schema_version"]
    if quote["artifact_kind"] != "s2_cost_quote" or type(
        version
    ) is not int or version != SCHEMA_VERSION:
        raise _Invalid("unknown quote kind or schema version")
    if quote["scope"] not in SCOPES or quote["currency"] != "USD":
        raise _Invalid("quote scope must be declared and currency USD")
    if type(quote["provider"]) is not str or not quote["provider"]:
        raise _Invalid("quote provider required")
    if quote["rate_basis"] not in ("per_gpu", "per_instance"):
        raise _Invalid("rate basis must be per_gpu or per_instance")
    granularity = quote["billing_granularity_seconds"]
    if type(granularity) is not int or granularity <= 0:
        raise _Invalid("billing granularity must be a positive integer")
    transfer = quote["transfer_usd_per_gb"]
    if type(transfer) is not dict or set(transfer) != {"download", "upload"}:
        raise _Invalid("transfer rates require download and upload")
    fees = quote["fees_and_tax"]
    if type(fees) is not dict or set(fees) != {"bound_usd", "basis"}:
        raise _Invalid("fees_and_tax requires bound_usd and basis")
    if type(fees["basis"]) is not str or not fees["basis"].strip():
        raise _Invalid("fees_and_tax basis must be stated")
    return {
        "compute": _usd(quote["compute_usd_per_hour"]),
        "storage": _usd(quote["storage_usd_per_hour"]),
        "download": _usd(transfer["download"]),
        "upload": _usd(transfer["upload"]),
        "granularity": granularity,
        "per_gpu": quote["rate_basis"] == "per_gpu",
        "fees": None if fees["bound_usd"] is None else _usd(fees["bound_usd"]),
    }


def _validate_scope(scope):
    keys = {"tier", "as_of_epoch", "caps_usd", "storage_retention_until_epoch"}
    if type(scope) is not dict or set(scope) != keys:
        raise _Invalid("scope keys must match the schema exactly")
    if scope["tier"] not in TIERS:
        raise _Invalid("scope tier must be pilot, stage or project")
    caps = scope["caps_usd"]
    if type(caps) is not dict or set(caps) != set(TIERS):
        raise _Invalid("caps for pilot, stage and project are required")
    return {
        "tier": scope["tier"],
        "as_of": _epoch(scope["as_of_epoch"]),
        "caps": {tier: _usd(caps[tier]) for tier in TIERS},
        "retention_until": _epoch(scope["storage_retention_until_epoch"], allow_none=True),
    }


def _nonempty(value):
    if type(value) is not str or not value.strip():
        raise _Invalid("text field required")
    return value


def _validate_event(event):
    if type(event) is not dict or event.get("kind") not in _EVENT_KEYS:
        raise _Invalid("unknown event kind")
    kind = event["kind"]
    if set(event) != _EVENT_KEYS[kind] | {"event_id", "kind"}:
        raise _Invalid(f"{kind} keys must match the schema exactly")
    _nonempty(event["event_id"])
    if kind in ("cash_deposit", "opening_credit", "prior_usage", "provider_charge_snapshot",
                "settlement_final"):
        _usd(event["amount_usd"])
    if kind == "liability":
        _nonempty(event["description"])
        if event["amount_usd"] is not None:
            _usd(event["amount_usd"])
    if kind == "credit_snapshot":
        _usd(event["credit_usd"])
    for key in ("at_epoch", "as_of_epoch"):
        if key in event:
            _epoch(event[key])
    if kind == "prior_usage":
        _nonempty(event["decision_ref"])
        if event["counts_toward"] not in TIERS:
            raise _Invalid("prior usage tier must be pilot, stage or project")
    if kind in ("compute_interval", "storage_interval"):
        _nonempty(event["resource_id"])
        start = _epoch(event["start_epoch"])
        end = _epoch(event["end_epoch"], allow_none=True)
        if end is not None and end < start:
            raise _Invalid("interval end precedes start")
        if kind == "compute_interval" and (
            type(event["gpu_count"]) is not int or event["gpu_count"] < 1
        ):
            raise _Invalid("gpu_count must be a positive integer")
    if kind == "provider_charge_snapshot":
        _nonempty(event["resource_id"])
    if kind == "transfer":
        if event["direction"] not in ("download", "upload"):
            raise _Invalid("transfer direction must be download or upload")
        _usd(event["gb"])
    if kind in ("liability_cleared", "settlement_final"):
        _nonempty(event["evidence_ref"])
    if kind == "liability_cleared":
        _nonempty(event["liability_event_id"])
    if kind == "provider_error":
        _nonempty(event["operation"])
    return event


def _interval_seconds(event, as_of, granularity):
    end = event["end_epoch"] if event["end_epoch"] is not None else max(
        as_of, event["start_epoch"]
    )
    return billed_seconds(end - event["start_epoch"], granularity)


def reconcile_cost(events, quote, scope):
    """Reconcile supplied cost events against a quote and nested caps. L1.

    ``events`` order is not semantic (sorted by ``event_id`` before use). Returns a
    ``s2_cost_ledger`` artifact; ``status`` is ``passed``, ``blocked`` or
    ``not_estimable``. Structural errors never raise: they block with issues.
    """
    issues = []
    input_hashes = {}
    ordered = events
    if type(events) is list and all(
        type(e) is dict and type(e.get("event_id")) is str for e in events
    ):
        ordered = sorted(events, key=lambda e: e["event_id"])  # order is not semantic
    for name, value in (("events", ordered), ("quote", quote), ("scope", scope)):
        try:
            input_hashes[name] = identity_hash(value)
        except ValueError:
            input_hashes[name] = None
    declared_scope = quote.get("scope") if type(quote) is dict else None
    try:
        rates = _validate_quote(quote)
    except _Invalid as error:
        issues.append(_issue("invalid_quote", "quote", str(error)))
        rates = None
    try:
        context = _validate_scope(scope)
    except _Invalid as error:
        issues.append(_issue("invalid_scope", "scope", str(error)))
        context = None
    valid = []
    if type(events) is not list:
        issues.append(_issue("invalid_event", "events", "events must be a list"))
        events = []
    seen = set()
    for index, event in enumerate(events):
        try:
            valid.append(_validate_event(event))
            if event["event_id"] in seen:
                raise _Invalid("duplicate event_id")
            seen.add(event["event_id"])
        except _Invalid as error:
            issues.append(_issue("invalid_event", f"events[{index}]", str(error)))
    if rates is None or context is None or any(i["code"] == "invalid_event" for i in issues):
        return _ledger(declared_scope, input_hashes, issues, None)
    valid.sort(key=lambda item: item["event_id"])
    by_kind = {kind: [e for e in valid if e["kind"] == kind] for kind in _EVENT_KEYS}

    with localcontext(_CONTEXT):
        # Compute: each interval billed separately at the provider granularity.
        compute_seconds, compute_usd, compute_gpu_hours = 0, Decimal(0), Decimal(0)
        by_resource = {}
        for event in by_kind["compute_interval"]:
            billed = _interval_seconds(event, context["as_of"], rates["granularity"])
            factor = event["gpu_count"] if rates["per_gpu"] else 1
            amount = rates["compute"] * factor * billed / 3600
            compute_seconds += billed
            compute_usd += amount
            compute_gpu_hours += gpu_hours(event["gpu_count"], billed)
            by_resource[event["resource_id"]] = by_resource.get(
                event["resource_id"], Decimal(0)) + amount
        # Storage: create -> destroy; open storage accrues to as_of plus forward liability.
        storage_seconds, storage_usd, open_storage, forward = 0, Decimal(0), [], Decimal(0)
        for event in by_kind["storage_interval"]:
            billed = _interval_seconds(event, context["as_of"], rates["granularity"])
            amount = rates["storage"] * billed / 3600
            storage_seconds += billed
            storage_usd += amount
            by_resource[event["resource_id"]] = by_resource.get(
                event["resource_id"], Decimal(0)) + amount
            if event["end_epoch"] is None:
                open_storage.append(event["resource_id"])
        liability_unknown = False
        if open_storage:
            retention = context["retention_until"]
            if retention is None:
                liability_unknown = True
                issues.append(_issue(
                    "open_storage_without_retention_bound", "scope.storage_retention_until_epoch",
                    "storage is still billable and no retention deadline bounds it"))
            elif context["as_of"] >= retention:
                liability_unknown = True  # overdue for destruction: future charges unbounded
                issues.append(_issue(
                    "open_storage_past_retention", "scope.storage_retention_until_epoch",
                    "storage is still open at or after its retention deadline"))
            else:
                remaining = billed_seconds(max(0, retention - context["as_of"]),
                                           rates["granularity"])
                forward = rates["storage"] * remaining * len(open_storage) / 3600
        transfer_usd = sum(
            (_usd(e["gb"]) * rates[e["direction"]] for e in by_kind["transfer"]), Decimal(0)
        )
        computed = compute_usd + storage_usd + transfer_usd

        # Provider cumulative snapshots: latest per resource; a decrease keeps the max.
        latest = {}
        for event in sorted(by_kind["provider_charge_snapshot"],
                            key=lambda e: (e["resource_id"], e["at_epoch"], e["event_id"])):
            amount = _usd(event["amount_usd"])
            previous = latest.get(event["resource_id"])
            if previous is not None and amount < previous:
                issues.append(_issue(
                    "provider_snapshot_decreased", f"events.{event['event_id']}",
                    "a later cumulative provider snapshot is lower; the maximum is kept",
                    severity="warning"))
                amount = previous
            latest[event["resource_id"]] = amount
        recorded = sum(latest.values(), Decimal(0))

        cash = sum((_usd(e["amount_usd"]) for e in by_kind["cash_deposit"]), Decimal(0))
        openings = by_kind["opening_credit"]
        if len(openings) > 1:
            issues.append(_issue("invalid_event", "events", "at most one opening_credit"))
            return _ledger(declared_scope, input_hashes, issues, None)
        opening = _usd(openings[0]["amount_usd"]) if openings else None
        observed = gap = None
        snapshots = sorted(by_kind["credit_snapshot"], key=lambda e: (e["at_epoch"], e["event_id"]))
        if opening is not None and snapshots:
            observed = opening + cash - _usd(snapshots[-1]["credit_usd"])
            if latest:
                gap = observed - recorded
                if gap != 0:
                    issues.append(_issue(
                        "provider_rounding_gap", "usage",
                        "observed credit consumption differs from displayed provider charges",
                        severity="warning"))
        # Per resource: max(quote-derived, provider snapshot); unmatched resources in full.
        per_resource = sum((max(by_resource.get(r, Decimal(0)), latest.get(r, Decimal(0)))
                            for r in sorted(set(by_resource) | set(latest))), Decimal(0))
        accrued = max([per_resource + transfer_usd]
                      + ([observed] if observed is not None else []))
        settlements = [_usd(e["amount_usd"]) for e in by_kind["settlement_final"]]
        settled = max(settlements) if settlements else None
        if settled is not None and settled > accrued:
            issues.append(_issue(
                "settlement_exceeds_accrued", "usage",
                "the final settled amount exceeds every accrued figure; usage is raised to it",
                severity="warning"))
            accrued = settled
        credit = None if opening is None else opening + cash - accrued

        prior = sum((_usd(e["amount_usd"]) for e in by_kind["prior_usage"]), Decimal(0))
        cleared = {e["liability_event_id"] for e in by_kind["liability_cleared"]}
        known_ids = {e["event_id"] for e in by_kind["liability"]}
        for ref in sorted(cleared - known_ids):
            issues.append(_issue("invalid_event", "events",
                                 f"liability_cleared references unknown liability {ref}"))
        liability = forward
        for event in by_kind["liability"]:
            if event["event_id"] in cleared:
                continue
            if event["amount_usd"] is None:
                liability_unknown = True
                issues.append(_issue("unknown_liability", f"events.{event['event_id']}",
                                     "a liability without a conservative bound blocks"))
            else:
                liability += _usd(event["amount_usd"])
        if rates["fees"] is None:
            liability_unknown = True
            issues.append(_issue("fees_and_tax_unbounded", "quote.fees_and_tax.bound_usd",
                                 "fees and tax exposure has no conservative bound"))
        else:
            liability += rates["fees"]
        if any(i["code"] == "invalid_event" for i in issues):
            return _ledger(declared_scope, input_hashes, issues, None)
        if by_kind["provider_error"]:
            issues.append(_issue(
                "provider_errors_recorded", "events",
                f"{len(by_kind['provider_error'])} provider error(s) recorded; they clear nothing",
                severity="warning"))

        # Nested tiers: this scope's usage and liability count from scope.tier upward.
        base = TIERS.index(context["tier"])
        tiers = {}
        for position, tier in enumerate(TIERS):
            used = sum((_usd(e["amount_usd"]) for e in by_kind["prior_usage"]
                        if TIERS.index(e["counts_toward"]) <= position), Decimal(0))
            owed = Decimal(0)
            if position >= base:
                used += accrued
                owed = liability
            cap = context["caps"][tier]
            headroom = None if liability_unknown else cap - used - owed
            tiers[tier] = {
                "cap_usd": _fmt(cap),
                "used_usd": _fmt(used),
                "liability_usd": None if liability_unknown else _fmt(owed),
                "headroom_usd": _fmt(headroom),
            }
            if (headroom is not None and headroom < 0) or used > cap:
                issues.append(_issue("cap_exceeded", f"tiers.{tier}",
                                     f"{tier} usage plus known liability exceeds its cap"))
        caps = [context["caps"][tier] for tier in TIERS]
        if not caps[0] <= caps[1] <= caps[2]:
            issues.append(_issue("caps_not_nested", "scope.caps_usd",
                                 "caps must satisfy pilot <= stage <= project"))
        used_values = [Decimal(tiers[tier]["used_usd"]) for tier in TIERS]
        if not used_values[0] <= used_values[1] <= used_values[2]:
            issues.append(_issue("budget_usage_not_nested", "tiers", "used is not nested"))

        settlement = "final" if by_kind["settlement_final"] else "pending"
        body = {
            "cash_paid_usd": _fmt(cash),
            "prior_usage_usd": _fmt(prior),
            "usage": {
                "computed_usd": _fmt(computed),
                "provider_recorded_usd": _fmt(recorded),
                "observed_credit_consumption_usd": _fmt(observed),
                "credit_vs_recorded_gap_usd": _fmt(gap),
                "settled_usd": _fmt(settled),
                "accrued_usd": _fmt(accrued),
                "basis": "max(sum over resources of max(quote-derived, latest provider "
                         "snapshot) + transfer, observed credit use, final settlement)",
            },
            "compute": {
                "billed_seconds": compute_seconds,
                "gpu_hours": _fmt(compute_gpu_hours),
                "usd": _fmt(compute_usd),
            },
            "storage": {
                "billed_seconds": storage_seconds,
                "usd": _fmt(storage_usd),
                "open_resources": sorted(open_storage),
            },
            "transfer_usd": _fmt(transfer_usd),
            "credit_remaining_usd": _fmt(credit),
            "credit_status": "not_estimable" if credit is None else "estimated",
            "liabilities_usd": None if liability_unknown else _fmt(liability),
            "liability_status": "not_estimable" if liability_unknown else "bounded",
            "settlement_status": settlement,
            "tier": context["tier"],
            "tiers": tiers,
            "rates": {
                "compute_usd_per_hour": _fmt(rates["compute"]),
                "storage_usd_per_hour": _fmt(rates["storage"]),
                "rate_basis": "per_gpu" if rates["per_gpu"] else "per_instance",
                "billing_granularity_seconds": rates["granularity"],
            },
        }
    return _ledger(declared_scope, input_hashes, issues, body, liability_unknown)


def _ledger(declared_scope, input_hashes, issues, body, unknown=False):
    blocking = any(issue["severity"] == "blocking" and issue["code"] not in (
        "unknown_liability", "fees_and_tax_unbounded", "open_storage_without_retention_bound"
    ) for issue in issues)
    if body is None or blocking:
        status = "blocked"
    elif unknown:
        status = "not_estimable"
    else:
        status = "passed"
    ledger = {
        "artifact_kind": "s2_cost_ledger",
        "schema_version": SCHEMA_VERSION,
        "scope": declared_scope if declared_scope in SCOPES else None,
        "producer": {"module": "gate.s2_cost"},
        "input_hashes": input_hashes,
        "units": {"currency": "USD", "time": "seconds", "gpu_time": "gpu_hours"},
        "status": status,
        "issues": issues,
        "cloud_launch_authorized": False,
    }
    if body is not None:
        ledger.update(body)
    return ledger


def _ledger_figures(ledger):
    """Parse every ledger field the continuation check reads; raise _Invalid if any is bad."""
    try:
        rates, tier = ledger["rates"], ledger["tier"]
        granularity = rates["billing_granularity_seconds"]
        if type(granularity) is not int or granularity <= 0:
            raise _Invalid("ledger billing granularity must be a positive integer")
        if rates["rate_basis"] not in ("per_gpu", "per_instance") or tier not in TIERS:
            raise _Invalid("ledger rate basis or tier unknown")
        tiers = {name: (_usd(ledger["tiers"][name]["cap_usd"]),
                        _usd(ledger["tiers"][name]["used_usd"]))
                 for name in TIERS[TIERS.index(tier):]}
        return {
            "granularity": granularity,
            "per_gpu": rates["rate_basis"] == "per_gpu",
            "compute": _usd(rates["compute_usd_per_hour"]),
            "storage": _usd(rates["storage_usd_per_hour"]),
            "tiers": tiers,
            "liability": _usd(ledger["liabilities_usd"]),
            "accrued": _usd(ledger["usage"]["accrued_usd"]),
        }
    except (KeyError, TypeError) as error:
        raise _Invalid(f"ledger field missing or malformed: {type(error).__name__}") from None


def can_start_next_unit(ledger, limits, next_unit, *, now_epoch):
    """Strict-headroom continuation check for one bounded unit. L1, never authority.

    ``limits``: ``task_allowance_usd`` (decimal string or ``None`` when no task
    allowance is recorded), ``reserves_usd`` (named decimal strings; ``cleanup``
    required) and the original absolute ``deadline_epoch``. ``next_unit``:
    ``upper_bound_usd``, ``max_seconds``, ``cleanup_seconds``, ``gpu_count``. Exposure is the
    maximum of the declared bound and the quote-derived cost of unit plus cleanup.
    Only a complete ``s2_cost_ledger`` of integer schema version 1 with status
    ``passed`` can allow continuation; anything else fails closed with an issue.
    """
    issues, remaining, exposure = [], {}, None
    try:
        if type(limits) is not dict or set(limits) != {
            "task_allowance_usd", "reserves_usd", "deadline_epoch"
        }:
            raise _Invalid("limits keys must match the schema exactly")
        if type(next_unit) is not dict or set(next_unit) != {
            "upper_bound_usd", "max_seconds", "cleanup_seconds", "gpu_count"
        }:
            raise _Invalid("next_unit keys must match the schema exactly")
        reserves = limits["reserves_usd"]
        if type(reserves) is not dict or "cleanup" not in reserves:
            raise _Invalid("a cleanup reserve is required")
        reserve_total = sum((_usd(v) for v in reserves.values()), Decimal(0))
        allowance = limits["task_allowance_usd"]
        allowance = None if allowance is None else _usd(allowance)
        deadline = _epoch(limits["deadline_epoch"])
        now = _epoch(now_epoch)
        declared = _usd(next_unit["upper_bound_usd"])
        unit_seconds = _epoch(next_unit["max_seconds"])
        cleanup_seconds = _epoch(next_unit["cleanup_seconds"])
        devices = next_unit["gpu_count"]
        if type(devices) is not int or devices < 1:
            raise _Invalid("gpu_count must be a positive integer")
    except _Invalid as error:
        issues.append(_issue("invalid_next_unit_input", "input", str(error)))
        return _decision(issues, remaining, exposure)
    version = ledger.get("schema_version") if type(ledger) is dict else None
    if type(ledger) is not dict or ledger.get("artifact_kind") != "s2_cost_ledger" or type(
        version
    ) is not int or version != SCHEMA_VERSION or "status" not in ledger:
        issues.append(_issue("invalid_ledger", "ledger",
                             "an s2_cost_ledger of schema version 1 is required"))
        return _decision(issues, remaining, exposure)
    status = ledger["status"]
    if status == "not_estimable" or ledger.get("liabilities_usd") is None:
        issues.append(_issue("unknown_charge_exposure", "ledger",
                             "unknown liabilities or fees block continuation"))
    if status == "blocked":
        issues.append(_issue("ledger_blocked", "ledger", "the cost ledger is blocked"))
    elif status not in ("passed", "not_estimable"):
        issues.append(_issue("ledger_not_passed", "ledger.status",
                             "only a passed ledger can allow continuation"))
    if issues:
        return _decision(issues, remaining, exposure)
    try:
        figures = _ledger_figures(ledger)
    except _Invalid as error:
        issues.append(_issue("invalid_ledger", "ledger", str(error)))
        return _decision(issues, remaining, exposure)
    with localcontext(_CONTEXT):
        billed = billed_seconds(unit_seconds + cleanup_seconds, figures["granularity"])
        factor = devices if figures["per_gpu"] else 1
        hourly = figures["compute"] * factor + figures["storage"]
        exposure = max(declared, hourly * billed / 3600)
        liability = figures["liability"]
        for tier, (cap, used) in figures["tiers"].items():
            left = cap - used - liability - reserve_total
            remaining[tier] = _fmt(left)
            if not exposure < left:
                issues.append(_issue(f"{tier}_headroom_exhausted", f"tiers.{tier}",
                                     "next-unit exposure does not fit strictly"))
        if allowance is not None:
            left = allowance - figures["accrued"] - liability - reserve_total
            remaining["task_allowance"] = _fmt(left)
            if not exposure < left:
                issues.append(_issue("task_allowance_headroom_exhausted", "limits",
                                     "next-unit exposure does not fit the task allowance"))
    if now >= deadline:
        issues.append(_issue("deadline_reached", "limits.deadline_epoch",
                             "the original absolute deadline has been reached"))
    elif not now + unit_seconds + cleanup_seconds < deadline:
        issues.append(_issue("deadline_headroom_exhausted", "limits.deadline_epoch",
                             "unit plus cleanup does not finish strictly before the deadline"))
    return _decision(issues, remaining, exposure)


def _decision(issues, remaining, exposure):
    return {
        "artifact_kind": "s2_next_unit_decision",
        "schema_version": SCHEMA_VERSION,
        "can_start_next_unit": not issues,
        "issues": issues,
        "exposure_usd": _fmt(exposure),
        "remaining_usd": remaining,
        "cloud_launch_authorized": False,
        "basis": "strict headroom: exposure < cap - used - liability - reserves "
                 "(agent technical default (D-018)); not provider billing or approval",
    }
