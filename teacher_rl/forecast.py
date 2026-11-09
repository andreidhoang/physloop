"""S4 cost forecast: the protocol-design section 8 formula with cash/usage/credit separation.

Tickets: protocol-design section 8 (cost and admission packet), E18 (cost boundary), build-spec
B3 item (4). Serves S4.1-S4.4/S4.6 [C] (G9, G7) under D-024. Agent-authored evaluator
engineering (D-018); an engineering forecast, never a budget approval or rental authority
(``cloud_launch_authorized`` is always ``False``).

Formula (protocol-design section 8, literal)::

    total = setup + training + development_selection + locked_confirmation
            + storage + transfer + fees + failure_reserve

- Each compute component is a list of lines ``{line_id, jobs, replicas, rerun_jobs,
  wall_seconds_per_job, gpu_count, basis}``. Every replica and every failed/rerun job is
  billed: ``billed_jobs = jobs x replicas + rerun_jobs``. Each job is billed separately at
  the quote's granularity (``gate.s2_cost.billed_seconds``; 5430 s at 60 s -> 5460 s), so
  ``usd = billed_jobs x rate x factor x billed_seconds / 3600`` with ``factor = gpu_count``
  for a per-GPU quote. ``gpu_count`` must equal ``host_gpu_count``, which must be
  ``teacher_rl.ops.REQUIRED_GPU_COUNT`` (2, the ``local_disaggregated_2gpu`` topology; review
  B3-13). The input ``scope`` must equal the quote's (review B3-07).
- Storage is billed per instance from create to destroy (retention included) at the
  quote's storage rate; transfer is ``GB x rate`` per direction.
- Schema v2 (module F4; completeness audit "S4 forecast") prices a separate evaluation host,
  for example the locked confirmation on a 1-GPU S2 host with its own quote:
  ``evaluation_host`` is ``null`` or ``{gpu_count >= 1, quote, basis}`` (the quote is checked
  by ``gate.s2_cost.reconcile_cost`` and must have the forecast's scope), and every compute
  line, storage entry and transfer entry (``transfer`` becomes a list) names its ``host``
  (``training`` | ``evaluation``). A line's ``gpu_count`` must equal its host's; setup and
  training lines run on the two-GPU training host only (``training_host_required``); a host
  with compute lines needs at least one storage and one transfer entry (missing is never
  zero). The evaluation quote's fees bound is an additional liability. v1 inputs keep their
  meaning: every line on the training host.
- Fees are the quote's ``fees_and_tax.bound_usd``; the failure reserve is a required
  explicit amount. Neither has a default.

Money separation (``gate.s2_cost.reconcile_cost`` over the supplied *actual* events, the same
ledger the S2 lifecycle uses): the current ledger gives cash paid, usage accrued, unused
credit and liabilities (the quote's fees bound is already one of its liabilities, so fees are
counted once). The projection adds the forecast usage to accrued usage and the failure reserve
to liabilities; prepaid credit absorbs usage first and is never a second payment, so
``additional_cash_needed = max(0, projected usage + liabilities - current credit)``. The
ledger's credit absorbs only training-host usage: evaluation-host usage plus its fees bound is
added as cash (``money.separate_host``), never absorbed by that credit (agent technical
default (D-018): credit is per provider account and the forecast cannot establish that two
quotes share one). The caps are all-in across hosts. Each tier
from the ledger's tier upward must fit strictly: ``used + liability < cap`` (agent technical
default (D-018), the same strict boundary as ``gate.s2_cost.can_start_next_unit``). Tiers
below the ledger's tier are unchanged by the forecast and only need ``used + liability <=
cap``, the bound ``reconcile_cost`` itself accepts (review B3-14).

Missing is never zero: a null measured input, an unbounded fee or a missing failure reserve
gives ``not_estimable`` (total ``None``) naming the field; a structural defect is ``blocked``.

Verification level per function:

- ``forecast``, ``crosscheck_rental``: **L1** (Decimal arithmetic with literal known answers
  and negative controls in ``tests/test_s4_forecast.py``). ``crosscheck_rental`` checks one
  host's lines (``host``, default the training host) against one rental record. The measured inputs themselves
  (setup time, episode wall time, rerun rates) are **L3**: they come from an admitted
  engineering qualification on the approved topology.

Owner inputs (required, never defaulted): the caps, the tier, the failure reserve, the
number of runs/arms/checkpoints (Huy's registration) and every measured duration. This module
never calls a provider, reads a credential, approves spending, funds an account or changes
a cap.
"""

import hashlib
import re
from decimal import ROUND_CEILING, Context, Decimal, InvalidOperation, localcontext
from pathlib import Path

from gate.s2_cost import billed_seconds, gpu_hours, reconcile_cost
from gate.s2_execution import validate_action_record
from gate.s2_identity import identity_hash
from teacher_rl.ops import REQUIRED_GPU_COUNT

ARTIFACT_KIND = "s4_cost_forecast"
INPUT_KIND = "s4_cost_forecast_input"
CROSSCHECK_KIND = "s4_forecast_rental_crosscheck"
SCHEMA_VERSION = 1
INPUT_SCHEMA_VERSIONS = (1, 2)
SCOPES = ("synthetic", "public_source_fixture", "actual")
HOSTS = ("training", "evaluation")
TRAINING_ONLY = ("setup", "training")
COMPUTE_COMPONENTS = ("setup", "training", "development_selection", "locked_confirmation")
COMPONENTS = (*COMPUTE_COMPONENTS, "storage", "transfer", "fees", "failure_reserve")
TIERS = ("pilot", "stage", "project")
_CONTEXT = Context(prec=60, rounding=ROUND_CEILING)
_INPUT_KEYS = {
    "artifact_kind", "schema_version", "scope", "quote", "host_gpu_count", "components",
    "storage", "transfer", "failure_reserve", "ledger_scope", "actual_events",
}
_INPUT_KEYS_V2 = _INPUT_KEYS | {"evaluation_host"}
_LINE_KEYS = {"line_id", "jobs", "replicas", "rerun_jobs", "wall_seconds_per_job", "gpu_count",
              "basis"}
_STORAGE_KEYS = {"line_id", "instances", "lifetime_seconds", "basis"}
_TRANSFER_KEYS = {"download_gb", "upload_gb", "basis"}
_HOST_KEYS = {"gpu_count", "quote", "basis"}
_LINE_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")


class _Invalid(ValueError):
    def __init__(self, code, path, message):
        super().__init__(message)
        self.code, self.path = code, path


def _issue(code, path, message, severity="blocking"):
    return {"code": code, "path": path, "severity": severity, "evidence_refs": [],
            "message": message}


def _fmt(value):
    if value is None:
        return None
    value = value.normalize(_CONTEXT)
    return "0" if value == 0 else format(value, "f")


def _producer():
    return {"module": "teacher_rl.forecast",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _keys(value, keys, path):
    if type(value) is not dict or set(value) != keys:
        raise _Invalid("invalid_input", path, f"{path} needs exactly {sorted(keys)}")
    return value


def _count(value, path, minimum=0):
    if type(value) is not int or value < minimum:
        raise _Invalid("invalid_input", path, f"{path} must be an int >= {minimum}")
    return value


def _decimal(value, path):
    """A decimal string, or None for a missing measurement; floats and ints are refused."""
    if value is None:
        return None
    if type(value) is not str:
        raise _Invalid("invalid_input", path, f"{path} must be a decimal string or null")
    try:
        amount = Decimal(value)
    except InvalidOperation:
        raise _Invalid("invalid_input", path, f"{path} is not a decimal") from None
    if not amount.is_finite() or amount < 0:
        raise _Invalid("invalid_input", path, f"{path} must be finite and >= 0")
    return amount


def _text(value, path):
    if type(value) is not str or not value.strip():
        raise _Invalid("invalid_input", path, f"{path} must be nonempty text")
    return value


def _rates(quote):
    """Quote rates; the quote itself is validated by ``reconcile_cost`` below."""
    try:
        return {
            "compute": Decimal(quote["compute_usd_per_hour"]),
            "storage": Decimal(quote["storage_usd_per_hour"]),
            "download": Decimal(quote["transfer_usd_per_gb"]["download"]),
            "upload": Decimal(quote["transfer_usd_per_gb"]["upload"]),
            "granularity": quote["billing_granularity_seconds"],
            "per_gpu": quote["rate_basis"] == "per_gpu",
            "fees": quote["fees_and_tax"]["bound_usd"],
        }
    except (KeyError, TypeError, InvalidOperation):
        return None


def _empty(scope, hashes, issues, status):
    return {
        "artifact_kind": ARTIFACT_KIND, "schema_version": SCHEMA_VERSION,
        "scope": scope if scope in SCOPES else None, "producer": _producer(),
        "input_hashes": hashes,
        "units": {"currency": "USD", "time": "seconds", "gpu_time": "gpu_hours",
                  "transfer": "GB (decimal)"},
        "status": status, "issues": issues, "level": "L1",
        "formula": "setup + training + development_selection + locked_confirmation + storage "
                   "+ transfer + fees + failure_reserve (protocol-design section 8)",
        "components_usd": None, "total_usd": None, "usage_usd": None, "lines": None,
        "billed_jobs_total": None, "gpu_hours_total": None, "money": None, "tiers": None,
        "quote_rates": None, "hosts": None, "usage_by_host_usd": None,
        "cloud_launch_authorized": False,
    }


def _host_of(entry, path, version, hosts, keys):
    """The entry's host (v1: always training) after an exact key check."""
    if version == 1:
        _keys(entry, keys, path)
        return "training"
    _keys(entry, keys | {"host"}, path)
    host = entry["host"]
    if host not in HOSTS:
        raise _Invalid("invalid_input", f"{path}.host", f"host is one of {list(HOSTS)}")
    if host not in hosts:
        raise _Invalid("evaluation_host_unregistered", f"{path}.host",
                       "a line on the evaluation host needs a registered evaluation_host")
    return host


def _price_line(line, path, hosts, version, issues):
    host = _host_of(line, path, version, hosts, _LINE_KEYS)
    if type(line["line_id"]) is not str or not _LINE_ID.fullmatch(line["line_id"]):
        raise _Invalid("invalid_input", f"{path}.line_id", "line_id is a lowercase slug")
    _text(line["basis"], f"{path}.basis")
    jobs = _count(line["jobs"], f"{path}.jobs")
    replicas = _count(line["replicas"], f"{path}.replicas", 1)
    reruns = _count(line["rerun_jobs"], f"{path}.rerun_jobs")
    gpus = _count(line["gpu_count"], f"{path}.gpu_count", 1)
    host_gpus, rates = hosts[host]["gpu_count"], hosts[host]["rates"]
    if gpus != host_gpus:
        issues.append(_issue("gpu_count_mismatch", f"{path}.gpu_count",
                             f"every job on the {host} host runs on its {host_gpus} GPU(s)"))
    wall = line["wall_seconds_per_job"]
    billed_jobs = jobs * replicas + reruns
    row = {"line_id": line["line_id"], "host": host, "billed_jobs": billed_jobs,
           "gpu_count": gpus, "wall_seconds_per_job": wall, "billed_seconds_per_job": None,
           "usd": None, "gpu_hours": None}
    if wall is None:
        issues.append(_issue("measured_input_missing", f"{path}.wall_seconds_per_job",
                             "measured wall time per job is required (never zero)",
                             severity="not_estimable"))
        return row, None, None
    _count(wall, f"{path}.wall_seconds_per_job")
    billed = billed_seconds(wall, rates["granularity"])
    factor = gpus if rates["per_gpu"] else 1
    with localcontext(_CONTEXT):
        usd = rates["compute"] * factor * billed * billed_jobs / 3600
        hours = gpu_hours(gpus, billed) * billed_jobs
    row.update(billed_seconds_per_job=billed, usd=_fmt(usd), gpu_hours=_fmt(hours))
    return row, usd, hours


class _ByHost:
    """Per-host usage sums; a host becomes None (not estimable) once any input is missing."""

    def __init__(self, hosts):
        self.usage = {host: Decimal(0) for host in hosts}

    def add(self, host, amount):
        if amount is None or self.usage[host] is None:
            self.usage[host] = None
        else:
            self.usage[host] += amount


def _compute_components(data, hosts, version, by_host, issues):
    components = _keys(data["components"], set(COMPUTE_COMPONENTS), "components")
    totals, lines, jobs, hours_total = {}, {}, 0, Decimal(0)
    estimable, used = True, set()
    for name in COMPUTE_COMPONENTS:
        entries = components[name]
        if type(entries) is not list:
            raise _Invalid("invalid_input", f"components.{name}", "a list of lines")
        if not entries:
            raise _Invalid("component_empty", f"components.{name}",
                           "missing is never zero: list a line (jobs 0 to state none)")
        seen, rows, total, ok = set(), [], Decimal(0), True
        for index, line in enumerate(entries):
            path = f"components.{name}[{index}]"
            row, usd, hours = _price_line(line, path, hosts, version, issues)
            if name in TRAINING_ONLY and row["host"] != "training":
                raise _Invalid("training_host_required", f"{path}.host",
                               f"{name} lines run on the {REQUIRED_GPU_COUNT}-GPU training "
                               "host (local_disaggregated_2gpu topology)")
            if row["line_id"] in seen:
                raise _Invalid("duplicate_line_id", path, "line ids are unique per component")
            seen.add(row["line_id"])
            used.add(row["host"])
            rows.append(row)
            jobs += row["billed_jobs"]
            by_host.add(row["host"], usd)
            if usd is None:
                ok = False
            else:
                total += usd
                hours_total += hours
        lines[name] = rows
        totals[name] = total if ok else None  # per component: one gap never hides another
        estimable = estimable and ok
    return totals, lines, jobs, hours_total, estimable, used


def _storage(data, hosts, version, by_host, issues):
    entries = data["storage"]
    if type(entries) is not list or not entries:
        raise _Invalid("component_empty", "storage", "list every instance's storage lifetime")
    total, seen, covered = Decimal(0), set(), set()
    for index, entry in enumerate(entries):
        path = f"storage[{index}]"
        host = _host_of(entry, path, version, hosts, _STORAGE_KEYS)
        _text(entry["basis"], f"{path}.basis")
        if type(entry["line_id"]) is not str or not _LINE_ID.fullmatch(entry["line_id"]) or (
                entry["line_id"] in seen):
            raise _Invalid("duplicate_line_id", f"{path}.line_id", "unique lowercase slug")
        seen.add(entry["line_id"])
        covered.add(host)
        instances = _count(entry["instances"], f"{path}.instances", 1)
        if entry["lifetime_seconds"] is None:
            issues.append(_issue("measured_input_missing", f"{path}.lifetime_seconds",
                                 "storage lifetime create->destroy is required",
                                 severity="not_estimable"))
            total = None
            by_host.add(host, None)
            continue
        lifetime = _count(entry["lifetime_seconds"], f"{path}.lifetime_seconds")
        rates = hosts[host]["rates"]
        with localcontext(_CONTEXT):
            amount = rates["storage"] * instances * billed_seconds(
                lifetime, rates["granularity"]) / 3600
        by_host.add(host, amount)
        if total is not None:
            total += amount
    return total, covered


def _transfer(data, hosts, version, by_host, issues):
    if version == 1:
        entries, prefix = [data["transfer"]], ["transfer"]
    else:
        entries = data["transfer"]
        if type(entries) is not list or not entries:
            raise _Invalid("invalid_input", "transfer",
                           "v2 transfer is a nonempty list of per-host entries")
        prefix = [f"transfer[{index}]" for index in range(len(entries))]
    total, covered, seen = Decimal(0), set(), set()
    for entry, path in zip(entries, prefix, strict=True):
        if version == 1:
            _keys(entry, _TRANSFER_KEYS, path)
            host = "training"
        else:
            host = _host_of(entry, path, version, hosts, _TRANSFER_KEYS | {"line_id"})
            if type(entry["line_id"]) is not str or not _LINE_ID.fullmatch(
                    entry["line_id"]) or entry["line_id"] in seen:
                raise _Invalid("duplicate_line_id", f"{path}.line_id", "unique lowercase slug")
            seen.add(entry["line_id"])
        covered.add(host)
        _text(entry["basis"], f"{path}.basis")
        down = _decimal(entry["download_gb"], f"{path}.download_gb")
        up = _decimal(entry["upload_gb"], f"{path}.upload_gb")
        for name, value in (("download_gb", down), ("upload_gb", up)):
            if value is None:
                issues.append(_issue("measured_input_missing", f"{path}.{name}",
                                     "transfer volume is required (never zero)",
                                     severity="not_estimable"))
        if down is None or up is None:
            total = None
            by_host.add(host, None)
            continue
        rates = hosts[host]["rates"]
        with localcontext(_CONTEXT):
            amount = down * rates["download"] + up * rates["upload"]
        by_host.add(host, amount)
        if total is not None:
            total += amount
    return total, covered


def _hosts(data, version):
    """Training host (the ledger quote) and, in v2, the optional evaluation host."""
    host = _count(data["host_gpu_count"], "host_gpu_count", 1)
    if host != REQUIRED_GPU_COUNT:
        raise _Invalid("host_gpu_count_mismatch", "host_gpu_count",
                       f"the S4 local_disaggregated_2gpu topology runs on exactly "
                       f"{REQUIRED_GPU_COUNT} GPUs")
    rates = _rates(data["quote"])
    if rates is None:
        raise _Invalid("invalid_input", "quote", "quote rates are unreadable")
    hosts = {"training": {"gpu_count": host, "quote": data["quote"], "rates": rates}}
    if version == 1 or data["evaluation_host"] is None:
        return hosts
    entry = _keys(data["evaluation_host"], _HOST_KEYS, "evaluation_host")
    _text(entry["basis"], "evaluation_host.basis")
    gpus = _count(entry["gpu_count"], "evaluation_host.gpu_count", 1)
    quote = entry["quote"]
    if type(quote) is not dict or quote.get("scope") != data["scope"]:
        raise _Invalid("scope_mismatch", "evaluation_host.quote.scope",
                       "the evaluation host's quote must have the forecast's scope")
    check = reconcile_cost([], quote, data["ledger_scope"])
    if any(issue["code"] == "invalid_quote" for issue in check["issues"]):
        raise _Invalid("invalid_evaluation_quote", "evaluation_host.quote",
                       "the evaluation host's quote is not a valid s2_cost_quote v1")
    rates = _rates(quote)
    if rates is None:
        raise _Invalid("invalid_evaluation_quote", "evaluation_host.quote",
                       "evaluation quote rates are unreadable")
    hosts["evaluation"] = {"gpu_count": gpus, "quote": quote, "rates": rates}
    return hosts


def _host_summary(hosts):
    return {name: {"gpu_count": host["gpu_count"], "provider": host["quote"]["provider"],
                   "rate_basis": "per_gpu" if host["rates"]["per_gpu"] else "per_instance",
                   "compute_usd_per_hour": _fmt(host["rates"]["compute"]),
                   "storage_usd_per_hour": _fmt(host["rates"]["storage"])}
            for name, host in hosts.items()}


def forecast(inputs):
    """Price the section 8 formula and project it onto the current cost ledger (L1)."""
    issues = []
    try:
        hashes = {"inputs": identity_hash(inputs)}
    except (TypeError, ValueError):
        hashes = {"inputs": None}
    scope = inputs.get("scope") if type(inputs) is dict else None
    try:
        version = inputs.get("schema_version") if type(inputs) is dict else None
        if type(version) is not int or version not in INPUT_SCHEMA_VERSIONS:
            raise _Invalid("invalid_input", "schema_version",
                           f"only {INPUT_KIND} v{list(INPUT_SCHEMA_VERSIONS)} are known")
        data = _keys(inputs, _INPUT_KEYS if version == 1 else _INPUT_KEYS_V2, "inputs")
        if data["artifact_kind"] != INPUT_KIND:
            raise _Invalid("invalid_input", "artifact_kind", f"only {INPUT_KIND} is known")
        if data["scope"] not in SCOPES:
            raise _Invalid("invalid_input", "scope", f"scope is one of {SCOPES}")
        if type(data["quote"]) is not dict or data["quote"].get("scope") != data["scope"]:
            raise _Invalid("scope_mismatch", "quote.scope",
                           "the quote (and the ledger priced with it) must have the "
                           "forecast's scope")
        current = reconcile_cost(data["actual_events"], data["quote"], data["ledger_scope"])
        if current["status"] == "blocked" or "tiers" not in current:
            codes = sorted({i["code"] for i in current["issues"]})
            raise _Invalid("current_ledger_blocked", "actual_events",
                           f"the current cost ledger is blocked: {codes}")
        hosts = _hosts(data, version)
        rates = hosts["training"]["rates"]
        by_host = _ByHost(hosts)
        totals, lines, jobs, hours, compute_ok, used = _compute_components(
            data, hosts, version, by_host, issues)
        totals["storage"], stored = _storage(data, hosts, version, by_host, issues)
        totals["transfer"], moved = _transfer(data, hosts, version, by_host, issues)
        for host in sorted(used):
            if host not in stored:
                raise _Invalid("storage_missing_for_host", "storage",
                               f"the {host} host runs jobs but has no storage entry")
            if host not in moved:
                raise _Invalid("transfer_missing_for_host", "transfer",
                               f"the {host} host runs jobs but has no transfer entry")
        reserve_entry = _keys(data["failure_reserve"], {"amount_usd", "basis"},
                              "failure_reserve")
        _text(reserve_entry["basis"], "failure_reserve.basis")
        reserve = _decimal(reserve_entry["amount_usd"], "failure_reserve.amount_usd")
    except _Invalid as error:
        issues.append(_issue(error.code, error.path, str(error)))
        return _finish(_empty(scope, hashes, issues, "blocked"))
    if any(issue["severity"] == "blocking" for issue in issues):
        return _finish(_empty(scope, hashes, issues, "blocked"))
    if reserve is None:
        issues.append(_issue("failure_reserve_missing", "failure_reserve.amount_usd",
                             "the failure reserve is an explicit owner amount (no default)",
                             severity="not_estimable"))
    fees = None if rates["fees"] is None else Decimal(rates["fees"])
    if fees is None:
        issues.append(_issue("fees_and_tax_unbounded", "quote.fees_and_tax.bound_usd",
                             "fees and tax need a conservative bound", severity="not_estimable"))
    separate_fees = Decimal(0)
    if "evaluation" in hosts:
        bound = hosts["evaluation"]["rates"]["fees"]
        separate_fees = None if bound is None else Decimal(bound)
        if separate_fees is None:
            issues.append(_issue("fees_and_tax_unbounded",
                                 "evaluation_host.quote.fees_and_tax.bound_usd",
                                 "the evaluation host's fees and tax need a conservative bound",
                                 severity="not_estimable"))
    totals["fees"] = None if fees is None or separate_fees is None else fees + separate_fees
    totals["failure_reserve"] = reserve
    report = _empty(scope, hashes, issues, "passed")
    report.update(lines=lines, billed_jobs_total=jobs,
                  quote_rates={"rate_basis": "per_gpu" if rates["per_gpu"] else "per_instance",
                               "compute_usd_per_hour": _fmt(rates["compute"]),
                               "storage_usd_per_hour": _fmt(rates["storage"])},
                  hosts=_host_summary(hosts),
                  usage_by_host_usd={host: _fmt(value) for host, value in by_host.usage.items()},
                  gpu_hours_total=_fmt(hours) if compute_ok else None,
                  components_usd={name: _fmt(totals[name]) for name in COMPONENTS})
    if any(totals[name] is None for name in COMPONENTS) or current["status"] != "passed":
        if current["status"] != "passed":
            issues.append(_issue("current_ledger_not_estimable", "actual_events",
                                 "the current ledger has unknown liabilities",
                                 severity="not_estimable"))
        report["status"] = "not_estimable"
        return _finish(report)
    usage = sum((totals[n] for n in (*COMPUTE_COMPONENTS, "storage", "transfer")), Decimal(0))
    report["total_usd"] = _fmt(usage + totals["fees"] + reserve)
    report["usage_usd"] = _fmt(usage)
    separate = None if "evaluation" not in hosts else {
        "usage": by_host.usage["evaluation"], "fees": separate_fees}
    _project(report, current, by_host.usage["training"], separate, reserve, issues)
    return _finish(report)


def _project(report, current, usage, separate, reserve, issues):
    """Project training-host usage onto the ledger; a separate host is paid in cash."""
    liability = Decimal(current["liabilities_usd"])
    accrued = Decimal(current["usage"]["accrued_usd"])
    credit = current["credit_remaining_usd"]
    credit = None if credit is None else Decimal(credit)
    extra_usage = Decimal(0) if separate is None else separate["usage"]
    extra_fees = Decimal(0) if separate is None else separate["fees"]
    projected_credit = None if credit is None else credit - usage
    needed = None if credit is None else max(
        Decimal(0), usage + liability + reserve - credit) + extra_usage + extra_fees
    if credit is None:
        issues.append(_issue("credit_not_estimable", "actual_events",
                             "no opening credit: cash needed cannot be separated from credit",
                             severity="not_estimable"))
        report["status"] = "not_estimable"
    report["money"] = {
        "current": {"cash_paid_usd": current["cash_paid_usd"],
                    "accrued_usd": current["usage"]["accrued_usd"],
                    "credit_remaining_usd": current["credit_remaining_usd"],
                    "liabilities_usd": current["liabilities_usd"]},
        "projected": {"accrued_usd": _fmt(accrued + usage),
                      "credit_remaining_usd": _fmt(projected_credit),
                      "liabilities_usd": _fmt(liability + reserve + extra_fees)},
        "separate_host": None if separate is None else {
            "usage_usd": _fmt(extra_usage), "fees_bound_usd": _fmt(extra_fees),
            "cash_needed_usd": _fmt(extra_usage + extra_fees),
            "rule": "the evaluation host is never paid from the ledger's credit (agent "
                    "technical default (D-018))"},
        "additional_cash_needed_usd": _fmt(needed),
        "rule": "prepaid credit absorbs training-host usage first; fees are a current liability "
                "counted once",
    }
    base = TIERS.index(current["tier"])
    tiers = {}
    for position, tier in enumerate(TIERS):
        row = current["tiers"][tier]
        cap, used = Decimal(row["cap_usd"]), Decimal(row["used_usd"])
        owed = Decimal(row["liability_usd"])
        if position >= base:
            used, owed = used + usage + extra_usage, owed + reserve + extra_fees
        headroom = cap - used - owed
        fits = headroom > 0 if position >= base else headroom >= 0
        tiers[tier] = {"cap_usd": row["cap_usd"], "projected_used_usd": _fmt(used),
                       "projected_liability_usd": _fmt(owed),
                       "projected_headroom_usd": _fmt(headroom), "fits": fits}
        if not fits:
            issues.append(_issue(f"forecast_exceeds_{tier}_cap", f"tiers.{tier}",
                                 "projected used + liability must stay strictly below the cap"
                                 if position >= base else
                                 "used + liability of a lower tier exceeds its cap"))
            report["status"] = "blocked"
    report["tiers"] = tiers


def _finish(report):
    report["issues"].sort(key=lambda i: (i["path"], i["code"]))
    for issue in report["issues"]:
        if issue["severity"] == "not_estimable":
            issue["severity"] = "warning"
    return report


def crosscheck_rental(forecast_report, rental_record, *, host="training"):
    """The admitted rental record's GPU count and max rate cover one host's lines (L1)."""
    issues = []
    admission = validate_action_record(rental_record)
    if not admission["admitted"]:
        issues.append(_issue("rental_record_not_admitted", "rental_record",
                             f"{sorted({i['code'] for i in admission['issues']})}"))
    if type(forecast_report) is not dict or forecast_report.get("artifact_kind") != (
            ARTIFACT_KIND) or forecast_report.get("status") != "passed":
        issues.append(_issue("forecast_not_passed", "forecast", "a passed forecast is required"))
    gpus = quoted = None
    if not issues and rental_record["scope"] != forecast_report["scope"]:
        issues.append(_issue("scope_mismatch", "rental_record.scope",
                             "forecast and rental record come from different scopes"))
    if not issues and host not in (forecast_report.get("hosts") or {}):
        issues.append(_issue("host_not_in_forecast", "host",
                             f"the forecast prices no {host!r} host"))
    if not issues:
        gpus = rental_record["resources"]["gpu_count"]
        lines = [row for rows in forecast_report["lines"].values() for row in rows
                 if row["host"] == host]
        if not lines or any(row["gpu_count"] != gpus for row in lines):
            issues.append(_issue("gpu_count_mismatch", "rental_record.resources.gpu_count",
                                 f"the record rents {gpus} GPU(s); the {host} host's forecast "
                                 "lines differ or are absent"))
    report = {
        "artifact_kind": CROSSCHECK_KIND, "schema_version": SCHEMA_VERSION,
        "scope": forecast_report.get("scope") if type(forecast_report) is dict else None,
        "producer": _producer(),
        "input_hashes": {"forecast": _hash(forecast_report), "rental_record": _hash(
            rental_record)},
        "units": {"currency": "USD", "rate": "USD per instance-hour"},
        "status": "blocked", "issues": issues, "gpu_count": gpus,
        "quoted_instance_usd_per_hour": quoted, "cloud_launch_authorized": False,
    }
    if not issues:
        report["quoted_instance_usd_per_hour"] = _quoted_rate(forecast_report, gpus, issues,
                                                              rental_record, host)
    report["status"] = "blocked" if issues else "passed"
    return report


def _hash(value):
    try:
        return identity_hash(value)
    except (TypeError, ValueError):
        return None


def _quoted_rate(forecast_report, gpus, issues, rental_record, host):
    quote = forecast_report["hosts"][host]
    with localcontext(_CONTEXT):
        factor = gpus if quote["rate_basis"] == "per_gpu" else 1
        rate = Decimal(quote["compute_usd_per_hour"]) * factor + Decimal(
            quote["storage_usd_per_hour"])
    if rate > Decimal(rental_record["max_hourly_usd"]):
        issues.append(_issue("quote_exceeds_max_hourly", "rental_record.max_hourly_usd",
                             f"quoted {_fmt(rate)} USD/h exceeds the approved maximum"))
    return _fmt(rate)
