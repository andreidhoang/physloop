"""One-command synthetic S2 rehearsal of the whole CPU pipeline (CPU-13; build-spec B2).

Usage::

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B scripts/rehearse_s2.py \\
        --fixture complete --output evidence/s2-cpu-UNIQUE-ID

Stages, each with literal known answers and at least one negative control that must be
rejected by the intended stage with the intended issue:

1. schedule and packets - the ten-unit pilot packet (``gate.s2_packet``) and the F1 full
   sweep (2 x 441 x 3 = 2,646 units);
2. runtime inventory and bootstrap plan (``gate.s2_runtime``, ``gate.s2_bootstrap``) and
   the fake guest bootstrap (``scripts/s2_guest_bootstrap.py`` with its fake runner),
   which stops at the first unverified output;
3. fake provider lifecycle (``gate.s2_execution.FakeProvider``) from admission to
   verified absence, with the D-020 ``exited`` shape;
4. fake unit runs (``scripts/s2_run_units.py`` with the ``FakeHost`` of
   ``tests/test_s2_run_units.py``): a pre-UUID crash, an internal retry and a failed
   rollout whose Compose exited 0, metered on the cost ledger from a starting budget
   snapshot (plus a control where prior pilot usage leaves less than one worst-case unit);
   a declared diagnostic qualification unit run alone in its own ledger; and the
   telemetry producers wired over the runner's own outputs (review F6):
   ``gate.s2_telemetry_sources.parse_dcgm_scrapes`` on the vendored exporter scrapes,
   ``useful_attempt_rows`` on the runner's ledger and unit records,
   ``allocation_from_journals`` on an invented lifecycle journal, the runner's ledger
   events and its recorded clock anchor, then ``gate.s2_telemetry.summarize_telemetry``;
5. ASL capture and actual-output inspection, synthetic ingestion, validity and population
   diagnostics, repeatability/probe/tensor comparators, telemetry, operational power and
   registered-gate glue (labelled synthetic), cost reconciliation and the next-unit stop;
6. readiness (CPU-02) for a fully linked synthetic case and for the real drafts;
7. seal, verify and fresh-directory replay of the evidence bundle (CPU-11);
8. the S2 report (CPU-13) in JSON and Markdown.

Every provider, runner, clock and port probe is a fake: no real ``vastai``, ``docker``,
``hf``, ``nvidia-smi``, ``uv`` or network call is made and no host is qualified. All
outputs go to the fresh ``--output`` directory (refused when it exists): ``bundle/``
(sealed), ``replay/`` (fresh-directory copy), ``work/`` (fake-host run directories and
journals), ``controls/`` (bundle negative controls), ``report.json``, ``report.md`` and
``rehearsal.json``. The synthetic guest-bootstrap canary secret files are written to a
temporary directory outside ``--output`` and removed when that stage returns. The
bundle carries every CPU-11 minimum role (``gate.s2_bundle.ACTUAL_MINIMUM_ROLES``;
the registration and export receipt are synthetic fixtures, not owner records) and a
source manifest of ``gate/``, ``scripts/``, ``tests/`` (with fixtures), ``pyproject.toml``
and ``uv.lock`` that replay re-hashes; git dirty state is not recorded. The S2.2
``unit_runs`` row cites the runner's run summary with its attempt ledger's own status and
issue codes. Every cited row carries the producer identity read from the cited artifact's
own bytes (review F4-02: ``_Rehearsal.cite``; never passed through by the caller);
artifacts the rehearsal composes itself name ``scripts.rehearse_s2`` as their producer,
and ``gate.s2_report`` passes ``CPU_REHEARSED`` only when every cited row is bound to the
verified bundle. Numbers are fixtures, not measurements; nothing here is a scientific
registration, admission, rental authority or ``HOST_QUALIFIED``.

Verification level: L1 (fakes and fixtures). Exit 0 when every check passed, else 2.
"""

import argparse
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "tests")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from s2_capture_fixture import wire_capture_request
from s2_ingestion_fixture import receipt
from s2_packet_fixture import full_sweep_request, packet_request
from s2_pilot_fixture import build_pilot_fixture

from gate import s2_execution as execution
from gate.s2_actual import inspect_actual_unit
from gate.s2_bootstrap import build_execution_plan
from gate.s2_bundle import ACTUAL_MINIMUM_ROLES, replay_bundle, seal_bundle, verify_bundle
from gate.s2_capture import inspect_capture
from gate.s2_cost import reconcile_cost
from gate.s2_diagnostics import describe_validity
from gate.s2_extract import extract_bundle
from gate.s2_gate import evaluate_registered
from gate.s2_identity import canonical_json_bytes, identity_hash, read_json_bytes
from gate.s2_ledger import reconcile_attempts
from gate.s2_packet import build_full_sweep_packet, build_pilot_packet
from gate.s2_pilot import next_unit_check
from gate.s2_population import audit_population
from gate.s2_power import run_sensitivity, zero_difference_minimum_roads
from gate.s2_preflight import validate_s2_preflight
from gate.s2_probes import compare_probe_artifacts, describe_repeats
from gate.s2_readiness import inspect_readiness
from gate.s2_report import build_s2_report, render_markdown
from gate.s2_runtime import validate_runtime_inventory
from gate.s2_telemetry import summarize_telemetry
from gate.tensor_manifest import build_manifest, compare_manifests

SCHEMA_VERSION = 1
# Source files whose bytes the bundle records (replay is meaningful against these bytes,
# and replay_bundle re-hashes them). Git dirty state is not recorded: the rehearsal never
# starts a process.
SOURCE_GLOBS = ("gate/*.py", "gate/**/*.py", "scripts/*.py", "tests/*.py",
                "tests/fixtures/**/*", "pyproject.toml", "uv.lock")
# Bundle roles beyond the CPU-11 minimum that the rehearsal also carries.
EXTRA_REQUIRED_ROLES = ("raw_artifact", "analysis_input", "analysis_output")


class _Rehearsal:
    def __init__(self, output):
        self.output = output
        self.bundle = output / "bundle"
        self.work = output / "work"
        self.checks = {}
        self.controls = {}
        self.entries = []
        self.evidence = {}

    # --------------------------------------------------------------- recording
    def check(self, name, observed, expected):
        self.checks[name] = {"observed": observed, "expected": expected,
                             "passed": observed == expected}

    def control(self, name, rejected):
        self.controls[name] = rejected is True

    def write(self, relative, value, role, replay=None):
        path = self.bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = canonical_json_bytes(value) + b"\n"
        with path.open("xb") as stream:
            stream.write(data)
        self.entries.append({"path": relative, "role": role, "restriction_class": "public_safe",
                             "licensed_lineage": False, "replay": replay})
        return relative, hashlib.sha256(data).hexdigest()

    def analysis(self, name, function, inputs, kwargs, result, roles=None):
        """Write each input and the output with its replay specification.

        ``roles`` maps an input key to its CPU-11 evidence role (for example the cost
        events are the ``cost`` evidence); other inputs are ``analysis_input``.
        """
        paths = {}
        for key, value in sorted(inputs.items()):
            role = (roles or {}).get(key, "analysis_input")
            paths[key] = self.write(f"inputs/{name}/{key}.json", value, role)[0]
        return self.write(f"outputs/{name}.json", result, "analysis_output", replay={
            "function": function, "inputs": paths, "kwargs": kwargs})

    def cite(self, key, written, kind, status, blockers=()):
        """Cite a written bundle artifact; its producer identity comes from its own bytes.

        Review F4-02: the bytes at ``written[0]`` are re-read, must hash to ``written[1]``,
        and the row's ``producer`` is ``{module, source_sha256}`` (plus ``function`` only
        when the artifact names one) taken from the artifact's own ``producer`` object; a
        caller never passes a producer through. An artifact without a producer object (or
        without a module name) gets no ``producer`` row field, so ``gate.s2_report`` reads
        it ``producer_identity_missing``; a missing ``source_sha256`` is recorded as null
        (unavailable), never invented.
        """
        data = (self.bundle / written[0]).read_bytes()
        if hashlib.sha256(data).hexdigest() != written[1]:
            raise ValueError(f"cited bytes changed: {written[0]}")
        artifact = read_json_bytes(data)
        row = {"path": written[0], "sha256": written[1], "artifact_kind": kind,
               "status": status, "scope": "synthetic", "blockers": sorted(set(blockers))}
        producer = artifact.get("producer") if type(artifact) is dict else None
        if type(producer) is dict and type(producer.get("module")) is str:
            identity = {"module": producer["module"],
                        "source_sha256": producer.get("source_sha256")}
            if "function" in producer:
                identity["function"] = producer["function"]
            row["producer"] = identity
        self.evidence[key] = row


def _source_sha256():
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return None


_SOURCE_SHA256 = _source_sha256()


def _producer(function):
    """This script's identity, embedded in artifacts the rehearsal itself composes."""
    return {"module": "scripts.rehearse_s2", "function": function,
            "source_sha256": _SOURCE_SHA256}


def _codes(report):
    return sorted({item["code"] for item in report.get("issues", [])})


def _blocking_codes(report):
    return sorted({item["code"] for item in report.get("issues", [])
                   if item.get("severity") in ("blocking", "error")})


# ------------------------------------------------------------------- stages


def _packets(r):
    request = packet_request()
    packet = build_pilot_packet(request)
    r.check("pilot_packet_units", len(packet["units"]), 10)
    written = r.analysis("pilot_packet", "build_pilot_packet", {"request": request}, {}, packet)
    r.cite("pilot_packet", written, packet["artifact_kind"],
           "passed" if packet["prepared"] else "blocked", _blocking_codes(packet))
    missing = copy.deepcopy(request)
    missing["execution_order"]["job_ids"] = []
    r.control("pilot_missing_order", "invalid_execution_order" in _codes(
        build_pilot_packet(missing)))
    sweep_request = full_sweep_request()
    sweep = build_full_sweep_packet(sweep_request)
    ports = [unit["generated_config_expectations"]["driver_port"] for unit in sweep["units"]]
    r.check("full_sweep_units", len(sweep["units"]), 2 * 441 * 3)
    r.check("full_sweep_distinct_ports_and_dirs",
            [len(set(ports)), len({unit["compose"]["cwd"] for unit in sweep["units"]})],
            [2646, 2646])
    summary = {key: sweep[key] for key in ("artifact_kind", "schema_version", "scope", "prepared",
                                           "sweep", "request_sha256", "schedule_sha256",
                                           "issues")}
    summary["units_sha256"] = identity_hash(sweep["units"])
    summary["packet_producer"] = sweep["producer"]
    summary["producer"] = _producer("_packets")
    written = r.write("reports/full-sweep-packet-summary.json", summary, "schedule")
    r.cite("full_sweep_packet", written, "s2_full_sweep_packet_summary",
           "passed" if sweep["prepared"] else "blocked", _blocking_codes(sweep))
    short = full_sweep_request()
    short["schedule_request"]["scenes"].pop()
    r.control("full_sweep_440_scenes", "invalid_schedule_request" in _codes(
        build_full_sweep_packet(short)))
    return packet


def _runtime_and_bootstrap(r):
    from test_s2_bootstrap import action
    from test_s2_guest_bootstrap import ENVIRON, FakeRunner
    from test_s2_runtime import fixture_inventory

    from scripts import s2_guest_bootstrap as guest

    inventory = fixture_inventory()
    report = validate_runtime_inventory(inventory)
    r.check("runtime_inventory_status", report["status"], "passed")
    written = r.analysis("runtime_inventory", "validate_runtime_inventory",
                         {"inventory": inventory}, {}, report,
                         roles={"inventory": "runtime_inventory"})
    r.cite("runtime_inventory", written, report["artifact_kind"], report["status"])
    draft = read_json_bytes((ROOT / "experiments/S2_SETUP_DRAFT.json").read_bytes())
    r.control("setup_draft_inventory_blocked",
              validate_runtime_inventory(draft["runtime_inventory"])["status"] == "blocked")
    record = action()
    plan = build_execution_plan(inventory, record, phases=list("ABCDE"))
    r.check("bootstrap_plan_compiled", plan["compiled"], True)
    written = r.write("reports/bootstrap-plan.json", plan, "effective_config")
    r.cite("bootstrap_plan", written, plan["artifact_kind"],
           "passed" if plan["compiled"] else "blocked", _codes(plan))
    r.control("bootstrap_without_authorization",
              build_execution_plan(inventory, action(keys=()), phases=list("ABCDE"))[
                  "compiled"] is False)
    # Review B2-08: the synthetic canary secret files live in a temporary directory
    # outside the output (evidence) tree and are removed when the stage returns.
    with tempfile.TemporaryDirectory(prefix="s2-rehearsal-secrets-") as secret_root:
        secrets = {}
        for name, value in (("hf_token", "synthetic-hf-canary-0123456789abcdef"),
                            ("ngc_api_key", "synthetic-ngc-canary-fedcba9876543210")):
            path = Path(secret_root) / f"{name}.secret"
            path.write_text(value + "\n")
            path.chmod(0o600)
            secrets[name] = str(path)
        runner = FakeRunner()
        result = guest.run_plan(plan, action_record=record, inventory=inventory, runner=runner,
                                journal_dir=str(r.work / "guest-journal"),
                                secret_files=secrets, environ=dict(ENVIRON),
                                clock=lambda: "2026-11-09T00:00:00Z")
    calls = json.dumps(runner.calls, default=str)
    r.check("guest_bootstrap_stops_at_first_unverified_output",
            [result["status"], result.get("failed_stage"), result.get("reason")],
            ["failed", "A1_host_probe", "verification_failed"])
    r.check("guest_bootstrap_secrets_never_reach_a_command",
            any(canary in calls for canary in ("synthetic-hf-canary", "synthetic-ngc-canary",
                                               "parent-env-canary")), False)
    r.write("reports/guest-bootstrap-result.json",
            {"result": {key: result.get(key) for key in ("status", "failed_stage", "reason")},
             "argv": [call["argv"][:2] for call in runner.calls]}, "lifecycle")


def _lifecycle(r):
    from test_s2_execution import Clock, exited_row, order_result, record, row

    provider = execution.FakeProvider({
        "create": [order_result()],
        "show_instance": [row(), exited_row(), execution.NOT_FOUND, execution.NOT_FOUND],
        "stop": ["acknowledged_unverified"], "destroy": ["requested_unverified"],
        "list_volumes": [[], []], "list_instances": [[], [], [], []],
    })
    action_record = record()
    admission = execution.validate_action_record(action_record)
    directory = r.work / "lifecycle"
    directory.mkdir(parents=True)
    # A real local export copy and its verification report (G1: the coordinator never
    # destroys on a bare hash; the report must re-verify from the exported bytes).
    export_root = directory / "export"
    export_root.mkdir()
    (export_root / "attempts.json").write_bytes(b"{}")
    receipts = [{"path": "attempts.json", "sha256": hashlib.sha256(b"{}").hexdigest(),
                 "size_bytes": 2}]
    export_report = execution.export_verification_report(str(export_root), receipts)
    exported = {"kind": "export_verified", "manifest_hash": export_report["manifest_sha256"],
                "verification": export_report}
    r.check("lifecycle_export_reverifies",
            [export_report["status"], execution.reverify_export(exported, str(export_root))],
            ["passed", []])
    events = directory / "watchdog-events.jsonl"
    coordinator = execution.Coordinator(
        directory / "journal.jsonl", provider, Clock(1_800_000_010), record=action_record,
        report=admission, intent_id="0f8e2c1a-5b7d-4c3e-9a10-2b3c4d5e6f70",
        watchdog_events=str(events))
    try:
        for _ in range(3):
            coordinator.step()
        events.write_text(json.dumps({
            "at_utc": "synthetic", "event": "armed", "instance_id": 58628722,
            "expected_label": "physloop-s2-0f8e2c1a-5b7d-4c3e-9a10-2b3c4d5e6f70",
            "deadline_utc_epoch": 1_800_003_600}) + "\n")
        armed = coordinator.observe_watchdog()["armed"]
        coordinator.step()
        coordinator.step()
        for observation in ({"kind": "bootstrap_complete"}, {"kind": "work_complete"},
                            exported):
            coordinator.observe(observation, export_root=str(export_root) if observation[
                "kind"] == "export_verified" else None)
            coordinator.step()
        for _ in range(5):
            coordinator.step()
        final = coordinator.state["state"]
    finally:
        coordinator.close()
    calls = [call[0] for call in provider.calls]
    r.check("lifecycle_reaches_verified_absence", [armed, final], [True, "absent_verified"])
    r.check("lifecycle_provider_calls", calls, [
        "list_instances", "list_instances", "create", "show_instance", "stop", "show_instance",
        "destroy", "show_instance", "list_volumes", "list_instances", "show_instance",
        "list_volumes", "list_instances"])
    try:
        execution.Coordinator(directory / "refused.jsonl", execution.FakeProvider({}),
                              Clock(1_800_000_000), record=record(max_hourly_usd="0"),
                              report=execution.validate_action_record(
                                  record(max_hourly_usd="0")),
                              intent_id="0f8e2c1a-5b7d-4c3e-9a10-2b3c4d5e6f70",
                              watchdog_events=str(events))
        refused = False
    except ValueError:
        refused = True
    r.control("lifecycle_unadmitted_record_refused", refused)
    r.write("reports/export-receipt.json", {
        "artifact_kind": "s2_export_receipt_rehearsal", "scope": "synthetic",
        "observation": exported, "destination_verified_independently": False,
        "note": "Local synthetic export copy verified on CPU; no provider export happened"},
        "export_receipt")
    written = r.write("reports/lifecycle.json", {
        "admission": admission, "final_state": final, "provider_calls": calls,
        "host_qualified": False, "producer": _producer("_lifecycle")}, "lifecycle")
    r.cite("lifecycle", written, "s2_lifecycle_rehearsal",
           "passed" if final == "absent_verified" else "blocked")


def _unit_runs(r):
    from test_s2_run_units import (
        HOST_ENV,
        RUN_CONFIG,
        Clock,
        FakeHost,
        authority,
        budget_ok,
        execute,
        make_packet,
        prior_usage,
        run_kw,
    )

    from scripts.s2_run_units import run_units

    base = r.work / "units"
    base.mkdir(parents=True)
    packet = make_packet(base)
    # The rehearsal's anchor uncertainty is 0 (an invented operator bound for fakes).
    summary, events, _host = execute(base, packet, scenarios={
        1: "pre_uuid_crash", 2: "internal_retry", 3: "failed_exit0"},
        run=run_kw(packet, clock_uncertainty_us=0))
    ledger = summary["ledger"]
    r.check("unit_runs_outcome", [summary["status"], summary["run_outcome"],
                                  len(summary["unit_records"])], ["passed", "completed", 10])
    r.check("unit_runs_attempt_counts",
            [ledger["counts"]["attempts"], ledger["counts"]["completed_attempts"],
             ledger["counts"]["failed_attempts"]], [11, 8, 3])
    r.check("unit_runs_ledger_not_admitted", [ledger["admitted"], _codes(ledger)],
            [False, ["job_not_completed", "retry_selection_unregistered"]])
    r.check("unit_runs_host_env_canary_absent", "hf_CANARY" in json.dumps(summary), False)
    r.check("unit_runs_clock_anchor_recorded",
            [sorted(summary["clock_anchor"]), summary["clock_anchor"]["uncertainty_us"]],
            [["anchor_ref", "clock_id", "monotonic_us", "uncertainty_us", "utc"], 0])
    r.check("unit_runs_budget_from_cost_ledger",
            [summary["budget"]["tier"], summary["budget"]["run_segments"]], ["pilot", 11])
    bad = authority(packet, job_ids=list(reversed(authority(packet)["job_ids"])))
    refused = run_units(packet, authority=bad, run_config=RUN_CONFIG,
                        journal_dir=base / "refused-journal", runner=FakeHost(base / "x"),
                        port_open_fn=lambda port: False, clock=Clock(),
                        budget=budget_ok(packet), host_env=HOST_ENV, **run_kw(packet))
    r.control("unit_runs_authority_mismatch_refused",
              refused["run_outcome"] == "refused" and "authority_jobs_mismatch" in _codes(refused)
              and not (base / "refused-journal").exists())
    # Ledger-fed budget: prior pilot usage $98 leaves less than one worst-case unit.
    stop_base = base / "budget-stop"
    stop_base.mkdir()
    stopped, stop_events, _ = execute(stop_base, packet, budget=budget_ok(
        packet, prior=[prior_usage("98", "pilot")]))
    r.control("unit_runs_ledger_budget_stop",
              stopped["stop_reason"] is not None
              and stopped["stop_reason"]["reason"] == "budget_stop" and stop_events == []
              and stopped["unit_records"][0]["steps"][0]["issues"]
              == ["pilot_headroom_exhausted"])
    unit_ledger = reconcile_attempts(packet["schedule"], events)
    r.analysis("unit_ledger", "reconcile_attempts",
               {"schedule": packet["schedule"], "events": events}, {}, unit_ledger)
    # Review B2-03 / F4-02: the cited artifact is the runner's own run summary (its producer
    # is scripts.s2_run_units); the row carries the summary's attempt-ledger admission and
    # issue codes, never the runner's engineering "passed".
    written = r.write("reports/unit-run-summary.json", summary, "attempts")
    r.cite("unit_runs", written, summary["artifact_kind"],
           "passed" if ledger["admitted"] is True else "blocked",
           _codes(ledger) if ledger["admitted"] is not True else ())
    return packet, summary, events


def _diagnostic_units(r):
    """A declared diagnostic qualification unit runs alone, in its own ledger."""
    from test_s2_run_units import execute, make_packet, run_kw

    base = r.work / "diagnostics"
    base.mkdir(parents=True)
    packet = make_packet(base, diagnostics=[(0, {})])
    summary, events, _ = execute(base, packet,
                                 run=run_kw(packet, unit_selection="diagnostics_only"))
    r.check("diagnostic_unit_separate_from_pilot_jobs",
            [summary["run_outcome"], len(summary["diagnostic_unit_records"]),
             len(summary["unit_records"]), events, summary["diagnostic_ledger"]["admitted"],
             summary["ledger"]["counts"]["scheduled_jobs"]],
            ["completed", 1, 0, [], True, 10])
    r.check("diagnostic_unit_budget_line",
            summary["diagnostic_unit_records"][0]["steps"][0]["decision"]["budget_line"],
            {"line_id": "pilot_diagnostics", "cap_usd": "5", "prior_spent_usd": "0",
             "run_spent_usd": "0", "spent_usd": "0", "remaining_usd": "5"})
    r.write("reports/diagnostic-unit-run-summary.json", summary, "attempts")


TELEMETRY_UUID = "GPU-e7664bde-fdf4-a368-400a-7f1d6a3d9f57"  # tests/fixtures/s2_telemetry_sources


def _transition(state):
    return {"type": "transition", "state": {"state": state}, "action": None,
            "observation": None, "issues": []}


def _telemetry_sources(r, packet, summary, events):
    """Review F6: DCGM scrapes -> useful rows -> allocation partition -> summary."""
    from datetime import datetime, timedelta

    from gate.s2_execution import LifecycleJournal
    from gate.s2_telemetry_sources import (
        allocation_from_journals,
        parse_dcgm_scrapes,
        useful_attempt_rows,
    )

    anchor = summary["clock_anchor"]
    clock_id, start = anchor["clock_id"], anchor["monotonic_us"]
    utc = datetime.fromisoformat(anchor["utc"])
    second = 1_000_000
    # Invented lifecycle around the runner (offsets in microseconds from the anchor; the
    # fake monotonic clock starts near 1 s, so pre-anchor time stays under 1 s): running
    # from 0.1 s before the anchor to 600 s after it, inside a billed interval from
    # -0.9 s to +1,000 s.
    entries = [(-900_000, "admitted"), (-900_000, "create_intent"),
               (-899_000, "create_requested"), (-890_000, "bound"), (-885_000, "armed"),
               (-300_000, "bootstrapping"), (-100_000, "running"), (600 * second, "exporting"),
               (650 * second, "export_verified"), (660 * second, "stopping"),
               (700 * second, "stopped"), (710 * second, "destroying"),
               (1000 * second, "absent_verified")]
    times = iter([(utc + timedelta(microseconds=offset)).isoformat()
                  for offset, _ in entries])
    path = r.work / "telemetry-lifecycle.jsonl"
    with LifecycleJournal(str(path), wall=lambda: next(times)) as journal:
        for _, state in entries:
            journal.append(_transition(state))
    lifecycle = LifecycleJournal.read(str(path))
    billed = {"allocation_id": "synthetic-allocation", "clock_id": clock_id, "gpu_count": 1,
              "devices": ["gpu0"], "start_us": start - 900_000,
              "end_us": start + 1000 * second, "source_ref": "synthetic billed interval"}
    rows = useful_attempt_rows(summary["ledger"], summary["unit_records"], clock_id=clock_id,
                               scope="synthetic")
    partition = allocation_from_journals(lifecycle, events, billed_interval=billed,
                                         clock_anchor=anchor, attempt_rows=rows["rows"],
                                         scope="synthetic")
    fixtures = ROOT / "tests" / "fixtures" / "s2_telemetry_sources"
    scrapes = parse_dcgm_scrapes(
        [{"t_us": start + offset * second, "text": (fixtures / name).read_text(), "ref": name}
         for offset, name in ((10, "sample-0.prom"), (11, "sample-1.prom"))],
        device_key="UUID", device_map={TELEMETRY_UUID: "gpu0"}, clock_id=clock_id,
        scope="synthetic")
    clock = {"clock_id": clock_id, "clock_kind": "monotonic", "clock_scope": "host",
             "utc_anchor": anchor["utc"]}
    telemetry = summarize_telemetry(scrapes["samples"], partition["telemetry_allocation"],
                                    rows["rows"], clock_metadata=clock,
                                    max_sample_gap_us=2 * second, scope="synthetic")
    # Independent oracle from the ledger events: each job spans its first to last event.
    jobs = [unit["job"]["job_id"] for unit in packet["units"]]
    spans = {job: max(e["timestamp_us"] for e in events if e["job_id"] == job)
             - min(e["timestamp_us"] for e in events if e["job_id"] == job) for job in jobs}
    useful = sum(spans[jobs[i]] for i in (0, 4, 5, 6, 7, 8, 9))
    r.check("telemetry_useful_rows",
            [rows["counts"][key] for key in ("attempts", "useful_true", "useful_false",
                                             "useful_unknown")], [11, 7, 3, 1])
    r.check("telemetry_partition_sums_to_billed",
            sum(partition["totals_us"].values()), billed["end_us"] - billed["start_us"])
    r.check("telemetry_partition_unit_phases",
            [partition["totals_us"]["useful_rollout"], partition["totals_us"]["retry_or_failed"],
             partition["totals_us"]["unknown"]],
            [useful, spans[jobs[1]] + spans[jobs[3]], spans[jobs[2]]])
    r.check("telemetry_dcgm_scrapes", [scrapes["status"], len(scrapes["scrapes"])],
            ["passed", 2])
    # The retried job waits for Huy's retry-selection rule, so its unit time is unknown
    # and the summary reports the partition gap instead of passing.
    r.check("telemetry_summary_reports_retry_gap",
            [telemetry["status"], "partition_gap" in _codes(telemetry)], ["blocked", True])
    r.write("reports/telemetry-sources.json", {
        "producer": _producer("_telemetry_sources"), "useful_rows": rows,
        "partition": partition, "scrapes": scrapes, "summary": telemetry}, "telemetry")


def _capture_and_actual(r):
    from test_s2_actual import BINDING, make_run

    root = r.work / "capture"
    root.mkdir(parents=True)
    request = wire_capture_request(root)
    capture = inspect_capture(root, request)
    torn = copy.deepcopy(request)
    torn["artifacts"][0]["asl"] = receipt(root, "torn.asl",
                                          (root / "capture.asl").read_bytes()[:-1])
    torn_report = inspect_capture(root, torn)
    decoded = capture["captures"][0]["inspection"]
    r.check("capture_inspection", [capture["capture_inspection_passed"],
                                   decoded["observations"]["pose_event_count"],
                                   capture["score_admitted"], capture["runtime_qualified"]],
            [True, 4, False, False])
    r.control("capture_torn_frame", not torn_report["capture_inspection_passed"])
    # gate.s2_capture's report carries no producer object; the rehearsal composes this
    # artifact around it and names itself (and the function whose output it holds).
    written = r.write("reports/capture.json", {
        "producer": _producer("_capture_and_actual"),
        "content": {"module": "gate.s2_capture", "function": "inspect_capture"},
        "inspection": capture}, "raw_artifact_index")
    r.cite("capture", written, "s2_capture_inspection",
           "passed" if capture["capture_inspection_passed"] else "blocked")
    actual_root = r.work / "actual"
    actual_root.mkdir(parents=True)
    run, receipts = make_run(actual_root)
    actual = inspect_actual_unit(run, receipts, BINDING, scope="synthetic")
    r.check("actual_inspection_never_admits", [actual["score_admitted"],
                                               actual["diagnostic_only"]], [False, True])
    r.check("actual_inspection_reports_partial_streams",
            {"folders_without_complete", "required_streams_missing"} <= set(_codes(actual)), True)
    written = r.write("reports/actual-inspection.json", actual, "extraction")
    r.cite("actual_inspection", written, actual["artifact_kind"], actual["status"])


def _ingestion_and_validity(r):
    from test_s2_population import manifest, roles

    raw = r.bundle / "raw-ingestion"
    raw.mkdir(parents=True)
    request = build_pilot_fixture(raw)
    for path in sorted(raw.rglob("*")):
        if path.is_file():
            r.entries.append({"path": path.relative_to(r.bundle).as_posix(),
                              "role": "raw_artifact", "restriction_class": "public_safe",
                              "licensed_lineage": False, "replay": None})
    extracted = extract_bundle(raw, request)
    records = [row for batch in extracted["batches"] for row in batch["normalization"]["records"]]
    r.check("ingestion_admitted_records", [extracted["admitted"], len(records)], [True, 10])
    r.check("ingestion_scores_and_deviations",
            [sorted({row["value"] for row in records}), sorted({row["deviation_m"]
                                                               for row in records})],
            [[0.8], [4.2]])
    missing = copy.deepcopy(request)
    missing["artifacts"].pop()
    rejected = extract_bundle(raw, missing)
    r.control("ingestion_missing_last_artifact", not rejected["admitted"]
              and "missing_attempt_artifacts" in _codes(rejected))
    written = r.write("reports/ingestion.json", {
        "producer": _producer("_ingestion_and_validity"),
        "content": {"module": "gate.s2_extract", "function": "extract_bundle"},
        "extraction": extracted}, "extraction")
    r.cite("ingestion", written, "s2_extraction", "passed" if extracted["admitted"]
           else "blocked")
    validity = describe_validity(extracted["ledger"], extracted["observations"],
                                 extracted["batches"], scope="synthetic")
    r.check("validity_deviation_bins", validity["deviation_bins"]["counts"],
            {"[0,1)": 0, "[1,2)": 0, "[2,4)": 0, "[4,inf)": 10, "unknown": 0})
    r.check("validity_never_admits", [validity["admitted_verdict"], validity["status"]],
            [None, "passed"])
    written = r.analysis("validity", "describe_validity", {
        "ledger_report": extracted["ledger"], "observations": extracted["observations"],
        "records": extracted["batches"]}, {"scope": "synthetic"}, validity)
    r.cite("validity", written, validity["artifact_kind"], validity["status"])
    population = audit_population(manifest(), roles())
    r.check("population_decisions_blocked_without_rule",
            [population["status"], population["decisions"]["code"]],
            ["passed", "overlap_rule_unregistered"])
    written = r.analysis("population", "audit_population",
                         {"manifest": manifest(), "roles": roles()}, {}, population,
                         roles={"manifest": "population"})
    r.cite("population", written, population["artifact_kind"], population["status"],
           [population["decisions"]["code"]])


def _probes_tensors_telemetry(r):
    from test_s2_probes import F32, obs, probe
    from test_s2_telemetry import CLOCK, S, allocation, rollout
    from test_tensor_manifest import FILE, SHA_A

    rows = [obs(0, rep="r0"), obs(1, rep="r1"), obs(0, rep="r2")]
    repeats = describe_repeats(rows, scope="synthetic")
    r.check("repeats_method_unregistered",
            [repeats["status"], repeats["descriptive_status"],
             "method_unregistered" in _codes(repeats)], ["blocked", "passed", True])
    written = r.analysis("repeats", "describe_repeats", {"observations": rows},
                         {"scope": "synthetic"}, repeats)
    r.cite("repeats", written, repeats["artifact_kind"], repeats["status"],
           ["method_unregistered"])
    layout = {"action": {"step": 0, "role": "x", "units": "m"}}
    first = probe([("action", "F32", [2], F32(1.0, 2.0))], layout)
    same = compare_probe_artifacts(first, copy.deepcopy(first), "fixed_input_policy")
    other = compare_probe_artifacts(first, probe([("action", "F32", [2], F32(1.0, 2.0))],
                                                 layout, input_sha="b" * 64),
                                    "fixed_input_policy")
    r.check("probe_identical", same["outcome"], "identical")
    r.control("probe_different_inputs_not_comparable", other["outcome"] == "not_comparable")
    written = r.write("reports/probe-comparison.json", same, "diagnostics")
    r.cite("probes", written, same["artifact_kind"], same["status"])
    manifest = build_manifest({"model.safetensors": FILE}, scope="synthetic")
    comparison = compare_manifests(manifest, copy.deepcopy(manifest))
    r.check("tensor_manifest_known_hash", [manifest["status"], SHA_A in json.dumps(manifest),
                                           comparison["tensors_identical"]],
            ["passed", True, True])
    written = r.write("reports/tensor-manifest-comparison.json",
                      {"manifest": manifest, "comparison": comparison,
                       "producer": _producer("_probes_tensors_telemetry")}, "diagnostics")
    r.cite("tensor_manifest", written, comparison["artifact_kind"], comparison["status"])
    arguments = {"samples": [], "allocation": allocation(),
                 "rollouts": [rollout(i) for i in range(4)]}
    kwargs = {"clock_metadata": CLOCK, "max_sample_gap_us": 2 * S, "scope": "synthetic"}
    telemetry = summarize_telemetry(arguments["samples"], arguments["allocation"],
                                    arguments["rollouts"], **kwargs)
    work = telemetry["useful_work"]
    r.check("telemetry_useful_work", [telemetry["allocation"]["gpu_hours"]["exact"],
                                      work["per_wall_hour"]["exact"],
                                      work["per_gpu_hour"]["exact"]], ["1", "8", "4"])
    written = r.analysis("telemetry", "summarize_telemetry", arguments, kwargs, telemetry,
                         roles={"samples": "telemetry", "allocation": "telemetry",
                                "rollouts": "telemetry"})
    r.cite("telemetry", written, telemetry["artifact_kind"], telemetry["status"])


def _power_gate_cost(r):
    from test_s2_cost import quote, scope
    from test_s2_gate import SCORE_METRIC, paired, registration
    from test_s2_power import continuous, design, roads

    power_design, scenarios = design({"urban": roads(3)}), [continuous()]
    power = run_sensitivity(power_design, scenarios, 20, 7)
    outcomes = power["scenarios"][0]["outcomes"]
    r.check("power_three_roads_insufficient",
            [outcomes["INSUFFICIENT"]["count"], outcomes["GO"]["count"],
             zero_difference_minimum_roads(0.05, 0.10)], [20, 0, 36])
    written = r.analysis("power", "run_sensitivity",
                         {"design": power_design, "scenarios": scenarios},
                         {"n_sim": 20, "seed": 7}, power)
    r.cite("power", written, power["artifact_kind"], power["status"])
    records, reg = paired(40), registration()
    gate = evaluate_registered(records, reg, scope="synthetic", adapter_metric=SCORE_METRIC)
    r.check("synthetic_gate_forty_equal_roads", [gate["status"], gate["verdict"],
                                                 gate["scope"]], ["passed", "GO", "synthetic"])
    r.control("gate_unregistered_margin_blocks", evaluate_registered(
        records, registration(margin=None), scope="synthetic",
        adapter_metric=SCORE_METRIC)["status"] == "blocked")
    written = r.analysis("gate", "evaluate_registered", {"records": records,
                                                         "registration": reg},
                         {"scope": "synthetic", "adapter_metric": SCORE_METRIC}, gate,
                         roles={"registration": "registration"})
    r.cite("gate", written, gate["artifact_kind"], gate["status"])
    events = [
        {"event_id": "o", "kind": "opening_credit", "amount_usd": "0", "as_of_epoch": 0},
        {"event_id": "d", "kind": "cash_deposit", "amount_usd": "10", "at_epoch": 1},
        {"event_id": "s", "kind": "provider_charge_snapshot", "resource_id": "i-1",
         "at_epoch": 2, "amount_usd": "3"},
    ]
    ledger = reconcile_cost(events, quote(), scope())
    r.check("cost_deposit_and_usage", [ledger["cash_paid_usd"], ledger["usage"]["accrued_usd"],
                                       ledger["credit_remaining_usd"]], ["10", "3", "7"])
    written = r.analysis("cost", "reconcile_cost", {"events": events, "quote": quote(),
                                                    "scope": scope()}, {}, ledger,
                         roles={"events": "cost", "quote": "cost"})
    r.cite("cost", written, ledger["artifact_kind"], ledger["status"])
    budget = {
        "scopes": {name: {"cap_usd": "10", "used_usd": "6", "liability_usd": "1",
                          "reserve_usd": "1"} for name in ("pilot", "stage", "project")},
        "next_unit_usd": "1.99", "now_epoch": 100, "deadline_epoch": 200,
        "next_unit_seconds": 60, "cleanup_seconds": 20,
    }
    affordable = next_unit_check(budget)
    r.check("next_unit_affordable", affordable["can_start_next_unit"], True)
    r.control("next_unit_equality_boundary",
              next_unit_check({**budget, "next_unit_usd": "2"})["can_start_next_unit"] is False)
    r.control("next_unit_deadline", next_unit_check({**budget, "now_epoch": 120})[
        "can_start_next_unit"] is False)
    r.analysis("next_unit", "next_unit_check", {"request": budget}, {}, affordable)


def _readiness(r):
    from test_s2_readiness import complete_case

    evidence_root = r.bundle / "readiness"
    evidence_root.mkdir(parents=True)
    plan, index, schedule, record = complete_case(evidence_root)
    for entry in index["entries"]:
        r.entries.append({"path": f"readiness/{entry['path']}", "role": "review",
                          "restriction_class": "public_safe", "licensed_lineage": False,
                          "replay": None})
    report = inspect_readiness(plan, index, schedule, record, root=evidence_root)
    r.check("readiness_synthetic_complete_case",
            [report["status"], report["sections"]["host_qualification"]["status"],
             report["cloud_launch_authorized"]], ["passed", "qualify_on_host", False])
    drafts = copy.deepcopy(plan)
    drafts["scientific_manifest"] = read_json_bytes(
        (ROOT / "experiments/S2_PILOT_DRAFT.json").read_bytes())
    draft_report = inspect_readiness(drafts, index, schedule, record, root=evidence_root)
    r.control("readiness_real_pilot_draft_blocked",
              draft_report["sections"]["scientific_admission"]["status"] == "blocked")
    closed = copy.deepcopy(record)
    closed["approval"]["decision_id"] = "D-019"
    r.control("readiness_closed_d019", "closed_authority_reused" in {
        item["code"] for item in inspect_readiness(plan, index, schedule, closed,
                                                   root=evidence_root)[
            "sections"]["action_scope"]["issues"]})
    preflight = validate_s2_preflight(drafts["scientific_manifest"])
    r.control("scientific_draft_not_ready", preflight["ready_for_planning"] is False)
    r.analysis("pilot_draft_preflight", "validate_s2_preflight",
               {"manifest": drafts["scientific_manifest"]}, {}, preflight)
    written = r.write("reports/readiness.json", report, "review")
    r.cite("readiness", written, report["artifact_kind"], report["status"])
    return report


def _source_manifest(r):
    files = sorted({path for pattern in SOURCE_GLOBS for path in ROOT.glob(pattern)
                    if path.is_file()})
    r.write("source/source-manifest.json", {
        "artifact_kind": "s2_source_manifest", "schema_version": 1, "scope": "synthetic",
        "files": [{"path": path.relative_to(ROOT).as_posix(),
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "size_bytes": path.stat().st_size} for path in files],
        "note": "byte identities of the working tree used; includes uncommitted changes"},
        "source_snapshot")


def _seal_and_replay(r):
    index = {"artifact_kind": "s2_bundle_index", "schema_version": 1, "scope": "synthetic",
             "bundle_id": "s2-cpu-rehearsal",
             # Review B2-10: the full CPU-11 actual-scope minimum, rehearsed on CPU.
             "required_roles": [*ACTUAL_MINIMUM_ROLES, *EXTRA_REQUIRED_ROLES],
             "entries": sorted(r.entries, key=lambda entry: entry["path"])}
    sealed = seal_bundle(r.bundle, index)
    r.check("bundle_sealed", [sealed["sealed"], _codes(sealed)], [True, []])
    verified = verify_bundle(r.bundle)
    r.check("bundle_verified", [verified["verified"], verified["seal_state"]], [True, "sealed"])
    replay = replay_bundle(r.bundle, r.output / "replay")
    mismatched = [row["path"] for row in replay["analyses"] if not row["identical"]]
    r.check("bundle_replay_identical", [replay["replay_identical"], mismatched], [True, []])
    r.check("bundle_replayed_analyses", len(replay["analyses"]), 12)
    comparison = replay["source_comparison"] or {}
    r.check("bundle_replay_source_identical",
            [comparison.get("status"), comparison.get("changed"), comparison.get("missing")],
            ["identical", [], []])
    controls = r.output / "controls"
    for name, data, expected in (("secret", b"JUPYTER_TOKEN=fake-jupyter-token\n",
                                  "secret_canary_detected"),
                                 ("changed", b"hello\n", "changed_bytes")):
        root = controls / name
        root.mkdir(parents=True)
        (root / "log.txt").write_bytes(data)
        control_index = {**index, "required_roles": [], "bundle_id": f"control-{name}",
                         "entries": [{"path": "log.txt", "role": "lifecycle",
                                      "restriction_class": "internal",
                                      "licensed_lineage": False, "replay": None}]}
        report = seal_bundle(root, control_index)
        if name == "changed":
            (root / "log.txt").write_bytes(b"HELLO\n")
            report = verify_bundle(root)
        r.control(f"bundle_{name}", expected in _codes(report))
    return verified, replay


def run_rehearsal(output):
    """Run every stage into a fresh ``output`` directory; return the rehearsal summary."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    output = output.resolve(strict=True)
    r = _Rehearsal(output)
    r.bundle.mkdir()
    r.work.mkdir()
    _packets(r)
    _runtime_and_bootstrap(r)
    _lifecycle(r)
    packet, unit_summary, unit_events = _unit_runs(r)
    _diagnostic_units(r)
    _telemetry_sources(r, packet, unit_summary, unit_events)
    _capture_and_actual(r)
    _ingestion_and_validity(r)
    _probes_tensors_telemetry(r)
    _power_gate_cost(r)
    readiness = _readiness(r)
    _source_manifest(r)
    verified, replay = _seal_and_replay(r)
    counts = {"pilot_units": 10, "pilot_attempts": r.checks["unit_runs_attempt_counts"][
        "observed"][0], "full_sweep_units": r.checks["full_sweep_units"]["observed"],
        "bundle_entries": verified["entries_checked"],
        "replayed_analyses": len(replay["analyses"])}
    passed = all(item["passed"] for item in r.checks.values()) and all(r.controls.values())
    summary = {"scope": "synthetic", "passed": passed, "negative_controls": dict(r.controls),
               "replay_identical": replay["replay_identical"], "counts": counts}
    report = build_s2_report(scope="synthetic", evidence=dict(r.evidence), readiness=readiness,
                             bundle_verification=verified, rehearsal=summary)
    r.check("report_milestones", {name: value["status"]
                                  for name, value in report["milestones"].items()}, {
        "CPU_REHEARSED": "passed", "PACKET_COMPLETE": "not_applicable",
        "HOST_QUALIFIED": "qualify_on_host", "ACTUAL_INPUT_QUALIFIED": "qualify_on_host",
        "S2_ACCEPTED": "not_run", "S2_STOPPED_OR_INSUFFICIENT": "not_run"})
    # Review F4-02 control: one cited row without its producer identity is unbound, and
    # an unbound row keeps CPU_REHEARSED from passing.
    stripped = copy.deepcopy(r.evidence)
    stripped["unit_runs"].pop("producer", None)
    unbound = build_s2_report(scope="synthetic", evidence=stripped, readiness=readiness,
                              bundle_verification=verified, rehearsal=summary)
    r.control("report_unbound_row_blocks_cpu_rehearsed",
              unbound["milestones"]["CPU_REHEARSED"]["status"] == "blocked"
              and "unit_runs (producer_identity_missing)"
              in unbound["milestones"]["CPU_REHEARSED"]["reason"])
    r.check("report_every_cited_row_bound",
            [report["evidence_binding"]["bound"], report["evidence_binding"]["unverified"]],
            [len(r.evidence), 0])
    unit_runs = report["items"]["S2.2"]
    r.check("report_unit_ledger_blocked",
            [unit_runs["engineering_status"],
             "retry_selection_unregistered" in unit_runs["engineering_blockers"]],
            ["blocked", True])
    with (output / "report.json").open("xb") as stream:
        stream.write(canonical_json_bytes(report) + b"\n")
    with (output / "report.md").open("x", encoding="utf-8") as stream:
        stream.write(render_markdown(report))
    passed = all(item["passed"] for item in r.checks.values()) and all(r.controls.values())
    result = {
        "artifact_kind": "s2_cpu_rehearsal", "schema_version": SCHEMA_VERSION,
        "scope": "synthetic", "passed": passed,
        "status": "passed" if passed else "blocked",
        "checks": r.checks, "negative_controls": r.controls, "counts": counts,
        "failed_checks": sorted(name for name, item in r.checks.items() if not item["passed"]),
        "unrejected_controls": sorted(name for name, ok in r.controls.items() if not ok),
        "bundle_manifest_sha256": verified["manifest_sha256"],
        "replay_identical": replay["replay_identical"],
        "milestones": {name: value["status"] for name, value in report["milestones"].items()},
        "runtime_qualified": False, "host_qualified": False, "cloud_launch_authorized": False,
        "real_tools_started": False,
        "note": "Synthetic fixtures and fakes only; not HOST_QUALIFIED, not an admission",
    }
    with (output / "rehearsal.json").open("xb") as stream:
        stream.write(canonical_json_bytes(result) + b"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", choices=("complete",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_rehearsal(args.output)
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc), "cloud_launch_authorized": False}))
        return 2
    print(json.dumps({key: result[key] for key in (
        "passed", "failed_checks", "unrejected_controls", "counts", "milestones",
        "host_qualified")}))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
