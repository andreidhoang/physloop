"""CLI known answers for scripts/s2_lifecycle.py (build-spec A3).

Provider access goes only through an injected factory returning ``FakeProvider``;
no test runs a real ``vastai`` or touches the network.
"""

import json
from pathlib import Path

import pytest

from gate import s2_execution as ex
from scripts import s2_lifecycle as cli
from tests.test_s2_execution import ID, INTENT, LABEL, WINDOW_END, WINDOW_START, record


class Clock:
    def __init__(self, now):
        self.now = now

    def time(self):
        return self.now


def write(path, value):
    path.write_text(json.dumps(value))
    return str(path)


def run(argv, capsys, **kwargs):
    code = cli.main(argv, **kwargs)
    out = capsys.readouterr().out
    return code, json.loads(out) if out.strip() else None


def lifecycle_args(tmp_path):
    return ["--journal", str(tmp_path / "journal.jsonl"),
            "--record", write(tmp_path / "record.json", record()),
            "--intent-id", INTENT, "--watchdog-events", str(tmp_path / "watchdog.jsonl")]


def test_validate_action_record_exit_codes(tmp_path, capsys):
    code, report = run(["validate-action-record", write(tmp_path / "r.json", record())], capsys)
    assert code == 0 and report["admitted"] is True
    code, report = run(["validate-action-record",
                        write(tmp_path / "b.json", record(max_hourly_usd="0"))], capsys)
    assert code == 1 and report["admitted"] is False
    (tmp_path / "dup.json").write_text('{"a": 1, "a": 2}')
    code, error = run(["validate-action-record", str(tmp_path / "dup.json")], capsys)
    assert code == 2 and error == {"error": "input_not_strict_json"}


def test_reconcile_cost_and_next_unit(tmp_path, capsys):
    events = [{"event_id": "o", "kind": "opening_credit", "amount_usd": "0", "as_of_epoch": 0},
              {"event_id": "d", "kind": "cash_deposit", "amount_usd": "10", "at_epoch": 1},
              {"event_id": "s", "kind": "provider_charge_snapshot", "resource_id": "i",
               "at_epoch": 2, "amount_usd": "3"}]
    quote = {"artifact_kind": "s2_cost_quote", "schema_version": 1, "scope": "synthetic",
             "currency": "USD", "provider": "vast", "rate_basis": "per_gpu",
             "compute_usd_per_hour": "0", "storage_usd_per_hour": "0",
             "transfer_usd_per_gb": {"download": "0", "upload": "0"},
             "billing_granularity_seconds": 1,
             "fees_and_tax": {"bound_usd": "0", "basis": "synthetic"}}
    scope = {"tier": "pilot", "as_of_epoch": 10,
             "caps_usd": {"pilot": "100", "stage": "1200", "project": "2500"},
             "storage_retention_until_epoch": None}
    ledger_path = tmp_path / "ledger.json"
    code, ledger = run(["reconcile-cost", "--events", write(tmp_path / "e.json", events),
                        "--quote", write(tmp_path / "q.json", quote),
                        "--scope", write(tmp_path / "s.json", scope)], capsys)
    assert code == 0
    assert (ledger["cash_paid_usd"], ledger["credit_remaining_usd"]) == ("10", "7")
    write(ledger_path, ledger)
    limits = {"task_allowance_usd": "10", "reserves_usd": {"cleanup": "1"},
              "deadline_epoch": 1000}
    unit = {"upper_bound_usd": "6", "max_seconds": 10, "cleanup_seconds": 10, "gpu_count": 1}
    code, decision = run(["next-unit", "--ledger", str(ledger_path),
                          "--limits", write(tmp_path / "l.json", limits),
                          "--unit", write(tmp_path / "u.json", unit), "--now", "0"], capsys)
    # Allowance 10 - accrued 3 - liability 0 - reserve 1 = 6; a $6 unit fails strictly.
    assert code == 1 and decision["can_start_next_unit"] is False


def test_step_refuses_without_execute_or_matching_authority(tmp_path, capsys):
    calls = []

    def factory():
        calls.append(1)
        return ex.FakeProvider({})

    args = lifecycle_args(tmp_path)
    code, error = run(["step", *args], capsys, provider_factory=factory,
                      clock=Clock(WINDOW_START + 10))
    assert code == 2 and error == {"error": "execution_requires_execute_and_authority"}
    code, error = run(["step", *args, "--execute", "--authority-ref", "D-001"], capsys,
                      provider_factory=factory, clock=Clock(WINDOW_START + 10))
    assert code == 2 and error == {"error": "authority_ref_does_not_match_record"}
    assert calls == []
    assert not (tmp_path / "journal.jsonl").exists()


def test_step_refuses_unadmitted_record(tmp_path, capsys):
    args = lifecycle_args(tmp_path)
    args[3] = write(tmp_path / "record.json", record(max_hourly_usd="0"))
    code, error = run(["step", *args, "--execute", "--authority-ref", "D-999"], capsys,
                      provider_factory=lambda: pytest.fail("provider constructed"),
                      clock=Clock(WINDOW_START + 10))
    assert code == 2 and error == {"error": "action_record_not_admitted"}


def test_step_observe_and_status_drive_the_fake_lifecycle(tmp_path, capsys):
    provider = ex.FakeProvider({"list_instances": [[], []],
                                "create": [{"outcome": "created", "instance_id": ID}]})
    args = lifecycle_args(tmp_path)
    execute = ["--execute", "--authority-ref", "D-999"]
    for _ in range(3):
        code, decision = run(["step", *args, *execute], capsys,
                             provider_factory=lambda: provider, clock=Clock(WINDOW_START + 10))
        assert code == 0
    assert decision["state"]["state"] == "bound"
    assert [call[0] for call in provider.calls] == ["list_instances", "list_instances",
                                                    "create"]
    events = tmp_path / "watchdog.jsonl"  # the plan's --watchdog-events path
    events.write_text(json.dumps({"event": "armed", "instance_id": ID,
                                  "expected_label": LABEL,
                                  "deadline_utc_epoch": WINDOW_END}) + "\n")
    code, receipt = run(["observe-watchdog", *args], capsys)
    assert code == 0 and receipt["armed"] is True
    code, status = run(["status", "--journal", str(tmp_path / "journal.jsonl"),
                        "--now", str(WINDOW_START + 10)], capsys)
    assert code == 0
    assert status["state"]["state"] == "bound"
    assert status["pending_observation"]["kind"] == "watchdog_receipt"
    assert status["next"]["state"]["state"] == "armed"
    obs = write(tmp_path / "obs.json", {"kind": "anything"})
    code, error = run(["observe", *args, "--observation", obs], capsys)
    assert code == 2 and error == {"error": "invalid_observation"}


def test_status_on_tampered_journal_fails_closed(tmp_path, capsys):
    path = tmp_path / "journal.jsonl"
    path.write_text('{"seq": 0}\n')
    code, error = run(["status", "--journal", str(path)], capsys)
    assert code == 2 and error == {"error": "journal_record_shape"}


def test_intent_id_is_a_canonical_uuid(capsys):
    code = cli.main(["intent-id"])
    value = capsys.readouterr().out.strip()
    assert code == 0
    assert ex.initial_state(record(), ex.validate_action_record(record()), value,
                            watchdog_events="w")["label"] == f"physloop-s2-{value}"


def test_charges_route_is_gated_and_emits_snapshot_events(tmp_path, capsys):
    from tests.test_s2_execution import EVIDENCE

    rows = json.loads((EVIDENCE / "charges-after-delete.json").read_text(),
                      parse_float=str)["task_rows"]
    args = lifecycle_args(tmp_path)
    window = ["--start", "1793491200", "--end", "1793577600"]
    code, error = run(["charges", *args, *window], capsys,
                      provider_factory=lambda: pytest.fail("provider constructed"),
                      clock=Clock(WINDOW_START + 10))
    assert code == 2 and error == {"error": "execution_requires_execute_and_authority"}
    provider = ex.FakeProvider({"list_instances": [[], []],
                                "create": [{"outcome": "created", "instance_id": ID}],
                                "charges": [rows]})
    execute = ["--execute", "--authority-ref", "D-999"]
    code, error = run(["charges", *args, *window, *execute], capsys,
                      provider_factory=lambda: provider, clock=Clock(WINDOW_START + 10))
    assert code == 2 and error == {"error": "journal_missing"}
    assert not (tmp_path / "journal.jsonl").exists()
    run(["step", *args, *execute], capsys, provider_factory=lambda: provider,
        clock=Clock(WINDOW_START + 10))
    code, error = run(["charges", *args, *window, *execute], capsys,
                      provider_factory=lambda: provider, clock=Clock(WINDOW_START + 10))
    assert code == 2 and error == {"error": "no_bound_instance"}
    for _ in range(2):
        run(["step", *args, *execute], capsys, provider_factory=lambda: provider,
            clock=Clock(WINDOW_START + 10))
    code, events = run(["charges", *args, *window, *execute], capsys,
                       provider_factory=lambda: provider, clock=Clock(WINDOW_START + 20))
    assert code == 0
    assert events == [{"event_id": f"charge-{ID}-{WINDOW_START + 20}",
                       "kind": "provider_charge_snapshot", "resource_id": str(ID),
                       "at_epoch": WINDOW_START + 20, "amount_usd": "0.6"}]
    assert provider.calls[-1] == ("charges", 1793491200, 1793577600)


# ------------------------------------------------------------ review regressions (A3)


def plan_args(tmp_path, rec=None):
    return ["--journal", str(tmp_path / "journal.jsonl"),
            "--record", write(tmp_path / "record.json", rec or record()),
            "--intent-id", INTENT, "--watchdog-events", str(tmp_path / "plan-events.jsonl")]


def to_bound(tmp_path, capsys, args):
    provider = ex.FakeProvider({"list_instances": [[], []],
                                "create": [{"outcome": "created", "instance_id": ID}]})
    for _ in range(3):
        code, decision = run(["step", *args, "--execute", "--authority-ref", "D-999"], capsys,
                             provider_factory=lambda: provider, clock=Clock(WINDOW_START + 10))
        assert code == 0
    assert decision["state"]["state"] == "bound"


def test_a3_02_observe_cli_refuses_a_forged_watchdog_receipt(tmp_path, capsys):
    args = plan_args(tmp_path)
    to_bound(tmp_path, capsys, args)
    forged = write(tmp_path / "forged.json", {"kind": "watchdog_receipt", "armed": True,
                                              "instance_id": ID, "label": LABEL,
                                              "deadline_epoch": WINDOW_END})
    code, error = run(["observe", *args, "--observation", forged], capsys)
    assert code == 2 and error == {"error": "invalid_observation"}
    code, status = run(["status", "--journal", str(tmp_path / "journal.jsonl")], capsys)
    assert status["pending_observation"] is None


def test_a3_02_observe_watchdog_reads_the_plan_events_path_only(tmp_path, capsys):
    args = plan_args(tmp_path)
    to_bound(tmp_path, capsys, args)
    armed = json.dumps({"event": "armed", "instance_id": ID, "expected_label": LABEL,
                        "deadline_utc_epoch": WINDOW_END}) + "\n"
    (tmp_path / "elsewhere.jsonl").write_text(armed)
    with pytest.raises(SystemExit):  # --events is no longer accepted
        cli.main(["observe-watchdog", *args, "--events", str(tmp_path / "elsewhere.jsonl")])
    capsys.readouterr()
    code, receipt = run(["observe-watchdog", *args], capsys)
    assert code == 0 and receipt["armed"] is False
    run(["step", *args, "--execute", "--authority-ref", "D-999"], capsys,
        provider_factory=lambda: ex.FakeProvider({}), clock=Clock(WINDOW_START + 10))
    (tmp_path / "plan-events.jsonl").write_text(armed)
    code, receipt = run(["observe-watchdog", *args], capsys)
    assert code == 0 and receipt["armed"] is True


def test_a3_11_status_is_read_only(tmp_path, capsys):
    missing = tmp_path / "never.jsonl"
    code, error = run(["status", "--journal", str(missing)], capsys)
    assert code == 2 and error == {"error": "journal_missing"}
    assert not missing.exists()
    path = tmp_path / "ro.jsonl"
    with ex.LifecycleJournal(path) as journal:
        journal.append({"type": "transition", "state": {"state": "admitted"}, "action": None,
                        "observation": None, "issues": []})
    path.chmod(0o444)
    try:
        code, status = run(["status", "--journal", str(path)], capsys)
    finally:
        path.chmod(0o644)
    assert code == 0 and status["state"] == {"state": "admitted"}


def test_a3_11_charges_reads_the_journal_without_a_write_lock(tmp_path, capsys):
    from tests.test_s2_execution import EVIDENCE

    rows = json.loads((EVIDENCE / "charges-after-delete.json").read_text(),
                      parse_float=str)["task_rows"]
    args = plan_args(tmp_path)
    to_bound(tmp_path, capsys, args)
    rec = record()
    holder = ex.Coordinator(tmp_path / "journal.jsonl", None, Clock(WINDOW_START + 10),
                            record=rec, report=ex.validate_action_record(rec),
                            intent_id=INTENT, watchdog_events=str(tmp_path / "plan-events.jsonl"))
    try:
        code, events = run(["charges", *args, "--start", "1", "--end", "2", "--execute",
                            "--authority-ref", "D-999"], capsys,
                           provider_factory=lambda: ex.FakeProvider({"charges": [rows]}),
                           clock=Clock(WINDOW_START + 20))
    finally:
        holder.close()
    assert code == 0 and events[0]["resource_id"] == str(ID)
    other = plan_args(tmp_path, record(max_hourly_usd="2.40"))
    code, error = run(["charges", *other, "--start", "1", "--end", "2", "--execute",
                       "--authority-ref", "D-999"], capsys,
                      provider_factory=lambda: pytest.fail("provider constructed"),
                      clock=Clock(WINDOW_START + 20))
    assert code == 2 and error == {"error": "journal_or_intent_mismatch"}


def _no_real_cli(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("real provider path reached")

    monkeypatch.setattr(ex, "subprocess_runner", refuse)
    monkeypatch.setattr(ex, "VastCliClient", refuse)


@pytest.mark.parametrize("command", ["step", "charges"])
def test_a3_13_real_provider_path_refuses_non_actual_scope(tmp_path, capsys, monkeypatch,
                                                           command):
    _no_real_cli(monkeypatch)
    window = ["--start", "1", "--end", "2"] if command == "charges" else []
    code, error = run([command, *plan_args(tmp_path), *window, "--execute",
                       "--authority-ref", "D-999"], capsys, clock=Clock(WINDOW_START + 10))
    assert code == 2 and error == {"error": "real_provider_requires_actual_scope"}
    assert not (tmp_path / "journal.jsonl").exists()


def test_a3_13_real_provider_path_requires_a_recorded_decision(tmp_path, capsys, monkeypatch):
    _no_real_cli(monkeypatch)
    decisions = tmp_path / "DECISIONS.md"
    decisions.write_text("- **D-019 · 2026-11-01 · Bounded rental.** text\n")
    monkeypatch.setattr(cli, "DECISIONS_PATH", decisions)
    args = plan_args(tmp_path, record(scope="actual"))
    code, error = run(["step", *args, "--execute", "--authority-ref", "D-999"], capsys,
                      clock=Clock(WINDOW_START + 10))
    assert code == 2 and error == {"error": "authority_decision_not_recorded"}
    assert cli.decision_recorded("D-019", decisions) is True
    assert cli.decision_recorded("D-01", decisions) is False
    assert not (tmp_path / "journal.jsonl").exists()


# ------------------------------------------------------------ re-verification residuals (A3R)


@pytest.mark.parametrize("mode,expected", [(0o444, "journal_unwritable"),
                                           (0o000, "journal_unreadable")])
@pytest.mark.parametrize("command", ["observe", "observe-watchdog", "step"])
def test_a3r_03_unwritable_or_unreadable_journal_is_a_static_error(tmp_path, capsys, command,
                                                                   mode, expected):
    args = plan_args(tmp_path)
    to_bound(tmp_path, capsys, args)
    journal = tmp_path / "journal.jsonl"
    before = journal.read_bytes()
    extra = {"observe": ["--observation", write(tmp_path / "o.json", {"kind": "work_complete"})],
             "step": ["--execute", "--authority-ref", "D-999"]}.get(command, [])
    journal.chmod(mode)
    try:
        code, error = run([command, *args, *extra], capsys,
                          provider_factory=lambda: pytest.fail("provider constructed"),
                          clock=Clock(WINDOW_START + 10))
    finally:
        journal.chmod(0o644)
    assert code == 2 and error == {"error": expected}
    assert journal.read_bytes() == before


# ------------------------------------------------------------ A3X: failed work


def test_a3x_observe_journals_work_failed_and_status_exports(tmp_path, capsys):
    from tests.test_s2_execution import to_state

    args = plan_args(tmp_path)
    running = to_state("running")
    running["plan"]["watchdog_events"] = str(tmp_path / "plan-events.jsonl")
    with ex.LifecycleJournal(tmp_path / "journal.jsonl") as journal:
        journal.append({"type": "transition", "state": running, "action": None,
                        "observation": None, "issues": []})
    unreasoned = write(tmp_path / "bad.json", {"kind": "work_failed",
                                               "decision_issue_codes": []})
    code, error = run(["observe", *args, "--observation", unreasoned], capsys)
    assert code == 2 and error == {"error": "invalid_observation"}
    failed = {"kind": "work_failed", "reason": "s4_watchdog:host_exited:1",
              "decision_issue_codes": ["s4_watchdog:host_exited:1", "s4_cleanup:dirty"]}
    code, observed = run(["observe", *args, "--observation",
                          write(tmp_path / "failed.json", failed)], capsys)
    assert code == 0 and observed == failed
    code, status = run(["status", "--journal", str(tmp_path / "journal.jsonl"),
                        "--now", str(WINDOW_START + 100)], capsys)
    assert code == 0 and status["pending_observation"] == failed
    assert status["next"]["state"]["state"] == "exporting"
    assert status["next"]["action"] == {"action": "export"}
    assert "work_failed" in status["next"]["state"]["issues"]
    assert "budget_exhausted" not in status["next"]["state"]["issues"]


# ------------------------------------------------------------ G1: hash-bound authority

WIRING = Path(__file__).resolve().parent / "fixtures" / "s2_authority_wiring"
DECISIONS = WIRING / "decisions_lifecycle.md"
REAL_DECISIONS = Path(__file__).resolve().parents[1] / "docs" / "DECISIONS.md"
# identity_hash(record(scope="actual", approval={"decision_id": <id>, ...})), as written in
# the fixture's authorizes lines.
RECORD_HASHES = {
    "D-019": "9950b2edd06a18ad9c726d1a3b2cb112067ad5e4ee9ff058fcb8052785dc96a3",
    "D-024": "d7ceeb8779d2e119af6cc421eb47c229b3562982541154ae608ca6437895d513",
    "D-030": "e59b6f706e36c63e715f06b73da37af2218b0c4d9ff68ab8fb3871b07b357a89",
    "D-031": "73310c98d542f29c21fa903d26bd55071f2a89379e6ff4d787f975abb5d5091f",
}


def actual(decision_id):
    return record(scope="actual", approval={
        "decision_id": decision_id,
        "quoted_instruction": "Synthetic fixture instruction for tests only."})


def test_g1_fixture_hashes_are_the_canonical_record_identities():
    from gate.s2_identity import identity_hash

    text = DECISIONS.read_text()
    for decision_id, value in RECORD_HASHES.items():
        assert identity_hash(actual(decision_id)) == value
        assert ex.validate_action_record(actual(decision_id))["input_hashes"]["record"] == value
    assert f"authorizes: s2_action_record sha256:{RECORD_HASHES['D-030']}" in text
    assert RECORD_HASHES["D-031"] not in text and RECORD_HASHES["D-024"] not in text


@pytest.mark.parametrize("command", ["step", "charges"])
def test_g1_correct_hash_line_authorizes_the_real_provider_path(tmp_path, capsys, monkeypatch,
                                                                command):
    provider = ex.FakeProvider({"list_instances": [[]]})
    monkeypatch.setattr(ex, "subprocess_runner", lambda *a, **k: pytest.fail("real CLI"))
    monkeypatch.setattr(ex, "VastCliClient", lambda: provider)  # the 'real' factory
    args = plan_args(tmp_path, actual("D-030"))
    if command == "charges":
        code, error = run(["charges", *args, "--start", "1", "--end", "2", "--execute",
                           "--authority-ref", "D-030"], capsys,
                          clock=Clock(WINDOW_START + 10), decisions_path=DECISIONS)
        assert code == 2 and error == {"error": "journal_missing"}  # past the gate
        return
    code, decision = run(["step", *args, "--execute", "--authority-ref", "D-030"], capsys,
                         clock=Clock(WINDOW_START + 10), decisions_path=DECISIONS)
    assert code == 0 and decision["state"]["state"] == "create_intent"
    assert provider.calls == [("list_instances",)]


@pytest.mark.parametrize("decision_id, expected", [
    ("D-019", "authority_closed"),        # closed by D-020 even with an exact hash line
    ("D-024", "authority_not_verified"),  # no authorizes line at all
    ("D-031", "authority_not_verified"),  # its line binds another record's hash
    ("D-040", "authority_decision_not_recorded"),
])
@pytest.mark.parametrize("command", ["step", "charges"])
def test_g1_fixture_refusals_construct_no_provider(tmp_path, capsys, monkeypatch, decision_id,
                                                  expected, command):
    _no_real_cli(monkeypatch)
    window = ["--start", "1", "--end", "2"] if command == "charges" else []
    code, error = run([command, *plan_args(tmp_path, actual(decision_id)), *window,
                       "--execute", "--authority-ref", decision_id], capsys,
                      clock=Clock(WINDOW_START + 10), decisions_path=DECISIONS)
    assert code == 2 and error == {"error": expected}
    assert not (tmp_path / "journal.jsonl").exists()


@pytest.mark.parametrize("decision_id, expected", [("D-019", "authority_closed"),
                                                   ("D-024", "authority_not_verified")])
def test_g1_real_decision_log_authorizes_no_rental(tmp_path, capsys, monkeypatch, decision_id,
                                                   expected):
    """The committed docs/DECISIONS.md (read, never edited) authorizes no action record."""
    _no_real_cli(monkeypatch)
    assert REAL_DECISIONS.is_file()
    code, error = run(["step", *plan_args(tmp_path, actual(decision_id)), "--execute",
                       "--authority-ref", decision_id], capsys, clock=Clock(WINDOW_START + 10))
    assert code == 2 and error == {"error": expected}
    assert not (tmp_path / "journal.jsonl").exists()


def test_g1_unreadable_decision_log_is_refused(tmp_path, capsys, monkeypatch):
    _no_real_cli(monkeypatch)
    code, error = run(["step", *plan_args(tmp_path, actual("D-030")), "--execute",
                       "--authority-ref", "D-030"], capsys, clock=Clock(WINDOW_START + 10),
                      decisions_path=tmp_path / "absent.md")
    assert code == 2 and error == {"error": "decisions_unreadable"}


def test_g1_fake_provider_path_is_unchanged_by_the_authority_gate(tmp_path, capsys):
    # An injected provider never reads the decision log (dry/fake path unchanged).
    code, decision = run(["step", *plan_args(tmp_path), "--execute", "--authority-ref",
                          "D-999"], capsys, provider_factory=lambda: ex.FakeProvider(
                              {"list_instances": [[]]}), clock=Clock(WINDOW_START + 10),
                         decisions_path=tmp_path / "absent.md")
    assert code == 0 and decision["state"]["state"] == "create_intent"


def exporting_journal(tmp_path, rec):
    """A journal in ``exporting`` that belongs to ``rec`` (its record hash and plan)."""
    from tests.test_s2_execution import to_state

    events = str(tmp_path / "plan-events.jsonl")
    init = ex.initial_state(rec, ex.validate_action_record(rec), INTENT, watchdog_events=events)
    exporting = to_state("exporting")
    exporting.update(record_hash=init["record_hash"], plan=init["plan"])
    with ex.LifecycleJournal(tmp_path / "journal.jsonl") as journal:
        journal.append({"type": "transition", "state": exporting, "action": None,
                        "observation": None, "issues": []})
    return plan_args(tmp_path, rec)


def test_g1_observe_export_verified_requires_a_reproducible_report(tmp_path, capsys):
    from tests.test_s2_execution import RECEIPTS, exported

    root = tmp_path / "export"
    args = exporting_journal(tmp_path, record(export_destination=str(root)))
    obs = write(tmp_path / "exported.json", exported())
    code, error = run(["observe", *args, "--observation", obs], capsys)
    assert code == 2 and error == {"error": "export_root_required"}
    root.mkdir()
    (root / RECEIPTS[0]["path"]).write_bytes(b"[]")  # other bytes than the report describes
    code, error = run(["observe", *args, "--observation", obs, "--export-root", str(root)],
                      capsys)
    assert code == 2 and error == {"error": "export_not_reverified"}
    code, status = run(["status", "--journal", str(tmp_path / "journal.jsonl")], capsys)
    assert status["pending_observation"] is None
    (root / RECEIPTS[0]["path"]).write_bytes(b"{}")
    code, observed = run(["observe", *args, "--observation", obs, "--export-root", str(root)],
                         capsys)
    assert code == 0 and observed["verification"]["status"] == "passed"


# ------------------------------------------------------------ G1 review regressions


def test_g1_03_step_executes_the_record_bytes_it_authorized(tmp_path, capsys, monkeypatch):
    """Review G1-03: a record path that serves other bytes on a second read never executes."""
    from gate.s2_identity import identity_hash

    authorized = actual("D-030")
    other = dict(actual("D-030"), offer={"offer_id": 77777777, "sku": "other offer"},
                 max_hourly_usd="2.40")
    assert ex.validate_action_record(other)["admitted"]
    assert identity_hash(other) not in DECISIONS.read_text()
    args = plan_args(tmp_path, authorized)
    record_path = args[args.index("--record") + 1]
    reads = []
    real_read = cli._read

    def alternating(path):  # A on the first read of the record, B on every later read
        if path != record_path:
            return real_read(path)
        reads.append(path)
        return authorized if len(reads) == 1 else other

    provider = ex.FakeProvider({"list_instances": [[], []],
                                "create": [{"outcome": "created", "instance_id": ID}]})
    monkeypatch.setattr(ex, "subprocess_runner", lambda *a, **k: pytest.fail("real CLI"))
    monkeypatch.setattr(ex, "VastCliClient", lambda: provider)
    monkeypatch.setattr(cli, "_read", alternating)
    for _ in range(3):
        reads.clear()
        code, _ = run(["step", *args, "--execute", "--authority-ref", "D-030"], capsys,
                             clock=Clock(WINDOW_START + 10), decisions_path=DECISIONS)
        assert code == 0 and len(reads) == 1  # one read per command
    assert ("create", authorized["offer"]["offer_id"], authorized["resources"]["image"],
            authorized["resources"]["disk_gb"], LABEL) in provider.calls
    assert all(call[1] != 77777777 for call in provider.calls if call[0] == "create")
    state, _ = ex.replay_records(ex.LifecycleJournal.read(tmp_path / "journal.jsonl"))
    assert state["record_hash"] == RECORD_HASHES["D-030"]


def test_g1_01_cli_export_must_be_the_records_own_destination(tmp_path, capsys):
    from tests.test_s2_execution import exported

    destination = tmp_path / "export"
    args = exporting_journal(tmp_path, record(export_destination=str(destination)))
    elsewhere = tmp_path / "elsewhere"
    for root in (destination, elsewhere):
        root.mkdir()
        (root / "attempts.json").write_bytes(b"{}")
    obs = write(tmp_path / "exported.json", exported())
    code, error = run(["observe", *args, "--observation", obs, "--export-root",
                       str(elsewhere)], capsys)
    assert code == 2 and error == {"error": "export_root_not_record_destination"}
    code, observed = run(["observe", *args, "--observation", obs, "--export-root",
                          str(destination)], capsys)  # control: the record's destination
    assert code == 0 and observed["kind"] == "export_verified"


def test_g1_01_synthetic_export_never_destroys_an_actual_rental(tmp_path, capsys, monkeypatch):
    """Review G1-01/G1-04: the reviewer's CLI path (hand-sealed empty synthetic bundle)."""
    from tests.test_s2_execution import _hand_sealed_empty_bundle, exported

    destination = tmp_path / "export"
    rec = actual("D-031")
    rec["export_destination"] = str(destination)
    args = exporting_journal(tmp_path, rec)
    report = _hand_sealed_empty_bundle(destination)
    assert report["status"] == "passed" and report["scope"] == "synthetic"
    obs = write(tmp_path / "obs.json", {"kind": "export_verified",
                                       "manifest_hash": report["manifest_sha256"],
                                       "verification": report})
    code, error = run(["observe", *args, "--observation", obs, "--export-root",
                       str(destination)], capsys)
    assert code == 2 and error == {"error": "export_not_bound_to_record"}
    (destination / "attempts.json").write_bytes(b"{}")
    code, error = run(["observe", *args, "--observation", write(
        tmp_path / "env.json", exported()), "--export-root", str(destination)], capsys)
    assert code == 2 and error == {"error": "export_not_bound_to_record"}  # envelope: no roles
    state, pending = ex.replay_records(ex.LifecycleJournal.read(tmp_path / "journal.jsonl"))
    assert state["state"] == "exporting" and pending is None


class _NotFake:
    """An injected adapter that is not the in-memory FakeProvider (a real client stand-in)."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, *args))
            raise AssertionError("injected real adapter reached")
        return call


def test_g1_12_injected_non_fake_provider_passes_the_real_gate(tmp_path, capsys):
    adapter = _NotFake()
    first, second = tmp_path / "synthetic", tmp_path / "actual"
    first.mkdir()
    second.mkdir()
    code, error = run(["step", *plan_args(first), "--execute", "--authority-ref", "D-999"],
                      capsys, provider_factory=lambda: adapter,
                      clock=Clock(WINDOW_START + 10), decisions_path=DECISIONS)
    assert code == 2 and error == {"error": "real_provider_requires_actual_scope"}
    code, error = run(["step", *plan_args(second, actual("D-024")), "--execute",
                       "--authority-ref", "D-024"], capsys, provider_factory=lambda: adapter,
                      clock=Clock(WINDOW_START + 10), decisions_path=DECISIONS)
    assert code == 2 and error == {"error": "authority_not_verified"}
    assert adapter.calls == []
    for directory in (first, second):  # at most the admitted genesis; no intent journaled
        records = ex.LifecycleJournal.read(directory / "journal.jsonl")
        assert [r["payload"]["state"]["state"] for r in records] == ["admitted"]
    # charges: refused at the same gate once the journal has a bound instance
    provider = ex.FakeProvider({"list_instances": [[], []],
                                "create": [{"outcome": "created", "instance_id": ID}]})
    third = tmp_path / "charges"
    third.mkdir()
    args = plan_args(third)
    for _ in range(3):
        run(["step", *args, "--execute", "--authority-ref", "D-999"], capsys,
            provider_factory=lambda: provider, clock=Clock(WINDOW_START + 10))
    code, error = run(["charges", *args, "--start", "1", "--end", "2", "--execute",
                       "--authority-ref", "D-999"], capsys, provider_factory=lambda: adapter,
                      clock=Clock(WINDOW_START + 10))
    assert code == 2 and error == {"error": "real_provider_requires_actual_scope"}
    assert adapter.calls == []
