"""Operator CLI for the S2 rental lifecycle, cost ledger and next-unit check.

Build-spec A3 (CPU-10, CPU-08; runbook section 4 steps 1-3, 9-10) under D-018/D-024;
serves S2.2/S2.3/S2.7 [C] (G2, G7). Agent-authored.

Subcommands (verification ceiling):

- ``validate-action-record RECORD`` - L1 pre-create admission; exit 0 admitted,
  1 blocked, 2 unreadable input.
- ``reconcile-cost --events --quote --scope`` - L1 ledger; exit 0 only when passed.
- ``next-unit --ledger --limits --unit --now`` - L1 strict-headroom decision; exit 0
  only when the next unit may start. Never rental authority.
- ``status --journal [--now]`` - read-only replay of the lifecycle journal (no write
  descriptor, never creates, never follows a symlink) and, with ``--now``, the pure
  next decision. Never calls a provider.
- ``observe`` - append one validated external observation (bootstrap, work complete,
  work failed, export, budget, cost). ``work_failed`` needs a static ``reason`` code and
  is journaled as its own cause, never as budget exhaustion. A watchdog receipt is
  refused here. An ``export_verified`` observation must carry its verification report
  and ``--export-root`` (the local export copy), which must resolve to the record's own
  ``export_destination`` (relative destinations resolve against the repository root;
  ``export_root_not_record_destination``); the report must be bound to the record's scope
  (``s2_execution.check_export_binding``: an actual rental needs an actual sealed bundle
  with every minimum role; ``export_not_bound_to_record``), and
  ``s2_execution.reverify_export`` re-runs the report's producer there and the identical
  report is required (``export_not_reverified`` otherwise), so destroy never follows a
  typed hash (G1, review G1-01). ``Coordinator.observe`` re-runs the binding and the
  re-verification itself.
- ``observe-watchdog`` - append the receipt read from the plan's own
  ``--watchdog-events`` file, the path the watchdog was armed with; never another path.
- ``step ... --execute --authority-ref D-NNN`` - execute exactly one lifecycle step
  (the only path here that creates, stops or destroys a resource). It refuses (exit 2,
  no provider constructed) unless ``--execute`` is given, the action record is admitted,
  and ``--authority-ref`` equals the record's approval decision ID. The record is read
  and admitted once; that same object is authorized and executed (review G1-03: a second
  read could execute bytes no decision binds). The real Vast client (no injected
  provider; or an injected provider that is not the in-memory ``FakeProvider``, review
  G1-12) additionally requires the record's ``scope`` to be ``actual``
  and hash-bound recorded authority (G1, L1 gate): the bytes of ``docs/DECISIONS.md``
  (``decisions_path`` is injectable for tests) must pass
  ``gate.s2_authority.verify_authority`` for that decision ID, ``artifact_kind``
  ``s2_action_record`` and the record's canonical ``identity_hash`` - one unclosed entry
  with the line ``authorizes: s2_action_record sha256:<hash>``. A heading alone, a
  closed decision (D-019) or a decision without that exact line (D-024) is refused with
  a static code (``decisions_unreadable``, ``authority_decision_not_recorded``,
  ``authority_closed`` or ``authority_not_verified``). Synthetic or fixture records
  never reach the real CLI. Injected (fake) providers and dry runs are unchanged.
  Against the real Vast CLI this is L3 (``qualify_on_host``).
- ``charges ... --start --end --execute --authority-ref D-NNN`` - read-only provider
  charge query for the bound instances, emitted as cumulative
  ``provider_charge_snapshot`` events for ``reconcile-cost``; the same refusal gate as
  ``step``, hash-bound authority included (it reaches the real provider with the
  operator's credentials although it changes nothing); reads the journal read-only.
  Provider amounts are rounded snapshots; settlement stays pending (D-020).
- ``intent-id`` - print a fresh canonical UUID for a new action intent.

All JSON inputs are parsed strictly (duplicate keys, NaN and coercions rejected).
Outputs are JSON on stdout; errors are ``{"error": <static code>}``; a journal that
cannot be opened for writing or read is ``journal_unwritable`` / ``journal_unreadable``. The CLI never
selects an offer, funds an account, accepts terms, extends a window, or prints raw
provider output or credentials.
"""

import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECISIONS_PATH = ROOT / "docs" / "DECISIONS.md"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gate import s2_cost, s2_execution
from gate.s2_authority import verify_authority
from gate.s2_identity import read_json_bytes


class _CliError(Exception):
    """Static error code for the operator; never wraps provider output."""


def decision_recorded(decision_id, path):
    """True when ``path`` records ``decision_id`` as a ``- **D-NNN · ...`` entry.

    Kept for callers; it is presence only and never authority (``recorded_authority``).
    """
    if type(decision_id) is not str or not re.fullmatch(r"D-\d{3}", decision_id):
        return False
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    return re.search(rf"^- \*\*{re.escape(decision_id)} \u00b7 ", text, re.MULTILINE) is not None


def recorded_authority(decision_id, report, decisions_path):
    """Hash-bound authority for one admitted action record; static error code or None (L1).

    ``report`` is the record's admission report (its ``input_hashes.record`` is the
    canonical ``identity_hash``). Never authenticates who wrote the decision bytes.
    """
    try:
        data = Path(decisions_path).read_bytes()
    except OSError:
        return "decisions_unreadable"
    result = verify_authority(data, decision_id=decision_id,
                              artifact_kind="s2_action_record",
                              artifact_sha256=report["input_hashes"]["record"])
    if result["authorized"]:
        return None
    codes = {issue["code"] for issue in result["issues"]}
    if "decision_not_found" in codes:
        return "authority_decision_not_recorded"
    if result["closed_by"] is not None:
        return "authority_closed"
    return "authority_not_verified"


def _read(path):
    try:
        data = Path(path).read_bytes()
    except OSError:
        raise _CliError("input_unreadable") from None
    try:
        return read_json_bytes(data)
    except ValueError:
        raise _CliError("input_not_strict_json") from None


def _emit(value):
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


class _SystemClock:
    @staticmethod
    def time():
        import time

        return time.time()


def _admitted(args):
    """Read and admit the action record exactly once per command (review G1-03)."""
    record = _read(args.record)
    report = s2_execution.validate_action_record(record)
    if not report["admitted"]:
        raise _CliError("action_record_not_admitted")
    return record, report


def _coordinator(args, provider, clock, record, report):
    """Coordinator over the record already read, admitted and (step) authorized - never a
    second read of ``--record``, so the authorized bytes are the executed bytes (G1-03)."""
    try:
        return s2_execution.Coordinator(
            args.journal, provider, clock, record=record, report=report,
            intent_id=args.intent_id, watchdog_events=args.watchdog_events), record
    except s2_execution.JournalError as error:
        raise _CliError(str(error)) from None
    except ValueError:
        raise _CliError("journal_or_intent_mismatch") from None


def _lifecycle_parser(sub, name, help_text):
    parser = sub.add_parser(name, help=help_text)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--intent-id", required=True)
    parser.add_argument("--watchdog-events", required=True)
    return parser


def _parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-action-record", help="pre-create admission")
    validate.add_argument("record")
    cost = sub.add_parser("reconcile-cost", help="cash/usage/credit/liability ledger")
    for name in ("--events", "--quote", "--scope"):
        cost.add_argument(name, required=True)
    unit = sub.add_parser("next-unit", help="strict-headroom continuation check")
    for name in ("--ledger", "--limits", "--unit"):
        unit.add_argument(name, required=True)
    unit.add_argument("--now", required=True, type=float)
    status = sub.add_parser("status", help="replay the journal (read-only)")
    status.add_argument("--journal", required=True)
    status.add_argument("--now", type=float)
    observe = _lifecycle_parser(sub, "observe", "append one external observation")
    observe.add_argument("--observation", required=True)
    observe.add_argument("--export-root",
                         help="local export copy an export_verified report is re-run on")
    _lifecycle_parser(sub, "observe-watchdog",
                      "append the arm receipt from the plan's watchdog event file")
    step = _lifecycle_parser(sub, "step", "execute one lifecycle step")
    step.add_argument("--execute", action="store_true")
    step.add_argument("--authority-ref")
    charges = _lifecycle_parser(sub, "charges", "cumulative charge snapshot events")
    charges.add_argument("--start", required=True, type=int)
    charges.add_argument("--end", required=True, type=int)
    charges.add_argument("--execute", action="store_true")
    charges.add_argument("--authority-ref")
    sub.add_parser("intent-id", help="print a fresh canonical intent UUID")
    return parser


def _read_journal(path):
    try:
        return s2_execution.LifecycleJournal.read(path)
    except s2_execution.JournalError as error:
        raise _CliError(str(error)) from None


def _destination(value):
    """The record's ``export_destination`` as a local path (relative to the repository)."""
    path = Path(value)
    return os.path.realpath(path if path.is_absolute() else ROOT / path)


def _observation(args, record):
    observation = _read(args.observation)
    if type(observation) is dict and observation.get("kind") == "export_verified":
        if not args.export_root:
            raise _CliError("export_root_required")
        # G1-01: the export copy is the record's own destination, and the report's scope is
        # the record's (an actual rental needs an actual bundle with every minimum role).
        if os.path.realpath(args.export_root) != _destination(record["export_destination"]):
            raise _CliError("export_root_not_record_destination")
        if s2_execution.check_export_binding(observation.get("verification"),
                                             record["scope"]):
            raise _CliError("export_not_bound_to_record")
        if s2_execution.reverify_export(observation, args.export_root):
            raise _CliError("export_not_reverified")
    return observation


def _run(args, provider_factory, clock, *, real_provider=False, decisions_path=None):
    if args.command == "intent-id":
        print(str(uuid.uuid4()))
        return 0
    if args.command == "validate-action-record":
        report = s2_execution.validate_action_record(_read(args.record))
        _emit(report)
        return 0 if report["admitted"] else 1
    if args.command == "reconcile-cost":
        ledger = s2_cost.reconcile_cost(_read(args.events), _read(args.quote), _read(args.scope))
        _emit(ledger)
        return 0 if ledger["status"] == "passed" else 1
    if args.command == "next-unit":
        decision = s2_cost.can_start_next_unit(_read(args.ledger), _read(args.limits),
                                               _read(args.unit), now_epoch=args.now)
        _emit(decision)
        return 0 if decision["can_start_next_unit"] else 1
    if args.command == "status":
        state, pending = s2_execution.replay_records(_read_journal(args.journal))
        if state is None:
            raise _CliError("journal_empty")
        result = {"state": state, "pending_observation": pending}
        if args.now is not None:
            try:
                result["next"] = s2_execution.next_action(state, pending, args.now)
            except ValueError:
                raise _CliError("invalid_journal_state_or_observation") from None
        _emit(result)
        return 0
    if args.command in ("observe", "observe-watchdog"):
        record, report = _admitted(args)
        external = _observation(args, record) if args.command == "observe" else None
        coordinator, _ = _coordinator(args, None, clock, record, report)
        with _closing(coordinator):
            try:
                if args.command == "observe":
                    observation = coordinator.observe(external, export_root=args.export_root)
                else:
                    observation = coordinator.observe_watchdog()
            except s2_execution.JournalError as error:  # a ValueError: map it first
                raise _CliError(str(error)) from None
            except ValueError:
                raise _CliError("invalid_observation") from None
        _emit(observation)
        return 0
    # step/charges: refuse before reading credentials or constructing a provider client.
    if not args.execute or not args.authority_ref:
        raise _CliError("execution_requires_execute_and_authority")
    record, report = _admitted(args)  # the one read: authority and execution share it
    if args.authority_ref != record["approval"]["decision_id"]:
        raise _CliError("authority_ref_does_not_match_record")
    path = DECISIONS_PATH if decisions_path is None else decisions_path

    def gate():
        # A synthetic or fixture record never drives the real Vast CLI (A3-13).
        if record["scope"] != "actual":
            raise _CliError("real_provider_requires_actual_scope")
        # G1: the decision must bind this exact record by its canonical hash.
        refused = recorded_authority(args.authority_ref, report, path)
        if refused:
            raise _CliError(refused)

    if real_provider:
        gate()  # before any client exists

    def make_provider():
        # G1-12: an injected factory is exempt only when it yields the in-memory FakeProvider;
        # any other adapter (an injected real client) passes the same gate before first use.
        # Constructed at the original point (after the journal checks); a client constructor
        # performs no provider call.
        provider = provider_factory()
        if not real_provider and not isinstance(provider, s2_execution.FakeProvider):
            gate()
        return provider

    if args.command == "charges":
        return _charges(args, make_provider, clock, record, report)
    coordinator, _ = _coordinator(args, None, clock, record, report)
    with _closing(coordinator):
        coordinator.provider = make_provider()
        try:
            decision = coordinator.step()
        except (s2_execution.ProviderError, s2_execution.JournalError) as error:
            raise _CliError(str(error)) from None
    _emit(decision)
    return 0


def _charges(args, make_provider, clock, record, report):
    """Read-only provider charge query -> cumulative snapshot events for reconcile-cost."""
    state, _ = s2_execution.replay_records(_read_journal(args.journal))
    try:
        s2_execution.verify_journal_owner(state, record, report, args.intent_id,
                                          watchdog_events=args.watchdog_events)
    except ValueError:
        raise _CliError("journal_or_intent_mismatch") from None
    if not state["instance_ids"]:
        raise _CliError("no_bound_instance")
    provider = make_provider()
    try:
        rows = provider.charges(args.start, args.end)
        events = s2_execution.charge_snapshot_events(rows, state["label"],
                                                     state["instance_ids"], clock.time())
    except s2_execution.ProviderError as error:
        raise _CliError(str(error)) from None
    except ValueError:
        raise _CliError("charge_rows_invalid") from None
    _emit(events)
    return 0


class _closing:
    def __init__(self, coordinator):
        self.coordinator = coordinator

    def __enter__(self):
        return self.coordinator

    def __exit__(self, *_):
        self.coordinator.close()


def main(argv=None, *, provider_factory=None, clock=None, decisions_path=None):
    """Run one command. ``decisions_path`` overrides ``docs/DECISIONS.md`` (tests only)."""
    args = _parser().parse_args(argv)
    factory = provider_factory or (lambda: s2_execution.VastCliClient())
    try:
        return _run(args, factory, clock or _SystemClock(),
                    real_provider=provider_factory is None, decisions_path=decisions_path)
    except _CliError as error:
        _emit({"error": str(error)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
