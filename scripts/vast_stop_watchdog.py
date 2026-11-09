"""Agent-authored S2/G7 stop-only watchdog; no create/start/destroy operations.

Build-spec A3 (R4 C5, P2-B, P2-C) under D-018/D-024; serves S2.2/S2.7 [C] (G2, G7).

Prepare and test before rental, then bind the exact new ID and unique label and an
absolute UTC deadline (the action record's cutoff). ``--max-window-seconds`` must
equal the action record's approved window; the deadline may not lie further ahead
than that window, and no window may exceed ``MAX_WINDOW_SECONDS`` (12 h,
``agent technical default (D-018)``). Wait for the durable ``armed`` event before
bootstrap. Keep this external host awake and supervised: this is not a
provider-enforced budget cap and cannot survive host/process/network failure.
Stopping does not end storage billing.

Behaviour (verification ceiling: L1 with a fake runner; L2 against the installed Vast
CLI 1.0.7 source and the live D-019/D-020 evidence; real provider behaviour is L3):

- Before the binding is verified (exact ID with exactly one row carrying the unique
  label), a scope mismatch exits 2 without stopping anything.
- Absence is never inferred from an inventory: when the bound ID is missing from
  ``show instances`` it must be ``NotFound`` on two consecutive successful exact-ID
  ``show instance <id>`` calls (``ABSENCE_CONFIRMATIONS``, agent technical default
  (D-018)); otherwise monitoring continues (R4 C5).
- After the binding is verified, a scope mismatch (duplicate label, changed label)
  triggers an immediate exact-ID stop of the bound instance only; once its stop is
  verified the watchdog exits 2 so the anomaly stays visible (R4 P2-B).
- At or after the deadline the watchdog issues exact-ID stops even if the inventory
  fails or is ambiguous, verifying by exact-ID state, with exponential backoff (cap
  ``MAX_BACKOFF_SECONDS``). If the stop is not confirmed within
  ``--stop-retry-limit-seconds`` (default ``STOP_RETRY_LIMIT_SECONDS``, agent technical
  default (D-018)) it exits 3 with ``stop_unconfirmed`` (R4 P2-C).
- The deadline is absolute and persisted in the ``armed`` event. Every invocation
  states which run it is: ``--first-arm`` or ``--prior-events PATH`` (exactly one is
  required). A restart's prior event file must exist; a ``--deadline-utc`` later than
  the earliest prior armed deadline for the same binding is refused (exit 2) before
  any event is written, so a restart never renews the allowance.

The installed Vast CLI's stop command prints prose even with --raw, and the CLI
exits 0 after an HTTP error while printing an error JSON to stderr. A response is
therefore never an acknowledgment; only a structured state read (``exited``/
``stopped`` with explicit stop intent, D-020) or confirmed absence completes. Raw CLI
output and credentials are never copied into the event log or diagnostics; the CLI
runs with closed stdin and an allowlisted environment without ``VAST_API_KEY``.

Exit codes: 0 verified stopped/absent; 2 scope mismatch or invalid input; 3
``stop_unconfirmed``.
"""

import argparse
import json
import math
import os
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

MAX_WINDOW_SECONDS = 12 * 3600  # agent technical default (D-018)
ABSENCE_CONFIRMATIONS = 2  # agent technical default (D-018)
STOP_RETRY_LIMIT_SECONDS = 900  # agent technical default (D-018)
MAX_BACKOFF_SECONDS = 60  # agent technical default (D-018)
NOT_FOUND_STATUS_CODES = frozenset({404})  # agent technical default (D-018); L3
ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "XDG_CONFIG_HOME")
_NOT_FOUND = object()


class CommandError(Exception):
    """Contains only our static diagnostic code, never subprocess output."""


class ScopeError(Exception):
    """Inventory does not establish the exact, uniquely labelled binding."""


class EventLog:
    """Exclusive creation and fsynced appends; existing evidence is never reused."""

    def __init__(self, path):
        self.fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL, 0o600)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        os.close(self.fd)

    def emit(self, event, **fields):
        record = {"at_utc": datetime.now(UTC).isoformat(), "event": event, **fields}
        line = json.dumps(record, allow_nan=False) + "\n"
        data = line.encode()
        while data:
            written = os.write(self.fd, data)
            if not written:
                raise OSError("event_write_failed")
            data = data[written:]
        os.fsync(self.fd)
        try:
            print(line, end="", flush=True)
        except BrokenPipeError:
            pass  # The durable event file remains the acknowledgment authority.


def subprocess_runner(argv, timeout):
    """Default runner: argv list, no shell, closed stdin, allowlisted environment."""
    env = {key: os.environ[key] for key in ENV_ALLOWLIST if key in os.environ}
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                          check=False, stdin=subprocess.DEVNULL, env=env)


def _run(cli, args, timeout, runner):
    try:
        result = runner([cli, "--raw", *args], timeout)
    except subprocess.TimeoutExpired:
        raise CommandError("cli_timeout") from None
    except UnicodeError:
        raise CommandError("cli_undecodable_output") from None
    except OSError:
        raise CommandError("cli_unavailable") from None
    if type(result.stdout) is not str or type(result.stderr) is not str:
        raise CommandError("cli_undecodable_output")
    return result


def _error_status(stderr):
    for line in stderr.splitlines():
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("error") is True:
            code = payload.get("status_code")
            return code if type(code) is int else 0
    return None


def _json(text):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        raise CommandError("cli_non_json") from None


def _inventory(cli, timeout, runner):
    result = _run(cli, ["show", "instances"], timeout, runner)
    if result.returncode != 0:
        raise CommandError("cli_nonzero")
    rows = _json(result.stdout)
    if not isinstance(rows, list):
        raise CommandError("inventory_not_full_list")
    return rows


def _exact(cli, instance_id, timeout, runner):
    """Exact-ID state: a row dict, ``_NOT_FOUND`` or CommandError."""
    result = _run(cli, ["show", "instance", str(instance_id)], timeout, runner)
    if result.returncode != 0:
        raise CommandError("cli_nonzero")
    if not result.stdout.strip():
        if _error_status(result.stderr) in NOT_FOUND_STATUS_CODES:
            return _NOT_FOUND
        raise CommandError("cli_error_status")
    row = _json(result.stdout)
    if not isinstance(row, dict) or row.get("id") != instance_id:
        raise CommandError("exact_identity_mismatch")
    return row


def _request_stop(cli, instance_id, timeout, runner):
    result = _run(cli, ["stop", "instance", str(instance_id)], timeout, runner)
    if result.returncode != 0 or _error_status(result.stderr) is not None:
        raise CommandError("stop_rejected")


def _bound_instance(rows, instance_id, label):
    ids = []
    for row in rows:
        if not isinstance(row, dict) or type(row.get("id")) is not int or row["id"] <= 0:
            raise ScopeError("invalid_inventory_identity")
        ids.append(row["id"])
    if len(set(ids)) != len(ids):
        raise ScopeError("duplicate_inventory_identity")
    matched = [row for row in rows if row["id"] == instance_id]
    labelled = sum(row.get("label") == label for row in rows)
    if not matched:
        if labelled:
            raise ScopeError("label_on_other_instance")
        return None
    if matched[0].get("label") != label or labelled != 1:
        raise ScopeError("label_not_exact_and_unique")
    return matched[0]


def _stopped(row):
    # Vast can report an actually stopped VM as "exited" (observed integration, D-020).
    # Either snapshot can precede a queued start. Require explicit stop intent.
    # Null/missing provisioning status remains nonterminal and eligible for stop.
    return (row.get("actual_status") in ("stopped", "exited")
            and row.get("intended_status") == "stopped"
            and row.get("next_state") in (None, "stopped"))


def prior_armed_deadline(path, instance_id, label):
    """Earliest armed deadline for this binding in a prior event file, else None.

    Fails closed on a missing file (FileNotFoundError), unreadable content or an armed
    event for another binding (ValueError), so a restart can never silently renew the
    allowance. ``None`` means the prior run never armed.
    """
    lines = Path(path).read_bytes().splitlines()
    deadlines = []
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            raise ValueError("prior_events_unreadable") from None
        if not isinstance(event, dict):
            raise TypeError("prior_events_unreadable")
        if event.get("event") != "armed":
            continue
        if event.get("instance_id") != instance_id or event.get("expected_label") != label:
            raise ValueError("prior_events_foreign_binding")
        deadline = event.get("deadline_utc_epoch")
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError("prior_events_invalid_deadline")
        deadlines.append(deadline)
    return min(deadlines) if deadlines else None


def _validate(instance_id, label, deadline, window, clock, poll, timeout, retry_limit):
    if type(instance_id) is not int or instance_id <= 0:
        raise ValueError("invalid_instance_id")
    if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", label):
        raise ValueError("invalid_expected_label")
    if type(window) is not int or not 0 < window <= MAX_WINDOW_SECONDS:
        raise ValueError("max_window_must_be_positive_and_within_hard_bound")
    if (type(deadline) not in (int, float) or not math.isfinite(deadline)
            or deadline - clock.time() > window):
        raise ValueError("deadline_must_be_finite_and_within_approved_window")
    for number in (poll, timeout):
        if type(number) not in (int, float) or not math.isfinite(number) or not 0 < number <= 30:
            raise ValueError("poll_and_timeout_must_be_within_30_seconds")
    if type(retry_limit) not in (int, float) or not 0 < retry_limit <= 3600:
        raise ValueError("stop_retry_limit_must_be_within_one_hour")


def watch(
    instance_id,
    expected_label,
    deadline_epoch,
    events,
    *,
    max_window_seconds,
    cli="vastai",
    clock=time,
    runner=None,
    poll_seconds=10,
    command_timeout=15,
    stop_retry_limit_seconds=STOP_RETRY_LIMIT_SECONDS,
    prior_deadline_epoch=None,
):
    """Monitor, then stop the exact bound instance at the deadline (see module doc).

    The earlier of the wall and initial monotonic deadlines triggers stopping; clock
    rollback cannot extend the allowance. No error implies absence or success.
    """
    _validate(instance_id, expected_label, deadline_epoch, max_window_seconds, clock,
              poll_seconds, command_timeout, stop_retry_limit_seconds)
    runner = runner or subprocess_runner
    if prior_deadline_epoch is not None and prior_deadline_epoch < deadline_epoch:
        deadline_epoch = prior_deadline_epoch
        events.emit("deadline_clamped_to_prior_arm", instance_id=instance_id,
                    deadline_utc_epoch=deadline_epoch)
    mono_deadline = clock.monotonic() + max(0, deadline_epoch - clock.time())
    armed = verified = scope_violation = False
    unverified_give_up = None

    def remaining():
        return min(deadline_epoch - clock.time(), mono_deadline - clock.monotonic())

    while True:
        left = remaining()
        if left <= 0 and verified:
            events.emit("deadline_reached", instance_id=instance_id)
            break
        if left <= 0:
            if unverified_give_up is None:
                unverified_give_up = clock.monotonic() + stop_retry_limit_seconds
            elif clock.monotonic() >= unverified_give_up:
                events.emit("binding_unverified_past_deadline", instance_id=instance_id)
                return 2
        try:
            instance = _bound_instance(_inventory(cli, command_timeout, runner), instance_id,
                                       expected_label)
        except ScopeError as error:
            if not verified:
                events.emit("scope_mismatch", instance_id=instance_id, error=str(error))
                return 2
            events.emit("scope_mismatch_after_arm", instance_id=instance_id, error=str(error))
            scope_violation = True
            break
        except CommandError as error:
            events.emit("command_error", operation="show", error=str(error))
            clock.sleep(poll_seconds if left <= 0 else min(poll_seconds, left))
            continue
        if instance is None:
            outcome = _confirm_absent(cli, instance_id, command_timeout, runner, clock,
                                      poll_seconds, events)
            if outcome is not None:
                return 0
        elif _stopped(instance):
            events.emit("stopped_verified", instance_id=instance_id)
            return 0
        else:
            verified = True
            if not armed and remaining() > 0:
                events.emit("armed", instance_id=instance_id, expected_label=expected_label,
                            deadline_utc_epoch=deadline_epoch)
                armed = True
        left = remaining()
        if left > 0:
            clock.sleep(min(poll_seconds, left))
        elif not verified:
            clock.sleep(poll_seconds)
    code = _stop_until_confirmed(cli, instance_id, command_timeout, runner, clock,
                                 poll_seconds, stop_retry_limit_seconds, events)
    return 2 if code == 0 and scope_violation else code


def _confirm_absent(cli, instance_id, timeout, runner, clock, poll, events):
    """Two consecutive successful exact-ID NotFound results; else None (keep watching)."""
    for attempt in range(ABSENCE_CONFIRMATIONS):
        if attempt:
            clock.sleep(poll)
        try:
            state = _exact(cli, instance_id, timeout, runner)
        except CommandError as error:
            events.emit("command_error", operation="show_exact", error=str(error))
            return None
        if state is not _NOT_FOUND:
            if _stopped(state):
                events.emit("stopped_verified", instance_id=instance_id)
                return "stopped"
            return None
    events.emit("instance_absent", instance_id=instance_id)
    return "absent"


def _stop_until_confirmed(cli, instance_id, timeout, runner, clock, poll, limit, events):
    give_up = clock.monotonic() + limit
    backoff, not_found = poll, 0
    while True:
        try:
            state = _exact(cli, instance_id, timeout, runner)
        except CommandError as error:
            events.emit("command_error", operation="show_exact", error=str(error))
            state, not_found = None, 0
        if state is _NOT_FOUND:
            not_found += 1
            if not_found >= ABSENCE_CONFIRMATIONS:
                events.emit("instance_absent", instance_id=instance_id)
                return 0
        elif isinstance(state, dict) and _stopped(state):
            events.emit("stopped_verified", instance_id=instance_id)
            return 0
        else:
            not_found = 0
        if state is not _NOT_FOUND:
            try:
                _request_stop(cli, instance_id, timeout, runner)
                events.emit("stop_requested", instance_id=instance_id)
            except CommandError as error:
                events.emit("command_error", operation="stop", error=str(error))
        if clock.monotonic() >= give_up:
            events.emit("stop_unconfirmed", instance_id=instance_id)
            return 3
        clock.sleep(backoff)
        backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", required=True, type=int)
    parser.add_argument("--expected-label", required=True)
    parser.add_argument("--deadline-utc", required=True,
                        help="ISO8601 with timezone: the action record's absolute cutoff")
    parser.add_argument("--max-window-seconds", required=True, type=int,
                        help="the action record's approved window; at most 12 h")
    parser.add_argument("--events", required=True, type=Path, help="New JSONL path; must not exist")
    run = parser.add_mutually_exclusive_group(required=True)
    run.add_argument("--first-arm", action="store_true",
                     help="first watchdog for this binding (no earlier event file)")
    run.add_argument("--prior-events", type=Path,
                     help="event file of an earlier run for this binding (restart)")
    parser.add_argument("--stop-retry-limit-seconds", type=int,
                        default=STOP_RETRY_LIMIT_SECONDS)
    parser.add_argument("--cli", default="vastai", help="Installed CLI executable path")
    args = parser.parse_args(argv)
    try:
        deadline = datetime.fromisoformat(args.deadline_utc)
        if deadline.tzinfo is None:
            raise ValueError("timezone_required")
        prior = None
        if args.prior_events is not None:
            prior = prior_armed_deadline(args.prior_events, args.instance_id,
                                         args.expected_label)
            if prior is not None and deadline.timestamp() > prior:
                raise ValueError("deadline_later_than_prior_arm")
        with EventLog(args.events) as events:
            return watch(
                args.instance_id, args.expected_label, deadline.timestamp(), events,
                max_window_seconds=args.max_window_seconds, cli=args.cli,
                stop_retry_limit_seconds=args.stop_retry_limit_seconds,
                prior_deadline_epoch=prior,
            )
    except (ValueError, TypeError, OSError, KeyboardInterrupt) as error:
        print(json.dumps({"event": "watchdog_failed", "error": type(error).__name__}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
