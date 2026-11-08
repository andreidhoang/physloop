"""Agent-authored S2/G7 stop-only watchdog; no create/start/destroy operations.

Prepare and test before rental, then bind the exact new ID and unique label and
an absolute UTC deadline measured from creation (at most 60 minutes ahead).
Wait for the durable ``armed`` event before bootstrap. Keep this external host
awake and supervised: this is not a provider-enforced budget cap and cannot
survive host/process/network failure. Stopping does not end storage billing.

The installed Vast CLI's stop command can print prose even with --raw. Such a
response is an error, never an acknowledgment; only a subsequent JSON inventory
showing the exact bound instance stopped/absent establishes completion. Raw CLI
output and credentials are never copied into the event log or diagnostics.
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


def _call(cli, args, timeout):
    try:
        result = subprocess.run(
            [cli, "--raw", *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise CommandError("cli_timeout") from None
    except OSError:
        raise CommandError("cli_unavailable") from None
    except UnicodeError:
        raise CommandError("cli_undecodable_output") from None
    if result.returncode != 0:
        raise CommandError("cli_nonzero")
    try:
        payload = json.loads(result.stdout)
    except (ValueError, TypeError):
        raise CommandError("cli_non_json") from None
    if isinstance(payload, dict) and payload.get("success") is not True:
        raise CommandError("cli_unsuccessful_json")
    return payload


def _bound_instance(rows, instance_id, label):
    if not isinstance(rows, list):
        raise CommandError("inventory_not_full_list")
    ids = []
    for row in rows:
        if not isinstance(row, dict) or type(row.get("id")) is not int or row["id"] <= 0:
            raise ScopeError("invalid_inventory_identity")
        ids.append(row["id"])
    if len(set(ids)) != len(ids):
        raise ScopeError("duplicate_inventory_identity")
    matched = [row for row in rows if row["id"] == instance_id]
    if not matched:
        return None
    if matched[0].get("label") != label or sum(row.get("label") == label for row in rows) != 1:
        raise ScopeError("label_not_exact_and_unique")
    return matched[0]


def watch(
    instance_id,
    expected_label,
    deadline_epoch,
    events,
    *,
    cli="vastai",
    clock=time,
    poll_seconds=10,
    command_timeout=15,
):
    """Return 0 only on verified stopped/absent, 2 on unsafe scope. Retry CLI errors.

    CLI errors remain visible in durable events and retry indefinitely, with a
    fresh exact identity/label check before each stop. No error implies absence.
    The earlier of the wall and initial monotonic deadlines triggers stopping;
    clock rollback cannot extend the allowance. No initial unbound wait exists.
    """
    if type(instance_id) is not int or instance_id <= 0:
        raise ValueError("invalid_instance_id")
    if not isinstance(expected_label, str) or not re.fullmatch(
        r"[A-Za-z0-9_.-]{1,128}", expected_label
    ):
        raise ValueError("invalid_expected_label")
    if (
        type(deadline_epoch) not in (int, float)
        or not math.isfinite(deadline_epoch)
        or deadline_epoch - clock.time() > 3600
    ):
        raise ValueError("deadline_must_be_finite_and_within_one_hour")
    for number in (poll_seconds, command_timeout):
        if type(number) not in (int, float) or not math.isfinite(number) or not 0 < number <= 30:
            raise ValueError("poll_and_timeout_must_be_within_30_seconds")
    mono_deadline = clock.monotonic() + max(0, deadline_epoch - clock.time())
    armed = False
    while True:
        try:
            instance = _bound_instance(
                _call(cli, ["show", "instances"], command_timeout),
                instance_id,
                expected_label,
            )
        except ScopeError as error:
            events.emit("scope_mismatch", instance_id=instance_id, error=str(error))
            return 2
        except CommandError as error:
            events.emit("command_error", operation="show", error=str(error))
            clock.sleep(poll_seconds)
            continue
        # Vast can report an actually stopped VM as "exited" (observed integration).
        # Either snapshot can precede a queued start. Require explicit stop intent.
        # Null/missing provisioning status remains nonterminal and eligible for stop.
        stopped = instance is not None and (
            instance.get("actual_status") in ("stopped", "exited")
            and instance.get("intended_status") == "stopped"
            and instance.get("next_state") in (None, "stopped")
        )
        if instance is None or stopped:
            events.emit(
                "instance_absent" if instance is None else "stopped_verified",
                instance_id=instance_id,
            )
            return 0
        remaining = min(deadline_epoch - clock.time(), mono_deadline - clock.monotonic())
        if not armed and remaining > 0:
            events.emit(
                "armed",
                instance_id=instance_id,
                expected_label=expected_label,
                deadline_utc_epoch=deadline_epoch,
            )
            armed = True
        if remaining <= 0:
            try:
                result = _call(cli, ["stop", "instance", str(instance_id)], command_timeout)
                if not isinstance(result, dict) or result.get("success") is not True:
                    raise CommandError("stop_acknowledgment_missing")
                events.emit("stop_requested", instance_id=instance_id)
            except CommandError as error:
                events.emit("command_error", operation="stop", error=str(error))
        clock.sleep(poll_seconds if remaining <= 0 else min(poll_seconds, remaining))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", required=True, type=int)
    parser.add_argument("--expected-label", required=True)
    parser.add_argument(
        "--deadline-utc", required=True, help="ISO8601 with timezone, from creation"
    )
    parser.add_argument("--events", required=True, type=Path, help="New JSONL path; must not exist")
    parser.add_argument("--cli", default="vastai", help="Installed CLI executable path")
    args = parser.parse_args(argv)
    try:
        deadline = datetime.fromisoformat(args.deadline_utc)
        if deadline.tzinfo is None:
            raise ValueError("timezone_required")
        with EventLog(args.events) as events:
            return watch(
                args.instance_id, args.expected_label, deadline.timestamp(), events, cli=args.cli
            )
    except (ValueError, OSError, KeyboardInterrupt) as error:
        print(json.dumps({"event": "watchdog_failed", "error": type(error).__name__}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
