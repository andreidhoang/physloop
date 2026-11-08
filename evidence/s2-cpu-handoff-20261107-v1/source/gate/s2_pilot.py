"""Small CPU pilot helpers: conservative affordability and verified local export.

No provisioning, stopping, deletion, asset download or authorization occurs here.
Cost inputs are recorded USD usage/liabilities, not cash deposits; do not count
prepaid consumption twice. A successful export is integrity, not authenticity.
"""

import math
import os
from decimal import Decimal, DecimalException, InvalidOperation
from pathlib import Path

from gate.s2_identity import read_artifact


def _usd(value):
    # Explicit decimal strings prevent accidentally inheriting binary float rounding.
    if type(value) is not str:
        raise ValueError("USD amounts require nonnegative decimal strings")
    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("Invalid USD amount") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError("Invalid USD amount")
    return amount


def next_unit_check(request):
    """Check all nested balances and an original absolute deadline; never authorize.

    Require strictly positive remaining headroom after next work and reserves.
    All fees/unsettled charges must have a conservative bound in liability_usd;
    null is unknown and blocks. Approval and truthful accounting are external.
    """
    issues, balances = [], {}
    try:
        if type(request) is not dict:
            raise ValueError("Expected an object")
        scopes = request.get("scopes")
        if type(scopes) is not dict or set(scopes) != {"pilot", "stage", "project"}:
            raise ValueError("Require pilot, stage and project budgets")
        next_cost = _usd(request.get("next_unit_usd"))
        caps = {}
        for name, scope in scopes.items():
            if type(scope) is not dict or set(scope) != {
                "cap_usd", "used_usd", "liability_usd", "reserve_usd"
            }:
                raise ValueError("Require complete usage, liability and reserve fields")
            amounts = {key: _usd(value) for key, value in scope.items()}
            caps[name] = amounts["cap_usd"]
            remaining = (amounts["cap_usd"] - amounts["used_usd"]
                         - amounts["liability_usd"] - amounts["reserve_usd"])
            balances[name] = str(remaining)
            if next_cost >= remaining:
                issues.append(f"{name}_headroom_exhausted")
        if not caps["pilot"] <= caps["stage"] <= caps["project"]:
            issues.append("caps_not_nested")
        for key in ("now_epoch", "deadline_epoch", "next_unit_seconds", "cleanup_seconds"):
            value = request.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Clock and duration fields must be finite nonnegative numbers")
        end = request["now_epoch"] + request["next_unit_seconds"] + request["cleanup_seconds"]
        if end >= request["deadline_epoch"]:
            issues.append("deadline_headroom_exhausted")
    except (ValueError, DecimalException, OverflowError):
        issues.append("invalid_or_unknown_budget_input")
    return {
        "can_start_next_unit": not issues,
        "issues": issues,
        "remaining_after_reserves_usd": balances,
        "cloud_launch_authorized": False,
        "basis": "supplied usage/liability bounds; not verified provider billing or approval",
    }


def verify_export(root, receipts):
    """Check every declared file using ingestion's no-follow, hash-checked reader."""
    issues, names = [], set()
    if not isinstance(receipts, list) or not receipts:
        issues.append("empty_or_invalid_export_manifest")
        receipts = []
    for index, item in enumerate(receipts):
        try:
            name = item.get("path") if isinstance(item, dict) else None
            if not isinstance(name, str) or name in names:
                raise ValueError("Invalid or duplicate path")
            names.add(name)
            read_artifact(root, item)
        except (OSError, ValueError, UnicodeError):
            issues.append(f"export_file_invalid:{index}")
    actual = set()
    for parent, directories, files in os.walk(root, followlinks=False,
                                               onerror=lambda _: issues.append("export_unreadable")):
        for name in directories + files:
            if (Path(parent) / name).is_symlink():
                issues.append("export_symlink")
        for name in files:
            actual.add((Path(parent) / name).relative_to(root).as_posix())
    if actual != names:
        issues.append("export_file_set_mismatch")
    return {
        "verified": not issues,
        "file_count": len(receipts),
        "issues": issues,
        "cloud_cleanup_authorized": False,
    }


def copy_verified_export(source, destination, receipts):
    """Copy verified bytes into a new local directory; never overwrite or delete.

    Used for the CPU rehearsal. A real remote transfer may use rsync/scp, then
    call verify_export on the destination with the same expected receipts.
    Partial failures leave the new directory visible and unverified.
    """
    if not verify_export(source, receipts)["verified"]:
        raise ValueError("Source export verification failed")
    # Read again through the verified descriptor path, so changed bytes cannot be copied.
    contents = [(item["path"], read_artifact(source, item)) for item in receipts]
    destination = Path(destination).absolute()
    if ".." in destination.parts or any(
        path.is_symlink() for path in (destination, *destination.parents)
    ):
        raise ValueError("Export destination must not contain traversal or symlinks")
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in contents:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    return verify_export(destination.resolve(strict=True), receipts)
