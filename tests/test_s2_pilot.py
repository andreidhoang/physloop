"""Literal CPU controls for the small pilot accounting/export boundary."""

import copy
import hashlib

import pytest

from gate.s2_pilot import copy_verified_export, next_unit_check, verify_export


def budget_request():
    return {
        "scopes": {
            name: {"cap_usd": "10", "used_usd": "6", "liability_usd": "1",
                   "reserve_usd": "1"}
            for name in ("pilot", "stage", "project")
        },
        "next_unit_usd": "1.99", "now_epoch": 100, "deadline_epoch": 200,
        "next_unit_seconds": 60, "cleanup_seconds": 20,
    }


def test_next_unit_preserves_each_nested_limit_and_cleanup():
    request = budget_request()
    result = next_unit_check(request)
    assert result["can_start_next_unit"]
    assert result["remaining_after_reserves_usd"]["pilot"] == "2"
    assert result["cloud_launch_authorized"] is False
    request["next_unit_usd"] = "2"
    assert not next_unit_check(request)["can_start_next_unit"]


@pytest.mark.parametrize("scope", ["pilot", "stage", "project"])
def test_any_scope_unknown_or_exhausted_blocks(scope):
    request = budget_request()
    request["scopes"][scope]["liability_usd"] = None
    assert not next_unit_check(request)["can_start_next_unit"]
    request["scopes"][scope]["liability_usd"] = "4"
    assert not next_unit_check(request)["can_start_next_unit"]


@pytest.mark.parametrize("bad", [True, -1, "NaN", "Infinity", "-1", None, 1.99,
                                 "1e999999999"])
def test_invalid_currency_never_looks_affordable(bad):
    request = budget_request()
    request["next_unit_usd"] = bad
    assert not next_unit_check(request)["can_start_next_unit"]


def test_deadline_and_restart_do_not_renew_allowance():
    request = budget_request()
    request["now_epoch"] = 120  # exact work + cleanup boundary
    assert not next_unit_check(request)["can_start_next_unit"]
    request["now_epoch"] = 190  # resumed with original absolute cutoff
    assert not next_unit_check(request)["can_start_next_unit"]
    assert request["deadline_epoch"] == 200


def test_missing_scope_and_boolean_clock_block():
    request = budget_request()
    del request["scopes"]["project"]
    assert not next_unit_check(request)["can_start_next_unit"]
    request = budget_request()
    request["now_epoch"] = True
    assert not next_unit_check(request)["can_start_next_unit"]


def receipt(name="raw.json", data=b"raw data"):
    return {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def test_export_verified_at_destination_and_detects_later_corruption(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "raw.json").write_bytes(b"raw data")
    destination = tmp_path / "destination"
    result = copy_verified_export(source, destination, [receipt()])
    assert result["verified"]
    assert result["cloud_cleanup_authorized"] is False
    assert (destination / "raw.json").read_bytes() == b"raw data"
    (destination / "raw.json").write_bytes(b"bad data")
    assert not verify_export(destination, [receipt()])["verified"]


@pytest.mark.parametrize("problem", ["missing", "hash", "duplicate", "escape", "symlink"])
def test_failed_export_never_creates_destination(tmp_path, problem):
    source = tmp_path / "source"
    source.mkdir()
    (source / "raw.json").write_bytes(b"raw data")
    receipts = [receipt()]
    if problem == "missing":
        receipts[0]["path"] = "missing.json"
    elif problem == "hash":
        receipts[0]["sha256"] = "0" * 64
    elif problem == "duplicate":
        receipts.append(copy.deepcopy(receipts[0]))
    elif problem == "escape":
        receipts[0]["path"] = "../raw.json"
    else:
        (source / "alias.json").symlink_to(source / "raw.json")
        receipts[0]["path"] = "alias.json"
    destination = tmp_path / "export"
    with pytest.raises(ValueError):
        copy_verified_export(source, destination, receipts)
    assert not destination.exists()


def test_export_never_overwrites_and_empty_manifest_cannot_pass(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "raw.json").write_bytes(b"raw data")
    with pytest.raises(FileExistsError):
        copy_verified_export(source, source, [receipt()])
    assert (source / "raw.json").read_bytes() == b"raw data"
    assert not verify_export(source, [])["verified"]


def test_export_rejects_unmanifested_files_and_symlinks(tmp_path):
    (tmp_path / "raw.json").write_bytes(b"raw data")
    (tmp_path / "unmanifested").write_bytes(b"not expected")
    assert not verify_export(tmp_path, [receipt()])["verified"]


def test_export_refuses_symlinked_destination_ancestor(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "raw.json").write_bytes(b"raw data")
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError):
        copy_verified_export(source, alias / "export", [receipt()])
    assert list(actual.iterdir()) == []


# --- R4 C7: usage and liability nest pilot <= stage <= project (agent-authored, D-024) ------


def nested_request(pilot, stage, project, field="used_usd"):
    request = {
        "scopes": {
            "pilot": {"cap_usd": "100", "used_usd": "0", "liability_usd": "0", "reserve_usd": "0"},
            "stage": {"cap_usd": "1200", "used_usd": "0", "liability_usd": "0", "reserve_usd": "0"},
            "project": {"cap_usd": "2500", "used_usd": "0", "liability_usd": "0",
                        "reserve_usd": "0"},
        },
        "next_unit_usd": "4.99", "now_epoch": 0, "deadline_epoch": 100,
        "next_unit_seconds": 10, "cleanup_seconds": 10,
    }
    for name, value in (("pilot", pilot), ("stage", stage), ("project", project)):
        request["scopes"][name][field] = value
    return request


def test_c7_repro_non_nested_usage_is_blocked():
    """repro_05(a): pilot used 95 while stage/project recorded 0 was affordable."""
    result = next_unit_check(nested_request("95", "0", "0"))
    assert not result["can_start_next_unit"]
    assert result["issues"] == ["budget_usage_not_nested"]


@pytest.mark.parametrize(
    "field,values",
    [
        ("used_usd", ("10", "9.99", "20")),
        ("used_usd", ("10", "20", "19.99")),
        ("liability_usd", ("1", "0", "1")),
        ("liability_usd", ("0", "2", "1")),
    ],
)
def test_c7_any_inversion_of_used_or_liability_blocks(field, values):
    result = next_unit_check(nested_request(*values, field=field))
    assert "budget_usage_not_nested" in result["issues"]
    assert not result["can_start_next_unit"]


@pytest.mark.parametrize(
    "field,values",
    [("used_usd", ("10", "10", "10")), ("used_usd", ("10", "30", "40.60")),
     ("liability_usd", ("1", "1.5", "2"))],
)
def test_c7_negative_control_nested_usage_is_affordable(field, values):
    result = next_unit_check(nested_request(*values, field=field))
    assert result["can_start_next_unit"], result["issues"]
    assert result["issues"] == []
