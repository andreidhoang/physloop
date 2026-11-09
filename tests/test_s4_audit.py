"""Known answers for E05, E16, E17 and E21 (build-spec A9).

Oracles: hand-derived literals; the pinned ``filter_trainable_rollouts`` (AlpaGym) and
Cosmos ``check_checkpoint_saving``/``save_check`` executed from hash-checked vendored
source (selected AST nodes, stub ``self``, restricted namespace).
"""

import importlib.util
import itertools
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from teacher_rl.checks import audit

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "s4_upstream_loader", ROOT / "tests/fixtures/s4_upstream_checks/upstream_loader.py"
)
upstream = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(upstream)

PIN = "36aeb4c5938cbc2eb2aed33b22434773da4ab639"
SHA = "c" * 64


def codes(report):
    return {i["code"] for i in report["issues"]}


# --- E05 ---------------------------------------------------------------------------


def test_e05_pinned_filter_only_warns_on_stale_rollouts():
    warnings = []
    ns = upstream.load(
        "alpagym/rollout_filter.py.txt",
        ["filter_trainable_rollouts"],
        {"logger": SimpleNamespace(warning=lambda *a: warnings.append(a)), "pathlib": None},
    )
    rollouts = [SimpleNamespace(completion="", weight_version=v) for v in (5, 2)]
    kept = ns["filter_trainable_rollouts"](
        rollouts, current_step=5, train_batch_per_replica=2, allowed_outdated_steps=0
    )
    assert [r.weight_version for r in kept] == [5, 2]  # stale rollout kept
    assert len(warnings) == 1 and warnings[0][1] == 1  # one stale kept, warning only


def _rollout(rid, behaviour, stamped):
    return {"rollout_id": rid, "behaviour_version": behaviour, "stamped_version": stamped}


def test_e05_fence_rejects_what_upstream_keeps():
    report = audit.check_version_fence(
        [_rollout("a", 5, 5), _rollout("b", 2, 2)],
        current_step=5,
        allowed_outdated_steps=0,
        scope="synthetic",
    )
    assert report["status"] == "blocked"
    assert "stale_rollout_consumed" in codes(report)
    assert [r["lag"] for r in report["rollouts"]] == [0, 3]
    ok = audit.check_version_fence(
        [_rollout("a", 4, 4)], current_step=5, allowed_outdated_steps=1, scope="synthetic"
    )
    assert ok["status"] == "passed" and ok["rollouts"][0]["stale"] is False


def test_e05_negative_controls():
    stamped_zero = audit.check_version_fence(
        [_rollout("a", 5, 0)], current_step=5, allowed_outdated_steps=0, scope="synthetic"
    )
    assert "version_stamp_mismatch" in codes(stamped_zero)  # packer stamps weight_version=0
    missing = audit.check_version_fence(
        [_rollout("a", None, 0)], current_step=5, allowed_outdated_steps=0, scope="synthetic"
    )
    assert missing["status"] == "unverified"
    unregistered = audit.check_version_fence(
        [_rollout("a", 5, 5)], current_step=5, allowed_outdated_steps=None, scope="synthetic"
    )
    assert unregistered["status"] == "blocked" and "allowed_outdated_steps_unregistered" in codes(
        unregistered
    )
    future = audit.check_version_fence(
        [_rollout("a", 6, 6)], current_step=5, allowed_outdated_steps=0, scope="synthetic"
    )
    assert "behaviour_version_from_future" in codes(future)
    with pytest.raises(ValueError):
        audit.check_version_fence(
            [_rollout("a", 1, 1), _rollout("a", 1, 1)],
            current_step=5,
            allowed_outdated_steps=0,
            scope="synthetic",
        )


# --- E16 ---------------------------------------------------------------------------


def path(path_id, **overrides):
    base = {
        "path_id": path_id,
        "image_size": [320, 512],
        "coc": False,
        "rope_offsets": "alpagym_bundle",
        "sampler": {"kind": "sde", "steps": 5},
        "control_rate_hz": 5.0,
        "controller": "linear_mpc",
        "model_package_commit": PIN,
        "vlm_name_or_path": "nvidia/Cosmos-Reason2-8B",
        "vlm_revision": "a9fae2cf89dc64db96b12860417f0eb403013bb9",
    }
    base.update(overrides)
    return base


def test_e16_reports_every_difference_and_never_explains_it():
    gate = path(
        "gate",
        image_size=[1080, 1920],
        coc=True,
        sampler={"kind": "ode", "steps": 10},
        control_rate_hz=10.0,
        controller="nonlinear_mpc",
        rope_offsets="driver_default",
    )
    report = audit.audit_adapter_paths(
        path("train", model_package_commit="4cda35d22bb257f0936ac272397627b9309ca211"),
        gate,
        expected_driver_commit=PIN,
        scope="public_source_fixture",
    )
    assert report["status"] == "passed" and report["paths_equivalent"] is False
    differing = sorted(r["factor"] for r in report["factors"] if r["state"] == "differs")
    assert differing == sorted([*audit.FACTORS, "model_package_commit"])
    assert sum(i["code"] == "adapter_factor_differs" for i in report["issues"]) == 7


def test_e16_identical_paths_are_equivalent():
    report = audit.audit_adapter_paths(
        path("train"), path("gate"), expected_driver_commit=PIN, scope="synthetic"
    )
    assert report["status"] == "passed" and report["paths_equivalent"] is True


def test_e16_negative_controls():
    unresolved = audit.audit_adapter_paths(
        path("train"),
        path("gate", model_package_commit=None),
        expected_driver_commit=PIN,
        scope="synthetic",
    )
    assert unresolved["status"] == "blocked" and "driver_commit_unresolved" in codes(unresolved)
    floating = audit.audit_adapter_paths(
        path("train"),
        path("gate", model_package_commit="f" * 40),
        expected_driver_commit=PIN,
        scope="synthetic",
    )
    assert "driver_commit_not_pinned" in codes(floating)
    unknown = audit.audit_adapter_paths(
        path("train", coc=None), path("gate"), expected_driver_commit=PIN, scope="synthetic"
    )
    assert unknown["status"] == "unverified" and unknown["paths_equivalent"] is False
    vlm = audit.audit_adapter_paths(
        path("train"),
        path("gate", vlm_revision="0" * 40),
        expected_driver_commit=PIN,
        scope="synthetic",
    )
    assert any(r["factor"] == "vlm_revision" and r["state"] == "differs" for r in vlm["factors"])
    with pytest.raises(ValueError):
        audit.audit_adapter_paths(
            path("train") | {"explanation": "expected"},
            path("gate"),
            expected_driver_commit=PIN,
            scope="synthetic",
        )
    with pytest.raises(ValueError):
        audit.audit_adapter_paths(
            path("train", sampler={"kind": "sde"}),
            path("gate"),
            expected_driver_commit=PIN,
            scope="synthetic",
        )


# --- E17 ---------------------------------------------------------------------------


def ev(run, road, scene, rep):
    return {"training_run_id": run, "road_id": road, "scene_id": scene, "replicate_id": rep}


def test_e17_rollout_multiplicity_is_not_road_support():
    rows = [
        ev(run, "road-1", scene, rep)
        for run in ("r1", "r2", "r3")
        for scene in ("s1", "s2")
        for rep in ("0", "1", "2")
    ]
    rows += [ev(run, "road-2", "s3", "0") for run in ("r1", "r2", "r3")]
    report = audit.uncertainty_axes(rows, scope="synthetic")
    assert report["n_rollouts"] == 21
    assert report["road_support"] == 2 and report["run_support"] == 3
    assert report["crossed"] is True and report["shared_roads"] == ["road-1", "road-2"]
    assert report["rollouts_by_road"] == {"road-1": 18, "road-2": 3}
    single = audit.uncertainty_axes(
        [ev("r1", "road-1", "s1", "0"), ev("r1", "road-1", "s1", "1")], scope="synthetic"
    )
    assert single["run_level_variation"] == "not_estimable" and single["road_support"] == 1


def test_e17_negative_controls():
    dup = audit.uncertainty_axes(
        [ev("r1", "d", "s", "0"), ev("r1", "d", "s", "0")], scope="synthetic"
    )
    assert dup["status"] == "blocked" and "duplicate_evaluation" in codes(dup)
    moved = audit.uncertainty_axes(
        [ev("r1", "d1", "s", "0"), ev("r2", "d2", "s", "0")], scope="synthetic"
    )
    assert "scene_road_inconsistent" in codes(moved)


# --- E21 ---------------------------------------------------------------------------


def test_e21_kl_not_computed_when_beta_nonpositive():
    off = audit.kl_status(0.0, [0.0, 0.0], source="in_process_capture")
    assert off["kl_status"] == "not_computed" and off["logged_values_ignored"] == 2
    assert "penalty_mean" not in off and "kl_mean" not in off
    on = audit.kl_status(0.01, [0.004, 0.006], source="in_process_capture")
    assert on["kl_status"] == "computed" and on["penalty_mean"] == pytest.approx(0.005)
    assert audit.kl_status(0.01)["kl_status"] == "not_logged"
    nonfinite = audit.kl_status(0.01, ["nan", 0.1], source="in_process_capture")
    assert nonfinite["kl_status"] == "nonfinite" and nonfinite["n_nonfinite"] == 1
    with pytest.raises(ValueError):
        audit.kl_status(float("nan"))


def retention(**overrides):
    kwargs = {
        "total_steps": 50,
        "save_freq": 1,
        "max_keep": 5,
        "export_safetensors": True,
        "enable_checkpoint": True,
        "registered_steps": [1],
        "zero_update_reference": {"kind": "converted_bundle_hash", "sha256": SHA},
        "scope": "synthetic",
    }
    kwargs.update(overrides)
    return audit.simulate_retention(**kwargs)


def test_e21_retention_known_answers():
    report = retention()
    assert report["surviving_checkpoints"] == [46, 47, 48, 49, 50]
    assert report["surviving_exports"] == [46, 47, 48, 49, 50]
    assert report["status"] == "blocked" and "registered_checkpoint_pruned" in codes(report)
    small = retention(total_steps=10, save_freq=2, max_keep=3, registered_steps=[6])
    assert small["saves"] == [2, 4, 6, 8, 10]
    assert small["surviving_checkpoints"] == [6, 8, 10]
    assert small["status"] == "passed"
    keep_all = retention(
        total_steps=7, save_freq=3, max_keep=-1, registered_steps=[3], export_safetensors=False
    )
    assert keep_all["saves"] == [3, 6, 7] and keep_all["surviving_checkpoints"] == [3, 6, 7]
    assert keep_all["surviving_exports"] == [7]  # intermediate exports off; final always exports
    assert "registered_export_not_retained" in codes(keep_all)


def test_e21_retention_negative_controls():
    assert "registered_step_not_saved" in codes(
        retention(total_steps=10, save_freq=4, registered_steps=[3])
    )
    trainer = retention(zero_update_reference={"kind": "trainer_save", "sha256": SHA})
    assert "zero_update_reference_not_a_trainer_save" in codes(trainer)
    assert retention()["trainer_saves_step_zero"] is False
    disabled = retention(enable_checkpoint=False, registered_steps="none")
    assert disabled["saves"] == [] and "no_final_export" in codes(disabled)
    unregistered = retention(
        save_freq=None, max_keep=None, registered_steps=None, zero_update_reference=None
    )
    assert unregistered["status"] == "blocked" and unregistered["surviving_checkpoints"] is None
    assert {
        "save_freq_unregistered",
        "max_keep_unregistered",
        "registered_steps_unregistered",
        "zero_update_reference_missing",
    } <= codes(unregistered)
    for bad in ({"max_keep": 0}, {"save_freq": 0}, {"registered_steps": [51]}):
        with pytest.raises(ValueError):
            retention(**bad)


@pytest.fixture(scope="module")
def pinned_functions():
    quiet = SimpleNamespace(info=lambda *a: None)
    status_ns = upstream.load(
        "cosmos_rl/status.py.txt",
        ["PolicyStatusManager.check_checkpoint_saving"],
        {"logger": quiet},
    )
    ckpt_ns = upstream.load(
        "cosmos_rl/checkpoint.py.txt",
        ["CheckpointMananger.save_check"],
        {"os": os, "logger": quiet},
    )
    return status_ns["check_checkpoint_saving"], ckpt_ns["save_check"]


def _pinned_retention(
    functions, total_steps, save_freq, max_keep, export_safetensors, enable_checkpoint
):
    """Run the pinned controller save decision and manager pruning with stub objects."""
    check_checkpoint_saving, save_check = functions
    deleted = []
    ckpt = SimpleNamespace(
        name="ckpt",
        ckpt_output_dir="/out/checkpoints",
        saved_ckpt_step_dirs=[],
        max_keep=max_keep,
        save_mode="sync",
        best_ckpt_abs_dir=None,
        _is_master_rank=lambda: True,
        _is_ckpt_dir_linked_as_best=lambda d: False,
        _delete_checkpoint=deleted.append,
    )
    config = SimpleNamespace(
        train=SimpleNamespace(
            ckpt=SimpleNamespace(
                save_freq_in_epoch=0, save_freq=save_freq, enable_checkpoint=enable_checkpoint
            )
        )
    )
    exports = set()
    for step in range(1, total_steps + 1):
        controller = SimpleNamespace(current_step=step, total_steps=total_steps, config=config)
        if check_checkpoint_saving(controller, required_rollouts=1):
            if step == total_steps or export_safetensors:
                exports.add(step)
            save_check(ckpt, step=step)
    pruned = {int(d.rsplit("_", 1)[1]) for d in deleted}
    kept = sorted(int(d.rsplit("_", 1)[1]) for d in ckpt.saved_ckpt_step_dirs)
    return kept, sorted(exports - pruned)


def test_e21_simulation_matches_pinned_source_on_a_grid(pinned_functions):
    for total, freq, keep, export, enabled in itertools.product(
        [1, 2, 7, 20], [1, 2, 3, 5, 25], [-1, 1, 2, 5], [True, False], [True, False]
    ):
        ours = retention(
            total_steps=total,
            save_freq=freq,
            max_keep=keep,
            export_safetensors=export,
            enable_checkpoint=enabled,
            registered_steps="none",
        )
        kept, exports = _pinned_retention(pinned_functions, total, freq, keep, export, enabled)
        assert ours["surviving_checkpoints"] == kept, (total, freq, keep)
        assert ours["surviving_exports"] == exports, (total, freq, keep, export)


def test_vendored_upstream_fixture_provenance_and_licences():
    import hashlib
    import json

    root = ROOT / "tests/fixtures/s4_upstream_checks"
    provenance = json.loads((root / "provenance.json").read_text())
    assert provenance["license"] == "Apache-2.0"
    listed = set(provenance["files"]) | {v["file"] for v in provenance["licenses"].values()}
    present = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.suffix in (".txt", ".LICENSE")
    }
    assert present == listed  # nothing unlisted, nothing missing
    for entry in provenance["licenses"].values():
        content = (root / entry["file"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == entry["sha256"]
        assert b"Apache License" in content and b"Version 2.0" in content
    for name, entry in provenance["files"].items():
        upstream.source_bytes(name)  # sha256 + git blob check
        assert entry["license_file"] in {v["file"] for v in provenance["licenses"].values()}
        assert entry["commit"] in entry["source_url"] and len(entry["commit"]) == 40
        assert b"Apache-2.0" in (root / name).read_bytes()[:400]  # SPDX header kept


# --- independent-review regressions (A9-03, A9-04, A9-16) ------------------------------

R1_COMMIT = "4cda35d22bb257f0936ac272397627b9309ca211"


def test_e16_model_package_commit_is_a_factor():
    differs = audit.audit_adapter_paths(
        path("train", model_package_commit=R1_COMMIT),
        path("gate"),
        expected_driver_commit=PIN,
        scope="synthetic",
    )
    assert differs["paths_equivalent"] is False
    row = {r["factor"]: r for r in differs["factors"]}["model_package_commit"]
    assert (row["train"], row["gate"], row["state"]) == (R1_COMMIT, PIN, "differs")
    assert "adapter_factor_differs" in codes(differs)
    unresolved = audit.audit_adapter_paths(
        path("train", model_package_commit=None),
        path("gate"),
        expected_driver_commit=PIN,
        scope="synthetic",
    )
    assert unresolved["status"] == "unverified" and unresolved["paths_equivalent"] is False
    assert "factor_unknown" in codes(unresolved)


def test_e21_kl_logged_text_zero_is_an_unverifiable_bound():
    printed = audit.kl_status(0.01, ["0.000000", "0.000000"], source="log_inferred")
    assert printed["kl_status"] == "unverified"
    assert printed["printed_zero"] == 2
    assert printed["penalty_mean_bounds"] == [-5e-7, 5e-7]  # '%.6f' half-unit on each side
    assert "kl_beta" in printed["unit"]
    logged = audit.kl_status(0.01, ["0.000040", "0.000060"], source="log_inferred")
    assert logged["kl_status"] == "computed"
    assert logged["penalty_mean"] == pytest.approx(5e-5)
    assert logged["kl_mean"] == pytest.approx(5e-3)  # penalty / kl_beta
    exact = audit.kl_status(0.1, [0.0005], source="in_process_capture")
    assert exact["kl_status"] == "computed" and exact["kl_mean"] == pytest.approx(0.005)
    assert exact["penalty_mean_bounds"] == [0.0005, 0.0005]
    for bad in (
        lambda: audit.kl_status(0.01, [0.0], source="log_inferred"),  # not the printed text
        lambda: audit.kl_status(0.01, ["5e-7"], source="log_inferred"),
        lambda: audit.kl_status(0.01, [0.0]),  # source required with values
    ):
        with pytest.raises(ValueError):
            bad()


def test_e21_retention_needs_explicit_none_or_nonempty_registered_steps():
    with pytest.raises(ValueError):
        retention(registered_steps=[])
    none = retention(registered_steps="none")
    assert none["status"] == "passed" and none["registered"] == "none"
