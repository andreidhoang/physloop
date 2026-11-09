"""Version fence, adapter audit, uncertainty axes, KL status and retention (E05, E16, E17, E21).

Serves S4.1/S4.4/S4.6 [C] (G9, G7); build-spec A9. Agent-authored verification:

- **E05** ``check_version_fence``: the pinned trainer filter
  (``alpagym@972d160:.../cosmos/rollout_filter.py:64-73``) only *warns* when it keeps
  a rollout older than ``current_step - allowed_outdated_steps``; packed rows carry
  ``weight_version=0`` (``packer.py:213, 237``). A qualification fence therefore needs
  the behaviour version captured at generation (H4) and rejects any stale or
  misreported rollout. L1; dispatcher path on a host L3.
- **E16** ``audit_adapter_paths``: the train path (AlpaGym policy) and the gate path
  (e2e_challenge driver) as explicit factor records - image size, chain-of-causation
  (CoC) text, RoPE offsets, sampler (e.g. SDE-5 vs ODE-10), control rate (5 vs
  10 Hz), controller (linear vs nonlinear MPC) - plus the resolved model-package
  commit and ``vlm_name_or_path``/revision, all compared train vs gate (an unresolved
  commit on either path is ``unknown``). Every difference is reported; nothing is
  explained away and no expected-difference field is accepted. L2 (values captured
  from pinned configs); host resolution L3.
- **E17** ``uncertainty_axes``: training-run and road axes counted separately;
  rollout multiplicity is never road support. L1.
- **E21** ``kl_status``: with ``kl_beta <= 0`` no reference exists
  (``trainer.py:614-615``) and the logged ``kl_avg`` 0.0 is *not computed*, never zero
  divergence. With ``kl_beta > 0`` the logged ``train/kl_avg`` is the beta-scaled
  penalty ``kl_beta * mean(valid-row KL)`` averaged over minibatches
  (``trainer.py:413-419``, ``replay_objective.py:80``), not KL; the log line prints it
  with ``%.6f``, so a printed ``0.000000`` is only a bound (``unverified``). ``simulate_retention`` reproduces Cosmos step-mode saving
  (``status.py:1275-1308``: save when ``step % save_freq == 0`` and always at the last
  step, only with checkpointing enabled) and pruning (``checkpoint.py:759-796``: after
  each save, if more than ``max_keep`` are kept and ``max_keep != -1``, delete the
  oldest *checkpoint and its safetensors export*, one per save). There is no step-0 save, so the
  zero-update reference can never be a trainer save. L1, checked against the pinned
  source executed in the tests. Assumptions (fail closed otherwise): fresh run (no
  pre-existing ``step_N`` directories), step-based saving (``save_freq_in_epoch <= 0``),
  validation off (no best-checkpoint protection).

Owner inputs (required, no defaults): ``allowed_outdated_steps``, ``save_freq``,
``max_keep``, the registered checkpoint steps and the zero-update reference.

This module never imports torch, never edits a config, never prunes or copies a
checkpoint and never interprets a train/gate difference as a model effect.
"""

import math
import re

from teacher_rl.checks import Issues, artifact, check_scope, is_count, is_real, is_text, require

PINNED_FILTER = (
    "alpagym@972d160eed0e23d388497851504a3a233fec5879:"
    "packages/runtime/src/alpagym_runtime/cosmos/rollout_filter.py"
)
PINNED_RETENTION = (
    "cosmos-rl@d2a2c57c4bd6496482bc42da19a59b4432705eda:"
    "cosmos_rl/{dispatcher/status.py,utils/checkpoint.py}"
)
FACTORS = ("image_size", "coc", "rope_offsets", "sampler", "control_rate_hz", "controller")
COMPARED = (*FACTORS, "model_package_commit", "vlm_name_or_path", "vlm_revision")
KL_SOURCES = ("in_process_capture", "log_inferred")
_LOGGED_KL = re.compile(r"-?\d+\.\d{6}|nan|inf|-inf")
_KL_TOKENS = ("nan", "inf", "-inf")
LOG_HALF_ULP = 5e-7  # '%.6f' rounds to within half of 1e-6
PATH_KEYS = set(FACTORS) | {"path_id", "model_package_commit", "vlm_name_or_path", "vlm_revision"}
REFERENCE_KINDS = ("converted_bundle_hash", "zero_update_export")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


def check_version_fence(rollouts, *, current_step, allowed_outdated_steps, scope):
    """E05: reject stale or misreported rollouts the pinned filter would only warn on.

    ``rollouts``: ``[{rollout_id, behaviour_version, stamped_version}]`` where
    ``behaviour_version`` is the policy version captured at generation (int or
    ``None`` if not captured) and ``stamped_version`` the ``weight_version`` the
    trainer received. Stale: ``behaviour_version < current_step -
    allowed_outdated_steps``.
    """
    check_scope(scope)
    require(is_count(current_step), "current_step must be an int >= 0")
    require(
        allowed_outdated_steps is None or is_count(allowed_outdated_steps),
        "allowed_outdated_steps must be an int >= 0 or None",
    )
    require(type(rollouts) is list and rollouts, "rollouts must be a nonempty list")
    keys = {"rollout_id", "behaviour_version", "stamped_version"}
    ids = set()
    for i, r in enumerate(rollouts):
        require(type(r) is dict and set(r) == keys, f"rollouts[{i}] needs exactly {sorted(keys)}")
        require(
            is_text(r["rollout_id"]) and r["rollout_id"] not in ids,
            f"rollouts[{i}].rollout_id must be unique text",
        )
        ids.add(r["rollout_id"])
        for k in ("behaviour_version", "stamped_version"):
            require(
                r[k] is None or is_count(r[k]), f"rollouts[{i}].{k} must be an int >= 0 or None"
            )
    issues = Issues()
    if allowed_outdated_steps is None:
        issues.add(
            "allowed_outdated_steps_unregistered",
            "allowed_outdated_steps",
            "Allowed lag is an owner input.",
        )
    rows = []
    for r in sorted(rollouts, key=lambda x: x["rollout_id"]):
        path = f"rollout:{r['rollout_id']}"
        version = r["behaviour_version"]
        lag = None if version is None else current_step - version
        stale = None
        if version is None:
            issues.add(
                "behaviour_version_missing",
                path,
                "Generation-time version not captured.",
                severity="warning",
            )
        elif version > current_step:
            issues.add(
                "behaviour_version_from_future", path, "Behaviour version exceeds the trainer step."
            )
        elif allowed_outdated_steps is not None:
            stale = version < current_step - allowed_outdated_steps
            if stale:
                issues.add(
                    "stale_rollout_consumed",
                    path,
                    f"Lag {lag} exceeds {allowed_outdated_steps}; "
                    "the pinned filter only warns (:64-73).",
                )
        if (
            r["stamped_version"] is not None
            and version is not None
            and r["stamped_version"] != version
        ):
            issues.add(
                "version_stamp_mismatch",
                path,
                "Stamped weight_version differs from the captured version.",
            )
        rows.append({**r, "lag": lag, "stale": stale})
    if issues.has("error"):
        status = "blocked"
    elif "behaviour_version_missing" in issues.codes():
        status = "unverified"
    else:
        status = "passed"
    return artifact(
        "s4_e05_version_fence",
        module=__name__,
        module_file=__file__,
        scope=scope,
        inputs={
            "rollouts": rollouts,
            "current_step": current_step,
            "allowed_outdated_steps": allowed_outdated_steps,
        },
        units={"version": "policy (trainer) steps", "lag": "policy steps"},
        status=status,
        issues=issues,
        body={
            "ticket": "E05",
            "pinned_source": PINNED_FILTER,
            "upstream_behaviour": (
                "keeps the freshest train_batch_per_replica and warns on stale; never rejects"
            ),
            "rollouts": rows,
        },
    )


def _check_factor(name, value, path):
    if value is None:
        return
    if name == "image_size":
        require(
            type(value) is list
            and len(value) == 2
            and all(type(v) is int and v > 0 for v in value),
            f"{path}.image_size must be [height, width]",
        )
    elif name == "coc":
        require(type(value) is bool, f"{path}.coc must be bool")
    elif name == "sampler":
        require(
            type(value) is dict
            and set(value) == {"kind", "steps"}
            and value["kind"] in ("sde", "ode")
            and type(value["steps"]) is int
            and value["steps"] > 0,
            f"{path}.sampler must be {{kind: sde|ode, steps: int > 0}}",
        )
    elif name == "control_rate_hz":
        require(is_real(value) and value > 0, f"{path}.control_rate_hz must be > 0")
    else:
        require(is_text(value), f"{path}.{name} must be text")


def _check_path(descriptor, path):
    require(
        type(descriptor) is dict and set(descriptor) == PATH_KEYS,
        f"{path} needs exactly {sorted(PATH_KEYS)}",
    )
    require(is_text(descriptor["path_id"]), f"{path}.path_id must be text")
    for name in FACTORS:
        _check_factor(name, descriptor[name], path)
    commit = descriptor["model_package_commit"]
    require(
        commit is None or (type(commit) is str and _HEX40.fullmatch(commit)),
        f"{path}.model_package_commit must be 40-hex or None",
    )
    for key in ("vlm_name_or_path", "vlm_revision"):
        require(
            descriptor[key] is None or is_text(descriptor[key]),
            f"{path}.{key} must be text or None",
        )


def audit_adapter_paths(train_path, gate_path, *, expected_driver_commit, scope):
    """E16: factor-by-factor train/gate adapter audit (L2).

    ``expected_driver_commit`` is the engineering pin of the gate driver's
    ``alpamayo1_5`` (40-hex, e.g. ``36aeb4c5938cbc2eb2aed33b22434773da4ab639``); an
    unresolved or different gate commit is reported. The model-package commit is also
    a compared row (train vs gate), so different model code is never equivalent.
    Unknown (``None``) values make the audit ``unverified``; differences are
    ``adapter_factor_differs`` warnings and ``paths_equivalent`` is ``False``.
    """
    check_scope(scope)
    _check_path(train_path, "train_path")
    _check_path(gate_path, "gate_path")
    require(
        type(expected_driver_commit) is str and _HEX40.fullmatch(expected_driver_commit),
        "expected_driver_commit must be 40-hex",
    )
    issues = Issues()
    table = []
    for name in COMPARED:
        a, b = train_path[name], gate_path[name]
        state = "unknown" if a is None or b is None else ("equal" if a == b else "differs")
        if state == "unknown":
            issues.add(
                "factor_unknown",
                f"factor:{name}",
                "Factor not resolved on one path.",
                severity="warning",
            )
        elif state == "differs":
            issues.add(
                "adapter_factor_differs",
                f"factor:{name}",
                f"train {a!r} vs gate {b!r}.",
                severity="warning",
            )
        table.append({"factor": name, "train": a, "gate": b, "state": state})
    commit = gate_path["model_package_commit"]
    if commit is None:
        issues.add(
            "driver_commit_unresolved",
            "gate_path.model_package_commit",
            "The driver installs alpamayo1_5 without a revision; resolve it.",
        )
    elif commit != expected_driver_commit:
        issues.add(
            "driver_commit_not_pinned",
            "gate_path.model_package_commit",
            f"{commit} != pinned {expected_driver_commit}.",
        )
    states = {row["state"] for row in table}
    if issues.has("error"):
        status = "blocked"
    elif "unknown" in states:
        status = "unverified"
    else:
        status = "passed"
    return artifact(
        "s4_e16_adapter_audit",
        module=__name__,
        module_file=__file__,
        scope=scope,
        inputs={
            "train_path": train_path,
            "gate_path": gate_path,
            "expected_driver_commit": expected_driver_commit,
        },
        units={
            "image_size": "pixels [height, width]",
            "control_rate_hz": "Hz",
            "sampler.steps": "integration steps",
        },
        status=status,
        issues=issues,
        body={
            "ticket": "E16",
            "status_meaning": (
                "audit complete; differences are reported, never attributed to the model"
            ),
            "paths_equivalent": states == {"equal"},
            "factors": table,
            "train_model_package_commit": train_path["model_package_commit"],
            "gate_model_package_commit": commit,
        },
    )


def uncertainty_axes(evaluations, *, scope):
    """E17: count training-run and road axes separately (L1).

    ``evaluations``: ``[{training_run_id, road_id, scene_id, replicate_id}]``. Road
    support is the number of distinct roads; repeated scenes or replicates on one road
    add rollouts, never road support. With fewer than two training runs the run-level
    (algorithm) variation is ``not_estimable``.
    """
    check_scope(scope)
    keys = {"training_run_id", "road_id", "scene_id", "replicate_id"}
    require(type(evaluations) is list and evaluations, "evaluations must be a nonempty list")
    for i, e in enumerate(evaluations):
        require(
            type(e) is dict and set(e) == keys and all(is_text(e[k]) for k in keys),
            f"evaluations[{i}] needs text {sorted(keys)}",
        )
    issues = Issues()
    seen, scene_road = set(), {}
    for e in evaluations:
        key = (e["training_run_id"], e["road_id"], e["scene_id"], e["replicate_id"])
        if key in seen:
            issues.add(
                "duplicate_evaluation",
                f"run:{key[0]}/scene:{key[2]}",
                f"Repeated record {list(key)}.",
            )
        seen.add(key)
        if scene_road.setdefault(e["scene_id"], e["road_id"]) != e["road_id"]:
            issues.add(
                "scene_road_inconsistent",
                f"scene:{e['scene_id']}",
                "One scene mapped to two roads.",
            )
    runs = sorted({e["training_run_id"] for e in evaluations})
    roads = sorted({e["road_id"] for e in evaluations})
    roads_by_run = {
        r: sorted({e["road_id"] for e in evaluations if e["training_run_id"] == r}) for r in runs
    }
    runs_by_road = {
        d: sorted({e["training_run_id"] for e in evaluations if e["road_id"] == d}) for d in roads
    }
    rollouts_by_road = {d: sum(e["road_id"] == d for e in evaluations) for d in roads}
    if len(runs) < 2:
        issues.add(
            "run_level_not_estimable",
            "training_run_id",
            "Fewer than two independent training runs.",
            severity="warning",
        )
    return artifact(
        "s4_e17_uncertainty_axes",
        module=__name__,
        module_file=__file__,
        scope=scope,
        inputs={"evaluations": evaluations},
        units={
            "support": "distinct roads / distinct training runs",
            "rollouts": "evaluation records",
        },
        status="blocked" if issues.has("error") else "passed",
        issues=issues,
        body={
            "ticket": "E17",
            "n_training_runs": len(runs),
            "n_roads": len(roads),
            "n_scenes": len({e["scene_id"] for e in evaluations}),
            "n_rollouts": len(evaluations),
            "road_support": len(roads),
            "run_support": len(runs),
            "run_level_variation": "not_estimable" if len(runs) < 2 else "estimable_descriptively",
            "crossed": all(len(v) == len(runs) for v in runs_by_road.values()),
            "shared_roads": [d for d in roads if len(runs_by_road[d]) > 1],
            "roads_by_run": roads_by_run,
            "rollouts_by_road": rollouts_by_road,
            "note": "rollout multiplicity is never counted as independent road support",
        },
    )


def _parse_kl(value, source):
    """Return the logged penalty as a float (``nan``/``inf`` for non-finite tokens)."""
    if source == "in_process_capture":
        if value in _KL_TOKENS:
            return float(value)
        require(is_real(value), "captured kl_avg must be a finite number or a token")
        return float(value)
    require(
        type(value) is str and _LOGGED_KL.fullmatch(value) is not None,
        "log_inferred kl_avg must be the '%.6f' text the trainer prints",
    )
    return float(value)


def kl_status(kl_beta, logged_kl_values=None, *, source=None):
    """E21: KL status from the run's ``kl_beta`` and the logged ``train/kl_avg`` (L1).

    ``kl_beta <= 0``: ``not_computed`` whatever the log prints. Otherwise
    ``logged_kl_values`` with ``source`` (required with values): ``in_process_capture``
    (the exact ``train/kl_avg`` floats, or tokens ``nan``/``inf``/``-inf``) or
    ``log_inferred`` (the printed ``%.6f`` text). Reported in the logged unit (the
    beta-scaled penalty) and as KL (``penalty / kl_beta``), with bounds that carry the
    print precision; any printed ``0.000000`` makes the status ``unverified``, a
    non-finite value ``nonfinite``; no values is ``not_logged``.
    """
    require(is_real(kl_beta), "kl_beta must be a finite number")
    values = None
    if logged_kl_values is not None:
        require(source in KL_SOURCES, f"source must be one of {KL_SOURCES} with logged values")
        require(type(logged_kl_values) is list, "logged_kl_values must be a list")
        values = [_parse_kl(v, source) for v in logged_kl_values]
    else:
        require(source is None or source in KL_SOURCES, f"source must be one of {KL_SOURCES}")
    if kl_beta <= 0:
        return {
            "kl_status": "not_computed",
            "kl_beta": kl_beta,
            "logged_values_ignored": 0 if logged_kl_values is None else len(logged_kl_values),
            "note": (
                "no reference model exists (trainer.py:614-615); "
                "a logged 0.0 is not zero divergence"
            ),
        }
    values = values or []
    finite = [v for v in values if math.isfinite(v)]
    printed_zero = sum(v == 0.0 for v in finite) if source == "log_inferred" and values else None
    half = LOG_HALF_ULP if source == "log_inferred" else 0.0
    mean = math.fsum(finite) / len(finite) if finite else None
    if not values:
        status = "not_logged"
    elif len(finite) != len(values):
        status = "nonfinite"
    elif printed_zero:
        status = "unverified"
    else:
        status = "computed"
    bounds = None if mean is None else [mean - half, mean + half]
    return {
        "kl_status": status,
        "kl_beta": kl_beta,
        "source": source,
        "unit": (
            "train/kl_avg = kl_beta * mean valid-row KL, averaged over minibatches "
            "(beta-scaled penalty); kl_mean = penalty / kl_beta"
        ),
        "n_values": len(values),
        "n_nonfinite": len(values) - len(finite),
        "printed_zero": printed_zero,
        "penalty_mean": mean,
        "penalty_max": max(finite) if finite else None,
        "penalty_mean_bounds": bounds,
        "kl_mean": None if mean is None else mean / kl_beta,
        "kl_mean_bounds": None if bounds is None else [b / kl_beta for b in bounds],
    }


def _retention_events(total_steps, save_freq, max_keep, export_safetensors, enable_checkpoint):
    saved, checkpoints, exports, saves, deletions = [], set(), set(), [], []
    for step in range(1, total_steps + 1):
        last = step == total_steps
        if not enable_checkpoint or not (last or step % save_freq == 0):
            continue
        saves.append(step)
        if last or export_safetensors:
            exports.add(step)
        checkpoints.add(step)
        saved.append(step)
        if len(saved) > max_keep and max_keep != -1:
            victim = saved.pop(0)
            checkpoints.discard(victim)
            exports.discard(victim)
            deletions.append({"at_step": step, "deleted_step": victim})
    return saves, deletions, sorted(checkpoints), sorted(exports)


def simulate_retention(
    *,
    total_steps,
    save_freq,
    max_keep,
    export_safetensors,
    enable_checkpoint,
    registered_steps,
    zero_update_reference,
    scope,
):
    """E21: which checkpoints and exports survive pinned pruning (L1).

    ``registered_steps``: the owner-registered trainer steps whose checkpoint *and*
    safetensors export must survive (e.g. early snapshots for drift checks): a
    nonempty list, or the explicit marker ``"none"`` when the owner registers no
    step (an empty list is refused: it cannot be told from an empty capture).
    ``zero_update_reference``: ``{"kind": "converted_bundle_hash" |
    "zero_update_export", "sha256": hex64}``; a trainer save cannot serve because no
    step-0 save exists. ``max_keep`` must be ``-1`` (keep all) or >= 1.
    """
    check_scope(scope)
    require(type(total_steps) is int and total_steps >= 1, "total_steps must be a positive int")
    require(
        type(export_safetensors) is bool and type(enable_checkpoint) is bool,
        "export/enable flags must be bool",
    )
    require(
        save_freq is None or (type(save_freq) is int and save_freq >= 1),
        "save_freq must be an int >= 1 or None",
    )
    require(
        max_keep is None or (type(max_keep) is int and (max_keep == -1 or max_keep >= 1)),
        "max_keep must be -1, >= 1 or None",
    )
    if registered_steps is not None and registered_steps != "none":
        require(
            type(registered_steps) is list
            and registered_steps
            and all(type(s) is int and 1 <= s <= total_steps for s in registered_steps)
            and len(set(registered_steps)) == len(registered_steps),
            "registered_steps must be 'none' or a nonempty list of distinct ints "
            "in [1, total_steps]",
        )
    if zero_update_reference is not None:
        require(
            type(zero_update_reference) is dict
            and set(zero_update_reference) == {"kind", "sha256"}
            and type(zero_update_reference["kind"]) is str,
            "zero_update_reference must be {kind, sha256}",
        )
    issues = Issues()
    for name, value in (
        ("save_freq", save_freq),
        ("max_keep", max_keep),
        ("registered_steps", registered_steps),
    ):
        if value is None:
            issues.add(f"{name}_unregistered", name, f"{name} is an owner input.")
    if zero_update_reference is None:
        issues.add(
            "zero_update_reference_missing",
            "zero_update_reference",
            "Register a hashed bundle or zero-update export.",
        )
    elif zero_update_reference["kind"] == "trainer_save":
        issues.add(
            "zero_update_reference_not_a_trainer_save",
            "zero_update_reference",
            "Cosmos never saves step 0.",
        )
    elif zero_update_reference["kind"] not in REFERENCE_KINDS:
        issues.add(
            "zero_update_reference_kind_unknown",
            "zero_update_reference.kind",
            f"Expected one of {REFERENCE_KINDS}.",
        )
    elif not (
        type(zero_update_reference["sha256"]) is str
        and _HEX64.fullmatch(zero_update_reference["sha256"])
    ):
        issues.add(
            "zero_update_reference_unhashed",
            "zero_update_reference.sha256",
            "Reference needs a sha256.",
        )
    body = {
        "ticket": "E21",
        "pinned_source": PINNED_RETENTION,
        "assumptions": [
            "fresh run",
            "step-based saving (save_freq_in_epoch <= 0)",
            "validation off",
        ],
        "trainer_saves_step_zero": False,
        "saves": None,
        "deletions": None,
        "surviving_checkpoints": None,
        "surviving_exports": None,
        "registered": None,
    }
    if save_freq is not None and max_keep is not None:
        saves, deletions, checkpoints, exports = _retention_events(
            total_steps, save_freq, max_keep, export_safetensors, enable_checkpoint
        )
        body.update(
            saves=saves,
            deletions=deletions,
            surviving_checkpoints=checkpoints,
            surviving_exports=exports,
        )
        if total_steps not in exports:
            issues.add(
                "no_final_export",
                "enable_checkpoint",
                "Without checkpointing no final export is written.",
            )
        if registered_steps == "none":
            body["registered"] = "none"
        elif registered_steps is not None:
            rows = []
            for step in sorted(registered_steps):
                row = {
                    "step": step,
                    "saved": step in saves,
                    "checkpoint_survives": step in checkpoints,
                    "export_survives": step in exports,
                }
                rows.append(row)
                path = f"registered_step:{step}"
                if not row["saved"]:
                    issues.add(
                        "registered_step_not_saved", path, "save_freq never saves this step."
                    )
                elif not row["checkpoint_survives"]:
                    issues.add(
                        "registered_checkpoint_pruned",
                        path,
                        "max_keep pruning deletes this checkpoint.",
                    )
                if row["saved"] and not row["export_survives"]:
                    issues.add(
                        "registered_export_not_retained",
                        path,
                        "No surviving safetensors export for this step.",
                    )
            body["registered"] = rows
    return artifact(
        "s4_e21_retention_simulation",
        module=__name__,
        module_file=__file__,
        scope=scope,
        inputs={
            "total_steps": total_steps,
            "save_freq": save_freq,
            "max_keep": max_keep,
            "export_safetensors": export_safetensors,
            "enable_checkpoint": enable_checkpoint,
            "registered_steps": registered_steps,
            "zero_update_reference": zero_update_reference,
        },
        units={"steps": "trainer (controller) steps, 1-indexed"},
        status="blocked" if issues.has("error") else "passed",
        issues=issues,
        body=body,
    )
