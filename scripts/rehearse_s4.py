"""Synthetic end-to-end S4 rehearsal with literal answers (build-spec B3 item 8).

Serves S4.1-S4.4/S4.6 [C] (G9, G7) under D-024. Agent-authored evaluator engineering
(D-018). Chain: registration -> compiled overrides -> H1 launch argv (TOML written in memory)
-> two-GPU bootstrap plan -> fake capture events through the real ``EventSink`` and
``read_stream`` -> attempt ledger -> learning-signal / step / skip checks -> G3 capture to
checks (the rollout stream's ``simulate_call`` reply blobs decoded by ``teacher_rl.simreturn``
into E10/E11/E14; the trainer stream's F5 records into E20/E07, its snapshots into E03, its
old/new log-probs into E04 under a strict-parity trace; a second capture with new session
UUIDs and the same seeds into E01) -> diagnostics ->
watchdog over fake processes with hand-off to the A3 lifecycle -> cost forecast -> S2
evaluation requests (each re-auditing its synthetic export chain) and the run-aggregation spec
read from an ``s4_run_registry`` v2 -> per-run ``gate.s2_gate.evaluate_registered`` on
synthetic records (primary endpoint and the synthetic guard endpoint) -> ``gate.superiority``
decisions bound to the registered guards, strata and look -> run aggregation (F4).

Every input is synthetic (``scope: "synthetic"``): the registration values exercise the schema
and are never owner (Huy) choices; authorization references are ``SYNTHETIC-*`` strings that
grant nothing. Nothing provisions, downloads, launches or touches the network: the watchdog's
processes, clock and command runner are fakes, and the only file effects are the capture
streams and their reply blobs written under the caller's ``--workdir`` (``capture/`` and the
E01 repeat ``capture_b/``). The synthetic capture builders (``write_synthetic_capture`` and
the event/reply helpers) are shared with ``tests/test_s4_adapters.py`` and
``tests/test_s4_replay_checks.py``; their values (seeds, rewards, log-probs, the E04
``max_ulp`` 8 stand-in, the ``collision_any`` required metric) exercise the schema and are
never owner choices. ``capture_checks.semantics.status`` is ``blocked`` by design: the
silent-retry attempt's reply has no collision series, and E11 detecting that is the intended
negative control inside the rehearsal, not a defect.

Verification ceiling: the rehearsal is **L1** (deterministic; ``tests/test_s4_rehearsal.py``
pins its literal summary). It proves the modules compose on synthetic data; it qualifies no
host, input, rental or scientific claim.

Usage::

    python scripts/rehearse_s4.py --workdir /abs/new/dir
"""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gate import superiority
from gate.s2_bootstrap import RunResult
from gate.s2_identity import canonical_json_bytes, identity_hash
from gate.tensor_manifest import build_manifest
from teacher_rl import pins

SCENE_A = "clipgt-a309e228-26e1-423e-a44c-cb00aa7378cb"
SCENE_B = "clipgt-01d503d4-449b-46fc-8d78-9085e70d3554"
SCENE_C = "clipgt-37113280-a033-4be4-afdd-cb199299e028"
REASON2 = "b" * 40  # synthetic revision; the real one is an admission-time input
RENDERER = "nvcr.io/nvidia/nre/nre-ga@sha256:" + "4" * 64
RUN_DIR_NAME = "20270115T080000Z-" + "0" * 32
RUN_IDS = ["run-a", "run-b", "run-c"]
PRIMARY_METRIC = {"name": "collision_any", "kind": "binary", "higher_is_worse": True,
                  "unit": "indicator"}
# Synthetic guard endpoint (schema exercise; the registered guards are Huy's). Its
# ``s2_protocol_hash`` (review F4-03) is filled below, once ``_s2_guard_registration`` exists.
_GUARD_ENDPOINT = {"name": "hard_brake_guard", "s2_registration_id": "synthetic-s2-guard",
                   "metric": {"name": "hard_brake_any", "kind": "binary",
                              "higher_is_worse": True, "unit": "indicator"},
                   "margin": 0.1, "alpha": 0.05}


def synthetic_registration(route="h1_resolved_toml"):
    """A complete synthetic S4 registration (schema exercise; not owner choices)."""
    return {
        "artifact_kind": "s4_run_registration", "schema_version": 2, "scope": "synthetic",
        "registration_id": "synthetic-s4-b3",
        "engineering": {
            "pins": {
                "alpagym": pins.UPSTREAM_PINS["alpagym"]["commit"],
                "alpamayo_recipes": pins.UPSTREAM_PINS["alpamayo_recipes"]["commit"],
                "cosmos_rl": pins.UPSTREAM_PINS["cosmos_rl"]["commit"],
                "training_alpasim": pins.UPSTREAM_PINS["training_alpasim"]["commit"],
                "alpamayo_r1": pins.UPSTREAM_PINS["alpamayo_r1"]["commit"],
                "teacher_revision": pins.TEACHER["revision"],
                "alpagym_uv_lock_sha256": pins.ALPAGYM_LOCK["uv_lock_sha256"],
            },
            "cosmos_reason2_revision": REASON2,
            "base_experiment": "alpamayo_1_5_local_2gpu_smoke",
            "policy_model_path": "/data/ckpt/alpagym-bundle",
            "run_root": "/data/alpagym-runs",
            "config_route": route,
        },
        "owner": {
            "group_size": 2, "rollouts_per_trainer_step": 2, "learning_rate": 0.0001,
            "lr_schedule": {"decay_type": "none", "decay_ratio": 0.0, "min_lr_factor": 0.0},
            "warmup": {"steps": 1, "start_factor": 0.0},
            "scene_set": [SCENE_A],
            "optimizer_steps_per_run": {"trainer_steps": 3, "epochs": 1, "mini_batch_rows": 1,
                                        "optimizer_iterations": 1},
            "checkpoint_retention": {"save_freq": 1, "max_keep": -1,
                                     "export_safetensors": True},
            "anchor": {"kind": "none"},
            "scene_roles": {"training": [SCENE_A], "development": [SCENE_B],
                            "confirmation": [SCENE_C]},
            "overlap_rule": {"rule_ref": "synthetic-rule-ref", "summary": "synthetic summary"},
            "evaluated_policy_path": "synthetic evaluated path",
            # Synthetic stand-ins for the F4 superiority slots (guard endpoint, guard and
            # gating strata, look), not owner values.
            "improvement_decision_rule": {
                "rule_id": superiority.RULE_ID, "registration_ref": "synthetic-ref",
                "guards": [dict(SYNTHETIC_GUARD)], "guard_strata": "primary_strata",
                "gating_strata": "all_guard_strata", "look": {"kind": "fixed"}},
            "smallest_improvement_of_interest": {"value": 0.03, "unit": "synthetic unit"},
            "seed_modes": {"train_seed": 7 if route == "h1_resolved_toml" else "unseeded",
                           "dataloader_seed": 0, "rollout_force_determinism": True,
                           "simulator": "upstream_random_unsettable"},
            "version_policy": {"on_policy": True, "allowed_outdated_steps": 0,
                               "prefetch_rollout": False},
            "reward_arm": {"arm": "A", "preset": "progress_safety"},
            "e07_pass_criterion": {"kind": "float32_tolerance", "tolerance": 1e-6},
            "parity_tolerances": {
                "log_density": {"abs": 0.0, "rel": 0.0, "justification_ref": "synthetic"},
                "action_trajectory": {"abs": 0.0, "rel": 0.0, "ulp": 0,
                                      "justification_ref": "synthetic"},
            },
            "video": {"render_video": False}, "decode_device": "cuda_nvjpeg",
            "independent_training_runs": 3,
            # Synthetic stand-in, not an owner value (matches the synthetic run registry).
            "run_aggregation": {"rule": "all", "k": None, "family_alpha": 0.05},
            "prediction_ref": "experiments/2026-11-09-synthetic-s4.md",
            # Synthetic stand-in for the AlpaGym issue #10 response, not an owner value.
            "response_to_alpagym_issue_10": {"response": "accept_as_shipped",
                                             "record_ref": "SYNTHETIC-issue-10-record"},
        },
    }


def synthetic_inventory():
    """A synthetic S4 runtime inventory (sizes/hashes are fixtures, not observed bytes)."""
    home = "/home/ubuntu"
    return {
        "artifact_kind": "s4_runtime_inventory", "schema_version": 1, "scope": "synthetic",
        "sources": {
            "alpagym": {"repository": pins.UPSTREAM_PINS["alpagym"]["repository"],
                        "commit": pins.UPSTREAM_PINS["alpagym"]["commit"]},
            "training_alpasim": {
                "repository": pins.UPSTREAM_PINS["training_alpasim"]["repository"],
                "commit": pins.UPSTREAM_PINS["training_alpasim"]["commit"],
                "tree": pins.UPSTREAM_PINS["training_alpasim"]["tree"]},
        },
        "alpagym_uv_lock_sha256": pins.ALPAGYM_LOCK["uv_lock_sha256"],
        "platform": "linux/amd64",
        "host_paths": {
            "physloop_root": f"{home}/physloop", "alpagym_checkout": "/data/src/alpagym",
            "alpasim_cache_dir": "/data/cache/alpagym-alpasim",
            "teacher_snapshot": "/data/models/alpamayo-1.5-10b",
            "converted_bundle": "/data/ckpt/alpagym-bundle", "hf_home": "/data/hf",
            "download_scratch": "/data/scratch", "evidence_dir": "/data/evidence",
            "docker_data_root": "/var/lib/docker", "home": home,
        },
        "images": [{"role": "renderer", "kind": "registry_digest", "reference": RENDERER,
                    "platform": "linux/amd64", "size_bytes": "unverified",
                    "authorization_ref": "SYNTHETIC-terms-renderer"}],
        "assets": [
            {"role": "teacher", "repository": pins.TEACHER["repository"], "repo_type": "model",
             "revision": pins.TEACHER["revision"],
             "authorization_ref": "SYNTHETIC-terms-teacher",
             "files": [{"path": "config.json", "size_bytes": 1000, "sha256": "1" * 64},
                       {"path": "model-00001-of-00002.safetensors", "size_bytes": 2000,
                        "sha256": "2" * 64},
                       {"path": "model-00002-of-00002.safetensors", "size_bytes": "unverified",
                        "sha256": "unverified"}]},
            {"role": "cosmos_reason2", "repository": pins.COSMOS_REASON2["repository"],
             "repo_type": "model", "revision": REASON2,
             "authorization_ref": "SYNTHETIC-terms-cosmos-reason2",
             "files": [{"path": "config.json", "size_bytes": 500, "sha256": "3" * 64},
                       {"path": "tokenizer.json", "size_bytes": 700, "sha256": "unverified"}]},
        ],
    }


def synthetic_bootstrap_record(phases=("host", "alpagym", "alpasim", "images", "assets",
                                       "convert", "nccl", "record")):
    return {"artifact_kind": "s4_bootstrap_action_record", "schema_version": 1,
            "action_record_ref": "SYNTHETIC-action-record",
            "authorized_phases": list(phases),
            "authorizations": {"host_setup": "SYNTHETIC-host-setup",
                               "acquire:renderer": "SYNTHETIC-acquire-renderer",
                               "acquire:teacher": "SYNTHETIC-acquire-teacher",
                               "acquire:cosmos_reason2": "SYNTHETIC-acquire-cosmos-reason2"}}


def synthetic_rental_record(gpu_count=2):
    """A synthetic gate.s2_execution action record for a two-GPU host (grants nothing)."""
    return {
        "artifact_kind": "s2_action_record", "schema_version": 1, "scope": "synthetic",
        "provider": "vast", "offer": {"offer_id": 1, "sku": "SYNTHETIC 2x GPU"},
        "region": "SYNTHETIC",
        "resources": {"gpu_count": gpu_count, "gpu_edition": "RTX PRO 6000 WS",
                      "disk_gb": 400, "image": "docker.io/vastai/kvm:ubuntu", "volumes": []},
        "max_hourly_usd": "3.50",
        "window": {"start_utc": "2027-01-15T08:00:00+00:00",
                   "end_utc": "2027-01-15T18:00:00+00:00"},
        "lifecycle_limits": {"startup_limit_seconds": 1200, "arm_limit_seconds": 120,
                             "export_attempt_limit": 2},
        "retention_after_stop": {"mode": "destroy_after_stop",
                                 "authority_ref": "SYNTHETIC-retention",
                                 "retain_until_utc": None},
        "export_destination": "evidence/s4-synthetic-export",
        "cap_headroom_snapshot": {
            "as_of_utc": "2027-01-15T07:00:00+00:00", "tier": "stage",
            "caps_usd": {"pilot": "100", "stage": "1200", "project": "2500"},
            "used_usd": {"pilot": "0.6", "stage": "0.6", "project": "0.6"},
            "liability_usd": {"pilot": "0", "stage": "0", "project": "0"}},
        "approval": {"decision_id": "D-999",
                     "quoted_instruction": "SYNTHETIC rehearsal record; grants nothing."},
        "license_attestation_refs": ["SYNTHETIC-terms-cosmos-reason2",
                                     "SYNTHETIC-terms-renderer", "SYNTHETIC-terms-teacher"],
        "scientific_registration_ref": None, "engineering_only": True,
    }


def synthetic_base_cosmos_toml():
    """Bytes shaped like the host-written cosmos_config.toml (run_artifacts.py:93-138).

    The host writes ``asdict(config.cosmos)`` minus nulls, so every registered ``cosmos.*``
    value appears under its TOML name (``num_epochs`` -> ``epoch``,
    ``grpo_optimization_iterations`` -> ``mu_iterations``) next to the fixed fields the host
    adds. Synthetic values equal the synthetic registration's.
    """
    return (
        'mode = "colocated"\n\n[train]\nepoch = 1\noutput_dir = "/data/alpagym-runs/'
        + RUN_DIR_NAME + '/cosmos"\nresume = false\nnon_text = true\ncompile = false\n'
        'optm_lr = 0.0001\noptm_decay_type = "none"\noptm_decay_ratio = 0.0\n'
        'optm_min_lr_factor = 0.0\noptm_warmup_steps = 1\nmax_num_steps = 3\n'
        'train_batch_per_replica = 2\n\n[train.ckpt]\nenable_checkpoint = true\n'
        'save_freq = 1\nmax_keep = -1\nexport_safetensors = true\n\n'
        '[train.train_policy]\ntype = "grpo"\n'
        'trainer_type = "alpagym_grpo"\nuse_remote_reward = false\nmini_batch = 1\n'
        'mu_iterations = 1\nkl_beta = 0.0\nreference_reset_interval = 0\non_policy = true\n'
        'allowed_outdated_steps = 0\n\n[policy]\n'
        'model_name_or_path = "/data/ckpt/alpagym-bundle"\n\n[rollout]\nn_generation = 2\n'
        'prefetch_rollout = false\n'
    ).encode()


# --- synthetic export chain (shapes per tests/test_s4_export_audit.py) -------------------


def _st(tensors, metadata=None):
    """Serialize ``{name: (dtype, shape, raw)}`` to safetensors bytes in name order."""
    import json
    import struct

    header, data, offset = {}, b"", 0
    if metadata is not None:
        header["__metadata__"] = metadata
    for name in sorted(tensors):
        dtype, shape, raw = tensors[name]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        data += raw
        offset += len(raw)
    blob = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<Q", len(blob)) + blob + data


def _bf16(*values):
    import struct

    return b"".join(struct.pack("<f", v)[2:] for v in values)


def _f32(*values):
    import struct

    return struct.pack(f"<{len(values)}f", *values)


SYNTHETIC_RELEASE = {
    "vlm.model.visual.proj.weight": ("BF16", [2], _bf16(1.0, 2.0)),
    "vlm.model.language_model.layers.0.weight": ("BF16", [2], _bf16(3.0, 4.0)),
    "vlm.lm_head.weight": ("BF16", [2], _bf16(5.0, 6.0)),
    "expert.layers.0.weight": ("BF16", [2], _bf16(7.0, 8.0)),
    "action_in_proj.proj.weight": ("BF16", [2], _bf16(9.0, 10.0)),
    "action_in_proj.encoder.freqs": ("F32", [1, 2], _f32(1.0, 100.0)),
    "action_out_proj.bias": ("F32", [2], _f32(0.5, -0.5)),
    "action_space.accel_mean": ("F32", [], _f32(0.25)),
}
SYNTHETIC_BUFFERS = {"action_in_proj.encoder.freqs", "action_space.accel_mean"}


def _shard(name):
    return "00000.safetensors" if name.startswith(("model.", "visual.", "lm_head")) else (
        "00001.safetensors")


def _sharded(tensors, shard_of, metadata=None):
    import json

    from teacher_rl.export_audit import EXPORT_INDEX_NAME

    shards = {}
    for name, value in tensors.items():
        shards.setdefault(shard_of(name), {})[name] = value
    files = {shard: _st(content, metadata) for shard, content in shards.items()}
    index = {"metadata": {"total_size": sum(len(raw) for _, _, raw in tensors.values())},
             "weight_map": {n: shard_of(n) for n in sorted(tensors)}}
    files[EXPORT_INDEX_NAME] = json.dumps(index).encode()
    return build_manifest(files, scope="synthetic", index_name=EXPORT_INDEX_NAME)


def _snapshot():
    import hashlib

    itemsize = {"F32": 4, "BF16": 2}
    tensors = {}
    for local, (dtype, shape, raw) in sorted(SYNTHETIC_RELEASE.items()):
        kind = "buffer" if local in SYNTHETIC_BUFFERS else "parameter"
        tensors[f"expert_model.{local}"] = {
            "kind": kind, "dtype": dtype, "shape": shape,
            "numel": len(raw) // itemsize[dtype], "byte_length": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "requires_grad": (not local.startswith("vlm.")) if kind == "parameter" else False,
            "persistent": True if kind == "buffer" else None}
    tensors = dict(sorted(tensors.items()))
    return {
        "artifact_kind": "s4_param_snapshot", "schema_version": 1, "scope": "synthetic",
        "producer": {"module": "teacher_rl.params", "source_sha256": "0" * 64},
        "input_hashes": {}, "units": {}, "hash_convention": "rule 11", "status": "passed",
        "source": "module", "tensors": tensors, "modules": {"": True},
        "tensor_set_sha256": identity_hash(
            {n: {k: e[k] for k in ("dtype", "shape", "byte_length", "sha256")}
             for n, e in tensors.items()}),
        "totals": {}, "issues": []}


def synthetic_export_chain(variant=0):
    """``(export_audit_inputs, inference_manifest)`` for one synthetic run's checkpoint.

    The expert weight differs per ``variant`` (a trained run changes bytes, never names,
    dtypes or shapes), so each run's inference checkpoint has its own tensor identity.
    """
    from teacher_rl import export_audit as ea

    export = {}
    for name in sorted(set(SYNTHETIC_RELEASE) - SYNTHETIC_BUFFERS):
        dtype, shape, raw = SYNTHETIC_RELEASE[name]
        if dtype == "F32":  # Cosmos exports in param_dtype bfloat16
            raw = b"".join(raw[i + 2:i + 4] for i in range(0, len(raw), 4))
            dtype = "BF16"
        if name == "expert.layers.0.weight":
            raw = _bf16(7.0 + variant, 8.0)
        export[ea.policy_name_to_export_key(name, hf_config_is_vlm=True)] = (dtype, shape, raw)
    renamed = {ea.export_key_to_inference_key(k): (k, v) for k, v in export.items()}
    inference = _sharded({n: v for n, (_, v) in renamed.items()},
                         lambda n: _shard(renamed[n][0]), metadata={"format": "pt"})
    inputs = {"release": build_manifest({"model.safetensors": _st(SYNTHETIC_RELEASE)},
                                        scope="synthetic"),
              "export": _sharded(export, _shard), "policy_snapshot": _snapshot(),
              "hf_config_is_vlm": True}
    return inputs, inference


def synthetic_run_registry(registration, *, rule="all", k=None, family_alpha=0.05):
    """An ``s4_run_registry`` v2 (synthetic run ids and aggregation inputs, not owner values)."""
    from teacher_rl.registration import validate_registration

    return {"artifact_kind": "s4_run_registry", "schema_version": 2, "scope": "synthetic",
            "registration_hash": validate_registration(registration)["registration_hash"],
            "run_ids": list(RUN_IDS),
            "aggregation": {"rule": rule, "k": k, "family_alpha": family_alpha}}


def _line(line_id, jobs, wall, reruns=0):
    return {"line_id": line_id, "jobs": jobs, "replicas": 1, "rerun_jobs": reruns,
            "wall_seconds_per_job": wall, "gpu_count": 2, "basis": "synthetic"}


def synthetic_forecast_input():
    return {
        "artifact_kind": "s4_cost_forecast_input", "schema_version": 1, "scope": "synthetic",
        "quote": {"artifact_kind": "s2_cost_quote", "schema_version": 1, "scope": "synthetic",
                  "currency": "USD", "provider": "vast", "rate_basis": "per_gpu",
                  "compute_usd_per_hour": "1.50", "storage_usd_per_hour": "0.10",
                  "transfer_usd_per_gb": {"download": "0.02", "upload": "0.05"},
                  "billing_granularity_seconds": 60,
                  "fees_and_tax": {"bound_usd": "1.00", "basis": "synthetic bound"}},
        "host_gpu_count": 2,
        "components": {"setup": [_line("bootstrap", 1, 1800, reruns=1)],
                       "training": [_line("baseline_runs", 3, 5430, reruns=1)],
                       "development_selection": [_line("checkpoint_evals", 6, 600)],
                       "locked_confirmation": [_line("fresh_runs", 3, 5400)]},
        "storage": [{"line_id": "host_disk", "instances": 1, "lifetime_seconds": 36000,
                     "basis": "synthetic"}],
        "transfer": {"download_gb": "100", "upload_gb": "20", "basis": "synthetic"},
        "failure_reserve": {"amount_usd": "5.00", "basis": "synthetic"},
        "ledger_scope": {"tier": "stage", "as_of_epoch": 200,
                         "caps_usd": {"pilot": "100", "stage": "1200", "project": "2500"},
                         "storage_retention_until_epoch": None},
        "actual_events": [
            {"event_id": "e1", "kind": "opening_credit", "amount_usd": "10.00",
             "as_of_epoch": 0},
            {"event_id": "e2", "kind": "cash_deposit", "amount_usd": "50.00", "at_epoch": 10},
            {"event_id": "e3", "kind": "credit_snapshot", "credit_usd": "60.00",
             "at_epoch": 100},
            {"event_id": "e4", "kind": "prior_usage", "amount_usd": "0.60",
             "decision_ref": "D-020", "counts_toward": "pilot"},
        ],
    }


# --- rehearsal ------------------------------------------------------------------------


def _codes(report):
    return sorted({i["code"] for i in report["issues"]})


# --- synthetic capture (G3): rollout and trainer streams through the real sink ------------

CAPTURE_LIMITS = {"max_event_bytes": 65_536, "max_stream_bytes": 1_048_576,
                  "max_blob_bytes": 65_536, "max_total_blob_bytes": 1_048_576}
SESSIONS_A = ("1" * 32, "2" * 32, "3" * 32)
SESSIONS_B = ("4" * 32, "5" * 32, "6" * 32)
SESSION_SEEDS = (101, 102, 103)
TIMESTAMPS_US = (0, 100_000, 200_000)
NONCE = "9" * 32
# E04 synthetic replay: row 0 replays bitwise; row 1 is 2**-20 above its stored float32
# old log-prob, i.e. 8 float32 ULP at magnitude 1.25 (spacing 2**-23 in [1, 2)).
OLD_LOGPROBS = (-0.5, -1.25)
NEW_LOGPROBS = (-0.5, -1.25 + 2.0 ** -20)
# Synthetic stand-in for a registered E04 tolerance (schema exercise; not an owner value).
SYNTHETIC_E04_TOLERANCE = {"registration_ref": "experiments/2026-11-09-synthetic-s4.md#e04",
                           "atol": None, "rtol": None, "max_ulp": 8}
H = {c: c * 64 for c in "0123456789abcdef"}


def _metric(name, values):
    return {"name": name, "timestamps_us": list(TIMESTAMPS_US), "values": list(values),
            "valid": [True] * len(values), "time_aggregation": 3}  # TIME_AGGREGATION_MAX


def synthetic_reply_bytes(session, *, collision):
    """A pinned-schema ``SimulationReturn`` for one session.

    ``collision`` is ``None`` (the scorer emitted no collision series, so the reward's
    ``collision_any`` lookup raises KeyError upstream), ``False`` or ``True`` (front and
    lateral contact at the last timestep). Every series is valid; the aggregates are the
    online maxima, as the pinned evaluator computes them, for every metric but
    ``eval_relevant`` (appended after aggregation upstream).
    """
    from teacher_rl import simreturn

    dense = [_metric("eval_relevant", [0.0, 1.0, 1.0]), _metric("offroad", [0.0] * 3),
             _metric("img_is_black", [0.0] * 3), _metric("collision_rear", [0.0] * 3)]
    if collision is not None:
        hit = [0.0, 0.0, 1.0 if collision else 0.0]
        dense += [_metric(n, hit) for n in ("collision_any", "collision_front",
                                            "collision_lateral")]
    message = simreturn.message_class()()
    ret = message.rollout_returns.add()
    ret.rollout_spec.scenario_id = SCENE_A
    ret.rollout_spec.nr_rollouts = 1
    ret.rollout_spec.session_uuids.append(session)
    ret.success = True
    ret.rollout_uuid = f"rollout-{session}"
    for metric in dense:
        added = ret.timestep_metrics.add()
        added.name = metric["name"]
        added.timestamps_us.extend(metric["timestamps_us"])
        added.values.extend(metric["values"])
        added.valid.extend(metric["valid"])
        added.time_aggregation = metric["time_aggregation"]
        if metric["name"] != "eval_relevant":
            ret.aggregated_metrics[metric["name"]] = max(metric["values"])
    return message.SerializeToString(deterministic=True)


def completion_path(session):
    return f"/data/alpagym-runs/{RUN_DIR_NAME}/rollouts/{SCENE_A}_{session}_{NONCE}.json"


def _bounded(values):
    return {"count": len(values), "values": list(values)}


def _tensor(kind, sha, requires_grad):
    return {"kind": kind, "dtype": "F32", "shape": [2], "numel": 2,
            "requires_grad": requires_grad, "sha256": sha, "hash_scope": "full", "shard": None}


def synthetic_snapshot(phase, trainer_step, *, expert, state, counter):
    """A schema-2 ``parameter_snapshot`` (``teacher_rl.capture.entrypoint`` docstring)."""
    return {"schema_version": 2, "phase": phase, "trainer_step": trainer_step,
            "tensors": {"expert.w": _tensor("param", expert, True),
                        "vlm.lm_head.weight": _tensor("param", H["e"], False),
                        "action_in_proj.freqs": _tensor("buffer", H["f"], False)},
            "training_flags": {"": True, "vlm": False},
            "optimizer_state": {"expert.w": state}, "step_counters": {"expert.w": counter},
            "applied_lr": [0.0001]}


def synthetic_rollout_events(sink, sessions, *, bytes_captured=True):
    """Prompt 7: ``a`` fails (reward KeyError) -> retry ``c``; ``b`` and ``c`` succeed."""
    a, b, c = sessions
    seeds = dict(zip(sessions, SESSION_SEEDS, strict=True))
    names = {a: ["collision_rear", "eval_relevant", "img_is_black", "offroad"]}
    for s in (b, c):
        names[s] = sorted(names[a] + ["collision_any", "collision_front", "collision_lateral"])
    collision = {a: None, b: False, c: True}

    def simulate(call, session):
        payload = {"call": call, "sessions": [session], "returns": [{
            "sessions": [session], "success": True, "error": "",
            "rollout_uuid": f"rollout-{session}", "timestep_metrics": names[session]}]}
        if bytes_captured:
            payload["reply_blob"] = sink.put_blob(synthetic_reply_bytes(
                session, collision=collision[session]))
        return "simulate_call", payload

    def att(kind, session, **fields):
        return kind, {"payload_instance": 1, "prompt_idx": 7, "session_uuid": session, **fields}

    def started(call, session):
        return "session_started", {"call": call, "session_uuid": session,
                                   "random_seed": seeds[session]}

    return [
        ("generation_entered", {"call": 100, "prompt_idxs": [7], "is_validation": False,
                                "current_weight_version": 0}),
        ("payload_bound", {"payload_instance": 1, "prompt_idx": 7, "n_target": 2,
                           "retries_left": 3}),
        ("payload_submitted", {"payload_instance": 1, "prompt_idx": 7}),
        att("attempt_started", a, scene_id=SCENE_A, attempt=0),
        att("attempt_started", b, scene_id=SCENE_A, attempt=0),
        started(201, a), started(202, b), simulate(301, a),
        att("attempt_failed", a, attempt=0, cause_type="KeyError",
            cause_message="'collision_any'", outcome="retry", retries_left_after=2),
        att("attempt_started", c, scene_id=SCENE_A, attempt=1),
        started(203, c), simulate(302, b),
        att("attempt_succeeded", b, attempt=0, collected_index=0, reward_total=1.0),
        simulate(303, c),
        att("attempt_succeeded", c, attempt=1, collected_index=1, reward_total=-9.0),
        ("generation_exited", {"call": 100, "results": 1}),
    ]


def synthetic_trainer_events(sessions, *, hashed=True, iterations=1):
    """One trainer step of one minibatch over the two group members (E03/E04/E07/E20)."""
    _a, b, c = sessions
    pick = (lambda d: d) if hashed else (lambda d: None)

    def snap(phase, step, expert, state, counter):
        event = synthetic_snapshot(phase, step, expert=expert, state=state, counter=counter)
        if not hashed:
            for entry in event["tensors"].values():
                entry["sha256"] = entry["hash_scope"] = None
            event["optimizer_state"] = {k: dict.fromkeys(v) for k, v in
                                        event["optimizer_state"].items()}
        return "parameter_snapshot", event

    moments = {"exp_avg": pick(H["c"]), "exp_avg_sq": pick(H["d"]), "step": pick(H["9"])}
    rows = [{"prompt_idx": 7, "completion": completion_path(s), "weight_version": 0,
             "advantage": adv, "reward": reward}
            for s, adv, reward in ((b, 0.9999998, 1.0), (c, -0.9999998, -9.0))]
    return [
        ("trainer_configured", {"call": 1, "mini_batch": 2, "iterations": iterations,
                                "group_size": 2, "allowed_outdated_steps": 0, "kl_beta": 0.0}),
        snap("init", None, H["a"], {}, 0),
        ("trainer_rollouts_received", {"call": 2, "current_step": 0,
                                       "train_batch_per_replica": 2,
                                       "allowed_outdated_steps": 0, "rollouts": rows}),
        ("trainer_step_started", {"call": 10, "current_step": 0, "total_steps": 1,
                                  "received_rollouts": 2, "do_save_checkpoint": False}),
        ("scheduler_state", {"call": 10, "phase": "pre_step", "trainer_step": 0,
                             "last_epoch": 0, "lrs": [0.0001]}),
        snap("pre_step", 0, H["a"], {}, 0),
        ("replay_pool_prepared", {"call": 11, "rows": 2,
                                  "rollouts": [completion_path(b), completion_path(c)],
                                  "rows_per_rollout": 1,
                                  "advantages": _bounded([0.9999998, -0.9999998])}),
        ("minibatch_started", {"call": 12, "rows": 2, "pool_call": 11,
                               "row_indices": _bounded([0, 1]),
                               "advantages": _bounded([0.9999998, -0.9999998]),
                               "old_logprobs": _bounded([[v] for v in OLD_LOGPROBS]),
                               "is_padding": _bounded([[False], [False]])}),
        ("replay_forward", {"call": 13, "minibatch_call": 12,
                            "log_probs": _bounded(list(NEW_LOGPROBS)), "kl_present": False}),
        ("optimizer_reduce", {"call": 14, "minibatch_call": 12, "grad_norm": 0.5,
                              "optimizer_step_calls": 1, "lr": [0.0001]}),
        ("minibatch_finished", {"call": 12, "metrics": [0.25]}),
        ("trainer_step_finished", {"call": 10, "current_step": 0,
                                   "metrics": {"train/num_batches": 1}}),
        ("scheduler_state", {"call": 10, "phase": "post_step", "trainer_step": 0,
                             "last_epoch": 1, "lrs": [0.0001]}),
        snap("post_step", 0, H["b"], moments, 1),
    ]


def read_capture_blobs(capture_dir):
    """``{sha256: bytes}`` of a capture blob store, read without following links."""
    import hashlib
    import os

    blobs = {}
    root = Path(capture_dir) / "blobs"
    for path in sorted(root.glob("*/*")) if root.is_dir() else []:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            data = b""
            while chunk := os.read(fd, 1 << 20):
                data += chunk
        finally:
            os.close(fd)
        if hashlib.sha256(data).hexdigest() != path.name:
            raise ValueError(f"blob {path} does not hash to its name")
        blobs[path.name] = data
    return blobs


def write_synthetic_capture(capture_dir, *, sessions=SESSIONS_A, trainer=True,
                            trainer_hashed=True, iterations=1, bytes_captured=True,
                            rollout_label="rollout_0"):
    """Write the synthetic capture through the real ``EventSink`` and read it back.

    Returns ``(streams, blobs)``: ``streams_from_reports`` over the verified ``read_stream``
    reports and the blob store read back from disk. Clocks and pid are fixed, so the chain
    heads are deterministic. ``rollout_label`` lets a test use the real capture label format
    (``<role>-<pid>-<uuid12>``, ``teacher_rl.capture.entrypoint.main``), which differs per run.
    """
    from teacher_rl.capture.sink import EventSink, read_stream
    from teacher_rl.records import streams_from_reports

    def sink(label):
        return EventSink(capture_dir, stream_label=label, limits=CAPTURE_LIMITS,
                         wall_ns=lambda: 0, mono_ns=lambda: 0, getpid=lambda: 1)

    rollout = sink(rollout_label)
    for kind, payload in synthetic_rollout_events(rollout, sessions,
                                                  bytes_captured=bytes_captured):
        rollout.emit(kind, payload)
    rollout.close()
    paths = [rollout.path]
    if trainer:
        trainer_sink = sink("trainer_0")
        for kind, payload in synthetic_trainer_events(sessions, hashed=trainer_hashed,
                                                      iterations=iterations):
            trainer_sink.emit(kind, payload)
        trainer_sink.close()
        paths.append(trainer_sink.path)
    reports = [read_stream(path, max_bytes=CAPTURE_LIMITS["max_stream_bytes"])
               for path in paths]
    streams, _ = streams_from_reports(reports)
    return streams, read_capture_blobs(capture_dir)


def synthetic_launch(run_id, effective_config_hash):
    """A synthetic E15 start record (hashes are fixtures, not observed bytes)."""
    return {"run_id": run_id, "launch_id": f"{run_id}-launch-0", "ordinal": 0,
            "phase": "start", "effective_config_hash": effective_config_hash,
            "uv_lock_sha256": H["1"], "base_weights_hash": H["2"],
            "capture_tree_hash": H["3"]}


def _capture(workdir):
    """Fake capture events through the real sink and reader (prompt 7: one silent retry)."""
    from teacher_rl.records import build_attempt_ledger

    streams, blobs = write_synthetic_capture(workdir / "capture")
    return build_attempt_ledger(streams, scope="synthetic"), streams, blobs


def _capture_checks(workdir, streams, blobs, ledger, effective_config_hash):
    """G3: the capture feeds the decoder, E10-E14, E20/E07, E03, E04 and E01."""
    from teacher_rl import adapters, records
    from teacher_rl.checks import replay

    decodes = adapters.decode_simulate_calls(streams, blobs=blobs, scope="synthetic")
    # Synthetic stand-in for the reward arm's required metric (not an owner value).
    sem = adapters.semantics_from_capture(decodes, ledger, required_metrics=["collision_any"],
                                          deviation_metric=None, scope="synthetic")
    trainer = records.build_trainer_step_records(streams, scope="synthetic", blobs=blobs)
    step_checks = adapters.step_checks_from_records(trainer, scope="synthetic")
    e03 = adapters.e03_from_capture(streams, stream_id="trainer_0", trainer_step=0,
                                    trainable_prefixes=["expert."], frozen_prefixes=["vlm."],
                                    scope="synthetic")
    trace = records.check_trace_integrity(ledger, streams, cohort="strict_parity",
                                          expected_version=0)
    e04 = replay.check_logprob_replay(trainer, trace_integrity=trace, trainer_weight_version=0,
                                      tolerance=SYNTHETIC_E04_TOLERANCE, scope="synthetic")
    streams_b, blobs_b = write_synthetic_capture(workdir / "capture_b", sessions=SESSIONS_B,
                                                 trainer=False)
    run_a = adapters.repeatability_run(
        streams, blobs=blobs, launch=synthetic_launch("e01-a", effective_config_hash),
        run_id="e01-a", scope="synthetic")
    run_b = adapters.repeatability_run(
        streams_b, blobs=blobs_b, launch=synthetic_launch("e01-b", effective_config_hash),
        run_id="e01-b", scope="synthetic")
    e01 = replay.check_repeatability(run_a, run_b, declaration={
        "weights": True, "inputs": True, "noise_seeds": True,
        "rollout_force_determinism": True}, scope="synthetic")
    by_label = dict(zip(SESSIONS_A, ("a", "b", "c"), strict=True))
    rows = sem["sessions"]
    return {
        "simulate": {"status": decodes["status"], "counts": decodes["counts"]},
        "semantics": {
            "status": sem["status"],
            "e10": {by_label[s]: rows[s]["e10"]["status"] for s in sorted(rows)},
            "e11": {by_label[s]: rows[s]["e11"]["status"] for s in sorted(rows)},
            "contact": {by_label[s]: [rows[s]["e11"]["any_contact"],
                                      rows[s]["e11"]["additive_front_lateral"]]
                        for s in sorted(rows)},
            "e14": {"status": sem["e14"]["status"],
                    "retry_causes": sem["e14"]["retry_causes"]}},
        "trainer_records": trainer["status"],
        "steps": {"status": step_checks["status"],
                  "e20": {sid: r["status"] for sid, r in step_checks["e20"].items()},
                  "e07": [r["report"]["status"] for r in step_checks["e07"]]},
        "e03": {"status": e03["status"], "changed": e03["diff"]["changed"]},
        "e04": {"status": e04["status"], "verdict": e04["verdict"],
                "paired_rows": e04["counts"]["paired_rows"], "max_ulp": e04["max_ulp"],
                "trace_integrity": trace["status"]},
        "e01": {"status": e01["status"], "verdict": e01["verdict"],
                "compared_units": e01["compared_units"]},
    }


def _checks(ledger):
    from teacher_rl import adapters, diagnostics
    from teacher_rl.checks import signal, steps

    rewards = ledger["groups"][0]["member_rewards"]
    advantages = [float(a) for a in signal.group_advantages_f32(rewards)]
    expected_steps = steps.expected_optimizer_steps(44, 1, 1, 0)
    digest = {c: c * 64 for c in "abcd"}
    params = {"artifact_kind": "s4_param_snapshot", "schema_version": 1, "scope": "synthetic",
              "status": "passed", "source": "module",
              "tensors": {"expert.w": {"kind": "parameter", "sha256": digest["a"]}}}
    optimizer = {"artifact_kind": "s4_optimizer_state_snapshot", "schema_version": 1,
                 "scope": "synthetic", "status": "passed",
                 "param_groups": [{"index": 0, "params": ["expert.w"]}],
                 "state": {"expert.w": {"exp_avg": {"sha256": digest["b"]},
                                        "step": {"sha256": digest["c"], "value": 1.0}}}}
    skip = adapters.check_skip(
        0, scheduler_before={"last_epoch": 0, "lrs": [0.0001]},
        scheduler_after={"last_epoch": 1, "lrs": [0.0001]},
        minibatches=[adapters.minibatch_record(0, grad_norm=0.0, optimizer_step_called=False,
                                               before=(params, optimizer),
                                               after=(params, optimizer))],
        scope="synthetic")
    diag = diagnostics.summarize_training_diagnostics(
        {"groups": {"groups": [{"group_id": ledger["groups"][0]["group_id"],
                                "rewards": rewards,
                                "components": [{"collision_any": 0.0},
                                               {"collision_any": 1.0}]}],
                    "failure_components": ["collision_any"]}}, scope="synthetic")
    return {"advantages": advantages, "expected_optimizer_steps": expected_steps,
            "skip_status": skip["status"], "diagnostics_status": diag["status"]}


class _FakeProcesses:
    """The rollout role dies while the host and controller live (lane-r3 finding 14)."""

    def __init__(self):
        self.alive_groups = {101, 102, 103, 104, 105}
        self.dead_pids = {104}
        self.signals = []

    def alive(self, pid):
        return pid not in self.dead_pids and pid in self.alive_groups

    def exit_code(self, pid):
        return None

    def group_alive(self, pgid):
        return pgid in self.alive_groups and pgid not in self.dead_pids

    def signal_group(self, pgid, sig):
        self.signals.append((pgid, int(sig)))
        self.alive_groups.discard(pgid)
        return "sent"


class _FakeClock:
    def __init__(self, now):
        self.now = now

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _watchdog(launch_spec):
    from gate import s2_execution
    from teacher_rl import watchdog

    roles = [{"name": n, "role": r, "pid": p, "pgid": p} for n, r, p in (
        ("host", "host", 101), ("wizard", "wizard", 102), ("controller", "controller", 103),
        ("rollout_0", "rollout", 104), ("policy_0", "policy", 105))]
    # s4_watch_spec v2 (G1): bound to the launch hash; containers scoped by Compose project
    spec = watchdog.build_watch_spec(
        launch_spec=launch_spec, run_id="run-a", role_records=roles,
        container_record={"compose_project": "physloop-run-a", "container_ids": []},
        limits={"stall_seconds": 600, "poll_seconds": 30, "deadline_epoch": 1_800_030_000,
                "provider_cutoff_epoch": 1_800_036_000, "export_reserve_seconds": 3600,
                "term_grace_seconds": 10})
    runs = iter([RunResult(0, b"", b"", False), RunResult(0, b"", b"", False)])
    dog = watchdog.TrainingWatchdog(spec, processes=_FakeProcesses(),
                                    progress=lambda: 5, runner=lambda argv, timeout_s: next(
                                        runs), clock=_FakeClock(1_800_000_000))
    report = dog.run(max_polls=3)
    observation = report["handoff_observation"]
    s2_execution.validate_observation(observation)
    return {"decision": report["decision"], "reasons": report["reasons"],
            "cleanup": report["cleanup_status"], "handoff": observation["kind"]}


def run_rehearsal(workdir):
    """Run the synthetic chain under ``workdir`` (must be a new, empty directory)."""
    from teacher_rl import evaluate, forecast, launch, ops
    from teacher_rl.registration import compile_overrides

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=False)
    registration = synthetic_registration()
    compiled = compile_overrides(registration)
    inventory = synthetic_inventory()
    launch_spec = launch.build_launch(
        registration, alpagym_checkout="/data/src/alpagym",
        capture_tree="/data/capture-tree", capture_manifest="/data/capture-tree.json",
        capture_config="/data/capture-config.json", worker_credentials="strip_offline",
        action_authority_ref="SYNTHETIC-launch-authority",
        run_dir=f"/data/alpagym-runs/{RUN_DIR_NAME}",
        base_cosmos_toml=synthetic_base_cosmos_toml(), resolved_config_sha256="5" * 64,
        alpasim_cache_dir=inventory["host_paths"]["alpasim_cache_dir"],
        nurec_authorization_refs={"terms": "SYNTHETIC-terms-nurec",
                                  "acquire": "SYNTHETIC-acquire-nurec"})
    plan = ops.build_s4_bootstrap_plan(inventory, synthetic_bootstrap_record(),
                                       synthetic_rental_record(),
                                       phases=list(ops.PHASES))
    ledger, streams, blobs = _capture(workdir)
    checks = _checks(ledger)
    capture_checks = _capture_checks(workdir, streams, blobs, ledger,
                                     compiled["effective_config_hash"])
    dog = _watchdog(launch_spec)
    cost = forecast.forecast(synthetic_forecast_input())
    registry = synthetic_run_registry(registration)
    requests = []
    for index, run in enumerate(RUN_IDS):
        audit_inputs, manifest = synthetic_export_chain(index)
        s2_registration = _s2_registration("tensor_set_sha256:" + manifest["tensor_set_sha256"])
        requests.append(evaluate.build_evaluation_request(
            inference_manifest=manifest, s4_registration=registration, run_registry=registry,
            run_id=run, checkpoint_step=3, s2_registration=s2_registration, scope="synthetic",
            export_audit_inputs=audit_inputs))
    spec = evaluate.aggregation_spec(requests=requests, s4_registration=registration,
                                     run_registry=registry, scope="synthetic")
    decided = evaluate.decide_and_aggregate(_evaluations(spec), spec)
    summary = {
        "compile": {"status": compiled["status"],
                    "cosmos_toml_overrides": compiled["cosmos_toml_overrides"],
                    "max_optimizer_steps_per_run": compiled["derived"][
                        "max_optimizer_steps_per_run"]},
        "launch": {"status": launch_spec["status"], "route": launch_spec["route"],
                   "argv_tail": launch_spec["argv"][-1],
                   "toml_train_seed": launch_spec["toml"]["merged"]["train"]["seed"]},
        "bootstrap": {"compiled": plan["compiled"], "stages": len(plan["stages"]),
                      "gpu_count": plan["gpu_count"],
                      "secrets_required": plan["secrets_required"]},
        "ledger": {"status": ledger["status"], "counts": ledger["counts"],
                   "failure_causes": ledger["failure_causes"]},
        "checks": checks,
        "capture_checks": capture_checks,
        "watchdog": dog,
        "forecast": {"status": cost["status"], "total_usd": cost["total_usd"],
                     "additional_cash_needed_usd": cost["money"]["additional_cash_needed_usd"]},
        "evaluation": {"requests_passed": sum(r["status"] == "passed" for r in requests),
                       "spec_status": spec["status"], "rule": spec["rule"],
                       "registered_run_ids": spec["registered_run_ids"]},
        "decision": {"status": decided["status"], "outcomes": decided["outcomes"],
                     "aggregate": (decided["aggregate"] or {}).get("outcome"),
                     "guards": [g["name"] for g in spec["decision_rule"]["guards"]]},
        "codes": {"launch": _codes(launch_spec), "plan": _codes(plan),
                  "ledger": _codes(ledger), "forecast": _codes(cost)},
        "cloud_launch_authorized": False,
        "scope": "synthetic",
    }
    summary["summary_sha256"] = identity_hash(summary)
    return summary


def _evaluations(spec):
    """Synthetic gate evaluations per run: the candidate never fails the primary endpoint and
    the reference always does (40 roads); the guard endpoint has identical outputs."""
    from gate.s2_gate import evaluate_registered

    out = {}
    for run in RUN_IDS:
        candidate = spec["run_identities"][run]["candidate"]
        primary, guard = _s2_registration(candidate), _s2_guard_registration(candidate)
        out[run] = {
            "primary": evaluate_registered(synthetic_records(reference_fail_roads=range(40)),
                                           primary, scope="synthetic",
                                           adapter_metric=primary["metric"]),
            "guards": {SYNTHETIC_GUARD["name"]: evaluate_registered(
                synthetic_records(), guard, scope="synthetic", adapter_metric=guard["metric"])},
            "guard_registrations": {SYNTHETIC_GUARD["name"]: guard},
        }
    return out


def _s2_registration(candidate):
    from gate.s2_gate import REGISTRATION_FIELDS

    return {"schema_version": 1, "registration_id": "synthetic-s2-eval",
            "reference": f"teacher:{pins.TEACHER['repository']}@{pins.TEACHER['revision']}",
            "candidate": candidate,
            "protocol": {"identity": "synthetic-protocol", "config_hash": "c" * 64,
                         "evaluator": "gate.s2_gate"},
            "seed_mode": "synthetic_seeded", "score_definition": "synthetic score",
            "deviation_convention": "synthetic convention",
            "estimand": REGISTRATION_FIELDS["estimand"], "required_strata": ["urban"],
            "margin": 0.1, "alpha": 0.05, "metric": dict(PRIMARY_METRIC),
            "analysis_mode": "fixed_sample"}


def _s2_guard_registration(candidate):
    """The synthetic guard endpoint's S2 registration: same protocol, its own endpoint."""
    guard = _GUARD_ENDPOINT
    return {**_s2_registration(candidate), "registration_id": guard["s2_registration_id"],
            "score_definition": "synthetic hard-brake score", "metric": dict(guard["metric"]),
            "margin": guard["margin"], "alpha": guard["alpha"]}


# The guard's S2 registration without its run-specific candidate, hashed as
# teacher_rl.evaluate._s2_protocol_hash does (one value for every run).
SYNTHETIC_GUARD = {**_GUARD_ENDPOINT, "s2_protocol_hash": identity_hash(
    {k: v for k, v in _s2_guard_registration(None).items() if k != "candidate"})}


def synthetic_records(candidate_fail_roads=(), reference_fail_roads=(), *, roads=40):
    """Paired binary records on ``roads`` urban roads (1 = event, higher is worse)."""
    rows = []
    for road in range(roads):
        for arm, failed in (("A", road in reference_fail_roads),
                            ("B", road in candidate_fail_roads)):
            rows.append({"arm": arm, "scene_id": f"urban/road-{road:03d}/scene-0",
                         "road_id": f"urban/road-{road:03d}", "stratum": "urban",
                         "replicate_id": 0, "value": 1 if failed else 0, "completed": True,
                         "deviation_m": 0.5, "seed": road})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", required=True, help="new directory for the capture stream")
    args = parser.parse_args(argv)
    if not Path(args.workdir).is_absolute():
        parser.error("--workdir must be absolute")
    try:
        summary = run_rehearsal(args.workdir)
    except FileExistsError:
        parser.error("--workdir must not exist (the rehearsal never reuses evidence)")
    sys.stdout.write(canonical_json_bytes(summary).decode() + "\n")
    passed = (summary["compile"]["status"] == "passed" and summary["bootstrap"]["compiled"]
              and summary["forecast"]["status"] == "passed"
              and summary["evaluation"]["spec_status"] == "passed"
              and summary["decision"]["status"] == "passed")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
