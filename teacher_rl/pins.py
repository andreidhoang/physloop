"""S4 engineering pins: upstream commits, model revisions, lock facts and preset facts.

Tickets: S4.1 (freeze the implemented learning problem: exact recipes PR dependency, Cosmos-RL,
training AlpaSim, checkpoint revision), E15 (run identity inputs), R3 §2 (pin/environment
manifest). Agent-authored under D-018/D-024; engineering preparation, not an S4 admission.

Verification ceiling:
- The constants are **L2** facts: read from the R3 lane clones at the pinned commits and from the
  vendored, hash-checked copies in ``tests/fixtures/s4_upstream/`` (``provenance.json``). They are
  untested on a host; whether a host checkout, lock or snapshot actually matches is **L3**.
- ``validate_observed_pins``: **L1** (pure comparison with literal known answers in
  ``tests/test_s4_registration.py``).
- ``pins_artifact``: **L1** (deterministic canonical artifact).

The Cosmos-Reason2-8B revision is an admission-time input. ``a9fae2cf…`` is recorded only as a
dated observation (2026-11-08, R3 finding 8); it is never used as a default.

This module never reads the network, a provider, a credential, a clock or the filesystem
(other than hashing its own source for producer identity), and never accepts a licence.
"""

import hashlib
import re
from pathlib import Path

from gate.s2_identity import identity_hash

SCHEMA_VERSION = 1
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")

# name -> pinned upstream source. ``source`` cites the lane report and the pinning file.
UPSTREAM_PINS = {
    "alpagym": {
        "repository": "https://github.com/NVlabs/alpagym",
        "commit": "972d160eed0e23d388497851504a3a233fec5879",
        "role": "training framework (host, runtime, policies); main head, no releases",
        "source": "lane-r3 §2; R3 clone HEAD",
    },
    "alpamayo_recipes": {
        "repository": "https://github.com/NVlabs/alpamayo-recipes",
        "commit": "1d99fc50370c96637157455da386323a44663c9d",
        "role": "expert-model RL recipes; open PR #18 head pinned by AlpaGym pyproject.toml:70",
        "source": "lane-r3 §2",
    },
    "cosmos_rl": {
        "repository": "https://github.com/nvidia-cosmos/cosmos-rl",
        "commit": "d2a2c57c4bd6496482bc42da19a59b4432705eda",
        "role": "GRPO dispatcher/trainer; override pinned by AlpaGym pyproject.toml:20-27,72",
        "source": "lane-r3 §2",
    },
    "training_alpasim": {
        "repository": "https://github.com/NVlabs/alpasim",
        "commit": "46b401beaec27977cbb86826df8e39cec2d49e49",
        "tree_identical_commit": "10cae1c2b943da72eabafad65df80f4e3c16d90d",
        "tree": "9b838542a6365bd34dc48b9c6eb795a94162fb8d",
        "role": "training simulator (only refs/pull/107/head; tree-identical to main 10cae1c)",
        "source": "lane-r3 §2; git rev-parse <commit>^{tree} on both commits in the R3 clone",
    },
    "alpamayo_r1": {
        "repository": "https://github.com/NVlabs/alpamayo",
        "commit": "4cda35d22bb257f0936ac272397627b9309ca211",
        "role": "model package pinned by AlpaGym pyproject.toml:69",
        "source": "lane-r3 §2",
    },
}

TEACHER = {
    "repository": "nvidia/Alpamayo-1.5-10B",
    "repo_type": "model",
    "revision": "7aba8293c09993f2e125c6819df05d7fa3e873ea",
    "gated": False,
    "terms": "OpenMDW-1.1 (Huy accepts)",
    "source": "lane-r3 §2; 02 S4.1 (Gym downloads without a revision; the manifest pins it)",
}

COSMOS_REASON2 = {
    "repository": "nvidia/Cosmos-Reason2-8B",
    "repo_type": "model",
    "gated": True,
    "terms": "gated Hugging Face access (Huy accepts)",
    "revision": None,  # admission-time input; never defaulted
    "observed_revision": {
        "value": "a9fae2cf89dc64db96b12860417f0eb403013bb9",
        "observed_on": "2026-11-08",
        "source": "lane-r3 §1 finding 8 (HF metadata, gated auto)",
        "use": "observation only; the registration supplies the revision at admission",
    },
    "why_required": (
        "the converter writes vlm_name_or_path into config.json and every model construction "
        "calls AutoProcessor/Qwen3VLConfig.from_pretrained on it (lane-r3 §1 finding 8)"
    ),
}

ALPAGYM_LOCK = {
    "uv_lock_sha256": "5148f9b0f5f20b1cc3753dec78e165491c881b0217d095bba9a74d017fcd674e",
    "packages": {
        "torch": "2.8.0+cu128",
        "transformers": "4.57.1",
        "vllm": "0.11.0",
        "numpy": "2.2.6",
        "safetensors": "0.7.0",
    },
    "platform": "linux x86_64 only (AlpaGym pyproject.toml:30)",
    "source": "lane-r3 §2; sha256 of uv.lock at the AlpaGym pin in the R3 clone",
}

# Shipped two-GPU base preset facts, cited from the vendored pinned YAML (alpagym repo).
_PRESET_DIR = "packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/configs/experiment"
EXPERIMENT_PRESETS = {
    "alpamayo_1_5_local_2gpu_smoke": {
        "source": f"alpagym:{_PRESET_DIR}/alpamayo_1_5_local_2gpu_smoke.yaml",
        "supported_base": True,
        "topology": "local_disaggregated_2gpu",
        "expected_valid_steps": 22,
        "n_generation": 2,
        "train_batch_per_replica": 1,
        "dp_shard_size": 1,
        "control_timestep_us": 200_000,
        "force_gt_duration_us": 1_600_000,
        "n_sim_steps": 30,
        "extra_overrides": (
            "+cameras=4cam_1080 runtime.simulation_config.pose_reporting_interval_us=100000"
        ),
        "ratio_clip": [0.2, 0.2],
        "reward_group_default": "distance_to_gt",
    },
    "alpamayo_1_5_clrl_test_run": {
        "source": f"alpagym:{_PRESET_DIR}/alpamayo_1_5_clrl_test_run.yaml",
        "supported_base": False,
        "unsupported_reason": "slurm_full_node_1_3_4 topology; not the two-GPU local host",
        "topology": "slurm_full_node_1_3_4",
        "expected_valid_steps": 22,
        "n_generation": 6,
        "train_batch_per_replica": 6,
        "dp_shard_size": 2,
        "control_timestep_us": 200_000,
        "force_gt_duration_us": 1_600_000,
        "n_sim_steps": 30,
        "extra_overrides": (
            "+cameras=4cam_1080 runtime.simulation_config.pose_reporting_interval_us=100000"
        ),
        "ratio_clip": [0.2, 0.2],
        "reward_group_default": "progress_safety",
    },
}

OBSERVED_PIN_FIELDS = (
    "alpagym", "alpamayo_recipes", "cosmos_rl", "training_alpasim", "alpamayo_r1",
    "alpagym_uv_lock_sha256", "teacher_revision", "cosmos_reason2_revision",
)


def _issue(code, path, message, severity="blocking"):
    return {"code": code, "path": path, "severity": severity, "evidence_refs": [],
            "message": message}


def _producer():
    source = Path(__file__).read_bytes()
    return {"module": "teacher_rl.pins", "source_sha256": hashlib.sha256(source).hexdigest()}


def pin_facts():
    """Return the pinned facts as one JSON-native mapping (copy; callers cannot mutate)."""
    return {
        "upstream": {name: dict(pin) for name, pin in UPSTREAM_PINS.items()},
        "teacher": dict(TEACHER),
        "cosmos_reason2": {**COSMOS_REASON2,
                           "observed_revision": dict(COSMOS_REASON2["observed_revision"])},
        "alpagym_lock": {**ALPAGYM_LOCK, "packages": dict(ALPAGYM_LOCK["packages"])},
        "experiment_presets": {
            name: {**facts, "ratio_clip": list(facts["ratio_clip"])}
            for name, facts in EXPERIMENT_PRESETS.items()
        },
    }


def pins_artifact():
    """Canonical pin artifact with its identity hash (no wall-clock content)."""
    facts = pin_facts()
    return {
        "artifact_kind": "s4_pins",
        "schema_version": SCHEMA_VERSION,
        "scope": "public_source_fixture",
        "producer": _producer(),
        "inputs": [],
        "units": {},
        "facts": facts,
        "facts_hash": identity_hash(facts),
        "issues": [],
    }


def validate_observed_pins(observed):
    """Compare a host-observed pin record with the pins; fail closed on missing or mismatched.

    ``observed`` maps every name in ``OBSERVED_PIN_FIELDS`` to the value reported by the host
    (``git rev-parse HEAD``, sha256 of ``uv.lock``, resolved snapshot revisions). The
    Cosmos-Reason2-8B revision has no pinned value: it must equal ``expected_cosmos_reason2``
    from the registration, passed inside ``observed`` as ``registered_cosmos_reason2_revision``.
    """
    issues = []
    if type(observed) is not dict:
        return {"status": "blocked", "issues": [_issue("observed_not_object", "$",
                                                       "observed pins must be an object")]}
    allowed = set(OBSERVED_PIN_FIELDS) | {"registered_cosmos_reason2_revision"}
    for key in sorted(set(observed) - allowed):
        issues.append(_issue("unknown_field", f"$.{key}", f"unknown observed pin field {key!r}"))
    expected = {name: pin["commit"] for name, pin in UPSTREAM_PINS.items()}
    expected["alpagym_uv_lock_sha256"] = ALPAGYM_LOCK["uv_lock_sha256"]
    expected["teacher_revision"] = TEACHER["revision"]
    registered = observed.get("registered_cosmos_reason2_revision")
    if type(registered) is not str or not _COMMIT.fullmatch(registered):
        issues.append(_issue("cosmos_reason2_revision_unregistered",
                             "$.registered_cosmos_reason2_revision",
                             "the Cosmos-Reason2-8B revision is an admission-time input"))
    else:
        expected["cosmos_reason2_revision"] = registered
    for name in OBSERVED_PIN_FIELDS:
        value = observed.get(name)
        pattern = _SHA256 if name == "alpagym_uv_lock_sha256" else _COMMIT
        if value is None:
            issues.append(_issue("pin_missing", f"$.{name}", f"{name} was not observed"))
        elif type(value) is not str or not pattern.fullmatch(value):
            issues.append(_issue("pin_malformed", f"$.{name}", f"{name} is not a full hash"))
        elif name in expected and value != expected[name]:
            issues.append(_issue("pin_mismatch", f"$.{name}",
                                 f"{name} {value} != pinned {expected[name]}"))
    return {"status": "blocked" if issues else "passed", "issues": issues}
