"""Pure schedule preparation for the adopted S2/v5 pilot and full matrix.

Inputs must come from the caller's explicit study records. Constructing a schedule
does not verify scientific registration, asset authenticity or execution authority.
No files, services or cloud resources are accessed.
"""

import re

from gate.s2_identity import identity_hash

_SCENE_KEYS = {"scene_id", "road_id", "stratum", "asset_sha256"}


def _text(value, name):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty string without surrounding whitespace")
    return value


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{name} must be a 64-character SHA256 hexadecimal digest")
    return value.lower()


def active_matrix_report(replicates=3):
    """Return exact scheduled counts, excluding diagnostics and retry attempts.

    Counts are arithmetic from the adopted matrix, not measured runtime, road
    independence, a price forecast or permission to run the study.
    """
    _positive_integer(replicates, "replicates")
    if replicates < 3:
        raise ValueError("full_sweep requires at least three simulator replicates")
    return {
        "schema_version": 1,
        "stage": "S2",
        "plan_version": "v5",
        "harmonizer_pilot": {
            "scene_count": 5,
            "arm_count": 1,
            "harmonizer_modes": ["off", "on"],
            "replicates": 1,
            "scheduled_policy_rollouts": 10,
        },
        "full_sweep": {
            "scene_count": 441,
            "arm_count": 2,
            "policy_roles": ["teacher", "reexport"],
            "replicates": replicates,
            "scheduled_policy_rollouts": 2 * 441 * replicates,
        },
        "independent_road_count": None,
        "counts_exclude_diagnostics_and_retries": True,
        "runtime_measurement_status": "not_measured",
        "measurement_basis": (
            "Schedule arithmetic only; historical timing scenarios are not measured demand."
        ),
        "spend_authorized": False,
        "cloud_launch_authorized": False,
    }


def build_schedule(
    *, study_id, phase, scenes, policies, protocol_hashes, replicates, requested_steps
):
    """Build deterministic jobs from explicit scene, policy and protocol records.

    ``scenes`` has exactly scene_id, road_id, stratum and asset_sha256 per row.
    The pilot requires five scenes, policy A, distinct off/on protocol hashes and
    one replicate. The full sweep requires 441 scenes, distinct A/B policy IDs,
    one fixed protocol and at least three replicates. Hashes bind declared bytes;
    the caller retains and verifies the source receipts separately.

    Job identity includes study and asset identity, although those fields remain
    in the source request rather than the ledger's compact job schema. Sorting
    inputs makes output independent of input order; this is not randomization or
    a runtime execution-order prescription.
    """
    _text(study_id, "study_id")
    if phase not in ("harmonizer_pilot", "full_sweep"):
        raise ValueError("phase must be harmonizer_pilot or full_sweep")
    _positive_integer(replicates, "replicates")
    _positive_integer(requested_steps, "requested_steps")
    pilot = phase == "harmonizer_pilot"
    if (pilot and replicates != 1) or (not pilot and replicates < 3):
        raise ValueError("pilot requires one replicate; full_sweep requires at least three")
    scene_count = 5 if pilot else 441
    if not isinstance(scenes, list) or len(scenes) != scene_count:
        raise ValueError(f"{phase} requires exactly {scene_count} scene records")
    expected_arms = {"A"} if pilot else {"A", "B"}
    if not isinstance(policies, dict) or set(policies) != expected_arms:
        raise ValueError(f"{phase} policies must have exactly arms {sorted(expected_arms)}")
    for arm, policy in policies.items():
        _text(policy, f"policies.{arm}")
    if len(set(policies.values())) != len(policies):
        raise ValueError("teacher and reexport must have distinct policy IDs")
    expected_modes = {"off", "on"} if pilot else {"fixed"}
    if not isinstance(protocol_hashes, dict) or set(protocol_hashes) != expected_modes:
        raise ValueError(f"{phase} protocol_hashes must have exactly {sorted(expected_modes)}")
    protocols = {
        mode: _sha256(value, f"protocol_hashes.{mode}") for mode, value in protocol_hashes.items()
    }
    if len(set(protocols.values())) != len(protocols):
        raise ValueError("pilot off/on modes require distinct protocol identities")
    canonical_scenes = []
    scene_ids, road_strata = set(), {}
    for index, scene in enumerate(scenes):
        if not isinstance(scene, dict) or set(scene) != _SCENE_KEYS:
            raise ValueError(f"scenes[{index}] must have exactly {sorted(_SCENE_KEYS)}")
        normalized = {
            key: _text(scene[key], f"scenes[{index}].{key}")
            for key in ("scene_id", "road_id", "stratum")
        }
        normalized["asset_sha256"] = _sha256(scene["asset_sha256"], "asset_sha256")
        scene_id, road, stratum = (
            normalized["scene_id"],
            normalized["road_id"],
            normalized["stratum"],
        )
        if scene_id in scene_ids:
            raise ValueError(f"duplicate scene_id: {scene_id}")
        scene_ids.add(scene_id)
        if road_strata.setdefault(road, stratum) != stratum:
            raise ValueError(f"road spans multiple strata: {road}")
        canonical_scenes.append(normalized)
    jobs = []
    for scene in sorted(canonical_scenes, key=lambda row: row["scene_id"]):
        for arm in sorted(policies):
            for mode in sorted(protocols):
                for replicate in range(replicates):
                    job = {
                        "phase": phase,
                        "arm": arm,
                        "policy_id": policies[arm],
                        "scene_id": scene["scene_id"],
                        "road_id": scene["road_id"],
                        "stratum": scene["stratum"],
                        "replicate_id": replicate,
                        "protocol_hash": protocols[mode],
                        "requested_steps": requested_steps,
                    }
                    job_id = identity_hash(
                        {
                            "job_schema_version": 1,
                            "study_id": study_id,
                            "asset_sha256": scene["asset_sha256"],
                            **job,
                        }
                    )
                    jobs.append({"job_id": job_id, **job})
    return jobs
