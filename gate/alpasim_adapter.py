"""Offline adapter for the inspected e2e_challenge summary writer.

No files, assets, network or inference are accessed. Sidecars are trusted declarations,
not authenticated evidence. This adapter has only been exercised on synthetic fixtures.
Public writer: NVlabs/alpasim e121e59141dc391f6c295513d6dadc840a419149,
src/eval/src/eval/aggregation/{processing,failed_rollouts}.py.
"""

import math
from copy import deepcopy

SUPPORTED_EVALUATOR_SHA = "e121e59141dc391f6c295513d6dadc840a419149"
IDENTITY_FIELDS = ("run_uuid", "clipgt_id", "rollout_id")
REQUIRED_METRICS = (
    "progress_clipped_rel",
    "gt_dist_traveled_m",
    "collision_at_fault",
    "offroad",
    "left_corridor_laterally",
)
SCORE_FIELDS = (
    *REQUIRED_METRICS,
    "progress_rel",
    "progress_score",
    "dist_to_gt_trajectory",
    "lateral_dist_to_gt_trajectory",
)


def _finite(value):
    try:
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )
    except OverflowError:
        return False


def normalize_summary(
    summary,
    identity_map,
    observations,
    *,
    arm,
    policy_id,
    expected_evaluator_sha,
    observed_evaluator_sha,
):
    """Return all admitted score records or none, with explicit quarantine reasons.

    identity_map binds upstream IDs to the planned scene/road/stratum/replicate schedule.
    observations supplies actual horizon and FULL-trajectory deviation; summary aggregates
    can be truncated and must not supply that maximum. Unresolved retries/missing rows block
    admission; a later attempt-selection adapter needs its own audited predeclared rule.
    """
    issues, diagnostics, records = [], [], []

    def issue(code, identity=None, message=""):
        issues.append({"code": code, "identity": identity, "message": message})

    def key(row):
        if not isinstance(row, dict) or any(
            not isinstance(row.get(k), str) or not row[k].strip() for k in IDENTITY_FIELDS
        ):
            raise ValueError("all upstream identity fields must be nonempty strings")
        return tuple(row[k] for k in IDENTITY_FIELDS)

    def index(rows, label):
        result = {}
        if not isinstance(rows, list):
            issue("invalid_sidecar", message=f"{label} must be an array")
            return result
        for row in rows:
            try:
                identity = key(row)
            except ValueError as exc:
                issue("invalid_identity", message=f"{label}: {exc}")
                continue
            if identity in result:
                issue("duplicate_identity", identity, f"duplicate {label}")
            result[identity] = row
        return result

    if arm not in ("A", "B") or not isinstance(policy_id, str) or not policy_id.strip():
        issue("invalid_policy", message="explicit arm and policy identity required")
    if not (expected_evaluator_sha == observed_evaluator_sha == SUPPORTED_EVALUATOR_SHA):
        issue("unsupported_evaluator", message="schema version alone is not a compatibility proof")
    if not isinstance(summary, dict):
        summary = {}
    if type(summary.get("schema_version")) is not int or summary.get("schema_version") != 1:
        issue("unsupported_schema")
    if summary.get("scene_score_enabled") is not True:
        issue("score_disabled")
    raw = summary.get("rollouts")
    if not isinstance(raw, list) or not raw:
        issue("empty_or_invalid_rollouts")
        raw = []
    upstream = index(raw, "rollouts")
    identities = index(identity_map, "identity map")
    ledger = index(observations, "observation ledger")
    for identity in sorted(set(identities) - set(upstream)):
        issue("missing_expected_rollout", identity)
    for identity in sorted(set(ledger) - set(upstream)):
        issue("unmatched_observation", identity)
    logical, scene_metadata, road_strata = set(), {}, {}
    out_of_envelope = 0
    for identity, row in upstream.items():
        before = len(issues)
        mapped = identities.get(identity)
        observed = ledger.get(identity)
        metrics = row.get("metrics")
        score_metrics = row.get("score_metrics")
        if not isinstance(metrics, dict):
            issue("missing_metric_provenance", identity)
        elif "error" in metrics:
            issue("infrastructure_error", identity, str(metrics["error"]))
        else:
            if any(metrics.get(k) != row.get(k) for k in (*IDENTITY_FIELDS, "run_name")):
                issue("nested_identity_mismatch", identity)
            if any(not _finite(metrics.get(k)) for k in REQUIRED_METRICS):
                issue("missing_required_scoring_metric", identity)
            if (
                not isinstance(score_metrics, dict)
                or any(k not in score_metrics for k in SCORE_FIELDS)
                or any(
                    not _finite(score_metrics.get(k)) for k in (*REQUIRED_METRICS, "progress_score")
                )
            ):
                issue("invalid_score_metrics", identity)
            elif any(metrics[k] != score_metrics[k] for k in REQUIRED_METRICS if k in metrics):
                issue("score_metric_mismatch", identity)
        score = row.get("score")
        if not _finite(score) or not 0 <= score <= 1:
            issue("invalid_or_unscored", identity)
        status = row.get("status")
        if status not in ("pass", "fail") or row.get("passed") is not (status == "pass"):
            issue("unscored_or_inconsistent_status", identity)
        reason = row.get("failure_reason")
        if status == "fail" and (score != 0 or reason not in REQUIRED_METRICS[2:]):
            issue("inconsistent_failure", identity)
        if status == "pass" and (
            reason is not None
            or not isinstance(score_metrics, dict)
            or score_metrics.get("progress_score") != score
        ):
            issue("inconsistent_pass", identity)
        if isinstance(metrics, dict) and all(_finite(metrics.get(k)) for k in REQUIRED_METRICS[2:]):
            reasons = [k for k in REQUIRED_METRICS[2:] if metrics[k] != 0]
            if reason != (reasons[0] if reasons else None):
                issue("failure_reason_mismatch", identity)
        if mapped is None:
            issue("missing_identity_mapping", identity)
        else:
            if mapped.get("policy_id") != policy_id:
                issue("policy_mismatch", identity)
            for field in ("scene_id", "road_id", "stratum"):
                if not isinstance(mapped.get(field), str) or not mapped[field].strip():
                    issue("missing_logical_identity", identity, field)
            replicate = mapped.get("replicate_id")
            if type(replicate) is not int or replicate < 0:
                issue("invalid_replicate", identity)
            elif isinstance(mapped.get("scene_id"), str):
                logical_id = (mapped.get("scene_id"), replicate)
                if logical_id in logical:
                    issue("duplicate_logical_observation", identity, "retry selection unresolved")
                logical.add(logical_id)
            scene = mapped.get("scene_id")
            metadata = (mapped.get("road_id"), mapped.get("stratum"))
            if isinstance(scene, str) and scene_metadata.setdefault(scene, metadata) != metadata:
                issue("scene_metadata_changed", identity)
            road, stratum = metadata
            if (
                isinstance(road, str)
                and isinstance(stratum, str)
                and road_strata.setdefault(road, stratum) != stratum
            ):
                issue("road_spans_strata", identity)
        if observed is None:
            issue("missing_observation_ledger", identity)
        else:
            req, actual = observed.get("requested_steps"), observed.get("actual_steps")
            if (
                type(req) is not int
                or req < 1
                or type(actual) is not int
                or actual != req
                or observed.get("completion_validated") is not True
            ):
                issue(
                    "horizon_unverified", identity, "v0 requires explicitly verified exact horizon"
                )
            dev = observed.get("full_max_deviation_m")
            if not _finite(dev) or dev < 0:
                issue("full_deviation_unverified", identity)
            if observed.get("required_metrics_verified") is not True:
                issue(
                    "metrics_unverified",
                    identity,
                    "missing offroad can be replaced by zero upstream",
                )
            if (
                not isinstance(observed.get("provenance_ref"), str)
                or not observed["provenance_ref"].strip()
            ):
                issue("missing_provenance_ref", identity)
        diagnostics.append(
            {
                "identity": list(identity),
                "status": status,
                "passed": row.get("passed"),
                "failure_reason": row.get("failure_reason"),
                "metrics": deepcopy(metrics),
                "score_metrics": deepcopy(score_metrics),
                "source_summary_metrics_may_be_truncated": True,
                "normalized": len(issues) == before,
            }
        )
        if len(issues) == before:
            records.append(
                {
                    "arm": arm,
                    "scene_id": mapped["scene_id"],
                    "road_id": mapped["road_id"],
                    "stratum": mapped["stratum"],
                    "replicate_id": mapped["replicate_id"],
                    "value": score,
                    "completed": True,
                    "deviation_m": dev,
                    "seed": None,
                }
            )
            out_of_envelope += int(dev >= 4)
    return {
        "schema_version": 1,
        "admitted": not issues,
        "records": records if not issues else [],
        "metric": {
            "name": "challenge_scene_score",
            "kind": "continuous",
            "higher_is_worse": False,
            "unit": "score_in_0_1",
        },
        "issues": issues,
        "diagnostics": diagnostics,
        "counts": {
            "total_rollouts": len(raw),
            "normalized_before_batch_admission": len(records),
            "out_of_envelope": out_of_envelope,
            "infrastructure_errors": sum(i["code"] == "infrastructure_error" for i in issues),
        },
        "sidecar_authenticity": "not_verified",
        "validation_scope": "synthetic schema fixtures; no real AlpaSim run yet",
    }
