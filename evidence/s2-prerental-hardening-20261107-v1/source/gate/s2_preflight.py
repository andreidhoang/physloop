"""Offline, agent-authored S2 schema-v2 planning admission; standard library only.

No file reads, network calls, license acceptance or launch actions occur here.
Manual attestations and referenced evidence are NOT authenticated or inspected.
`ready_for_planning` means logical field consistency, never authorization to act.

Schema: top-level schema_version=2, stage='S2', phase='harmonizer_pilot' or
'full_sweep', status='preregistered', registration={owner,source_ref,frozen},
prerequisites, planned={source,sampling,simulator,driver,measurement,resources},
realized={}. Planned and realized fields are separate; outputs are not required.
Required exact field names and conditional rules are explicit below.

Attestation: {status:'recorded',actor,source_ref,recorded_at:ISO date/time}.
Budget additionally: tier ('zero','up_to_2500','up_to_6000','collaborator'),
project_cap_usd,stage_cap_usd,pilot_cap_usd,approved_by='Huy'. License acceptance
also requires actor and approved_by Huy. Frozen owner choices have
{value,owner:'Huy',source_ref,frozen:true}. The module never fills such records.
Budget also records phase-specific approved_compute_gpu_hours_cap and
approved_rate_per_gpu_hour_cap. All phases need a frozen experiment_prediction.
Full sweeps need owner full_sweep_release with pilot_evidence_ref, forecast_ref,
and remaining_stage_cap_usd. Resources.other_cap_usd reserves setup/tax/other
noncompute charges; an unused reserve does not increase the compute ceiling.
"""

import math
import re
from datetime import datetime


def _text(value):
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value.strip().lower() not in {"pending", "unknown", "tbd", "____", "none"}
    )


def _number(value, positive=False, integer=False):
    try:
        finite = math.isfinite(value)
    except (OverflowError, TypeError):
        return False
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and finite
        and (value > 0 if positive else value >= 0)
        and (not integer or isinstance(value, int))
    )


def _date(value):
    if not isinstance(value, str):
        return False
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


def _sha(value, length):
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", value) is not None


def _attested(value, owner=False):
    return (
        isinstance(value, dict)
        and value.get("status") == "recorded"
        and _text(value.get("actor"))
        and _text(value.get("source_ref"))
        and _date(value.get("recorded_at"))
        and (not owner or value.get("actor") == value.get("approved_by") == "Huy")
    )


def validate_s2_preflight(manifest):
    """Return blockers, warnings and logical readiness; never launch authorization.

    Every issue is {code,path,message}. Publication eligibility is separate from
    dataset-use planning eligibility and remains conditional on unverified manual
    attestations. Budget arithmetic does not inspect cumulative provider spending.
    """
    blockers, warnings = [], []

    def issue(code, path, message):
        blockers.append({"code": code, "path": path, "message": message})

    def warn(code, path, message):
        warnings.append({"code": code, "path": path, "message": message})

    def section(parent, key, path):
        value = parent.get(key)
        if not isinstance(value, dict):
            issue("invalid_section", path, "Expected an object.")
            return {}
        return value

    warn(
        "unverified_attestations",
        "prerequisites",
        "Manual approval text, evidence references, pins and provider capabilities are not authenticated.",
    )
    warn(
        "not_launch_authorization",
        "",
        "Readiness is an offline consistency result; no cloud launch, download or publication is authorized.",
    )
    if not isinstance(manifest, dict):
        issue("invalid_manifest", "", "Expected a schema-v2 object.")
        manifest = {}
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 2:
        issue("unsupported_schema", "schema_version", "Migrate the draft to schema version 2.")
    if manifest.get("stage") != "S2":
        issue("wrong_stage", "stage", "This validator supports S2 only.")
    phase = manifest.get("phase")
    if phase not in ("harmonizer_pilot", "full_sweep"):
        issue("invalid_phase", "phase", "Choose harmonizer_pilot or full_sweep explicitly.")
    full = phase == "full_sweep"
    if manifest.get("status") != "preregistered":
        issue("not_preregistered", "status", "A draft is not a frozen preregistration.")
    registration = section(manifest, "registration", "registration")
    if (
        registration.get("owner") != "Huy"
        or registration.get("frozen") is not True
        or not _text(registration.get("source_ref"))
    ):
        issue("owner_choice_unfrozen", "registration", "Record Huy's frozen protocol reference.")
    prereq = section(manifest, "prerequisites", "prerequisites")
    planned = section(manifest, "planned", "planned")
    section(manifest, "realized", "realized")
    if not _text(prereq.get("S1_evidence")):
        issue(
            "s1_evidence_missing", "prerequisites.S1_evidence", "Reference qualified S1 evidence."
        )
    for key in ("challenge_closed_verified", "latest_upstream_release_checked"):
        if not _attested(prereq.get(key)):
            issue(
                "prerequisite_unrecorded",
                f"prerequisites.{key}",
                "A dated evidence attestation is required.",
            )
    if not _attested(prereq.get("license_acceptance_by_owner"), owner=True):
        issue(
            "license_unaccepted",
            "prerequisites.license_acceptance_by_owner",
            "Dataset-use planning needs a recorded owner license acceptance; publication is separate.",
        )
    host = prereq.get("host_compatibility_verified")
    if not _attested(host) or any(
        host.get(key) is not True for key in ("docker", "nvidia_runtime", "full_host_access")
    ):
        issue(
            "host_incompatible",
            "prerequisites.host_compatibility_verified",
            "Record full-host access, Docker and NVIDIA runtime compatibility before run planning.",
        )
    upstream = prereq.get("latest_upstream_release_checked")
    if isinstance(upstream, dict) and upstream.get("release_merged") is not True:
        warn(
            "provisional_upstream",
            "prerequisites.latest_upstream_release_checked",
            "No merged post-challenge protocol is attested; future results remain provisional.",
        )
    if not _attested(prereq.get("hour_reforecast_P003")):
        warn(
            "hours_unrecorded",
            "prerequisites.hour_reforecast_P003",
            "Human weekly hours and the remaining schedule cannot be inferred from software delivery.",
        )

    def reject_realized(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                if (
                    isinstance(key, str)
                    and (
                        "_actual" in key
                        or key.startswith("actual_")
                        or key in {"renderer_oom_requeues_per_rollout", "measured_gpu_hours"}
                    )
                    and child is not None
                ):
                    issue(
                        "realized_in_plan",
                        f"{path}.{key}",
                        "Realized outputs belong under realized.",
                    )
                reject_realized(child, f"{path}.{key}")
        elif isinstance(value, list):
            for i, child in enumerate(value):
                reject_realized(child, f"{path}[{i}]")

    reject_realized(planned, "planned")
    source = section(planned, "source", "planned.source")
    sampling = section(planned, "sampling", "planned.sampling")
    simulator = section(planned, "simulator", "planned.simulator")
    driver = section(planned, "driver", "planned.driver")
    measurement = section(planned, "measurement", "planned.measurement")
    resources = section(planned, "resources", "planned.resources")
    for parent, prefix, keys, size in (
        (source, "source", ("git_hash",), 40),
        (source, "source", ("config_sha256", "lockfile_sha256"), 64),
        (simulator, "simulator", ("alpasim_sha", "evaluator_sha", "scorer_file_sha"), 40),
        (driver, "driver", ("resolved_alpamayo1_5_commit",), 40),
    ):
        for key in keys:
            if not _sha(parent.get(key), size):
                issue(
                    "unpinned_identity",
                    f"planned.{prefix}.{key}",
                    f"Require a full {size}-hex identity.",
                )
    second = simulator.get("second_alpasim_sha_if_applicable")
    if second is not None and not _sha(second, 40):
        issue(
            "unpinned_identity",
            "planned.simulator.second_alpasim_sha_if_applicable",
            "Require a full SHA.",
        )
    image = simulator.get("renderer_image")
    if not isinstance(image, str) or re.fullmatch(r"\S+@sha256:[0-9a-fA-F]{64}", image) is None:
        issue(
            "unpinned_identity",
            "planned.simulator.renderer_image",
            "Pin the image by sha256 digest, not tag.",
        )
    for parent, prefix, names in (
        (
            sampling,
            "sampling",
            ("target_population", "estimand", "road_independence_justification"),
        ),
        (
            simulator,
            "simulator",
            ("branch", "topology", "controller_identity", "postprocessing_identity"),
        ),
        (driver, "driver", ("policy_clock_source",)),
        (measurement, "measurement", ("score_definition", "deviation_convention")),
    ):
        for name in names:
            if not _text(parent.get(name)):
                issue(
                    "missing_planned_field",
                    f"planned.{prefix}.{name}",
                    "Provide the planned definition.",
                )
    for name in ("n_sim_steps_requested", "concurrency", "replicas", "nre_workers"):
        if not _number(simulator.get(name), positive=True, integer=True):
            issue(
                "missing_planned_field",
                f"planned.simulator.{name}",
                "Require a positive planned integer.",
            )
    if not _number(driver.get("max_batch_size"), positive=True, integer=True):
        issue(
            "missing_planned_field",
            "planned.driver.max_batch_size",
            "Require a positive batch size.",
        )
    if driver.get("max_batch_size") != 1:
        issue(
            "gated_batching",
            "planned.driver.max_batch_size",
            "Batched-driver proposal P-015 is outside this validator.",
        )
    if simulator.get("traffic_mode") != "log_replay":
        issue(
            "gated_traffic",
            "planned.simulator.traffic_mode",
            "Reactive traffic requires separate P-013 admission.",
        )
    if simulator.get("run_method") not in ("docker_compose", "docker-compose", "SLURM"):
        issue(
            "run_method_unresolved",
            "planned.simulator.run_method",
            "Pin a supported S2 run method.",
        )
    if not isinstance(simulator.get("rendering_flags"), dict):
        issue(
            "missing_planned_field",
            "planned.simulator.rendering_flags",
            "Record rendering flags explicitly.",
        )
    if not isinstance(simulator.get("camera_request"), dict) or not simulator.get("camera_request"):
        issue(
            "missing_planned_field",
            "planned.simulator.camera_request",
            "Record the camera request.",
        )
    size = simulator.get("decoded_size")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or not all(_number(x, True, True) for x in size)
    ):
        issue(
            "missing_planned_field",
            "planned.simulator.decoded_size",
            "Record positive [height,width].",
        )
    scene_list, road_list = sampling.get("scene_list"), sampling.get("road_list")
    for name, values in (("scene_list", scene_list), ("road_list", road_list)):
        if (
            not isinstance(values, list)
            or not values
            or not all(_text(x) for x in values)
            or (all(isinstance(x, str) for x in values) and len(set(values)) != len(values))
        ):
            issue(
                "sampling_unresolved",
                f"planned.sampling.{name}",
                "Supply unique explicit identities.",
            )
    if isinstance(scene_list, list) and len(scene_list) != (441 if full else 5):
        issue(
            "scene_count",
            "planned.sampling.scene_list",
            "S2 specifies 441 full-sweep or 5 pilot scenes.",
        )
    if (
        isinstance(road_list, list)
        and isinstance(scene_list, list)
        and len(road_list) > len(scene_list)
    ):
        issue(
            "sampling_unresolved",
            "planned.sampling.road_list",
            "Road count cannot exceed scene count.",
        )
    replicates = sampling.get("replicates")
    if (
        not _number(replicates, True, True)
        or (full and replicates < 3)
        or (not full and replicates != 1)
    ):
        issue(
            "replicate_count",
            "planned.sampling.replicates",
            "Use >=3 replicates for full sweeps, 1 for pilot.",
        )
    mode, force = sampling.get("seed_mode"), sampling.get("force_determinism")
    if mode not in ("seeded", "unseeded") or type(force) is not bool:
        issue(
            "seed_mode_inconsistent",
            "planned.sampling",
            "Explicit seed mode and boolean force_determinism required.",
        )
    elif (mode == "seeded") != force:
        issue(
            "seed_mode_inconsistent",
            "planned.sampling",
            "Unseeded requires false; seeded requires true.",
        )
    if mode == "seeded" and (
        not _sha(driver.get("determinism_port_commit"), 40)
        or not _text(driver.get("determinism_evidence_ref"))
    ):
        issue(
            "seed_mode_unqualified",
            "planned.driver",
            "Seeded mode needs the pinned reseed port and evidence.",
        )
    harmonizer = simulator.get("harmonizer")
    if (full and type(harmonizer) is not bool) or (not full and harmonizer != "compare_off_on"):
        issue(
            "harmonizer_unresolved",
            "planned.simulator.harmonizer",
            "Pilot compares off/on; full sweep must freeze one boolean state.",
        )
    if not full:
        warn(
            "pilot_not_verdict",
            "phase",
            "Cost pilot does not produce a release verdict; final Harmonizer choice may remain pending.",
        )
    if type(driver.get("expert_graph")) is not bool:
        issue(
            "graph_unresolved",
            "planned.driver.expert_graph",
            "Explicitly default graph off until qualification.",
        )
    elif driver["expert_graph"]:
        qualification = driver.get("graph_qualification")
        if (
            not isinstance(qualification, dict)
            or qualification.get("status") != "passed"
            or qualification.get("bitwise") is not True
            or qualification.get("force_determinism") is not True
            or qualification.get("negative_controls_pass") is not True
            or not _text(qualification.get("evidence_ref"))
            or qualification.get("resolved_alpamayo1_5_commit")
            != driver.get("resolved_alpamayo1_5_commit")
        ):
            issue(
                "graph_unqualified",
                "planned.driver.graph_qualification",
                "Record bitwise deterministic qualification and negative controls on the pinned driver.",
            )
        elif mode == "unseeded":
            warn(
                "graph_transfer_assumed",
                "planned.driver.graph_qualification",
                "Bitwise qualification is deterministic; transfer to unseeded sweep is an assumption.",
            )
    if measurement.get("metric_direction") not in ("higher_is_better", "higher_is_worse"):
        issue(
            "invalid_measurement",
            "planned.measurement.metric_direction",
            "Declare metric direction.",
        )
    owner_choices = ("experiment_prediction",)
    if full:
        owner_choices += (
            "margin",
            "reliability_floor",
            "noise_floor_prediction",
            "stabilization_rule",
            "strata_definition",
        )
    for name in owner_choices:
        choice = measurement.get(name)
        if (
            not isinstance(choice, dict)
            or choice.get("owner") != "Huy"
            or choice.get("frozen") is not True
            or not _text(choice.get("source_ref"))
        ):
            issue(
                "owner_choice_unfrozen",
                f"planned.measurement.{name}",
                "Record a frozen owner value and source.",
            )
            continue
        value = choice.get("value")
        valid = (
            _number(value, positive=name != "noise_floor_prediction")
            and (value < 1 if name == "margin" else value <= 1)
            if name in ("margin", "reliability_floor", "noise_floor_prediction")
            else _text(value)
            if name in ("stabilization_rule", "experiment_prediction")
            else isinstance(value, list)
            and bool(value)
            and all(_text(x) for x in value)
            and len(set(value)) == len(value)
        )
        if not valid:
            issue(
                "invalid_measurement",
                f"planned.measurement.{name}.value",
                "Provide a valid pre-run value.",
            )
    looks = measurement.get("interim_looks")
    if not isinstance(looks, dict) or looks.get("mode") not in ("fixed", "sequential"):
        issue(
            "sequential_unqualified",
            "planned.measurement.interim_looks",
            "Explicit fixed mode or qualified sequential schedule required.",
        )
    elif looks["mode"] == "sequential" and (
        not full
        or looks.get("frozen") is not True
        or not _text(looks.get("qualification_ref"))
        or not _text(looks.get("schedule_ref"))
    ):
        issue(
            "sequential_unqualified",
            "planned.measurement.interim_looks",
            "Full-sweep sequential mode needs a frozen S1-qualified schedule.",
        )

    cost_fields = (
        "gpu_count",
        "rate_per_gpu_hour_usd",
        "wall_hours_cap",
        "storage_cap_usd",
        "egress_cap_usd",
        "other_cap_usd",
        "total_cap_usd",
    )
    for name in resources:
        if isinstance(name, str) and name.endswith("_cap_usd") and name not in cost_fields:
            issue(
                "unknown_cost_field",
                f"planned.resources.{name}",
                "Use other_cap_usd for setup, tax and other noncompute reserves; unknown cost caps are not ignored.",
            )
    for name in cost_fields:
        if not _number(
            resources.get(name),
            positive=name in ("gpu_count", "rate_per_gpu_hour_usd", "wall_hours_cap"),
            integer=name == "gpu_count",
        ):
            issue(
                "invalid_resource",
                f"planned.resources.{name}",
                "Require a finite nonnegative cost input and positive compute quantities.",
            )
    if not _date(resources.get("quoted_at")) or not _text(resources.get("gpu_edition")):
        issue("invalid_resource", "planned.resources", "Record quote date and exact GPU edition.")
    budget = prereq.get("budget_decision_P005")
    budget_ok = _attested(budget, owner=True)
    cap_names = ("project_cap_usd", "stage_cap_usd", "pilot_cap_usd")
    compute_cap_names = ("approved_compute_gpu_hours_cap", "approved_rate_per_gpu_hour_cap")
    if not budget_ok or any(not _number(budget.get(k)) for k in cap_names):
        issue(
            "budget_unapproved",
            "prerequisites.budget_decision_P005",
            "Explicit owner-approved tier and cash caps required; filled cost is not approval.",
        )
        budget_ok = False
    if budget_ok:
        tier_limits = {"zero": 0, "up_to_2500": 2500, "up_to_6000": 6000, "collaborator": None}
        tier = budget.get("tier")
        if (
            not isinstance(tier, str)
            or tier not in tier_limits
            or not budget["pilot_cap_usd"] <= budget["stage_cap_usd"] <= budget["project_cap_usd"]
            or (
                tier in tier_limits
                and tier_limits[tier] is not None
                and budget["project_cap_usd"] > tier_limits[tier]
            )
        ):
            issue(
                "budget_unapproved",
                "prerequisites.budget_decision_P005",
                "Tier and nested explicit caps must agree.",
            )
            budget_ok = False
    compute_limits_ok = budget_ok and all(
        _number(budget.get(k), positive=True) for k in compute_cap_names
    )
    if budget_ok and not compute_limits_ok:
        issue(
            "phase_limits_unapproved",
            "prerequisites.budget_decision_P005",
            "Cash ceilings are recorded; phase GPU-hour and hourly-rate limits still need approval.",
        )
    release_cap = None
    if full:
        release = prereq.get("full_sweep_release")
        if (
            not _attested(release, owner=True)
            or not _text(release.get("pilot_evidence_ref"))
            or not _text(release.get("forecast_ref"))
            or not _number(release.get("remaining_stage_cap_usd"), positive=True)
            or (budget_ok and release["remaining_stage_cap_usd"] > budget["stage_cap_usd"])
        ):
            issue(
                "full_sweep_unreleased",
                "prerequisites.full_sweep_release",
                "Record Huy's post-pilot release, measured forecast reference and remaining stage cap.",
            )
        else:
            release_cap = release["remaining_stage_cap_usd"]
    estimated_cost = None
    if all(_number(resources.get(k)) for k in cost_fields):
        try:
            compute_gpu_hours = resources["gpu_count"] * resources["wall_hours_cap"]
            compute_cost = compute_gpu_hours * resources["rate_per_gpu_hour_usd"]
            estimated_cost = (
                compute_cost
                + resources["storage_cap_usd"]
                + resources["egress_cap_usd"]
                + resources["other_cap_usd"]
            )
        except OverflowError:
            compute_gpu_hours = None
            compute_cost = None
            estimated_cost = None
        if (
            not _number(compute_gpu_hours, positive=True)
            or not _number(compute_cost, positive=True)
            or not _number(estimated_cost)
        ):
            issue(
                "invalid_resource",
                "planned.resources",
                "Computed costs must be finite and positive compute cost must not underflow to zero.",
            )
            estimated_cost = None
        elif estimated_cost > resources["total_cap_usd"]:
            issue(
                "cost_exceeds_cap",
                "planned.resources.total_cap_usd",
                "Compute plus storage, egress and other reserves exceeds total cap.",
            )
        if (
            compute_limits_ok
            and _number(compute_gpu_hours)
            and (
                compute_gpu_hours > budget["approved_compute_gpu_hours_cap"]
                or resources["rate_per_gpu_hour_usd"] > budget["approved_rate_per_gpu_hour_cap"]
            )
        ):
            issue(
                "compute_exceeds_cap",
                "planned.resources",
                "Compute GPU-hours and hourly rate must each stay inside the approved phase limits.",
            )
        if budget_ok and estimated_cost is not None:
            phase_cap = budget["stage_cap_usd" if full else "pilot_cap_usd"]
            if resources["total_cap_usd"] > phase_cap:
                issue(
                    "cost_exceeds_cap",
                    "planned.resources.total_cap_usd",
                    "Requested total cap exceeds the approved phase cap.",
                )
        if release_cap is not None and resources["total_cap_usd"] > release_cap:
            issue(
                "cost_exceeds_cap",
                "planned.resources.total_cap_usd",
                "Requested total cap exceeds the released remaining stage budget.",
            )
    warn(
        "budget_state_unverified",
        "planned.resources",
        "Quote availability, freshness, cumulative spend and provider billing are not checked offline.",
    )
    return {
        "schema_version": 2,
        "phase": phase if isinstance(phase, str) else None,
        "ready_for_planning": not blockers,
        "cloud_launch_authorized": False,
        "attestation_authenticity": "not_verified",
        "blocking_reasons": blockers,
        "warnings": warnings,
        "estimated_total_cost_usd": estimated_cost,
        "publication": {
            "public_numbers_eligible_by_attestation": _attested(
                prereq.get("public_numbers_KCL"), owner=True
            ),
            "derived_weights_labels_eligible_by_attestation": _attested(
                prereq.get("derived_weights_labels_KCL2"), owner=True
            ),
            "publication_authorized": False,
        },
    }
