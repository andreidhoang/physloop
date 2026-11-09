"""S4.6 evaluation handoff: exported inference checkpoint -> registered S2 evaluation request.

Tickets: S4.6 (an improvement claim needs independent held-out evaluation through the S2 gate
and a registered superiority rule), E17 (training runs and roads are separate axes; a run is a
registered identity, never inferred), build-spec B3 item (5). Serves S4.6 [C] (G9, G7) under
D-024. Agent-authored evaluator engineering (D-018).

``build_evaluation_request`` binds one exported, converted inference checkpoint to the
registered S2 protocol:

- the candidate identity is the ``tensor_set_sha256`` of a ``passed``
  ``gate.tensor_manifest`` manifest of the inference checkpoint (tensor identity, rule 11);
  the manifest is re-verified through ``gate.tensor_manifest.compare_manifests`` (its seal
  must recompute), never trusted from its status field alone;
- the export chain is re-audited here (review B3-11): ``export_audit_inputs`` (the release
  and Cosmos export manifests, the policy snapshot and the host-recorded ``hf_config_is_vlm``)
  go through ``teacher_rl.export_audit.audit_export`` with this inference manifest, and the
  audit and its converter section must be ``passed``; the audit hash and input hashes enter
  the request body;
- ``checkpoint_step`` must be a step the registered run exports: the final
  ``trainer_steps``, plus every multiple of ``save_freq`` up to it when ``export_safetensors``
  is registered (``checkpoint_step_not_exported`` otherwise);
- the S2 registration must carry every ``gate.s2_gate.REGISTRATION_FIELDS`` value (checked by
  the gate glue's own registration validator) and its ``candidate`` must name exactly that
  identity (``tensor_set_sha256:<hex>``), so the gate cannot evaluate a different checkpoint
  under this request; the whole S2 registration is carried in the request body;
- the run id must be in an ``s4_run_registry`` v2 bound to the S4 registration hash whose
  length equals the registered ``independent_training_runs`` and whose aggregation copy
  equals the registered run aggregation (below); its hash enters the request body.

Run aggregation inputs (B3 re-verification): ``rule`` (``all`` | ``at_least_k``), ``k`` and
``family_alpha`` are the S4 registration's owner slot ``owner.run_aggregation``, validated by
``teacher_rl.registration`` (no defaults), so they are fixed before any request is built and
never free arguments at aggregation time. ``aggregation_spec`` reads them from the validated
registration. Run registry v2 (review B3-20): ``{artifact_kind, schema_version: 2, scope,
registration_hash, run_ids, aggregation: {rule, k, family_alpha}}``; its ``aggregation`` is
checked against the registered slot and any difference is ``aggregation_mismatch`` (the
registry never overrides the registration). v1 registries are refused
(``run_registry_invalid``).

``aggregation_spec`` re-derives every request from its body (review B3-03): the request hash,
the S2 registration hash and protocol hash, the S2 registration's validity and candidate
label, the scope, the S4 registration hash, the run registry hash and the checkpoint step are
recomputed or re-checked; a hand-written or edited request is ``request_invalid``. The S4
registration's ``improvement_decision_rule.rule_id`` must be ``gate.superiority.RULE_ID`` and
``delta_min`` is its ``smallest_improvement_of_interest.value``. Exactly one request per
registered run, all under one S2 protocol (every field except the run-specific
``candidate``), with pairwise distinct candidates (one checkpoint cannot stand for two
independent runs). The spec records the S2 registered ``required_strata``, ``metric``,
``margin``, ``alpha`` and ``analysis_mode``.

Decision-rule binding (module F4; completeness audit "S4.6 superiority binding"). The spec
also carries the registered ``owner.improvement_decision_rule`` superiority slots under
``decision_rule``: the guard endpoints (name, S2 registration id, metric, margin, alpha), the
registered ``guard_strata``/``gating_strata`` tokens or lists with their resolution against the
S2 strata (``primary_strata`` -> the S2 ``required_strata``; ``all_guard_strata`` -> the guard
strata), and the ``look``. Listed gating strata must lie within the resolved guard strata
(``gating_strata_not_subset``). A registered sequential look is refused at spec time
(``look_not_evaluable``): ``gate.s2_gate`` evaluates ``fixed_sample`` only, so no look can be
taken through this glue. ``run_identities`` records, per run, the candidate label, the
reference and the S2 registration hash its decision must be computed from.

``decide_run`` is the glue from ``gate.s2_gate.evaluate_registered`` output to
``gate.superiority.decide``: the primary evaluation must be a passed
``s2_registered_gate_evaluation`` of exactly the run's S2 registration (its
``input_hashes.registration``), in the spec scope, whose ``gate_response.comparison`` names the
run's candidate and reference and the registered strata, margin, alpha and metric; each
registered guard needs exactly one passed evaluation under its registered S2 registration id
whose comparison names the same candidate, reference, protocol identity, config hash,
evaluator, seed mode and deviation convention as the primary (the score definition belongs to
the endpoint), the resolved guard strata and the guard's registered metric, margin and alpha. Each guard
evaluation is also bound by bytes, symmetrically with the primary (review F4-03): the caller
supplies the guard's S2 registration for the run, which must be valid, carry the guard's
registered ``s2_registration_id`` and the run's candidate and reference, recompute to the
registered ``s2_protocol_hash`` (the registration without its candidate) and hash to the
evaluation's ``input_hashes.registration``; an evaluation under an edited same-id registration
(for example another score definition) is refused. Every evaluation must carry
``input_hashes.registration`` (review F4-04). ``decide`` then runs with the registered
``delta_min``, guard strata and gating strata (it also checks that each guard covers the
primary's paired observations). ``decide_and_aggregate`` runs it for every registered run, then
``aggregate`` (``gate.superiority.aggregate_runs`` with the registered run aggregation), and
returns an ``s4_registered_decision`` artifact (blocked with ``decision_binding_failed`` naming
the run on any binding defect; nothing is substituted) that also carries, per run and endpoint,
the evaluation's request hash, registration hash and unchanged ``out_of_envelope`` (>= 4 m)
section (review F4-08).

``aggregate`` takes ``{run_id: {"request_hash", "decision"}}`` (review B3-02): each decision
is bound to its registered request hash, and its ``delta_min``, ``required_strata``,
``metric`` (only a ``gate.api`` response populates it; ``None`` is refused), gate method
(``fixed_sample`` -> ``gate.superiority.FIXED_METHOD``) and primary level
(``claims.ruled_out.level``) must equal the spec; any difference raises ``ValueError``. The
primary margin is not part of ``superiority.decide`` output, so it is bound through the
request's S2 registration (the gate glue evaluates under it), not re-read from the decision.
Under F4 every decision must also carry exactly the registered guard names, each with the
registered metric, margin, alpha and the fixed-sample gate method, the resolved guard and
gating strata, and the registered look (``None`` with ``terminal`` ``None`` for a fixed look);
a ``decide(guards={})`` output under registered guards is refused.
The decisions are then handed to ``gate.superiority.aggregate_runs`` unchanged; the claim
concerns named checkpoints, not a training method.

Verification level per function:

- ``build_evaluation_request``, ``aggregation_spec``, ``aggregate``, ``decide_run``,
  ``decide_and_aggregate``: **L1** (literal known answers and negative controls in
  ``tests/test_s4_evaluate.py``; the glue is exercised on real ``evaluate_registered`` output
  of synthetic records). Running the request -
  driver load of the checkpoint, the S2 rollouts and the protected gate on actual records -
  is **L3** (``execution_status: qualify_on_host``). That the export manifest belongs to the
  named run (its run directory) is asserted by the caller, not derived here.

This module never runs a simulator, never calls the gate on records (it reads the gate's
outputs), never recomputes a bound (protected
``gate/verdict.py`` and ``gate/stats/`` are imported by the gate glue, never copied), never
chooses a run id, rule, alpha, margin or delta, and never authorizes anything.
"""

import hashlib
import re
from pathlib import Path

from gate import superiority
from gate.s2_gate import _registration_issues as s2_registration_issues
from gate.s2_identity import identity_hash
from gate.tensor_manifest import compare_manifests
from teacher_rl.export_audit import audit_export
from teacher_rl.registration import ALL_GUARD_STRATA, PRIMARY_STRATA, validate_registration

REQUEST_KIND = "s4_evaluation_request"
SPEC_KIND = "s4_run_aggregation_spec"
AGGREGATE_KIND = "s4_run_aggregation"
DECISION_KIND = "s4_registered_decision"
EVALUATION_KIND = "s2_registered_gate_evaluation"
# Protocol fields a guard evaluation shares with the primary (score_definition is per endpoint).
SHARED_PROTOCOL_KEYS = ("identity", "config_hash", "evaluator", "seed_mode",
                        "deviation_convention")
REGISTRY_KIND = "s4_run_registry"
SCHEMA_VERSION = 1
REGISTRY_SCHEMA_VERSION = 2
SCOPES = ("synthetic", "public_source_fixture", "actual")
CANDIDATE_PREFIX = "tensor_set_sha256:"
GATE_ENTRY = "gate.s2_gate.evaluate_registered"
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_REGISTRY_KEYS = {"artifact_kind", "schema_version", "scope", "registration_hash", "run_ids",
                  "aggregation"}
_AGGREGATION_KEYS = {"rule", "k", "family_alpha"}
_AUDIT_INPUT_KEYS = {"release", "export", "policy_snapshot", "hf_config_is_vlm"}
REQUEST_BODY_KEYS = (
    "s4_registration_hash", "s4_registration_id", "evaluated_policy_path", "run_id",
    "checkpoint_step", "candidate", "reference", "s2_registration", "s2_registration_hash",
    "s2_protocol_hash", "s2_registration_id", "protocol", "run_registry_hash", "export_audit",
)
S2_REGISTERED_KEYS = ("registration_id", "required_strata", "metric", "margin", "alpha",
                      "analysis_mode")
_SPEC_BODY_KEYS = ("registered_run_ids", "rule", "k", "family_alpha", "delta_min", "rule_id",
                   "request_hashes", "run_registry_hash", "s2_protocol_hash", "s2_registered",
                   "decision_rule", "run_identities")
_ANALYSIS_METHODS = {"fixed_sample": superiority.FIXED_METHOD}
_EVALUATION_ENTRY_KEYS = {"primary", "guards", "guard_registrations"}


def _issue(code, path, message, severity="blocking"):
    return {"code": code, "path": path, "severity": severity, "evidence_refs": [],
            "message": message}


def _producer():
    return {"module": "teacher_rl.evaluate",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _hash(value):
    try:
        return identity_hash(value)
    except (TypeError, ValueError):
        return None


def _aggregation_issues(aggregation, m, issues):
    """Owner-registered run aggregation inputs inside the run registry (no defaults)."""
    rule, k, alpha = aggregation["rule"], aggregation["k"], aggregation["family_alpha"]
    path = "run_registry.aggregation"
    if rule not in superiority.RUN_RULES:
        issues.append(_issue("aggregation_rule_unregistered", f"{path}.rule",
                             f"rule is one of {list(superiority.RUN_RULES)} (owner input)"))
    if rule == "all" and k is not None:
        issues.append(_issue("aggregation_k_invalid", f"{path}.k",
                             "k applies only to at_least_k"))
    if rule == "at_least_k" and (type(k) is not int or not 1 <= k <= m):
        issues.append(_issue("aggregation_k_invalid", f"{path}.k",
                             "at_least_k needs 1 <= k <= runs"))
    if type(alpha) is not float or not 0 < alpha < 1:
        issues.append(_issue("family_alpha_invalid", f"{path}.family_alpha",
                             "family_alpha is a float strictly between 0 and 1 (owner input)"))


def _check_registry(registry, report, issues, registered_aggregation=None):
    """Run ids registered for this S4 registration; returns the list or None."""
    if type(registry) is not dict or set(registry) != _REGISTRY_KEYS or registry.get(
            "artifact_kind") != REGISTRY_KIND or registry.get("schema_version") != (
            REGISTRY_SCHEMA_VERSION) or type(registry.get("schema_version")) is not int:
        issues.append(_issue("run_registry_invalid", "run_registry",
                             f"{REGISTRY_KIND} v{REGISTRY_SCHEMA_VERSION} with exactly "
                             f"{sorted(_REGISTRY_KEYS)} (v1 lacks the aggregation inputs)"))
        return None
    ids = registry["run_ids"]
    if type(ids) is not list or not ids or len(set(ids)) != len(ids) or not all(
            type(i) is str and _RUN_ID.fullmatch(i) for i in ids):
        issues.append(_issue("run_registry_invalid", "run_registry.run_ids",
                             "unique lowercase run id slugs"))
        return None
    aggregation = registry["aggregation"]
    if type(aggregation) is not dict or set(aggregation) != _AGGREGATION_KEYS:
        issues.append(_issue("aggregation_unregistered", "run_registry.aggregation",
                             f"the registry carries exactly {sorted(_AGGREGATION_KEYS)}"))
        return None
    before = len(issues)
    _aggregation_issues(aggregation, len(ids), issues)
    if len(issues) != before:
        return None
    if report is None or report["status"] != "passed":
        return None
    if registry["registration_hash"] != report["registration_hash"]:
        issues.append(_issue("run_registry_invalid", "run_registry.registration_hash",
                             "the registry belongs to a different S4 registration"))
        return None
    if _hash(aggregation) != _hash(registered_aggregation):
        issues.append(_issue("aggregation_mismatch", "run_registry.aggregation",
                             "the registry's aggregation differs from the registered "
                             "owner.run_aggregation"))
        return None
    return ids


def _registered(s4_registration, registry, scope, issues):
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    report = validate_registration(s4_registration)
    if report["status"] != "passed":
        issues.append(_issue("s4_registration_blocked", "s4_registration",
                             f"{sorted({i['code'] for i in report['issues']})}"))
        report = None
    elif s4_registration["scope"] != scope:
        issues.append(_issue("scope_mismatch", "s4_registration.scope",
                             "registration scope differs from the declared scope"))
    registered = None if report is None else s4_registration["owner"]["run_aggregation"]
    ids = _check_registry(registry, report, issues, registered)
    if ids is not None:
        if registry["scope"] != scope:
            issues.append(_issue("scope_mismatch", "run_registry.scope", "scope differs"))
        runs = s4_registration["owner"]["independent_training_runs"]
        if len(ids) != runs:
            issues.append(_issue("run_registry_invalid", "run_registry.run_ids",
                                 f"{len(ids)} run ids for {runs} registered training runs"))
            ids = None
    return report, ids


def exported_steps(s4_registration):
    """Trainer steps the registered run exports (final always; intermediates by save_freq)."""
    owner = s4_registration["owner"]
    final = owner["optimizer_steps_per_run"]["trainer_steps"]
    retention = owner["checkpoint_retention"]
    steps = {final}
    if retention["export_safetensors"] is True:
        steps.update(range(retention["save_freq"], final + 1, retention["save_freq"]))
    return sorted(steps)


def _step_issues(s4_registration, step, report, issues):
    if type(step) is not int or step < 0:
        issues.append(_issue("invalid_checkpoint_step", "checkpoint_step",
                             "an exported trainer step is an int >= 0"))
    elif report is not None and step not in exported_steps(s4_registration):
        issues.append(_issue("checkpoint_step_not_exported", "checkpoint_step",
                             f"step {step} is not exported by the registered run "
                             f"(exported: {exported_steps(s4_registration)})"))


def _export_audit(inputs, inference_manifest, scope, issues):
    """Re-run the export audit for this inference manifest; return its binding or None."""
    if type(inputs) is not dict or set(inputs) != _AUDIT_INPUT_KEYS:
        issues.append(_issue("export_audit_missing", "export_audit_inputs",
                             f"the export chain {sorted(_AUDIT_INPUT_KEYS)} is required"))
        return None
    try:
        report = audit_export(release=inputs["release"], export=inputs["export"],
                              hf_config_is_vlm=inputs["hf_config_is_vlm"],
                              inference=inference_manifest,
                              policy_snapshot=inputs["policy_snapshot"])
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        issues.append(_issue("export_audit_blocked", "export_audit_inputs",
                             f"{type(exc).__name__}: the export audit cannot run"))
        return None
    if report["status"] != "passed" or (report.get("converter") or {}).get(
            "status") != "passed":
        codes = sorted({i["code"] for i in report["issues"] if i["severity"] == "error"})
        issues.append(_issue("export_audit_blocked", "export_audit_inputs",
                             f"the export audit with this inference manifest is not passed "
                             f"(converter status {(report.get('converter') or {}).get('status')}"
                             f"; {codes})"))
        return None
    if report["scope"] != scope:
        issues.append(_issue("scope_mismatch", "export_audit_inputs", "scope differs"))
        return None
    return {"audit_hash": identity_hash(report), "input_hashes": report["input_hashes"]}


def _s2_protocol_hash(s2_registration):
    return identity_hash({k: v for k, v in s2_registration.items() if k != "candidate"})


def build_evaluation_request(*, inference_manifest, s4_registration, run_registry, run_id,
                             checkpoint_step, s2_registration, scope, export_audit_inputs):
    """Bind one inference checkpoint to the registered S2 protocol (L1; execution L3)."""
    issues = []
    report, ids = _registered(s4_registration, run_registry, scope, issues)
    if ids is not None and run_id not in ids:
        issues.append(_issue("run_id_unregistered", "run_id",
                             f"{run_id!r} is not a registered training run"))
    _step_issues(s4_registration, checkpoint_step, report, issues)
    check = compare_manifests(inference_manifest, inference_manifest)
    candidate = audit = None
    if check["status"] != "passed":
        issues.append(_issue("manifest_invalid", "inference_manifest",
                             f"{sorted({i['code'] for i in check['issues']})}"))
    else:
        if inference_manifest["scope"] != scope:
            issues.append(_issue("scope_mismatch", "inference_manifest.scope", "scope differs"))
        files = [{"name": f["name"], "sha256": f["sha256"]} for f in inference_manifest["files"]]
        if inference_manifest["index"] is not None:
            files.append({"name": inference_manifest["index"]["name"],
                          "sha256": inference_manifest["index"]["sha256"]})
        candidate = {
            "label": CANDIDATE_PREFIX + inference_manifest["tensor_set_sha256"],
            "tensor_set_sha256": inference_manifest["tensor_set_sha256"],
            "tensor_count": inference_manifest["totals"]["tensor_count"],
            "tensor_bytes": inference_manifest["totals"]["tensor_bytes"],
            "files": sorted(files, key=lambda f: f["name"]),
            "manifest_hash": _hash(inference_manifest),
        }
        audit = _export_audit(export_audit_inputs, inference_manifest, scope, issues)
    s2_problems = s2_registration_issues(s2_registration)
    if s2_problems:
        issues.append(_issue("s2_registration_invalid", "s2_registration",
                             f"{sorted({p['path'] for p in s2_problems})}"))
    elif candidate is not None and s2_registration["candidate"] != candidate["label"]:
        issues.append(_issue("candidate_identity_mismatch", "s2_registration.candidate",
                             "the S2 registration must name exactly this checkpoint's tensor "
                             f"identity {candidate['label']}"))
    issues.sort(key=lambda i: (i["path"], i["code"]))
    blocked = bool(issues)
    body = None if blocked else {
        "s4_registration_hash": report["registration_hash"],
        "s4_registration_id": s4_registration["registration_id"],
        "evaluated_policy_path": s4_registration["owner"]["evaluated_policy_path"],
        "run_id": run_id, "checkpoint_step": checkpoint_step, "candidate": candidate,
        "reference": s2_registration["reference"],
        "s2_registration": s2_registration,
        "s2_registration_hash": identity_hash(s2_registration),
        "s2_protocol_hash": _s2_protocol_hash(s2_registration),
        "s2_registration_id": s2_registration["registration_id"],
        "protocol": s2_registration["protocol"],
        "run_registry_hash": identity_hash(run_registry),
        "export_audit": audit,
    }
    return {
        "artifact_kind": REQUEST_KIND, "schema_version": SCHEMA_VERSION, "scope": scope,
        "producer": _producer(),
        "input_hashes": {"inference_manifest": _hash(inference_manifest),
                         "s4_registration": _hash(s4_registration),
                         "run_registry": _hash(run_registry),
                         "s2_registration": _hash(s2_registration),
                         "export_audit_inputs": _hash(export_audit_inputs)},
        "units": {"checkpoint_step": "trainer steps", "tensor_bytes": "bytes"},
        "status": "blocked" if blocked else "passed", "issues": issues, "level": "L1",
        "execution_status": "qualify_on_host", "gate_entry": GATE_ENTRY,
        **(body or {key: None for key in REQUEST_BODY_KEYS}
           | {"run_id": run_id, "checkpoint_step": checkpoint_step}),
        "qualify_on_host": [
            "the gate driver loads this inference checkpoint and its bytes match the manifest",
            "the S2 rollouts run under the registered protocol; the protected gate decides",
            "the E16 train/gate adapter audit is attached before attributing differences",
            "the export manifest was produced by the named run's run directory",
        ],
        "request_hash": None if blocked else identity_hash(body),
        "cloud_launch_authorized": False,
    }


def _request_problem(req, *, report, registry_hash, s4_registration, scope):
    """Recompute a request exactly as ``build_evaluation_request`` does; None when sound."""
    try:
        if req["artifact_kind"] != REQUEST_KIND or req["schema_version"] != SCHEMA_VERSION or (
                req["status"] != "passed"):
            return "request_not_passed", "a passed s4_evaluation_request v1"
        body = {key: req[key] for key in REQUEST_BODY_KEYS}
        if identity_hash(body) != req["request_hash"]:
            return "request_invalid", "the request body does not match its request_hash"
        if req["scope"] != scope:
            return "scope_mismatch", "the request scope differs"
        s2 = body["s2_registration"]
        if s2_registration_issues(s2):
            return "request_invalid", "the carried S2 registration is invalid"
        candidate = body["candidate"]
        if (identity_hash(s2) != body["s2_registration_hash"]
                or _s2_protocol_hash(s2) != body["s2_protocol_hash"]
                or s2["registration_id"] != body["s2_registration_id"]
                or s2["protocol"] != body["protocol"] or s2["reference"] != body["reference"]
                or s2["candidate"] != candidate["label"]
                or candidate["label"] != CANDIDATE_PREFIX + candidate["tensor_set_sha256"]):
            return "request_invalid", "the request's S2 binding does not recompute"
        if report is not None and body["s4_registration_hash"] != report["registration_hash"]:
            return "request_registration_mismatch", "the request names another S4 registration"
        if body["run_registry_hash"] != registry_hash:
            return "request_registry_mismatch", "the request was built under another registry"
        if report is not None and body["checkpoint_step"] not in exported_steps(
                s4_registration):
            return "checkpoint_step_not_exported", "the step is not exported by the run"
        if type(body["export_audit"]) is not dict:
            return "request_invalid", "the request carries no export audit binding"
    except (KeyError, TypeError, AttributeError, ValueError):
        return "request_invalid", "a structurally incomplete evaluation request"
    return None


def _decision_rule(rule, s2, issues):
    """The registered superiority slots resolved against the S2 strata (spec body part)."""
    required = sorted(s2["required_strata"])
    path = "s4_registration.owner.improvement_decision_rule"
    guard_strata = (required if rule["guard_strata"] == PRIMARY_STRATA
                    else sorted(rule["guard_strata"]))
    gating = (guard_strata if rule["gating_strata"] == ALL_GUARD_STRATA
              else sorted(rule["gating_strata"]))
    if not set(gating) <= set(guard_strata):
        issues.append(_issue("gating_strata_not_subset", f"{path}.gating_strata",
                             f"gating strata {gating} are not within the guard strata "
                             f"{guard_strata}"))
    look = rule["look"]
    if look["kind"] != "fixed" or s2["analysis_mode"] != "fixed_sample":
        issues.append(_issue("look_not_evaluable", f"{path}.look",
                             "gate.s2_gate evaluates fixed_sample only; a registered "
                             f"{look['kind']} look cannot be taken through this glue"))
    return {
        "guards": sorted((dict(guard) for guard in rule["guards"]), key=lambda g: g["name"]),
        "guard_strata": rule["guard_strata"], "gating_strata": rule["gating_strata"],
        "look": dict(look), "resolved_guard_strata": guard_strata,
        "resolved_gating_strata": gating,
    }


def aggregation_spec(*, requests, s4_registration, run_registry, scope):
    """Registered inputs of ``superiority.aggregate_runs`` (L1); read from the registration."""
    issues = []
    report, ids = _registered(s4_registration, run_registry, scope, issues)
    registry_hash = _hash(run_registry)
    delta_min = None
    if report is not None:
        owner = s4_registration["owner"]
        if owner["improvement_decision_rule"]["rule_id"] != superiority.RULE_ID:
            issues.append(_issue("decision_rule_mismatch", "s4_registration.owner."
                                 "improvement_decision_rule.rule_id",
                                 f"aggregation implements only {superiority.RULE_ID}"))
        delta_min = owner["smallest_improvement_of_interest"]["value"]
    by_run = {}
    if type(requests) is not list:
        issues.append(_issue("invalid_requests", "requests", "a list of evaluation requests"))
        requests = []
    for index, req in enumerate(requests):
        path = f"requests[{index}]"
        if type(req) is not dict:
            issues.append(_issue("request_not_passed", path, "a passed evaluation request"))
            continue
        problem = _request_problem(req, report=report, registry_hash=registry_hash,
                                   s4_registration=s4_registration, scope=scope)
        if problem is not None:
            issues.append(_issue(problem[0], path, problem[1]))
            continue
        if req["run_id"] in by_run:
            issues.append(_issue("run_duplicated", path, f"second request for {req['run_id']}"))
            continue
        by_run[req["run_id"]] = req
    if ids is not None:
        for run in ids:
            if run not in by_run:
                issues.append(_issue("run_missing_request", f"run:{run}",
                                     "every registered run needs exactly one request"))
        for run in sorted(set(by_run) - set(ids)):
            issues.append(_issue("run_id_unregistered", f"run:{run}", "not a registered run"))
    if len({r["s2_protocol_hash"] for r in by_run.values()}) > 1:
        issues.append(_issue("protocol_mismatch", "requests",
                             "every run is evaluated under one S2 registration (all fields "
                             "except its run-specific candidate)"))
    candidates = [r["candidate"]["tensor_set_sha256"] for r in by_run.values()]
    if len(set(candidates)) != len(candidates):
        issues.append(_issue("candidate_reused", "requests",
                             "one checkpoint cannot stand for two independent runs"))
    decision_rule = None
    if report is not None and ids is not None and by_run and all(r in by_run for r in ids):
        decision_rule = _decision_rule(owner["improvement_decision_rule"],
                                       by_run[ids[0]]["s2_registration"], issues)
    issues.sort(key=lambda i: (i["path"], i["code"]))
    blocked = bool(issues)
    body = None
    if not blocked:
        aggregation = s4_registration["owner"]["run_aggregation"]
        s2 = by_run[ids[0]]["s2_registration"]
        body = {
            "registered_run_ids": list(ids), "rule": aggregation["rule"], "k": aggregation["k"],
            "family_alpha": aggregation["family_alpha"], "delta_min": delta_min,
            "rule_id": superiority.RULE_ID,
            "request_hashes": {run: by_run[run]["request_hash"] for run in ids},
            "run_registry_hash": registry_hash,
            "s2_protocol_hash": by_run[ids[0]]["s2_protocol_hash"],
            "s2_registered": {key: s2[key] for key in S2_REGISTERED_KEYS},
            "decision_rule": decision_rule,
            "run_identities": {run: {"candidate": by_run[run]["candidate"]["label"],
                                     "reference": by_run[run]["reference"],
                                     "s2_registration_hash": by_run[run]["s2_registration_hash"]}
                               for run in ids},
        }
    return {
        "artifact_kind": SPEC_KIND, "schema_version": SCHEMA_VERSION, "scope": scope,
        "producer": _producer(),
        "input_hashes": {"requests": _hash(requests), "s4_registration": _hash(s4_registration),
                         "run_registry": registry_hash},
        "units": {"family_alpha": "probability", "delta_min": "registered metric units"},
        "status": "blocked" if blocked else "passed", "issues": issues, "level": "L1",
        **(body or {key: None for key in _SPEC_BODY_KEYS} | {"rule_id": superiority.RULE_ID}),
        "spec_hash": None if blocked else identity_hash(body),
    }


def _bound_decisions(decisions, spec):
    """Check each run's decision against its request hash and the registered S2 values."""
    if type(decisions) is not dict or set(decisions) != set(spec["registered_run_ids"]):
        raise ValueError("decisions must cover exactly the registered runs")
    registered = spec["s2_registered"]
    method = _ANALYSIS_METHODS.get(registered["analysis_mode"])
    plain = {}
    for run in spec["registered_run_ids"]:
        entry = decisions[run]
        if type(entry) is not dict or set(entry) != {"request_hash", "decision"}:
            raise ValueError(f"run {run!r}: pass {{'request_hash', 'decision'}}")
        if entry["request_hash"] != spec["request_hashes"][run]:
            raise ValueError(f"run {run!r}: the decision is not bound to its registered request")
        d = entry["decision"]
        if type(d) is not dict:
            raise ValueError(f"run {run!r}: a superiority.decide output is required")
        if d.get("delta_min") != spec["delta_min"] or type(d.get("delta_min")) is not type(
                spec["delta_min"]):
            raise ValueError(f"run {run!r}: decided with delta_min {d.get('delta_min')!r}, "
                             f"registered {spec['delta_min']!r}")
        if list(d.get("required_strata") or ()) != sorted(registered["required_strata"]):
            raise ValueError(f"run {run!r}: decided on strata {d.get('required_strata')!r}, "
                             f"registered {sorted(registered['required_strata'])!r}")
        if d.get("metric") is None or d.get("metric") != registered["metric"]:
            raise ValueError(f"run {run!r}: the decision's metric is not the registered metric "
                             "(decide from the gate.api response, which carries it)")
        if method is None or d.get("gate_method") != method:
            raise ValueError(f"run {run!r}: gate method {d.get('gate_method')!r} does not match "
                             f"analysis_mode {registered['analysis_mode']!r}")
        level = ((d.get("claims") or {}).get("ruled_out") or {}).get("level")
        if level != registered["alpha"]:
            raise ValueError(f"run {run!r}: primary level {level!r} is not the registered alpha "
                             f"{registered['alpha']!r}")
        _check_decision_rule(run, d, spec["decision_rule"], method)
        plain[run] = d
    return plain


def _check_decision_rule(run, d, rule, method):
    """F4: a decision carries exactly the registered guards, strata and look."""
    guards = d.get("guards")
    names = [guard["name"] for guard in rule["guards"]]
    if type(guards) is not dict or sorted(guards) != names:
        raise ValueError(f"run {run!r}: decided with guards {sorted(guards or {})!r}, "
                         f"registered {names!r}")
    for guard in rule["guards"]:
        got = guards[guard["name"]]
        for key in ("metric", "margin", "alpha"):
            if type(got) is not dict or got.get(key) != guard[key] or type(
                    got.get(key)) is not type(guard[key]):
                raise ValueError(f"run {run!r}: guard {guard['name']!r} {key} "
                                 f"{(got or {}).get(key)!r}, registered {guard[key]!r}")
        if got.get("method") != method:
            raise ValueError(f"run {run!r}: guard {guard['name']!r} gate method "
                             f"{got.get('method')!r} does not match the registered analysis")
    for key in ("guard_strata", "gating_strata"):
        if list(d.get(key) or ()) != rule[f"resolved_{key}"]:
            raise ValueError(f"run {run!r}: decided with {key} {d.get(key)!r}, registered "
                             f"{rule[f'resolved_{key}']!r}")
    look = rule["look"]
    expected = (None, None) if look["kind"] == "fixed" else (look["look_index"], True)
    if (d.get("look"), d.get("terminal")) != expected:
        raise ValueError(f"run {run!r}: decided at look {d.get('look')!r} (terminal "
                         f"{d.get('terminal')!r}), registered {look!r}")


def aggregate(decisions, spec):
    """``gate.superiority.aggregate_runs`` over bound per-run decisions under a passed spec (L1).

    ``decisions`` maps each registered run id to ``{"request_hash": <its request hash>,
    "decision": <superiority.decide output>}``; any unbound or off-registration decision
    raises ``ValueError``.
    """
    _verified_spec(spec)
    plain = _bound_decisions(decisions, spec)
    result = superiority.aggregate_runs(plain, registered_run_ids=spec["registered_run_ids"],
                                        rule=spec["rule"], family_alpha=spec["family_alpha"],
                                        k=spec["k"])
    return {"artifact_kind": AGGREGATE_KIND, "schema_version": SCHEMA_VERSION,
            "scope": spec["scope"], "producer": _producer(),
            "input_hashes": {"spec": spec["spec_hash"],
                             "decisions": _hash(decisions)},
            "units": {}, "status": "passed", "issues": [], "aggregate": result,
            "scope_of_claim": "named checkpoints of the registered runs, not a training method"}


def _verified_spec(spec):
    if type(spec) is not dict or spec.get("artifact_kind") != SPEC_KIND or spec.get(
            "status") != "passed":
        raise ValueError("a passed s4_run_aggregation_spec is required")
    try:
        body = {key: spec[key] for key in _SPEC_BODY_KEYS}
    except KeyError:
        raise ValueError("spec lacks a registered field") from None
    if identity_hash(body) != spec["spec_hash"]:
        raise ValueError("spec content does not match its spec_hash")
    return body


def _evaluation(value, what, spec, identity, registration_id):
    """A passed evaluate_registered artifact of the expected registration; its comparison."""
    if type(value) is not dict or value.get("artifact_kind") != EVALUATION_KIND or value.get(
            "schema_version") != 1 or value.get("status") != "passed" or value.get(
            "verdict") is None or type(value.get("gate_response")) is not dict:
        raise ValueError(f"{what}: a passed {EVALUATION_KIND} with a verdict is required")
    if value.get("scope") != spec["scope"]:
        raise ValueError(f"{what}: scope {value.get('scope')!r}, spec {spec['scope']!r}")
    if value.get("registration_id") != registration_id:
        raise ValueError(f"{what}: evaluated under {value.get('registration_id')!r}, "
                         f"registered {registration_id!r}")
    hashes = value.get("input_hashes")
    if type(hashes) is not dict or type(hashes.get("registration")) is not str:
        raise ValueError(f"{what}: the evaluation carries no input_hashes.registration "
                         "(review F4-04)")
    comparison = value["gate_response"].get("comparison")
    if type(comparison) is not dict:
        raise ValueError(f"{what}: the gate response carries no comparison")
    for key in ("candidate", "reference"):
        if comparison.get(key) != identity[key]:
            raise ValueError(f"{what}: compares {key} {comparison.get(key)!r}, the run's "
                             f"{identity[key]!r}")
    return comparison


def _expect(what, comparison, key, expected):
    got = comparison.get(key)
    if key == "required_strata":
        got = sorted(got) if type(got) is list else got
    if got != expected or type(got) is not type(expected):
        raise ValueError(f"{what}: comparison {key} {got!r}, registered {expected!r}")


def _guard_registration(what, registration, guard, identity, evaluation):
    """Review F4-03: the guard's S2 registration is the registered one, for this run."""
    if type(registration) is not dict or s2_registration_issues(registration):
        raise ValueError(f"{what}: a valid S2 registration of the guard endpoint is required")
    if registration["registration_id"] != guard["s2_registration_id"] or any(
            registration[key] != identity[key] for key in ("candidate", "reference")):
        raise ValueError(f"{what}: the supplied S2 registration is not the guard's "
                         "registration for this run's candidate and reference")
    if _s2_protocol_hash(registration) != guard["s2_protocol_hash"]:
        raise ValueError(f"{what}: the supplied S2 registration does not match the registered "
                         "s2_protocol_hash")
    if identity_hash(registration) != evaluation["input_hashes"]["registration"]:
        raise ValueError(f"{what}: not evaluated under the guard's registered S2 registration")


def decide_run(*, spec, run_id, primary, guards, guard_registrations):
    """``gate.superiority.decide`` on one run's ``evaluate_registered`` outputs (L1).

    ``primary`` is the evaluation under the run's S2 registration; ``guards`` maps each
    registered guard name to its evaluation and ``guard_registrations`` maps it to the S2
    registration that evaluation used (bound to the registered ``s2_protocol_hash``; review
    F4-03). Returns ``{"request_hash", "decision"}``, the entry ``aggregate`` takes. Any binding
    defect raises ``ValueError`` naming it.
    """
    body = _verified_spec(spec)
    if run_id not in body["registered_run_ids"]:
        raise ValueError(f"run {run_id!r} is not a registered run")
    identity = body["run_identities"][run_id]
    registered, rule = body["s2_registered"], body["decision_rule"]
    what = f"run {run_id!r} primary"
    comparison = _evaluation(primary, what, spec, identity, registered["registration_id"])
    if primary["input_hashes"]["registration"] != identity["s2_registration_hash"]:
        raise ValueError(f"{what}: not evaluated under the run's registered S2 registration")
    _expect(what, comparison, "required_strata", sorted(registered["required_strata"]))
    for key in ("margin", "alpha", "metric"):
        _expect(what, comparison, key, registered[key])
    protocol = comparison.get("protocol")
    if type(guards) is not dict or sorted(guards) != [g["name"] for g in rule["guards"]]:
        raise ValueError(f"run {run_id!r}: guard evaluations {sorted(guards or {})!r}, "
                         f"registered {[g['name'] for g in rule['guards']]!r}")
    if type(guard_registrations) is not dict or sorted(guard_registrations) != sorted(guards):
        raise ValueError(f"run {run_id!r}: guard registrations "
                         f"{sorted(guard_registrations or {})!r}, registered "
                         f"{[g['name'] for g in rule['guards']]!r}")
    results = {}
    for guard in rule["guards"]:
        what = f"run {run_id!r} guard {guard['name']!r}"
        got = _evaluation(guards[guard["name"]], what, spec, identity,
                          guard["s2_registration_id"])
        _expect(what, got, "required_strata", rule["resolved_guard_strata"])
        for key in ("margin", "alpha", "metric"):
            _expect(what, got, key, guard[key])
        if type(protocol) is not dict or type(got.get("protocol")) is not dict or any(
                got["protocol"].get(key) != protocol.get(key) for key in SHARED_PROTOCOL_KEYS):
            raise ValueError(f"{what}: protocol differs from the primary's in "
                             f"{list(SHARED_PROTOCOL_KEYS)}")
        _guard_registration(what, guard_registrations[guard["name"]], guard, identity,
                            guards[guard["name"]])
        results[guard["name"]] = guards[guard["name"]]["gate_response"]
    decision = superiority.decide(
        primary["gate_response"], required_strata=registered["required_strata"],
        guards=results, delta_min=body["delta_min"],
        guard_strata=(None if rule["guard_strata"] == PRIMARY_STRATA
                      else rule["resolved_guard_strata"]),
        gating_strata=(None if rule["gating_strata"] == ALL_GUARD_STRATA
                       else rule["resolved_gating_strata"]))
    return {"request_hash": body["request_hashes"][run_id], "decision": decision}


def decide_and_aggregate(evaluations, spec):
    """Decide every registered run from its evaluations, then aggregate (L1).

    ``evaluations`` maps each registered run id to ``{"primary": <evaluation>, "guards":
    {name: <evaluation>}, "guard_registrations": {name: <its S2 registration>}}``. Returns an
    ``s4_registered_decision`` artifact; its ``evaluations`` section reports, per run and
    endpoint, each evaluation's ``request_hash``, registration hash and its ``out_of_envelope``
    (>= 4 m) section unchanged, so those rollouts are reported separately (review F4-08).
    """
    issues, decisions, aggregated, carried = [], None, None, None
    try:
        body = _verified_spec(spec)
        if type(evaluations) is not dict or set(evaluations) != set(body["registered_run_ids"]):
            raise ValueError("evaluations must cover exactly the registered runs")
        decisions = {}
        for run in body["registered_run_ids"]:
            entry = evaluations[run]
            if type(entry) is not dict or set(entry) != _EVALUATION_ENTRY_KEYS:
                raise ValueError(f"run {run!r}: pass {sorted(_EVALUATION_ENTRY_KEYS)}")
            decisions[run] = decide_run(spec=spec, run_id=run, primary=entry["primary"],
                                        guards=entry["guards"],
                                        guard_registrations=entry["guard_registrations"])
        aggregated = aggregate(decisions, spec)
        carried = {run: {"primary": _carried(evaluations[run]["primary"],
                                             f"run {run!r} primary"),
                         "guards": {name: _carried(value, f"run {run!r} guard {name!r}")
                                    for name, value in
                                    sorted(evaluations[run]["guards"].items())}}
                   for run in body["registered_run_ids"]}
    except (ValueError, TypeError) as exc:
        issues.append(_issue("decision_binding_failed", "evaluations", str(exc)))
        decisions = aggregated = carried = None
    return {
        "artifact_kind": DECISION_KIND, "schema_version": SCHEMA_VERSION,
        "scope": spec.get("scope") if type(spec) is dict else None, "producer": _producer(),
        "input_hashes": {"spec": spec.get("spec_hash") if type(spec) is dict else None,
                         "evaluations": _hash(evaluations)},
        "units": {"delta_min": "registered metric units"},
        "status": "blocked" if issues else "passed", "issues": issues, "level": "L1",
        "decisions": None if decisions is None else {
            run: entry["decision"] for run, entry in decisions.items()},
        "outcomes": None if decisions is None else {
            run: entry["decision"]["outcome"] for run, entry in decisions.items()},
        "aggregate": None if aggregated is None else aggregated["aggregate"],
        "evaluations": carried,
        "scope_of_claim": "named checkpoints of the registered runs, not a training method",
        "cloud_launch_authorized": False,
    }


def _carried(evaluation, what):
    """Review F4-08: what the decision artifact keeps of one bound evaluation.

    G3: a passed evaluation missing its ``request_hash`` (64 lowercase hex) or its
    ``out_of_envelope`` section (an object) raises ``ValueError`` naming the run and endpoint,
    so ``decide_and_aggregate`` returns the blocked ``decision_binding_failed`` artifact
    instead of a ``KeyError``; nothing is substituted.
    """
    request_hash = evaluation.get("request_hash")
    if type(request_hash) is not str or _HEX64.fullmatch(request_hash) is None:
        raise ValueError(f"{what}: the evaluation carries no request_hash (64 lowercase hex)")
    envelope = evaluation.get("out_of_envelope")
    if type(envelope) is not dict:
        raise ValueError(f"{what}: the evaluation carries no out_of_envelope section")
    return {"request_hash": request_hash,
            "registration_hash": evaluation["input_hashes"]["registration"],
            "out_of_envelope": envelope}
