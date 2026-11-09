"""``python -m teacher_rl``: dry-run CLI over the S4 engineering harness (build-spec B3 item 7).

Serves S4.1-S4.4/S4.6 [C] (G9, G7) under D-024. Agent-authored evaluator engineering (D-018).

Commands (each reads strict JSON files and writes one canonical JSON artifact to stdout)::

    validate-registration REG.json
    compile REG.json
    bootstrap-plan --inventory INV.json --action-record REC.json --rental-record RENT.json
                   --phases host,alpagym,...
    forecast INPUT.json
    export-audit --release R.json --export E.json --hf-config-is-vlm true|false
                 [--inference I.json] [--policy-snapshot P.json]
    parity --reference A.json --candidate B.json [--tolerance T.json]
    evaluation-request --manifest M.json --s4-registration REG.json --run-registry RR.json
                       --run-id ID --checkpoint-step N --s2-registration S2.json --scope S
                       --export-audit-inputs EA.json
    aggregation-spec --request Q.json [--request ...] --s4-registration REG.json
                     --run-registry RR.json --scope S
    launch-plan --registration REG.json --alpagym-checkout DIR --capture-tree DIR
                --capture-manifest FILE --capture-config FILE --worker-credentials MODE
                --action-authority-ref REF [--alpasim-cache-dir DIR --nurec-terms-ref REF
                --nurec-acquire-ref REF] [--run-dir DIR --base-cosmos-toml FILE
                --resolved-config-sha256 HEX]
    decide-and-aggregate --spec AGG.json --evaluations EVALS.json
    watch-spec --launch-spec LS.json --run-id ID --roles ROLES.json --containers C.json
               --limits LIMITS.json
    watch --spec WATCH.json [--execute --launch-spec LS.json --registration REG.json
          --decision-id D-NNN --progress-dir DIR [--decisions docs/DECISIONS.md]]
    rehearse --workdir /abs/new/dir

``decide-and-aggregate`` calls ``teacher_rl.evaluate.decide_and_aggregate(evaluations, spec)``:
``AGG.json`` is an ``aggregation-spec`` output and ``EVALS.json`` maps every registered run id
to ``{primary, guards, guard_registrations}``. ``watch-spec`` composes an ``s4_watch_spec`` v2
(``teacher_rl.watchdog.build_watch_spec``) from the launch spec, the role PIDs/PGIDs captured
on the host (``ROLES.json``: ``[{name, role, pid, pgid}]``), the launch's container record
(``C.json``: ``{compose_project, container_ids}``) and the six timing limits.
``watch`` without ``--execute`` validates the spec and prints the teardown plan
(``s4_watch_plan``); it signals nothing and runs nothing. ``watch --execute`` (G1) first
requires the same hash-bound authority as the launch
(``teacher_rl.capture.host_launch.verify_launch_authority`` over the bytes of
``docs/DECISIONS.md``, ``--decisions`` overriding the path for tests), an actual-scope spec
whose ``launch_hash`` is the authorized launch's, a registration that passes
``validate_registration``, and - in the same decision entry - the line ``authorizes:
s4_watch_spec sha256:<identity_hash of the watch spec>`` (review G1-02: the spec's container
scope and process groups are operator records outside the launch hash, so the decision binds
them too; the dry-run plan prints the exact line), then runs ``TrainingWatchdog`` until a
terminal decision and prints its report. It is the only command here that signals processes
or runs ``docker``/``nvidia-smi``; ``main``'s ``watch_runtime`` keyword injects the processes,
progress counter, runner and clock (tests use fakes; the default real adapters are L3).
``rehearse`` passes only when its registered decision artifact is ``passed`` as well.

``EA.json`` holds ``{release, export, policy_snapshot, hf_config_is_vlm}`` (the export chain
re-audited against the manifest). The run aggregation rule, ``k`` and ``family_alpha`` are
read from the S4 registration's ``owner.run_aggregation`` slot (the ``s4_run_registry`` v2
copy must equal it), never from flags. The launch-plan cache directory and NuRec references
have no default: omitting them blocks the plan.

Every command except ``watch --execute`` is pure or a dry run: nothing provisions, downloads,
launches, signals a process or calls a provider; ``watch --execute`` signals only the spec's
declared process groups and removes only this run's containers, under verified authority. ``bootstrap-plan`` and ``launch-plan`` take recorded-authority inputs
(action records, ``--action-authority-ref``) only to *compile* argv; they still execute
nothing. ``rehearse`` writes only its synthetic capture stream under a new ``--workdir``.

Exit codes: 0 when the artifact is ``passed`` (or ``qualify_on_host`` for checks whose last rung
is host-only, or a compiled plan); 1 when it is blocked, not estimable or otherwise not passed;
2 for unreadable or malformed input - an artifact argument must be a JSON object - and for any
exception a command raises (an error JSON names the exception type and message, never a
traceback; review B3-16); argparse usage errors also exit 2.

Verification level: **L1** (``tests/test_s4_cli.py`` drives ``main`` with temporary files and
captured stdout; ``watch --execute`` with injected fakes). The real watch runtime is **L3**.
Never: network, credential reads, owner-value defaults; no subprocess or signal outside
``watch --execute``.
"""

import argparse
import math
import os
import stat
import sys
from pathlib import Path

from gate.s2_identity import canonical_json_bytes, read_json_bytes

WATCH_ARTIFACT_KIND = "s4_watch_spec"  # decision line kind binding one exact watch spec (G1-02)

MAX_INPUT_BYTES = 64 * 1024 * 1024  # agent technical default (D-018)
DECISIONS_PATH = Path(__file__).resolve().parents[1] / "docs" / "DECISIONS.md"
OK_STATUSES = ("passed", "qualify_on_host")


class InputError(ValueError):
    """An input file is missing, not regular, too large or not strict JSON."""


def _read_bytes(path):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise InputError(f"{path}: {exc.strerror}") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise InputError(f"{path}: not a regular file")
        if info.st_size > MAX_INPUT_BYTES:
            raise InputError(f"{path}: larger than {MAX_INPUT_BYTES} bytes")
        chunks = []
        while chunk := os.read(fd, 1 << 20):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _json(path):
    try:
        return read_json_bytes(_read_bytes(path))
    except ValueError as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError(f"{path}: {exc}") from None


def _artifact(path):
    """A strict-JSON object (every artifact the commands take is a JSON object)."""
    value = _json(path)
    if type(value) is not dict:
        raise InputError(f"{path}: expected a JSON object, got {type(value).__name__}")
    return value


def _bool(text):
    if text not in ("true", "false"):
        raise argparse.ArgumentTypeError("expected true or false")
    return text == "true"


def _cmd_validate_registration(args):
    from teacher_rl.registration import validate_registration
    return validate_registration(_artifact(args.registration))


def _cmd_compile(args):
    from teacher_rl.registration import compile_overrides
    return compile_overrides(_artifact(args.registration))


def _cmd_bootstrap_plan(args):
    from teacher_rl.ops import build_s4_bootstrap_plan
    phases = [p for p in args.phases.split(",") if p]
    return build_s4_bootstrap_plan(_artifact(args.inventory), _artifact(args.action_record),
                                   _artifact(args.rental_record), phases=phases)


def _cmd_forecast(args):
    from teacher_rl.forecast import forecast
    return forecast(_artifact(args.input))


def _cmd_export_audit(args):
    from teacher_rl.export_audit import audit_export
    return audit_export(release=_artifact(args.release), export=_artifact(args.export),
                        hf_config_is_vlm=args.hf_config_is_vlm,
                        inference=_artifact(args.inference) if args.inference else None,
                        policy_snapshot=(_artifact(args.policy_snapshot) if args.policy_snapshot
                                         else None))


def _cmd_parity(args):
    from teacher_rl.parity import ladder_from_manifests
    tolerance = _artifact(args.tolerance) if args.tolerance else None
    return ladder_from_manifests(_artifact(args.reference), _artifact(args.candidate),
                                 tolerance=tolerance)


def _cmd_evaluation_request(args):
    from teacher_rl.evaluate import build_evaluation_request
    return build_evaluation_request(
        inference_manifest=_artifact(args.manifest),
        s4_registration=_artifact(args.s4_registration),
        run_registry=_artifact(args.run_registry), run_id=args.run_id,
        checkpoint_step=args.checkpoint_step, s2_registration=_artifact(args.s2_registration),
        scope=args.scope, export_audit_inputs=_artifact(args.export_audit_inputs))


def _cmd_aggregation_spec(args):
    from teacher_rl.evaluate import aggregation_spec
    return aggregation_spec(requests=[_artifact(p) for p in args.request],
                            s4_registration=_artifact(args.s4_registration),
                            run_registry=_artifact(args.run_registry), scope=args.scope)


def _cmd_launch_plan(args):
    from teacher_rl.launch import build_launch
    return build_launch(
        _artifact(args.registration), alpagym_checkout=args.alpagym_checkout,
        capture_tree=args.capture_tree, capture_manifest=args.capture_manifest,
        capture_config=args.capture_config, worker_credentials=args.worker_credentials,
        action_authority_ref=args.action_authority_ref, run_dir=args.run_dir,
        base_cosmos_toml=_read_bytes(args.base_cosmos_toml) if args.base_cosmos_toml else None,
        resolved_config_sha256=args.resolved_config_sha256,
        alpasim_cache_dir=args.alpasim_cache_dir,
        nurec_authorization_refs=None if None in (args.nurec_terms_ref, args.nurec_acquire_ref)
        else {"terms": args.nurec_terms_ref, "acquire": args.nurec_acquire_ref})


def _cmd_decide_and_aggregate(args):
    from teacher_rl.evaluate import decide_and_aggregate
    return decide_and_aggregate(_artifact(args.evaluations), _artifact(args.spec))


def _cmd_watch_spec(args):
    from teacher_rl.watchdog import build_watch_spec
    return build_watch_spec(launch_spec=_artifact(args.launch_spec), run_id=args.run_id,
                            role_records=_json(args.roles),
                            container_record=_artifact(args.containers),
                            limits=_artifact(args.limits))


def _watch_plan(spec):
    from gate.s2_identity import identity_hash
    from teacher_rl import watchdog

    return {"artifact_kind": "s4_watch_plan", "schema_version": 1, "scope": spec["scope"],
            "producer": {"module": "teacher_rl.__main__", "function": "watch"},
            "input_hashes": {"spec": identity_hash(spec)}, "status": "passed", "issues": [],
            "executes": False, "run_id": spec["run_id"], "launch_hash": spec["launch_hash"],
            "signal_order": list(watchdog.ROLE_ORDER),
            "process_groups": sorted({role["pgid"] for role in spec["roles"]}),
            "container_scope": {"compose_project": spec["compose_project"],
                                "container_ids": list(spec["container_ids"])},
            "teardown_commands": [list(watchdog.CONTAINER_LIST),
                                  ["docker", "rm", "-f", "<this run's listed IDs>"],
                                  list(watchdog.CONTAINER_LIST), list(watchdog.GPU_LIST)],
            "execute_requires": {
                "flags": ["--launch-spec", "--registration", "--decision-id", "--progress-dir"],
                "decision_lines": [
                    "authorizes: s4_launch_plan sha256:" + spec["launch_hash"],
                    "authorizes: s4_registration sha256:<identity_hash of --registration>",
                    f"authorizes: {WATCH_ARTIFACT_KIND} sha256:" + identity_hash(spec),
                    "records: acquire:nurec_scenes", "records: terms:nurec_scenes"]}}


def _real_watch_runtime(spec, progress_dir):
    from teacher_rl import watchdog
    return (watchdog.SystemProcesses(spec["roles"]), watchdog.directory_progress(progress_dir),
            watchdog.subprocess_runner, watchdog.SystemClock())


def _cmd_watch(args):
    from teacher_rl import watchdog
    from teacher_rl.capture.host_launch import verify_launch_authority

    spec = watchdog.validate_watch_spec(_artifact(args.spec))
    if not args.execute:
        return _watch_plan(spec)
    missing = [flag for flag, value in (("--launch-spec", args.launch_spec),
                                        ("--registration", args.registration),
                                        ("--decision-id", args.decision_id),
                                        ("--progress-dir", args.progress_dir)) if not value]
    if missing:
        raise InputError(f"watch --execute needs {missing} (recorded launch authority)")
    if not os.path.isabs(args.progress_dir):
        raise InputError("--progress-dir must be absolute")
    from gate.s2_authority import verify_authority
    from gate.s2_identity import identity_hash
    from teacher_rl.registration import validate_registration

    launch_spec, registration = _artifact(args.launch_spec), _artifact(args.registration)
    decisions = _read_bytes(args.decisions)
    check = verify_launch_authority(launch_spec, registration, decisions,
                                    decision_id=args.decision_id)
    codes = [issue["code"] for issue in check["issues"]]
    if validate_registration(registration)["status"] != "passed":  # review G1-09
        codes.append("registration_not_validated")
    if spec["scope"] != "actual":
        codes.append("watch_scope_not_actual")
    if spec["launch_hash"] != launch_spec.get("launch_hash"):
        codes.append("watch_spec_not_for_this_launch")
    # Review G1-02: the container scope (compose_project, container_ids) and the process groups
    # are operator records, not part of the launch hash, so the same decision must bind this
    # exact watch spec as well; otherwise a spec naming another run's project would remove it.
    watch_check = verify_authority(decisions, decision_id=args.decision_id,
                                   artifact_kind=WATCH_ARTIFACT_KIND,
                                   artifact_sha256=identity_hash(spec))
    if not watch_check["authorized"]:
        codes.append("watch_spec_not_authorized")
    if codes:
        return {"artifact_kind": "s4_watch_refusal", "schema_version": 1,
                "scope": spec["scope"], "status": "blocked", "executes": False,
                "issues": codes, "authority": check}
    runtime = args.watch_runtime or _real_watch_runtime
    processes, progress, runner, clock = runtime(spec, args.progress_dir)
    dog = watchdog.TrainingWatchdog(spec, processes=processes, progress=progress,
                                    runner=runner, clock=clock)
    polls = max(1, math.ceil((spec["deadline_epoch"] - clock.time()) / spec["poll_seconds"]) + 1)
    return dog.run(max_polls=polls)


def _cmd_rehearse(args):
    from importlib import import_module
    if not os.path.isabs(args.workdir):
        raise InputError("--workdir must be absolute")
    if os.path.lexists(args.workdir):
        raise InputError("--workdir must not exist (the rehearsal never reuses evidence)")
    try:
        module = import_module("scripts.rehearse_s4")
    except ImportError as exc:
        raise InputError(f"run from the repository root: {exc}") from None
    summary = module.run_rehearsal(args.workdir)
    passed = (summary["compile"]["status"] == "passed" and summary["bootstrap"]["compiled"]
              and summary["forecast"]["status"] == "passed"
              and summary["evaluation"]["spec_status"] == "passed"
              and summary["decision"]["status"] == "passed")
    return dict(summary, status="passed" if passed else "blocked")


def _parser():
    parser = argparse.ArgumentParser(prog="python -m teacher_rl",
                                     description="S4 harness dry-run CLI (nothing executes)")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-registration", "compile"):
        cmd = sub.add_parser(name)
        cmd.add_argument("registration")
    cmd = sub.add_parser("bootstrap-plan")
    for flag in ("--inventory", "--action-record", "--rental-record", "--phases"):
        cmd.add_argument(flag, required=True)
    cmd = sub.add_parser("forecast")
    cmd.add_argument("input")
    cmd = sub.add_parser("export-audit")
    cmd.add_argument("--release", required=True)
    cmd.add_argument("--export", required=True)
    cmd.add_argument("--hf-config-is-vlm", required=True, type=_bool)
    cmd.add_argument("--inference")
    cmd.add_argument("--policy-snapshot")
    cmd = sub.add_parser("parity")
    cmd.add_argument("--reference", required=True)
    cmd.add_argument("--candidate", required=True)
    cmd.add_argument("--tolerance")
    cmd = sub.add_parser("evaluation-request")
    for flag in ("--manifest", "--s4-registration", "--run-registry", "--run-id",
                 "--s2-registration", "--scope", "--export-audit-inputs"):
        cmd.add_argument(flag, required=True)
    cmd.add_argument("--checkpoint-step", required=True, type=int)
    cmd = sub.add_parser("aggregation-spec")
    cmd.add_argument("--request", required=True, action="append")
    for flag in ("--s4-registration", "--run-registry", "--scope"):
        cmd.add_argument(flag, required=True)
    cmd = sub.add_parser("launch-plan")
    for flag in ("--registration", "--alpagym-checkout", "--capture-tree",
                 "--capture-manifest", "--capture-config", "--worker-credentials",
                 "--action-authority-ref"):
        cmd.add_argument(flag, required=True)
    for flag in ("--run-dir", "--base-cosmos-toml", "--resolved-config-sha256",
                 "--alpasim-cache-dir", "--nurec-terms-ref", "--nurec-acquire-ref"):
        cmd.add_argument(flag)
    cmd = sub.add_parser("decide-and-aggregate")
    cmd.add_argument("--spec", required=True)
    cmd.add_argument("--evaluations", required=True)
    cmd = sub.add_parser("watch-spec")
    for flag in ("--launch-spec", "--run-id", "--roles", "--containers", "--limits"):
        cmd.add_argument(flag, required=True)
    cmd = sub.add_parser("watch")
    cmd.add_argument("--spec", required=True)
    cmd.add_argument("--execute", action="store_true")
    for flag in ("--launch-spec", "--registration", "--decision-id", "--progress-dir"):
        cmd.add_argument(flag)
    cmd.add_argument("--decisions", default=str(DECISIONS_PATH))
    cmd = sub.add_parser("rehearse")
    cmd.add_argument("--workdir", required=True)
    return parser


COMMANDS = {
    "validate-registration": _cmd_validate_registration, "compile": _cmd_compile,
    "bootstrap-plan": _cmd_bootstrap_plan, "forecast": _cmd_forecast,
    "export-audit": _cmd_export_audit, "parity": _cmd_parity,
    "evaluation-request": _cmd_evaluation_request, "aggregation-spec": _cmd_aggregation_spec,
    "launch-plan": _cmd_launch_plan, "rehearse": _cmd_rehearse,
    "decide-and-aggregate": _cmd_decide_and_aggregate, "watch-spec": _cmd_watch_spec,
    "watch": _cmd_watch,
}


def _succeeded(artifact):
    if artifact.get("artifact_kind") == "s4_bootstrap_plan":
        return artifact.get("compiled") is True
    if artifact.get("artifact_kind") == "s4_watch_spec":
        return True  # build_watch_spec raised on any gap, so a returned spec is valid
    return artifact.get("status") in OK_STATUSES


def main(argv=None, *, stdout=None, watch_runtime=None):
    """Run one command; returns the exit code (0 passed, 1 not passed, 2 input error).

    ``watch_runtime(spec, progress_dir) -> (processes, progress, runner, clock)`` replaces the
    real watch adapters (tests only).
    """
    stdout = stdout or sys.stdout
    args = _parser().parse_args(argv)
    args.watch_runtime = watch_runtime
    try:
        artifact = COMMANDS[args.command](args)
        # Serialized inside the try: an argument that is not strict UTF-8 (a surrogate-escaped
        # byte in argv) reaches the artifact and fails here, not as a traceback.
        text = canonical_json_bytes(artifact).decode() + "\n"
    except Exception as exc:  # noqa: BLE001 - every failure is an error JSON, never a traceback
        message = f"{type(exc).__name__}: {exc}"
        error = {"artifact_kind": "s4_cli_error", "schema_version": 1,
                 "command": args.command, "status": "blocked",
                 "error": message.encode("utf-8", "backslashreplace").decode("utf-8")}
        stdout.write(canonical_json_bytes(error).decode() + "\n")
        return 2
    stdout.write(text)
    return 0 if _succeeded(artifact) else 1


if __name__ == "__main__":
    sys.exit(main())
