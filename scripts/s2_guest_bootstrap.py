#!/usr/bin/env python3
"""Execute a compiled S2 or S4 bootstrap plan on the guest, one verified stage at a time.

Tickets: CPU-09 (staged bootstrap), runbook section 4 steps 4-6, R2 phases A-E; build-spec B3
item (1) and A2X (S4 two-GPU plans). Agent-authored under D-018/D-024; serves S2.1/S2.2/S2.7
and S4.1-S4.4 [C] (G2, G7, G9). Plans come from the compiler registered for their kind in
``gate.s2_bootstrap.PLAN_KINDS`` (``s2_runtime``: ``gate.s2_bootstrap.build_execution_plan``;
``s4_runtime``: ``teacher_rl.ops.build_s4_bootstrap_plan``); this runner adds no command of its
own and is the only bootstrap runner.

Usage (on the admitted guest, after recorded authority)::

    python3 scripts/s2_guest_bootstrap.py PLAN.json --action-record ACTION.json \
        --inventory INVENTORY.json [--rental-record RENTAL.json] --journal-dir /abs/new/journal \
        [--secret-file hf_token=/abs/file] [--secret-file ngc_api_key=/abs/file] \
        [--resume] [--execute --decision-id D-NNN [--decisions docs/DECISIONS.md]]

``--rental-record`` is required for an ``s4_runtime`` plan (its compiler takes the rental action
record) and refused for an ``s2_runtime`` plan.

Without ``--execute`` it runs every check below, lists the stages and runs nothing; it also
prints ``authority_request``: the exact ``authorizes:`` and ``records:`` lines a decision entry
needs before this plan may execute (with ``--decision-id`` the dry run checks them too).

Hash-bound authority (G1; completeness-audit Safety: bootstrap refs were free text never
checked against the decision log). Executing a plan (``--execute``; and any ``run_plan`` that
reaches the real ``subprocess_runner``, or is given ``authority``) requires, over the bytes of
``docs/DECISIONS.md`` (``--decisions`` overrides the path for tests), that
``gate.s2_authority.verify_authority`` authorizes ``--decision-id`` for ``artifact_kind`` =
the plan kind's registered artifact kind (``s2_bootstrap_plan`` or ``s4_bootstrap_plan``) and
``artifact_sha256`` = ``gate.s2_identity.identity_hash`` of the whole plan as supplied (which
seals ``plan_sha256``, argv, requires and authorizations), and that
``gate.s2_authority.verify_records`` finds a ``records: <key>`` line in that same entry for
every ``terms:``/``acquire:`` key any stage ``requires``. The plan's free-text authorization
values no longer suffice on their own (``plan_not_authorized``, ``records_missing``,
``authority_required``, ``decisions_unreadable``). Level L1 (pure check over supplied bytes);
it never authenticates who wrote the decision.

Behaviour:
- Authority is re-derived, never trusted from the plan. The runner refuses before touching
  anything unless: the plan passes ``validate_plan`` (registered kind and compiler identity,
  hash seal, schema, credential hygiene, rule confinement, every ``requires`` key present); its
  ``producer.module`` is the registered compiler's module (else ``compiler_mismatch``); the
  inventory's ``artifact_kind`` is the kind's registered inventory kind
  (``inventory_kind_mismatch``); exactly the kind's inputs are given (``input_missing``,
  ``unexpected_input``); the action record's, inventory's and (S4) rental record's hashes are the
  ones the plan was compiled from; the plan's phases are all in the record's
  ``authorized_phases`` and every stage's phase is in the plan's phases; every non-``terms:``
  authorization equals the record's value; and recompiling through the kind's registered
  compiler here reproduces the plan's ``plan_sha256`` exactly. ``plan_sha256`` is an
  unkeyed hash anyone can recompute, so only the recompilation binds argv, requires and
  authorizations to the record (``plan_not_reproducible`` names both producer hashes, so a
  different compiler source version is told apart from an edited plan).
- The real ``subprocess_runner`` (no injected runner) additionally requires plan scope ``actual``
  and no authorization or action-record reference containing ``SYNTHETIC``: fixture authority is
  never treated as recorded authority (contract section 2.1/2.6).
- Secrets: a required secret file that is missing, not a private regular file (mode & 077 == 0,
  no symlink) or shorter than 8 bytes is refused, as is any secret value appearing in the plan.
- Environment: only the stage's ``env_allowlist`` names copied from the caller, the plan's
  ``env_set`` values, and ``secret_env`` values read from the declared secret files. Nothing
  else (parent tokens, provider keys) reaches a stage. ``stdin_secret`` pipes a secret file's
  bytes to stdin (``docker login --password-stdin``).
- Every secret value is redacted from captured stdout/stderr before it is written or verified;
  journal records carry no secret values.
- Journal: ``<journal>/journal.jsonl`` canonical-JSON lines, each fsynced; ``run_started`` carries
  the artifact header (kind, schema, scope, producer, input hashes, units). Stage output in
  ``<journal>/stages/<id>.<attempt>.{stdout,stderr}``; archived files in ``<journal>/archive``.
  An existing journal is never overwritten; ``--resume`` appends to a journal of the same plan.
- Stops at the first failure (non-zero exit, timeout, runner error, a required rule failing, a
  receipt conflict or any unexpected exception, reported as ``internal_error``). Cleanup commands
  of stages that were started (``docker logout``) and ``run_finished`` run at the end either way;
  ``run_plan`` never raises.
- Attempt-scoped stages (the host probe) get ``ATTEMPT_TOKEN`` replaced by the attempt number in
  argv and rule paths, so a retried probe writes a fresh directory; the substituted argv is
  journalled and the completed attempt drives idempotency on resume.
- Resume (CPU-09: "resumes only when verified existing artifacts match expected hashes;
  otherwise block instead of overwriting evidence"): every ``file`` rule whose path has a size and
  sha256 receipt in the journal (from any stage's completion) is re-hashed first. Present but
  different bytes block the run (``receipt_mismatch``) and keep blocking on later resumes; a
  missing artifact has nothing to overwrite, so its stage may run again. A completed stage is then
  skipped only when its ``idempotency_check`` rules re-verify now; stages without such rules run
  again.

Verification ceiling: run logic is **L1** with a fake runner (``tests/test_s2_guest_bootstrap.py``);
``subprocess_runner`` is L1 against the local Python interpreter only; behaviour of the real
guest tools (apt, docker, uv, hf, nvidia-smi) is **L3** - a completed run is a host record, not
``HOST_QUALIFIED``. The runner never chooses an owner value, never creates or destroys a rental,
never accepts terms and never runs an unlisted command. Agent technical defaults (D-018): cleanup
commands get ``CLEANUP_TIMEOUT_S`` = 120 s; secrets shorter than 8 bytes are refused so that
redaction cannot mangle unrelated output. ``run_plan``'s ``compiler`` keyword exists so runner
mechanics can be tested on hand-built stages; ``main`` never exposes it, and the registered
compiler is always resolved from ``PLAN_KINDS``, never from a name the plan carries.
"""

import argparse
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gate.s2_authority import verify_authority, verify_records
from gate.s2_bootstrap import (
    PLAN_KINDS,
    RunResult,
    evaluate_rule,
    plan_kind,
    resolve_compiler,
    substitute_attempt,
    validate_plan,
)
from gate.s2_identity import canonical_json_bytes, identity_hash, read_json_bytes
from gate.s2_runtime import SCOPES

DECISIONS_PATH = ROOT / "docs" / "DECISIONS.md"
GATED_PREFIXES = ("terms:", "acquire:")

CLEANUP_TIMEOUT_S = 120
MIN_SECRET_BYTES = 8
MAX_SECRET_BYTES = 64 * 1024
KILL_GRACE_S = 10
SCHEMA_VERSION = 1
UNITS = {"timeout_s": "seconds", "size_bytes": "bytes"}
RECEIPT_KINDS = ("file", "archive_file")


def _issue(code, path, message):
    return {"code": code, "path": path, "severity": "error", "evidence_refs": [],
            "message": message}


def _producer():
    try:
        source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        source_sha256 = None
    return {"module": "scripts.s2_guest_bootstrap", "source_sha256": source_sha256}


def _hash_or_none(value):
    try:
        return identity_hash(value)
    except (TypeError, ValueError, UnicodeError):
        return None


def _header(plan, action_record, inventory, rental_record=None):
    plan = plan if type(plan) is dict else {}
    scope = plan.get("scope") if type(plan.get("scope")) is str else None
    return {
        "artifact_kind": "s2_bootstrap_run_summary", "schema_version": SCHEMA_VERSION,
        "scope": scope if scope in SCOPES else None, "producer": _producer(),
        "inputs": {"plan_sha256": plan.get("plan_sha256") if type(plan.get("plan_sha256")) is str
                   else None,
                   "action_record_sha256": _hash_or_none(action_record),
                   "inventory_sha256": _hash_or_none(inventory),
                   "rental_record_sha256": (None if rental_record is None
                                            else _hash_or_none(rental_record))},
        "units": dict(UNITS),
    }


def subprocess_runner(argv, *, cwd, env, stdin, timeout_s):
    """Run argv without a shell in its own session; on timeout TERM then KILL the group (L3)."""
    process = subprocess.Popen(
        list(argv), cwd=cwd, env=dict(env), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True, close_fds=True)
    try:
        stdout, stderr = process.communicate(input=stdin or b"", timeout=timeout_s)
        return RunResult(process.returncode, stdout, stderr, False)
    except subprocess.TimeoutExpired:
        stdout = stderr = b""
        for sig, grace in ((signal.SIGTERM, KILL_GRACE_S), (signal.SIGKILL, None)):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass  # group already gone; collect what it wrote
            try:
                stdout, stderr = process.communicate(timeout=grace)
                break
            except subprocess.TimeoutExpired:
                continue
        return RunResult(None, stdout or b"", stderr or b"", True)


def _read_secret(path):
    """Read a private regular secret file without following symlinks."""
    if type(path) is not str or not os.path.isabs(path):
        return None, "secret_file_missing"
    try:
        info = os.lstat(path)
    except OSError:
        return None, "secret_file_missing"
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
        return None, "secret_file_permissions"
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        with os.fdopen(descriptor, "rb") as stream:
            data = stream.read(MAX_SECRET_BYTES + 1)
    except OSError:
        return None, "secret_file_missing"
    value = data.strip()
    if len(data) > MAX_SECRET_BYTES or len(value) < MIN_SECRET_BYTES or b"\x00" in value:
        return None, "secret_invalid"
    try:
        return value.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, "secret_invalid"


def _plan_strings(value):
    if type(value) is str:
        yield value
    elif type(value) is dict:
        for item in value.values():
            yield from _plan_strings(item)
    elif type(value) is list:
        for item in value:
            yield from _plan_strings(item)


class _Journal:
    def __init__(self, directory, plan_sha256, clock):
        self.directory = directory
        self.path = os.path.join(directory, "journal.jsonl")
        self.plan_sha256 = plan_sha256
        self.clock = clock
        self.seq = 0
        self.attempts = {}

    def load(self):
        """Return ({completed stage_id: attempt}, {path: receipt}) from an existing journal."""
        completed, receipts = {}, {}
        with open(self.path, "rb") as stream:
            for line in stream:
                record = read_json_bytes(line.rstrip(b"\n"))
                if record.get("plan_sha256") != self.plan_sha256:
                    raise ValueError("journal_plan_mismatch")
                self.seq = max(self.seq, record.get("seq", 0))
                stage_id = record.get("stage_id")
                event = record.get("event")
                if event == "stage_started":
                    self.attempts[stage_id] = self.attempts.get(stage_id, 0) + 1
                elif event in ("stage_completed", "stage_skipped"):
                    completed[stage_id] = record.get("attempt")
                    _record_receipts(receipts, record.get("rules"), record.get("seq"))
                elif event == "stage_failed":
                    completed.pop(stage_id, None)
        return completed, receipts

    def append(self, event, **fields):
        self.seq += 1
        record = {"seq": self.seq, "event": event, "plan_sha256": self.plan_sha256,
                  "wall_time_utc": self.clock(), **fields}
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW,
                             0o600)
        try:
            os.write(descriptor, canonical_json_bytes(record) + b"\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def write_output(self, stage_id, attempt, suffix, data):
        target = os.path.join(self.directory, "stages", f"{stage_id}.{attempt}.{suffix}")
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())


def _record_receipts(receipts, rules, seq):
    """Latest size/sha256 receipt per path from a journal record's rule summaries."""
    for rule in rules if type(rules) is list else []:
        recorded = rule.get("recorded") if type(rule) is dict else None
        if (type(recorded) is dict and rule.get("kind") in RECEIPT_KINDS and rule.get("ok")
                and type(rule.get("path")) is str and type(recorded.get("sha256")) is str
                and type(recorded.get("size_bytes")) is int):
            receipts[rule["path"]] = {"size_bytes": recorded["size_bytes"],
                                      "sha256": recorded["sha256"], "seq": seq}


def _receipt_conflicts(stage, receipts, roots):
    """Paths of this stage's file rules whose present bytes differ from their journal receipt."""
    conflicts, seen = [], set()
    for rule in list(stage["expected_outputs"]) + list(stage["idempotency_check"] or []):
        path = rule.get("path")
        if rule.get("kind") != "file" or path not in receipts or path in seen:
            continue
        seen.add(path)
        receipt = receipts[path]
        result = evaluate_rule({"kind": "file", "root": rule.get("root"), "path": path,
                                "size_bytes": receipt["size_bytes"], "sha256": receipt["sha256"]},
                               roots=roots)
        if not result["ok"] and result["reason"] != "missing":
            conflicts.append({"path": path, "reason": result["reason"],
                              "receipt_seq": receipt["seq"]})
    return conflicts


def _redact(data, secrets):
    for name, value in sorted(secrets.items()):
        data = data.replace(value.encode("utf-8"), f"[REDACTED:{name}]".encode())
    return data


def _stage_env(stage, environ, secrets=None):
    env = {name: environ[name] for name in stage["env_allowlist"] if name in environ}
    env.update(stage["env_set"])
    for variable, name in (stage["secret_env"].items() if secrets is not None else ()):
        env[variable] = secrets[name]
    return env


def _summarize(rules, results):
    return [dict({key: result[key] for key in ("kind", "ok", "required", "reason", "recorded")},
                 path=rule.get("path") if type(rule.get("path")) is str else None)
            for rule, result in zip(rules, results)]


def _refused(header, issues, journal_dir=None):
    return dict(header, status="refused", failed_stage=None, reason=issues[0]["code"],
                issues=issues, stages=[], journal=journal_dir)


def _authority_issues(plan, record):
    """The plan never claims more than the action record grants (review A2-F1)."""
    if (type(record) is not dict or type(record.get("authorized_phases")) is not list
            or type(record.get("authorizations")) is not dict):
        return [_issue("invalid_action_record", "action_record", "Malformed action record.")]
    phases = plan["phases"]
    if type(phases) is not list or not all(type(p) is str for p in phases):
        return [_issue("invalid_plan", "phases", "phases is a list of phase letters.")]
    issues = []
    unauthorized = [p for p in phases if p not in record["authorized_phases"]]
    if unauthorized:
        issues.append(_issue("phase_not_authorized", "phases",
                             f"Phases {unauthorized} are not in the record's authorized_phases."))
    for index, stage in enumerate(plan["stages"]):
        if stage["phase"] not in phases:
            issues.append(_issue("stage_phase_not_in_plan", f"stages[{index}].phase",
                                 "Every stage belongs to one of the plan's phases."))
    for key, value in sorted(plan["authorizations"].items()):
        if not key.startswith("terms:") and record["authorizations"].get(key) != value:
            issues.append(_issue("authorization_not_recorded", f"authorizations.{key}",
                                 "Plan authority differs from the action record's."))
    return issues


def _reproduce(plan, kind, inputs, compiler):
    """Recompile through the kind's registered compiler (or the injected test seam)."""
    function = PLAN_KINDS[kind]["function"]
    try:
        compile_plan = compiler if compiler is not None else resolve_compiler(kind)
        compiled = compile_plan(*[inputs[name] for name in PLAN_KINDS[kind]["inputs"]],
                                phases=list(plan["phases"]))
        ok = (type(compiled) is dict and compiled.get("compiled") is True
              and compiled.get("plan_sha256") == plan["plan_sha256"])
    except Exception as exc:  # noqa: BLE001 - a compiler failure is a refusal, never a crash
        compiled, ok = {"error": type(exc).__name__}, False
    if ok:
        return []
    compiled = compiled if type(compiled) is dict else {}
    theirs = (compiled.get("producer") or {}).get("source_sha256") if type(
        compiled.get("producer")) is dict else None
    return [_issue("plan_not_reproducible", "plan_sha256",
                   f"Recompiling through {function} does not reproduce this "
                   f"plan (recompiled {compiled.get('plan_sha256')}, plan "
                   f"{plan['plan_sha256']}; compiler source {theirs}, plan producer "
                   f"{plan['producer'].get('source_sha256')}). A differing producer hash means "
                   "another compiler source version; otherwise the plan was edited.")]


def _real_runner_issues(plan, record):
    """Fixture authority never reaches the real subprocess runner (review A2-F2)."""
    if plan["scope"] != "actual":
        return [_issue("fixture_scope_refused", "scope",
                       f"The real runner executes only scope 'actual' plans, not "
                       f"{plan['scope']!r}.")]
    refs = [("action_record.action_record_ref", record.get("action_record_ref"))]
    refs += [(f"authorizations.{k}", v) for k, v in sorted(plan["authorizations"].items())]
    synthetic = [where for where, value in refs
                 if type(value) is not str or "synthetic" in value.casefold()]
    if synthetic:
        return [_issue("synthetic_authority_refused", synthetic[0],
                       "Synthetic/fixture references are not recorded authority.")]
    return []


def authority_request(plan, kind):
    """The decision lines this plan needs before it may execute (L1; pure)."""
    return {"artifact_kind": PLAN_KINDS[kind]["plan_artifact_kind"],
            "sha256": identity_hash(plan),
            "records": sorted({key for stage in plan["stages"] for key in stage["requires"]
                               if key.startswith(GATED_PREFIXES)})}


def _decision_issues(plan, kind, authority):
    """Hash-bound recorded authority for this exact plan and its gated stages (G1, L1)."""
    if type(authority) is not dict or set(authority) != {"decisions_md", "decision_id"}:
        return [_issue("authority_required", "authority",
                       "Executing a plan needs --decision-id and the decision log bytes.")]
    if authority["decisions_md"] is None:
        return [_issue("decisions_unreadable", "decisions", "The decision log is unreadable.")]
    wanted = authority_request(plan, kind)
    checked = verify_authority(authority["decisions_md"], decision_id=authority["decision_id"],
                               artifact_kind=wanted["artifact_kind"],
                               artifact_sha256=wanted["sha256"])
    if not checked["authorized"]:
        codes = sorted({issue["code"] for issue in checked["issues"]})
        return [_issue("plan_not_authorized", "authority",
                       f"{authority['decision_id']!r} does not record 'authorizes: "
                       f"{wanted['artifact_kind']} sha256:{wanted['sha256']}' as live "
                       f"authority ({', '.join(codes)}).")]
    if wanted["records"]:
        records = verify_records(authority["decisions_md"],
                                 decision_id=authority["decision_id"],
                                 refs=tuple(wanted["records"]))
        if not records["satisfied"]:
            missing = records["missing_refs"] or wanted["records"]
            return [_issue("records_missing", "authority",
                           f"{authority['decision_id']!r} lacks 'records:' lines for "
                           f"{missing} (terms/acquisition of gated stages).")]
    return []


def _input_issues(plan, kind, inputs, compiler):
    """The plan's kind, producer and inputs agree with its registered compiler (A2X)."""
    spec = PLAN_KINDS[kind]
    module = spec["function"].rpartition(".")[0]
    producer = plan["producer"] if type(plan["producer"]) is dict else {}
    if compiler is None and producer.get("module") != module:
        return [_issue("compiler_mismatch", "producer.module",
                       f"A {kind} plan is produced by {module}.")]
    inventory = inputs["inventory"]
    if type(inventory) is not dict or inventory.get("artifact_kind") != spec[
            "inventory_artifact_kind"]:
        return [_issue("inventory_kind_mismatch", "inventory.artifact_kind",
                       f"A {kind} plan is compiled from a {spec['inventory_artifact_kind']}.")]
    for name in ("rental_record",):
        if name in spec["inputs"] and inputs[name] is None:
            return [_issue("input_missing", name, f"A {kind} plan is recompiled from its {name}.")]
        if name not in spec["inputs"] and inputs[name] is not None:
            return [_issue("unexpected_input", name, f"A {kind} plan takes no {name}.")]
    for name, code in (("action_record", "action_record_mismatch"),
                       ("inventory", "inventory_mismatch"),
                       ("rental_record", "rental_record_mismatch")):
        if name in spec["inputs"] and _hash_or_none(inputs[name]) != plan["inputs"].get(
                f"{name}_sha256"):
            return [_issue(code, name, f"The {name.replace('_', ' ')} is not the one this plan "
                                       "was compiled from.")]
    return []


def _verify(plan, record, inventory, runner, compiler=None, rental_record=None,
            authority=None, require_authority=False, dry_run=False):
    """All refusal checks that need no secret; empty means the plan may run."""
    issues = validate_plan(plan)
    if issues:
        return issues
    kind, _ = plan_kind(plan)
    inputs = {"inventory": inventory, "action_record": record, "rental_record": rental_record}
    issues = (_input_issues(plan, kind, inputs, compiler) or _authority_issues(plan, record)
              or _reproduce(plan, kind, inputs, compiler))
    if issues:
        return issues
    real = runner is subprocess_runner
    if real:
        issues = _real_runner_issues(plan, record)
        if issues:
            return issues
    if (real and not dry_run) or require_authority or authority is not None:
        return _decision_issues(plan, kind, authority)
    return []


def _load_secrets(plan, secret_files):
    secrets = {}
    for name in plan["secrets_required"]:
        value, problem = _read_secret((secret_files or {}).get(name))
        if problem:
            return None, [_issue(problem, f"secret_files.{name}",
                                 "Provide a private regular file holding the secret.")]
        secrets[name] = value
    for text in _plan_strings(plan):
        if any(value in text for value in secrets.values()):
            return None, [_issue("secret_in_plan", "plan",
                                 "A secret value appears in the plan; refusing to run.")]
    return secrets, []


class _Run:
    """Mutable state of one execution; every journal write may fail without escaping."""

    def __init__(self, plan, journal, summary, runner, secrets, environ, completed, receipts,
                 resume):
        self.plan, self.journal, self.summary = plan, journal, summary
        self.runner, self.secrets, self.environ = runner, secrets, environ
        self.completed, self.receipts, self.resume = completed, receipts, resume
        self.cleanups = []
        self.current = None
        self.archive_dir = os.path.join(journal.directory, "archive")

    def safe_append(self, event, **fields):
        try:
            self.journal.append(event, **fields)
        except Exception as exc:  # noqa: BLE001 - recorded in the summary, never raised
            self.summary["issues"].append(_issue(
                "journal_write_failed", "journal",
                f"{type(exc).__name__} writing '{event}'; the journal is incomplete."))
            if self.summary["status"] == "completed":
                self.summary.update(status="failed", reason="journal_write_failed")

    def fail(self, stage_id, reason, **fields):
        self.summary.update(status="failed", failed_stage=stage_id, reason=reason)
        self.summary["stages"].append({"id": stage_id, "outcome": "failed", "reason": reason})
        self.journal.append("stage_failed", stage_id=stage_id, reason=reason, **fields)

    def stages(self):
        plan, roots = self.plan, self.plan["roots"]
        for stage in plan["stages"]:
            stage_id = stage["id"]
            self.current = stage_id
            missing = [key for key in stage["requires"] if not plan["authorizations"].get(key)]
            if missing:
                self.fail(stage_id, "authorization_missing", missing=missing)
                return
            probe_env = _stage_env(stage, self.environ)
            next_attempt = self.journal.attempts.get(stage_id, 0) + 1
            done = self.resume and stage_id in self.completed
            view = substitute_attempt(stage, self.completed[stage_id]) if done else (
                substitute_attempt(stage, next_attempt) if stage["argv"] is not None else stage)
            if self.resume:
                conflicts = _receipt_conflicts(view, self.receipts, roots)
                if conflicts:
                    self.fail(stage_id, "receipt_mismatch", conflicts=conflicts)
                    return
            if done and stage["idempotency_check"] is not None:
                rules = view["idempotency_check"]
                results = [evaluate_rule(rule, roots=roots, runner=self.runner, env=probe_env)
                           for rule in rules]
                if all(result["ok"] for result in results):
                    summary = _summarize(rules, results)
                    self.journal.append("stage_skipped", stage_id=stage_id,
                                        attempt=self.completed[stage_id], rules=summary)
                    _record_receipts(self.receipts, summary, self.journal.seq)
                    self.summary["stages"].append({"id": stage_id, "outcome": "skipped"})
                    continue
            if not self.run_stage(stage, probe_env):
                return

    def run_stage(self, stage, probe_env):
        stage_id, roots = stage["id"], self.plan["roots"]
        stdout, attempt, view = b"", None, stage
        if stage["argv"] is not None:
            attempt = self.journal.attempts.get(stage_id, 0) + 1
            self.journal.attempts[stage_id] = attempt
            view = substitute_attempt(stage, attempt)
            if stage["cleanup_argv"]:
                self.cleanups.append(stage)
            self.journal.append("stage_started", stage_id=stage_id, attempt=attempt,
                                argv=view["argv"], cwd=stage["cwd"], level=stage["level"])
            secrets = self.secrets
            stdin = secrets[stage["stdin_secret"]].encode() if stage["stdin_secret"] else None
            try:
                result = self.runner(list(view["argv"]), cwd=stage["cwd"],
                                     env=_stage_env(stage, self.environ, secrets), stdin=stdin,
                                     timeout_s=stage["timeout_s"])
            except Exception as exc:  # noqa: BLE001 - any runner failure stops the run
                self.fail(stage_id, "runner_error", error=type(exc).__name__)
                return False
            stdout = _redact(result.stdout or b"", secrets)
            self.journal.write_output(stage_id, attempt, "stdout", stdout)
            self.journal.write_output(stage_id, attempt, "stderr",
                                      _redact(result.stderr or b"", secrets))
            if result.timed_out:
                self.fail(stage_id, "timeout", timeout_s=stage["timeout_s"])
                return False
            if result.returncode != 0:
                self.fail(stage_id, "exit_status", returncode=result.returncode)
                return False
        rules = view["expected_outputs"]
        results = [evaluate_rule(rule, roots=roots, stdout=stdout, runner=self.runner,
                                 env=probe_env, archive_dir=self.archive_dir) for rule in rules]
        summary = _summarize(rules, results)
        if not all(result["ok"] for result in results):
            self.fail(stage_id, "verification_failed", rules=summary)
            return False
        self.journal.append("stage_completed", stage_id=stage_id, attempt=attempt, rules=summary,
                            first_delivery=bool(stage["records_first_delivery"]))
        _record_receipts(self.receipts, summary, self.journal.seq)
        self.summary["stages"].append({"id": stage_id, "outcome": "completed"})
        return True

    def cleanup(self):
        for stage in reversed(self.cleanups):
            try:
                result = self.runner(list(stage["cleanup_argv"]), cwd=stage["cwd"],
                                     env=_stage_env(stage, self.environ), stdin=None,
                                     timeout_s=CLEANUP_TIMEOUT_S)
                ok = not result.timed_out and result.returncode == 0
                returncode = result.returncode
            except Exception:  # noqa: BLE001 - recorded, never raised
                ok, returncode = False, None
            self.safe_append("cleanup", stage_id=stage["id"], argv=stage["cleanup_argv"], ok=ok,
                             returncode=returncode)
            if not ok:
                self.summary["issues"].append(_issue(
                    "cleanup_failed", f"stages.{stage['id']}",
                    "Cleanup command failed; inspect the guest."))
                self.summary["status"] = "failed"
                self.summary["reason"] = self.summary["reason"] or "cleanup_failed"


def run_plan(plan, *, action_record, inventory, runner, journal_dir, secret_files, environ,
             clock, resume=False, rental_record=None, compiler=None, authority=None,
             require_authority=False):
    """Verify, then execute a plan with the given runner; returns a summary (never raises).

    ``rental_record`` is the S4 compiler's third input (``s4_runtime`` plans only).
    ``compiler`` overrides the registered recompilation reference only for tests of runner
    mechanics, which inject one that returns their hand-built plan; ``main`` never passes it,
    so the plan's kind always selects its compiler from ``PLAN_KINDS``. ``authority`` is
    ``{"decisions_md": bytes | None, "decision_id": str}``; it is checked whenever given, and
    required for the real ``subprocess_runner`` or when ``require_authority`` (``main
    --execute``) is set. Injected runners without either keep the runner-mechanics behaviour.
    """
    header = _header(plan, action_record, inventory, rental_record)
    try:
        issues = _verify(plan, action_record, inventory, runner, compiler, rental_record,
                         authority, require_authority)
    except Exception as exc:  # noqa: BLE001 - malformed input is a refusal, never a crash
        issues = [_issue("invalid_input", "plan", f"{type(exc).__name__} while verifying.")]
    if issues:
        return _refused(header, issues)
    secrets, issues = _load_secrets(plan, secret_files)
    if issues:
        return _refused(header, issues)
    if type(journal_dir) is not str or not os.path.isabs(journal_dir):
        return _refused(header, [_issue("invalid_journal_dir", "journal_dir", "Absolute path.")])
    journal = _Journal(journal_dir, plan["plan_sha256"], clock)
    completed, receipts = {}, {}
    if os.path.lexists(journal.path):
        if not resume:
            return _refused(header, [_issue("journal_exists", "journal_dir",
                                            "Never overwrite a journal; pass --resume to "
                                            "continue.")], journal_dir)
        try:
            completed, receipts = journal.load()
        except (OSError, ValueError) as exc:
            code = str(exc) if str(exc) == "journal_plan_mismatch" else "journal_unreadable"
            return _refused(header, [_issue(code, "journal_dir", "Journal belongs to another "
                                            "plan or is unreadable.")], journal_dir)
    summary = dict(header, status="completed", failed_stage=None, reason=None, issues=[],
                   stages=[], journal=journal_dir)
    state = _Run(plan, journal, summary, runner, secrets, environ, completed, receipts, resume)
    try:
        for sub in ("", "stages", "archive"):
            os.makedirs(os.path.join(journal_dir, sub), mode=0o700, exist_ok=True)
        journal.append("run_started", **{key: header[key] for key in (
            "schema_version", "scope", "producer", "inputs", "units")},
                       artifact_kind="s2_bootstrap_journal", resume=bool(resume),
                       stage_count=len(plan["stages"]), secrets_provided=sorted(secrets))
        state.stages()
    except Exception as exc:  # noqa: BLE001 - becomes a structured failure (review A2-F4)
        stage_id = state.current
        summary.update(status="failed", failed_stage=stage_id, reason="internal_error")
        summary["issues"].append(_issue(
            "internal_error", f"stages.{stage_id}" if stage_id else "run",
            f"{type(exc).__name__} during the run; inspect the guest and the journal."))
        last = summary["stages"][-1] if summary["stages"] else {}
        if stage_id is not None and (last.get("id"), last.get("outcome")) != (stage_id, "failed"):
            summary["stages"].append({"id": stage_id, "outcome": "failed",
                                      "reason": "internal_error"})
        state.safe_append("stage_failed", stage_id=stage_id, reason="internal_error",
                          error=type(exc).__name__)
    state.cleanup()
    state.safe_append("run_finished", status=summary["status"],
                      failed_stage=summary["failed_stage"], reason=summary["reason"])
    return summary


def _secret_args(values):
    files = {}
    for value in values or []:
        name, sep, path = value.partition("=")
        if not sep or not name or name in files:
            raise ValueError("--secret-file takes NAME=/absolute/path once per name")
        files[name] = path
    return files


def main(argv=None, *, runner=None, environ=None, clock=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("plan")
    parser.add_argument("--action-record", required=True)
    parser.add_argument("--inventory", required=True,
                        help="The runtime inventory the plan was compiled from (recompiled here).")
    parser.add_argument("--rental-record",
                        help="The rental action record an s4_runtime plan was compiled from.")
    parser.add_argument("--journal-dir", required=True)
    parser.add_argument("--secret-file", action="append")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true",
                        help="Run the stages; without it the plan is only verified.")
    parser.add_argument("--decision-id",
                        help="Decision entry that authorizes this plan's hash (required with "
                             "--execute).")
    parser.add_argument("--decisions", default=str(DECISIONS_PATH),
                        help="Decision log bytes to verify against (default docs/DECISIONS.md).")
    args = parser.parse_args(argv)
    try:
        plan = read_json_bytes(Path(args.plan).read_bytes())
        record = read_json_bytes(Path(args.action_record).read_bytes())
        inventory = read_json_bytes(Path(args.inventory).read_bytes())
        rental = (None if args.rental_record is None
                  else read_json_bytes(Path(args.rental_record).read_bytes()))
        secret_files = _secret_args(args.secret_file)
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "refused", "reason": "invalid_input",
                          "message": type(exc).__name__}))
        return 2
    effective_runner = runner if runner is not None else subprocess_runner
    authority = None
    if args.decision_id is not None:
        try:
            decisions_md = Path(args.decisions).read_bytes()
        except OSError:
            decisions_md = None
        authority = {"decisions_md": decisions_md, "decision_id": args.decision_id}
    if not args.execute:
        header = _header(plan, record, inventory, rental)
        request = None
        try:
            # The dry run never needs authority; a given --decision-id is checked as well.
            issues = _verify(plan, record, inventory, effective_runner, rental_record=rental,
                             authority=authority, dry_run=True)
            if not issues:
                request = authority_request(plan, plan_kind(plan)[0])
        except Exception as exc:  # noqa: BLE001 - malformed input is a refusal
            issues = [_issue("invalid_input", "plan", f"{type(exc).__name__} while verifying.")]
        status = "refused" if issues else "validated"
        stages = [] if issues else [stage["id"] for stage in plan["stages"]]
        print(json.dumps(dict(header, status=status, issues=issues, stages=stages,
                              authority_request=request), indent=2))
        return 2 if issues else 0
    if clock is None:
        from datetime import datetime, timezone  # Python 3.10 has no datetime.UTC

        def clock():
            # Phase A runs on the guest's system Python 3.10 (no datetime.UTC there).
            return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: UP017

    result = run_plan(plan, action_record=record, inventory=inventory, runner=effective_runner,
                      journal_dir=args.journal_dir, secret_files=secret_files,
                      environ=dict(os.environ if environ is None else environ), clock=clock,
                      resume=args.resume, rental_record=rental, authority=authority,
                      require_authority=True)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())
