"""External process-group watchdog for one S4 training run (E18).

Tickets: E18 (cost or deadline boundary: export evidence, terminate task resources, verify
absence, no automatic extension), lane-r3 section 1 finding 14 (a non-controller Cosmos worker
crash does not stop the run: ``launch_all`` kills everything only when the controller fails),
lane-r3 section 3 teardown (host ``finally`` sends SIGTERM then SIGKILL after 10 s; manual
crash cleanup removes ``wizard`` containers), build-spec B3 item (3). Serves S4.1-S4.4 [C]
(G9, G7) under D-024. Agent-authored evaluator engineering (D-018).

Watch spec (``s4_watch_spec`` v2, every field required, no defaults): ``run_id``;
``launch_hash`` (the ``teacher_rl.launch`` spec's hash: the watch belongs to that authorized
launch); ``compose_project`` (the Compose project label of this run's Wizard containers, or
``null``) and ``container_ids`` (full 64-hex IDs recorded at launch; at least one of the two
scopes is required); ``roles`` -
``{name, role, pid, pgid}`` with role in ``host | wizard | controller | policy | rollout``
(exactly one ``host``, one ``controller``, at least one ``policy`` and one ``rollout``;
roles may share a process group, which is then signalled once);
``stall_seconds`` (no progress for this long -> hung); ``poll_seconds``; ``deadline_epoch``
(absolute, persisted in the spec, never renewed); ``provider_cutoff_epoch`` (the A3 lifecycle
cutoff); ``export_reserve_seconds`` (``deadline_epoch + export_reserve_seconds <=
provider_cutoff_epoch`` so evidence export fits before the provider stop);
``term_grace_seconds`` (SIGTERM -> SIGKILL; 10 s mirrors the upstream host teardown).
The export reserve must also cover the watchdog's own worst-case teardown (review B3-05):
``export_reserve_seconds > 0`` and ``poll_seconds + 2 x term_grace_seconds +
TEARDOWN_COMMANDS x COMMAND_TIMEOUT_S <= export_reserve_seconds`` (four commands: container
list, removal, re-list, GPU list); ``run`` clamps every sleep to the deadline, so the deadline
teardown starts at ``deadline_epoch`` and never runs into the provider cutoff.

Decision (``assess``, pure): ``terminate`` when the deadline is reached, any non-host role is
dead while the host lives (``role_dead:<name>``), the host exited (``host_exited``, with its
exit code when known), progress stalled for ``stall_seconds`` (an unreadable progress counter
is not progress) or the progress counter went backwards; ``completed`` only when the host and
every role have exited and the host exit code is known to be 0; otherwise ``continue``.

Teardown (``TrainingWatchdog.terminate``): SIGTERM to every live process group in role order
(host, controller, policy, rollout, wizard), wait ``term_grace_seconds`` on the injected clock,
SIGKILL survivors, wait again; then ``CONTAINER_LIST`` (``docker ps -a --no-trunc --format
'{{.ID}} {{.Label "com.docker.compose.project"}}'``), parsed strictly (any line that is not a
64-hex ID plus an optional label fails the check closed), selecting only containers whose ID
was recorded at launch (``container_ids``) or whose Compose project label equals
``compose_project`` - never a name substring (G1; completeness-audit Safety: v1 force-removed
every container matching ``name=wizard``, which is any container of any run). Selected
containers are removed by exact ID with ``docker rm -f`` and the list is re-read; then
``nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader``. Cleanup is
``verified_clean`` only when every group is gone and both listings succeeded with none of this
run's containers and no GPU process left; a failed listing is ``unverified`` (never clean) and a
remaining run container/GPU process is ``dirty``. Containers of other runs are never touched.
GPU processes are reported, never killed by pattern (that could hit a process that is not this
run's). A v1 spec (no container scope) is refused.

Producer (``build_watch_spec``, L1): composes a v2 spec from the launch spec (``launch_hash``
and ``scope``; it must be ``passed``), the role records captured on the host (``{name, role,
pid, pgid}`` each; capturing them from the live process table is L3), the container record of
the launch (``{compose_project, container_ids}``) and the timing limits (all six required).

Operator entry point: ``python -m teacher_rl watch --spec SPEC.json`` (``teacher_rl.__main__``)
prints the teardown plan and executes nothing; ``--execute`` needs the same hash-bound
authority as the launch (``teacher_rl.capture.host_launch.verify_launch_authority``) for the
spec's ``launch_hash`` and, in the same decision entry, ``authorizes: s4_watch_spec
sha256:<identity_hash(spec)>`` (review G1-02: ``compose_project``/``container_ids`` and the
roles are operator records the launch hash does not cover), then runs ``TrainingWatchdog`` with ``SystemProcesses``,
``subprocess_runner``, ``SystemClock`` and ``directory_progress`` (all L3 on a host).

Hand-off to the A3 provider lifecycle (``gate.s2_execution``): ``handoff_observation`` maps the
report onto an observation kind the coordinator accepts in ``running`` - ``work_complete`` for
a completed run with a verified-clean teardown, ``work_failed`` for every other end (A3X;
``reason``: the first ``s4_watchdog:<reason>`` of a termination, else
``s4_cleanup:<status>``; ``decision_issue_codes``: every reason and the cleanup code) - so the
lifecycle exports evidence and then stops/destroys the instance, recording failed work as its
own cause, never as budget exhaustion; at ``provider_cutoff_epoch`` the lifecycle's own
``next_action`` stops regardless of export. ``hand_off`` journals the observation and steps a
coordinator (``observe`` + ``step``); nothing here calls a provider directly. In
``bootstrapping`` (which does not accept ``work_complete``) it journals ``work_failed`` with
reason ``s4_watchdog:lifecycle_state:bootstrapping`` so a finished run is still exported and
stopped (review B3-06); before ``bootstrapping`` it raises, and a step result carrying
``unexpected_observation`` raises ``RuntimeError`` rather than leaving the instance billed.

Verification level per function:

- ``validate_watch_spec``, ``build_watch_spec``, ``parse_container_listing``, ``assess``,
  ``handoff_observation``: **L1** (literal known answers and negative controls in
  ``tests/test_s4_watchdog.py``).
- ``directory_progress``: **L1** over a temporary directory; that the capture stream grows
  while training progresses is **L3**. ``SystemClock`` is **L3** (never used by tests).
- ``TrainingWatchdog`` (poll/terminate/run) and ``hand_off``: **L1** over injected fake
  processes, clock, runner and coordinator; real signals, docker and nvidia-smi behaviour on
  a host are **L3** (``qualify_on_host``).
- ``parse_proc_stat`` and the identity logic of ``SystemProcesses`` (review B3-09): **L1**
  over an injected stat reader and ``killpg``. At construction the adapter binds every
  declared group to its leader (``pid == pgid``) start time from ``/proc/<pid>/stat`` and
  checks each role's ``pgrp``; ``signal_group`` refuses an unbound group, a group whose leader
  start time or group changed (``refused_identity_changed``: the id was reused) or an
  unreadable one, and never signals its own group. A leaderless group stays signalable while
  no process owns its id (Linux does not reuse an id still used as a process-group id). The
  default ``/proc`` reader, real ``os.killpg`` and ``subprocess_runner`` are **L3** and never
  used by tests.

Report ``status``: ``passed`` means the teardown was verified clean (it says nothing about
whether the run succeeded - ``decision``/``reasons`` do), ``blocked`` a dirty or unverified
cleanup, ``not_run`` a watch that ended without a terminal decision.

Agent technical defaults (D-018): role signal order, the 1 s wait step while a grace period
runs, and the progress counter of the operator entry point (total bytes of regular files under
the run's capture directory, which only grows while roles emit events). Never: extends a
deadline, retries a run, kills a process outside the declared groups, removes a container that
is not this run's, calls a provider, reads a credential or decides what a failed run means.
"""

import hashlib
import math
import os
import re
import signal
import stat
import subprocess
from pathlib import Path

from gate.s2_bootstrap import RunResult
from gate.s2_execution import ALLOWED_OBSERVATIONS, validate_observation
from gate.s2_identity import identity_hash

SPEC_KIND = "s4_watch_spec"
REPORT_KIND = "s4_watchdog_report"
SCHEMA_VERSION = 2
SCOPES = ("synthetic", "public_source_fixture", "actual")
ROLE_ORDER = ("host", "controller", "policy", "rollout", "wizard")
COMPOSE_PROJECT_LABEL = "com.docker.compose.project"
CONTAINER_LIST = ("docker", "ps", "-a", "--no-trunc", "--format",
                  '{{.ID}} {{.Label "' + COMPOSE_PROJECT_LABEL + '"}}')
GPU_LIST = ("nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader")
COMMAND_TIMEOUT_S = 60  # agent technical default (D-018)
TEARDOWN_COMMANDS = 4  # docker ps, docker rm, docker ps, nvidia-smi (worst case)
LIMIT_KEYS = ("stall_seconds", "poll_seconds", "deadline_epoch", "provider_cutoff_epoch",
              "export_reserve_seconds", "term_grace_seconds")
_SPEC_KEYS = {"artifact_kind", "schema_version", "scope", "run_id", "launch_hash",
              "compose_project", "container_ids", "roles", *LIMIT_KEYS}
_ROLE_KEYS = {"name", "role", "pid", "pgid"}
_CONTAINER_KEYS = {"compose_project", "container_ids"}
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
# Compose project names: lowercase letters, digits, dashes and underscores, starting with a
# letter or digit (docker compose documentation).
_PROJECT = re.compile(r"[a-z0-9][a-z0-9_-]{0,127}")
_LISTING_LINE = re.compile(r"([0-9a-f]{64}) ?(.*)")


class SpecError(ValueError):
    """A watch spec that cannot be enforced; nothing is watched."""


def _issue(code, path, message, severity="blocking"):
    return {"code": code, "path": path, "severity": severity, "evidence_refs": [],
            "message": message}


def _producer():
    return {"module": "teacher_rl.watchdog",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise SpecError(f"{name} must be a positive int")


def _epoch(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise SpecError(f"{name} must be a finite positive epoch")


def _container_scope(project, ids):
    if project is not None and (type(project) is not str or not _PROJECT.fullmatch(project)):
        raise SpecError("compose_project is null or a Compose project name")
    if type(ids) is not list or any(type(i) is not str or not _HEX64.fullmatch(i) for i in ids):
        raise SpecError("container_ids are full 64-hex container IDs recorded at launch")
    if len(set(ids)) != len(ids):
        raise SpecError("container_ids are unique")
    if project is None and not ids:
        raise SpecError("record this run's containers (compose_project or container_ids); "
                        "the watchdog never removes containers by name pattern")


def validate_watch_spec(spec):
    """Return the spec unchanged or raise ``SpecError`` (L1; every field required)."""
    if type(spec) is dict and spec.get("schema_version") == 1:
        raise SpecError("s4_watch_spec v1 removed containers by name pattern; v2 records the "
                        "run's Compose project or container IDs and its launch_hash")
    if type(spec) is not dict or set(spec) != _SPEC_KEYS:
        raise SpecError(f"spec needs exactly {sorted(_SPEC_KEYS)}")
    if spec["artifact_kind"] != SPEC_KIND or spec["schema_version"] != SCHEMA_VERSION or type(
            spec["schema_version"]) is not int:
        raise SpecError(f"only {SPEC_KIND} v{SCHEMA_VERSION} is known")
    if spec["scope"] not in SCOPES:
        raise SpecError(f"scope is one of {SCOPES}")
    if type(spec["run_id"]) is not str or not _NAME.fullmatch(spec["run_id"]):
        raise SpecError("run_id is a slug")
    if type(spec["launch_hash"]) is not str or not _HEX64.fullmatch(spec["launch_hash"]):
        raise SpecError("launch_hash is the 64-hex hash of the authorized launch spec")
    _container_scope(spec["compose_project"], spec["container_ids"])
    for name in ("stall_seconds", "poll_seconds", "term_grace_seconds"):
        _positive_int(spec[name], name)
    if type(spec["export_reserve_seconds"]) is not int or spec["export_reserve_seconds"] <= 0:
        raise SpecError("export_reserve_seconds must be a positive int")
    teardown = (spec["poll_seconds"] + 2 * spec["term_grace_seconds"]
                + TEARDOWN_COMMANDS * COMMAND_TIMEOUT_S)
    if teardown > spec["export_reserve_seconds"]:
        raise SpecError(f"poll + 2 x grace + {TEARDOWN_COMMANDS} x {COMMAND_TIMEOUT_S} s "
                        f"teardown ({teardown} s) must fit the export reserve")
    _epoch(spec["deadline_epoch"], "deadline_epoch")
    _epoch(spec["provider_cutoff_epoch"], "provider_cutoff_epoch")
    if spec["deadline_epoch"] + spec["export_reserve_seconds"] > spec["provider_cutoff_epoch"]:
        raise SpecError("deadline + export reserve must not pass the provider cutoff")
    roles = spec["roles"]
    if type(roles) is not list or not roles:
        raise SpecError("roles must be a nonempty list")
    names = set()
    for role in roles:
        if type(role) is not dict or set(role) != _ROLE_KEYS:
            raise SpecError(f"each role needs exactly {sorted(_ROLE_KEYS)}")
        if type(role["name"]) is not str or not _NAME.fullmatch(role["name"]) or (
                role["name"] in names):
            raise SpecError("role names are unique slugs")
        if role["role"] not in ROLE_ORDER:
            raise SpecError(f"role kind is one of {ROLE_ORDER}")
        for key in ("pid", "pgid"):
            if type(role[key]) is not int or role[key] <= 1:
                raise SpecError(f"{key} must be an int > 1 (never 0, 1 or a negative group)")
        names.add(role["name"])
    kinds = [r["role"] for r in roles]
    if kinds.count("host") != 1 or kinds.count("controller") != 1 or (
            "policy" not in kinds or "rollout" not in kinds):
        raise SpecError("one host, one controller, and at least one policy and rollout role")
    return spec


def build_watch_spec(*, launch_spec, run_id, role_records, container_record, limits):
    """Compose a v2 watch spec from the launch spec, captured roles and launch records (L1).

    ``role_records``: ``[{name, role, pid, pgid}, ...]`` captured on the host;
    ``container_record``: ``{compose_project, container_ids}`` recorded at launch; ``limits``:
    exactly ``LIMIT_KEYS``. Nothing is defaulted; raises ``SpecError`` on any gap.
    """
    if type(launch_spec) is not dict or launch_spec.get("artifact_kind") != "s4_launch_spec" or (
            launch_spec.get("status") != "passed"):
        raise SpecError("watch only a passed s4_launch_spec (it carries the launch_hash)")
    if type(container_record) is not dict or set(container_record) != _CONTAINER_KEYS:
        raise SpecError(f"container record needs exactly {sorted(_CONTAINER_KEYS)}")
    if type(limits) is not dict or set(limits) != set(LIMIT_KEYS):
        raise SpecError(f"limits need exactly {list(LIMIT_KEYS)} (no defaults)")
    if type(role_records) is not list:
        raise SpecError("role records are a list")
    spec = {"artifact_kind": SPEC_KIND, "schema_version": SCHEMA_VERSION,
            "scope": launch_spec.get("scope"), "run_id": run_id,
            "launch_hash": launch_spec.get("launch_hash"),
            "compose_project": container_record["compose_project"],
            "container_ids": container_record["container_ids"],
            "roles": [dict(role) if type(role) is dict else role for role in role_records],
            **{key: limits[key] for key in LIMIT_KEYS}}
    return validate_watch_spec(spec)


def parse_container_listing(text, spec):
    """IDs of this run's containers in a ``CONTAINER_LIST`` output; ``None`` if unparseable (L1).

    A container is this run's when its full ID was recorded at launch or its Compose project
    label equals the spec's ``compose_project``; every other container is ignored.
    """
    selected = []
    for line in text.splitlines():
        line = line.rstrip("\r")
        if not line.strip():
            continue
        match = _LISTING_LINE.fullmatch(line)
        if match is None:
            return None
        container, project = match.group(1), match.group(2).strip()
        if container in spec["container_ids"] or (
                spec["compose_project"] is not None and project == spec["compose_project"]):
            selected.append(container)
    return selected


def directory_progress(path):
    """A progress counter: total bytes of regular files under ``path`` (no links followed).

    Returns a callable giving an int, or ``None`` when the directory cannot be read (an
    unreadable counter is not progress). L1 over a temporary directory.
    """
    root = Path(path)

    def progress():
        total = 0
        try:
            for dirpath, _dirnames, filenames in os.walk(root, followlinks=False,
                                                         onerror=_raise):
                for name in filenames:
                    info = os.lstat(Path(dirpath) / name)
                    if stat.S_ISREG(info.st_mode):
                        total += info.st_size
        except OSError:
            return None
        return total if root.is_dir() and not root.is_symlink() else None

    return progress


def _raise(error):
    raise error


class SystemClock:
    """Wall clock with a real sleep (L3; never used by tests)."""

    @staticmethod
    def time():
        import time

        return time.time()

    @staticmethod
    def sleep(seconds):
        import time

        time.sleep(seconds)


def initial_state(spec, now):
    validate_watch_spec(spec)
    return {"started_epoch": now, "last_progress": None, "last_progress_epoch": now,
            "polls": 0}


def assess(spec, state, observation, now):
    """Pure decision over one observation ``{alive: {name: bool}, host_exit_code, progress}``.

    Returns ``{"decision": continue|terminate|completed, "reasons": [...], "state": new}``.
    """
    state = dict(state, polls=state["polls"] + 1)
    reasons = []
    if now >= spec["deadline_epoch"]:
        reasons.append("training_deadline_reached")
    alive = observation["alive"]
    host = next(r for r in spec["roles"] if r["role"] == "host")
    host_alive = alive.get(host["name"]) is True
    others = [r for r in spec["roles"] if r["role"] != "host"]
    if host_alive:
        reasons.extend(f"role_dead:{r['name']}" for r in others if alive.get(r["name"]) is not
                       True)
    else:
        code = observation["host_exit_code"]
        everything_gone = not any(alive.get(r["name"]) is True for r in others)
        if code == 0 and everything_gone and not reasons:
            return {"decision": "completed", "reasons": ["completed"], "state": state}
        reasons.append("host_exited" if code is None else f"host_exited:{code}")
    progress = observation["progress"]
    if type(progress) is int and not isinstance(progress, bool) and progress >= 0:
        last = state["last_progress"]
        if last is not None and progress < last:
            reasons.append("progress_regressed")
        elif last is None or progress > last:
            state.update(last_progress=progress, last_progress_epoch=now)
    if now - state["last_progress_epoch"] >= spec["stall_seconds"]:
        reasons.append("progress_stalled")
    return {"decision": "terminate" if reasons else "continue", "reasons": reasons,
            "state": state}


def handoff_observation(report):
    """The A3 lifecycle observation for a terminal watchdog report (validated; L1)."""
    if report["decision"] == "completed" and report["cleanup_status"] == "verified_clean":
        observation = {"kind": "work_complete"}
    else:
        codes = [f"s4_watchdog:{reason}" for reason in report["reasons"]]
        codes.append(f"s4_cleanup:{report['cleanup_status']}")
        # A3X: the cause is the first termination reason, else the unclean teardown.
        cause = codes[0] if report["decision"] == "terminate" else codes[-1]
        observation = {"kind": "work_failed", "reason": cause, "decision_issue_codes": codes}
    return validate_observation(observation)


_BEFORE_WORK = ("admitted", "create_intent", "create_requested", "uncertain_create", "bound",
                "arm_failed", "armed")


def hand_off(coordinator, report):
    """Journal the hand-off observation on a lifecycle coordinator and step it (L1 with fakes).

    In ``running`` the observation moves the lifecycle to ``exporting``. In ``bootstrapping``
    a kind the state does not accept is journaled as ``work_failed`` naming the state.
    Before ``bootstrapping`` no training run can be handed off (``RuntimeError``); after
    ``running`` the coordinator only steps (its ``next_action`` enforces the cutoff by
    itself). A step that reports ``unexpected_observation`` raises ``RuntimeError``.
    """
    observation = report["handoff_observation"]
    name = coordinator.state["state"]
    if name in _BEFORE_WORK:
        raise RuntimeError(f"no training run can end while the lifecycle is in {name!r}")
    if name in ("bootstrapping", "running"):
        if observation["kind"] not in ALLOWED_OBSERVATIONS[name]:
            codes = [f"s4_watchdog:{reason}" for reason in report["reasons"]]
            codes.append(f"s4_watchdog:lifecycle_state:{name}")
            codes.append(f"s4_cleanup:{report['cleanup_status']}")
            observation = validate_observation({
                "kind": "work_failed", "reason": f"s4_watchdog:lifecycle_state:{name}",
                "decision_issue_codes": codes})
        coordinator.observe(observation)
    decision = coordinator.step()
    rejected = [i for i in decision.get("issues", ()) if i.get("code") == (
        "unexpected_observation")]
    if rejected:
        raise RuntimeError(f"the lifecycle rejected the hand-off: unexpected_observation in "
                           f"state {name!r}")
    return decision


class TrainingWatchdog:
    """Poll a training run through injected interfaces and tear it down on a terminal reason.

    ``processes``: ``alive(pid) -> bool``, ``exit_code(pid) -> int | None``,
    ``group_alive(pgid) -> bool``, ``signal_group(pgid, sig) -> str``; ``progress()`` returns a
    monotonic int counter or ``None``; ``runner(argv, timeout_s) -> RunResult``; ``clock`` has
    ``time()`` and ``sleep(seconds)``.
    """

    def __init__(self, spec, *, processes, progress, runner, clock):
        self.spec = validate_watch_spec(spec)
        self.processes, self.progress, self.runner, self.clock = (processes, progress, runner,
                                                                  clock)
        self.state = initial_state(spec, clock.time())
        self.events = []

    def _observe(self):
        alive = {}
        for role in self.spec["roles"]:
            try:
                alive[role["name"]] = self.processes.alive(role["pid"]) is True
            except Exception:  # noqa: BLE001 - an unreadable process is not alive evidence
                alive[role["name"]] = False
                self.events.append({"event": "process_check_failed", "role": role["name"]})
        host = next(r for r in self.spec["roles"] if r["role"] == "host")
        try:
            code = self.processes.exit_code(host["pid"])
        except Exception:  # noqa: BLE001
            code = None
        code = code if type(code) is int else None
        try:
            progress = self.progress()
        except Exception:  # noqa: BLE001 - unreadable progress is not progress
            progress = None
        return {"alive": alive, "host_exit_code": code, "progress": progress}

    def poll(self):
        result = assess(self.spec, self.state, self._observe(), self.clock.time())
        self.state = result["state"]
        return result

    def run(self, *, max_polls):
        """Poll until a terminal decision (or ``max_polls``), then tear down and report."""
        if type(max_polls) is not int or max_polls < 1:
            raise ValueError("max_polls must be a positive int")
        result = None
        for index in range(max_polls):
            result = self.poll()
            if result["decision"] != "continue":
                break
            if index + 1 < max_polls:
                remaining = self.spec["deadline_epoch"] - self.clock.time()
                self.clock.sleep(max(0, min(self.spec["poll_seconds"], remaining)))
        if result["decision"] == "continue":
            return self._report("continue", [], None)
        return self.terminate(result["decision"], result["reasons"])

    def _live_groups(self):
        groups = []
        for kind in ROLE_ORDER:
            for role in self.spec["roles"]:
                if role["role"] == kind and role["pgid"] not in groups:
                    groups.append(role["pgid"])
        live = []
        for pgid in groups:
            try:
                if self.processes.group_alive(pgid):
                    live.append(pgid)
            except Exception:  # noqa: BLE001 - unknown state: keep it in the kill set
                live.append(pgid)
        return live

    def _signal_all(self, sig):
        for pgid in self._live_groups():
            try:
                outcome = self.processes.signal_group(pgid, sig)
            except Exception as exc:  # noqa: BLE001
                outcome = f"error:{type(exc).__name__}"
            self.events.append({"event": "signal", "pgid": pgid, "signal": int(sig),
                                "outcome": str(outcome)})

    def _wait(self):
        end = self.clock.time() + self.spec["term_grace_seconds"]
        while self._live_groups() and self.clock.time() < end:
            self.clock.sleep(min(1, end - self.clock.time()))
        return self._live_groups()

    def _run(self, argv):
        try:
            result = self.runner(list(argv), timeout_s=COMMAND_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001
            return None, f"runner_error:{type(exc).__name__}"
        if not isinstance(result, RunResult) or result.timed_out or result.returncode != 0:
            return None, "command_failed"
        try:
            return result.stdout.decode("utf-8"), None
        except (AttributeError, UnicodeDecodeError):
            return None, "unreadable_output"

    def _listed(self, issues):
        text, error = self._run(CONTAINER_LIST)
        if error:
            issues.append(_issue("container_check_failed", "docker", error))
            return None
        ids = parse_container_listing(text, self.spec)
        if ids is None:
            issues.append(_issue("container_check_failed", "docker", "unexpected listing line"))
        return ids

    def _containers(self, issues):
        ids = self._listed(issues)
        if ids is None:
            return None
        if ids:
            self.events.append({"event": "remove_containers", "ids": ids})
            _, error = self._run(("docker", "rm", "-f", *ids))
            if error:
                issues.append(_issue("container_remove_failed", "docker", error))
            ids = self._listed(issues)
            if ids is None:
                return None
        if ids:
            issues.append(_issue("run_containers_remain", "docker", f"{ids}"))
        return ids

    def _gpu_processes(self, issues):
        text, error = self._run(GPU_LIST)
        if error:
            issues.append(_issue("gpu_check_failed", "nvidia-smi", error))
            return None
        rows = [line.strip() for line in text.splitlines() if line.strip()]
        if rows:
            issues.append(_issue("gpu_processes_remain", "nvidia-smi", f"{rows}"))
        return rows

    def terminate(self, decision, reasons):
        """SIGTERM, grace, SIGKILL, grace; then verify containers and GPU processes."""
        issues = []
        self._signal_all(signal.SIGTERM)
        if self._wait():
            self._signal_all(signal.SIGKILL)
        survivors = self._wait()
        if survivors:
            issues.append(_issue("process_group_survived", "processes", f"{survivors}"))
        containers = self._containers(issues)
        gpu = self._gpu_processes(issues)
        if containers is None or gpu is None:
            cleanup = "unverified"
        elif survivors or containers or gpu:
            cleanup = "dirty"
        else:
            cleanup = "verified_clean"
        report = self._report(decision, reasons, cleanup, issues=issues, survivors=survivors,
                              containers=containers, gpu=gpu)
        report["handoff_observation"] = handoff_observation(report)
        return report

    def _report(self, decision, reasons, cleanup, *, issues=(), survivors=None,
                containers=None, gpu=None):
        body = {"run_id": self.spec["run_id"], "decision": decision, "reasons": list(reasons),
                "cleanup_status": cleanup, "survivor_groups": survivors,
                "remaining_containers": containers, "remaining_gpu_processes": gpu,
                "events": list(self.events), "polls": self.state["polls"]}
        return {"artifact_kind": REPORT_KIND, "schema_version": SCHEMA_VERSION,
                "scope": self.spec["scope"], "producer": _producer(),
                "input_hashes": {"spec": identity_hash(self.spec)},
                "units": {"time": "seconds since the epoch on the injected clock"},
                "status": "passed" if cleanup == "verified_clean" else (
                    "not_run" if cleanup is None else "blocked"),
                "issues": sorted(issues, key=lambda i: (i["path"], i["code"])),
                "level": "L1", "execution_status": "qualify_on_host", **body,
                "handoff_observation": None, "report_hash": identity_hash(body)}


def parse_proc_stat(data):
    """``pgrp`` (field 5) and ``starttime`` (field 22) from ``/proc/<pid>/stat`` bytes (L1)."""
    try:
        fields = data[data.rindex(b")") + 1:].split()
        return {"pgrp": int(fields[2]), "starttime": int(fields[19])}
    except (ValueError, IndexError, AttributeError, TypeError):
        raise ValueError("unparseable /proc/<pid>/stat") from None


def _read_proc_stat(pid):
    """Linux ``/proc`` reader (L3): ``None`` when no such process exists."""
    try:
        with open(f"/proc/{int(pid)}/stat", "rb") as handle:
            return parse_proc_stat(handle.read())
    except (FileNotFoundError, ProcessLookupError):
        return None


class SystemProcesses:
    """Real process adapter (POSIX; the default reader and signals are L3, never in tests).

    A reaped child's exit code is cached: ``os.kill(pid, 0)`` still succeeds on an unreaped
    zombie, and after ``waitpid`` reaps it a second ``waitpid`` raises ``ChildProcessError``,
    so without the cache a cleanly finished host would read as ``host_exited`` (code unknown).

    ``roles`` are the watch spec's roles; each declared group is bound to its leader's start
    time at construction (review B3-09), and ``signal_group`` re-verifies it before any signal.
    """

    def __init__(self, roles, *, read_stat=_read_proc_stat, killpg=os.killpg,
                 getpgrp=os.getpgrp):
        self._reaped = {}
        self._read_stat, self._killpg, self._getpgrp = read_stat, killpg, getpgrp
        self._groups = {}
        for role in roles:
            self._groups.setdefault(role["pgid"], self._leader_token(role["pgid"]))
        for role in roles:
            try:
                stat = self._read_stat(role["pid"])
            except Exception:  # noqa: BLE001 - unreadable identity cannot be bound
                stat = {"pgrp": None}
            if stat is not None and stat["pgrp"] != role["pgid"]:
                self._groups[role["pgid"]] = "unverifiable"

    def _leader_token(self, pgid):
        try:
            leader = self._read_stat(pgid)
        except Exception:  # noqa: BLE001
            return "unverifiable"
        if leader is None:
            return None  # leaderless: members keep the id from being reused
        if leader["pgrp"] != pgid:
            return "unverifiable"
        return leader["starttime"]

    def _reap(self, pid):
        if pid in self._reaped:
            return self._reaped[pid]
        try:
            done, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return None  # not our child: the exit code is unknown, never assumed 0
        if not done:
            return None
        self._reaped[pid] = os.waitstatus_to_exitcode(status)
        return self._reaped[pid]

    def alive(self, pid):
        if self._reap(pid) is not None:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def exit_code(self, pid):
        return self._reap(pid)

    def group_alive(self, pgid):
        try:
            self._killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def signal_group(self, pgid, sig):
        if pgid == self._getpgrp():
            return "refused_own_group"  # never signal the watchdog's own process group
        if pgid not in self._groups:
            return "refused_unbound_group"
        bound = self._groups[pgid]
        if bound == "unverifiable":
            return "refused_identity_changed"
        current = self._leader_token(pgid)
        if current == "unverifiable" or (current is not None and current != bound):
            return "refused_identity_changed"
        try:
            self._killpg(pgid, sig)
        except ProcessLookupError:
            return "no_such_group"
        return "sent"


def subprocess_runner(argv, timeout_s):
    """Real command runner with a minimal environment (L3, never used by tests)."""
    env = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL") if k in os.environ}
    try:
        done = subprocess.run(list(argv), capture_output=True, timeout=timeout_s, env=env,
                              check=False)
    except subprocess.TimeoutExpired:
        return RunResult(None, b"", b"", True)
    return RunResult(done.returncode, done.stdout, done.stderr, False)
