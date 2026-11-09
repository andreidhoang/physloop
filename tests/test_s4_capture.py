"""Known answers for the H4 capture adapter, its sink and the host launcher patch (A8; L1/L2).

Upstream is never imported. Where the pinned code is stdlib-only, the exact pinned AST nodes are
executed from the vendored, hash-checked sources in ``tests/fixtures/s4_upstream/`` (the real
``StreamingRolloutWorker`` retry/collect logic, ``AlpagymRollout.rollout_generation``,
``filter_trainable_rollouts``, ``_build_cosmos_launcher_command``/``_build_cosmos_command`` and
``ArtifactPaths``/``build_artifact_paths``);
other upstream modules are ``sys.modules`` fakes whose method signatures are AST-checked against
the same pinned files. No subprocess, network, provider, GPU or real tool is used.

The F5 section (parameter snapshot schema 2, scheduler/trainer settings, pinned-shaped trainer
feeding ``teacher_rl.records.build_trainer_step_records``) replaces ``torch`` with a
``sys.modules`` fake whose tensors expose hand-packed little-endian bytes; every expected hash is
``hashlib`` over those literal bytes, and the E07/E20 consumers (``teacher_rl.checks.steps``,
``teacher_rl.adapters``) are run on the produced inputs.
"""

import ast
import builtins
import contextlib
import dataclasses
import hashlib
import inspect
import itertools
import json
import logging
import os
import pathlib
import queue
import struct
import threading
import types
import typing
import uuid
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

from gate.s2_identity import canonical_json_bytes
from gate.tensor_manifest import DTYPE_ITEMSIZE, build_manifest
from teacher_rl import adapters, records
from teacher_rl.capture import entrypoint, host_launch, sink
from teacher_rl.capture.entrypoint import (
    CHAINED_TARGETS,
    SENTINEL,
    WRAP_TARGETS,
    Capture,
    CaptureFatal,
    CaptureInstallError,
)
from teacher_rl.capture.sink import EventSink, SinkError, SinkLimitError, read_stream
from teacher_rl.checks import steps
from teacher_rl.records import build_attempt_ledger, check_trace_integrity, streams_from_reports

FIXTURES = Path(__file__).parent / "fixtures" / "s4_upstream"
PROVENANCE = json.loads((FIXTURES / "provenance.json").read_text())
REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE = "clipgt-a309e228-26e1-423e-a44c-cb00aa7378cb"
LIMITS = {"max_event_bytes": 65536, "max_stream_bytes": 4_000_000, "max_blob_bytes": 100_000,
          "max_total_blob_bytes": 1_000_000}
CONFIG = {"schema": "physloop.s4.capture.config.v1", "limits": LIMITS, "max_inline_items": 8,
          "max_message_chars": 200, "max_completion_file_bytes": 1_000_000,
          "max_checkpoint_file_bytes": 1_000_000,
          "parameter_snapshots": "init_and_each_step", "parameter_hashing": "metadata_only",
          "capture_simulator_bytes": True}
_G_RT = "alpagym/packages/runtime/src/alpagym_runtime"


def pinned_source(fixture):
    for repo in PROVENANCE["repositories"].values():
        for entry in repo["files"]:
            if entry["fixture"] == f"{fixture}.txt":
                data = (FIXTURES / entry["fixture"]).read_bytes()
                assert hashlib.sha256(data).hexdigest() == entry["sha256"], fixture
                return data
    raise AssertionError(f"{fixture} is not vendored")


def pinned_tree(fixture):
    return ast.parse(pinned_source(fixture), filename=fixture)


def find_def(tree, qualname):
    parts = qualname.split(".")
    scope = tree.body
    for part in parts[:-1]:
        scope = next(n for n in scope if isinstance(n, ast.ClassDef) and n.name == part).body
    return next(n for n in scope if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == parts[-1])


def ast_params(node):
    a = node.args
    names = [p.arg for p in a.posonlyargs + a.args]
    if a.vararg:
        names.append(a.vararg.arg)
    names += [p.arg for p in a.kwonlyargs]
    if a.kwarg:
        names.append(a.kwarg.arg)
    return tuple(names)


def ast_signature(node):
    """(name, kind, has_default) per parameter, from the pinned FunctionDef."""
    a = node.args
    positional = a.posonlyargs + a.args
    defaults = [False] * (len(positional) - len(a.defaults)) + [True] * len(a.defaults)
    out = [(p.arg, "POSITIONAL_ONLY" if p in a.posonlyargs else "POSITIONAL_OR_KEYWORD", d)
           for p, d in zip(positional, defaults, strict=True)]
    if a.vararg:
        out.append((a.vararg.arg, "VAR_POSITIONAL", False))
    out += [(p.arg, "KEYWORD_ONLY", d is not None)
            for p, d in zip(a.kwonlyargs, a.kw_defaults, strict=True)]
    if a.kwarg:
        out.append((a.kwarg.arg, "VAR_KEYWORD", False))
    return out


def runtime_signature(func):
    return [(p.name, p.kind.name, p.default is not inspect.Parameter.empty)
            for p in inspect.signature(func).parameters.values()]


SAFE_BUILTINS = {name: getattr(builtins, name) for name in (
    "__build_class__", "len", "list", "dict", "set", "str", "int", "float", "bool", "range",
    "sorted", "sum", "next", "isinstance", "super", "object", "property", "staticmethod",
    "ValueError", "RuntimeError", "KeyError", "FileNotFoundError", "Exception",
    "BaseException", "tuple", "type", "enumerate", "zip", "min", "max", "getattr")}


def exec_pinned(fixture, names, namespace, module_name):
    """Execute only the named top-level nodes (classes, functions, assignments) of pinned code."""
    tree = pinned_tree(fixture)
    picked = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names or isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in names for t in node.targets):
            picked.append(node)
    assert len(picked) == len(names), (fixture, names)
    module = types.ModuleType(module_name)
    module.__dict__.update(namespace)
    module.__dict__["__builtins__"] = SAFE_BUILTINS
    module.__dict__["__name__"] = module_name
    code = compile(ast.Module(body=picked, type_ignores=[]), fixture, "exec")
    exec(code, module.__dict__)  # noqa: S102 - hash-checked pinned public source only
    return module


def exec_pinned_method(fixture, qualname, namespace):
    node = find_def(pinned_tree(fixture), qualname)
    module = types.ModuleType("pinned_method")
    module.__dict__.update(namespace)
    module.__dict__["__builtins__"] = SAFE_BUILTINS
    code = compile(ast.Module(body=[node], type_ignores=[]), fixture, "exec")
    exec(code, module.__dict__)  # noqa: S102 - hash-checked pinned public source only
    return module.__dict__[node.name]


# ------------------------------------------------------------------- pinned-source agreement


@pytest.mark.parametrize("target", [t for t in WRAP_TARGETS + CHAINED_TARGETS
                                    if t.kind != "class_binding"], ids=lambda t: t.target_id)
def test_wrap_target_parameters_equal_pinned_source(target):
    node = find_def(pinned_tree(target.fixture), target.defined_in)
    assert ast_params(node) == target.params


def test_class_and_function_bindings_are_module_global_lookups_in_pinned_source():
    backend = pinned_tree(f"{_G_RT}/cosmos/rollout_backend.py")
    imports = [n for n in backend.body if isinstance(n, ast.ImportFrom)
               and n.module == "alpasim_grpc.v0.runtime_pb2_grpc"]
    assert [a.name for n in imports for a in n.names] == ["RuntimeServiceStub"]
    init_engine = find_def(backend, "AlpagymRollout.init_engine")
    calls = [n for n in ast.walk(init_engine) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "RuntimeServiceStub"]
    assert len(calls) == 1
    trainer = pinned_tree(f"{_G_RT}/cosmos/trainer.py")
    assert any(isinstance(n, ast.ImportFrom) and n.module == "alpagym_runtime.cosmos.rollout_filter"
               and [a.name for a in n.names] == ["filter_trainable_rollouts"] for n in trainer.body)
    step = find_def(trainer, "AlpagymGRPOTrainer.step_training")
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "filter_trainable_rollouts" for n in ast.walk(step))
    bundle = pinned_tree("alpagym/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/bundle.py")
    setup = find_def(bundle, "setup_tokenizer")
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "install_alpamayo_r1_runtime_bridge" for n in ast.walk(setup))
    worker = pinned_tree(f"{_G_RT}/episode_runner/streaming_worker.py")
    run = find_def(worker, "StreamingRolloutWorker._run_rollout")
    sims = [n for n in ast.walk(run) if isinstance(n, ast.Attribute) and n.attr == "simulate"]
    assert len(sims) == 1 and ast.unparse(sims[0].value) == "self._alpasim_runtime_stub"
    calls_self = {n.func.attr for n in ast.walk(worker) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and ast.unparse(n.func.value) == "self"}
    assert {"_run_rollout", "_on_rollout_succeeded", "_on_rollout_failed"} <= calls_self


def test_upstream_entrypoint_and_cli_sequences_match_what_main_reproduces():
    tree = pinned_tree(f"{_G_RT}/cosmos/entrypoint.py")
    main = find_def(tree, "main")
    assert ast_signature(main) == [("argv", "POSITIONAL_OR_KEYWORD", True)]
    imported = {a.name for n in tree.body if isinstance(n, ast.ImportFrom)
                and n.module == "alpagym_runtime.cosmos" for a in n.names}
    assert {"rollout_backend", "trainer"} <= imported  # registrations happen on import
    cli = pinned_tree("alpagym/packages/host/src/alpagym_host/cli.py")
    guard = cli.body[-1]
    assert isinstance(guard, ast.If) and "__main__" in ast.unparse(guard.test)
    assert [ast.unparse(s) for s in guard.body] == ["register_config_schema()", "main()"]
    lifecycle = pinned_tree("alpagym/packages/host/src/alpagym_host/run_lifecycle.py")
    launcher = find_def(lifecycle, "_build_cosmos_launcher_command")
    assert ast_params(launcher) == host_launch.LAUNCHER_PARAMS
    literals = [n.value for n in ast.walk(launcher) if isinstance(n, ast.Constant)]
    assert literals.count(host_launch.UPSTREAM_MODULE) == 1


# ------------------------------------------------------------------- sink


def make_sink(tmp_path, label="rollout-test", limits=None, **kwargs):
    return EventSink(tmp_path / "physloop_capture", stream_label=label,
                     limits=limits or LIMITS, **kwargs)


def test_sink_canonical_bytes_match_gate_identity():
    for value in ({"b": 1, "a": [1.5, "é", None, True]}, {"x": {"z": -0.0, "y": 10**20}}):
        assert sink.canonical_bytes(value) == canonical_json_bytes(value)


def test_sink_hash_chain_known_answer_and_clock_independence(tmp_path):
    first = make_sink(tmp_path, "a", wall_ns=lambda: 1, mono_ns=lambda: 2, getpid=lambda: 77)
    first.emit("probe", {"value": 1})
    first.close()
    second = make_sink(tmp_path, "b", wall_ns=lambda: 9, mono_ns=lambda: 9, getpid=lambda: 77)
    second.emit("probe", {"value": 1})
    second.close()
    lines_a = first.path.read_bytes().splitlines()
    record = json.loads(lines_a[1])
    header = json.loads(lines_a[0])
    expected = hashlib.sha256(canonical_json_bytes(
        {"seq": 1, "prev": header["hash"], "event": {"kind": "probe", "value": 1}})).hexdigest()
    assert record["hash"] == expected and record["time"] == {"wall_ns": 1, "monotonic_ns": 2}
    report = read_stream(first.path, max_bytes=10**6)
    assert report["status"] == "passed" and report["issues"] == []
    assert [e["kind"] for e in report["events"]] == ["stream_opened", "probe", "stream_closed"]
    # Same events and pid, different label/clock: only the header label changes the chain.
    rec_b = [json.loads(x) for x in second.path.read_bytes().splitlines()]
    assert rec_b[1]["event"] == record["event"] and rec_b[1]["time"] != record["time"]


@pytest.mark.parametrize(("damage", "code"), [
    (lambda lines: lines[:1] + [lines[1].replace(b'"value":1', b'"value":2')] + lines[2:],
     "hash_mismatch"),
    (lambda lines: lines[:1] + lines[2:], "sequence_gap"),
    (lambda lines: [lines[0], lines[2], lines[1], *lines[3:]], "sequence_gap"),
    (lambda lines: lines + [b'{"seq": 9'], "malformed_record"),
])
def test_read_stream_detects_tampering(tmp_path, damage, code):
    s = make_sink(tmp_path)
    s.emit("probe", {"value": 1})
    s.emit("probe", {"value": 3})
    s.close()
    lines = s.path.read_bytes().splitlines()
    s.path.write_bytes(b"\n".join(damage(lines)) + b"\n")
    report = read_stream(s.path, max_bytes=10**6)
    assert report["status"] == "blocked"
    assert code in {i["code"] for i in report["issues"]}


def test_read_stream_torn_tail_symlink_and_unclosed(tmp_path):
    s = make_sink(tmp_path)
    s.emit("probe", {"value": 1})
    data = s.path.read_bytes()
    s.path.write_bytes(data[:-5])
    assert "truncated_tail" in {i["code"] for i in read_stream(s.path, max_bytes=10**6)["issues"]}
    s.path.write_bytes(data)
    report = read_stream(s.path, max_bytes=10**6)
    assert report["status"] == "passed"
    assert [i["code"] for i in report["issues"]] == ["stream_not_closed"]
    link = tmp_path / "link.jsonl"
    link.symlink_to(s.path)
    with pytest.raises(SinkError):
        read_stream(link, max_bytes=10**6)
    with pytest.raises(SinkLimitError):
        read_stream(s.path, max_bytes=10)


def test_sink_refuses_reuse_bad_labels_symlinked_dirs_and_non_json(tmp_path):
    make_sink(tmp_path, "dup")
    with pytest.raises(SinkError):
        make_sink(tmp_path, "dup")
    with pytest.raises(SinkError):
        make_sink(tmp_path, "../escape")
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "linked").symlink_to(real)
    with pytest.raises(SinkError):
        EventSink(tmp_path / "linked" / "cap", stream_label="x", limits=LIMITS)
    s = make_sink(tmp_path, "nan")
    with pytest.raises(SinkError):
        s.emit("probe", {"value": float("nan")})
    with pytest.raises(SinkError):
        s.emit("probe", {"kind": "spoof"})
    for bad in ({}, {**LIMITS, "max_blob_bytes": 0}, {**LIMITS, "extra": 1},
                {**LIMITS, "max_stream_bytes": LIMITS["max_event_bytes"]}):
        with pytest.raises(SinkError):
            EventSink(tmp_path / "physloop_capture", stream_label="bad", limits=bad)


def test_oversize_event_is_replaced_by_its_hash_never_dropped(tmp_path):
    limits = {**LIMITS, "max_event_bytes": 2048, "max_stream_bytes": 100_000}
    s = make_sink(tmp_path, limits=limits)
    payload = {"blob": "x" * 5000}
    s.emit("big", payload)
    s.close()
    events = read_stream(s.path, max_bytes=10**6)["events"]
    oversize = events[1]
    assert oversize["kind"] == "event_oversize" and oversize["original_kind"] == "big"
    assert oversize["sha256"] == hashlib.sha256(
        canonical_json_bytes({"kind": "big", **payload})).hexdigest()


def test_stream_budget_writes_a_limit_record_then_refuses(tmp_path):
    limits = {**LIMITS, "max_event_bytes": 2048, "max_stream_bytes": 4096}
    s = make_sink(tmp_path, limits=limits)
    with pytest.raises(SinkLimitError):
        for index in range(100):
            s.emit("probe", {"index": index, "pad": "y" * 400})
    with pytest.raises(SinkError):
        s.emit("probe", {"index": -1})
    report = read_stream(s.path, max_bytes=10**6)
    # A8-S-01: the sink refused evidence, so the stream is blocked for evidence use, not closed.
    assert report["status"] == "blocked"
    (limit,) = [i for i in report["issues"] if i["code"] == "stream_limit_reached"]
    assert limit["severity"] == "blocking" and "probe" in limit["message"]
    assert report["events"][-1]["kind"] == "stream_limit_reached"
    assert report["events"][-1]["rejected_kind"] == "probe"
    assert s.path.stat().st_size <= limits["max_stream_bytes"]
    streams, _ = streams_from_reports([report])
    assert build_attempt_ledger(streams, scope="synthetic")["status"] != "passed"


def test_blobs_are_content_addressed_deduplicated_and_budgeted(tmp_path):
    limits = {**LIMITS, "max_blob_bytes": 10, "max_total_blob_bytes": 15}
    s = make_sink(tmp_path, limits=limits)
    first = s.put_blob(b"0123456789")
    digest = hashlib.sha256(b"0123456789").hexdigest()
    assert first == {"sha256": digest, "bytes": 10, "stored": True,
                     "path": f"blobs/{digest[:2]}/{digest}"}
    assert (s.directory / first["path"]).read_bytes() == b"0123456789"
    assert s.put_blob(b"0123456789")["deduplicated"] is True
    assert s.put_blob(b"x" * 11)["stored"] is False
    assert s.put_blob(b"abcdef") == {"sha256": hashlib.sha256(b"abcdef").hexdigest(),
                                     "bytes": 6, "stored": False, "reason": "blob_over_budget"}


def test_forked_child_opens_its_own_linked_stream(tmp_path):
    pid = {"value": 100}
    s = make_sink(tmp_path, "parent", getpid=lambda: pid["value"])
    s.emit("probe", {"n": 1})
    pid["value"] = 200
    s.emit("probe", {"n": 2})
    child = read_stream(s.directory / "parent-fork200.jsonl", max_bytes=10**6)
    assert child["events"][0]["parent_stream"]["label"] == "parent"
    assert child["events"][0]["parent_stream"]["last_seq"] == 1
    assert [e.get("n") for e in child["events"][1:]] == [2]
    parent = read_stream(s.directory / "parent.jsonl", max_bytes=10**6)
    assert [e.get("n") for e in parent["events"][1:]] == [1]


# ------------------------------------------------------------------- fake upstream


@dataclasses.dataclass
class EpisodeMetrics:
    aggregated: dict
    dense: dict


@dataclasses.dataclass
class EpisodeOutput:
    scene_id: str
    session_uuid: str
    num_steps: int
    policy_outputs: tuple
    executed_ego_trajectory: tuple
    route_waypoints: tuple
    metrics: object = None
    reward: object = None


class FakeServicer:
    """AlpaGym driver servicer stand-in; signatures checked against pinned driver_server.py."""

    def __init__(self):
        self._session_records = {}

    def start_session(self, request, context):
        return SimpleNamespace(status="ok")

    def close_session(self, request, context):
        self._session_records[str(request.session_uuid)] = SimpleNamespace(
            outputs=(1, 2, 3), executed_ego_trajectory=(), ground_truth=None)
        return SimpleNamespace()

    def pop_session_record(self, session_uuid):
        return self._session_records.pop(session_uuid)


class FakeChannel:
    """Plays AlpaSim: drives the driver session and replies per a script of outcomes."""

    def __init__(self, servicer, script):
        self.servicer, self.script, self.seed = servicer, list(script), itertools.count(100)

    def simulate(self, request, timeout):
        session = request.rollout_specs[0].session_uuids[0]
        outcome = self.script.pop(0)
        self.servicer.start_session(SimpleNamespace(session_uuid=session,
                                                    random_seed=next(self.seed)), None)
        self.servicer.close_session(SimpleNamespace(session_uuid=session), None)
        ret = SimpleNamespace(
            rollout_spec=SimpleNamespace(session_uuids=[session]),
            success=outcome != "sim_fail", error="" if outcome != "sim_fail" else "boom",
            rollout_uuid=f"r-{session[:6]}", aggregated_metrics={"progress": 0.5},
            timestep_metrics=[SimpleNamespace(name="collision_any")])
        return SimpleNamespace(rollout_returns=[ret],
                               SerializeToString=lambda: f"reply:{session}".encode())


def build_request(*, scene_ids, n_generation, driver_host, driver_port, n_concurrent_per_driver,
                  session_uuid):
    return SimpleNamespace(
        rollout_specs=[SimpleNamespace(session_uuids=[session_uuid], scenario_id=scene_ids[0])],
        SerializeToString=lambda: f"request:{session_uuid}".encode())


class RewardScript:
    def __init__(self, failing_calls):
        self.calls, self.failing = 0, set(failing_calls)

    def __call__(self, episode, ground_truth, reward_config):
        self.calls += 1
        if self.calls in self.failing:
            raise KeyError("collision_any")  # missing metric -> silent retry (R3 finding 6)
        return SimpleNamespace(total=float(self.calls), report_metrics={"progress": 0.5})


FAKE_TRAINER_SOURCE = '''
class AlpagymGRPOTrainer:
    def __init__(self, config, parallel_dims, **kwargs):
        self.config = config
        self.model = kwargs["model"]
        self.optimizers = kwargs["optimizers"]
        self.lr_schedulers = kwargs["lr_schedulers"]
        self.grad_norms = list(kwargs["grad_norms"])
        self.export_gate = kwargs.get("export_gate")  # test hook: writer blocks mid-export
        self.export_async = kwargs.get("export_async", True)
        self.upload_thread = None

    def step_training(self, rollouts, current_step, total_steps, remain_samples_num,
                      inter_policy_nccl, is_master_replica, do_save_checkpoint=False):
        kept = filter_trainable_rollouts(
            rollouts, current_step=current_step,
            train_batch_per_replica=int(self.config.train.train_batch_per_replica),
            allowed_outdated_steps=0)
        samples, advantages = self._prepare_training_data(kept)
        if samples:  # reversed order stands in for the pinned unseeded randperm
            order = list(reversed(range(len(samples))))
            self._train_minibatch([samples[i] for i in order], [advantages[i] for i in order],
                                  inter_policy_nccl)
        self.all_reduce_states(inter_policy_nccl)
        if is_master_replica and do_save_checkpoint:
            self._save_checkpoint(current_step, total_steps, remain_samples_num)
        return {"loss": 0.25, "kept": len(kept)}

    def _prepare_training_data(self, rollouts):
        samples, advantages = [], []
        for rollout in rollouts:  # two replay rows per rollout: one valid step, one padding row
            for step in range(2):
                signal = SimpleNamespace(old_logprobs=[-1.0 - step], is_padding=[step == 1])
                samples.append(SimpleNamespace(training_signal=signal))
                advantages.append(0.0 if step == 1 else float(rollout.advantage))
        return samples, advantages

    def _forward_with_reference(self, model_inputs):
        return [-1.5, -2.5], None

    def _train_minibatch(self, minibatch_samples, minibatch_advantages, inter_policy_nccl):
        self._forward_with_reference({"rows": len(minibatch_samples)})
        return (0.1, 1.0, 0.0, -1.5, -1.5, 0.0)

    def all_reduce_states(self, inter_policy_nccl):
        grad_norm = self.grad_norms.pop(0)
        if grad_norm != 0.0:
            self.optimizers.step()
        return grad_norm

    def _save_checkpoint(self, current_step, total_steps, remain_samples_num):
        if current_step == total_steps or self.config.train.ckpt.export_safetensors:
            self.export_safetensors(
                output_dir=self.config.train.output_dir,
                rel_path=os.path.join("safetensors", f"step_{current_step}"),
                trainable_only=False, is_final=current_step == total_steps, dtype=None)
        return None

    def export_safetensors(self, output_dir, rel_path, trainable_only, is_final, dtype):
        # Shaped like pinned cosmos-rl LLMTrainer.export_safetensors (d2a2c57,
        # llm_trainer.py:771-790): the files are written on a daemon thread named
        # "save_and_upload_safetensors", kept on self.upload_thread, never joined here.
        out = os.path.join(output_dir, rel_path)
        os.makedirs(out)
        step = rel_path.rsplit("_", 1)[1]
        first_written = threading.Event()
        gate = self.export_gate

        def write():
            with open(os.path.join(out, "model.safetensors"), "wb") as handle:
                handle.write(b"weights-" + step.encode())
            first_written.set()
            if gate is not None:
                gate.wait()
            with open(os.path.join(out, "config.json"), "wb") as handle:
                handle.write(b"{}")

        if not self.export_async:  # upstream drift: synchronous write, no thread handle
            write()
            return
        self.upload_thread = threading.Thread(target=write, name="save_and_upload_safetensors",
                                              daemon=True)
        self.upload_thread.start()
        first_written.wait()  # deterministic: one complete file exists when this returns
'''


def trainer_config(tmp_path, export=True, batch=1):
    return SimpleNamespace(train=SimpleNamespace(
        train_batch_per_replica=batch, output_dir=str(tmp_path / "outputs"),
        ckpt=SimpleNamespace(export_safetensors=export)))

FAKE_BACKEND_SOURCE = '''
class RuntimeServiceStub:
    def __init__(self, channel):
        self._channel = channel

    def simulate(self, request, timeout=None):
        return self._channel.simulate(request, timeout)
'''


class FakeOptimizers:
    """A plain optimizer with no param groups (the F5 snapshot then records none)."""

    def __init__(self):
        self.steps = 0
        self.param_groups = []
        self.state = {}

    def step(self):
        self.steps += 1


class FakeParam:
    def __init__(self, dtype, shape, requires_grad):
        self.dtype, self.shape, self.requires_grad = dtype, shape, requires_grad


class FakeModel:
    def named_parameters(self):
        return [("vlm.lm_head.weight", FakeParam("torch.bfloat16", (4, 2), False)),
                ("expert.layers.0.weight", FakeParam("torch.bfloat16", (2, 2), True))]

    def named_buffers(self):
        return [("action_in_proj.freqs", FakeParam("torch.float32", (8,), False))]

    def named_modules(self):
        return [("", SimpleNamespace(training=True)), ("vlm", SimpleNamespace(training=False))]


def install_fakes(monkeypatch, reward):
    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys_modules(), name, mod)
        return mod

    worker_ns = {"itertools": itertools, "logging": logging, "queue": queue,
                 "threading": threading, "uuid": uuid, "Future": Future,
                 "dataclass": dataclasses.dataclass, "field": dataclasses.field,
                 "replace": dataclasses.replace, "Callable": typing.Callable,
                 "RewardConfig": object, "RuntimeServiceStub": object, "RLPayload": object,
                 "EgodriverServer": object, "build_simulation_request_proto": build_request,
                 "measure_perf": lambda *a, **k: (lambda f: f),
                 "timed_scope": lambda *a, **k: contextlib.nullcontext(),
                 "compute_reward": reward, "EpisodeMetrics": EpisodeMetrics,
                 "EpisodeOutput": EpisodeOutput}
    worker_mod = exec_pinned(f"{_G_RT}/episode_runner/streaming_worker.py",
                             {"logger", "SharedPayloadState", "_RolloutJob",
                              "StreamingRolloutWorker"}, worker_ns,
                             "alpagym_runtime.episode_runner.streaming_worker")
    monkeypatch.setitem(sys_modules(), worker_mod.__name__, worker_mod)
    filter_mod = exec_pinned(f"{_G_RT}/cosmos/rollout_filter.py",
                             {"logger", "filter_trainable_rollouts"},
                             {"logging": logging, "pathlib": pathlib, "Any": typing.Any},
                             "alpagym_runtime.cosmos.rollout_filter")
    backend = module("alpagym_runtime.cosmos.rollout_backend")
    exec(FAKE_BACKEND_SOURCE, backend.__dict__)  # noqa: S102 - local fake source
    generation = exec_pinned_method(
        f"{_G_RT}/cosmos/rollout_backend.py", "AlpagymRollout.rollout_generation",
        {"measure_perf": lambda *a, **k: (lambda f: f), "RLPayload": object, "Any": typing.Any,
         "RolloutResult": lambda completions: SimpleNamespace(completions=completions),
         "logger": logging.getLogger("pinned.rollout_backend")})
    backend.AlpagymRollout = type("AlpagymRollout", (), {"rollout_generation": generation})
    trainer = module("alpagym_runtime.cosmos.trainer",
                     filter_trainable_rollouts=filter_mod.filter_trainable_rollouts,
                     SimpleNamespace=SimpleNamespace, os=os, threading=threading)
    exec(FAKE_TRAINER_SOURCE, trainer.__dict__)  # noqa: S102 - local fake source
    servicer_cls = type("EgodriverGrpcServicer", (FakeServicer,), {})  # fresh per test
    module("alpagym_runtime.alpasim.driver_server", EgodriverGrpcServicer=servicer_cls)
    wrapper_mod = module("alpamayo1_x_rl.models.expert_model.cosmos_wrapper")

    class ExpertModelCosmos:
        def load_hf_weights(self, model_name_or_path, parallel_dims=None, device=None,
                            revision=None):
            return None

    wrapper_mod.ExpertModelCosmos = ExpertModelCosmos

    def install_alpamayo_r1_runtime_bridge():
        sys_modules()["alpamayo1_x_rl.models.expert_model.cosmos_wrapper"]

    module("alpagym_alpamayo_r1.bundle",
           install_alpamayo_r1_runtime_bridge=install_alpamayo_r1_runtime_bridge)
    upstream_calls = []
    module("alpagym_runtime.cosmos.entrypoint",
           main=lambda argv=None: upstream_calls.append(argv) or "upstream-done")
    return SimpleNamespace(worker=worker_mod, backend=backend, trainer=trainer,
                           wrapper=wrapper_mod, upstream_calls=upstream_calls,
                           servicer_cls=servicer_cls)


def sys_modules():
    import sys

    return sys.modules


def test_fake_signatures_match_pinned_source(monkeypatch):
    fakes = install_fakes(monkeypatch, RewardScript(()))
    checks = [
        (f"{_G_RT}/alpasim/driver_server.py", "EgodriverGrpcServicer.start_session",
         FakeServicer.start_session),
        (f"{_G_RT}/alpasim/driver_server.py", "EgodriverGrpcServicer.close_session",
         FakeServicer.close_session),
        (f"{_G_RT}/alpasim/driver_server.py", "EgodriverGrpcServicer.pop_session_record",
         FakeServicer.pop_session_record),
        ("alpagym/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/bundle.py",
         "install_alpamayo_r1_runtime_bridge",
         sys_modules()["alpagym_alpamayo_r1.bundle"].install_alpamayo_r1_runtime_bridge),
        ("alpamayo-recipes/recipes/alpamayo1_x_rl/models/expert_model/cosmos_wrapper.py",
         "ExpertModelCosmos.load_hf_weights", fakes.wrapper.ExpertModelCosmos.load_hf_weights),
        (f"{_G_RT}/cosmos/entrypoint.py", "main",
         sys_modules()["alpagym_runtime.cosmos.entrypoint"].main),
    ]
    for name in ("__init__", "step_training", "_prepare_training_data",
                 "_forward_with_reference", "_train_minibatch", "all_reduce_states",
                 "_save_checkpoint"):
        checks.append((f"{_G_RT}/cosmos/trainer.py", f"AlpagymGRPOTrainer.{name}",
                       getattr(fakes.trainer.AlpagymGRPOTrainer, name)))
    for fixture, qualname, fake in checks:
        node = find_def(pinned_tree(fixture), qualname)
        assert runtime_signature(fake) == ast_signature(node), qualname


# ------------------------------------------------------------------- capture install


def new_capture(tmp_path, label="rollout-test", fatal=None, config=None):
    s = make_sink(tmp_path, label)
    return Capture(s, config or CONFIG, on_fatal=fatal or (lambda message: None)), s


def test_install_wraps_every_target_once_and_is_idempotent(monkeypatch, tmp_path):
    fakes = install_fakes(monkeypatch, RewardScript(()))
    capture, _ = new_capture(tmp_path)
    original_submit = fakes.worker.StreamingRolloutWorker.submit_payload
    first = capture.install()
    assert first == {"installed": [t.target_id for t in WRAP_TARGETS], "already_installed": []}
    wrapped = fakes.worker.StreamingRolloutWorker.submit_payload
    assert getattr(wrapped, SENTINEL) == "submit_payload"
    assert wrapped.__wrapped__ is original_submit
    second = capture.install()
    assert second == {"installed": [], "already_installed": [t.target_id for t in WRAP_TARGETS]}
    assert fakes.worker.StreamingRolloutWorker.submit_payload is wrapped
    assert getattr(fakes.backend.RuntimeServiceStub, SENTINEL) == "runtime_stub"


def test_missing_target_aborts_install(monkeypatch, tmp_path):
    fakes = install_fakes(monkeypatch, RewardScript(()))
    del fakes.trainer.AlpagymGRPOTrainer._save_checkpoint
    capture, _ = new_capture(tmp_path)
    with pytest.raises(CaptureInstallError, match="save_checkpoint"):
        capture.install()
    monkeypatch.delitem(sys_modules(), "alpagym_alpamayo_r1.bundle")
    with pytest.raises(CaptureInstallError):
        new_capture(tmp_path, "other")[0].install(
            [t for t in WRAP_TARGETS if t.target_id == "runtime_bridge"])


def test_signature_drift_aborts_install(monkeypatch, tmp_path):
    fakes = install_fakes(monkeypatch, RewardScript(()))

    def drifted(self, rollout_job, exc, extra):
        return None

    fakes.worker.StreamingRolloutWorker._on_rollout_failed = drifted
    with pytest.raises(CaptureInstallError, match="differs from the pinned"):
        new_capture(tmp_path)[0].install()


def test_capture_config_is_required_and_validated():
    assert entrypoint.validate_capture_config(CONFIG) == CONFIG
    for bad in ({**CONFIG, "parameter_hashing": "fast"}, {k: v for k, v in CONFIG.items()
                                                          if k != "limits"},
                {**CONFIG, "max_inline_items": 0}, {**CONFIG, "capture_simulator_bytes": 1},
                {**CONFIG, "schema": "v0"}):
        with pytest.raises(ValueError):
            entrypoint.validate_capture_config(bad)


# ------------------------------------------------------------------- end to end on pinned code


def run_pinned_rollouts(monkeypatch, tmp_path):
    reward = RewardScript(failing_calls={2})
    fakes = install_fakes(monkeypatch, reward)
    capture, s = new_capture(tmp_path)
    capture.install()
    servicer = fakes.servicer_cls()
    # Script: A ok; B simulates ok but reward raises KeyError -> retry C ok. Then prompt 9:
    # four simulator failures exhaust a budget of 3.
    channel = FakeChannel(servicer, ["ok", "ok", "ok", "sim_fail", "sim_fail", "sim_fail",
                                     "sim_fail"])
    stub = fakes.backend.RuntimeServiceStub(channel)  # module-global lookup, as init_engine does
    driver = SimpleNamespace(servicer=servicer,
                             topology_endpoint=SimpleNamespace(host="localhost", port=5000))
    worker = fakes.worker.StreamingRolloutWorker(
        alpasim_runtime_stub=stub, driver_server=driver, simulation_timeout_s=5.0,
        reward_config=object(), max_concurrent_rollouts=1, rollouts_per_payload=2,
        scene_id_resolver=lambda payload: SCENE, max_scene_retries=3)
    rollout = fakes.backend.AlpagymRollout()
    rollout._worker = worker
    results = rollout.rollout_generation([SimpleNamespace(prompt_idx=7)], None, None,
                                         current_weight_version=4)
    with pytest.raises(RuntimeError, match="exhausted retries"):
        rollout.rollout_generation([SimpleNamespace(prompt_idx=9)], None, None,
                                   current_weight_version=4)
    worker.shutdown()
    return fakes, capture, s, results, servicer


def test_pinned_worker_capture_reconstructs_silent_retry_and_drop(monkeypatch, tmp_path):
    _fakes, _capture, s, results, _servicer = run_pinned_rollouts(monkeypatch, tmp_path)
    s.close()
    report = read_stream(s.path, max_bytes=10**7)
    assert report["status"] == "passed", report["issues"]
    streams, issues = streams_from_reports([report])
    assert issues == []
    ledger = build_attempt_ledger(streams, scope="synthetic")
    assert ledger["status"] == "passed", ledger["issues"]
    # prompt 7: A ok, B reward KeyError -> retry C ok (3 attempts, 2 successes); prompt 9:
    # four simulator failures with budget 3 (4 attempts), sibling never started.
    assert ledger["counts"] == {"payloads": 2, "attempts": 7, "succeeded": 2,
                                "silent_retries": 4, "dropped_payloads": 1,
                                "unterminated_attempts": 0, "never_started_siblings": 1}
    assert ledger["failure_causes"] == {"KeyError": 1, "RuntimeError": 4}
    retry_group, dropped_group = ledger["groups"]
    episodes = results[0].completions
    assert retry_group["members"] == [e.session_uuid for e in episodes]
    assert retry_group["member_rewards"] == [1.0, 3.0]  # reward call 2 raised KeyError
    assert retry_group["member_is_retry"] == [False, True]
    assert [f["cause_type"] for f in retry_group["failed_attempts"]] == ["KeyError"]
    assert dropped_group["dropped"] and dropped_group["members"] == []
    assert [f["outcome"] for f in dropped_group["failed_attempts"]] == [
        "failed_retry", "failed_retry", "failed_retry", "failed_dropped"]
    events = report["events"]
    sims = [e for e in events if e["kind"] == "simulate_call"]
    assert len(sims) == 7 and all(e["request_blob"]["stored"] for e in sims)
    assert sims[0]["returns"][0]["timestep_metrics"] == ["collision_any"]
    seeds = [e["random_seed"] for e in events if e["kind"] == "session_started"]
    assert seeds == list(range(100, 107))
    closes = [e["recorded_steps"] for e in events if e["kind"] == "session_closed"]
    assert closes == [3] * 7


def test_pinned_filter_is_captured_before_it_unlinks(monkeypatch, tmp_path):
    fakes, _capture, s, results, _ = run_pinned_rollouts(monkeypatch, tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    rollouts = []
    for index, episode in enumerate(results[0].completions):
        path = artifacts / f"{SCENE}_{episode.session_uuid}_{'a' * 31}{index}.json"
        path.write_bytes(f"episode-{index}".encode())
        rollouts.append(SimpleNamespace(prompt_idx=7, completion=str(path), weight_version=4,
                                        advantage=0.5, reward=1.0))
    optimizers = FakeOptimizers()
    trainer = fakes.trainer.AlpagymGRPOTrainer(
        trainer_config(tmp_path), None, model=FakeModel(), optimizers=optimizers,
        lr_schedulers=SimpleNamespace(get_last_lr=lambda: [0.0001], last_epoch=0), grad_norms=[0.0, 2.0])
    trainer.step_training(rollouts, 5, 10, 0, None, True, True)
    trainer.step_training([], 6, 10, 0, None, True)
    dropped = Path(rollouts[1].completion)
    assert not dropped.exists()  # the pinned filter kept one and unlinked the other member
    assert optimizers.steps == 1 and "step" not in vars(optimizers)
    s.close()
    report = read_stream(s.path, max_bytes=10**7)
    events = report["events"]
    received = [e for e in events if e["kind"] == "trainer_rollouts_received"]
    assert [len(e["rollouts"]) for e in received] == [2, 0]
    assert received[0]["rollouts"][1]["completion_file"] == {
        "status": "hashed", "sha256": hashlib.sha256(b"episode-1").hexdigest(), "bytes": 9}
    kept = [e["kept"] for e in events if e["kind"] == "trainer_rollouts_kept"]
    assert kept == [[rollouts[0].completion], []]
    reduces = [e for e in events if e["kind"] == "optimizer_reduce"]
    assert [(e["grad_norm"], e["optimizer_step_calls"]) for e in reduces] == [(0.0, 0), (2.0, 1)]
    assert reduces[0]["lr"] == [0.0001]
    snaps = [e for e in events if e["kind"] == "parameter_snapshot"]
    assert [e["phase"] for e in snaps] == ["init", "pre_step", "post_step", "pre_step",
                                           "post_step"]
    # F5 schema 2 under metadata_only: rule-11 dtype names, nothing hashed (null, never zero)
    assert [(e["schema_version"], e["trainer_step"]) for e in snaps] == [
        (2, None), (2, 5), (2, 5), (2, 6), (2, 6)]
    # the emitted top-level keys, pinned (the sink's ``kind``/``seq`` plus the docstring
    # schema; review F5-03): any drift fails here, not silently downstream
    assert {frozenset(e) for e in snaps} == {frozenset({
        "kind", "seq", "schema_version", "phase", "trainer_step", "tensors", "training_flags",
        "optimizer_state", "step_counters", "applied_lr"})}
    assert snaps[0]["tensors"]["vlm.lm_head.weight"] == {
        "kind": "param", "dtype": "BF16", "shape": [4, 2], "numel": 8, "requires_grad": False,
        "sha256": None, "hash_scope": None, "shard": None}
    assert snaps[0]["tensors"]["action_in_proj.freqs"]["kind"] == "buffer"
    assert snaps[0]["training_flags"] == {"": True, "vlm": False}
    assert (snaps[0]["optimizer_state"], snaps[0]["step_counters"], snaps[0]["applied_lr"]) == (
        {}, {}, [])
    schedules = [(e["phase"], e["trainer_step"], e["last_epoch"]) for e in events
                 if e["kind"] == "scheduler_state"]
    assert schedules == [("pre_step", 5, 0), ("post_step", 5, 0), ("pre_step", 6, 0),
                         ("post_step", 6, 0)]
    (configured,) = [e for e in events if e["kind"] == "trainer_configured"]
    assert {k: configured[k] for k in ("mini_batch", "iterations", "kl_beta")} == {
        "mini_batch": None, "iterations": None, "kl_beta": None}  # absent here: null
    forward = next(e for e in events if e["kind"] == "replay_forward")
    assert forward["log_probs"] == {"count": 2, "values": [-1.5, -2.5]}
    assert forward["kl_present"] is False
    # A8-C-03: per-row old log-probs, padding mask and pool row indices (E04), and the
    # export manifest hashed before pruning (E02/E21).
    pool = next(e for e in events if e["kind"] == "replay_pool_prepared")
    assert pool["rows"] == 2 and pool["rows_per_rollout"] == 2
    assert pool["rollouts"] == [rollouts[0].completion]
    started = next(e for e in events if e["kind"] == "minibatch_started")
    assert started["pool_call"] == pool["call"]
    assert started["row_indices"] == {"count": 2, "values": [1, 0]}
    assert started["old_logprobs"] == {"count": 2, "values": [[-2.0], [-1.0]]}
    assert started["is_padding"] == {"count": 2, "values": [[True], [False]]}
    assert started["advantages"] == {"count": 2, "values": [0.0, 0.5]}
    assert forward["minibatch_call"] == started["call"]
    saved = [e for e in events if e["kind"] == "checkpoint_save" and e["phase"] == "returned"]
    assert len(saved) == 1 and saved[0]["current_step"] == 5
    export = saved[0]["export"]
    assert export["status"] == "hashed"
    assert export["path"] == str(tmp_path / "outputs" / "safetensors" / "step_5")
    assert export["files"] == {"count": 2, "values": [
        {"path": "config.json", "status": "hashed", "bytes": 2,
         "sha256": hashlib.sha256(b"{}").hexdigest()},
        {"path": "model.safetensors", "status": "hashed", "bytes": 9,
         "sha256": hashlib.sha256(b"weights-5").hexdigest()}]}
    # E09 over the same capture: every member received once at the generated version.
    streams, _ = streams_from_reports([report])
    ledger = build_attempt_ledger(streams, scope="synthetic")
    integrity = check_trace_integrity(ledger, streams, cohort="strict_parity",
                                      expected_version=4)
    assert integrity["status"] == "passed", integrity["issues"]
    assert integrity["received"] == 2


def test_extraction_failure_is_recorded_not_raised(monkeypatch, tmp_path):
    install_fakes(monkeypatch, RewardScript(()))
    capture, s = new_capture(tmp_path)
    capture.install()
    request = SimpleNamespace(session_uuid="abc", random_seed="not-an-int")
    result = sys_modules()["alpagym_runtime.alpasim.driver_server"].EgodriverGrpcServicer(
    ).start_session(request, None)
    assert result.status == "ok"
    s.close()
    events = read_stream(s.path, max_bytes=10**6)["events"]
    errors = [e for e in events if e["kind"] == "capture_extract_error"]
    assert errors[0]["target"] == "session_start" and errors[0]["error_type"] == "ValueError"


def test_upstream_exception_is_recorded_and_propagates(monkeypatch, tmp_path):
    fakes = install_fakes(monkeypatch, RewardScript(()))
    capture, s = new_capture(tmp_path)
    capture.install()
    trainer = fakes.trainer.AlpagymGRPOTrainer(
        trainer_config(tmp_path), None, model=FakeModel(), optimizers=FakeOptimizers(),
        lr_schedulers=None, grad_norms=[])
    with pytest.raises(IndexError):
        trainer.all_reduce_states(None)  # pop from an empty script
    s.close()
    raised = [e for e in read_stream(s.path, max_bytes=10**6)["events"]
              if e["kind"] == "optimizer_reduce_raised"]
    assert raised[0]["error_type"] == "IndexError" and raised[0]["optimizer_step_calls"] == 0


def test_sink_failure_is_fatal_and_cannot_be_swallowed(monkeypatch, tmp_path):
    install_fakes(monkeypatch, RewardScript(()))
    fatal = []
    capture, s = new_capture(tmp_path, fatal=fatal.append)
    capture.install()
    s.close()
    servicer = sys_modules()["alpagym_runtime.alpasim.driver_server"].EgodriverGrpcServicer()
    with pytest.raises(CaptureFatal):
        try:
            servicer.start_session(SimpleNamespace(session_uuid="s", random_seed=1), None)
        except Exception:  # noqa: BLE001 - proves upstream-style handlers cannot swallow it
            pytest.fail("CaptureFatal must not be an Exception")
    assert fatal and "capture sink failure" in fatal[0]


def test_chained_load_hf_weights_reports_load_gaps(monkeypatch, tmp_path):
    fakes = install_fakes(monkeypatch, RewardScript(()))
    capture, s = new_capture(tmp_path)
    capture.install()
    bridge = sys_modules()["alpagym_alpamayo_r1.bundle"].install_alpamayo_r1_runtime_bridge
    bridge()
    bridge()  # idempotent second call
    load = fakes.wrapper.ExpertModelCosmos.load_hf_weights
    assert getattr(load, SENTINEL) == "load_hf_weights"
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "expert.a": "00000.safetensors", "vlm.b": "00000.safetensors",
        "extra.c": "00000.safetensors"}}))
    model = fakes.wrapper.ExpertModelCosmos()
    model.expert_model = SimpleNamespace(state_dict=lambda: {"expert.a": 0, "vlm.b": 0,
                                                             "action_in_proj.freqs": 0})
    model.load_hf_weights(str(ckpt))
    s.close()
    events = read_stream(s.path, max_bytes=10**6)["events"]
    loaded = next(e for e in events if e["kind"] == "weights_loaded")
    assert loaded["file_keys"] == 3 and loaded["model_keys"] == 3
    assert loaded["missing_in_file"] == {"count": 1, "values": ["action_in_proj.freqs"]}
    assert loaded["unexpected_in_file"] == {"count": 1, "values": ["extra.c"]}
    installs = [e for e in events if e["kind"] == "chained_wrap_installed"]
    assert [e["installed"] for e in installs] == [["load_hf_weights"], []]


def test_bridge_fails_closed_when_chained_target_is_missing(monkeypatch, tmp_path):
    fakes = install_fakes(monkeypatch, RewardScript(()))
    del fakes.wrapper.ExpertModelCosmos.load_hf_weights
    capture, _ = new_capture(tmp_path)
    capture.install()
    with pytest.raises(CaptureInstallError):
        sys_modules()["alpagym_alpamayo_r1.bundle"].install_alpamayo_r1_runtime_bridge()


def test_large_lists_spill_to_blobs(monkeypatch, tmp_path):
    install_fakes(monkeypatch, RewardScript(()))
    capture, s = new_capture(tmp_path, config={**CONFIG, "max_inline_items": 2})
    spilled = capture.bounded([1, 2, 3])
    assert spilled["count"] == 3 and spilled["blob"]["stored"] is True
    assert (s.directory / spilled["blob"]["path"]).read_bytes() == b"[1,2,3]"
    assert capture.bounded([float("nan")]) == {"count": 1, "values": ["nan"]}


# ------------------------------------------------------------------- entrypoint main


def write_cosmos_toml(tmp_path, resolved):
    path = tmp_path / "tmp-cosmos.toml"
    path.write_text(f'[custom]\nresolved_config_path = "{resolved}"\n')
    return path


def write_capture_config(tmp_path, config=CONFIG):
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(config))
    return path


def test_main_installs_then_calls_upstream_with_untouched_argv(monkeypatch, tmp_path):
    fakes = install_fakes(monkeypatch, RewardScript(()))
    run_dir = tmp_path / "run"
    toml = write_cosmos_toml(tmp_path, run_dir / "resolved_config.yaml")
    argv = ["--config", str(toml), "--extra", "x"]
    canary = "hf_" + "C" * 34
    environ = {**worker_env(tmp_path), "COSMOS_ROLE": "Rollout", "HF_TOKEN": canary,
               "NGC_API_KEY": "nvapi-" + "D" * 20, "PATH": "/usr/bin"}
    assert entrypoint.main(argv, environ=environ) == "upstream-done"
    assert fakes.upstream_calls == [argv]
    (stream_path,) = (run_dir / "physloop_capture").glob("rollout-*.jsonl")
    report = read_stream(stream_path, max_bytes=10**6)
    assert report["status"] == "passed"
    installed = next(e for e in report["events"] if e["kind"] == "capture_installed")
    assert installed["installed"] == [t.target_id for t in WRAP_TARGETS]
    assert installed["role"] == "Rollout"
    # A8-H-02: credentials leave the Cosmos role environment before upstream is imported.
    assert installed["worker_credentials"] == {
        "mode": "strip_offline", "removed_names": ["HF_TOKEN", "NGC_API_KEY"],
        "set": {"HF_HUB_OFFLINE": "1"}}
    assert "HF_TOKEN" not in environ and "NGC_API_KEY" not in environ
    assert environ["HF_HUB_OFFLINE"] == "1" and environ["PATH"] == "/usr/bin"
    assert canary not in stream_path.read_text()


def worker_env(tmp_path, mode="strip_offline"):
    return {"PHYSLOOP_CAPTURE_CONFIG": str(write_capture_config(tmp_path)),
            "PYTHONDONTWRITEBYTECODE": "1", "PHYSLOOP_WORKER_CREDENTIALS": mode}


def test_main_inherit_mode_discloses_credential_names_only(monkeypatch, tmp_path):
    install_fakes(monkeypatch, RewardScript(()))
    run_dir = tmp_path / "run"
    toml = write_cosmos_toml(tmp_path, run_dir / "resolved_config.yaml")
    canary = "hf_" + "E" * 34
    environ = {**worker_env(tmp_path, "inherit_disclosed"), "COSMOS_ROLE": "Policy",
               "HF_TOKEN": canary}
    entrypoint.main(["--config", str(toml)], environ=environ)
    (stream_path,) = (run_dir / "physloop_capture").glob("policy-*.jsonl")
    installed = next(e for e in read_stream(stream_path, max_bytes=10**6)["events"]
                     if e["kind"] == "capture_installed")
    assert installed["worker_credentials"] == {"mode": "inherit_disclosed",
                                               "present_credential_names": ["HF_TOKEN"]}
    assert environ["HF_TOKEN"] == canary and canary not in stream_path.read_text()


def _pinned_artifact_paths(run_root):
    import datetime as dt

    paths_mod = exec_pinned("alpagym/packages/host/src/alpagym_host/config.py",
                            {"ArtifactPaths"}, {"dataclass": dataclasses.dataclass, "Path": Path},
                            "pinned_config")
    art = exec_pinned("alpagym/packages/host/src/alpagym_host/run_artifacts.py",
                      {"build_artifact_paths"},
                      {"ArtifactPaths": paths_mod.ArtifactPaths, "Path": Path,
                       "datetime": SimpleNamespace(now=lambda tz: SimpleNamespace(
                           strftime=lambda fmt: "20261109T000000Z")),
                       "UTC": dt.UTC, "uuid4": uuid.uuid4,
                       "DictConfig": object}, "pinned_run_artifacts")
    return art.build_artifact_paths(SimpleNamespace(run_root=run_root))


def test_capture_directory_from_the_pinned_artifact_paths(tmp_path):
    """A8-C-02: the pinned default run_root is relative and refused; an absolute one works."""
    default = _pinned_artifact_paths("./tmp/alpagym-runs")  # conf/default.yaml:17
    assert not default.resolved_config_path.is_absolute()
    with pytest.raises(CaptureInstallError, match="absolute"):
        entrypoint.capture_directory_from_toml(
            write_cosmos_toml(tmp_path, default.resolved_config_path))
    absolute = _pinned_artifact_paths(str(tmp_path / "runs"))
    toml = write_cosmos_toml(tmp_path, absolute.resolved_config_path)
    assert entrypoint.capture_directory_from_toml(toml) == absolute.run_dir / "physloop_capture"


@pytest.mark.parametrize("problem", ["no_env", "relative_run_dir", "bad_config", "bad_role",
                                     "bytecode_writes", "no_credential_mode", "bad_mode"])
def test_main_refuses_before_upstream_runs(monkeypatch, tmp_path, problem):
    fakes = install_fakes(monkeypatch, RewardScript(()))
    resolved = "run/resolved.yaml" if problem == "relative_run_dir" else str(
        tmp_path / "run" / "resolved.yaml")
    toml = write_cosmos_toml(tmp_path, resolved)
    config = {**CONFIG, "parameter_hashing": "fast"} if problem == "bad_config" else CONFIG
    environ = {**worker_env(tmp_path), "PHYSLOOP_CAPTURE_CONFIG": str(
        write_capture_config(tmp_path, config)),
               "COSMOS_ROLE": "Policy;rm" if problem == "bad_role" else "Policy"}
    if problem == "no_env":
        environ.pop("PHYSLOOP_CAPTURE_CONFIG")
    elif problem == "bytecode_writes":  # A8-H-01: workers must not write into the tree
        environ.pop("PYTHONDONTWRITEBYTECODE")
    elif problem == "no_credential_mode":  # A8-H-02: the credential mode has no default
        environ.pop("PHYSLOOP_WORKER_CREDENTIALS")
    elif problem == "bad_mode":
        environ["PHYSLOOP_WORKER_CREDENTIALS"] = "keep"
    with pytest.raises((CaptureInstallError, ValueError)):
        entrypoint.main(["--config", str(toml)], environ=environ)
    assert fakes.upstream_calls == []


# ------------------------------------------------------------------- host launcher


def pinned_lifecycle(monkeypatch):
    ns = {"RunConfig": object, "ExecutionBackend": object, "RunTopologyPlan": object,
          "Path": Path, "alpagym_project_root": lambda: Path("/opt/alpagym")}
    mod = exec_pinned("alpagym/packages/host/src/alpagym_host/run_lifecycle.py",
                      {"_build_cosmos_command", "_build_cosmos_launcher_command"}, ns,
                      "alpagym_host.run_lifecycle")
    monkeypatch.setitem(sys_modules(), mod.__name__, mod)
    return mod


def run_config():
    return SimpleNamespace(
        artifact_paths=SimpleNamespace(cosmos_config_path=Path("/run/cosmos_config.toml"),
                                       log_dir=Path("/run/logs")),
        cosmos=SimpleNamespace(launch=SimpleNamespace(policy_replicas=1, rollout_replicas=1,
                                                      controller_port=29500)))


UPSTREAM_COMMAND = ["uv", "run", "--all-packages", "--project", "/opt/alpagym", "python", "-m",
                    "cosmos_rl.launcher.launch_all", "--config", "/run/cosmos_config.toml",
                    "--policy", "1", "--rollout", "1", "--num-workers", "1", "--worker-idx",
                    "0", "--port", "29500", "--log-dir", "/run/logs",
                    "alpagym_runtime.cosmos.entrypoint"]


def test_pinned_launcher_command_known_answer_and_patch(monkeypatch):
    lifecycle = pinned_lifecycle(monkeypatch)
    local = SimpleNamespace(is_slurm_run=False)
    assert lifecycle._build_cosmos_command(run_config(), local, None, None) == UPSTREAM_COMMAND
    assert host_launch.install_launcher_patch(lifecycle) == "installed"
    assert host_launch.install_launcher_patch(lifecycle) == "already_installed"
    patched = lifecycle._build_cosmos_command(run_config(), local, None, None)
    assert patched == [*UPSTREAM_COMMAND[:-1], "teacher_rl.capture.entrypoint"]
    worker = lifecycle._build_cosmos_launcher_command(
        run_config(), project_root=Path("/w"), no_sync=True, worker_count=2, worker_index=1,
        controller_url="host0:29500")
    assert worker[:5] == ["uv", "run", "--no-sync", "--project", "/w"]
    assert worker[-3:] == ["--log-dir", "/run/logs", "teacher_rl.capture.entrypoint"]


@pytest.mark.parametrize("command", [
    UPSTREAM_COMMAND[:-1],
    [*UPSTREAM_COMMAND, "--tail"],
    [*UPSTREAM_COMMAND[:-1], "alpagym_runtime.cosmos.entrypoint",
     "alpagym_runtime.cosmos.entrypoint"],
    "uv run python -m alpagym_runtime.cosmos.entrypoint",
])
def test_rewrite_refuses_unexpected_command_shapes(command):
    with pytest.raises(host_launch.LaunchRefused):
        host_launch.rewrite_launcher_command(command)


def test_launcher_patch_refuses_signature_drift():
    mod = types.ModuleType("alpagym_host.run_lifecycle")
    mod._build_cosmos_launcher_command = lambda config, project_root: []
    with pytest.raises(host_launch.LaunchRefused):
        host_launch.install_launcher_patch(mod)
    with pytest.raises(host_launch.LaunchRefused):
        host_launch.install_launcher_patch(types.ModuleType("empty"))


def test_capture_tree_stage_and_verify(tmp_path):
    tree = tmp_path / "tree"
    manifest = host_launch.stage_capture_tree(REPO_ROOT, tree, scope="synthetic")
    assert [f["path"] for f in manifest["files"]] == list(host_launch.CAPTURE_FILES)
    for entry in manifest["files"]:
        assert entry["sha256"] == hashlib.sha256((REPO_ROOT / entry["path"]).read_bytes(
        )).hexdigest()
    assert host_launch.verify_capture_tree(tree, manifest)["status"] == "passed"
    with pytest.raises(host_launch.LaunchRefused):
        host_launch.stage_capture_tree(REPO_ROOT, tree, scope="synthetic")
    (tree / "teacher_rl" / "capture" / "sink.py").write_bytes(b"# tampered\n")
    assert host_launch.verify_capture_tree(tree, manifest)["status"] == "blocked"


@pytest.mark.parametrize("mutation", ["extra", "symlink", "tree_hash"])
def test_capture_tree_rejects_extra_links_and_bad_manifest(tmp_path, mutation):
    tree = tmp_path / "tree"
    manifest = host_launch.stage_capture_tree(REPO_ROOT, tree, scope="synthetic")
    if mutation == "extra":
        (tree / "gate.py").write_text("x = 1\n")
    elif mutation == "symlink":
        (tree / "teacher_rl" / "link.py").symlink_to(tree / "teacher_rl" / "__init__.py")
    else:
        manifest = {**manifest, "tree_hash": "0" * 64}
    assert host_launch.verify_capture_tree(tree, manifest)["status"] == "blocked"


def test_capture_environment_adds_only_four_names_and_no_secrets(tmp_path):
    canary = "hf_" + "Q" * 34
    base = {"HF_TOKEN": canary, "PATH": "/usr/bin"}
    kwargs = {"capture_tree": tmp_path / "tree", "capture_config_path": tmp_path / "c.json",
              "worker_credentials": "strip_offline"}
    updates = host_launch.capture_environment(base, **kwargs)
    assert updates == {"PYTHONPATH": str(tmp_path / "tree"),
                       "PHYSLOOP_CAPTURE_CONFIG": str(tmp_path / "c.json"),
                       "PYTHONDONTWRITEBYTECODE": "1",
                       "PHYSLOOP_WORKER_CREDENTIALS": "strip_offline"}
    assert canary not in json.dumps(updates)
    same = {"PYTHONPATH": str(tmp_path / "tree")}
    assert host_launch.capture_environment(same, **kwargs)
    with pytest.raises(host_launch.LaunchRefused):
        host_launch.capture_environment({"PYTHONPATH": "/elsewhere"}, **kwargs)
    with pytest.raises(host_launch.LaunchRefused):
        host_launch.capture_environment({}, **{**kwargs, "capture_tree": Path("rel")})
    with pytest.raises(host_launch.LaunchRefused):
        host_launch.capture_environment({}, **{**kwargs, "worker_credentials": "keep"})


NUREC_RECORDS = ["acquire:nurec_scenes", "terms:nurec_scenes"]


def launch_inputs(tmp_path, tree, manifest_path, *, scope="actual", route="hydra_only"):
    """A real build_launch spec over this test's staged tree (actual scope, hydra route)."""
    from scripts.rehearse_s4 import (
        RUN_DIR_NAME,
        synthetic_base_cosmos_toml,
        synthetic_registration,
    )
    from teacher_rl import launch

    registration = synthetic_registration(route=route)
    registration["scope"] = scope
    extra = {} if route == "hydra_only" else {
        "run_dir": f"/data/alpagym-runs/{RUN_DIR_NAME}",
        "base_cosmos_toml": synthetic_base_cosmos_toml(), "resolved_config_sha256": "5" * 64}
    spec = launch.build_launch(
        registration, alpagym_checkout=str(tmp_path / "alpagym"), capture_tree=str(tree),
        capture_manifest=str(manifest_path),
        capture_config=str(write_capture_config(tmp_path)), worker_credentials="strip_offline",
        action_authority_ref="D-032 launch approval",
        alpasim_cache_dir="/data/cache/alpagym-alpasim",
        nurec_authorization_refs={"terms": "D-032 NuRec terms",
                                  "acquire": "D-032 NuRec acquisition"}, **extra)
    assert spec["status"] == "passed", spec["issues"]
    return spec, registration


def launch_decisions(tmp_path, spec, registration, *, grants=None, records=NUREC_RECORDS):
    from gate.s2_identity import identity_hash
    from tests.test_s2_guest_bootstrap import render_decisions

    grants = [("s4_launch_plan", spec["launch_hash"]),
              ("s4_registration", identity_hash(registration))] if grants is None else grants
    return render_decisions(tmp_path / "DECISIONS.md", grants, records)


def host_main_setup(monkeypatch, tmp_path, *, scope="actual"):
    lifecycle = pinned_lifecycle(monkeypatch)
    calls = []
    config_mod = types.ModuleType("alpagym_host.config")
    config_mod.register_config_schema = lambda: calls.append("register")
    cli = types.ModuleType("alpagym_host.cli")
    cli.main = lambda: calls.append("cli.main") or "ran"
    monkeypatch.setitem(sys_modules(), "alpagym_host.config", config_mod)
    monkeypatch.setitem(sys_modules(), "alpagym_host.cli", cli)
    tree = tmp_path / "tree"
    manifest = host_launch.stage_capture_tree(REPO_ROOT, tree, scope="synthetic")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    spec, registration = launch_inputs(tmp_path, tree, manifest_path, scope=scope)
    (tmp_path / "launch-spec.json").write_text(json.dumps(spec))
    (tmp_path / "registration.json").write_text(json.dumps(registration))
    decisions = launch_decisions(tmp_path, spec, registration)
    tail = spec["argv"][spec["argv"].index("teacher_rl.capture.host_launch") + 1:]
    split = tail.index("--")
    args = [*tail[:split], "--launch-spec", str(tmp_path / "launch-spec.json"),
            "--registration", str(tmp_path / "registration.json"),
            "--decisions", str(decisions), "--decision-id", "D-032", *tail[split:]]
    return lifecycle, calls, tree, args, spec


def run_host(args, tmp_path, environ, set_argv=lambda values: None):
    return host_launch.main(args, environ=environ, set_argv=set_argv,
                            getcwd=lambda: str(tmp_path / "alpagym"))


def test_host_main_patches_registers_and_hands_off(monkeypatch, tmp_path):
    lifecycle, calls, tree, args, spec = host_main_setup(monkeypatch, tmp_path)
    canary = "hf_" + "Z" * 34
    environ = {"HF_TOKEN": canary}
    argv_seen = []
    assert run_host(args, tmp_path, environ, argv_seen.append) == "ran"
    assert calls == ["register", "cli.main"]
    overrides = spec["argv"][spec["argv"].index("--") + 1:]
    assert "run_root='/data/alpagym-runs'" in overrides
    assert argv_seen == [["teacher_rl.capture.entrypoint", *overrides]]
    # The host CLI keeps HF_TOKEN for its own preflight; each Cosmos role strips it on entry
    # (PHYSLOOP_WORKER_CREDENTIALS=strip_offline; see test_main_installs_...).
    assert environ == {"HF_TOKEN": canary, "PYTHONPATH": str(tree),
                       "PHYSLOOP_CAPTURE_CONFIG": str(tmp_path / "capture.json"),
                       "PYTHONDONTWRITEBYTECODE": "1",
                       "PHYSLOOP_WORKER_CREDENTIALS": "strip_offline"}
    assert canary not in json.dumps(argv_seen)
    assert getattr(lifecycle._build_cosmos_launcher_command, host_launch.SENTINEL)


def _run_root_index(args):
    return next(i for i, a in enumerate(args) if a.startswith("run_root="))


@pytest.mark.parametrize("mutation", ["placeholder_authority", "tampered_tree", "no_overrides",
                                      "no_run_root", "relative_run_root", "two_run_roots",
                                      "pycache"])
def test_host_main_refuses_without_authority_or_integrity(monkeypatch, tmp_path, mutation):
    lifecycle, calls, tree, args, _ = host_main_setup(monkeypatch, tmp_path)
    if mutation == "placeholder_authority":
        args[1] = "TBD"
    elif mutation == "tampered_tree":
        (tree / "teacher_rl" / "capture" / "entrypoint.py").write_text("# swapped\n")
    elif mutation == "no_overrides":
        args = args[: args.index("--")]
    elif mutation == "no_run_root":  # A8-C-02: the pinned default run_root is relative
        del args[_run_root_index(args)]
    elif mutation == "relative_run_root":
        args[_run_root_index(args)] = "run_root=./tmp/alpagym-runs"
    elif mutation == "two_run_roots":
        args.append("++run_root='/other'")
    else:  # A8-H-01: bytecode in the tree is unverified code
        (tree / "teacher_rl" / "__pycache__").mkdir()
        (tree / "teacher_rl" / "__pycache__" / "__init__.cpython-312.pyc").write_bytes(b"x")
    environ = {}
    with pytest.raises(host_launch.LaunchRefused):
        run_host(args, tmp_path, environ)
    assert calls == [] and environ == {}
    assert not getattr(lifecycle._build_cosmos_launcher_command, host_launch.SENTINEL, False)


def test_host_main_requires_the_authority_flag(monkeypatch, tmp_path):
    _lifecycle, calls, _tree, args, _ = host_main_setup(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        run_host(args[2:], tmp_path, {})
    index = args.index("--worker-credentials")
    for bad in (args[:index] + args[index + 2:],  # A8-H-02: no default credential mode
                [*args[:index + 1], "keep", *args[index + 2:]]):
        with pytest.raises(SystemExit):
            run_host(bad, tmp_path, {})
    for flag in host_launch.AUTHORITY_FLAGS:  # G1: every authority input is required
        index = args.index(flag)
        with pytest.raises(SystemExit):
            run_host(args[:index] + args[index + 2:], tmp_path, {})
    assert calls == []


# ------------------------------------------------------------------- G1: launch authority


def _refusal(monkeypatch, tmp_path, mutate, *, scope="actual"):
    lifecycle, calls, _tree, args, spec = host_main_setup(monkeypatch, tmp_path, scope=scope)
    args = mutate(args, spec) or args
    environ = {"HF_TOKEN": "hf_" + "Y" * 34}
    before = dict(environ)
    with pytest.raises(host_launch.LaunchRefused) as raised:
        run_host(args, tmp_path, environ)
    assert calls == [] and environ == before
    assert not getattr(lifecycle._build_cosmos_launcher_command, host_launch.SENTINEL, False)
    return str(raised.value)


def _decision(decision_id):
    def mutate(args, spec):
        args[args.index("--decision-id") + 1] = decision_id
    return mutate


def _rewrite_decisions(tmp_path, **kwargs):
    def mutate(args, spec):
        registration = json.loads(Path(args[args.index("--registration") + 1]).read_text())
        assert Path(args[args.index("--decisions") + 1]) == tmp_path / "DECISIONS.md"
        launch_decisions(tmp_path, spec, registration, **kwargs)
    return mutate


@pytest.mark.parametrize("mutate_name, code", [
    ("closed_d019", "launch_not_authorized"),
    ("no_authorizes_d024", "launch_not_authorized"),
    ("wrong_hash_d033", "launch_not_authorized"),
    ("no_nurec_records_d034", "nurec_records_missing"),
    ("registration_grant_missing", "launch_not_authorized"),
    ("only_terms_recorded", "nurec_records_missing"),
])
def test_g1_launch_needs_the_launch_and_registration_hash_and_nurec_records(
        monkeypatch, tmp_path, mutate_name, code):
    mutations = {
        "closed_d019": _decision("D-019"),
        "no_authorizes_d024": _decision("D-024"),
        "wrong_hash_d033": _decision("D-033"),
        "no_nurec_records_d034": _decision("D-034"),
        "registration_grant_missing": lambda args, spec: _rewrite_decisions(
            tmp_path, grants=[("s4_launch_plan", spec["launch_hash"])])(args, spec),
        "only_terms_recorded": _rewrite_decisions(tmp_path, records=["terms:nurec_scenes"]),
    }
    assert code in _refusal(monkeypatch, tmp_path, mutations[mutate_name])


def test_g1_launch_refuses_non_actual_scope(monkeypatch, tmp_path):
    message = _refusal(monkeypatch, tmp_path, lambda args, spec: None, scope="synthetic")
    assert "launch_scope_not_actual" in message and "registration_scope_not_actual" in message


def test_g1_launch_refuses_edited_spec_registration_argv_and_cwd(monkeypatch, tmp_path):
    def path(args, flag):
        return Path(args[args.index(flag) + 1])

    def edit_spec(args, spec):  # an extra override, launch_hash left as it was
        edited = dict(spec, argv=[*spec["argv"], "reward=other"])
        path(args, "--launch-spec").write_text(json.dumps(edited))
        args.append("reward=other")

    def reseal_spec(args, spec):  # re-derived hash: self-consistent but never authorized
        from gate.s2_identity import identity_hash

        edited = dict(spec, argv=[*spec["argv"], "reward=other"])
        edited["launch_hash"] = identity_hash(host_launch.launch_body(edited))
        path(args, "--launch-spec").write_text(json.dumps(edited))
        args.append("reward=other")

    def edit_registration(args, spec):
        registration = json.loads(path(args, "--registration").read_text())
        registration["registration_id"] = registration["registration_id"] + "-edited"
        path(args, "--registration").write_text(json.dumps(registration))

    cases = {"launch_hash_mismatch": edit_spec, "launch_not_authorized": reseal_spec,
             "registration_mismatch": edit_registration,
             "launch_argv_mismatch": lambda args, spec: args.append("reward=other")}
    for index, (code, mutate) in enumerate(cases.items()):
        sub = tmp_path / f"case{index}"
        sub.mkdir()
        assert code in _refusal(monkeypatch, sub, mutate), code
    (tmp_path / "cwd").mkdir()
    _lifecycle, calls, _tree, args, _ = host_main_setup(monkeypatch, tmp_path / "cwd")
    with pytest.raises(host_launch.LaunchRefused, match="launch_cwd_mismatch"):
        host_launch.main(args, environ={}, set_argv=lambda values: None,
                         getcwd=lambda: "/somewhere/else")
    assert calls == []


def test_g1_real_decision_log_authorizes_no_launch(monkeypatch, tmp_path):
    real = REPO_ROOT / "docs" / "DECISIONS.md"

    def use_real(decision_id):
        def mutate(args, spec):
            args[args.index("--decisions") + 1] = str(real)
            args[args.index("--decision-id") + 1] = decision_id
        return mutate

    for decision_id in ("D-019", "D-024"):
        sub = tmp_path / decision_id
        sub.mkdir()
        assert "launch_not_authorized" in _refusal(monkeypatch, sub, use_real(decision_id))


@pytest.mark.parametrize("route", ["hydra_only", "h1_resolved_toml"])
def test_g1_launch_body_rederives_build_launch_hash(tmp_path, route):
    from gate.s2_identity import identity_hash

    spec, registration = launch_inputs(tmp_path, tmp_path / "tree", tmp_path / "m.json",
                                       route=route)
    assert identity_hash(host_launch.launch_body(spec)) == spec["launch_hash"]
    check = host_launch.verify_launch_authority(
        spec, registration, launch_decisions(tmp_path, spec, registration).read_bytes(),
        decision_id="D-032")
    assert check["authorized"] is True and check["issues"] == []
    assert check["registration_sha256"] == identity_hash(registration)
    assert check["authenticity"] == "supplied_not_authenticated"
    blocked = dict(spec, status="blocked")
    assert [i["code"] for i in host_launch.verify_launch_authority(
        blocked, registration, b"", decision_id="D-032")["issues"]] == ["launch_spec_not_passed"]


def test_g1_capture_tree_carries_the_stdlib_authority_verifier():
    assert "gate/s2_authority.py" in host_launch.CAPTURE_FILES
    source = (REPO_ROOT / "gate" / "s2_authority.py").read_text()
    imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert imports == ["import hashlib", "import re"]


def test_planted_bytecode_and_pycache_named_files_are_rejected(tmp_path):
    """A8-H-01: an unchecked hash-based pyc would run instead of the verified source."""
    import importlib.util
    import marshal

    tree = tmp_path / "tree"
    manifest = host_launch.stage_capture_tree(REPO_ROOT, tree, scope="synthetic")
    target = tree / "teacher_rl" / "capture" / "sink.py"
    code = compile("PLANTED = True\n", str(target), "exec")
    header = (importlib.util.MAGIC_NUMBER + (0b01).to_bytes(4, "little")
              + importlib.util.source_hash(target.read_bytes()))
    cache = Path(importlib.util.cache_from_source(str(target)))
    cache.parent.mkdir(parents=True)
    cache.write_bytes(header + marshal.dumps(code))
    report = host_launch.verify_capture_tree(tree, manifest)
    assert report["status"] == "blocked"
    assert any("__pycache__" in issue for issue in report["issues"])
    tree2 = tmp_path / "tree2"
    manifest2 = host_launch.stage_capture_tree(REPO_ROOT, tree2, scope="synthetic")
    (tree2 / "teacher_rl" / "capture" / "__pycache__x.py").write_text("print('unlisted')\n")
    assert host_launch.verify_capture_tree(tree2, manifest2)["status"] == "blocked"


# ------------------------------------------------------------------- import discipline


@pytest.mark.parametrize("module", ["teacher_rl/capture/__init__.py",
                                    "teacher_rl/capture/entrypoint.py",
                                    "teacher_rl/capture/host_launch.py",
                                    "teacher_rl/capture/sink.py"])
def test_capture_modules_import_only_stdlib_and_capture_siblings(module):
    import sys

    tree = ast.parse((REPO_ROOT / module).read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            assert top in sys.stdlib_module_names or name.startswith("teacher_rl.capture"), (
                module, name)


def test_staged_tree_lists_every_capture_module():
    on_disk = sorted(str(p.relative_to(REPO_ROOT)) for p in
                     (REPO_ROOT / "teacher_rl" / "capture").glob("*.py"))
    assert sorted(f for f in host_launch.CAPTURE_FILES if "/capture/" in f) == on_disk
    assert os.sep == "/"  # POSIX paths in manifests


def test_read_stream_rejects_a_rehashed_record_with_a_forged_link(tmp_path):
    s = make_sink(tmp_path)
    s.emit("probe", {"value": 1})
    s.emit("probe", {"value": 2})
    s.close()
    lines = s.path.read_bytes().splitlines()
    record = json.loads(lines[2])
    body = {"seq": record["seq"], "prev": "0" * 64, "event": record["event"]}
    forged = {**body, "hash": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
              "time": record["time"]}
    lines[2] = canonical_json_bytes(forged)
    s.path.write_bytes(b"\n".join(lines) + b"\n")
    report = read_stream(s.path, max_bytes=10**6)
    assert [i["code"] for i in report["issues"]] == ["chain_broken"]
    assert len(report["events"]) == 2


def test_concurrent_pinned_worker_still_yields_a_consistent_ledger(monkeypatch, tmp_path):
    fakes = install_fakes(monkeypatch, RewardScript(()))
    capture, s = new_capture(tmp_path)
    capture.install()
    servicer = fakes.servicer_cls()
    stub = fakes.backend.RuntimeServiceStub(FakeChannel(servicer, ["ok"] * 12))
    driver = SimpleNamespace(servicer=servicer,
                             topology_endpoint=SimpleNamespace(host="localhost", port=5000))
    worker = fakes.worker.StreamingRolloutWorker(
        alpasim_runtime_stub=stub, driver_server=driver, simulation_timeout_s=5.0,
        reward_config=object(), max_concurrent_rollouts=4, rollouts_per_payload=3,
        scene_id_resolver=lambda payload: SCENE, max_scene_retries=3)
    rollout = fakes.backend.AlpagymRollout()
    rollout._worker = worker
    rollout.rollout_generation([SimpleNamespace(prompt_idx=i) for i in range(4)], None, None,
                               current_weight_version=2)
    worker.shutdown()
    s.close()
    streams, issues = streams_from_reports([read_stream(s.path, max_bytes=10**7)])
    ledger = build_attempt_ledger(streams, scope="synthetic")
    assert issues == [] and ledger["status"] == "passed", ledger["issues"]
    assert ledger["counts"]["succeeded"] == 12
    assert sorted(g["prompt_idx"] for g in ledger["groups"]) == [0, 1, 2, 3]
    assert all(g["complete"] and len(g["members"]) == 3 for g in ledger["groups"])


def test_ledger_inputs_bind_the_verified_stream_chain_head(tmp_path):
    s = make_sink(tmp_path)
    s.emit("probe", {"value": 1})
    s.close()
    report = read_stream(s.path, max_bytes=10**6)
    last = json.loads(s.path.read_bytes().splitlines()[-1])
    assert report["chain_head"] == last["hash"]
    streams, _ = streams_from_reports([report])
    ledger = build_attempt_ledger(streams, scope="synthetic")
    assert ledger["inputs"] == [{"artifact_kind": "s4_capture_stream",
                                 "stream_id": "rollout-test", "chain_head": last["hash"],
                                 "reader_status": "passed"}]
    assert ledger["status"] == "not_run"  # a probe-only stream attempted nothing (A8-R-02)
    manifest = host_launch.stage_capture_tree(REPO_ROOT, tmp_path / "tree", scope="synthetic")
    assert {"artifact_kind", "schema_version", "scope", "producer", "inputs", "units",
            "issues"} <= set(manifest)
    with pytest.raises(host_launch.LaunchRefused):
        host_launch.stage_capture_tree(REPO_ROOT, tmp_path / "tree2", scope="prod")


def test_failure_outcome_is_taken_under_the_upstream_lock(monkeypatch, tmp_path):
    """A8-C-01: a sibling exhausting the budget between upstream's lock release and the capture
    after-hook must not turn a failure that enqueued a retry into "dropped"."""
    fakes = install_fakes(monkeypatch, RewardScript(()))
    worker_cls = fakes.worker.StreamingRolloutWorker
    pinned_failed = worker_cls._on_rollout_failed
    state = fakes.worker.SharedPayloadState(payload=SimpleNamespace(prompt_idx=7), n_target=2,
                                            future=Future(), retries_left=1)
    job_a = fakes.worker._RolloutJob(shared_payload_state=state, session_uuid="a" * 32,
                                     scene_id=SCENE)
    job_b = fakes.worker._RolloutJob(shared_payload_state=state, session_uuid="b" * 32,
                                     scene_id=SCENE)

    interleave = [True]

    def _on_rollout_failed(self, rollout_job, exc):  # the pinned signature
        pinned_failed(self, rollout_job, exc)
        if rollout_job is job_a and interleave:
            interleave.clear()
            # Legal interleaving: after A's lock section, another simulate-pool thread runs
            # its whole (wrapped) failure before A's capture after-hook executes.
            sibling = threading.Thread(target=self._on_rollout_failed,
                                       args=(job_b, RuntimeError("sibling failure")))
            sibling.start()
            sibling.join()

    worker_cls._on_rollout_failed = _on_rollout_failed
    worker = object.__new__(worker_cls)  # no threads: only what _on_rollout_failed touches
    worker._lock = threading.Lock()
    worker._rollout_job_queue = queue.PriorityQueue()
    worker._rollout_job_seq = itertools.count()
    worker._active_payload_states = {7: state}
    capture, s = new_capture(tmp_path)
    capture.install([t for t in WRAP_TARGETS if t.target_id == "rollout_failed"])
    worker._on_rollout_failed(job_a, RuntimeError("A failed"))
    worker._on_rollout_failed(job_a, RuntimeError("late duplicate"))  # after the drop
    queued = []
    while not worker._rollout_job_queue.empty():
        queued.append(worker._rollout_job_queue.get_nowait())
    assert [(priority, job.attempts) for priority, _seq, job in queued] == [(0, 1)]
    s.close()
    failed = [e for e in read_stream(s.path, max_bytes=10**6)["events"]
              if e["kind"] == "attempt_failed"]
    outcomes = sorted((e["session_uuid"][0], e["outcome"], e["retries_left_after"])
                      for e in failed)
    assert outcomes == [("a", "ignored_after_drop", -1), ("a", "retry", 0),
                        ("b", "dropped", -1)]
    with worker._lock:  # the probe still behaves as the original mutex
        assert worker._lock.locked()
    assert not worker._lock.locked()


# ------------------------------------------------- A8-C-03: asynchronous safetensors export

BOTH_EXPORT_FILES = [
    {"path": "config.json", "status": "hashed", "bytes": 2,
     "sha256": hashlib.sha256(b"{}").hexdigest()},
    {"path": "model.safetensors", "status": "hashed", "bytes": 9,
     "sha256": hashlib.sha256(b"weights-5").hexdigest()}]


def export_trainer(fakes, tmp_path, **kwargs):
    return fakes.trainer.AlpagymGRPOTrainer(
        trainer_config(tmp_path), None, model=FakeModel(), optimizers=FakeOptimizers(),
        lr_schedulers=SimpleNamespace(get_last_lr=lambda: [0.0001], last_epoch=0),
        grad_norms=[0.0] * 4, **kwargs)


def export_events(stream_path):
    events = read_stream(stream_path, max_bytes=10**7)["events"]
    returned = [e for e in events if e["kind"] == "checkpoint_save" and e["phase"] == "returned"]
    settled = [e for e in events if e["kind"] == "checkpoint_export_settled"]
    return events, returned, settled


def test_export_written_by_the_upload_thread_is_hashed_after_the_join(monkeypatch, tmp_path):
    """A8-C-03: the pinned writer thread is joined before the export is hashed."""
    fakes = install_fakes(monkeypatch, RewardScript(()))
    capture, s = new_capture(tmp_path)
    capture.install()
    trainer = export_trainer(fakes, tmp_path)
    trainer.step_training([], 5, 10, 0, None, True, True)
    assert not trainer.upload_thread.is_alive()
    s.close()
    _events, returned, settled = export_events(s.path)
    export = returned[0]["export"]
    assert export["status"] == "hashed" and export["synchronized"] is True
    assert export["files"] == {"count": 2, "values": BOTH_EXPORT_FILES}
    assert settled == []


def test_partial_async_export_is_never_labelled_hashed(monkeypatch, tmp_path):
    """A8-C-03 (reverifier probe R11b): pinned ``export_safetensors`` returns while its daemon
    thread is still writing; one complete file must not make the directory "hashed"."""
    monkeypatch.setattr(entrypoint, "EXPORT_JOIN_TIMEOUT_S", 0.05, raising=False)
    fakes = install_fakes(monkeypatch, RewardScript(()))
    capture, s = new_capture(tmp_path)
    capture.install()
    gate = threading.Event()
    trainer = export_trainer(fakes, tmp_path, export_gate=gate)
    try:
        trainer.step_training([], 5, 10, 0, None, True, True)
        writer = trainer.upload_thread
        assert writer.name == "save_and_upload_safetensors" and writer.is_alive()
        trainer.step_training([], 6, 10, 0, None, True)  # writer still running: still pending
    finally:
        gate.set()
        trainer.upload_thread.join(5)
    trainer.step_training([], 7, 10, 0, None, True)
    s.close()
    events, returned, settled = export_events(s.path)
    export_dir = tmp_path / "outputs" / "safetensors" / "step_5"
    assert returned[0]["export"] == {
        "path": str(export_dir), "status": "pending_async_write",
        "reason": "writer_thread_alive_after_join",
        "writer_thread": "save_and_upload_safetensors", "files": None}
    assert len(settled) == 1
    assert {k: settled[0][k] for k in ("save_call", "current_step", "trigger", "final")} == {
        "save_call": returned[0]["call"], "current_step": 5, "trigger": "step_start",
        "final": False}
    assert settled[0]["export"]["status"] == "hashed"
    assert settled[0]["export"]["synchronized"] is True
    assert settled[0]["export"]["files"] == {"count": 2, "values": BOTH_EXPORT_FILES}
    started = [e for e in events if e["kind"] == "trainer_step_started"]
    finished = [e for e in events if e["kind"] == "trainer_step_finished"]
    # re-hashed at the start of step 7, after its trainer_step_started record (A8R2)
    assert started[-1]["seq"] < settled[0]["seq"] < finished[-1]["seq"]


def test_export_without_a_writer_thread_handle_stays_pending(monkeypatch, tmp_path):
    """A8-C-03: when no new ``upload_thread`` appears the write cannot be synchronized."""
    fakes = install_fakes(monkeypatch, RewardScript(()))
    capture, s = new_capture(tmp_path)
    capture.install()
    trainer = export_trainer(fakes, tmp_path, export_async=False)
    trainer.step_training([], 5, 10, 0, None, True, True)
    trainer.step_training([], 6, 10, 0, None, True)
    s.close()
    _events, returned, settled = export_events(s.path)
    assert returned[0]["export"]["status"] == "pending_async_write"
    assert returned[0]["export"]["reason"] == "writer_thread_not_found"
    assert returned[0]["export"]["writer_thread"] is None
    assert returned[0]["export"]["files"] is None
    assert len(settled) == 1 and settled[0]["trigger"] == "step_start"
    assert settled[0]["export"]["status"] == "hashed"
    assert settled[0]["export"]["synchronized"] is False
    assert settled[0]["export"]["files"] == {"count": 2, "values": BOTH_EXPORT_FILES}


def test_pending_exports_are_settled_at_shutdown(monkeypatch, tmp_path):
    """A8-C-03: ``main`` re-hashes pending exports before closing its stream; a writer still
    running at shutdown is recorded as pending, never hashed."""
    monkeypatch.setattr(entrypoint, "EXPORT_JOIN_TIMEOUT_S", 0.05, raising=False)
    fakes = install_fakes(monkeypatch, RewardScript(()))
    run_dir = tmp_path / "run"
    toml = write_cosmos_toml(tmp_path, run_dir / "resolved_config.yaml")
    finished, running = threading.Event(), threading.Event()
    writers = []

    def upstream_main(argv=None):
        trainer = export_trainer(fakes, tmp_path, export_gate=finished)
        trainer.step_training([], 5, 10, 0, None, True, True)
        writers.append(trainer.upload_thread)
        trainer.export_gate = running
        trainer.step_training([], 6, 10, 0, None, True, True)
        writers.append(trainer.upload_thread)
        finished.set()
        writers[0].join(5)
        return "upstream-done"

    sys_modules()["alpagym_runtime.cosmos.entrypoint"].main = upstream_main
    environ = {**worker_env(tmp_path), "COSMOS_ROLE": "Policy"}
    try:
        assert entrypoint.main(["--config", str(toml)], environ=environ) == "upstream-done"
    finally:
        running.set()
        for writer in writers:
            writer.join(5)
    (stream_path,) = (run_dir / "physloop_capture").glob("policy-*.jsonl")
    _events, returned, settled = export_events(stream_path)
    assert [e["export"]["status"] for e in returned] == ["pending_async_write"] * 2
    assert [(e["current_step"], e["trigger"], e["final"]) for e in settled] == [
        (5, "shutdown", True), (6, "shutdown", True)]
    assert settled[0]["export"]["status"] == "hashed"
    assert settled[0]["export"]["files"] == {"count": 2, "values": BOTH_EXPORT_FILES}
    assert settled[1]["export"] == {
        "path": str(tmp_path / "outputs" / "safetensors" / "step_6"),
        "status": "pending_async_write", "reason": "writer_thread_alive_at_shutdown",
        "writer_thread": "save_and_upload_safetensors", "files": None}


def test_failed_step_start_settle_keeps_the_step_record_and_the_pending_export(
        monkeypatch, tmp_path):
    """A8R2 (reverifier probe_c03 s6): a manifest failure during the step-start settle (a
    file pruned mid-walk) must not suppress ``trainer_step_started`` or the ``pre_step``
    snapshot, and must not discard the pending export: it is recorded and settled later."""
    monkeypatch.setattr(entrypoint, "EXPORT_JOIN_TIMEOUT_S", 0.05, raising=False)
    fakes = install_fakes(monkeypatch, RewardScript(()))
    capture, s = new_capture(tmp_path)  # CONFIG takes init_and_each_step snapshots
    capture.install()
    gate = threading.Event()
    trainer = export_trainer(fakes, tmp_path, export_gate=gate)
    try:
        trainer.step_training([], 5, 10, 0, None, True, True)  # pending: writer gated
    finally:
        gate.set()
        trainer.upload_thread.join(5)
    real, calls = entrypoint._directory_manifest, {"n": 0}

    def flaky(directory, limit):
        calls["n"] += 1
        if calls["n"] == 1:
            raise FileNotFoundError(2, "No such file or directory", str(directory))
        return real(directory, limit)

    monkeypatch.setattr(entrypoint, "_directory_manifest", flaky)
    trainer.step_training([], 6, 10, 0, None, True)
    assert len(capture._pending_exports) == 1  # kept for a later settle
    capture.settle_exports("shutdown")
    s.close()
    events, returned, settled = export_events(s.path)
    export_dir = tmp_path / "outputs" / "safetensors" / "step_5"
    started = [e for e in events if e["kind"] == "trainer_step_started"]
    assert [e["current_step"] for e in started] == [5, 6]
    snapshots = [e for e in events if e["kind"] == "parameter_snapshot"
                 and e.get("phase") == "pre_step"]
    assert len(snapshots) == 2
    assert not [e for e in events if e["kind"] == "capture_extract_error"]
    (error,) = [e for e in events if e["kind"] == "checkpoint_export_settle_error"]
    assert {k: error[k] for k in ("save_call", "current_step", "path", "trigger", "final",
                                  "error_type")} == {
        "save_call": returned[0]["call"], "current_step": 5, "path": str(export_dir),
        "trigger": "step_start", "final": False, "error_type": "FileNotFoundError"}
    assert started[1]["seq"] < error["seq"] < snapshots[1]["seq"]
    assert [(e["current_step"], e["trigger"], e["final"]) for e in settled] == [
        (5, "shutdown", True)]
    assert settled[0]["export"]["status"] == "hashed"
    assert settled[0]["export"]["files"] == {"count": 2, "values": BOTH_EXPORT_FILES}
    assert capture._pending_exports == []


def test_settle_isolates_each_export_and_restores_on_a_fatal_sink(tmp_path):
    """A8R2: one export failing to settle neither blocks the others nor drops itself; an
    exception escaping the settle (a fatal sink failure) restores every unsettled export."""
    capture, s = new_capture(tmp_path)
    done = threading.Thread(target=lambda: None, name=entrypoint.EXPORT_THREAD_NAME)
    done.start()
    done.join(5)
    good = tmp_path / "good"
    good.mkdir()
    (good / "model.safetensors").write_bytes(b"weights-5")
    items = [{"call": 1, "step": 5, "path": tmp_path / "bad", "thread": done},
             {"call": 2, "step": 6, "path": good, "thread": done}]
    real = entrypoint._directory_manifest

    def flaky(directory, limit):
        if Path(directory).name == "bad":
            raise FileNotFoundError(2, "No such file or directory", str(directory))
        return real(directory, limit)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(entrypoint, "_directory_manifest", flaky)
        capture._pending_exports = list(items)
        capture.settle_exports("step_start")
        assert capture._pending_exports == items[:1]  # the failed export stays pending
        capture._pending_exports = list(items)
        real_emit = capture.emit

        def fatal_emit(kind, payload):
            if kind == "checkpoint_export_settle_error":
                raise entrypoint.CaptureFatal("sink failed")
            return real_emit(kind, payload)

        mp.setattr(capture, "emit", fatal_emit)
        with pytest.raises(entrypoint.CaptureFatal):
            capture.settle_exports("step_start")
        assert capture._pending_exports == items  # nothing discarded
    s.close()
    events = read_stream(s.path, max_bytes=10**7)["events"]
    assert [(e["kind"], e["current_step"]) for e in events
            if e["kind"].startswith("checkpoint_export")] == [
        ("checkpoint_export_settle_error", 5), ("checkpoint_export_settled", 6)]


# ------------------------------------------------- A8-H-02: credential name filter

CREDENTIAL_ENV_NAMES = [
    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "WANDB_API_KEY",
    "NGC_API_KEY", "VAST_API_KEY", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "GITHUB_TOKEN",
    "GH_TOKEN", "hf_token", "OPENAI_API_KEY", "DB_PASSWORD", "APP_SECRET", "DJANGO_SECRET_KEY",
    "SERVICE_ACCESS_TOKEN", "REGISTRY_AUTH_TOKEN",
    # A8R2: Vast session credentials (tests/test_vast_s2_host_probe.py) and common providers
    "JUPYTER_TOKEN", "OPEN_BUTTON_TOKEN", "AWS_ACCESS_KEY_ID", "SSH_PRIVATE_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS", "DB_PASSWD", "HUGGINGFACEHUB_API_TOKEN", "HF_API_TOKEN",
    "NPM_TOKEN", "MLFLOW_TRACKING_TOKEN", "SOME_SERVICE_TOKEN", "DEPLOY_PRIVATE_KEY",
    "S3_KEY_ID", "CLOUD_CREDENTIALS", "ldap_passwd"]
NON_CREDENTIAL_ENV_NAMES = [
    "TOKENIZERS_PARALLELISM", "VLLM_MAX_TOKENS", "MAX_NEW_TOKENS", "TOKEN_BUDGET",
    "SECRETS_DIR", "HF_HUB_OFFLINE", "HF_HOME", "PATH", "PHYSLOOP_WORKER_CREDENTIALS",
    "PHYSLOOP_CAPTURE_CONFIG", "MAX_TOKENS", "TOKEN", "TOKENIZER_PATH", "KEY_ID_PREFIX",
    "CREDENTIALS_DIR", "PRIVATE_KEY_PATH", "PASSWD_FILE"]


@pytest.mark.parametrize("mode", ["strip_offline", "inherit_disclosed"])
def test_worker_credential_filter_matches_credentials_only(mode):
    """A8-H-02: exact credential names plus conservative suffixes; tokenizer and max-token
    settings are configuration, not credentials, and stay in the worker environment."""
    environ = {name: "value" for name in CREDENTIAL_ENV_NAMES + NON_CREDENTIAL_ENV_NAMES}
    report = entrypoint.apply_worker_credentials(environ, mode)
    names = report["removed_names" if mode == "strip_offline" else "present_credential_names"]
    assert names == sorted(CREDENTIAL_ENV_NAMES)
    for name in NON_CREDENTIAL_ENV_NAMES:  # strip_offline sets HF_HUB_OFFLINE=1 by design
        expected = "1" if mode == "strip_offline" and name == "HF_HUB_OFFLINE" else "value"
        assert environ[name] == expected, name
    if mode == "strip_offline":
        assert not set(CREDENTIAL_ENV_NAMES) & set(environ)


# ------------------------------------------------------------------- F5 snapshot schema 2
# Rule-11 known answers: bytes are hand-packed little-endian values and every expected hash is
# hashlib over those literal bytes (never the module under test). torch is a sys.modules fake.

def sha(data):
    return hashlib.sha256(data).hexdigest()


class FakeDtype:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"torch.{self.name}"


class FakeTensor:
    """A plain strided CPU tensor whose rule-11 bytes are exactly ``raw``."""

    layout = "strided"
    is_quantized = False

    def __init__(self, raw, dtype, shape, requires_grad=False):
        self.raw, self.dtype = bytes(raw), FakeDtype(dtype)
        self.shape, self.requires_grad = tuple(shape), requires_grad
        self.device = SimpleNamespace(type="cpu")

    def detach(self):
        return self

    def cpu(self):
        return self

    def resolve_conj(self):
        return self

    def resolve_neg(self):
        return self

    def contiguous(self):
        return self

    def reshape(self, *shape):
        assert shape == (-1,)
        return self

    def view(self, dtype):
        assert str(dtype) == "torch.uint8"
        return SimpleNamespace(numpy=lambda: self.raw)

    def item(self):
        assert str(self.dtype) == "torch.float32" and len(self.raw) == 4
        return struct.unpack("<f", self.raw)[0]


def f32(*values, shape=None, requires_grad=False):
    return FakeTensor(struct.pack(f"<{len(values)}f", *values), "float32",
                      (len(values),) if shape is None else shape, requires_grad)


class FakeShard:
    def __init__(self, dim):
        self.dim = dim

    def is_shard(self):
        return True

    def is_replicate(self):
        return False


class FakeReplicate:
    def is_shard(self):
        return False

    def is_replicate(self):
        return True


class FakePartial:
    def is_shard(self):
        return False

    def is_replicate(self):
        return False


class FakeStridedShard(FakeShard):
    """torch's ``_StridedShard`` shape: a ``Shard`` subclass whose rows are not contiguous."""

    def __init__(self, dim, split_factor):
        super().__init__(dim)
        self.split_factor = split_factor


class LookalikeShard:
    """Quacks like ``Shard`` (``is_shard`` True) but is not the registered type."""

    def __init__(self, dim):
        self.dim = dim

    def is_shard(self):
        return True

    def is_replicate(self):
        return False


class FakeDTensor:
    """DTensor shape: global ``shape``, ``placements``, ``device_mesh``, ``to_local()``."""

    def __init__(self, local, global_shape, placements, coordinate, mesh_sizes,
                 requires_grad=True):
        self._local, self.shape = local, tuple(global_shape)
        self.placements, self.requires_grad = tuple(placements), requires_grad
        self.device_mesh = SimpleNamespace(get_coordinate=lambda: coordinate,
                                           size=lambda dim: mesh_sizes[dim])

    def detach(self):
        return self

    def to_local(self):
        return self._local


class TensorModel:
    def __init__(self, params, buffers, modules=(("", True), ("vlm", False))):
        self._params, self._buffers, self._modules = list(params), list(buffers), modules

    def named_parameters(self):
        return list(self._params)

    def named_buffers(self):
        return list(self._buffers)

    def named_modules(self):
        return [(name, SimpleNamespace(training=flag)) for name, flag in self._modules]


class FakeAdamW:
    """One param group; a real step sets step+1, exp_avg 0.25 and subtracts 0.5."""

    def __init__(self, params, lr):
        self.param_groups = [{"params": list(params), "lr": lr}]
        self.state = {}

    def step(self):
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state.setdefault(param, {})
                count = int(state["step"].item()) + 1 if "step" in state else 1
                state["step"] = f32(float(count), shape=())
                state["exp_avg"] = f32(0.25, 0.25)
                values = struct.unpack("<2f", param.raw)
                param.raw = struct.pack("<2f", *(v - 0.5 for v in values))

    def zero_grad(self):
        return None


class FakeOptimizersContainer:
    """Cosmos OptimizersContainer shape (optm/__init__.py:97-207): iterates inner optimizers."""

    def __init__(self, inner):
        self.optimizers = [list(inner)]

    def __iter__(self):
        return itertools.chain(*self.optimizers)

    def step(self):
        for optimizer in self:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self:
            optimizer.zero_grad()


class FakeLambdaLR:
    def __init__(self, lr):
        self.last_epoch, self.lr = 0, lr

    def get_last_lr(self):
        return [self.lr]

    def step(self):
        self.last_epoch += 1


class FakeSchedulersContainer:
    """Cosmos LRSchedulersContainer shape (optm/__init__.py:448-471)."""

    def __init__(self, inner):
        self.schedulers = [list(inner)]

    def __iter__(self):
        return itertools.chain(*self.schedulers)

    def step(self):
        for scheduler in self:
            scheduler.step()

    def get_last_lr(self, model_part_idx=0):
        return self.schedulers[model_part_idx][0].get_last_lr()


@pytest.fixture
def fake_torch(monkeypatch):
    module = types.ModuleType("torch")
    module.uint8, module.strided = FakeDtype("uint8"), "strided"
    monkeypatch.setitem(sys_modules(), "torch", module)
    # the exact placement types shard_layout compares against (type identity, no subclasses)
    placements = types.ModuleType("torch.distributed.tensor.placement_types")
    placements.Shard, placements.Replicate = FakeShard, FakeReplicate
    for name in ("torch.distributed", "torch.distributed.tensor"):
        monkeypatch.setitem(sys_modules(), name, types.ModuleType(name))
    monkeypatch.setitem(sys_modules(), placements.__name__, placements)
    return module


BF16_ONE_TWO = b"\x80\x3f\x00\x40"  # bfloat16 1.0 = 0x3f80, 2.0 = 0x4000, little-endian


def policy(state=None):
    w = f32(1.0, 2.0, requires_grad=True)
    model = TensorModel([("expert.w", w),
                         ("vlm.lm_head.weight", FakeTensor(BF16_ONE_TWO, "bfloat16", (2,)))],
                        [("action_in_proj.freqs", f32(0.5))])
    adam = FakeAdamW([w], lr=0.0001)
    if state is not None:
        adam.state[w] = state
    return model, FakeOptimizersContainer([adam]), w


def test_rule11_dtype_table_is_the_tensor_manifest_table():
    from teacher_rl import params

    assert set(entrypoint.RULE11_DTYPES.values()) == set(DTYPE_ITEMSIZE)
    assert entrypoint.RULE11_DTYPES == params._DTYPE_NAMES
    assert entrypoint.rule11_dtype(FakeDtype("bfloat16")) == "BF16"
    with pytest.raises(ValueError, match="outside rule 11"):
        entrypoint.rule11_dtype(FakeDtype("complex64"))


def test_snapshot_v2_hashes_rule11_bytes_known_answer(fake_torch):
    model, optimizers, _w = policy({"step": f32(3.0, shape=()), "exp_avg": f32(0.25, 0.25)})
    snapshot = entrypoint.build_parameter_snapshot(model, optimizers, phase="pre_step",
                                                   trainer_step=5, hashing="sha256_local")
    full = {"hash_scope": "full", "shard": None}
    assert snapshot == {
        "schema_version": 2, "phase": "pre_step", "trainer_step": 5,
        "tensors": {
            "action_in_proj.freqs": {"kind": "buffer", "dtype": "F32", "shape": [1], "numel": 1,
                                     "requires_grad": False,
                                     "sha256": sha(struct.pack("<f", 0.5)), **full},
            "expert.w": {"kind": "param", "dtype": "F32", "shape": [2], "numel": 2,
                         "requires_grad": True, "sha256": sha(struct.pack("<2f", 1.0, 2.0)),
                         **full},
            "vlm.lm_head.weight": {"kind": "param", "dtype": "BF16", "shape": [2], "numel": 2,
                                   "requires_grad": False, "sha256": sha(BF16_ONE_TWO), **full}},
        "training_flags": {"": True, "vlm": False},
        "optimizer_state": {"expert.w": {"exp_avg": sha(struct.pack("<2f", 0.25, 0.25)),
                                         "step": sha(struct.pack("<f", 3.0))}},
        "step_counters": {"expert.w": 3}, "applied_lr": [0.0001]}
    # the consumer accepts exactly this shape (no drift between producer and adapter)
    event = {"kind": "parameter_snapshot", "seq": 3, **snapshot}
    assert records.snapshot_digest(event) == {
        "parameters": {"expert.w": sha(struct.pack("<2f", 1.0, 2.0)),
                       "vlm.lm_head.weight": sha(BF16_ONE_TWO)},
        "optimizer_state": {"expert.w": {"exp_avg": sha(struct.pack("<2f", 0.25, 0.25)),
                                         "step": sha(struct.pack("<f", 3.0))}},
        "step_counters": {"expert.w": 3}}
    # rule 11: the loaded-tensor hash equals the hash of its stored safetensors byte range
    header = json.dumps({"expert.w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}},
                        separators=(",", ":")).encode()
    stored = len(header).to_bytes(8, "little") + header + struct.pack("<2f", 1.0, 2.0)
    manifest = build_manifest({"model.safetensors": stored}, scope="synthetic")
    assert manifest["tensors"]["expert.w"]["sha256"] == snapshot["tensors"]["expert.w"]["sha256"]


def test_metadata_only_snapshot_records_nulls_and_never_imports_torch(monkeypatch):
    monkeypatch.setitem(sys_modules(), "torch", None)  # any torch import would now raise
    model, optimizers, _w = policy({"step": f32(3.0, shape=()), "exp_avg": f32(0.25, 0.25)})
    snapshot = entrypoint.build_parameter_snapshot(model, optimizers, phase="init",
                                                   trainer_step=None, hashing="metadata_only")
    assert {(e["sha256"], e["hash_scope"]) for e in snapshot["tensors"].values()} == {
        (None, None)}
    assert snapshot["optimizer_state"] == {"expert.w": {"exp_avg": None, "step": None}}
    assert snapshot["step_counters"] == {"expert.w": 3}
    with pytest.raises(records.RecordsError, match="snapshot_unhashed"):
        records.snapshot_digest({"kind": "parameter_snapshot", "seq": 0, **snapshot})


def test_member_without_state_maps_to_empty_and_zero(fake_torch):
    model, optimizers, _w = policy()
    snapshot = entrypoint.build_parameter_snapshot(model, optimizers, phase="post_step",
                                                   trainer_step=0, hashing="sha256_local")
    assert (snapshot["optimizer_state"], snapshot["step_counters"]) == ({"expert.w": {}},
                                                                         {"expert.w": 0})


def test_dtensor_local_shard_offsets_follow_the_chunk_rule(fake_torch):
    local = f32(1.0, 2.0, 3.0, 4.0, shape=(2, 2))
    # global 5 rows over a mesh of 2: chunk 3, rank 1 holds rows 3..4
    param = FakeDTensor(local, (5, 2), [FakeShard(0)], [1], [2])
    entry = entrypoint.tensor_entry(param, "param", "sha256_local")
    assert entry == {"kind": "param", "dtype": "F32", "shape": [2, 2], "numel": 4,
                     "requires_grad": True,
                     "sha256": sha(struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)),
                     "hash_scope": "local_shard",
                     "shard": {"global_shape": [5, 2], "offsets": [3, 0],
                               "placement": "[Shard(dim=0)]"}}
    # 2-D mesh (HSDP): Replicate x Shard(0); global 3 over 2: chunk 2, coordinate 1 -> row 2
    tail = FakeDTensor(f32(9.0), (3,), [FakeReplicate(), FakeShard(0)], [0, 1], [2, 2])
    assert entrypoint.tensor_entry(tail, "param", "sha256_local")["shard"] == {
        "global_shape": [3], "offsets": [2], "placement": "[Replicate(), Shard(dim=0)]"}
    # global 1 over 2: rank 1 holds nothing at offset 1 (torch _local_shard_size_and_offset)
    empty = FakeDTensor(FakeTensor(b"", "float32", (0,)), (1,), [FakeShard(-1)], [1], [2])
    assert entrypoint.tensor_entry(empty, "param", "sha256_local")["shard"]["offsets"] == [1]


@pytest.mark.parametrize("make", [
    lambda: FakeDTensor(f32(1.0, 2.0, 3.0), (5,), [FakeShard(0)], [1], [2]),  # wrong extent
    lambda: FakeDTensor(f32(1.0), (1,), [FakePartial()], [0], [2]),
    lambda: FakeDTensor(f32(1.0), (2,), [FakeShard(0)], None, [2]),  # rank not on the mesh
    lambda: FakeDTensor(f32(1.0), (2,), [FakeShard(0), FakeShard(0)], [0, 0], [2, 1]),
    lambda: FakeDTensor(f32(1.0, 2.0), (3,), [FakeReplicate()], [0], [2]),  # replicate != global
])
def test_unprovable_shard_layouts_raise(fake_torch, make):
    with pytest.raises(ValueError):
        entrypoint.tensor_entry(make(), "param", "sha256_local")


@pytest.mark.parametrize("placements", [
    [FakeStridedShard(0, split_factor=2), FakeReplicate()],  # lone strided shard (review F5-02)
    [FakeStridedShard(0, split_factor=1)],
    [LookalikeShard(0)],
])
def test_shard_subclasses_and_lookalikes_are_refused(fake_torch, placements):
    # torch gives a lone _StridedShard(0, split_factor=2) coordinate 0 of 2 the global rows
    # [0, 1, 4, 5] of 8, not the contiguous [0..3] the chunk rule would record as offset 0
    local = f32(*([0.0] * 12), shape=(4, 3))
    sizes = [2] * len(placements)
    param = FakeDTensor(local, (8, 3), placements, [0] * len(placements), sizes)
    with pytest.raises(ValueError, match="exactly Shard or Replicate"):
        entrypoint.tensor_entry(param, "param", "sha256_local")
    # the same layout with the exact Shard type is accepted (positive control)
    exact = FakeDTensor(local, (8, 3), [FakeShard(0)] + [FakeReplicate()] * (len(placements) - 1),
                        [0] * len(placements), sizes)
    assert entrypoint.tensor_entry(exact, "param", "sha256_local")["shard"]["offsets"] == [0, 0]


def test_snapshot_defects_raise_instead_of_emitting_a_partial_record(fake_torch):
    model, optimizers, w = policy()
    stranger = f32(0.0, 0.0, requires_grad=True)
    cases = [
        (TensorModel([("x", FakeTensor(b"\x00" * 8, "complex64", (1,)))], []), optimizers),
        (model, FakeOptimizersContainer([FakeAdamW([stranger], 0.1)])),  # unnamed member
        (model, FakeOptimizersContainer([FakeAdamW([w], 0.1), FakeAdamW([w], 0.1)])),
        policy({"step": f32(2.5, shape=())})[:2],  # non-integral step count
        (model, None),
        (TensorModel([], []), optimizers),
    ]
    for case_model, case_optimizers in cases:
        with pytest.raises(ValueError):
            entrypoint.build_parameter_snapshot(case_model, case_optimizers, phase="pre_step",
                                                trainer_step=1, hashing="sha256_local")
    sharded_state = {"exp_avg": FakeDTensor(f32(1.0), (2,), [FakeShard(0)], [0], [2])}
    sharded_model, sharded_optimizers, _ = policy(sharded_state)
    with pytest.raises(ValueError, match="unsharded parameter"):
        entrypoint.build_parameter_snapshot(sharded_model, sharded_optimizers, phase="pre_step",
                                            trainer_step=1, hashing="sha256_local")
    with pytest.raises(ValueError, match="integral count"):
        entrypoint.build_parameter_snapshot(*policy({"step": f32(2.5, shape=())})[:2],
                                            phase="pre_step", trainer_step=1,
                                            hashing="sha256_local")
    for phase, step in (("init", 0), ("pre_step", None), ("during", 1), ("pre_step", -1)):
        with pytest.raises(ValueError):
            entrypoint.build_parameter_snapshot(model, optimizers, phase=phase, trainer_step=step,
                                                hashing="sha256_local")


def test_scheduler_state_reads_the_container_and_refuses_disagreement():
    container = FakeSchedulersContainer([FakeLambdaLR(0.0001), FakeLambdaLR(0.0002)])
    assert entrypoint.scheduler_state(container) == {"last_epoch": 0, "lrs": [0.0001, 0.0002]}
    container.step()
    assert entrypoint.scheduler_state(container)["last_epoch"] == 1
    container.schedulers[0][1].last_epoch = 7
    for bad in (container, None, FakeSchedulersContainer([]),
                SimpleNamespace(get_last_lr=lambda: [0.1])):
        with pytest.raises((ValueError, AttributeError)):
            entrypoint.scheduler_state(bad)


def test_trainer_settings_record_pinned_attributes_or_null():
    trainer = SimpleNamespace(_mini_batch=2, _grpo_optimization_iterations=1, _group_size=6,
                              _allowed_outdated_steps=0, _kl_beta=0.0)
    assert entrypoint.trainer_settings(trainer) == {
        "mini_batch": 2, "iterations": 1, "group_size": 6, "allowed_outdated_steps": 0,
        "kl_beta": 0.0}
    assert entrypoint.trainer_settings(SimpleNamespace(_mini_batch="2", _kl_beta=True)) == {
        "mini_batch": None, "iterations": None, "group_size": None,
        "allowed_outdated_steps": None, "kl_beta": None}


# A pinned-shaped trainer: _train_minibatch calls all_reduce_states (trainer.py:418-422) and
# step_training runs iterations x ceil(rows / min(mini_batch, rows)) minibatches, then
# lr_schedulers.step() once (trainer.py:195-211). ``order`` stands in for the unseeded randperm.
PINNED_SHAPED_TRAINER_SOURCE = '''
class AlpagymGRPOTrainer:
    def __init__(self, config, parallel_dims, **kwargs):
        self.config = config
        self.model = kwargs["model"]
        self.optimizers = kwargs["optimizers"]
        self.lr_schedulers = kwargs["lr_schedulers"]
        self.grad_norms = list(kwargs["grad_norms"])
        self.order = kwargs["order"]
        self._mini_batch = kwargs["mini_batch"]
        self._grpo_optimization_iterations = kwargs["iterations"]
        self._group_size = 2
        self._kl_beta = 0.0
        self._allowed_outdated_steps = 0
        self.upload_thread = None

    def step_training(self, rollouts, current_step, total_steps, remain_samples_num,
                      inter_policy_nccl, is_master_replica, do_save_checkpoint=False):
        samples, advantages = self._prepare_training_data(rollouts)
        if not samples:
            raise ValueError("no trainable samples")
        size = min(self._mini_batch, len(samples))
        count = (len(samples) + size - 1) // size
        batches = 0
        for iteration in range(self._grpo_optimization_iterations):
            indices = self.order[iteration]
            for index in range(count):
                chosen = indices[index * size:(index + 1) * size]
                self._train_minibatch([samples[i] for i in chosen],
                                      [advantages[i] for i in chosen], inter_policy_nccl)
                batches += 1
        self.lr_schedulers.step()
        return {"train/num_batches": batches,
                "train/learning_rate": self.lr_schedulers.get_last_lr()[0]}

    def _prepare_training_data(self, rollouts):
        samples, advantages = [], []
        for rollout in rollouts:
            for old, pad in rollout.steps:
                signal = SimpleNamespace(old_logprobs=[old], is_padding=[pad])
                samples.append(SimpleNamespace(training_signal=signal))
                advantages.append(0.0 if pad else float(rollout.advantage))
        return samples, advantages

    def _forward_with_reference(self, model_inputs):
        return [-1.0] * model_inputs["rows"], None

    def _train_minibatch(self, minibatch_samples, minibatch_advantages, inter_policy_nccl):
        self._forward_with_reference({"rows": len(minibatch_samples)})
        grad_norm = self.all_reduce_states(inter_policy_nccl)
        return (0.1, 0.0, 1.0, 1.0, 0.0, grad_norm)

    def all_reduce_states(self, inter_policy_nccl):
        grad_norm = self.grad_norms.pop(0)
        if grad_norm != 0.0:
            self.optimizers.step()
        self.optimizers.zero_grad()
        return grad_norm

    def _save_checkpoint(self, current_step, total_steps, remain_samples_num):
        return None
'''


def pinned_shaped_trainer(fakes):
    exec(PINNED_SHAPED_TRAINER_SOURCE, fakes.trainer.__dict__)  # noqa: S102 - local fake source
    return fakes.trainer.AlpagymGRPOTrainer


def test_pinned_shaped_trainer_signatures_match_pinned_source(monkeypatch):
    trainer_cls = pinned_shaped_trainer(install_fakes(monkeypatch, RewardScript(())))
    for name in ("__init__", "step_training", "_prepare_training_data",
                 "_forward_with_reference", "_train_minibatch", "all_reduce_states",
                 "_save_checkpoint"):
        node = find_def(pinned_tree(f"{_G_RT}/cosmos/trainer.py"), f"AlpagymGRPOTrainer.{name}")
        assert runtime_signature(getattr(trainer_cls, name)) == ast_signature(node), name


def run_pinned_shaped_steps(monkeypatch, tmp_path, *, trainer_steps, grad_norms, mini_batch=2,
                            iterations=1, order=((1, 0),), hashing="sha256_local"):
    fakes = install_fakes(monkeypatch, RewardScript(()))
    trainer_cls = pinned_shaped_trainer(fakes)
    capture, s = new_capture(tmp_path, config={**CONFIG, "parameter_hashing": hashing})
    capture.install()
    model, optimizers, _w = policy()
    schedulers = FakeSchedulersContainer([FakeLambdaLR(0.0001)])
    trainer = trainer_cls(trainer_config(tmp_path), None, model=model, optimizers=optimizers,
                          lr_schedulers=schedulers, grad_norms=grad_norms, order=list(order),
                          mini_batch=mini_batch, iterations=iterations)
    rollout = SimpleNamespace(completion="/run/a.json", advantage=0.5,
                              steps=[(-1.0, False), (-2.0, True)])
    for step in trainer_steps:
        trainer.step_training([rollout], step, 10, 0, None, True)
    s.close()
    report = read_stream(s.path, max_bytes=10**7)
    assert report["status"] == "passed", report["issues"]
    return report


def test_pinned_shaped_capture_feeds_e20_and_e07_exactly(monkeypatch, tmp_path, fake_torch):
    """Skip at step 5 (grad 0.0), real step at step 6 (grad 2.0); one minibatch each."""
    report = run_pinned_shaped_steps(monkeypatch, tmp_path, trainer_steps=(5, 6),
                                     grad_norms=[0.0, 2.0])
    events = report["events"]
    started = [e for e in events if e["kind"] == "minibatch_started"]
    reduces = [e for e in events if e["kind"] == "optimizer_reduce"]
    assert [r["minibatch_call"] for r in reduces] == [m["call"] for m in started]
    (configured,) = [e for e in events if e["kind"] == "trainer_configured"]
    assert {k: configured[k] for k in ("mini_batch", "iterations", "group_size")} == {
        "mini_batch": 2, "iterations": 1, "group_size": 2}
    streams, _ = streams_from_reports([report])
    built = records.build_trainer_step_records(streams, scope="synthetic")
    assert built["status"] == "passed", built["issues"]
    sid = report["stream_label"]
    assert built["e20_trainer_steps"] == {sid: [
        {"trainer_step": 5, "rows": 2, "mini_batch": 2, "iterations": 1,
         "source": "in_process_capture",
         "minibatches": [{"grad_norm": 0.0, "optimizer_step_called": False}],
         "reported_num_batches": 1},
        {"trainer_step": 6, "rows": 2, "mini_batch": 2, "iterations": 1,
         "source": "in_process_capture",
         "minibatches": [{"grad_norm": 2.0, "optimizer_step_called": True}],
         "reported_num_batches": 1}]}
    e20 = steps.reconcile_optimizer_steps(built["e20_trainer_steps"][sid], scope="synthetic")
    assert e20["status"] == "passed", e20["issues"]
    assert [(r["expected_optimizer_steps"], r["counted_optimizer_steps"])
            for r in e20["trainer_steps"]] == [(0, 0), (1, 1)]
    w0 = sha(struct.pack("<2f", 1.0, 2.0))
    w1 = sha(struct.pack("<2f", 0.5, 1.5))
    frozen = sha(BF16_ONE_TWO)
    skip, taken = built["e07_trainer_steps"]
    assert skip == {"stream_id": sid, "trainer_step": 5, "hash_comparability": "full", "input": {
        "trainer_step": 5, "scheduler_before": {"last_epoch": 0, "lrs": [0.0001]},
        "scheduler_after": {"last_epoch": 1, "lrs": [0.0001]},
        "minibatches": [{"index": 0, "grad_norm": 0.0, "optimizer_step_called": False,
                         "before": {"parameters": {"expert.w": w0, "vlm.lm_head.weight": frozen},
                                    "optimizer_state": {"expert.w": {}},
                                    "step_counters": {"expert.w": 0}},
                         "after": {"parameters": {"expert.w": w0, "vlm.lm_head.weight": frozen},
                                   "optimizer_state": {"expert.w": {}},
                                   "step_counters": {"expert.w": 0}}}]}}
    assert taken["input"]["minibatches"][0]["after"] == {
        "parameters": {"expert.w": w1, "vlm.lm_head.weight": frozen},
        "optimizer_state": {"expert.w": {"exp_avg": sha(struct.pack("<2f", 0.25, 0.25)),
                                         "step": sha(struct.pack("<f", 1.0))}},
        "step_counters": {"expert.w": 1}}
    for item in (skip, taken):
        verdict = steps.check_skip_semantics(item["input"], scope="synthetic")
        assert verdict["status"] == "passed", verdict["issues"]
        given = item["input"]
        assert adapters.check_skip(
            given["trainer_step"], scheduler_before=given["scheduler_before"],
            scheduler_after=given["scheduler_after"], minibatches=given["minibatches"],
            scope="synthetic")["status"] == "passed"
    step6 = built["streams"][0]["steps"][1]
    assert step6["parameter_changes"] == {
        "hash_comparability": "full", "changed": ["expert.w"],
        "unchanged": ["action_in_proj.freqs", "vlm.lm_head.weight"],
        "only_in_pre_step": [], "only_in_post_step": []}
    assert [m["optimizer_step"] for s in built["streams"][0]["steps"]
            for m in s["minibatches"]] == ["skipped", "taken"]
    assert step6["minibatches"][0]["row_indices"] == [1, 0]
    assert step6["minibatches"][0]["advantages"] == [0.0, 0.5]
    # E03 over the captured init snapshot: a full-identity s4_param_snapshot v1 artifact
    init = next(e for e in events if e["kind"] == "parameter_snapshot" and e["phase"] == "init")
    artifact = records.param_snapshot_artifact(init, scope="synthetic")
    from teacher_rl.export_audit import tensor_table

    table, problem = tensor_table(artifact)
    assert problem is None and table["expert.w"]["byte_length"] == 8


def test_two_minibatches_leave_e07_not_estimable_but_e20_exact(monkeypatch, tmp_path,
                                                                fake_torch):
    report = run_pinned_shaped_steps(monkeypatch, tmp_path, trainer_steps=(5,),
                                     grad_norms=[0.0, 3.0], mini_batch=1, order=((1, 0),))
    streams, _ = streams_from_reports([report])
    built = records.build_trainer_step_records(streams, scope="synthetic")
    assert built["status"] == "passed", built["issues"]
    (step,) = built["streams"][0]["steps"]
    assert step["e07"] == {"assembly": "not_estimable",
                           "reasons": ["per_minibatch_snapshots_unavailable"],
                           "hash_comparability": "full"}
    (e20_input,) = built["e20_trainer_steps"][report["stream_label"]]
    assert e20_input["minibatches"] == [{"grad_norm": 0.0, "optimizer_step_called": False},
                                        {"grad_norm": 3.0, "optimizer_step_called": True}]
    assert steps.reconcile_optimizer_steps([e20_input], scope="synthetic")["status"] == "passed"


def test_failed_snapshot_is_an_extract_error_and_blocks_the_records(monkeypatch, tmp_path,
                                                                     fake_torch):
    fakes = install_fakes(monkeypatch, RewardScript(()))
    trainer_cls = pinned_shaped_trainer(fakes)
    capture, s = new_capture(tmp_path, config={**CONFIG, "parameter_hashing": "sha256_local"})
    capture.install()
    model = TensorModel([("x", FakeTensor(b"\x00" * 8, "complex64", (1,), True))], [])
    trainer = trainer_cls(trainer_config(tmp_path), None, model=model,
                          optimizers=FakeOptimizersContainer([]),
                          lr_schedulers=FakeSchedulersContainer([FakeLambdaLR(0.1)]),
                          grad_norms=[0.0], order=[[0]], mini_batch=1, iterations=1)
    trainer.step_training([SimpleNamespace(completion="/r.json", advantage=1.0,
                                           steps=[(-1.0, False)])], 3, 10, 0, None, True)
    s.close()
    report = read_stream(s.path, max_bytes=10**7)
    events = report["events"]
    assert not [e for e in events if e["kind"] == "parameter_snapshot"]
    assert sorted(e["phase"] for e in events if e["kind"] == "capture_extract_error") == [
        "init_snapshot", "post_step_snapshot", "pre_step_snapshot"]
    assert [e["kind"] for e in events].count("trainer_step_started") == 1
    built = records.build_trainer_step_records(streams_from_reports([report])[0],
                                               scope="synthetic")
    assert built["status"] == "blocked"
    assert built["streams"][0]["steps"][0]["e07"]["reasons"] == ["snapshot_missing"]
    assert built["e20_trainer_steps"][report["stream_label"]][0]["rows"] == 1


def test_capture_tree_stages_the_rule11_modules():
    assert {"gate/__init__.py", "gate/s2_identity.py",
            "gate/tensor_manifest.py"} <= set(host_launch.CAPTURE_FILES)


# ------------------------------------------------------------------- G1 review regressions


def test_g1_06_non_list_argv_is_refused_never_raised(tmp_path):
    """Review G1-06: a self-consistent spec whose argv is not a list of str is a refusal."""
    from gate.s2_identity import identity_hash

    spec, registration = launch_inputs(tmp_path, tmp_path / "tree", tmp_path / "m.json")
    decisions = launch_decisions(tmp_path, spec, registration).read_bytes()
    for argv, cwd in ((5, spec["cwd"]), (["ok", 7], spec["cwd"]), (spec["argv"], 3)):
        bad = dict(spec, argv=argv, cwd=cwd)
        bad["launch_hash"] = identity_hash(host_launch.launch_body(bad))  # self-consistent
        check = host_launch.verify_launch_authority(bad, registration, decisions,
                                                    decision_id="D-032", launch_args=["x"],
                                                    cwd="/x")
        assert check["authorized"] is False
        assert [i["code"] for i in check["issues"]] == ["launch_spec_invalid"]
    control = host_launch.verify_launch_authority(spec, registration, decisions,
                                                  decision_id="D-032")
    assert control["authorized"] is True


def test_g1_09_bound_hash_of_a_non_registration_is_refused(tmp_path):
    """Review G1-09: a decision binding the hash of an arbitrary object is not a registration."""
    import copy

    from gate.s2_identity import identity_hash

    spec, registration = launch_inputs(tmp_path, tmp_path / "tree", tmp_path / "m.json")
    for invalid in ({"scope": "actual"}, dict(registration, artifact_kind="s4_registration"),
                    dict(registration, schema_version=1),
                    {k: v for k, v in registration.items() if k != "owner"}):
        edited = copy.deepcopy(spec)
        edited["input_hashes"]["registration"] = identity_hash(invalid)  # outside launch_hash
        assert edited["launch_hash"] == spec["launch_hash"]
        sub = tmp_path / str(len(list(tmp_path.iterdir())))
        sub.mkdir()
        decisions = launch_decisions(sub, edited, invalid).read_bytes()
        check = host_launch.verify_launch_authority(edited, invalid, decisions,
                                                    decision_id="D-032")
        assert check["authorized"] is False
        assert "registration_invalid" in [i["code"] for i in check["issues"]]


def test_g1_09_registration_shape_mirrors_the_registration_module():
    from teacher_rl import registration

    assert host_launch.REGISTRATION_KIND == registration.ARTIFACT_KIND
    assert host_launch.REGISTRATION_SCHEMA_VERSION == registration.SCHEMA_VERSION
    assert host_launch.REGISTRATION_FIELDS == registration.TOP_FIELDS


# ------------------------------------------------------------------- G3R residual regressions


def test_g3r_genuinely_blocked_build_launch_spec_is_not_passed_never_invalid(tmp_path):
    """G3R (2): a blocked build_launch spec (argv/cwd None) is refused as not passed."""
    from teacher_rl import launch

    spec, registration = launch_inputs(tmp_path, tmp_path / "tree", tmp_path / "m.json")
    blocked = launch.build_launch(
        registration, alpagym_checkout=spec["cwd"], capture_tree=str(tmp_path / "tree"),
        capture_manifest=str(tmp_path / "m.json"), capture_config=str(tmp_path / "c.json"),
        worker_credentials="strip_offline", action_authority_ref="D-032 launch approval",
        alpasim_cache_dir=None, nurec_authorization_refs={"terms": "D-032 NuRec terms",
                                                          "acquire": "D-032 NuRec acquisition"})
    assert blocked["status"] == "blocked" and blocked["argv"] is None and blocked["cwd"] is None
    decisions = launch_decisions(tmp_path, spec, registration).read_bytes()
    check = host_launch.verify_launch_authority(blocked, registration, decisions,
                                                decision_id="D-032", launch_args=["x"], cwd="/x")
    assert check["authorized"] is False
    assert [i["code"] for i in check["issues"]] == ["launch_spec_not_passed"]
    # still fail closed: an incomplete spec stays invalid, a passed one with bad argv too
    partial = {k: v for k, v in blocked.items() if k != "argv"}
    assert [i["code"] for i in host_launch.verify_launch_authority(
        partial, registration, decisions, decision_id="D-032")["issues"]] == [
            "launch_spec_invalid"]


@pytest.mark.parametrize("route", ["hydra_only", "h1_resolved_toml"])
def test_g3r_authority_request_lines_satisfy_the_real_verifier(tmp_path, route):
    """G3R (3): the exact lines build_launch lists authorize that launch, no more."""
    from string import Template

    from tests.test_s2_guest_bootstrap import WIRING

    spec, registration = launch_inputs(tmp_path, tmp_path / "tree", tmp_path / "m.json",
                                       route=route)
    lines = spec["authority_request"]
    grants = [f"  - {line}" for line in lines if line.startswith("authorizes: ")]
    recorded = [f"  - {line}" for line in lines if line.startswith("records: ")]
    assert len(grants) == 2 and len(recorded) == 2
    decisions = Template((WIRING / "decisions_template.md").read_text()).substitute(
        grants="\n".join(grants), wrong_grants="", records="\n".join(recorded)).encode()
    check = host_launch.verify_launch_authority(spec, registration, decisions,
                                                decision_id="D-032")
    assert check["authorized"] is True, check["issues"]
    # negative control: without the registration line the same entry does not authorize
    partial = Template((WIRING / "decisions_template.md").read_text()).substitute(
        grants=grants[0], wrong_grants="", records="\n".join(recorded)).encode()
    assert [i["code"] for i in host_launch.verify_launch_authority(
        spec, registration, partial, decision_id="D-032")["issues"]] == ["launch_not_authorized"]
