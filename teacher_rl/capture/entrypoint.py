"""Zero-fork in-process capture entrypoint for AlpaGym's Cosmos workers (lane-r3 §6 H4).

Tickets: S4.3 (preserve per-timestep simulator replies, every attempt, retry and dropped payload
before upstream cleanup), E01 (session seeds), E02/E03 (load gaps; parameter/buffer metadata),
E05/E09 (behaviour version at generation; trainer-received rollouts hashed before the filter
unlinks them), E06/E14 (attempt ledger with causes), E07/E20 (exact reduced grad norm, whether
``optimizer.step`` ran, learning rate), E15 (run identity inputs). Agent-authored under
D-018/D-024.

Cosmos runs ``python -m <module> --config <toml>`` for the controller and every replica. The host
launcher (``host_launch.py``) substitutes this module for ``alpagym_runtime.cosmos.entrypoint``.
``main`` then: reads ``custom.resolved_config_path`` from the TOML to locate the run directory
(it must be absolute: the registration compiles an absolute ``run_root`` because the pinned
default ``./tmp/alpagym-runs`` is relative), validates the capture configuration named by
``PHYSLOOP_CAPTURE_CONFIG``, refuses unless ``PYTHONDONTWRITEBYTECODE`` is set (no bytecode is
written into the verified capture tree), applies the required worker credential mode
``PHYSLOOP_WORKER_CREDENTIALS`` (``strip_offline``: credential-named variables are removed from
this role's environment and ``HF_HUB_OFFLINE=1`` is set; ``inherit_disclosed``: the names, never
the values, are recorded), opens a per-process stream under ``<run_dir>/physloop_capture/``,
imports the upstream entrypoint (which performs the Cosmos registrations), installs idempotent
sentinel wrappers on the lookup bindings in ``WRAP_TARGETS`` and calls the upstream ``main``
with the untouched argv. Upstream code is never edited, copied or re-implemented.

Failure outcomes (E06/E14) are read under the worker's own lock: the ``rollout_failed`` wrapper
installs, once per worker, a ``_LockProbe`` that delegates to the same ``threading.Lock`` and
snapshots ``permanently_failed``/``retries_left`` on the first acquire/release pair of the
failing thread, so a concurrent sibling failure cannot change the recorded outcome.

E04 replay rows: ``_prepare_training_data`` is wrapped to record the replay pool and to map
each pool sample to its row index; ``_train_minibatch`` then records the row indices (the pinned
``torch.randperm`` is unseeded, so they are observed, never reproduced), the per-row old
log-probs and padding mask, and the advantages; ``_forward_with_reference`` names its
minibatch. E02/E21: after ``_save_checkpoint`` returns, the ``safetensors/step_N`` export (when
the pinned export condition held) is hashed file by file before any later pruning. Pinned
cosmos-rl ``LLMTrainer.export_safetensors`` (d2a2c57, ``llm_trainer.py:771-790``) writes the
export on a daemon thread ``save_and_upload_safetensors`` kept on ``trainer.upload_thread`` and
returns without joining it, so the after-hook first joins that thread (bounded by
``EXPORT_JOIN_TIMEOUT_S``). If it is still alive, or no new thread handle appeared, the export
is recorded as ``pending_async_write`` (never ``hashed``) and re-hashed as
``checkpoint_export_settled`` at the next ``step_training`` start (after
``trainer_step_started``) or at shutdown (A8-C-03); a failure while re-hashing one export is
recorded as ``checkpoint_export_settle_error`` and keeps it pending (A8R2).

Trainer evidence for E03/E07/E20 (F5; consumed by ``teacher_rl.records.build_trainer_step_records``):

- ``trainer_configured`` (``trainer_init`` after-hook): ``{call, mini_batch, iterations,
  group_size, allowed_outdated_steps, kl_beta}`` read from the pinned trainer attributes
  ``_mini_batch``, ``_grpo_optimization_iterations``, ``_group_size``,
  ``_allowed_outdated_steps``, ``_kl_beta``; an absent or non-numeric attribute is recorded as
  ``null``, never guessed.
- ``scheduler_state`` around every ``step_training``: ``{call, phase: pre_step|post_step,
  trainer_step, last_epoch, lrs}``. The pinned ``step_training`` calls ``lr_schedulers.step()``
  once after the loop, so pre and post differ by one epoch. A Cosmos ``LRSchedulersContainer``
  (``schedulers`` attribute) is iterated; its schedulers must agree on ``last_epoch``.
- ``optimizer_reduce`` / ``optimizer_reduce_raised`` carry ``minibatch_call`` (the enclosing
  ``_train_minibatch`` call; pinned ``_train_minibatch`` calls ``all_reduce_states``), so a
  reduce is paired with its minibatch by id, never by position.
- ``parameter_snapshot`` (``SNAPSHOT_SCHEMA_VERSION`` 2; after ``trainer_init`` with phase
  ``init`` and ``trainer_step: null``, and with ``init_and_each_step`` also as ``pre_step`` /
  ``post_step`` of each ``step_training`` with ``trainer_step = current_step``)::

      {schema_version: 2, phase, trainer_step,
       tensors: {name: {kind: param|buffer, dtype: <rule-11 safetensors name>, shape, numel,
                        requires_grad, sha256, hash_scope: full|local_shard,
                        shard: null | {global_shape, offsets, placement}}},
       training_flags: {module_name: bool}, optimizer_state: {param_name: {state_key: sha256}},
       step_counters: {param_name: int}, applied_lr: [float per param group]}

  The event kind is ``parameter_snapshot``; no other field is added. ``init`` is a deliberate
  third phase (``init_only`` would otherwise record nothing). ``shape``/``numel`` describe
  the bytes hashed: for a DTensor (FSDP2) that is ``to_local()``, ``hash_scope`` is
  ``local_shard`` and ``shard`` records the global shape, the shard offsets (torch ``Shard``
  chunk rule: chunk ``ceil(global / mesh_size)``, offset ``min(chunk * coordinate, global)``,
  checked against the local extent) and the placement text. A local-shard hash is comparable
  only with the same shard of the same rank (same stream, same ``shard``), never with a full
  tensor or an export manifest. Optimizer-state tensors follow the same rule (FSDP2 shards
  moments like their parameter; a sharded state entry under an unsharded parameter refuses);
  a non-tensor state value is hashed as its canonical JSON bytes. ``optimizer_state`` and
  ``step_counters`` name every member of every param group of every inner optimizer of a
  Cosmos ``OptimizersContainer``; a member with no state yet maps to ``{}`` and counter ``0``.
  ``applied_lr`` is each group's ``lr`` (``get_last_lr()`` reports only the first scheduler).
  With ``parameter_hashing: metadata_only`` every ``sha256`` and ``hash_scope`` is ``null``
  (nothing was hashed; the consumer cannot run E07 on it). Any defect (unknown dtype, a
  Partial, nested or Shard-subclass placement such as ``_StridedShard``, an optimizer member
  without a model name, a non-integral step) raises, so no partial snapshot is emitted: the failure is a ``capture_extract_error``.
- Hashing uses ``gate.tensor_manifest.tensor_record`` (rule 11) through a lazy import inside
  the trainer process. **Shared change required:** the staged capture tree
  (``host_launch.CAPTURE_FILES``) must also list ``gate/__init__.py``, ``gate/s2_identity.py``
  and ``gate/tensor_manifest.py``; until it does, ``sha256_local`` snapshots on a host fail as
  ``capture_extract_error`` (``ModuleNotFoundError``). ``RULE11_DTYPES`` maps torch dtype names
  to the twelve rule-11 names; its value set is tested equal to ``DTYPE_ITEMSIZE``.

Fail closed: a missing wrap target, a signature that differs from the pinned source, an invalid
capture configuration or a missing run directory aborts at startup; a sink failure during the
run calls the fatal handler (default: message on stderr, ``os._exit(70)``) because evidence
loss must stop qualification visibly and must not masquerade as a rollout failure (upstream
catches ``Exception`` in ``_run_rollout`` and would turn it into a silent retry). Extraction
problems are recorded as ``capture_extract_error`` events, never ignored.

Verification ceiling:
- wrapper installation, idempotency, signature checks and event semantics: **L2** — tested with
  ``sys.modules`` fakes whose signatures are AST-checked against the vendored pinned sources in
  ``tests/fixtures/s4_upstream/`` (``tests/test_s4_capture.py``).
- ``rule11_dtype``, ``shard_layout``, ``tensor_entry``, ``build_parameter_snapshot``,
  ``scheduler_state``, ``trainer_settings``: **L1/L2** — literal byte known answers with
  ``sys.modules`` torch fakes (fake DTensor, optimizer container, scheduler container).
- behaviour inside real Cosmos/AlpaGym processes, FSDP/DTensor parameter hashing, the actual
  event volume and timing: **L3** (``qualify_on_host``). Hashing a 10B policy on CPU twice
  per trainer step takes minutes, and one snapshot event must fit ``max_event_bytes`` (an
  oversize event is replaced by ``event_oversize`` and its content is lost): both are host
  sizing items.

Imports at module level are stdlib only (plus ``teacher_rl.capture.sink``): this module runs
in the AlpaGym environment and must not import torch, numpy, upstream or PhysLoop's ``gate``
at import time. It never reads credentials and never edits or deletes upstream files.
"""

import argparse
import functools
import hashlib
import importlib
import inspect
import itertools
import json
import math
import os
import re
import stat
import sys
import threading
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path

from teacher_rl.capture.sink import EventSink, SinkError, canonical_bytes, validate_limits

CONFIG_SCHEMA = "physloop.s4.capture.config.v1"
ENV_CONFIG = "PHYSLOOP_CAPTURE_CONFIG"
UPSTREAM_ENTRYPOINT = "alpagym_runtime.cosmos.entrypoint"
SENTINEL = "_physloop_capture_wrap"
PAYLOAD_ATTR = "_physloop_payload_instance"
CAPTURE_SUBDIR = "physloop_capture"
SNAPSHOT_MODES = ("init_only", "init_and_each_step")
HASHING_MODES = ("metadata_only", "sha256_local")
CONFIG_FIELDS = ("schema", "limits", "max_inline_items", "max_message_chars",
                 "max_completion_file_bytes", "max_checkpoint_file_bytes", "parameter_snapshots",
                 "parameter_hashing", "capture_simulator_bytes")
ENV_WORKER_CREDENTIALS = "PHYSLOOP_WORKER_CREDENTIALS"
WORKER_CREDENTIAL_MODES = ("strip_offline", "inherit_disclosed")
# Worker-environment credential names (agent technical default, D-018; A8-H-02, A8R2): exact
# names plus end-of-name suffixes, compared upper-cased. A bare substring such as TOKEN would
# also strip configuration like TOKENIZERS_PARALLELISM or VLLM_MAX_TOKENS; the "_TOKEN" suffix
# matches neither. JUPYTER_TOKEN and OPEN_BUTTON_TOKEN are Vast session credentials.
CREDENTIAL_NAMES = frozenset({
    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "WANDB_API_KEY",
    "NGC_API_KEY", "VAST_API_KEY", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "GITHUB_TOKEN",
    "GH_TOKEN", "JUPYTER_TOKEN", "OPEN_BUTTON_TOKEN", "AWS_ACCESS_KEY_ID", "SSH_PRIVATE_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS", "HUGGINGFACEHUB_API_TOKEN", "HF_API_TOKEN", "NPM_TOKEN",
    "MLFLOW_TRACKING_TOKEN"})
CREDENTIAL_SUFFIXES = ("_API_KEY", "_SECRET", "_SECRET_KEY", "_PASSWORD", "_ACCESS_TOKEN",
                       "_AUTH_TOKEN", "_TOKEN", "_PASSWD", "_PRIVATE_KEY", "_KEY_ID",
                       "_CREDENTIALS")
# Pinned cosmos-rl export writer (d2a2c57, llm_trainer.py:771-790) and the bounded join on it
# (agent technical default, D-018; A8-C-03). A longer write is re-hashed on a later event.
EXPORT_THREAD_NAME = "save_and_upload_safetensors"
EXPORT_JOIN_TIMEOUT_S = 30.0
_ROLE = re.compile(r"[A-Za-z0-9_-]{1,32}")
SNAPSHOT_SCHEMA_VERSION = 2
SNAPSHOT_PHASES = ("init", "pre_step", "post_step")
# torch dtype name -> rule-11 safetensors name (build-spec section 2 rule 11).
RULE11_DTYPES = {
    "float64": "F64", "float32": "F32", "float16": "F16", "bfloat16": "BF16",
    "int64": "I64", "int32": "I32", "int16": "I16", "int8": "I8", "uint8": "U8",
    "bool": "BOOL", "float8_e4m3fn": "F8_E4M3", "float8_e5m2": "F8_E5M2"}
# Pinned AlpagymGRPOTrainer.__init__ attributes (trainer.py:118-131) recorded per process.
TRAINER_SETTINGS = (("mini_batch", "_mini_batch"),
                    ("iterations", "_grpo_optimization_iterations"),
                    ("group_size", "_group_size"),
                    ("allowed_outdated_steps", "_allowed_outdated_steps"))

_G_RT = "alpagym/packages/runtime/src/alpagym_runtime"
_G_POL = "alpagym/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1"


class CaptureInstallError(RuntimeError):
    """A wrap target is missing or differs from the pinned source; capture refuses to run."""


class CaptureFatal(BaseException):
    """Evidence could not be recorded. Deliberately not an ``Exception``: upstream
    ``_run_rollout`` catches ``Exception`` and would turn evidence loss into a silent retry."""


@dataclass(frozen=True)
class WrapTarget:
    """One upstream lookup binding wrapped in place (``owner.attribute`` at import time)."""

    target_id: str
    module: str
    qualname: str
    kind: str  # "method" | "function_binding" | "class_binding" | "chained_method"
    params: tuple
    evidence: tuple
    fixture: str  # vendored pinned source (tests/fixtures/s4_upstream/<fixture>.txt)
    defined_in: str  # "Class.method" or "function" inside the fixture

    @property
    def owner_path(self):
        return self.qualname.rpartition(".")[0]

    @property
    def attribute(self):
        return self.qualname.rpartition(".")[2]


WRAP_TARGETS = (
    WrapTarget("runtime_stub", "alpagym_runtime.cosmos.rollout_backend", "RuntimeServiceStub",
               "class_binding", (), ("E09", "E10", "E12", "E13", "E14"),
               f"{_G_RT}/cosmos/rollout_backend.py", "RuntimeServiceStub"),
    WrapTarget("submit_payload", "alpagym_runtime.episode_runner.streaming_worker",
               "StreamingRolloutWorker.submit_payload", "method", ("self", "payload"),
               ("E06", "E09", "E14", "E15"), f"{_G_RT}/episode_runner/streaming_worker.py",
               "StreamingRolloutWorker.submit_payload"),
    WrapTarget("run_rollout", "alpagym_runtime.episode_runner.streaming_worker",
               "StreamingRolloutWorker._run_rollout", "method", ("self", "rollout_job"),
               ("E06", "E09", "E14"), f"{_G_RT}/episode_runner/streaming_worker.py",
               "StreamingRolloutWorker._run_rollout"),
    WrapTarget("rollout_succeeded", "alpagym_runtime.episode_runner.streaming_worker",
               "StreamingRolloutWorker._on_rollout_succeeded", "method",
               ("self", "rollout_job", "episode"), ("E06", "E09"),
               f"{_G_RT}/episode_runner/streaming_worker.py",
               "StreamingRolloutWorker._on_rollout_succeeded"),
    WrapTarget("rollout_failed", "alpagym_runtime.episode_runner.streaming_worker",
               "StreamingRolloutWorker._on_rollout_failed", "method",
               ("self", "rollout_job", "exc"), ("E06", "E14"),
               f"{_G_RT}/episode_runner/streaming_worker.py",
               "StreamingRolloutWorker._on_rollout_failed"),
    WrapTarget("session_start", "alpagym_runtime.alpasim.driver_server",
               "EgodriverGrpcServicer.start_session", "method", ("self", "request", "context"),
               ("E01",), f"{_G_RT}/alpasim/driver_server.py",
               "EgodriverGrpcServicer.start_session"),
    WrapTarget("session_close", "alpagym_runtime.alpasim.driver_server",
               "EgodriverGrpcServicer.close_session", "method", ("self", "request", "context"),
               ("E01",), f"{_G_RT}/alpasim/driver_server.py",
               "EgodriverGrpcServicer.close_session"),
    WrapTarget("rollout_generation", "alpagym_runtime.cosmos.rollout_backend",
               "AlpagymRollout.rollout_generation", "method",
               ("self", "payloads", "stream", "data_packer", "data_fetcher", "is_validation",
                "current_weight_version"), ("E05", "E09"), f"{_G_RT}/cosmos/rollout_backend.py",
               "AlpagymRollout.rollout_generation"),
    WrapTarget("filter_trainable", "alpagym_runtime.cosmos.trainer",
               "filter_trainable_rollouts", "function_binding",
               ("rollouts", "current_step", "train_batch_per_replica", "allowed_outdated_steps"),
               ("E05", "E09"), f"{_G_RT}/cosmos/rollout_filter.py", "filter_trainable_rollouts"),
    WrapTarget("trainer_init", "alpagym_runtime.cosmos.trainer", "AlpagymGRPOTrainer.__init__",
               "method", ("self", "config", "parallel_dims", "kwargs"), ("E03",),
               f"{_G_RT}/cosmos/trainer.py", "AlpagymGRPOTrainer.__init__"),
    WrapTarget("step_training", "alpagym_runtime.cosmos.trainer",
               "AlpagymGRPOTrainer.step_training", "method",
               ("self", "rollouts", "current_step", "total_steps", "remain_samples_num",
                "inter_policy_nccl", "is_master_replica", "do_save_checkpoint"),
               ("E03", "E20"), f"{_G_RT}/cosmos/trainer.py", "AlpagymGRPOTrainer.step_training"),
    WrapTarget("prepare_training_data", "alpagym_runtime.cosmos.trainer",
               "AlpagymGRPOTrainer._prepare_training_data", "method", ("self", "rollouts"),
               ("E04",), f"{_G_RT}/cosmos/trainer.py",
               "AlpagymGRPOTrainer._prepare_training_data"),
    WrapTarget("forward_with_reference", "alpagym_runtime.cosmos.trainer",
               "AlpagymGRPOTrainer._forward_with_reference", "method", ("self", "model_inputs"),
               ("E04",), f"{_G_RT}/cosmos/trainer.py",
               "AlpagymGRPOTrainer._forward_with_reference"),
    WrapTarget("train_minibatch", "alpagym_runtime.cosmos.trainer",
               "AlpagymGRPOTrainer._train_minibatch", "method",
               ("self", "minibatch_samples", "minibatch_advantages", "inter_policy_nccl"),
               ("E04", "E07", "E20"), f"{_G_RT}/cosmos/trainer.py",
               "AlpagymGRPOTrainer._train_minibatch"),
    WrapTarget("all_reduce_states", "alpagym_runtime.cosmos.trainer",
               "AlpagymGRPOTrainer.all_reduce_states", "method", ("self", "inter_policy_nccl"),
               ("E07", "E20"), f"{_G_RT}/cosmos/trainer.py",
               "AlpagymGRPOTrainer.all_reduce_states"),
    WrapTarget("save_checkpoint", "alpagym_runtime.cosmos.trainer",
               "AlpagymGRPOTrainer._save_checkpoint", "method",
               ("self", "current_step", "total_steps", "remain_samples_num"), ("E02", "E21"),
               f"{_G_RT}/cosmos/trainer.py", "AlpagymGRPOTrainer._save_checkpoint"),
    WrapTarget("runtime_bridge", "alpagym_alpamayo_r1.bundle",
               "install_alpamayo_r1_runtime_bridge", "function_binding", (), ("E02",),
               f"{_G_POL}/bundle.py", "install_alpamayo_r1_runtime_bridge"),
)
CHAINED_TARGETS = (
    WrapTarget("load_hf_weights", "alpamayo1_x_rl.models.expert_model.cosmos_wrapper",
               "ExpertModelCosmos.load_hf_weights", "chained_method",
               ("self", "model_name_or_path", "parallel_dims", "device", "revision"),
               ("E02", "E03"),
               "alpamayo-recipes/recipes/alpamayo1_x_rl/models/expert_model/cosmos_wrapper.py",
               "ExpertModelCosmos.load_hf_weights"),
)


def validate_capture_config(config):
    """Return a validated capture configuration or raise ``ValueError`` (every field required)."""
    if type(config) is not dict or set(config) != set(CONFIG_FIELDS):
        raise ValueError(f"capture config must contain exactly {list(CONFIG_FIELDS)}")
    if config["schema"] != CONFIG_SCHEMA:
        raise ValueError(f"capture config schema must be {CONFIG_SCHEMA!r}")
    validate_limits(config["limits"])
    for name in ("max_inline_items", "max_message_chars", "max_completion_file_bytes",
                 "max_checkpoint_file_bytes"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config["parameter_snapshots"] not in SNAPSHOT_MODES:
        raise ValueError(f"parameter_snapshots must be one of {SNAPSHOT_MODES}")
    if config["parameter_hashing"] not in HASHING_MODES:
        raise ValueError(f"parameter_hashing must be one of {HASHING_MODES}")
    if type(config["capture_simulator_bytes"]) is not bool:
        raise ValueError("capture_simulator_bytes must be true or false")
    return json.loads(json.dumps(config))


def load_capture_config(path):
    """Read and validate the capture configuration file (strict JSON, regular file)."""
    data = Path(path).read_bytes()

    def no_duplicates(pairs):
        keys = [key for key, _ in pairs]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate key in capture config")
        return dict(pairs)

    return validate_capture_config(json.loads(data, object_pairs_hook=no_duplicates))


def _default_fatal(message):
    sys.stderr.write(f"[physloop-capture] FATAL: {message}\n")
    sys.stderr.flush()
    os._exit(70)


class Capture:
    """Holds the sink and configuration; builds the per-target wrappers."""

    def __init__(self, sink, config, *, on_fatal=_default_fatal, import_module=None):
        self.sink = sink
        self.config = validate_capture_config(config)
        self.on_fatal = on_fatal
        self.import_module = import_module or importlib.import_module
        self._calls = itertools.count(1)
        self._payloads = itertools.count(1)
        self._bind_lock = threading.Lock()
        self._tls = threading.local()
        self._pool = None  # (samples, {id(sample): row}, call) of the latest replay pool
        self.export_join_timeout = EXPORT_JOIN_TIMEOUT_S
        self._export_lock = threading.Lock()
        self._pending_exports = []  # exports whose writer was not confirmed finished

    # ---------- emission ----------

    def emit(self, kind, payload):
        try:
            return self.sink.emit(kind, payload)
        except SinkError as exc:
            message = f"capture sink failure on {kind}: {exc}"
            self.on_fatal(message)
            raise CaptureFatal(message) from exc

    def _blob(self, data):
        try:
            return self.sink.put_blob(data)
        except SinkError as exc:
            message = f"capture blob failure: {exc}"
            self.on_fatal(message)
            raise CaptureFatal(message) from exc

    def extract_error(self, target_id, phase, exc):
        self.emit("capture_extract_error", {"target": target_id, "phase": phase,
                                            "error_type": type(exc).__name__,
                                            "message": self.text(str(exc))})

    def text(self, value):
        value = str(value)
        limit = self.config["max_message_chars"]
        return value if len(value) <= limit else value[:limit] + f"...[+{len(value) - limit}]"

    def bounded(self, values):
        """Inline a JSON list up to ``max_inline_items``; otherwise store it as a blob."""
        values = [v if type(v) in (dict, list) else _json_scalar(v) for v in values]
        if len(values) <= self.config["max_inline_items"]:
            return {"count": len(values), "values": values}
        return {"count": len(values), "blob": self._blob(canonical_bytes(values))}

    def next_call(self):
        return next(self._calls)

    def payload_instance(self, state):
        """Tag a ``SharedPayloadState`` with a capture-assigned instance number exactly once.

        Upstream enqueues sibling jobs inside ``submit_payload`` and worker threads may start
        them before ``submit_payload`` returns, so ordering by submission is unreliable. The
        first observer (submission or attempt start) binds the instance under a lock and emits
        ``payload_bound`` before any attempt event of that instance; no failure can precede
        the first attempt start, so ``retries_left`` is still the initial budget here.
        """
        with self._bind_lock:
            instance = getattr(state, PAYLOAD_ATTR, None)
            if instance is None:
                instance = next(self._payloads)
                setattr(state, PAYLOAD_ATTR, instance)
                self.emit("payload_bound", {"payload_instance": instance,
                                            "prompt_idx": state.payload.prompt_idx,
                                            "n_target": state.n_target,
                                            "retries_left": state.retries_left})
            return instance

    # ---------- installation ----------

    def install(self, targets=WRAP_TARGETS):
        """Wrap every target; idempotent (sentinel on the current binding)."""
        installed, present = [], []
        for target in targets:
            owner, original = self._resolve(target)
            if getattr(original, SENTINEL, None) == target.target_id:
                present.append(target.target_id)
                continue
            if target.kind != "class_binding":
                _check_signature(target, original)
            wrapper = self._build(target, original)
            setattr(wrapper, SENTINEL, target.target_id)
            setattr(owner, target.attribute, wrapper)
            installed.append(target.target_id)
        return {"installed": installed, "already_installed": present}

    def _resolve(self, target):
        try:
            module = self.import_module(target.module)
        except ImportError as exc:
            raise CaptureInstallError(f"{target.target_id}: cannot import {target.module}: "
                                      f"{exc}") from exc
        owner = module
        for part in filter(None, target.owner_path.split(".")):
            owner = getattr(owner, part, None)
            if owner is None:
                raise CaptureInstallError(f"{target.target_id}: {target.module}."
                                          f"{target.owner_path} is missing")
        original = inspect.getattr_static(owner, target.attribute, None)
        if original is None or not callable(getattr(owner, target.attribute, None)):
            raise CaptureInstallError(f"{target.target_id}: {target.module}.{target.qualname} "
                                      "is missing")
        return owner, getattr(owner, target.attribute)

    def _build(self, target, original):
        builder = getattr(self, f"_wrap_{target.target_id}", None)
        if builder is None:
            raise CaptureInstallError(f"no capture builder for {target.target_id}")
        return builder(target, original)

    # ---------- generic method wrapper ----------

    def _method(self, target, original, before=None, after=None, cleanup=None):
        signature = inspect.signature(original)

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            call = self.next_call()
            bound, ctx = None, {}
            try:
                try:
                    bound = signature.bind(*args, **kwargs).arguments
                    if before is not None:
                        before(call, bound, ctx)
                except Exception as exc:  # noqa: BLE001 - recorded; never alters upstream
                    self.extract_error(target.target_id, "before", exc)
                try:
                    result = original(*args, **kwargs)
                except CaptureFatal:
                    raise
                except BaseException as exc:
                    self.emit("call_raised", {"target": target.target_id, "call": call,
                                              "error_type": type(exc).__name__,
                                              "message": self.text(exc)})
                    raise
                if after is not None and bound is not None:
                    try:
                        after(call, bound, result, ctx)
                    except Exception as exc:  # noqa: BLE001 - recorded as capture_extract_error
                        self.extract_error(target.target_id, "after", exc)
                return result
            finally:
                if cleanup is not None:
                    cleanup(ctx)

        return wrapper

    # ---------- per-target builders ----------

    def _wrap_submit_payload(self, target, original):
        def after(call, a, state, ctx):
            self.emit("payload_submitted", {
                "call": call, "payload_instance": self.payload_instance(state),
                "prompt_idx": a["payload"].prompt_idx})

        return self._method(target, original, None, after)

    def _wrap_run_rollout(self, target, original):
        def before(call, a, ctx):
            job = a["rollout_job"]
            state = job.shared_payload_state
            self.emit("attempt_started", {
                "call": call, "payload_instance": self.payload_instance(state),
                "prompt_idx": state.payload.prompt_idx, "session_uuid": job.session_uuid,
                "scene_id": job.scene_id, "attempt": job.attempts})

        return self._method(target, original, before)

    def _wrap_rollout_succeeded(self, target, original):
        def after(call, a, result, ctx):
            job, episode = a["rollout_job"], a["episode"]
            state = job.shared_payload_state
            index = next((i for i, item in enumerate(state.collected) if item is episode), None)
            reward = getattr(episode, "reward", None)
            metrics = getattr(episode, "metrics", None)
            aggregated = getattr(metrics, "aggregated", None) if metrics is not None else None
            self.emit("attempt_succeeded", {
                "call": call, "payload_instance": self.payload_instance(state),
                "prompt_idx": state.payload.prompt_idx,
                "session_uuid": job.session_uuid, "attempt": job.attempts,
                "collected_index": index, "num_steps": getattr(episode, "num_steps", None),
                "reward_total": _json_scalar(getattr(reward, "total", None)),
                "reward_components": _float_map(getattr(reward, "report_metrics", None)),
                "aggregated_metrics": _float_map(aggregated)})

        return self._method(target, original, None, after)

    def _wrap_rollout_failed(self, target, original):
        """Outcome from snapshots taken under the worker lock by the failing thread itself.

        Pinned ``_on_rollout_failed`` holds ``self._lock`` for its whole decision: entry with
        ``permanently_failed`` -> ignored; it sets ``permanently_failed`` -> dropped; otherwise
        it enqueued a retry. Reading the state after the call returns races with sibling
        failures (A8-C-01), so the probe snapshots on acquire and before release.
        """

        def before(call, a, ctx):
            worker, state = a["self"], a["rollout_job"].shared_payload_state
            with self._bind_lock:
                if not isinstance(worker._lock, _LockProbe):
                    worker._lock = _LockProbe(worker._lock, self._tls)
            probe = {"state": state, "enter": None, "exit": None, "error": None}
            self._tls.probe = probe
            ctx["probe"] = probe

        def after(call, a, result, ctx):
            job, exc = a["rollout_job"], a["exc"]
            state = job.shared_payload_state
            probe = ctx.get("probe") or {}
            enter, leave = probe.get("enter"), probe.get("exit")
            if enter is None or leave is None:
                outcome, retries_left = None, None
                self.extract_error(target.target_id, "lock_probe", RuntimeError(
                    probe.get("error") or "no snapshot under the worker lock"))
            elif enter["permanently_failed"]:
                outcome, retries_left = "ignored_after_drop", leave["retries_left"]
            elif leave["permanently_failed"]:
                outcome, retries_left = "dropped", leave["retries_left"]
            else:
                outcome, retries_left = "retry", leave["retries_left"]
            self.emit("attempt_failed", {
                "call": call, "payload_instance": self.payload_instance(state),
                "prompt_idx": state.payload.prompt_idx,
                "session_uuid": job.session_uuid, "attempt": job.attempts,
                "cause_type": type(exc).__name__, "cause_message": self.text(exc),
                "outcome": outcome, "retries_left_after": retries_left,
                "outcome_basis": "worker_lock_snapshot"})

        def cleanup(ctx):
            self._tls.probe = None

        return self._method(target, original, before, after, cleanup)

    def _wrap_session_start(self, target, original):
        def after(call, a, result, ctx):
            request = a["request"]
            self.emit("session_started", {"call": call,
                                          "session_uuid": str(request.session_uuid),
                                          "random_seed": int(request.random_seed)})

        return self._method(target, original, None, after)

    def _wrap_session_close(self, target, original):
        def after(call, a, result, ctx):
            session = str(a["request"].session_uuid)
            records = getattr(a["self"], "_session_records", {})
            record = records.get(session)
            outputs = getattr(record, "outputs", None)
            self.emit("session_closed", {"call": call, "session_uuid": session,
                                         "recorded_steps": None if outputs is None
                                         else len(outputs)})

        return self._method(target, original, None, after)

    def _wrap_rollout_generation(self, target, original):
        def before(call, a, ctx):
            version = a.get("current_weight_version")
            self.emit("generation_entered", {
                "call": call, "prompt_idxs": [p.prompt_idx for p in a["payloads"]],
                "is_validation": bool(a.get("is_validation", False)),
                "current_weight_version": version if type(version) is int else None})

        def after(call, a, result, ctx):
            self.emit("generation_exited", {"call": call, "results": len(result)})

        return self._method(target, original, before, after)

    def _wrap_filter_trainable(self, target, original):
        limit = self.config["max_completion_file_bytes"]

        def before(call, a, ctx):
            rows = []
            for rollout in a["rollouts"]:
                completion = getattr(rollout, "completion", None)
                row = {"prompt_idx": getattr(rollout, "prompt_idx", None),
                       "completion": str(completion) if completion else None,
                       "weight_version": getattr(rollout, "weight_version", None),
                       "advantage": _json_scalar(getattr(rollout, "advantage", None)),
                       "reward": _json_scalar(getattr(rollout, "reward", None))}
                if completion:
                    row["completion_file"] = _hash_file(str(completion), limit)
                rows.append(row)
            self.emit("trainer_rollouts_received", {
                "call": call, "current_step": a["current_step"],
                "train_batch_per_replica": a["train_batch_per_replica"],
                "allowed_outdated_steps": a["allowed_outdated_steps"], "rollouts": rows})

        def after(call, a, kept, ctx):
            self.emit("trainer_rollouts_kept", {
                "call": call, "kept": [str(getattr(r, "completion", "")) for r in kept]})

        return self._method(target, original, before, after)

    def _guarded(self, target, phase, function, *args):
        """Run one evidence extraction; a failure is a ``capture_extract_error``, never a
        partial record and never an exception into upstream (``CaptureFatal`` propagates)."""
        try:
            function(*args)
        except Exception as exc:  # noqa: BLE001 - recorded as capture_extract_error
            self.extract_error(target.target_id, phase, exc)

    def _wrap_trainer_init(self, target, original):
        def after(call, a, result, ctx):
            trainer = a["self"]
            self.emit("trainer_configured", {"call": call, **trainer_settings(trainer)})
            self._guarded(target, "init_snapshot", self._snapshot, trainer, "init", None)

        return self._method(target, original, None, after)

    def _wrap_step_training(self, target, original):
        each = self.config["parameter_snapshots"] == "init_and_each_step"

        def before(call, a, ctx):
            trainer, step = a["self"], a["current_step"]
            self.emit("trainer_step_started", {
                "call": call, "current_step": step,
                "total_steps": a["total_steps"], "received_rollouts": len(a["rollouts"]),
                "do_save_checkpoint": bool(a.get("do_save_checkpoint", False))})
            self.settle_exports("step_start")  # isolated per export (A8R2)
            self._guarded(target, "pre_step_scheduler", self._scheduler_event, call, trainer,
                          "pre_step", step)
            if each:
                self._guarded(target, "pre_step_snapshot", self._snapshot, trainer, "pre_step",
                              step)

        def after(call, a, metrics, ctx):
            trainer, step = a["self"], a["current_step"]
            self.emit("trainer_step_finished", {"call": call, "current_step": step,
                                                "metrics": _float_map(metrics)})
            self._guarded(target, "post_step_scheduler", self._scheduler_event, call, trainer,
                          "post_step", step)
            if each:
                self._guarded(target, "post_step_snapshot", self._snapshot, trainer,
                              "post_step", step)

        def cleanup(ctx):
            self._pool = None  # release the replay pool references after the step

        return self._method(target, original, before, after, cleanup)

    def _wrap_prepare_training_data(self, target, original):
        def after(call, a, result, ctx):
            samples, advantages = result
            self._pool = (samples, {id(sample): row for row, sample in enumerate(samples)}, call)
            rollouts = list(a["rollouts"])
            rows = len(samples)
            self.emit("replay_pool_prepared", {
                "call": call, "rows": rows,
                "rollouts": [str(getattr(r, "completion", None)) for r in rollouts],
                "rows_per_rollout": rows // len(rollouts) if rollouts and
                rows % len(rollouts) == 0 else None,
                "advantages": self.bounded(_tensor_values(advantages))})

        return self._method(target, original, None, after)

    def _wrap_forward_with_reference(self, target, original):
        def after(call, a, result, ctx):
            log_probs, kl = result
            self.emit("replay_forward", {
                "call": call, "minibatch_call": getattr(self._tls, "minibatch_call", None),
                "log_probs": self.bounded(_tensor_values(log_probs)),
                "kl_present": kl is not None})

        return self._method(target, original, None, after)

    def _wrap_train_minibatch(self, target, original):
        def before(call, a, ctx):
            self._tls.minibatch_call = call
            samples = list(a["minibatch_samples"])
            pool = self._pool
            rows = [None] * len(samples)
            if pool is not None:
                for position, sample in enumerate(samples):
                    row = pool[1].get(id(sample))
                    if row is not None and pool[0][row] is sample:
                        rows[position] = row
            payload = {"call": call, "rows": len(samples),
                       "pool_call": None if pool is None else pool[2],
                       "row_indices": self.bounded(rows),
                       "advantages": self.bounded(_tensor_values(a["minibatch_advantages"])),
                       "old_logprobs": None, "is_padding": None}
            try:
                payload["old_logprobs"] = self.bounded(
                    [_tensor_values(s.training_signal.old_logprobs) for s in samples])
                payload["is_padding"] = self.bounded(
                    [[bool(v) for v in _tensor_values(s.training_signal.is_padding)]
                     for s in samples])
            except Exception as exc:  # noqa: BLE001 - recorded; the row event is still kept
                self.extract_error(target.target_id, "training_signal", exc)
            self.emit("minibatch_started", payload)

        def after(call, a, result, ctx):
            self.emit("minibatch_finished", {"call": call,
                                             "metrics": [_json_scalar(v) for v in result]})

        def cleanup(ctx):
            self._tls.minibatch_call = None

        return self._method(target, original, before, after, cleanup)

    def _wrap_all_reduce_states(self, target, original):
        signature = inspect.signature(original)

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            call = self.next_call()
            minibatch_call = getattr(self._tls, "minibatch_call", None)
            trainer = signature.bind(*args, **kwargs).arguments["self"]
            optimizers = getattr(trainer, "optimizers", None)
            counter = {"calls": 0}
            patched, own = False, None
            try:
                original_step = optimizers.step
                own = vars(optimizers).get("step") if hasattr(optimizers, "__dict__") else None

                @functools.wraps(original_step)
                def counted_step(*s_args, **s_kwargs):
                    counter["calls"] += 1
                    return original_step(*s_args, **s_kwargs)

                optimizers.step = counted_step
                patched = True
            except Exception as exc:  # noqa: BLE001 - recorded; never blocks the update
                self.extract_error(target.target_id, "before", exc)
            try:
                grad_norm = original(*args, **kwargs)
            except CaptureFatal:
                raise
            except BaseException as exc:
                self.emit("optimizer_reduce_raised", {
                    "call": call, "minibatch_call": minibatch_call,
                    "error_type": type(exc).__name__, "message": self.text(exc),
                    "optimizer_step_calls": counter["calls"] if patched else None})
                raise
            finally:
                if patched:
                    if own is None:
                        del optimizers.step  # the class-level binding is visible again
                    else:
                        optimizers.step = own
            lr = None
            try:
                schedulers = getattr(trainer, "lr_schedulers", None)
                if schedulers is not None and hasattr(schedulers, "get_last_lr"):
                    lr = [_json_scalar(v) for v in schedulers.get_last_lr()]
            except Exception as exc:  # noqa: BLE001 - recorded as capture_extract_error
                self.extract_error(target.target_id, "after", exc)
            self.emit("optimizer_reduce", {
                "call": call, "minibatch_call": minibatch_call,
                "grad_norm": _json_scalar(grad_norm),
                "optimizer_step_calls": counter["calls"] if patched else None, "lr": lr})
            return grad_norm

        return wrapper

    def _wrap_save_checkpoint(self, target, original):
        def before(call, a, ctx):
            ctx["prior_thread"] = getattr(a["self"], "upload_thread", None)
            self.emit("checkpoint_save", {"call": call, "phase": "enter",
                                          "current_step": a["current_step"]})

        def after(call, a, result, ctx):
            step = a["current_step"]
            payload = {"call": call, "phase": "returned", "current_step": step,
                       "export": {"status": "not_exported"}}
            train = a["self"].config.train
            # The pinned export condition (trainer.py _save_checkpoint): last step or
            # train.ckpt.export_safetensors; the export lands in safetensors/step_<N>.
            if step == a["total_steps"] or bool(train.ckpt.export_safetensors):
                export = Path(str(train.output_dir)) / "safetensors" / f"step_{step}"
                # A thread handle is only this export's writer if it is new since entry.
                thread = getattr(a["self"], "upload_thread", None)
                if (thread is None or thread is ctx.get("prior_thread")
                        or not isinstance(thread, threading.Thread)):
                    thread, reason = None, "writer_thread_not_found"
                else:
                    thread.join(self.export_join_timeout)
                    reason = "writer_thread_alive_after_join" if thread.is_alive() else None
                if reason is None:
                    payload["export"] = {"path": str(export), "synchronized": True,
                                         **self._export_manifest(export)}
                else:
                    payload["export"] = _pending_export(export, reason)
                    with self._export_lock:
                        self._pending_exports.append({"call": call, "step": step,
                                                      "path": export, "thread": thread})
            self.emit("checkpoint_save", payload)

        return self._method(target, original, before, after)

    def _export_manifest(self, export):
        manifest = _directory_manifest(export, self.config["max_checkpoint_file_bytes"])
        if manifest.get("files") is not None:
            manifest["files"] = self.bounded(manifest["files"])
        return manifest

    def settle_exports(self, trigger):
        """Re-hash pending exports whose writer has finished (A8-C-03).

        ``step_start`` never blocks: a writer still alive stays pending. ``shutdown`` joins
        each writer (bounded) and records every remaining export; one still being written is
        recorded as ``pending_async_write``, never ``hashed``. When no writer handle was found,
        the export is hashed once no thread named ``EXPORT_THREAD_NAME`` is alive and is
        marked ``synchronized: false``.

        Each export is settled in isolation (A8R2): an exception while settling one (e.g. a
        file pruned mid-walk) is recorded as ``checkpoint_export_settle_error`` and that export
        stays pending; any export not yet settled when an exception escapes (a fatal sink
        failure) is restored to the pending list.
        """
        final = trigger == "shutdown"
        with self._export_lock:
            pending, self._pending_exports = self._pending_exports, []
        keep, remaining = [], list(pending)
        try:
            while remaining:
                item = remaining[0]
                try:
                    export = self._settle_export(item, final)
                except CaptureFatal:
                    raise
                except Exception as exc:  # noqa: BLE001 - recorded; the export stays pending
                    self.emit("checkpoint_export_settle_error", {
                        "save_call": item["call"], "current_step": item["step"],
                        "path": str(item["path"]), "trigger": trigger, "final": final,
                        "error_type": type(exc).__name__, "message": self.text(exc)})
                    export = None
                if export is None:
                    keep.append(remaining.pop(0))
                    continue
                self.emit("checkpoint_export_settled", {
                    "save_call": item["call"], "current_step": item["step"], "trigger": trigger,
                    "final": final, "export": export})
                remaining.pop(0)
        finally:
            with self._export_lock:
                self._pending_exports[:0] = keep + remaining

    def _settle_export(self, item, final):
        """The settled export record for one pending export, or None while it stays pending."""
        writers = ([item["thread"]] if item["thread"] is not None else
                   [t for t in threading.enumerate() if t.name == EXPORT_THREAD_NAME])
        for writer in writers:
            if final:
                writer.join(self.export_join_timeout)
        if any(writer.is_alive() for writer in writers):
            if not final:
                return None
            return _pending_export(item["path"], "writer_thread_alive_at_shutdown")
        return {"path": str(item["path"]), "synchronized": item["thread"] is not None,
                **self._export_manifest(item["path"])}

    def _wrap_runtime_bridge(self, target, original):
        def after(call, a, result, ctx):
            report = self.install(CHAINED_TARGETS)
            self.emit("chained_wrap_installed", {"call": call, **report})

        wrapper = self._method(target, original, None, after)

        @functools.wraps(original)
        def strict(*args, **kwargs):
            result = wrapper(*args, **kwargs)
            for chained in CHAINED_TARGETS:  # fail closed if the chained target is absent
                _owner, current = self._resolve(chained)
                if getattr(current, SENTINEL, None) != chained.target_id:
                    raise CaptureInstallError(f"{chained.target_id} was not wrapped")
            return result

        return strict

    def _wrap_load_hf_weights(self, target, original):
        def after(call, a, result, ctx):
            path = str(a["model_name_or_path"])
            file_keys = _checkpoint_keys(path)
            model = getattr(a["self"], "expert_model", None)
            model_keys = set(model.state_dict().keys()) if model is not None else None
            payload = {"call": call, "model_name_or_path": path,
                       "file_keys": None if file_keys is None else len(file_keys),
                       "model_keys": None if model_keys is None else len(model_keys)}
            if file_keys is not None and model_keys is not None:
                payload["missing_in_file"] = self.bounded(sorted(model_keys - file_keys))
                payload["unexpected_in_file"] = self.bounded(sorted(file_keys - model_keys))
            self.emit("weights_loaded", payload)

        return self._method(target, original, None, after)

    def _wrap_runtime_stub(self, target, original_class):
        capture = self

        class CapturingStubFactory:
            """Constructs the upstream stub and returns a recording proxy around it."""

            def __new__(cls, *args, **kwargs):
                return _SimulateProxy(original_class(*args, **kwargs), capture)

        CapturingStubFactory.__wrapped__ = original_class
        return CapturingStubFactory

    # ---------- parameters ----------

    def _snapshot(self, trainer, phase, trainer_step):
        model = getattr(trainer, "model", None)
        if model is None:
            raise AttributeError("trainer.model missing")
        self.emit("parameter_snapshot", build_parameter_snapshot(
            model, getattr(trainer, "optimizers", None), phase=phase, trainer_step=trainer_step,
            hashing=self.config["parameter_hashing"]))

    def _scheduler_event(self, call, trainer, phase, trainer_step):
        state = scheduler_state(getattr(trainer, "lr_schedulers", None))
        self.emit("scheduler_state", {"call": call, "phase": phase,
                                      "trainer_step": trainer_step, **state})


class _LockProbe:
    """Delegates to a worker's original ``threading.Lock``; snapshots for an armed thread.

    The failing thread arms ``tls.probe`` before calling pinned ``_on_rollout_failed``; on its
    first acquire the probe records ``permanently_failed`` and on the matching release (still
    holding the lock) ``permanently_failed`` and ``retries_left``. Other threads, and later
    acquisitions by the same thread, pass straight through. Mutual exclusion is the original
    lock's: threads that captured the original object before the swap still use it.
    """

    def __init__(self, lock, tls):
        self._physloop_inner = lock
        self._physloop_tls = tls

    def _probe(self):
        return getattr(self._physloop_tls, "probe", None)

    def acquire(self, *args, **kwargs):
        acquired = self._physloop_inner.acquire(*args, **kwargs)
        probe = self._probe()
        if acquired and probe is not None and probe["enter"] is None:
            try:
                probe["enter"] = {"permanently_failed": bool(probe["state"].permanently_failed)}
            except Exception as exc:  # noqa: BLE001 - never break the upstream lock
                probe["error"] = f"enter snapshot failed: {type(exc).__name__}"
        return acquired

    def release(self):
        probe = self._probe()
        if probe is not None and probe["enter"] is not None and probe["exit"] is None:
            try:
                state = probe["state"]
                probe["exit"] = {"permanently_failed": bool(state.permanently_failed),
                                 "retries_left": int(state.retries_left)}
            except Exception as exc:  # noqa: BLE001 - never break the upstream lock
                probe["error"] = f"exit snapshot failed: {type(exc).__name__}"
        self._physloop_inner.release()

    def locked(self):
        return self._physloop_inner.locked()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc_info):
        self.release()
        return False


class _SimulateProxy:
    """Forwards to the upstream stub; ``simulate`` records request/reply bytes per session."""

    def __init__(self, stub, capture):
        self._stub = stub
        self._capture = capture

    def __getattr__(self, name):
        return getattr(self._stub, name)

    def simulate(self, request, *args, **kwargs):
        cap = self._capture
        call = cap.next_call()
        payload = {"call": call}
        try:
            payload["sessions"] = [str(s) for spec in request.rollout_specs
                                   for s in spec.session_uuids]
            if cap.config["capture_simulator_bytes"]:
                payload["request_blob"] = cap._blob(request.SerializeToString())
        except Exception as exc:  # noqa: BLE001 - recorded as capture_extract_error
            cap.extract_error("runtime_stub", "before", exc)
        try:
            reply = self._stub.simulate(request, *args, **kwargs)
        except CaptureFatal:
            raise
        except BaseException as exc:
            cap.emit("simulate_raised", {**payload, "error_type": type(exc).__name__,
                                         "message": cap.text(exc)})
            raise
        try:
            payload["returns"] = [{
                "sessions": [str(s) for s in ret.rollout_spec.session_uuids],
                "success": bool(ret.success), "error": cap.text(ret.error) if ret.error else "",
                "rollout_uuid": str(ret.rollout_uuid),
                "timestep_metrics": sorted(str(m.name) for m in ret.timestep_metrics),
            } for ret in reply.rollout_returns]
            if cap.config["capture_simulator_bytes"]:
                payload["reply_blob"] = cap._blob(reply.SerializeToString())
        except Exception as exc:  # noqa: BLE001 - recorded as capture_extract_error
            cap.extract_error("runtime_stub", "after", exc)
        cap.emit("simulate_call", payload)
        return reply


def _check_signature(target, original):
    try:
        names = tuple(inspect.signature(original).parameters)
    except (TypeError, ValueError) as exc:
        raise CaptureInstallError(f"{target.target_id}: no inspectable signature") from exc
    if names != target.params:
        raise CaptureInstallError(f"{target.target_id}: signature {names} differs from the "
                                  f"pinned {target.params}")


def _json_scalar(value):
    if value is None or type(value) in (bool, int, str):
        return value
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    if type(value) is bool or type(value) is int:
        return value
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def _float_map(mapping):
    if mapping is None:
        return None
    result = {}
    for key, value in sorted(dict(mapping).items(), key=lambda item: str(item[0])):
        try:
            result[str(key)] = _json_scalar(value)
        except (TypeError, ValueError):
            result[str(key)] = {"unconverted": type(value).__name__}
    return result


def _tensor_values(tensor):
    if tensor is None:
        return []
    if hasattr(tensor, "detach"):
        return tensor.detach().float().cpu().reshape(-1).tolist()
    return list(tensor)


# ---------- parameter snapshot v2 (rule 11; F5) ----------


def rule11_dtype(dtype):
    """The rule-11 safetensors name of a torch dtype (``torch.bfloat16`` -> ``BF16``) or raise."""
    name = str(dtype).removeprefix("torch.")
    if name not in RULE11_DTYPES:
        raise ValueError(f"dtype {dtype} is outside rule 11")
    return RULE11_DTYPES[name]


def _is_dtensor(value):
    return all(hasattr(value, name) for name in ("to_local", "placements", "device_mesh"))


def _placement_text(placement):
    """``Replicate()`` / ``Shard(dim=d)`` for exactly those torch types, else raise.

    Type identity, never ``is_shard()``: torch's ``_StridedShard`` is a ``Shard`` subclass
    (``is_shard()`` True) whose local rows are not contiguous (a lone
    ``_StridedShard(0, split_factor=2)`` at coordinate 0 of 2 holds global rows [0, 1, 4, 5]
    of 8), so the chunk rule's offset would be a guess. A Partial holds no stored values.
    """
    from torch.distributed.tensor.placement_types import Replicate, Shard  # worker-only

    if type(placement) is Replicate:
        return "Replicate()"
    if type(placement) is Shard:
        return f"Shard(dim={int(placement.dim)})"
    raise ValueError(f"placement {placement!r} ({type(placement).__name__}) is not exactly "
                     "Shard or Replicate: a Partial holds no stored values and a Shard "
                     "subclass (e.g. _StridedShard) has no contiguous chunk offset")


def shard_layout(dtensor, local):
    """``{global_shape, offsets, placement}`` of this rank's local shard, checked (L1/L2).

    Supports exactly-typed Replicate and at most one exactly-typed ``Shard(dim)`` per tensor
    dimension (no subclass such as ``_StridedShard``; see ``_placement_text``), with torch's
    chunk rule (chunk ``ceil(global / mesh_size)``; offset ``min(chunk * coordinate, global)``;
    extent ``min(global, offset + chunk) - offset``). The computed extent must equal the local
    tensor's; anything else raises, so offsets are never guessed.
    """
    global_shape = [int(d) for d in dtensor.shape]
    local_shape = [int(d) for d in local.shape]
    if len(local_shape) != len(global_shape):
        raise ValueError("local shard rank differs from the global rank")
    placements = tuple(dtensor.placements)
    mesh = dtensor.device_mesh
    coordinate = mesh.get_coordinate()
    if coordinate is None or len(coordinate) != len(placements):
        raise ValueError("this rank holds no coordinate on the tensor's device mesh")
    offsets, sharded, texts = [0] * len(global_shape), set(), []
    for mesh_dim, placement in enumerate(placements):
        texts.append(_placement_text(placement))
        if texts[-1] == "Replicate()":
            continue
        dim = int(placement.dim)
        dim = dim + len(global_shape) if dim < 0 else dim
        if not 0 <= dim < len(global_shape) or dim in sharded:
            raise ValueError("nested or out-of-range Shard placement is not supported")
        sharded.add(dim)
        size, index = int(mesh.size(mesh_dim)), int(coordinate[mesh_dim])
        chunk = -(-global_shape[dim] // size)
        start = min(chunk * index, global_shape[dim])
        extent = min(global_shape[dim], start + chunk) - start
        if local_shape[dim] != extent:
            raise ValueError(f"local extent {local_shape[dim]} on dim {dim} differs from the "
                             f"chunk rule's {extent}")
        offsets[dim] = start
    for dim, (lsize, gsize) in enumerate(zip(local_shape, global_shape, strict=True)):
        if dim not in sharded and lsize != gsize:
            raise ValueError(f"unsharded dim {dim} has local size {lsize} != global {gsize}")
    return {"global_shape": global_shape, "offsets": offsets,
            "placement": "[" + ", ".join(texts) + "]"}


def _local_view(tensor):
    """(tensor whose bytes are hashed, shard layout or None); DTensor -> ``to_local()``."""
    value = tensor.detach() if callable(getattr(tensor, "detach", None)) else tensor
    if not _is_dtensor(value):
        return value, None
    local = value.to_local()
    if _is_dtensor(local):
        raise ValueError("to_local() returned a distributed tensor")
    return local, shard_layout(value, local)


def rule11_sha256(local, dtype, shape):
    """Rule-11 hash of a plain local tensor via ``gate.tensor_manifest.tensor_record``.

    Lazy imports: torch exists only inside the trainer process, and ``gate`` must be in the
    staged capture tree (module docstring). Non-contiguous, negated or conjugated views are
    materialized first, so the hash is of the logical values in row-major order.
    """
    if sys.byteorder != "little":
        raise ValueError("rule 11 is little-endian; this host is not")
    import torch  # runs only inside the worker, never at module import

    from gate.tensor_manifest import tensor_record  # rule 11; never re-derived here

    if getattr(local, "is_quantized", False) or getattr(local, "layout", None) != torch.strided:
        raise ValueError("only strided, non-quantized tensors have rule-11 bytes")
    if getattr(getattr(local, "device", None), "type", None) == "meta":
        raise ValueError("meta tensors carry no bytes")
    plain = local.cpu().resolve_conj().resolve_neg().contiguous()
    raw = plain.reshape(-1).view(torch.uint8).numpy()
    return tensor_record(dtype, list(shape), raw)["sha256"]


def tensor_entry(tensor, kind, hashing):
    """One v2 ``tensors`` entry (``kind`` is ``param`` or ``buffer``)."""
    local, shard = _local_view(tensor)
    dtype = rule11_dtype(local.dtype)
    shape = [int(d) for d in local.shape]
    entry = {"kind": kind, "dtype": dtype, "shape": shape, "numel": math.prod(shape),
             "requires_grad": bool(tensor.requires_grad), "sha256": None, "hash_scope": None,
             "shard": shard}
    if hashing == "sha256_local":
        entry["sha256"] = rule11_sha256(local, dtype, shape)
        entry["hash_scope"] = "full" if shard is None else "local_shard"
    return entry


def _inner_optimizers(optimizers):
    """The optimizers holding state: a Cosmos ``OptimizersContainer`` (``optimizers``
    attribute) is iterated; any other object is one optimizer."""
    if optimizers is None:
        raise ValueError("trainer.optimizers is missing")
    inner = list(iter(optimizers)) if hasattr(optimizers, "optimizers") else [optimizers]
    for optimizer in inner:
        if type(getattr(optimizer, "param_groups", None)) is not list or not hasattr(
                getattr(optimizer, "state", None), "get"):
            raise ValueError("an optimizer lacks param_groups (list) or state (mapping)")
    return inner


def _step_count(value):
    raw = value.item() if callable(getattr(value, "item", None)) else value
    if type(raw) is bool or type(raw) not in (int, float) or not math.isfinite(raw) or (
            raw < 0 or raw != int(raw)):
        raise ValueError(f"optimizer step {raw!r} is not a finite non-negative integral count")
    return int(raw)


def _state_digest(value, param_sharded, hashing):
    if callable(getattr(value, "detach", None)):
        local, shard = _local_view(value)
        if shard is not None and not param_sharded:
            raise ValueError("a sharded optimizer state entry belongs to an unsharded parameter")
        if hashing != "sha256_local":
            return None
        shape = [int(d) for d in local.shape]
        return rule11_sha256(local, rule11_dtype(local.dtype), shape)
    scalar = _json_scalar(value)
    if hashing != "sha256_local":
        return None
    return hashlib.sha256(canonical_bytes(scalar)).hexdigest()


def build_parameter_snapshot(model, optimizers, *, phase, trainer_step, hashing):
    """The v2 ``parameter_snapshot`` payload (module docstring); raises on any defect."""
    if phase not in SNAPSHOT_PHASES:
        raise ValueError(f"phase must be one of {SNAPSHOT_PHASES}")
    if (trainer_step is None) != (phase == "init") or (
            trainer_step is not None and (type(trainer_step) is not int or trainer_step < 0)):
        raise ValueError("trainer_step is null exactly at init and otherwise an int >= 0")
    if hashing not in HASHING_MODES:
        raise ValueError(f"hashing must be one of {HASHING_MODES}")
    named = list(model.named_parameters())
    tensors = {}
    for name, param in named:
        if name in tensors:
            raise ValueError(f"parameter name {name!r} repeated")
        tensors[name] = tensor_entry(param, "param", hashing)
    for name, buffer in model.named_buffers():
        if name in tensors:
            raise ValueError(f"{name!r} is both a parameter and a buffer")
        tensors[name] = tensor_entry(buffer, "buffer", hashing)
    if not tensors:
        raise ValueError("the model has no parameter or buffer to snapshot")
    names = {id(param): name for name, param in named}
    optimizer_state, step_counters, applied_lr = {}, {}, []
    for optimizer in _inner_optimizers(optimizers):
        for group in optimizer.param_groups:
            applied_lr.append(_json_scalar(group["lr"]))
            for param in group["params"]:
                name = names.get(id(param))
                if name is None:
                    raise ValueError("an optimizer member is not a model.named_parameters() entry")
                if name in optimizer_state:
                    raise ValueError(f"{name!r} is in two param groups")
                state = optimizer.state.get(param) or {}
                sharded = tensors[name]["shard"] is not None
                optimizer_state[name] = {
                    str(key): _state_digest(value, sharded, hashing)
                    for key, value in sorted(state.items(), key=lambda item: str(item[0]))}
                step_counters[name] = _step_count(state["step"]) if "step" in state else 0
    return {"schema_version": SNAPSHOT_SCHEMA_VERSION, "phase": phase,
            "trainer_step": trainer_step, "tensors": dict(sorted(tensors.items())),
            "training_flags": dict(sorted((str(name), bool(module.training))
                                          for name, module in model.named_modules())),
            "optimizer_state": dict(sorted(optimizer_state.items())),
            "step_counters": dict(sorted(step_counters.items())), "applied_lr": applied_lr}


def scheduler_state(lr_schedulers):
    """``{last_epoch, lrs}``; a Cosmos ``LRSchedulersContainer`` is iterated (L1/L2)."""
    if lr_schedulers is None:
        raise ValueError("trainer.lr_schedulers is missing")
    inner = (list(iter(lr_schedulers)) if hasattr(lr_schedulers, "schedulers")
             else [lr_schedulers])
    if not inner:
        raise ValueError("no learning-rate scheduler")
    epochs, lrs = set(), []
    for scheduler in inner:
        epoch = scheduler.last_epoch
        if type(epoch) is not int:
            raise ValueError("scheduler last_epoch is not an int")
        epochs.add(epoch)
        lrs.extend(_json_scalar(value) for value in scheduler.get_last_lr())
    if len(epochs) != 1:
        raise ValueError(f"schedulers disagree on last_epoch: {sorted(epochs)}")
    return {"last_epoch": epochs.pop(), "lrs": lrs}


def trainer_settings(trainer):
    """Replay geometry recorded from the pinned trainer attributes; absent -> ``None``."""
    settings = {}
    for field, attribute in TRAINER_SETTINGS:
        value = getattr(trainer, attribute, None)
        settings[field] = value if type(value) is int else None
    beta = getattr(trainer, "_kl_beta", None)
    settings["kl_beta"] = _json_scalar(beta) if type(beta) in (int, float) else None
    return settings


def _hash_file(path, limit):
    """Hash a completion artifact before upstream may unlink it (no-follow, bounded)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        return {"status": "unreadable", "error": type(exc).__name__}
    try:
        digest, total = hashlib.sha256(), 0
        while chunk := os.read(fd, 1 << 20):
            total += len(chunk)
            if total > limit:
                return {"status": "over_limit", "bytes_read": total}
            digest.update(chunk)
        return {"status": "hashed", "sha256": digest.hexdigest(), "bytes": total}
    finally:
        os.close(fd)


def _pending_export(export, reason):
    found = reason != "writer_thread_not_found"
    return {"path": str(export), "status": "pending_async_write", "reason": reason,
            "writer_thread": EXPORT_THREAD_NAME if found else None, "files": None}


def _directory_manifest(directory, limit):
    """Hash every regular file under ``directory`` (no-follow, bounded per file, sorted)."""
    root = Path(directory)
    try:
        info = os.lstat(root)
    except FileNotFoundError:
        return {"status": "missing", "files": None}
    if not stat.S_ISDIR(info.st_mode):
        return {"status": "not_a_directory", "files": None}
    files = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in sorted(filenames) + [d for d in dirnames
                                         if os.path.islink(os.path.join(dirpath, d))]:
            full = Path(dirpath) / name
            relative = full.relative_to(root).as_posix()
            mode = os.lstat(full).st_mode
            if stat.S_ISLNK(mode):
                files.append({"path": relative, "status": "symlink"})
            elif stat.S_ISREG(mode):
                files.append({"path": relative, **_hash_file(str(full), limit)})
            else:
                files.append({"path": relative, "status": "special_file"})
    files.sort(key=lambda item: item["path"])
    complete = all(item.get("status") == "hashed" for item in files)
    return {"status": "hashed" if files and complete else "incomplete" if files else "empty",
            "manifest_sha256": hashlib.sha256(canonical_bytes(files)).hexdigest(),
            "files": files}


def _checkpoint_keys(path):
    """Tensor names from ``model.safetensors.index.json`` or a single file header (stdlib)."""
    root = Path(path)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        return set(json.loads(index.read_bytes())["weight_map"])
    single = root / "model.safetensors"
    if not single.is_file():
        return None
    with single.open("rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        if size <= 0 or size > 100_000_000:
            raise ValueError("implausible safetensors header length")
        header = json.loads(handle.read(size))
    return {key for key in header if key != "__metadata__"}


def capture_directory_from_toml(config_path):
    """``<run_dir>/physloop_capture`` from ``custom.resolved_config_path`` in the Cosmos TOML."""
    document = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    resolved = (document.get("custom") or {}).get("resolved_config_path")
    if type(resolved) is not str or not Path(resolved).is_absolute():
        raise CaptureInstallError("Cosmos TOML lacks an absolute custom.resolved_config_path")
    return Path(resolved).parent / CAPTURE_SUBDIR


def is_credential_name(name):
    """Whether an environment name is a credential (agent technical default, D-018).

    Exact ``CREDENTIAL_NAMES`` or a ``CREDENTIAL_SUFFIXES`` ending, compared upper-cased;
    ``PHYSLOOP_*`` names are PhysLoop's own configuration and never credentials.
    """
    upper = name.upper()
    return not upper.startswith("PHYSLOOP_") and (
        upper in CREDENTIAL_NAMES or upper.endswith(CREDENTIAL_SUFFIXES))


def apply_worker_credentials(environ, mode):
    """Apply the required worker credential mode to this role's environment; names only.

    ``strip_offline`` removes every variable for which ``is_credential_name`` holds and sets
    ``HF_HUB_OFFLINE=1`` (gated Hugging Face files must then already be in the cache: an
    L3 qualification item). ``inherit_disclosed`` changes nothing and records which
    credential-named variables the role inherited. No default: a missing or unknown mode
    refuses. Values are never read into the record.
    """
    if mode not in WORKER_CREDENTIAL_MODES:
        raise CaptureInstallError(f"{ENV_WORKER_CREDENTIALS} must be one of "
                                  f"{list(WORKER_CREDENTIAL_MODES)} (no default)")
    names = sorted(name for name in environ if is_credential_name(name))
    if mode == "inherit_disclosed":
        return {"mode": mode, "present_credential_names": names}
    for name in names:
        del environ[name]
    environ["HF_HUB_OFFLINE"] = "1"
    return {"mode": mode, "removed_names": names, "set": {"HF_HUB_OFFLINE": "1"}}


def main(argv=None, *, environ=None, import_module=None, sink_factory=EventSink,
         on_fatal=_default_fatal):
    """Install capture, then hand control to the upstream entrypoint with the same argv."""
    environ = os.environ if environ is None else environ
    import_module = import_module or importlib.import_module
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    config_path = environ.get(ENV_CONFIG)
    if not config_path:
        raise CaptureInstallError(f"{ENV_CONFIG} is not set; capture refuses to run")
    capture_config = load_capture_config(config_path)
    directory = capture_directory_from_toml(args.config)
    role = environ.get("COSMOS_ROLE") or "unknown"
    if not _ROLE.fullmatch(role):
        raise CaptureInstallError(f"unexpected COSMOS_ROLE {role!r}")
    if not environ.get("PYTHONDONTWRITEBYTECODE"):
        raise CaptureInstallError("PYTHONDONTWRITEBYTECODE is not set: this role would write "
                                  "bytecode into the verified capture tree")
    credentials = apply_worker_credentials(environ, environ.get(ENV_WORKER_CREDENTIALS))
    label = f"{role.lower()}-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    sink = sink_factory(directory, stream_label=label, limits=capture_config["limits"])
    upstream = import_module(UPSTREAM_ENTRYPOINT)
    capture = Capture(sink, capture_config, on_fatal=on_fatal, import_module=import_module)
    report = capture.install()
    capture.emit("capture_installed", {
        "role": role, "targets": [t.target_id for t in WRAP_TARGETS],
        "chained_targets": [t.target_id for t in CHAINED_TARGETS], **report,
        "worker_credentials": credentials,
        "capture_config_sha256": hashlib.sha256(canonical_bytes(capture_config)).hexdigest(),
        "capture_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    try:
        return upstream.main(argv)
    finally:
        try:
            capture.settle_exports("shutdown")
        finally:
            sink.close({"role": role})


if __name__ == "__main__":
    main()
