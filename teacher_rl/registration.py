"""S4 run registration schema v2, fail-closed validation and upstream override compilation.

Tickets: S4.1 (freeze the implemented learning problem; every ``02`` S4 ``____`` learning-run
slot), D-023 slots (scene roles and overlap rule, evaluated policy path, improvement decision
rule, smallest improvement of interest), lane-r3 §4-§6 (shipped presets, interface anchors, H1
configuration route), module F4 (completeness audit "S4.6 superiority binding" and "S4.1 issue
#10 slot"). Agent-authored under D-018/D-024; engineering preparation only.

F4 owner slots (required, no defaults):

- ``improvement_decision_rule`` = ``{rule_id, registration_ref, guards, guard_strata,
  gating_strata, look}``. ``guards`` is an explicit list (``[]`` registers "no guard") of
  ``{name (slug), s2_registration_id, metric {name, kind, higher_is_worse, unit}, margin in
  (0, 1), alpha float in (0, 1), s2_protocol_hash}``: each guard endpoint is evaluated by its
  own S2 registration and bound to these values at aggregation (``teacher_rl.evaluate``).
  ``s2_protocol_hash`` (64 lowercase hex; review F4-03) is
  ``gate.s2_identity.identity_hash`` of that S2 registration without its run-specific
  ``candidate`` (the convention of ``teacher_rl.evaluate._s2_protocol_hash``), so the guard's
  score definition and every other registered field are pinned by bytes, as the primary is
  through its request; one hash covers every run because only the candidate differs. ``guard_strata`` is
  ``"primary_strata"`` or a nonempty list of distinct strata; ``gating_strata`` is
  ``"all_guard_strata"`` or a nonempty list that must be a subset of a listed
  ``guard_strata`` (checked against the S2 strata at aggregation when the guard strata are the
  primary's). With ``guards == []`` the two strata slots must hold the inert tokens
  (``guard_strata_without_guards`` otherwise; agent technical default (D-018): nothing to
  gate, so no strata are registered). ``look`` is ``{"kind": "fixed"}`` or
  ``{"kind": "sequential", "look_index": k >= 1}``; whether the evaluation glue can take that
  look is decided at aggregation, not here.
- ``response_to_alpagym_issue_10`` = ``{response, record_ref}`` with ``response`` one of
  ``accept_as_shipped`` (train with the shipped dtypes; lane-r3 finding 4: trainable weights
  and Adam state may be bf16 at runtime, unverified, recorded by E03) or ``change_request``
  (a recorded request to change the trainable master dtype). It is harness-only; a change
  request cannot be compiled onto the pinned trainer (no upstream key; casting masters is an
  owner training-method decision that needs a new pin), so ``compile_overrides`` blocks with
  ``alpagym_issue_10_change_not_compilable``.

Schema versions: v2 (G3) is v1 plus the required guard field ``s2_protocol_hash`` (review
F4-03), which v1 lacked while keeping version 1; only v2 validates (a v1 document is
``unsupported_schema_version``, never silently upgraded). The compiled ``s4_effective_config``
keeps its own ``EFFECTIVE_CONFIG_SCHEMA_VERSION`` 1 (its shape did not change; consumed by
``teacher_rl.records``).

Owner slots are required inputs with **no defaults**. This module checks types, upstream domains
and internal consistency; it never chooses a learning setting, reward arm, tolerance, scene,
role, rule or prediction, and never declares a registration scientifically admitted.

Verification ceiling:
- ``validate_registration``: **L1** (pure structure/domain checks; literal known answers and
  negative controls in ``tests/test_s4_registration.py``). Its batch-geometry rule is **L2**:
  it agrees with the pinned host ``_validate_cosmos_grpo_batch_geometry`` executed from the
  vendored ``config_validation.py`` on the same shapes. ``engineering.run_root`` must be
  absolute (the pinned default is relative and the capture entrypoint cannot locate the run).
- ``compile_overrides``: **L2**. Every mapped key cites the pinned AlpaGym Hydra schema field
  (``alpagym_host/config.py`` dataclasses) or the pinned Cosmos-RL config field
  (``cosmos_rl/policy/config/__init__.py``); the tests resolve every cited key against the
  vendored, hash-checked copies. That Hydra composes these overrides as intended, and that
  ``launch_all`` round-trips unknown TOML keys on the H1 route, is **L3** (``qualify_on_host``).

Field categories in the compiled mapping:
- ``hydra_key``: an AlpaGym Hydra schema field or config-group choice exists.
- ``cosmos_toml_key``: only a Cosmos-RL config field exists; allowed only when the registration
  declares ``config_route = "h1_resolved_toml"`` (R3 §6 H1), unless the registered value equals
  the pinned Cosmos default, in which case nothing is overridden and the match is recorded.
- ``fixed_by_upstream``: the pinned code offers no setting; only the shipped behaviour compiles,
  any other value is refused as ``no_upstream_key`` (simulator session seeds; decode device).
- ``harness_only``: PhysLoop evaluator inputs with no upstream key by nature (scene roles,
  overlap rule, evaluated path, decision rule, E07 criterion, parity tolerances, run count,
  run aggregation, prediction reference); recorded, never passed to upstream.

This module never reads the network, a provider, a credential, a clock or the filesystem (other
than hashing its own source for producer identity).
"""

import hashlib
import math
import re
from pathlib import Path, PurePosixPath

from gate.s2_identity import identity_hash
from gate.superiority import RUN_RULES as RUN_AGGREGATION_RULES
from teacher_rl import pins

SCHEMA_VERSION = 2  # s4_run_registration (v2 adds guard s2_protocol_hash; G3)
EFFECTIVE_CONFIG_SCHEMA_VERSION = 1  # s4_effective_config, unchanged
ARTIFACT_KIND = "s4_run_registration"
SCOPES = ("synthetic", "public_source_fixture", "actual")
CONFIG_ROUTES = ("hydra_only", "h1_resolved_toml")
PLACEHOLDERS = {"tbd", "todo", "later", "____", "placeholder", "xxx", "n/a", "?"}
DECAY_TYPES = ("sqrt", "cosine", "linear", "none")  # Cosmos TrainingConfig.optm_decay_type
SIMULATOR_SEED_MODE = "upstream_random_unsettable"
DECODE_DEVICE = "cuda_nvjpeg"
REWARD_ARMS = {"A": "progress_safety"}  # baseline A is the shipped preset (02 S4.3)
PARITY_KEYS = ("log_density", "action_trajectory")
DECISION_RULE_FIELDS = ("rule_id", "registration_ref", "guards", "guard_strata",
                        "gating_strata", "look")
GUARD_FIELDS = ("name", "s2_registration_id", "metric", "margin", "alpha", "s2_protocol_hash")
METRIC_FIELDS = ("name", "kind", "higher_is_worse", "unit")
METRIC_KINDS = ("binary", "continuous")
PRIMARY_STRATA = "primary_strata"
ALL_GUARD_STRATA = "all_guard_strata"
LOOK_KINDS = ("fixed", "sequential")
ISSUE_10_RESPONSES = ("accept_as_shipped", "change_request")
_I10 = ("lane-r3 finding 4; third-party AlpaGym issue #10 (2026-11-02); cosmos_rl/policy/model/"
        "base.py:625-630; models/base_model.py:373-381; rl/utils/fsdp.py:78-85")

_SLUG = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_SCENE = re.compile(r"clipgt-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_PATH = re.compile(r"/[A-Za-z0-9._/-]*")
_PREDICTION = re.compile(r"experiments/\d{4}-\d{2}-\d{2}-[a-z0-9][a-z0-9-]*\.md")
_TEXT = re.compile(r"[^\x00-\x1f\x7f]{1,400}")

_G_CONFIG = "alpagym:packages/host/src/alpagym_host/config.py"
_C_CONFIG = "cosmos-rl:cosmos_rl/policy/config/__init__.py"
_G_REWARD = "alpagym:packages/host/src/alpagym_host/conf/reward/progress_safety.yaml"
_G_POLICY = ("alpagym:packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/configs/policy/"
             "alpamayo_r1.yaml")
_A_BASE = "alpasim-gympin:src/wizard/configs/base_config.yaml"
_G_GEOMETRY = ("alpagym:packages/host/src/alpagym_host/config_validation.py:"
               "_validate_cosmos_grpo_batch_geometry (lines 332-371; called unconditionally by "
               "validate_run_config)")
_G_RUN_ROOT = ("alpagym:packages/host/src/alpagym_host/conf/default.yaml:17 (run_root: "
               "./tmp/alpagym-runs); run_artifacts.py build_artifact_paths and "
               "custom.resolved_config_path")

# Cosmos defaults for fields AlpaGym's schema does not expose (cited Cosmos fields).
COSMOS_TOML_DEFAULTS = {
    "train.optm_warmup_start_factor": (0.0, f"{_C_CONFIG}:TrainingConfig.optm_warmup_start_factor"),
    "train.seed": (None, f"{_C_CONFIG}:TrainingConfig.seed"),
    "train.train_policy.dataloader_seed": (0, f"{_C_CONFIG}:GrpoConfig.dataloader_seed"),
}

OWNER_FIELDS = (
    "group_size", "rollouts_per_trainer_step", "learning_rate", "lr_schedule", "warmup",
    "scene_set", "optimizer_steps_per_run", "checkpoint_retention", "anchor", "scene_roles",
    "overlap_rule", "evaluated_policy_path", "improvement_decision_rule",
    "smallest_improvement_of_interest", "seed_modes", "version_policy", "reward_arm",
    "e07_pass_criterion", "parity_tolerances", "video", "decode_device",
    "independent_training_runs", "run_aggregation", "prediction_ref",
    "response_to_alpagym_issue_10",
)
ENGINEERING_FIELDS = (
    "pins", "cosmos_reason2_revision", "base_experiment", "policy_model_path", "run_root",
    "config_route",
)
PIN_FIELDS = (
    "alpagym", "alpamayo_recipes", "cosmos_rl", "training_alpasim", "alpamayo_r1",
    "teacher_revision", "alpagym_uv_lock_sha256",
)
TOP_FIELDS = (
    "artifact_kind", "schema_version", "scope", "registration_id", "engineering", "owner",
)


class _Checker:
    """Collect issues while walking one registration; never raises on bad input."""

    def __init__(self):
        self.issues = []

    def add(self, code, path, message, severity="blocking"):
        self.issues.append({"code": code, "path": path, "severity": severity,
                            "evidence_refs": [], "message": message})

    def obj(self, value, path, fields):
        if type(value) is not dict:
            self.add("not_object", path, "expected an object")
            return None
        for key in sorted(set(value) - set(fields)):
            self.add("unknown_field", f"{path}.{key}", f"unknown field {key!r}")
        missing = [key for key in fields if key not in value]
        for key in missing:
            self.add("owner_slot_unregistered" if path == "$.owner" else "field_missing",
                     f"{path}.{key}", f"required field {key!r} is missing (no default)")
        return value

    def integer(self, value, path, minimum=None):
        if type(value) is not int:
            self.add("not_integer", path, "expected an integer")
            return False
        if minimum is not None and value < minimum:
            self.add("out_of_range", path, f"must be >= {minimum}")
            return False
        return True

    def number(self, value, path, low=None, high=None, low_open=False):
        if type(value) not in (int, float) or not math.isfinite(value):
            self.add("not_finite_number", path, "expected a finite number")
            return False
        if low is not None and (value < low or (low_open and value == low)):
            self.add("out_of_range", path, f"must be {'>' if low_open else '>='} {low}")
            return False
        if high is not None and value > high:
            self.add("out_of_range", path, f"must be <= {high}")
            return False
        return True

    def boolean(self, value, path):
        if type(value) is not bool:
            self.add("not_boolean", path, "expected true or false")
            return False
        return True

    def text(self, value, path):
        if type(value) is not str or not _TEXT.fullmatch(value) or value != value.strip():
            self.add("not_text", path, "expected a non-empty single-line string")
            return False
        if value.strip().lower() in PLACEHOLDERS or "____" in value:
            self.add("placeholder_value", path, "placeholder text is not a registered value")
            return False
        return True

    def choice(self, value, path, options, code="unsupported_value"):
        if value not in options:
            self.add(code, path, f"must be one of {list(options)}")
            return False
        return True

    def scenes(self, value, path, allow_empty=False):
        if type(value) is not list or (not value and not allow_empty):
            self.add("not_scene_list", path, "expected a list of clipgt scene IDs")
            return None
        good = True
        for index, scene in enumerate(value):
            if type(scene) is not str or not _SCENE.fullmatch(scene):
                self.add("malformed_scene_id", f"{path}[{index}]", "not a clipgt-<uuid> ID")
                good = False
        if good and len(set(value)) != len(value):
            self.add("duplicate_scene_id", path, "scene IDs must be unique")
            good = False
        return value if good else None


def _safe_absolute_path(path):
    if type(path) is not str or _SAFE_PATH.fullmatch(path) is None:
        return False
    pure = PurePosixPath(path)
    return str(pure) == path and ".." not in pure.parts and path != "/"


def _preset_for(eng):
    """The shipped preset named by ``engineering.base_experiment`` (None for any bad value)."""
    name = eng.get("base_experiment") if type(eng) is dict else None
    return pins.EXPERIMENT_PRESETS.get(name) if type(name) is str else None


def _check_engineering(chk, eng):
    if chk.obj(eng, "$.engineering", ENGINEERING_FIELDS) is None:
        return
    pin_record = eng.get("pins")
    if chk.obj(pin_record, "$.engineering.pins", PIN_FIELDS) is not None:
        expected = {name: pin["commit"] for name, pin in pins.UPSTREAM_PINS.items()}
        expected["teacher_revision"] = pins.TEACHER["revision"]
        expected["alpagym_uv_lock_sha256"] = pins.ALPAGYM_LOCK["uv_lock_sha256"]
        for name in PIN_FIELDS:
            if name in pin_record and pin_record[name] != expected[name]:
                chk.add("pin_mismatch", f"$.engineering.pins.{name}",
                        f"must equal the engineering pin {expected[name]}")
    if "cosmos_reason2_revision" in eng:
        value = eng["cosmos_reason2_revision"]
        if type(value) is not str or not _COMMIT.fullmatch(value):
            chk.add("cosmos_reason2_revision_unregistered", "$.engineering.cosmos_reason2_revision",
                    "a full 40-hex revision is an admission-time input (no default)")
    if "base_experiment" in eng:
        preset = _preset_for(eng)
        if preset is None or not preset["supported_base"]:
            chk.add("unsupported_base_experiment", "$.engineering.base_experiment",
                    "must be a shipped preset on the two-GPU local topology "
                    f"{[n for n, p in pins.EXPERIMENT_PRESETS.items() if p['supported_base']]}")
    if "policy_model_path" in eng and not _safe_absolute_path(eng["policy_model_path"]):
        chk.add("unsafe_model_path", "$.engineering.policy_model_path",
                "must be a normalized absolute POSIX path of [A-Za-z0-9._/-]")
    if "run_root" in eng and not _safe_absolute_path(eng["run_root"]):
        # The pinned default run_root is relative; Cosmos workers then get a relative
        # custom.resolved_config_path and the capture entrypoint cannot locate the run.
        chk.add("unsafe_run_root", "$.engineering.run_root",
                "must be a normalized absolute POSIX path of [A-Za-z0-9._/-] (the pinned "
                f"default is relative: {_G_RUN_ROOT})")
    if "config_route" in eng:
        chk.choice(eng["config_route"], "$.engineering.config_route", CONFIG_ROUTES)


def _check_owner(chk, owner, eng):
    if chk.obj(owner, "$.owner", OWNER_FIELDS) is None:
        return
    p = "$.owner"
    if ("group_size" in owner and chk.integer(owner["group_size"], f"{p}.group_size", 1)
            and owner["group_size"] < 2):
        chk.add("group_size_below_two", f"{p}.group_size",
                    "a group-relative advantage needs at least two candidates (02 S4.1)")
    if "rollouts_per_trainer_step" in owner:
        chk.integer(owner["rollouts_per_trainer_step"], f"{p}.rollouts_per_trainer_step", 1)
    if "learning_rate" in owner:
        chk.number(owner["learning_rate"], f"{p}.learning_rate", 0, low_open=True)
    sched = owner.get("lr_schedule")
    if "lr_schedule" in owner and chk.obj(
            sched, f"{p}.lr_schedule", ("decay_type", "decay_ratio", "min_lr_factor")):
        if "decay_type" in sched:
            chk.choice(sched["decay_type"], f"{p}.lr_schedule.decay_type", DECAY_TYPES)
        if "decay_ratio" in sched:
            chk.number(sched["decay_ratio"], f"{p}.lr_schedule.decay_ratio", 0, 1)
        if "min_lr_factor" in sched:
            chk.number(sched["min_lr_factor"], f"{p}.lr_schedule.min_lr_factor", 0, 1)
    warm = owner.get("warmup")
    if "warmup" in owner and chk.obj(warm, f"{p}.warmup", ("steps", "start_factor")):
        if "steps" in warm:
            chk.integer(warm["steps"], f"{p}.warmup.steps", 0)
        if "start_factor" in warm:
            chk.number(warm["start_factor"], f"{p}.warmup.start_factor", 0, 1)
    scene_set = chk.scenes(owner["scene_set"], f"{p}.scene_set") if "scene_set" in owner else None
    steps = owner.get("optimizer_steps_per_run")
    step_fields = ("trainer_steps", "epochs", "mini_batch_rows", "optimizer_iterations")
    if "optimizer_steps_per_run" in owner and chk.obj(
            steps, f"{p}.optimizer_steps_per_run", step_fields):
        for name in step_fields:
            if name in steps:
                chk.integer(steps[name], f"{p}.optimizer_steps_per_run.{name}", 1)
    ret = owner.get("checkpoint_retention")
    if "checkpoint_retention" in owner and chk.obj(
            ret, f"{p}.checkpoint_retention", ("save_freq", "max_keep", "export_safetensors")):
        if "save_freq" in ret:
            chk.integer(ret["save_freq"], f"{p}.checkpoint_retention.save_freq", 1)
        if ("max_keep" in ret and chk.integer(ret["max_keep"],
                                               f"{p}.checkpoint_retention.max_keep", -1)
                and ret["max_keep"] == 0):
            chk.add("out_of_range", f"{p}.checkpoint_retention.max_keep",
                        "must be >= 1, or -1 to keep every checkpoint (Cosmos semantics)")
        if "export_safetensors" in ret:
            chk.boolean(ret["export_safetensors"], f"{p}.checkpoint_retention.export_safetensors")
    _check_anchor(chk, owner)
    _check_roles(chk, owner, scene_set)
    for name in ("evaluated_policy_path",):
        if name in owner:
            chk.text(owner[name], f"{p}.{name}")
    _check_decision_rule(chk, owner)
    _check_issue_10(chk, owner)
    smallest = owner.get("smallest_improvement_of_interest")
    if "smallest_improvement_of_interest" in owner and chk.obj(
            smallest, f"{p}.smallest_improvement_of_interest", ("value", "unit")):
        if "value" in smallest:
            chk.number(smallest["value"], f"{p}.smallest_improvement_of_interest.value", 0,
                       low_open=True)
        if "unit" in smallest:
            chk.text(smallest["unit"], f"{p}.smallest_improvement_of_interest.unit")
    _check_seeds(chk, owner)
    _check_versions(chk, owner, eng)
    arm = owner.get("reward_arm")
    if ("reward_arm" in owner and chk.obj(arm, f"{p}.reward_arm", ("arm", "preset"))
            and (type(arm.get("arm")) is not str
                 or REWARD_ARMS.get(arm["arm"]) != arm.get("preset"))):
        chk.add("reward_arm_not_approved", f"{p}.reward_arm",
                    "schema v2 compiles only baseline A = shipped progress_safety; another arm "
                    "needs Huy's approved evaluator term and tests (02 S4.3)")
    _check_e07_and_parity(chk, owner)
    video = owner.get("video")
    if ("video" in owner and chk.obj(video, f"{p}.video", ("render_video",))
            and "render_video" in video):
        chk.boolean(video["render_video"], f"{p}.video.render_video")
    if "decode_device" in owner and owner["decode_device"] != DECODE_DEVICE:
        chk.add("no_upstream_key", f"{p}.decode_device",
                "the pinned decode device follows policy.model.device (cuda: nvJPEG; "
                "alpagym buffers.py ObservationBuffers.__init__); no decode-only setting exists")
    if ("independent_training_runs" in owner and chk.integer(
            owner["independent_training_runs"], f"{p}.independent_training_runs", 1)
            and owner["independent_training_runs"] < 3):
        chk.add("fewer_than_three_training_runs", f"{p}.independent_training_runs",
                    "02 S4.4 baseline needs at least three; fewer is an engineering smoke only",
                    severity="warning")
    _check_run_aggregation(chk, owner)
    if "prediction_ref" in owner:
        ref = owner["prediction_ref"]
        if type(ref) is not str or not _PREDICTION.fullmatch(ref):
            chk.add("malformed_prediction_ref", f"{p}.prediction_ref",
                    "must be experiments/<YYYY-MM-DD>-<slug>.md (the written prediction)")


def _strata(chk, value, path):
    """A nonempty list of distinct single-line strata names, or None (issue recorded)."""
    if type(value) is not list or not value or not all(
            type(name) is str and _TEXT.fullmatch(name) and name == name.strip()
            for name in value) or len(set(value)) != len(value):
        chk.add("not_strata_list", path, "expected a nonempty list of distinct strata names")
        return None
    return value


def _check_guard(chk, guard, path):
    if chk.obj(guard, path, GUARD_FIELDS) is None:
        return None
    name = guard.get("name")
    if "name" in guard and (type(name) is not str or not _SLUG.fullmatch(name)):
        chk.add("malformed_guard_name", f"{path}.name", "a guard name is a lowercase slug")
    if "s2_registration_id" in guard:
        chk.text(guard["s2_registration_id"], f"{path}.s2_registration_id")
    digest = guard.get("s2_protocol_hash")
    if "s2_protocol_hash" in guard and (type(digest) is not str
                                        or not _SHA256.fullmatch(digest)):
        chk.add("malformed_guard_protocol_hash", f"{path}.s2_protocol_hash",
                "the guard's S2 registration hash (without its candidate) is 64 lowercase hex")
    metric = guard.get("metric")
    if "metric" in guard and chk.obj(metric, f"{path}.metric", METRIC_FIELDS) is not None:
        for key in ("name", "unit"):
            if key in metric:
                chk.text(metric[key], f"{path}.metric.{key}")
        if "kind" in metric:
            chk.choice(metric["kind"], f"{path}.metric.kind", METRIC_KINDS)
        if "higher_is_worse" in metric:
            chk.boolean(metric["higher_is_worse"], f"{path}.metric.higher_is_worse")
    if "margin" in guard and chk.number(guard["margin"], f"{path}.margin", 0, 1,
                                        low_open=True) and guard["margin"] >= 1:
        chk.add("out_of_range", f"{path}.margin", "must be < 1")
    alpha = guard.get("alpha")
    if "alpha" in guard and (type(alpha) is not float or not 0 < alpha < 1):
        chk.add("guard_alpha_invalid", f"{path}.alpha",
                "alpha is a float strictly between 0 and 1 (owner input)")
    return name if type(name) is str else None


def _check_decision_rule(chk, owner):
    """``improvement_decision_rule``: rule identity plus the F4 superiority slots."""
    path = "$.owner.improvement_decision_rule"
    rule = owner.get("improvement_decision_rule")
    if "improvement_decision_rule" not in owner or chk.obj(
            rule, path, DECISION_RULE_FIELDS) is None:
        return
    for name in ("rule_id", "registration_ref"):
        if name in rule:
            chk.text(rule[name], f"{path}.{name}")
    guards = rule.get("guards")
    names = []
    if "guards" in rule:
        if type(guards) is not list:
            chk.add("not_guard_list", f"{path}.guards",
                    "guards is an explicit list of guard endpoints ([] registers none)")
            guards = None
        else:
            names = [_check_guard(chk, guard, f"{path}.guards[{index}]")
                     for index, guard in enumerate(guards)]
            listed = [name for name in names if name is not None]
            if len(set(listed)) != len(listed):
                chk.add("duplicate_guard_name", f"{path}.guards", "guard names must be unique")
    guard_strata = rule.get("guard_strata")
    listed_strata = None
    if "guard_strata" in rule:
        if type(guard_strata) is list:
            listed_strata = _strata(chk, guard_strata, f"{path}.guard_strata")
        elif guard_strata != PRIMARY_STRATA:
            chk.add("unsupported_value", f"{path}.guard_strata",
                    f"guard_strata is {PRIMARY_STRATA!r} or a list of strata")
    gating = rule.get("gating_strata")
    if "gating_strata" in rule:
        if type(gating) is list:
            gating_list = _strata(chk, gating, f"{path}.gating_strata")
            if gating_list is not None and listed_strata is not None and not set(
                    gating_list) <= set(listed_strata):
                chk.add("gating_strata_not_subset", f"{path}.gating_strata",
                        "gating strata must be a subset of the registered guard strata")
        elif gating != ALL_GUARD_STRATA:
            chk.add("unsupported_value", f"{path}.gating_strata",
                    f"gating_strata is {ALL_GUARD_STRATA!r} or a list of strata")
    if guards == [] and ("guard_strata" in rule and guard_strata != PRIMARY_STRATA or (
            "gating_strata" in rule and gating != ALL_GUARD_STRATA)):
        chk.add("guard_strata_without_guards", path,
                f"with no guards, guard_strata is {PRIMARY_STRATA!r} and gating_strata "
                f"{ALL_GUARD_STRATA!r} (agent technical default (D-018))")
    look = rule.get("look")
    if "look" in rule:
        kind = look.get("kind") if type(look) is dict else None
        if kind == "fixed":
            chk.obj(look, f"{path}.look", ("kind",))
        elif kind == "sequential":
            if chk.obj(look, f"{path}.look", ("kind", "look_index")) and "look_index" in look:
                chk.integer(look["look_index"], f"{path}.look.look_index", 1)
        else:
            chk.add("unsupported_value", f"{path}.look",
                    "look is {'kind': 'fixed'} or {'kind': 'sequential', 'look_index': k}")


def _check_issue_10(chk, owner):
    path = "$.owner.response_to_alpagym_issue_10"
    value = owner.get("response_to_alpagym_issue_10")
    if "response_to_alpagym_issue_10" not in owner or chk.obj(
            value, path, ("response", "record_ref")) is None:
        return
    if "response" in value:
        chk.choice(value["response"], f"{path}.response", ISSUE_10_RESPONSES)
    if "record_ref" in value:
        chk.text(value["record_ref"], f"{path}.record_ref")


def _check_run_aggregation(chk, owner):
    """``gate.superiority.aggregate_runs`` inputs (owner values; no defaults, fail closed).

    ``rule`` is one of ``RUN_AGGREGATION_RULES``; ``k`` is ``None`` under ``all`` and an int
    in ``[1, independent_training_runs]`` under ``at_least_k``; ``family_alpha`` is a float
    strictly between 0 and 1 (the same domain ``teacher_rl.evaluate`` and ``aggregate_runs``
    accept).
    """
    path = "$.owner.run_aggregation"
    agg = owner.get("run_aggregation")
    if "run_aggregation" not in owner or chk.obj(
            agg, path, ("rule", "k", "family_alpha")) is None:
        return
    rule, k = agg.get("rule"), agg.get("k")
    if "rule" in agg and rule not in RUN_AGGREGATION_RULES:
        chk.add("aggregation_rule_unregistered", f"{path}.rule",
                f"rule is one of {list(RUN_AGGREGATION_RULES)} (owner input, no default)")
    runs = owner.get("independent_training_runs")
    m = runs if type(runs) is int and runs >= 1 else None
    if "k" in agg and rule == "all" and k is not None:
        chk.add("aggregation_k_invalid", f"{path}.k", "k applies only to at_least_k")
    if "k" in agg and rule == "at_least_k" and (
            type(k) is not int or k < 1 or (m is not None and k > m)):
        chk.add("aggregation_k_invalid", f"{path}.k",
                "at_least_k needs an integer 1 <= k <= independent_training_runs")
    alpha = agg.get("family_alpha")
    if "family_alpha" in agg and (type(alpha) is not float or not 0 < alpha < 1):
        chk.add("family_alpha_invalid", f"{path}.family_alpha",
                "family_alpha is a float strictly between 0 and 1 (owner input)")


def _check_anchor(chk, owner):
    if "anchor" not in owner:
        return
    anchor, path = owner["anchor"], "$.owner.anchor"
    if type(anchor) is not dict or anchor.get("kind") not in ("none", "kl_to_reference"):
        chk.add("unsupported_value", path,
                "anchor must be {'kind': 'none'} or {'kind': 'kl_to_reference', ...}")
        return
    if anchor["kind"] == "none":
        chk.obj(anchor, path, ("kind",))
        return
    if chk.obj(anchor, path, ("kind", "kl_beta", "reference_reset_interval")) is None:
        return
    if "kl_beta" in anchor:
        chk.number(anchor["kl_beta"], f"{path}.kl_beta", 0, low_open=True)
    if "reference_reset_interval" in anchor:
        chk.integer(anchor["reference_reset_interval"], f"{path}.reference_reset_interval", 0)


def _check_roles(chk, owner, scene_set):
    roles_ok = None
    if "scene_roles" in owner:
        roles = owner["scene_roles"]
        if chk.obj(roles, "$.owner.scene_roles", ("training", "development", "confirmation")):
            parsed = {}
            for role in ("training", "development", "confirmation"):
                if role in roles:
                    parsed[role] = chk.scenes(roles[role], f"$.owner.scene_roles.{role}",
                                              allow_empty=role != "training")
            if all(parsed.get(r) is not None for r in ("training", "development",
                                                        "confirmation")):
                roles_ok = parsed
    if (roles_ok is not None and scene_set is not None
            and set(roles_ok["training"]) != set(scene_set)):
        chk.add("training_role_differs_from_scene_set", "$.owner.scene_roles.training",
                    "the training role must list exactly the registered scene_set")
    rule = owner.get("overlap_rule")
    if "overlap_rule" in owner and chk.obj(rule, "$.owner.overlap_rule",
                                           ("rule_ref", "summary")):
        for name in ("rule_ref", "summary"):
            if name in rule:
                chk.text(rule[name], f"$.owner.overlap_rule.{name}")
    return roles_ok


def _check_seeds(chk, owner):
    seeds = owner.get("seed_modes")
    fields = ("train_seed", "dataloader_seed", "rollout_force_determinism", "simulator")
    if "seed_modes" not in owner or chk.obj(seeds, "$.owner.seed_modes", fields) is None:
        return
    path = "$.owner.seed_modes"
    train_seed = seeds.get("train_seed")
    if "train_seed" in seeds and train_seed != "unseeded" and not (
            type(train_seed) is int and train_seed >= 0):
        chk.add("unsupported_value", f"{path}.train_seed",
                "train_seed is a non-negative integer or the string 'unseeded'")
    if "dataloader_seed" in seeds:
        chk.integer(seeds["dataloader_seed"], f"{path}.dataloader_seed", 0)
    if "rollout_force_determinism" in seeds:
        chk.boolean(seeds["rollout_force_determinism"], f"{path}.rollout_force_determinism")
    if "simulator" in seeds and seeds["simulator"] != SIMULATOR_SEED_MODE:
        chk.add("no_upstream_key", f"{path}.simulator",
                "AlpaSim session seeds are random and unsettable at the training pin "
                f"(lane-r3 finding 11); only {SIMULATOR_SEED_MODE!r} compiles")


def _check_versions(chk, owner, eng):
    pol = owner.get("version_policy")
    fields = ("on_policy", "allowed_outdated_steps", "prefetch_rollout")
    if "version_policy" not in owner or chk.obj(pol, "$.owner.version_policy", fields) is None:
        return
    path = "$.owner.version_policy"
    ok = all([
        "on_policy" not in pol or chk.boolean(pol["on_policy"], f"{path}.on_policy"),
        "allowed_outdated_steps" not in pol
        or chk.integer(pol["allowed_outdated_steps"], f"{path}.allowed_outdated_steps", 0),
        "prefetch_rollout" not in pol
        or chk.boolean(pol["prefetch_rollout"], f"{path}.prefetch_rollout"),
    ])
    if ok and pol.get("on_policy") is True and pol.get("allowed_outdated_steps", 0) != 0:
        chk.add("on_policy_lag_contradiction", path,
                "Cosmos forces allowed_outdated_steps = 0 when on_policy is true "
                f"({_C_CONFIG}:TrainingConfig validator)")
    preset = _preset_for(eng)
    if ok and pol.get("prefetch_rollout") is True and preset and preset["dp_shard_size"] != 1:
        chk.add("prefetch_requires_dp_shard_one", f"{path}.prefetch_rollout",
                f"prefetch_rollout requires dp_shard_size == 1 ({_G_CONFIG}:"
                "CosmosRLRolloutConfig.prefetch_rollout)")


def _check_batch_geometry(chk, owner, eng):
    """The pinned host validator's GRPO batch-geometry rule, applied before launch.

    ``train_batch_per_replica`` (= ``rollouts_per_trainer_step``) must be divisible by
    ``mini_batch`` and by ``dp_shard_size * mini_batch`` (``_G_GEOMETRY``); upstream raises
    ``ValueError`` before writing a Cosmos config otherwise, so such a shape cannot run.
    """
    preset = _preset_for(eng)
    steps = owner.get("optimizer_steps_per_run") if type(owner) is dict else None
    rollouts = owner.get("rollouts_per_trainer_step") if type(owner) is dict else None
    mini = steps.get("mini_batch_rows") if type(steps) is dict else None
    if preset is None or not all(type(v) is int and v > 0 for v in (rollouts, mini)):
        return  # missing/invalid inputs are reported by their own checks
    shard = preset["dp_shard_size"]
    problems = []
    if rollouts % mini:
        problems.append(f"train_batch_per_replica({rollouts}) must be divisible by "
                        f"mini_batch({mini})")
    if rollouts % (shard * mini):
        problems.append(f"train_batch_per_replica({rollouts}) must be divisible by "
                        f"dp_shard_size({shard}) * mini_batch({mini})")
    if problems:
        chk.issues.append({"code": "batch_geometry_rejected_upstream",
                           "path": "$.owner.optimizer_steps_per_run.mini_batch_rows",
                           "severity": "blocking", "evidence_refs": [_G_GEOMETRY],
                           "message": "the pinned host validator rejects this shape before "
                                      "launch: " + "; ".join(problems)})


def _check_e07_and_parity(chk, owner):
    e07 = owner.get("e07_pass_criterion")
    path = "$.owner.e07_pass_criterion"
    if "e07_pass_criterion" in owner:
        kind = e07.get("kind") if type(e07) is dict else None
        if kind == "exact_zero":
            chk.obj(e07, path, ("kind",))
        elif kind == "float32_tolerance":
            if chk.obj(e07, path, ("kind", "tolerance")) and "tolerance" in e07:
                chk.number(e07["tolerance"], f"{path}.tolerance", 0, low_open=True)
        else:
            chk.add("unsupported_value", path,
                    "kind must be 'exact_zero' or 'float32_tolerance' (owner input, no default)")
    tol = owner.get("parity_tolerances")
    if "parity_tolerances" not in owner or chk.obj(
            tol, "$.owner.parity_tolerances", PARITY_KEYS) is None:
        return
    for key in PARITY_KEYS:
        entry = tol.get(key)
        sub = f"$.owner.parity_tolerances.{key}"
        if key not in tol or type(entry) is not dict:
            if key in tol:
                chk.add("not_object", sub, "expected an object")
            continue
        allowed = ("abs", "rel", "ulp", "justification_ref")
        for extra in sorted(set(entry) - set(allowed)):
            chk.add("unknown_field", f"{sub}.{extra}", f"unknown field {extra!r}")
        for name in ("abs", "rel", "justification_ref"):
            if name not in entry:
                chk.add("field_missing", f"{sub}.{name}", f"required field {name!r} is missing")
        if "abs" in entry:
            chk.number(entry["abs"], f"{sub}.abs", 0)
        if "rel" in entry:
            chk.number(entry["rel"], f"{sub}.rel", 0)
        if "ulp" in entry:
            chk.integer(entry["ulp"], f"{sub}.ulp", 0)
        if "justification_ref" in entry:
            chk.text(entry["justification_ref"], f"{sub}.justification_ref")


def _producer():
    source = Path(__file__).read_bytes()
    return {"module": "teacher_rl.registration",
            "source_sha256": hashlib.sha256(source).hexdigest()}


def _role_facts(owner):
    roles = owner.get("scene_roles") if type(owner) is dict else None
    if type(roles) is not dict or not all(type(roles.get(r)) is list for r in
                                          ("training", "development", "confirmation")):
        return None
    sets = {role: {s for s in roles[role] if type(s) is str} for role in roles
            if role in ("training", "development", "confirmation")}
    pairs = (("training", "development"), ("training", "confirmation"),
             ("development", "confirmation"))
    return {f"{a}&{b}": sorted(sets[a] & sets[b]) for a, b in pairs}


def validate_registration(registration):
    """Validate an S4 run registration; ``passed`` only when no blocking issue remains.

    Returns ``{"status", "issues", "facts", "registration_hash"}``. ``facts`` reports scene-ID
    intersections between roles: the overlap decision belongs to the registered rule, so
    intersections are surfaced as warnings, never silently accepted or rejected here.
    """
    chk = _Checker()
    if chk.obj(registration, "$", TOP_FIELDS) is None:
        return {"status": "blocked", "issues": chk.issues, "facts": {},
                "registration_hash": None}
    if registration.get("artifact_kind") != ARTIFACT_KIND:
        chk.add("wrong_artifact_kind", "$.artifact_kind", f"must be {ARTIFACT_KIND!r}")
    if registration.get("schema_version") != SCHEMA_VERSION or type(
            registration.get("schema_version")) is not int:
        chk.add("unsupported_schema_version", "$.schema_version",
                f"only schema_version {SCHEMA_VERSION} is supported")
    if "scope" in registration:
        chk.choice(registration["scope"], "$.scope", SCOPES)
    rid = registration.get("registration_id")
    if "registration_id" in registration and (type(rid) is not str or not _SLUG.fullmatch(rid)):
        chk.add("malformed_registration_id", "$.registration_id", "expected a lowercase slug")
    eng = registration.get("engineering")
    if "engineering" in registration:
        _check_engineering(chk, eng)
    if "owner" in registration:
        _check_owner(chk, registration["owner"], eng)
        _check_batch_geometry(chk, registration["owner"], eng)
    facts = {}
    role_facts = _role_facts(registration.get("owner"))
    if role_facts is not None:
        facts["role_scene_id_intersections"] = role_facts
        for pair, shared in role_facts.items():
            if shared:
                chk.add("role_scene_id_overlap", "$.owner.scene_roles",
                        f"{pair} share {len(shared)} scene ID(s); the registered overlap rule "
                        "decides admissibility", severity="warning")
    blocking = any(issue["severity"] == "blocking" for issue in chk.issues)
    try:
        registration_hash = identity_hash(registration)
    except ValueError:
        registration_hash = None
        chk.add("not_json_native", "$", "registration is not strict JSON")
        blocking = True
    return {"status": "blocked" if blocking else "passed", "issues": chk.issues,
            "facts": facts, "registration_hash": registration_hash}


def _hydra_value(value):
    """Render a JSON value in Hydra override grammar (strings single-quoted)."""
    if value is None:
        return "null"
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    if type(value) is float:
        return repr(value)
    if type(value) is str:
        if "'" in value or "\\" in value:
            raise ValueError("override strings may not contain quotes or backslashes")
        return f"'{value}'"
    if type(value) is list:
        return "[" + ",".join(_hydra_value(item) for item in value) + "]"
    raise ValueError(f"unsupported override value {type(value).__name__}")


def _ceil_div(a, b):
    return -(-a // b)


def compile_overrides(registration):
    """Map a valid registration to pinned Hydra overrides and (H1) Cosmos TOML overrides.

    Fails closed: a blocked registration, a value needing a Cosmos-only key on the
    ``hydra_only`` route, or a value the pinned code cannot express returns ``blocked`` with no
    overrides. The ``effective_config_hash`` binds pins, base preset, route and every override.
    """
    report = validate_registration(registration)
    issues = list(report["issues"])
    base = {
        "artifact_kind": "s4_effective_config",
        "schema_version": EFFECTIVE_CONFIG_SCHEMA_VERSION,
        "scope": registration.get("scope") if type(registration) is dict else None,
        "producer": _producer(),
        "inputs": [{"artifact_kind": ARTIFACT_KIND,
                    "identity_hash": report["registration_hash"]}],
        "units": {
            "max_optimizer_steps_per_trainer_step": "optimizer steps per trainer step per "
                                                    "policy replica (upper bound)",
            "max_optimizer_steps_per_run": "optimizer steps per run (upper bound)",
        },
    }
    if report["status"] != "passed":
        return {**base, "status": "blocked", "issues": issues, "effective_config_hash": None}
    eng, owner = registration["engineering"], registration["owner"]
    preset = pins.EXPERIMENT_PRESETS[eng["base_experiment"]]
    route = eng["config_route"]
    mapping, hydra_values, toml = [], [], {}

    def hydra(field, key, value, source):
        mapping.append({"field": field, "category": "hydra_key", "key": key, "value": value,
                        "source": source})
        hydra_values.append(f"{key}={_hydra_value(value)}")

    def cosmos(field, key, value):
        default, source = COSMOS_TOML_DEFAULTS[key]
        entry = {"field": field, "category": "cosmos_toml_key", "key": key, "value": value,
                 "source": source, "upstream_default": default}
        if value == default and type(value) is type(default):
            entry["applied"] = "upstream_default_matches"
        elif route == "h1_resolved_toml":
            entry["applied"] = "h1_toml_override"
            toml[key] = value
        else:
            entry["applied"] = "refused"
            issues.append({"code": "requires_h1_toml_route", "path": f"$.owner.{field}",
                           "severity": "blocking", "evidence_refs": [source],
                           "message": f"{key} has no AlpaGym Hydra field; declare "
                                      "config_route = 'h1_resolved_toml' or register the "
                                      "Cosmos default"})
        mapping.append(entry)

    def fixed(field, value, source):
        mapping.append({"field": field, "category": "fixed_by_upstream", "key": None,
                        "value": value, "source": source, "applied": "shipped_behaviour"})

    def harness(field):
        mapping.append({"field": field, "category": "harness_only", "key": None,
                        "value": owner[field], "source": "PhysLoop evaluator input",
                        "applied": "recorded"})

    group_sel = [f"experiment={eng['base_experiment']}",
                 f"reward={owner['reward_arm']['preset']}"]
    mapping.append({"field": "engineering.base_experiment", "category": "hydra_key",
                    "key": "experiment", "value": eng["base_experiment"],
                    "source": preset["source"]})
    mapping.append({"field": "reward_arm", "category": "hydra_key", "key": "reward",
                    "value": owner["reward_arm"]["preset"], "source": _G_REWARD})
    hydra("engineering.policy_model_path", "policy.model.path", eng["policy_model_path"],
          f"{_G_CONFIG}:ModelConfig.path")
    hydra("engineering.run_root", "run_root", eng["run_root"],
          f"{_G_CONFIG}:RunConfigSchema.run_root")
    hydra("group_size", "cosmos.rollout.n_generation", owner["group_size"],
          f"{_G_CONFIG}:CosmosRLRolloutConfig.n_generation")
    hydra("rollouts_per_trainer_step", "cosmos.train.train_batch_per_replica",
          owner["rollouts_per_trainer_step"],
          f"{_G_CONFIG}:CosmosRLTrainConfig.train_batch_per_replica")
    hydra("learning_rate", "cosmos.train.optm_lr", float(owner["learning_rate"]),
          f"{_G_CONFIG}:CosmosRLTrainConfig.optm_lr")
    sched = owner["lr_schedule"]
    hydra("lr_schedule.decay_type", "cosmos.train.optm_decay_type", sched["decay_type"],
          f"{_G_CONFIG}:CosmosRLTrainConfig.optm_decay_type")
    hydra("lr_schedule.decay_ratio", "cosmos.train.optm_decay_ratio",
          float(sched["decay_ratio"]), f"{_G_CONFIG}:CosmosRLTrainConfig.optm_decay_ratio")
    hydra("lr_schedule.min_lr_factor", "cosmos.train.optm_min_lr_factor",
          float(sched["min_lr_factor"]), f"{_G_CONFIG}:CosmosRLTrainConfig.optm_min_lr_factor")
    hydra("warmup.steps", "cosmos.train.optm_warmup_steps", owner["warmup"]["steps"],
          f"{_G_CONFIG}:CosmosRLTrainConfig.optm_warmup_steps")
    cosmos("warmup.start_factor", "train.optm_warmup_start_factor",
           float(owner["warmup"]["start_factor"]))
    hydra("scene_set", "dataset.scene_ids", list(owner["scene_set"]),
          f"{_G_CONFIG}:DatasetConfig.scene_ids")
    hydra("scene_set", "dataset.test_suite_id", None, f"{_G_CONFIG}:DatasetConfig.test_suite_id")
    steps = owner["optimizer_steps_per_run"]
    hydra("optimizer_steps_per_run.trainer_steps", "cosmos.train.max_num_steps",
          steps["trainer_steps"], f"{_G_CONFIG}:CosmosRLTrainConfig.max_num_steps")
    hydra("optimizer_steps_per_run.epochs", "cosmos.train.num_epochs", steps["epochs"],
          f"{_G_CONFIG}:CosmosRLTrainConfig.num_epochs")
    hydra("optimizer_steps_per_run.mini_batch_rows", "cosmos.train.train_policy.mini_batch",
          steps["mini_batch_rows"], f"{_G_CONFIG}:CosmosRLTrainPolicyConfig.mini_batch")
    hydra("optimizer_steps_per_run.optimizer_iterations",
          "cosmos.train.train_policy.grpo_optimization_iterations", steps["optimizer_iterations"],
          f"{_G_CONFIG}:CosmosRLTrainPolicyConfig.grpo_optimization_iterations")
    ret = owner["checkpoint_retention"]
    hydra("checkpoint_retention", "cosmos.train.ckpt.enable_checkpoint", True,
          f"{_G_CONFIG}:CosmosRLTrainCkptConfig.enable_checkpoint")
    hydra("checkpoint_retention.save_freq", "cosmos.train.ckpt.save_freq", ret["save_freq"],
          f"{_G_CONFIG}:CosmosRLTrainCkptConfig.save_freq")
    hydra("checkpoint_retention.max_keep", "cosmos.train.ckpt.max_keep", ret["max_keep"],
          f"{_G_CONFIG}:CosmosRLTrainCkptConfig.max_keep")
    hydra("checkpoint_retention.export_safetensors", "cosmos.train.ckpt.export_safetensors",
          ret["export_safetensors"], f"{_G_CONFIG}:CosmosRLTrainCkptConfig.export_safetensors")
    anchor = owner["anchor"]
    kl_beta = 0.0 if anchor["kind"] == "none" else float(anchor["kl_beta"])
    reset = 0 if anchor["kind"] == "none" else anchor["reference_reset_interval"]
    hydra("anchor", "cosmos.train.train_policy.kl_beta", kl_beta,
          f"{_G_CONFIG}:CosmosRLTrainPolicyConfig.kl_beta")
    hydra("anchor", "cosmos.train.train_policy.reference_reset_interval", reset,
          f"{_G_CONFIG}:CosmosRLTrainPolicyConfig.reference_reset_interval")
    seeds = owner["seed_modes"]
    cosmos("seed_modes.train_seed", "train.seed",
           None if seeds["train_seed"] == "unseeded" else seeds["train_seed"])
    cosmos("seed_modes.dataloader_seed", "train.train_policy.dataloader_seed",
           seeds["dataloader_seed"])
    hydra("seed_modes.rollout_force_determinism", "policy.inference.sampling.force_determinism",
          seeds["rollout_force_determinism"],
          f"{_G_CONFIG}:SamplingParamsConfig.force_determinism")
    fixed("seed_modes.simulator", seeds["simulator"],
          "alpasim-gympin driver_service.py:75 / runtime.proto:14-20 (lane-r3 finding 11)")
    pol = owner["version_policy"]
    hydra("version_policy.on_policy", "cosmos.train.train_policy.on_policy", pol["on_policy"],
          f"{_G_CONFIG}:CosmosRLTrainPolicyConfig.on_policy")
    hydra("version_policy.allowed_outdated_steps",
          "cosmos.train.train_policy.allowed_outdated_steps", pol["allowed_outdated_steps"],
          f"{_G_CONFIG}:CosmosRLTrainPolicyConfig.allowed_outdated_steps")
    hydra("version_policy.prefetch_rollout", "cosmos.rollout.prefetch_rollout",
          pol["prefetch_rollout"], f"{_G_CONFIG}:CosmosRLRolloutConfig.prefetch_rollout")
    video_flag = "true" if owner["video"]["render_video"] else "false"
    extra = f"{preset['extra_overrides']} eval.video.render_video={video_flag}"
    mapping.append({"field": "video.render_video", "category": "hydra_key",
                    "key": "alpasim.wizard_args.extra_overrides", "value": extra,
                    "source": f"{_G_CONFIG}:AlpaSimWizardArgs.extra_overrides; "
                              f"{_A_BASE}:eval.video.render_video",
                    "composition": {"preset_part": preset["extra_overrides"],
                                    "preset_source": preset["source"],
                                    "registered_part": f"eval.video.render_video={video_flag}"}})
    hydra_values.append(f"alpasim.wizard_args.extra_overrides={_hydra_value(extra)}")
    fixed("decode_device", owner["decode_device"],
          f"{_G_POLICY}:model.device (cuda); alpagym buffers.py ObservationBuffers.__init__")
    for field in ("scene_roles", "overlap_rule", "evaluated_policy_path",
                  "improvement_decision_rule", "smallest_improvement_of_interest",
                  "e07_pass_criterion", "parity_tolerances", "independent_training_runs",
                  "run_aggregation", "prediction_ref", "response_to_alpagym_issue_10"):
        harness(field)
    issue_10 = owner["response_to_alpagym_issue_10"]
    if issue_10["response"] == "change_request":
        issues.append({"code": "alpagym_issue_10_change_not_compilable",
                       "path": "$.owner.response_to_alpagym_issue_10", "severity": "blocking",
                       "evidence_refs": [_I10],
                       "message": "the pinned trainer has no setting for the trainable master "
                                  "dtype; the recorded change request needs a new pin or an "
                                  "approved training-method change before compilation"})
    if owner["rollouts_per_trainer_step"] < owner["group_size"]:
        issues.append({"code": "partial_group_trained", "path": "$.owner",
                       "severity": "warning", "evidence_refs": [
                           "alpagym:packages/runtime/src/alpagym_runtime/cosmos/rollout_filter.py"],
                       "message": "rollouts_per_trainer_step < group_size: the trainer filter "
                                  "keeps the freshest rollouts per replica, so part of each "
                                  "group is unlinked before training"})
    blocking = any(issue["severity"] == "blocking" for issue in issues)
    if blocking:
        return {**base, "status": "blocked", "issues": issues, "mapping": mapping,
                "effective_config_hash": None}
    rows = owner["rollouts_per_trainer_step"] * preset["expected_valid_steps"]
    # The geometry check guarantees mini_batch_rows divides the rollouts, hence the rows; the
    # ceiling is kept as the upstream minibatch count formula (trainer.py:322-323).
    per_step = steps["optimizer_iterations"] * _ceil_div(rows, steps["mini_batch_rows"])
    hydra_overrides = group_sel + sorted(hydra_values)
    effective = {
        "pins": dict(eng["pins"]),
        "cosmos_reason2_revision": eng["cosmos_reason2_revision"],
        "base_experiment": eng["base_experiment"],
        "config_route": route,
        "hydra_overrides": hydra_overrides,
        "cosmos_toml_overrides": dict(sorted(toml.items())),
    }
    return {
        **base,
        "status": "passed",
        "issues": issues,
        "hydra": {"overrides": hydra_overrides},
        "cosmos_toml_overrides": dict(sorted(toml.items())),
        "mapping": mapping,
        "inherited_from_preset": {
            "source": preset["source"],
            "expected_valid_steps": preset["expected_valid_steps"],
            "ratio_clip_low_high": list(preset["ratio_clip"]),
            "topology": preset["topology"],
            "n_sim_steps": preset["n_sim_steps"],
            "control_timestep_us": preset["control_timestep_us"],
            "force_gt_duration_us": preset["force_gt_duration_us"],
        },
        "inherited_cosmos_defaults": {
            "source": "lane-r3 §4 (Cosmos defaults not exposed by AlpaGym)",
            "optimizer": "AdamW fused, weight_decay 0.01, betas (0.9, 0.999)",
            "grad_norm_clip": 1.0, "variant": "grpo", "master_dtype": "float32",
            "param_dtype": "bfloat16",
            "alpagym_issue_10": {
                "configured_master_dtype": "float32",
                "runtime_trainable_dtype": "unverified (E03 records it at load)",
                "risk": "trainable weights and Adam state may be bf16 at runtime",
                "source": _I10,
                "response": issue_10["response"],
                "record_ref": issue_10["record_ref"],
            },
        },
        "derived": {
            "replay_rows_per_trainer_step": rows,
            "max_optimizer_steps_per_trainer_step": per_step,
            "max_optimizer_steps_per_run": per_step * steps["trainer_steps"],
            "rule": "optimizer_iterations x ceil(rows / mini_batch_rows); zero-gradient "
                    "minibatches skip the step (02 S4.1)",
        },
        "qualify_on_host": [
            "Hydra composes these overrides onto the pinned preset (resolved_config.yaml hash)",
            "h1 route: the controller's effective Cosmos config contains every TOML override",
            "AlpaSim wizard receives eval.video.render_video via extra_overrides",
        ],
        "effective_config": effective,
        "effective_config_hash": identity_hash(effective),
    }
