"""Known answers for optimizer-step accounting, skip semantics and the LR lambda.

E20/E07 (build-spec A9). Oracles: hand-derived literals; the pinned
``build_lr_schedulers`` executed from the hash-checked vendored
``cosmos_rl/policy/trainer/optm/__init__.py`` (only that function node; its
``LRSchedulersContainer`` replaced by a capture stub); AST checks of the pinned trainer
text for the minibatch formula and the skip condition.
"""

import ast
import functools
import importlib.util
import itertools
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from teacher_rl.checks import steps

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "s4_upstream_loader", ROOT / "tests/fixtures/s4_upstream_checks/upstream_loader.py"
)
upstream = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(upstream)

H = "a" * 64
H2 = "b" * 64


# --- E20 formula ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rows", "mini_batch", "iterations", "zero", "expected"),
    [
        (22, 1, 1, 0, 22),  # smoke preset: 22 valid rows, mini_batch 1, mu 1
        (22, 4, 1, 0, 6),  # ceil(22/4) = 6
        (22, 4, 2, 3, 9),  # 2 * 6 - 3
        (3, 8, 1, 0, 1),  # mini_batch larger than rows -> one minibatch (min(mb, rows))
        (132, 1, 1, 132, 0),  # every minibatch zero-norm -> no optimizer step
    ],
)
def test_e20_expected_optimizer_steps_literals(rows, mini_batch, iterations, zero, expected):
    assert steps.expected_optimizer_steps(rows, mini_batch, iterations, zero) == expected


@pytest.mark.parametrize(
    "args",
    [
        (0, 1, 1, 0),
        (5, 0, 1, 0),
        (5, 1, 0, 0),
        (5, 2, 1, 4),
        (5, 2, 1, -1),
        (True, 1, 1, 0),
        (5.0, 1, 1, 0),
    ],
)
def test_e20_rejects_invalid_inputs(args):
    with pytest.raises(ValueError):
        steps.expected_optimizer_steps(*args)


def test_e20_formula_matches_pinned_trainer_text():
    source = upstream.source_bytes("alpagym/trainer.py.txt").decode()
    loop = ast.unparse(upstream._find(ast.parse(source), "AlpagymGRPOTrainer._run_training_loop"))
    assert "mini_batch_size = min(self._mini_batch, num_steps)" in loop
    assert "num_mini_batches = (num_steps + mini_batch_size - 1) // mini_batch_size" in loop
    assert "for _ in range(self._grpo_optimization_iterations):" in loop
    skip = ast.unparse(upstream._find(ast.parse(source), "AlpagymGRPOTrainer.all_reduce_states"))
    assert "if grad_norm_value != 0.0:\n            self.optimizers.step()" in skip
    step = ast.unparse(upstream._find(ast.parse(source), "AlpagymGRPOTrainer.step_training"))
    assert step.count("self.lr_schedulers.step()") == 1  # once per trainer step, always
    # Exhaustive agreement of the closed form with the pinned loop arithmetic.
    for rows, mb, it in itertools.product(range(1, 30), range(1, 12), range(1, 4)):
        size = min(mb, rows)
        assert steps.minibatch_count(rows, mb, it) == it * ((rows + size - 1) // size)
        assert steps.minibatch_count(rows, mb, it) == it * math.ceil(rows / mb)


def _step_record(**overrides):
    record = {
        "trainer_step": 1,
        "rows": 3,
        "mini_batch": 2,
        "iterations": 2,
        "source": "in_process_capture",
        "minibatches": [
            {"grad_norm": 0.5, "optimizer_step_called": True},
            {"grad_norm": 0.0, "optimizer_step_called": False},
            {"grad_norm": 1e-9, "optimizer_step_called": True},
            {"grad_norm": 0.0, "optimizer_step_called": False},
        ],
        "reported_num_batches": 4,
    }
    record.update(overrides)
    return record


def test_e20_reconcile_in_process_capture_known_answer():
    report = steps.reconcile_optimizer_steps([_step_record()], scope="synthetic")
    assert report["status"] == "passed"
    row = report["trainer_steps"][0]
    assert row["expected_minibatches"] == 4
    assert row["zero_norm_minibatches"] == 2
    assert row["expected_optimizer_steps"] == row["counted_optimizer_steps"] == 2
    assert report["units_of_measure"]["optimizer_steps"].startswith("optimizer.step() calls")


def test_e20_negative_controls():
    called_at_zero = _step_record(
        minibatches=[
            {"grad_norm": 0.0, "optimizer_step_called": True},
            {"grad_norm": 0.5, "optimizer_step_called": True},
            {"grad_norm": 0.5, "optimizer_step_called": True},
            {"grad_norm": 0.5, "optimizer_step_called": True},
        ]
    )
    missing = _step_record(minibatches=_step_record()["minibatches"][:3])
    wrong_batches = _step_record(reported_num_batches=2)
    nonfinite = _step_record(
        minibatches=[{"grad_norm": "nan", "optimizer_step_called": False}]
        + _step_record()["minibatches"][1:]
    )
    no_rows = _step_record(rows=0, minibatches=[], reported_num_batches=0)
    expected = {
        "skip_rule_violated": called_at_zero,
        "minibatch_count_mismatch": missing,
        "num_batches_mismatch": wrong_batches,
        "nonfinite_gradient_upstream_raises": nonfinite,
        "no_trainable_samples": no_rows,
    }
    for code, record in expected.items():
        report = steps.reconcile_optimizer_steps([record], scope="synthetic")
        assert report["status"] == "blocked", code
        assert code in {i["code"] for i in report["issues"]}, code


def test_e20_log_inferred_counts_are_unverifiable_when_a_zero_is_printed():
    logged = _step_record(
        source="log_inferred",
        minibatches=[
            {"grad_norm": "0.500000", "optimizer_step_called": None},
            {"grad_norm": "0.000000", "optimizer_step_called": None},
            {"grad_norm": "0.000001", "optimizer_step_called": None},
            {"grad_norm": "0.000000", "optimizer_step_called": None},
        ],
    )
    report = steps.reconcile_optimizer_steps([logged], scope="synthetic")
    assert report["status"] == "unverified"
    row = report["trainer_steps"][0]
    assert row["counted_optimizer_steps"] is None
    assert row["optimizer_steps_bounds"] == [2, 4]
    assert "unverifiable_log_precision" in {i["code"] for i in report["issues"]}
    nonzero = _step_record(
        source="log_inferred",
        minibatches=[{"grad_norm": "0.500000", "optimizer_step_called": None}] * 4,
    )
    exact = steps.reconcile_optimizer_steps([nonzero], scope="synthetic")
    assert exact["status"] == "passed"
    assert exact["trainer_steps"][0]["counted_optimizer_steps"] == 4
    bad = _step_record(
        source="log_inferred",
        minibatches=[{"grad_norm": "5e-7", "optimizer_step_called": None}] * 4,
    )
    with pytest.raises(ValueError):
        steps.reconcile_optimizer_steps([bad], scope="synthetic")


# --- E07 skip semantics ---------------------------------------------------------------


def _snap(param=H, moment=H, step=3):
    return {
        "parameters": {"expert.w": param},
        "optimizer_state": {"expert.w": {"exp_avg": moment, "exp_avg_sq": moment}},
        "step_counters": {"expert.w": step},
    }


def _skip_case(**changes):
    case = {
        "trainer_step": 4,
        "scheduler_before": {"last_epoch": 3, "lrs": [0.0]},
        "scheduler_after": {"last_epoch": 4, "lrs": [5e-5]},
        "minibatches": [
            {
                "index": 0,
                "grad_norm": 0.0,
                "optimizer_step_called": False,
                "before": _snap(),
                "after": _snap(),
            },
            {
                "index": 1,
                "grad_norm": 0.25,
                "optimizer_step_called": True,
                "before": _snap(),
                "after": _snap(
                    param=H, moment=H2, step=4
                ),  # lr 0 at warmup start: weights may stay
            },
        ],
    }
    case.update(changes)
    return case


def test_e07_skip_leaves_state_unchanged_while_scheduler_advances():
    report = steps.check_skip_semantics(_skip_case(), scope="synthetic")
    assert report["status"] == "passed", report["issues"]
    rows = report["minibatches"]
    assert rows[0]["expected"] == "skip" and rows[0]["parameters_changed"] is False
    assert rows[1]["expected"] == "step" and rows[1]["parameters_changed"] is False
    assert report["scheduler_advanced"] is True


def test_e07_skip_negative_controls():
    moved = _skip_case()
    moved["minibatches"][0]["after"] = _snap(param=H2)
    moments = _skip_case()
    moments["minibatches"][0]["after"] = _snap(moment=H2)
    counter = _skip_case()
    counter["minibatches"][0]["after"] = _snap(step=4)
    stepped = _skip_case()
    stepped["minibatches"][0]["optimizer_step_called"] = True
    frozen_scheduler = _skip_case(scheduler_after={"last_epoch": 3, "lrs": [0.0]})
    no_counter = _skip_case()
    no_counter["minibatches"][1]["after"] = _snap(moment=H2, step=3)
    gap = _skip_case()
    gap["minibatches"][1]["before"] = _snap(param=H2)
    keys = _skip_case()
    keys["minibatches"][0]["after"] = {**_snap(), "parameters": {"expert.w": H, "expert.v": H}}
    expected = {
        "parameters_changed_at_skip": moved,
        "optimizer_state_changed_at_skip": moments,
        "step_counter_changed_at_skip": counter,
        "skip_rule_violated": stepped,
        "scheduler_not_advanced": frozen_scheduler,
        "step_counter_not_advanced": no_counter,
        "snapshot_discontinuity": gap,
        "snapshot_keys_changed": keys,
    }
    for code, case in expected.items():
        report = steps.check_skip_semantics(case, scope="synthetic")
        assert report["status"] == "blocked", code
        assert code in {i["code"] for i in report["issues"]}, code


# --- pinned warmup_stable_decay lambda --------------------------------------------


@pytest.fixture(scope="module")
def pinned_lambda():
    captured = {}

    def container(optimizers, lr_lambda):
        captured["lr_lambda"] = lr_lambda
        return lr_lambda

    warnings = []
    ns = upstream.load(
        "cosmos_rl/optm_init.py.txt",
        ["build_lr_schedulers"],
        {
            "math": math,
            "functools": functools,
            "logger": SimpleNamespace(warning=warnings.append),
            "LRSchedulersContainer": container,
        },
    )

    def build(training_steps, warmup, decay_ratio, decay_type, min_lr, start):
        train = SimpleNamespace(
            optm_warmup_steps=warmup,
            optm_decay_ratio=decay_ratio,
            optm_decay_type=decay_type,
            optm_min_lr_factor=min_lr,
            optm_warmup_start_factor=start,
        )
        return ns["build_lr_schedulers"](None, SimpleNamespace(train=train), training_steps)

    return build


def _schedule(training_steps, warmup, decay_ratio=None, decay_type="none", min_lr=0.0, start=0.0):
    return steps.resolve_lr_schedule(
        training_steps=training_steps,
        warmup_steps=warmup,
        decay_ratio=decay_ratio,
        decay_type=decay_type,
        min_lr_factor=min_lr,
        warmup_start_factor=start,
    )


def test_warmup_literals():
    assert steps.lr_factors(_schedule(50, 1), 3) == [1.0, 1.0, 1.0]  # both 1.5 presets
    assert steps.lr_factors(_schedule(10, 3), 4) == [0.0, 0.5, 1.0, 1.0]  # starts at lr 0
    clipped = _schedule(5, 20)  # base default warmup 20 clipped to the run length
    assert clipped["warmup_steps"] == 5 and clipped["stable_steps"] == 1
    assert steps.lr_factors(clipped, 6) == [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
    assert steps.lr_factors(_schedule(10, 3, start=0.5), 3) == [0.5, 0.75, 1.0]
    decay = _schedule(10, 0, decay_ratio=1.0, decay_type="linear")
    assert (decay["warmup_steps"], decay["stable_steps"], decay["decay_steps"]) == (0, 1, 10)
    assert steps.lr_factors(decay, 3) == [1.0, 0.9, 0.8]
    assert _schedule(10, 0.5)["warmup_steps"] == 5  # float <= 1.0 is a fraction
    assert _schedule(10, 1.0)["warmup_steps"] == 10  # float 1.0 means the whole run
    assert _schedule(10, 1)["warmup_steps"] == 1  # int 1 means one step
    assert _schedule(9, 0, decay_ratio=0.5)["decay_steps"] == 4  # Python round half-even


def test_lr_lambda_matches_pinned_source_on_a_grid(pinned_lambda):
    grid = itertools.product(
        [1, 2, 5, 13],
        [0, 1, 2, 4, 20, 0.25, 1.0],
        [None, 0.0, 0.5, 1.0],
        ["linear", "sqrt", "cosine", "none", None],
        [0.0, 0.1],
        [0.0, 0.3],
    )
    for training, warmup, ratio, kind, min_lr, start in grid:
        schedule = _schedule(training, warmup, ratio, kind, min_lr, start)
        upstream_lambda = pinned_lambda(training, warmup, ratio, kind, min_lr, start)
        for step in range(training + 1):
            try:
                expected = upstream_lambda(step)
            except AssertionError:
                with pytest.raises(ValueError):
                    steps.lr_factor(step, schedule)
                continue
            assert steps.lr_factor(step, schedule) == expected, (
                training,
                warmup,
                ratio,
                kind,
                step,
            )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"warmup_steps": -1},
        {"warmup_steps": True},
        {"training_steps": 0},
        {"decay_type": "exp"},
        {"decay_ratio": float("nan")},
        {"min_lr_factor": None},
        {"warmup_start_factor": None},
    ],
)
def test_lr_schedule_rejects_invalid_or_missing_owner_settings(kwargs):
    base = {
        "training_steps": 10,
        "warmup_steps": 1,
        "decay_ratio": None,
        "decay_type": "none",
        "min_lr_factor": 0.0,
        "warmup_start_factor": 0.0,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        steps.resolve_lr_schedule(**base)
    with pytest.raises(TypeError):
        steps.resolve_lr_schedule(training_steps=10, warmup_steps=1)  # no defaults


# --- independent-review regressions (A9-01, A9-11) -------------------------------------


def test_e07_skip_with_empty_snapshots_is_blocked_not_vacuously_passed():
    empty = {"parameters": {}, "optimizer_state": {}, "step_counters": {}}
    case = _skip_case()
    case["minibatches"] = [
        {
            "index": 0,
            "grad_norm": 0.0,
            "optimizer_step_called": False,
            "before": empty,
            "after": {k: {} for k in empty},
        }
    ]
    report = steps.check_skip_semantics(case, scope="synthetic")
    assert report["status"] == "blocked"
    assert "snapshot_empty" in {i["code"] for i in report["issues"]}


def test_e07_optimizer_keys_must_match_counters_and_name_parameters():
    stray = _skip_case()
    for mb in stray["minibatches"]:
        for side in ("before", "after"):
            mb[side]["optimizer_state"] = {"expert.other": mb[side]["optimizer_state"]["expert.w"]}
    no_state = _skip_case()
    for mb in no_state["minibatches"]:
        for side in ("before", "after"):
            mb[side]["optimizer_state"] = {}
            mb[side]["step_counters"] = {}
    expected = {"snapshot_keys_mismatch": stray, "snapshot_empty": no_state}
    for code, case in expected.items():
        report = steps.check_skip_semantics(case, scope="synthetic")
        assert report["status"] == "blocked", code
        assert code in {i["code"] for i in report["issues"]}, code


def test_e07_frozen_parameters_without_optimizer_state_are_allowed():
    case = _skip_case()
    for mb in case["minibatches"]:
        for side in ("before", "after"):
            mb[side]["parameters"]["vlm.lm_head.weight"] = H
    report = steps.check_skip_semantics(case, scope="synthetic")
    assert report["status"] == "passed", report["issues"]


def test_lr_factor_validates_a_hand_built_schedule():
    schedule = steps.resolve_lr_schedule(
        training_steps=10,
        warmup_steps=0,
        decay_ratio=1.0,
        decay_type="linear",
        min_lr_factor=0.0,
        warmup_start_factor=0.0,
    )
    assert steps.lr_factor(5, schedule) == 0.5
    for bad in (
        {**schedule, "decay_type": "exponential"},
        {k: v for k, v in schedule.items() if k != "min_lr_factor"},
        {**schedule, "stable_steps": schedule["stable_steps"] + 1},
        {**schedule, "warmup_steps": -1},
        {**schedule, "min_lr_factor": float("nan")},
    ):
        with pytest.raises(ValueError):
            steps.lr_factor(5, bad)
