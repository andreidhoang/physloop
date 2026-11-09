"""Pinned replay objective, SDE log-density reduction, KL and skip path vs independent oracles.

Build-spec A9 torch part (E07, E20, E21; protocol-design section 5 oracles). Executes
hash-checked vendored upstream nodes only (``tests/fixtures/s4_upstream_checks``):

- ``alpagym@972d160`` ``cosmos/replay_objective.py``: ``compute_ppo_surrogate``,
  ``compute_kl_penalty`` (torch-only).
- ``alpamayo-recipes@1d99fc5`` ``diffusion/flow_matching.py``:
  ``FlowMatching._batched_sde_logprob`` (trainer log-density and KL; ``self`` unused).
- ``alpagym@972d160`` ``cosmos/trainer.py``: ``_run_training_loop`` and
  ``all_reduce_states`` with stub ``self`` (CUDA stream calls replaced by no-ops).

Oracles are hand-derived (closed forms below) or ``torch.distributions``; no oracle
calls the code under test. These record shipped behaviour; they propose no change.
"""

import contextlib
import hashlib
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from teacher_rl.checks import audit, steps

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "s4_upstream_loader", ROOT / "tests/fixtures/s4_upstream_checks/upstream_loader.py"
)
upstream = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(upstream)

T64 = torch.float64


@pytest.fixture(scope="module")
def objective():
    ns = upstream.load(
        "alpagym/replay_objective.py.txt",
        ["compute_ppo_surrogate", "compute_kl_penalty"],
        {"torch": torch},
    )
    return ns["compute_ppo_surrogate"], ns["compute_kl_penalty"]


@pytest.fixture(scope="module")
def sde_logprob():
    ns = upstream.load(
        "recipes/flow_matching.py.txt",
        ["FlowMatching._batched_sde_logprob"],
        {"torch": torch, "math": math},
    )
    return lambda **kw: ns["_batched_sde_logprob"](None, **kw)


def _surrogate(objective, log_ratio, advantage, padding=None):
    surrogate, _ = objective
    n = len(log_ratio)
    new = torch.tensor(log_ratio, dtype=T64, requires_grad=True)
    old = torch.zeros(n, dtype=T64)
    pad = torch.tensor(padding or [False] * n)
    loss, ratio = surrogate(new, old, torch.tensor(advantage, dtype=T64), 0.2, 0.2, pad)
    return loss, ratio, new


def test_clipped_minimum_positive_and_negative_advantage(objective):
    loss, ratio, _ = _surrogate(objective, [math.log(1.5)], [1.0])
    assert ratio.item() == pytest.approx(1.5, abs=1e-15)
    assert loss.item() == pytest.approx(-1.2, abs=1e-15)  # -min(1.5*1, 1.2*1)
    loss, _, _ = _surrogate(objective, [math.log(0.5)], [-1.0])
    assert loss.item() == pytest.approx(0.8, abs=1e-15)  # -min(-0.5, -0.8)
    loss, _, _ = _surrogate(
        objective, [math.log(1.5), math.log(0.5), 0.0], [1.0, -1.0, 0.0], [False, False, True]
    )
    assert loss.item() == pytest.approx((-1.2 + 0.8) / 2, abs=1e-15)  # mean over valid rows


def test_gradient_flows_only_inside_the_clip_region(objective):
    loss, _, new = _surrogate(objective, [math.log(1.1)], [1.0])
    loss.backward()
    assert new.grad.item() == pytest.approx(-1.1, abs=1e-12)  # d(-r*A)/dlogp = -r
    loss, _, new = _surrogate(objective, [math.log(1.5)], [1.0])
    loss.backward()
    assert new.grad.item() == 0.0  # clipped branch is constant in logp


def test_log_ratio_is_clamped_to_plus_minus_five(objective):
    _, ratio, _ = _surrogate(objective, [10.0, -10.0], [1.0, 1.0])
    assert ratio.tolist() == pytest.approx([math.exp(5.0), math.exp(-5.0)], rel=1e-15)


def test_all_padding_minibatch_is_a_graph_connected_zero(objective):
    loss, _, new = _surrogate(objective, [0.3, -0.2], [1.0, 1.0], [True, True])
    assert loss.item() == 0.0 and loss.requires_grad
    loss.backward()
    assert new.grad.tolist() == [0.0, 0.0]


def _unit_step(v_model, x_next, v_ref=None):
    """One SDE step at t = 0 with timesteps [0, 0.5] and noise_level 1.

    sigma = 1, sigma_max = 0.5, std = sqrt(1 / 0.5) = sqrt(2), dt = 0.5 so the
    transition std is std * sqrt(dt) = 1, and its mean is 0.5 * x + 0.5 * v.
    """
    n = len(v_model)
    kwargs = {
        "model_output": torch.tensor([[v_model]], dtype=T64),
        "timesteps": torch.tensor([[0.0, 0.5]], dtype=T64),
        "sample": torch.tensor([[[0.0] * n, x_next]], dtype=T64),
        "noise_level": 1.0,
    }
    if v_ref is not None:
        kwargs["ref_model_output"] = torch.tensor([[v_ref]], dtype=T64)
    return kwargs


def test_mean_reduction_gives_geometric_mean_ratio_not_joint(sde_logprob, objective):
    a = math.sqrt(2 * math.log(4.0))  # N(0; 0, 1) / N(0; a/1, 1) = exp(a^2/2) = 4
    x_next = [0.0, 0.7]
    new_lp, _ = sde_logprob(**_unit_step([0.0, 0.3], x_next))
    old_lp, _ = sde_logprob(**_unit_step([2 * a, 0.3], x_next))

    def log_density(mean, x):
        loc, scale = torch.tensor(mean, dtype=T64), torch.tensor(1.0, dtype=T64)
        return torch.distributions.Normal(loc, scale).log_prob(torch.tensor(x, dtype=T64))

    joint = sum(
        log_density(0.5 * vn, x) - log_density(0.5 * vo, x)
        for vn, vo, x in zip([0.0, 0.3], [2 * a, 0.3], x_next)
    )
    assert math.exp(joint.item()) == pytest.approx(4.0, abs=1e-12)  # joint ratio
    surrogate, _ = objective
    _, ratio = surrogate(new_lp, old_lp, torch.ones(1, dtype=T64), 0.2, 0.2, torch.tensor([False]))
    assert ratio.item() == pytest.approx(2.0, abs=1e-12)  # upstream mean -> geometric mean
    same_new, _ = sde_logprob(**_unit_step([0.3, 0.3], x_next))
    same_old, _ = sde_logprob(**_unit_step([0.3, 0.3], x_next))
    assert torch.equal(same_new, same_old)  # parity alone cannot tell the two apart


def test_directed_gaussian_kl_is_0_005_in_both_directions(sde_logprob, objective):
    _, penalty = objective
    _, kl_ab = sde_logprob(**_unit_step([0.2, 0.2], [0.0, 0.0], v_ref=[0.0, 0.0]))
    _, kl_ba = sde_logprob(**_unit_step([0.0, 0.0], [0.0, 0.0], v_ref=[0.2, 0.2]))
    oracle = torch.distributions.kl_divergence(
        torch.distributions.Normal(torch.tensor(0.1, dtype=T64), 1.0),
        torch.distributions.Normal(torch.tensor(0.0, dtype=T64), 1.0),
    )
    assert oracle.item() == pytest.approx(0.005, abs=1e-15)
    assert kl_ab.item() == pytest.approx(0.005, abs=1e-15) and kl_ba.item() == pytest.approx(
        0.005, abs=1e-15
    )
    kl = torch.tensor([0.005, 99.0], dtype=T64)
    pad = torch.tensor([False, True])
    assert penalty(kl, pad, kl_beta=0.1, device="cpu").item() == pytest.approx(0.0005, abs=1e-15)
    off = penalty(kl, pad, kl_beta=0.0, device="cpu")
    assert off.item() == 0.0
    assert (
        audit.kl_status(0.0, [off.item()], source="in_process_capture")["kl_status"]
        == "not_computed"
    )  # zero is not divergence


@pytest.fixture(scope="module")
def trainer_methods():
    cuda = SimpleNamespace(
        current_stream=lambda: object(), stream=lambda s: contextlib.nullcontext()
    )
    proxy = SimpleNamespace(
        cuda=cuda, stack=torch.stack, _foreach_norm=torch._foreach_norm, randperm=torch.randperm
    )

    def clip(params, max_norm, foreach=True, pp_mesh=None, return_norm_only=False):
        return torch.nn.utils.clip_grad_norm_(params, max_norm, foreach=foreach)

    dist = SimpleNamespace(
        gradient_reduce_across_dp_replicas_=lambda params, nccl: None, gradient_norm_clipping=clip
    )
    ns = upstream.load(
        "alpagym/trainer.py.txt",
        ["AlpagymGRPOTrainer._run_training_loop", "AlpagymGRPOTrainer.all_reduce_states"],
        {"torch": proxy, "dist_util": dist},
    )
    return ns["_run_training_loop"], ns["all_reduce_states"]


@pytest.mark.parametrize(
    ("rows", "mini_batch", "iterations"), [(22, 1, 1), (22, 4, 2), (3, 8, 3), (7, 3, 2)]
)
def test_e20_pinned_loop_counts_match_formula(trainer_methods, rows, mini_batch, iterations):
    run_loop, _ = trainer_methods
    calls = []
    norms = iter([0.0 if k % 3 == 0 else 0.5 for k in range(1000)])

    def train_minibatch(samples, advantages, nccl):
        calls.append(len(samples))
        return 0.0, 0.0, 1.0, 1.0, 0.0, next(norms)

    stub = SimpleNamespace(
        _ensure_reference_model=lambda: None,
        _mini_batch=mini_batch,
        _grpo_optimization_iterations=iterations,
        _train_minibatch=train_minibatch,
    )
    torch.manual_seed(0)
    result = run_loop(stub, list(range(rows)), torch.zeros(rows), None)
    zero = sum(1 for k in range(len(calls)) if k % 3 == 0)
    assert result[2] == len(calls) == steps.minibatch_count(rows, mini_batch, iterations)
    assert sum(calls) == rows * iterations  # every row once per iteration
    assert steps.expected_optimizer_steps(rows, mini_batch, iterations, zero) == len(calls) - zero


def _digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def _snapshot(param, optimizer):
    state = optimizer.state.get(param, {})
    return {
        "parameters": {"expert.w": _digest(param)},
        "optimizer_state": {
            "expert.w": {k: _digest(v) for k, v in sorted(state.items()) if k != "step"}
        },
        "step_counters": {"expert.w": int(state["step"].item()) if "step" in state else 0},
    }


def test_e07_pinned_skip_leaves_adamw_state_unchanged_while_scheduler_advances(trainer_methods):
    _, reduce_states = trainer_methods
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))
    optimizer = torch.optim.AdamW([param], lr=0.1, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda k: 1.0)
    stepped = []
    wrapper = SimpleNamespace(
        step=lambda: (stepped.append(True), optimizer.step()), zero_grad=optimizer.zero_grad
    )
    stub = SimpleNamespace(
        train_stream=SimpleNamespace(wait_stream=lambda s: None),
        model=SimpleNamespace(parameters=lambda: iter([param])),
        config=SimpleNamespace(train=SimpleNamespace(optm_grad_norm_clip=1.0)),
        parallel_dims=SimpleNamespace(pp_enabled=False),
        optimizers=wrapper,
    )
    scheduler_before = {"last_epoch": scheduler.last_epoch, "lrs": scheduler.get_last_lr()}
    minibatches = []
    for index, grad in enumerate(([0.0, 0.0, 0.0], [0.3, 0.0, -0.4])):
        before, calls = _snapshot(param, optimizer), len(stepped)
        param.grad = torch.tensor(grad)
        norm = reduce_states(stub, None)
        after = _snapshot(param, optimizer)
        minibatches.append(
            {
                "index": index,
                "grad_norm": norm,
                "optimizer_step_called": len(stepped) > calls,
                "before": before,
                "after": after,
            }
        )
    scheduler.step()
    assert minibatches[0]["grad_norm"] == 0.0 and minibatches[1]["grad_norm"] == pytest.approx(0.5)
    assert len(stepped) == 1
    assert minibatches[0]["before"] == minibatches[0]["after"]  # bitwise: weights, moments, counter
    report = steps.check_skip_semantics(
        {
            "trainer_step": 1,
            "scheduler_before": scheduler_before,
            "scheduler_after": {"last_epoch": scheduler.last_epoch, "lrs": scheduler.get_last_lr()},
            "minibatches": minibatches,
        },
        scope="synthetic",
    )
    assert report["status"] == "passed", report["issues"]
    assert report["minibatches"][1]["parameters_changed"] is True


def test_nonfinite_reduced_gradient_raises(trainer_methods):
    _, reduce_states = trainer_methods
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([param], lr=0.1)
    stub = SimpleNamespace(
        train_stream=SimpleNamespace(wait_stream=lambda s: None),
        model=SimpleNamespace(parameters=lambda: iter([param])),
        config=SimpleNamespace(train=SimpleNamespace(optm_grad_norm_clip=1.0)),
        parallel_dims=SimpleNamespace(pp_enabled=False),
        optimizers=optimizer,
    )
    param.grad = torch.tensor([float("nan")])
    with pytest.raises(FloatingPointError):
        reduce_states(stub, None)
