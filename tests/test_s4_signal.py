"""Known answers for the pinned float32 GRPO advantage (E06-E08, E19; build-spec A9).

Oracles are independent of ``teacher_rl.checks.signal``:

1. ``emulated_advantages`` - IEEE float32 emulated in pure Python (``struct`` rounding
   after each exact double operation; correctly rounded for + - * / sqrt), with
   numpy's documented reduction order (sequential below 8 elements, eight-way
   pairwise blocks up to 128). No numpy is used by the oracle.
2. The pinned ``GRPO.compute_advantage`` method executed from the hash-checked
   vendored ``cosmos_rl/dispatcher/algo/grpo.py`` (AST-selected, restricted namespace).
3. R3's printed literals (lane-r3 section 9, E19), compared after the derivation.
"""

import importlib.util
import math
import os
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from teacher_rl.checks import signal

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "s4_upstream_loader", ROOT / "tests/fixtures/s4_upstream_checks/upstream_loader.py"
)
upstream = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(upstream)

# Resolved Cosmos GRPO settings the pinned formula assumes (engineering fact, not owner).
PINNED_CONFIG = {"unbiased_advantage": False, "eps": 1e-6}


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def emulated_sum(values):
    if len(values) < 8:
        total = 0.0
        for v in values:
            total = f32(total + v)
        return total
    assert len(values) <= 128
    acc = list(values[:8])
    i = 8
    while i < len(values) - len(values) % 8:
        acc = [f32(acc[k] + values[i + k]) for k in range(8)]
        i += 8
    total = f32(
        f32(f32(acc[0] + acc[1]) + f32(acc[2] + acc[3]))
        + f32(f32(acc[4] + acc[5]) + f32(acc[6] + acc[7]))
    )
    for v in values[i:]:
        total = f32(total + v)
    return total


def emulated_advantages(rewards, eps=1e-6):
    r = [f32(x) for x in rewards]
    n = len(r)
    mean = f32(emulated_sum(r) / n)
    centred = [f32(x - mean) for x in r]
    var = f32(emulated_sum([f32(c * c) for c in centred]) / n)
    denominator = f32(f32(math.sqrt(var)) + f32(eps))
    return [f32(c / denominator) for c in centred]


@pytest.fixture(scope="module")
def pinned_compute_advantage():
    ns = upstream.load("cosmos_rl/grpo.py.txt", ["GRPO.compute_advantage"], {"np": np})
    stub = SimpleNamespace(unbiased=False, eps=1e-6)
    return lambda rewards: ns["compute_advantage"](stub, rewards)


E19_CASES = {
    "g2_gap_0.001": ([-0.213, -0.214], [0.99799, -0.99802]),
    "g2_gap_5": ([0.0, -5.0], [0.9999996, -0.9999996]),
    "g2_gap_10": ([1.0, -9.0], [0.9999998, -0.9999998]),
    "g6_one_collision": ([0.5] * 5 + [-9.5], [0.4472135] * 5 + [-2.2360673]),
}


@pytest.mark.parametrize("name", sorted(E19_CASES))
def test_e19_literals_match_independent_float32_emulation(name, pinned_compute_advantage):
    rewards, printed = E19_CASES[name]
    got = signal.group_advantages_f32(rewards)
    assert got == emulated_advantages(rewards)  # exact float32 equality
    assert got == pinned_compute_advantage(rewards)
    for value, literal in zip(got, printed):
        digits = len(repr(literal).split(".")[1])
        assert abs(value - literal) <= 0.5 * 10.0**-digits


def test_e19_reproduces_r3_full_precision_literals():
    assert signal.group_advantages_f32([-0.213, -0.214]) == [
        0.9979891180992126,
        -0.9980188608169556,
    ]
    assert signal.group_advantages_f32([1.0, -9.0]) == [0.9999998211860657, -0.9999998211860657]
    six = signal.group_advantages_f32([0.5] * 5 + [-9.5])
    assert six == [0.44721347093582153] * 5 + [-2.236067295074463]
    # -sqrt(5) in float64 differs: the gap is float32 rounding, not an error.
    assert six[-1] != -math.sqrt(5) and abs(six[-1] + math.sqrt(5)) < 1e-6


def test_e19_noise_only_group_has_unit_rms_and_magnitude_is_lost():
    noise = [0.13, 0.07, 0.11, 0.05, 0.09, 0.10]
    adv = signal.group_advantages_f32(noise)
    assert adv == emulated_advantages(noise)
    assert abs(math.sqrt(sum(a * a for a in adv) / 6) - 1.0) < 1e-4
    scaled = signal.group_advantages_f32([x * 1000 for x in noise])
    assert abs(math.sqrt(sum(a * a for a in scaled) / 6) - 1.0) < 1e-6


@pytest.mark.parametrize("size", [2, 4, 8, 16])
@pytest.mark.parametrize("value", [0.7, -9.3, 0.1, -0.213, 1.0 / 3.0, 0.123456789])
def test_e07_tied_groups_are_exact_zero_at_power_of_two_sizes(size, value):
    adv = signal.group_advantages_f32([value] * size)
    assert adv == emulated_advantages([value] * size) == [0.0] * size


def test_e07_tied_group_of_six_gives_nonzero_same_sign_advantage(pinned_compute_advantage):
    rewards = [-0.213] * 6
    adv = signal.group_advantages_f32(rewards)
    emulated = emulated_advantages(rewards)
    assert adv == emulated == pinned_compute_advantage(rewards)
    assert adv == [0.014682376757264137] * 6
    assert all(a > 0 for a in adv)


def test_e08_near_tie_asymmetry_literal():
    adv = signal.group_advantages_f32([-0.2, -0.2000001])
    assert adv == emulated_advantages([-0.2, -0.2000001])
    assert adv == [0.042466215789318085, -0.05662161856889725]


def test_pinned_reproduction_matches_emulation_on_random_groups(pinned_compute_advantage):
    rng = np.random.default_rng(20261109)
    for size in (2, 3, 5, 6, 7, 9, 12, 17):
        for _ in range(40):
            rewards = rng.uniform(-10.0, 1.0, size).tolist()
            got = signal.group_advantages_f32(rewards)
            assert got == emulated_advantages(rewards) == pinned_compute_advantage(rewards)
            tied = [rewards[0]] * size
            assert signal.group_advantages_f32(tied) == emulated_advantages(tied)


def test_advantage_scalar_stays_float32():
    r = np.array([0.0, -5.0]).astype(np.float32)
    assert (np.std(r) + 1e-6).dtype == np.float32  # NEP 50 assumption the module relies on


@pytest.mark.parametrize(
    "bad", [[], [float("nan"), 1.0], [float("inf")], [True, 1.0], ["1"], [1e39, 0.0], None]
)
def test_group_advantages_reject_invalid_rewards(bad):
    with pytest.raises(ValueError):
        signal.group_advantages_f32(bad)


def test_group_of_one_is_zero_like_upstream(pinned_compute_advantage):
    assert signal.group_advantages_f32([3.5]) == pinned_compute_advantage([3.5]) == [0.0]


# --- E07 equal-reward control -------------------------------------------------------


def _g(gid, rewards, advantages=None):
    if advantages is None:
        advantages = emulated_advantages(rewards)
    return {"group_id": gid, "rewards": rewards, "advantages": advantages}


def test_e07_control_blocks_without_owner_criterion_but_reports_facts():
    report = signal.check_equal_reward_control(
        [_g("a", [0.7, 0.7])], None, scope="synthetic", advantage_config=PINNED_CONFIG
    )
    assert report["status"] == "blocked"
    assert "e07_criterion_unregistered" in {i["code"] for i in report["issues"]}
    row = report["groups"][0]
    assert row["captured_exact_zero"] is True and row["meets_criterion"] is None
    assert report["artifact_kind"] == "s4_e07_equal_reward_control"
    assert report["schema_version"] == 1 and report["scope"] == "synthetic"


def test_e07_exact_zero_passes_at_g2_and_fails_at_g6_tied_value():
    ok = signal.check_equal_reward_control(
        [_g("g2", [0.7, 0.7])],
        {"mode": "exact_zero"},
        scope="synthetic",
        advantage_config=PINNED_CONFIG,
    )
    assert ok["status"] == "passed"
    six = signal.check_equal_reward_control(
        [_g("g6", [-0.213] * 6)],
        {"mode": "exact_zero"},
        scope="synthetic",
        advantage_config=PINNED_CONFIG,
    )
    assert six["status"] == "blocked"
    codes = {i["code"] for i in six["issues"]}
    assert {"e07_criterion_not_met", "pinned_tied_group_nonzero"} <= codes
    assert six["groups"][0]["captured_sign_pattern"] == [1]


def test_e07_float32_tolerance_criterion_is_applied_as_given():
    report = signal.check_equal_reward_control(
        [_g("g6", [-0.213] * 6)],
        {"mode": "float32_tolerance", "value": 0.02},
        scope="synthetic",
        advantage_config=PINNED_CONFIG,
    )
    assert report["status"] == "passed"
    assert report["groups"][0]["captured_max_abs_advantage"] == 0.014682376757264137


def test_e07_negative_controls():
    untied = signal.check_equal_reward_control(
        [_g("u", [0.7, 0.6])],
        {"mode": "exact_zero"},
        scope="synthetic",
        advantage_config=PINNED_CONFIG,
    )
    assert "rewards_not_tied" in {i["code"] for i in untied["issues"]}
    wrong = signal.check_equal_reward_control(
        [_g("w", [0.7, 0.7], [0.0, 1e-9])],
        {"mode": "exact_zero"},
        scope="synthetic",
        advantage_config=PINNED_CONFIG,
    )
    assert "captured_advantage_mismatch" in {i["code"] for i in wrong["issues"]}
    for criterion in (
        {"mode": "float32_tolerance"},
        {"mode": "loose"},
        {"mode": "exact_zero", "value": 0},
    ):
        with pytest.raises(ValueError):
            signal.check_equal_reward_control(
                [_g("a", [1.0, 1.0])], criterion, scope="synthetic", advantage_config=PINNED_CONFIG
            )
    with pytest.raises(ValueError):
        signal.check_equal_reward_control(
            [_g("a", [1.0, 1.0])],
            {"mode": "exact_zero"},
            scope=None,
            advantage_config=PINNED_CONFIG,
        )
    with pytest.raises(ValueError):
        signal.check_equal_reward_control(
            [_g("a", [1.0, 1.0]), _g("a", [1.0, 1.0])],
            None,
            scope="synthetic",
            advantage_config=PINNED_CONFIG,
        )


# --- E06 recomputation --------------------------------------------------------------


def _members(rewards, advantages=None):
    advantages = emulated_advantages(rewards) if advantages is None else advantages
    return [
        {"member_id": f"m{i}", "reward": r, "advantage": a}
        for i, (r, a) in enumerate(zip(rewards, advantages))
    ]


def test_e06_bitwise_recomputation_passes_and_blocks_without_owner_inputs():
    groups = [{"group_id": "p0", "members": _members([0.5, -9.5, 0.4, 0.3, 0.2, 0.1])}]
    ok = signal.check_advantage_recomputation(
        groups,
        group_size=6,
        tolerance={"mode": "bitwise"},
        scope="synthetic",
        advantage_config=PINNED_CONFIG,
    )
    assert ok["status"] == "passed" and ok["groups"][0]["bitwise_equal"] is True
    blocked = signal.check_advantage_recomputation(
        groups, group_size=None, tolerance=None, scope="synthetic", advantage_config=PINNED_CONFIG
    )
    assert blocked["status"] == "blocked"
    assert {"group_size_unregistered", "tolerance_unregistered"} <= {
        i["code"] for i in blocked["issues"]
    }
    assert blocked["groups"][0]["max_abs_difference"] == 0.0


def test_e06_negative_controls_subset_duplicate_and_drift():
    full = _members([1.0, 0.0, -1.0])
    subset = [{"group_id": "p0", "members": full[:2]}]
    report = signal.check_advantage_recomputation(
        subset,
        group_size=3,
        tolerance={"mode": "bitwise"},
        scope="synthetic",
        advantage_config=PINNED_CONFIG,
    )
    codes = {i["code"] for i in report["issues"]}
    assert "group_size_mismatch" in codes and "advantage_recomputation_mismatch" in codes
    dup = [
        {"group_id": "p0", "members": _members([1.0, 0.0])},
        {"group_id": "p1", "members": _members([1.0, 0.0])},
    ]
    assert "duplicate_member" in {
        i["code"]
        for i in signal.check_advantage_recomputation(
            dup,
            group_size=2,
            tolerance={"mode": "bitwise"},
            scope="synthetic",
            advantage_config=PINNED_CONFIG,
        )["issues"]
    }
    drift = [{"group_id": "p0", "members": _members([1.0, 0.0], [1.0, -0.9999])}]
    loose = signal.check_advantage_recomputation(
        drift,
        group_size=2,
        tolerance={"mode": "absolute", "value": 1e-3},
        scope="synthetic",
        advantage_config=PINNED_CONFIG,
    )
    assert loose["status"] == "passed" and loose["groups"][0]["bitwise_equal"] is False
    tight = signal.check_advantage_recomputation(
        drift,
        group_size=2,
        tolerance={"mode": "bitwise"},
        scope="synthetic",
        advantage_config=PINNED_CONFIG,
    )
    assert tight["status"] == "blocked"


def test_e06_reports_float32_order_sensitivity():
    rng = np.random.default_rng(7)
    sensitive = None
    for _ in range(500):
        rewards = rng.uniform(-10, 1, 6).tolist()
        order = sorted(range(6), key=lambda i: f"m{i}")
        base = emulated_advantages(rewards)
        perm = emulated_advantages([rewards[i] for i in reversed(order)])
        if any(perm[k] != base[i] for k, i in enumerate(reversed(order))):
            sensitive = rewards
            break
    assert sensitive is not None  # summation order matters in float32 for some groups
    members = _members(sensitive)
    members = list(reversed(members))  # captured order differs from member-id order
    captured = [
        {**m, "advantage": a}
        for m, a in zip(members, emulated_advantages([m["reward"] for m in members]))
    ]
    report = signal.check_advantage_recomputation(
        [{"group_id": "p", "members": captured}],
        group_size=6,
        tolerance={"mode": "bitwise"},
        scope="synthetic",
        advantage_config=PINNED_CONFIG,
    )
    assert report["groups"][0]["bitwise_equal"] is True
    assert report["groups"][0]["order_sensitive"] is True


# --- E08 ordering -------------------------------------------------------------------


def test_e08_reversal_and_sign_inversion_detection():
    groups = [{"group_id": "g", "rewards": [0.3, -1.0, 2.0, 0.5], "advantages": None}]
    ok = signal.check_advantage_ordering(groups, scope="synthetic")
    assert ok["status"] == "passed"
    assert ok["groups"][0]["order_violations"] == [] == ok["groups"][0]["reversal_violations"]
    inverted = signal.check_advantage_ordering(
        groups,
        scope="synthetic",
        advantage_fn=lambda r: [-a for a in signal.group_advantages_f32(r)],
    )
    codes = {i["code"] for i in inverted["issues"]}
    assert {"advantage_order_violation", "reversal_order_violation"} <= codes
    captured_inverted = [
        {"group_id": "c", "rewards": [1.0, 0.0], "advantages": [-0.9999995, 0.9999995]}
    ]
    report = signal.check_advantage_ordering(captured_inverted, scope="synthetic")
    assert report["status"] == "blocked" and report["groups"][0]["order_violations"] == [[0, 1]]


def test_e08_float32_collapse_and_asymmetry_are_reported_as_facts():
    collapse = signal.check_advantage_ordering(
        [{"group_id": "c", "rewards": [0.1, 0.1 + 1e-12], "advantages": None}], scope="synthetic"
    )
    assert collapse["status"] == "passed"
    assert collapse["groups"][0]["float32_collapsed_pairs"] == [[0, 1]]
    assert collapse["groups"][0]["advantages"] == [0.0, 0.0]
    near = signal.check_advantage_ordering(
        [{"group_id": "n", "rewards": [-0.2, -0.2000001], "advantages": None}], scope="synthetic"
    )
    assert near["groups"][0]["magnitude_asymmetric"] is True
    assert "float32_near_tie_asymmetry" in {i["code"] for i in near["issues"]}


def test_reports_are_deterministic_and_input_order_invariant():
    a = [_g("x", [1.0, 1.0]), _g("y", [0.2, 0.2])]
    r1 = signal.check_equal_reward_control(
        a, {"mode": "exact_zero"}, scope="synthetic", advantage_config=PINNED_CONFIG
    )
    r2 = signal.check_equal_reward_control(
        list(reversed(a)), {"mode": "exact_zero"}, scope="synthetic", advantage_config=PINNED_CONFIG
    )
    assert r1["groups"] == r2["groups"] and r1["status"] == r2["status"]
    assert r1["producer"]["module"] == "teacher_rl.checks.signal"


def test_non_torch_check_modules_import_without_torch():
    code = (
        "import sys\n"
        "import teacher_rl.checks, teacher_rl.checks.signal, teacher_rl.checks.steps\n"
        "import teacher_rl.checks.semantics, teacher_rl.checks.audit, teacher_rl.diagnostics\n"
        "assert 'torch' not in sys.modules, 'torch imported'\n"
    )
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        check=False,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_e06_bitwise_means_no_rerounding_of_captured_values():
    exact = emulated_advantages([1.0, 0.0])
    nudged = [exact[0] + 2**-40, exact[1]]  # same float32 after rounding, different double
    assert f32(nudged[0]) == exact[0] and nudged[0] != exact[0]
    report = signal.check_advantage_recomputation(
        [{"group_id": "p", "members": _members([1.0, 0.0], nudged)}],
        group_size=2,
        tolerance={"mode": "bitwise"},
        scope="synthetic",
        advantage_config=PINNED_CONFIG,
    )
    assert report["status"] == "blocked" and report["groups"][0]["bitwise_equal"] is False


# --- independent-review regressions (A9-07, A9-10, A9-15) ------------------------------

# Found once by a seeded search (numpy default_rng(7), uniform(-10, 1, 6)); reversing this
# group changes float32 bits, so it is order sensitive in the pinned formula.
ORDER_SENSITIVE = [
    -3.123949867348663,
    -0.1306481893346697,
    -1.4674574073028714,
    -7.522720910103489,
    -6.69817086597652,
    -0.39091210064111914,
]


def test_e06_order_sensitivity_found_when_captured_order_is_member_id_order():
    reversed_adv = emulated_advantages(list(reversed(ORDER_SENSITIVE)))
    base = emulated_advantages(ORDER_SENSITIVE)
    assert any(reversed_adv[5 - k] != base[k] for k in range(6))  # independent oracle
    report = signal.check_advantage_recomputation(
        [{"group_id": "p", "members": _members(ORDER_SENSITIVE)}],  # m0..m5 = sorted order
        group_size=6,
        tolerance={"mode": "bitwise"},
        advantage_config=PINNED_CONFIG,
        scope="synthetic",
    )
    row = report["groups"][0]
    assert row["bitwise_equal"] is True and row["order_sensitive"] is True
    single = signal.check_advantage_recomputation(
        [{"group_id": "s", "members": _members([0.5])}],
        group_size=1,
        tolerance={"mode": "bitwise"},
        advantage_config=PINNED_CONFIG,
        scope="synthetic",
    )
    assert single["groups"][0]["order_sensitive"] == "not_tested"


# A9R residual: re-verifier literal. Members 0 and 1 carry the same reward, so the
# member-id sort [1, 0, 2, ...] only swaps tied members and reproduces the captured
# reward sequence; the fallback must then use the reversal.
TIED_ORDER_SENSITIVE = [
    0.21032125256406453,
    0.21032125256406453,
    -9.225373662303834,
    -8.572486556607721,
    0.4316129862095257,
    -3.159280479239789,
]


def test_e06_member_id_sort_that_only_swaps_tied_rewards_falls_back_to_reversal():
    base = emulated_advantages(TIED_ORDER_SENSITIVE)
    reversed_adv = emulated_advantages(list(reversed(TIED_ORDER_SENSITIVE)))
    assert any(reversed_adv[5 - k] != base[k] for k in range(6))  # independent oracle
    ids = ["m1", "m0", "m2", "m3", "m4", "m5"]
    members = [
        {"member_id": i, "reward": r, "advantage": a}
        for i, r, a in zip(ids, TIED_ORDER_SENSITIVE, base)
    ]
    report = signal.check_advantage_recomputation(
        [{"group_id": "t", "members": members}],
        group_size=6,
        tolerance={"mode": "bitwise"},
        advantage_config=PINNED_CONFIG,
        scope="synthetic",
    )
    row = report["groups"][0]
    assert row["bitwise_equal"] is True and row["order_sensitive"] is True


def test_e06_order_not_tested_when_every_permutation_tried_reproduces_rewards():
    # All rewards equal: neither the member-id sort nor the reversal changes the
    # reward sequence, so no different summation order was exercised.
    rewards = [0.25, 0.25, 0.25, 0.25]
    members = [
        {"member_id": i, "reward": r, "advantage": a}
        for i, r, a in zip(["m2", "m0", "m3", "m1"], rewards, emulated_advantages(rewards))
    ]
    report = signal.check_advantage_recomputation(
        [{"group_id": "e", "members": members}],
        group_size=4,
        tolerance={"mode": "bitwise"},
        advantage_config=PINNED_CONFIG,
        scope="synthetic",
    )
    assert report["groups"][0]["order_sensitive"] == "not_tested"


# --- A9R2 residual: "reproduces the sequence" means equal float32 VALUES -----------------


def _order_row(rewards, ids):
    members = [
        {"member_id": i, "reward": r, "advantage": a}
        for i, r, a in zip(ids, rewards, emulated_advantages(rewards))
    ]
    report = signal.check_advantage_recomputation(
        [{"group_id": "v", "members": members}],
        group_size=len(rewards),
        tolerance={"mode": "bitwise"},
        advantage_config=PINNED_CONFIG,
        scope="synthetic",
    )
    row = report["groups"][0]
    return report["status"], row["bitwise_equal"], row["order_sensitive"]


def _reversal_changes_bits(rewards):
    base = emulated_advantages(rewards)
    reversed_adv = emulated_advantages(list(reversed(rewards)))
    n = len(rewards)
    return any(
        struct.pack("<f", reversed_adv[n - 1 - k]) != struct.pack("<f", base[k]) for k in range(n)
    )


# Re-verifier literal: the member-id sort [1, 0, 2, ...] swaps 0.0 and -0.0 (different
# bytes, equal float32 values), so it reproduces the summed sequence; the reversal does not.
SIGNED_ZERO_SWAP = [
    0.0,
    -0.0,
    -9.057859161420133,
    -7.395084427442903,
    -1.185980882729634,
    -3.5962176032919544,
]

# Re-verifier literal: members 0 and 1 are distinct doubles (Python !=) that round to the
# same float32 (same bytes and value), so the member-id sort reproduces the sequence.
ROUND_TO_SAME_F32 = [
    0.1,
    0.10000000000100001,
    -1.144967838800818,
    -1.112651312898569,
    -4.331418828536438,
    -6.856184819030442,
]

SWAP_FIRST_TWO = ["m1", "m0", "m2", "m3", "m4", "m5"]


def test_e06_signed_zero_swap_reproduces_the_sequence_and_falls_back_to_reversal():
    r = SIGNED_ZERO_SWAP
    assert r[0] == r[1] and struct.pack("<f", r[0]) != struct.pack("<f", r[1])
    assert _reversal_changes_bits(r)  # independent oracle
    assert _order_row(r, SWAP_FIRST_TWO) == ("passed", True, True)


def test_e06_all_tied_signed_zeros_are_not_tested():
    r = [0.0, -0.0, 0.0, -0.0]  # reversal [-0.0, 0.0, -0.0, 0.0]: equal float32 values
    assert _order_row(r, ["m1", "m0", "m3", "m2"]) == ("passed", True, "not_tested")


def test_e06_distinct_doubles_rounding_to_one_float32_reproduce_the_sequence():
    r = ROUND_TO_SAME_F32
    assert r[0] != r[1] and struct.pack("<f", r[0]) == struct.pack("<f", r[1])
    assert _reversal_changes_bits(r)  # independent oracle
    assert _order_row(r, SWAP_FIRST_TWO) == ("passed", True, True)


# Seeded search (random.Random(11), palindromes [a, b, c, c, b, a] rounded to 6 places):
# the reversal reproduces this sequence, while the member-id sort [2, 1, 0, 3, 4, 5]
# changes it and changes float32 bits, so only the member-id order can detect sensitivity.
PALINDROME_MEMBER_ORDER_SENSITIVE = [-7.968736, -4.369005, -3.07129, -3.07129, -4.369005, -7.968736]


def test_e06_member_id_order_is_used_when_it_changes_the_sequence():
    r = PALINDROME_MEMBER_ORDER_SENSITIVE
    order = [2, 1, 0, 3, 4, 5]
    base = emulated_advantages(r)
    perm = emulated_advantages([r[i] for i in order])
    assert any(perm[p] != base[i] for p, i in enumerate(order))  # independent oracle
    assert not _reversal_changes_bits(r)
    assert _order_row(r, ["m2", "m1", "m0", "m3", "m4", "m5"]) == ("passed", True, True)


def test_nonfinite_captures_use_tokens_and_reach_the_nonfinite_issue():
    e07 = signal.check_equal_reward_control(
        [_g("t", ["nan", 1.0], [0.0, 0.0])],
        {"mode": "exact_zero"},
        advantage_config=PINNED_CONFIG,
        scope="synthetic",
    )
    assert e07["status"] == "blocked" and e07["groups"][0]["evaluated"] is False
    assert "nonfinite_value" in {i["code"] for i in e07["issues"]}
    e06 = signal.check_advantage_recomputation(
        [{"group_id": "p", "members": _members([1.0, 0.0], [1.0, "-inf"])}],
        group_size=2,
        tolerance={"mode": "bitwise"},
        advantage_config=PINNED_CONFIG,
        scope="synthetic",
    )
    assert "nonfinite_value" in {i["code"] for i in e06["issues"]}
    e08 = signal.check_advantage_ordering(
        [{"group_id": "g", "rewards": [1.0, 0.0], "advantages": ["inf", 0.0]}], scope="synthetic"
    )
    assert "nonfinite_value" in {i["code"] for i in e08["issues"]}
    injected = signal.check_advantage_ordering(
        [{"group_id": "g", "rewards": [1.0, 0.0], "advantages": None}],
        scope="synthetic",
        advantage_fn=lambda r: [math.nan] * len(r),
    )
    assert "nonfinite_value" in {i["code"] for i in injected["issues"]}
    for call in (
        lambda: signal.check_equal_reward_control(
            [_g("t", [math.nan, 1.0], [0.0, 0.0])],
            {"mode": "exact_zero"},
            advantage_config=PINNED_CONFIG,
            scope="synthetic",
        ),
        lambda: signal.check_advantage_recomputation(
            [{"group_id": "p", "members": _members([1.0, 0.0], [1.0, math.inf])}],
            group_size=2,
            tolerance={"mode": "bitwise"},
            advantage_config=PINNED_CONFIG,
            scope="synthetic",
        ),
        lambda: signal.check_advantage_ordering(
            [{"group_id": "g", "rewards": [1.0, 0.0], "advantages": [math.inf, 0.0]}],
            scope="synthetic",
        ),
    ):
        with pytest.raises(ValueError, match="token"):
            call()


def test_resolved_advantage_config_must_match_the_reproduced_convention():
    source = upstream.source_bytes("cosmos_rl/grpo.py.txt").decode()
    assert "unbiased: bool = False, eps: float = 1e-6" in source  # pinned defaults
    group = [_g("g2", [0.7, 0.7])]
    members = [{"group_id": "p", "members": _members([1.0, 0.0])}]
    cases = {
        "advantage_config_unresolved": None,
        "advantage_config_not_pinned": {"unbiased_advantage": True, "eps": 1e-6},
    }
    for code, config in cases.items():
        e07 = signal.check_equal_reward_control(
            group, {"mode": "exact_zero"}, advantage_config=config, scope="synthetic"
        )
        e06 = signal.check_advantage_recomputation(
            members,
            group_size=2,
            tolerance={"mode": "bitwise"},
            advantage_config=config,
            scope="synthetic",
        )
        for report in (e07, e06):
            assert report["status"] == "blocked", code
            assert code in {i["code"] for i in report["issues"]}, code
    eps = signal.check_advantage_recomputation(
        members,
        group_size=2,
        tolerance={"mode": "bitwise"},
        advantage_config={"unbiased_advantage": False, "eps": 1e-5},
        scope="synthetic",
    )
    assert "advantage_config_not_pinned" in {i["code"] for i in eps["issues"]}
    for bad in ({"unbiased_advantage": 0, "eps": 1e-6}, {"eps": 1e-6}, "default"):
        with pytest.raises(ValueError):
            signal.check_equal_reward_control(
                group, {"mode": "exact_zero"}, advantage_config=bad, scope="synthetic"
            )
