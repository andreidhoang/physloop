"""Group-advantage checks for the shipped GRPO signal (E06, E07, E08, E19; build-spec A9).

Serves S4.2/S4.3 [C] (G9). Agent-authored verification of the *pinned* Cosmos-RL
advantage, ``cosmos-rl@d2a2c57:cosmos_rl/dispatcher/algo/grpo.py:38-47``::

    rewards = np.array(rewards).astype(np.float32)
    mean = np.mean(rewards)
    result = (rewards - mean) / (np.std(rewards) + self.eps)   # eps = 1e-6, unbiased False

used for every payload (``cosmos_rl/reward/base.py:73``) with no tied-group filter
(the drop exists only for variant ``dapo``; the default is ``grpo``) and no advantage
clipping. It reproduces that convention; it never proposes a different estimator.

Findings this module makes checkable (R3 section 1, finding 1):

- In float32 an all-tied group gives exactly zero for G = 2, 4, 8, 16 but can give a
  nonzero *same-sign* advantage for other G (literal: six rewards of ``-0.213`` give
  ``0.014682376757264137`` each), so the protocol's E07 "exact zero" is not a property
  of the pinned path for every group size. The E07 pass criterion is Huy's input.
- E19 magnitudes: any untied group is standardized to unit RMS, so reward magnitude
  is lost; at G = 2 the advantage is about +-1 whatever the gap.
- E08: in float32 distinct rewards can collapse to one value (ordering lost) and near
  ties give asymmetric magnitudes (``[-0.2, -0.2000001]`` -> ``+0.0425 / -0.0566``).

Verification level per function (build-spec section 1):

- ``group_advantages_f32`` - L1 reproduction of the pinned formula; literal known
  answers derived in the tests by an independent IEEE float32 emulation. L2 caveat:
  the AlpaGym lock pins numpy 2.2.6 (Linux); this CPU run uses the local numpy 2.x.
  numpy >= 2 is required because NEP 50 keeps ``np.std(r) + 1e-6`` in float32; numpy
  1.x would promote to float64 and change every value, so it is refused.
- ``check_equal_reward_control`` - L1 over captured groups (E07); actual capture L3.
- ``check_advantage_recomputation`` - L1 over captured groups (E06); actual L3.
- ``check_advantage_ordering`` - L1 (E08).

Owner inputs (required, never defaulted): the E07 pass criterion, the recomputation
tolerance and the registered group size. Missing -> ``blocked`` naming the field.

Resolved-config input (engineering fact, not an owner slot): E06/E07 take
``advantage_config = {"unbiased_advantage": bool, "eps": float}``, the resolved Cosmos
GRPO settings (``train.train_policy.unbiased_advantage``; ``GRPO(eps=...)``). The
oracle reproduces only ``unbiased_advantage=False, eps=1e-6`` (the pinned defaults,
``grpo.py:24``); ``None`` blocks with ``advantage_config_unresolved`` and any other
value with ``advantage_config_not_pinned``, so a configuration difference is never
reported as a captured-advantage mismatch alone.

Captured value convention: finite int/float, or the token ``"nan"``, ``"inf"`` or
``"-inf"`` for a non-finite capture (JSON-safe, so the artifact can be hashed and the
``nonfinite_value`` issue is reported). A raw float NaN/inf is refused with
``ValueError`` naming the tokens.

This module never imports torch, never filters or re-weights groups, never changes
an advantage, and never treats a recomputation match as a learning pass.
"""

import struct

import numpy as np

from teacher_rl.checks import (
    Issues,
    artifact,
    check_scope,
    is_real,
    is_text,
    require,
)

PINNED_EPS = 1e-6
PINNED_SOURCE = (
    "cosmos-rl@d2a2c57c4bd6496482bc42da19a59b4432705eda:cosmos_rl/dispatcher/algo/grpo.py:38-47"
)
E07_MODES = ("exact_zero", "float32_tolerance")
NONFINITE_TOKENS = ("nan", "inf", "-inf")
PINNED_ADVANTAGE_CONFIG = {"unbiased_advantage": False, "eps": PINNED_EPS}
TOLERANCE_MODES = ("bitwise", "absolute")

_GROUP_KEYS = {"group_id", "rewards", "advantages"}
_MEMBER_KEYS = {"member_id", "reward", "advantage"}


def _require_numpy2():
    major = int(np.__version__.split(".")[0])
    if major < 2:
        raise RuntimeError(
            "numpy >= 2 (NEP 50) is required to reproduce the pinned float32 advantage; "
            f"found {np.__version__}"
        )


def _reward_list(rewards, path="rewards"):
    require(type(rewards) is list and rewards, f"{path} must be a nonempty list")
    require(all(is_real(r) for r in rewards), f"{path} must hold finite int/float values")
    return rewards


def float32_values(values):
    """Return each value rounded to float32 (as Python floats)."""
    with np.errstate(over="ignore"):
        return np.array(_reward_list(values)).astype(np.float32).tolist()


def group_advantages_f32(rewards):
    """Return the pinned float32 group-standardized advantages for one group (L1).

    Exactly the pinned expression (float32 cast, numpy mean, population std, eps
    1e-6, no tied-group filter). Non-finite or non-numeric rewards, and rewards that
    overflow float32, raise ``ValueError``: the check never computes on them.
    """
    _require_numpy2()
    _reward_list(rewards)
    with np.errstate(over="ignore"):
        r = np.array(rewards).astype(np.float32)
    require(bool(np.all(np.isfinite(r))), "rewards overflow float32")
    mean = np.mean(r)
    result = (r - mean) / (np.std(r) + PINNED_EPS)
    require(result.dtype == np.float32, "advantage left float32 (NEP 50 assumption broken)")
    return result.tolist()


def _same_bits(a, b):
    """Bitwise equality of a captured value with a recomputed float32-as-float value.

    No re-rounding: a captured float64 that merely rounds to the same float32 differs.
    """
    return is_real(a) and struct.pack("<d", float(a)) == struct.pack("<d", float(b))


def _check_captured_values(values, path):
    """Captured numbers are finite int/float or a non-finite token (never raw NaN/inf)."""
    for value in values:
        if type(value) is str:
            require(value in NONFINITE_TOKENS, f"{path}: unknown token {value!r}")
            continue
        require(
            type(value) in (int, float),
            f"{path} must hold numbers or a non-finite token {NONFINITE_TOKENS}",
        )
        require(
            is_real(value),
            f"{path}: encode non-finite captures as the token 'nan', 'inf' or '-inf'",
        )


def _check_advantage_config(config, issues):
    """Validate the resolved Cosmos GRPO config and add the blocking issue, if any."""
    if config is None:
        issues.add(
            "advantage_config_unresolved",
            "advantage_config",
            "Resolve train.train_policy.unbiased_advantage and the GRPO eps from the run "
            "config; the oracle reproduces only unbiased_advantage=False, eps=1e-6.",
        )
        return
    require(
        type(config) is dict
        and set(config) == {"unbiased_advantage", "eps"}
        and type(config["unbiased_advantage"]) is bool
        and type(config["eps"]) in (int, float)
        and is_real(config["eps"]),
        "advantage_config must be {'unbiased_advantage': bool, 'eps': finite number} or None",
    )
    if config != PINNED_ADVANTAGE_CONFIG:
        issues.add(
            "advantage_config_not_pinned",
            "advantage_config",
            f"Resolved {config} differs from the reproduced {PINNED_ADVANTAGE_CONFIG}; "
            "the recomputation does not apply.",
        )


def _check_group(group, path):
    require(
        type(group) is dict and set(group) == _GROUP_KEYS,
        f"{path} needs exactly {sorted(_GROUP_KEYS)}",
    )
    require(is_text(group["group_id"]), f"{path}.group_id must be text")
    require(
        type(group["rewards"]) is list and group["rewards"],
        f"{path}.rewards must be a nonempty list",
    )
    require(
        type(group["advantages"]) is list and len(group["advantages"]) == len(group["rewards"]),
        f"{path}.advantages must be a list aligned with rewards",
    )
    _check_captured_values(group["rewards"], f"{path}.rewards")
    _check_captured_values(group["advantages"], f"{path}.advantages")


def _unique_ids(groups, key, path):
    ids = [g[key] for g in groups]
    require(len(set(ids)) == len(ids), f"{path} repeats a {key}")


def check_equal_reward_control(captured, criterion, *, advantage_config, scope):
    """E07 equal-reward control over captured all-tied groups (L1; actual capture L3).

    ``captured``: list of ``{group_id, rewards, advantages}`` (captured advantages as
    the trainer received them). ``criterion`` (owner input, no default) is
    ``{"mode": "exact_zero"}`` or ``{"mode": "float32_tolerance", "value": v}`` with
    finite ``v > 0``; ``None`` blocks with ``e07_criterion_unregistered`` while the
    facts (pinned recomputation, maximum magnitude, sign pattern) are still reported.
    ``advantage_config``: the resolved GRPO settings (module docstring).
    """
    check_scope(scope)
    require(type(captured) is list and captured, "captured must be a nonempty list of groups")
    for index, group in enumerate(captured):
        _check_group(group, f"captured[{index}]")
    _unique_ids(captured, "group_id", "captured")
    if criterion is not None:
        require(
            type(criterion) is dict
            and criterion.get("mode") in E07_MODES
            and (
                set(criterion) == {"mode"}
                if criterion["mode"] == "exact_zero"
                else set(criterion) == {"mode", "value"}
                and is_real(criterion["value"])
                and criterion["value"] > 0
            ),
            "criterion must be {'mode': 'exact_zero'} or "
            "{'mode': 'float32_tolerance', 'value': finite > 0}",
        )
    issues = Issues()
    _check_advantage_config(advantage_config, issues)
    if criterion is None:
        issues.add(
            "e07_criterion_unregistered",
            "criterion",
            "The E07 pass criterion is an owner input (exact_zero or float32_tolerance).",
        )
    rows = []
    for group in sorted(captured, key=lambda g: g["group_id"]):
        gid, rewards, advantages = group["group_id"], group["rewards"], group["advantages"]
        path = f"group:{gid}"
        row = {"group_id": gid, "group_size": len(rewards)}
        if not all(is_real(r) for r in rewards) or not all(is_real(a) for a in advantages):
            issues.add("nonfinite_value", path, "Captured reward or advantage is not finite.")
            row.update(evaluated=False)
            rows.append(row)
            continue
        tied = len(set(rewards)) == 1
        oracle = group_advantages_f32(rewards)
        captured_max = max(abs(a) for a in advantages)
        signs = sorted({(a > 0) - (a < 0) for a in advantages})
        matches = all(_same_bits(a, o) for a, o in zip(advantages, oracle))
        row.update(
            evaluated=True,
            rewards_tied=tied,
            oracle_advantages=oracle,
            oracle_exact_zero=all(o == 0.0 for o in oracle),
            captured_max_abs_advantage=captured_max,
            captured_exact_zero=all(a == 0.0 for a in advantages),
            captured_sign_pattern=signs,
            captured_matches_pinned_recomputation=matches,
        )
        if not tied:
            issues.add("rewards_not_tied", path, "An E07 control group must have equal rewards.")
        if not matches:
            issues.add(
                "captured_advantage_mismatch",
                path,
                "Captured advantages differ bitwise (float32) from the pinned recomputation.",
            )
        if tied and not row["oracle_exact_zero"]:
            issues.add(
                "pinned_tied_group_nonzero",
                path,
                "The pinned float32 formula gives a nonzero advantage for this tied group.",
                severity="info",
            )
        if criterion is None:
            row["meets_criterion"] = None
        elif criterion["mode"] == "exact_zero":
            row["meets_criterion"] = row["captured_exact_zero"]
        else:
            row["meets_criterion"] = captured_max <= criterion["value"]
        if row["meets_criterion"] is False:
            issues.add("e07_criterion_not_met", path, "Captured advantages fail the criterion.")
        rows.append(row)
    status = "blocked" if issues.has("error") else "passed"
    return artifact(
        "s4_e07_equal_reward_control",
        module=__name__,
        module_file=__file__,
        scope=scope,
        inputs={
            "captured": captured,
            "criterion": criterion,
            "advantage_config": advantage_config,
        },
        units={"advantage": "dimensionless (float32 values as Python floats)"},
        status=status,
        issues=issues,
        body={
            "ticket": "E07",
            "pinned_source": PINNED_SOURCE,
            "criterion": criterion,
            "advantage_config": advantage_config,
            "groups": rows,
            "note": (
                "A zero advantage is necessary, not sufficient, for the pinned "
                "optimizer skip: the skip needs an exactly-zero reduced gradient norm "
                "(KL off, no auxiliary loss); see teacher_rl.checks.steps."
            ),
        },
    )


def _reproduces_f32_sequence(rewards, order):
    """True when ``order`` leaves the float32 reward VALUES the pinned path sums unchanged.

    Equality is float32 value equality, position by position (``np.array_equal`` on the
    float32 casts): 0.0 equals -0.0, and distinct doubles that round to the same float32
    are equal. Such a permutation (identity, or swaps of members whose float32 rewards
    are equal) sums the same values in the same order, so it cannot test
    summation-order sensitivity.
    """
    reordered = np.array([rewards[i] for i in order]).astype(np.float32)
    return bool(np.array_equal(reordered, np.array(rewards).astype(np.float32)))


def check_advantage_recomputation(
    captured_groups, *, group_size, tolerance, advantage_config, scope
):
    """E06: recompute each captured group's advantages in captured order (L1).

    ``captured_groups``: list of ``{group_id, members: [{member_id, reward,
    advantage}]}`` with members in the order the upstream computed them.
    ``group_size`` (registered n_generation) and ``tolerance`` (``{"mode":
    "bitwise"}`` or ``{"mode": "absolute", "value": v}``) are owner inputs with no
    default; ``None`` blocks while the facts are still computed. A group whose size
    differs from the registration is a rank-local or partial subset and cannot
    redefine the global group (``group_size_mismatch``). ``order_sensitive`` reports
    whether a different member order changes the float32 result (summation order):
    the member-id order is tried, or the reversal when the member-id order reproduces
    the captured float32 reward sequence (equal float32 values at every position, so
    the identity, a swap of 0.0 and -0.0, or a swap of distinct doubles that round to
    one float32 all reproduce it), so a permutation is compared only when it changes
    the summed values; ``True`` is proof, ``False`` means not detected by that
    permutation; a one-member group, or one whose reversal also reproduces the
    captured sequence (e.g. all float32 rewards equal), is ``"not_tested"``.
    ``advantage_config``: the resolved GRPO settings.
    """
    check_scope(scope)
    require(
        type(captured_groups) is list and captured_groups, "captured_groups must be a nonempty list"
    )
    for index, group in enumerate(captured_groups):
        path = f"captured_groups[{index}]"
        require(
            type(group) is dict and set(group) == {"group_id", "members"},
            f"{path} needs group_id, members",
        )
        require(is_text(group["group_id"]), f"{path}.group_id must be text")
        members = group["members"]
        require(type(members) is list and members, f"{path}.members must be a nonempty list")
        for member in members:
            require(
                type(member) is dict
                and set(member) == _MEMBER_KEYS
                and is_text(member["member_id"]),
                f"{path} members need exactly member_id, reward, advantage",
            )
            _check_captured_values([member["reward"], member["advantage"]], f"{path} member")
    _unique_ids(captured_groups, "group_id", "captured_groups")
    require(
        group_size is None or (type(group_size) is int and group_size >= 1),
        "group_size must be a positive int or None",
    )
    if tolerance is not None:
        require(
            type(tolerance) is dict
            and tolerance.get("mode") in TOLERANCE_MODES
            and (
                set(tolerance) == {"mode"}
                if tolerance["mode"] == "bitwise"
                else set(tolerance) == {"mode", "value"}
                and is_real(tolerance["value"])
                and tolerance["value"] >= 0
            ),
            "tolerance must be {'mode': 'bitwise'} or {'mode': 'absolute', 'value': finite >= 0}",
        )
    issues = Issues()
    _check_advantage_config(advantage_config, issues)
    if group_size is None:
        issues.add(
            "group_size_unregistered", "group_size", "Registered group size is an owner input."
        )
    if tolerance is None:
        issues.add(
            "tolerance_unregistered", "tolerance", "Recomputation tolerance is an owner input."
        )
    seen_members = {}
    rows = []
    for group in sorted(captured_groups, key=lambda g: g["group_id"]):
        gid, members = group["group_id"], group["members"]
        path = f"group:{gid}"
        for member in members:
            if member["member_id"] in seen_members:
                issues.add(
                    "duplicate_member",
                    path,
                    f"Member {member['member_id']} also appears in group "
                    f"{seen_members[member['member_id']]}.",
                )
            seen_members.setdefault(member["member_id"], gid)
        rewards = [m["reward"] for m in members]
        advantages = [m["advantage"] for m in members]
        row = {"group_id": gid, "group_size": len(members)}
        if group_size is not None and len(members) != group_size:
            issues.add(
                "group_size_mismatch", path, f"{len(members)} members, registered {group_size}."
            )
        if not all(is_real(v) for v in rewards + advantages):
            issues.add("nonfinite_value", path, "Captured reward or advantage is not finite.")
            row["evaluated"] = False
            rows.append(row)
            continue
        oracle = group_advantages_f32(rewards)
        diffs = [abs(a - o) for a, o in zip(advantages, oracle)]
        bitwise = all(_same_bits(a, o) for a, o in zip(advantages, oracle))
        order = sorted(range(len(members)), key=lambda i: members[i]["member_id"])
        if _reproduces_f32_sequence(rewards, order):
            order = list(range(len(members)))[::-1]
        if len(members) < 2 or _reproduces_f32_sequence(rewards, order):
            order_sensitive = "not_tested"
        else:
            reordered = group_advantages_f32([rewards[i] for i in order])
            order_sensitive = any(
                not _same_bits(reordered[pos], oracle[i]) for pos, i in enumerate(order)
            )
        if tolerance is None:
            within = None
        elif tolerance["mode"] == "bitwise":
            within = bitwise
        else:
            within = max(diffs) <= tolerance["value"]
        if within is False:
            issues.add(
                "advantage_recomputation_mismatch",
                path,
                "Captured advantages exceed the tolerance.",
            )
        row.update(
            evaluated=True,
            bitwise_equal=bitwise,
            max_abs_difference=max(diffs),
            within_tolerance=within,
            order_sensitive=order_sensitive,
        )
        rows.append(row)
    status = "blocked" if issues.has("error") else "passed"
    return artifact(
        "s4_e06_advantage_recomputation",
        module=__name__,
        module_file=__file__,
        scope=scope,
        inputs={
            "captured_groups": captured_groups,
            "group_size": group_size,
            "tolerance": tolerance,
            "advantage_config": advantage_config,
        },
        units={"advantage": "dimensionless", "max_abs_difference": "advantage units"},
        status=status,
        issues=issues,
        body={
            "ticket": "E06",
            "pinned_source": PINNED_SOURCE,
            "tolerance": tolerance,
            "advantage_config": advantage_config,
            "groups": rows,
        },
    )


def _order_violations(rewards32, advantages):
    """Pairs whose advantage order disagrees with the float32 reward order."""
    violations = []
    n = len(rewards32)
    for i in range(n):
        for j in range(i + 1, n):
            r = (rewards32[i] > rewards32[j]) - (rewards32[i] < rewards32[j])
            a = (advantages[i] > advantages[j]) - (advantages[i] < advantages[j])
            if r != a:
                violations.append([i, j])
    return violations


def check_advantage_ordering(groups, *, scope, advantage_fn=None):
    """E08: ordering, reversal and float32 near-tie facts per group (L1).

    ``groups``: list of ``{group_id, rewards, advantages}``; ``advantages`` may be
    ``None`` to check the pinned recomputation only. Advantages must order members
    exactly as their float32-rounded rewards (equal rewards -> equal advantages); a
    sign inversion is ``advantage_order_violation``. Reversal applies the advantage
    function (default: the pinned ``group_advantages_f32``; injectable so a
    negative control can prove detection) to the reversed rewards and requires the
    reversed order. Distinct rewards that collapse in float32 and asymmetric
    magnitudes are reported as facts (``info``), not errors.
    """
    check_scope(scope)
    fn = group_advantages_f32 if advantage_fn is None else advantage_fn
    require(callable(fn), "advantage_fn must be callable")
    require(type(groups) is list and groups, "groups must be a nonempty list")
    for index, group in enumerate(groups):
        path = f"groups[{index}]"
        require(
            type(group) is dict and set(group) == _GROUP_KEYS,
            f"{path} needs exactly {sorted(_GROUP_KEYS)}",
        )
        require(is_text(group["group_id"]), f"{path}.group_id must be text")
        _reward_list(group["rewards"], f"{path}.rewards")
        require(
            group["advantages"] is None
            or (
                type(group["advantages"]) is list
                and len(group["advantages"]) == len(group["rewards"])
            ),
            f"{path}.advantages must be None or aligned with rewards",
        )
        if group["advantages"] is not None:
            _check_captured_values(group["advantages"], f"{path}.advantages")
    _unique_ids(groups, "group_id", "groups")
    issues = Issues()
    rows = []
    for group in sorted(groups, key=lambda g: g["group_id"]):
        gid, rewards = group["group_id"], group["rewards"]
        path = f"group:{gid}"
        rewards32 = float32_values(rewards)
        computed = list(fn(list(rewards)))
        reversed_adv = list(fn(list(reversed(rewards))))
        subject = computed if group["advantages"] is None else group["advantages"]
        if not all(is_real(a) for a in subject + computed + reversed_adv):
            issues.add("nonfinite_value", path, "An advantage is not finite.")
            rows.append({"group_id": gid, "evaluated": False})
            continue
        violations = _order_violations(rewards32, subject)
        reversal_violations = _order_violations(list(reversed(rewards32)), reversed_adv)
        merged = [
            [i, j]
            for i in range(len(rewards))
            for j in range(i + 1, len(rewards))
            if rewards[i] != rewards[j] and rewards32[i] == rewards32[j]
        ]
        asymmetric = len(subject) == 2 and subject[0] != 0.0 and abs(subject[0]) != abs(subject[1])
        if violations:
            issues.add("advantage_order_violation", path, f"Order disagrees on pairs {violations}.")
        if reversal_violations:
            issues.add(
                "reversal_order_violation", path, "Reversed rewards do not reverse the order."
            )
        if merged:
            issues.add(
                "float32_reward_collapse",
                path,
                f"Distinct rewards equal in float32: {merged}.",
                severity="info",
            )
        if asymmetric:
            issues.add(
                "float32_near_tie_asymmetry",
                path,
                "G = 2 advantages differ in magnitude.",
                severity="info",
            )
        rows.append(
            {
                "group_id": gid,
                "evaluated": True,
                "subject": "pinned_recomputation" if group["advantages"] is None else "captured",
                "rewards_float32": rewards32,
                "advantages": subject,
                "order_violations": violations,
                "reversal_violations": reversal_violations,
                "float32_collapsed_pairs": merged,
                "magnitude_asymmetric": asymmetric,
            }
        )
    status = "blocked" if issues.has("error") else "passed"
    return artifact(
        "s4_e08_advantage_ordering",
        module=__name__,
        module_file=__file__,
        scope=scope,
        inputs={"groups": groups},
        units={"advantage": "dimensionless"},
        status=status,
        issues=issues,
        body={
            "ticket": "E08",
            "advantage_function": "pinned_group_advantages_f32"
            if advantage_fn is None
            else "injected",
            "groups": rows,
        },
    )
