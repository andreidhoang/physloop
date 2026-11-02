"""The one place the tests name a ★ core.

The names and signatures are proposed. Reshape them here, or in the wrappers in conftest.py, when
the cores exist; the tests stay put.
"""

import importlib

import pytest

CORES = {
    "exact_mcnemar_p": "gate.stats:exact_mcnemar_p",  # (b, c, sided) -> p
    "exact_mcnemar_power": "gate.stats:exact_mcnemar_power",  # (delta, psi, n, alpha) -> power
    "exact_mcnemar_mde": "gate.stats:exact_mcnemar_mde",  # (psi, n, alpha, power) -> delta
    "ni_margin": "gate.stats:ni_margin",  # (psi, n, alpha, power) -> margin at true delta = 0
    "connor_n": "gate.stats:connor_n",  # (delta, psi, alpha, power, sided) -> scenes
    "continuous_mde": "gate.stats:continuous_mde",  # (sigma_d, n, alpha, power) -> delta
    "zero_failure_upper": "gate.stats:zero_failure_upper",  # (n, tail) -> upper bound
    "overall_go_power": "gate.stats:overall_go_power",  # (stratum_powers) -> power
    "per_stratum_power": "gate.stats:per_stratum_power",  # (overall, k) -> power
    "equal_precision_scenes": "gate.stats:equal_precision_scenes",  # (m, rho) -> scene multiple
    "obrien_fleming": "gate.stats:obrien_fleming",  # (K, alpha) -> z boundary per look
    "pocock": "gate.stats:pocock",  # (K, alpha) -> z boundary per look
    "verdict": "gate.verdict:verdict",  # (rows, margin, alpha) -> .overall, .strata
}


def load_core(name):
    module, attr = CORES[name].split(":")
    try:
        return getattr(importlib.import_module(module), attr)
    except ModuleNotFoundError as exc:
        if not (exc.name == module or module.startswith(exc.name + ".")):
            raise
        pytest.fail(f"★ core not written yet: {CORES[name]} ({exc})", pytrace=False)
    except AttributeError as exc:
        pytest.fail(f"★ core not written yet: {CORES[name]} ({exc})", pytrace=False)
