import pytest
from cores import load_core


@pytest.fixture
def core():
    return load_core


@pytest.fixture
def run_verdict():
    """Returns (overall, {stratum: state}) as plain strings, whatever type verdict() returns."""

    def plain(state):
        return str(getattr(state, "value", state))

    def run(rows, margin, alpha=0.05):
        result = load_core("verdict")(rows, margin=margin, alpha=alpha)
        return plain(result.overall), {name: plain(s) for name, s in result.strata.items()}

    return run
