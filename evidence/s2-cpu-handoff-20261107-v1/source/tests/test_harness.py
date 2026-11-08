"""Plumbing tests. Unlike the known answers, these pass before any ★ core exists."""

from pathlib import Path

import pytest
from cores import CORES, load_core

import gate

ROOT = Path(__file__).resolve().parent.parent


def test_package_imports():
    assert gate.__doc__


def test_every_core_lives_in_a_starred_path():
    lines = (ROOT / ".claude" / "starred-cores.txt").read_text().splitlines()
    starred = {line.strip() for line in lines if line.strip() and not line.startswith("#")}
    for target in CORES.values():
        path = target.split(":")[0].replace(".", "/")
        assert {f"{path}.py", f"{path}/"} & starred, f"{target} is not guarded"


def test_missing_core_fails_and_names_it(monkeypatch):
    monkeypatch.setitem(CORES, "absent", "gate.absent:nothing")
    with pytest.raises(pytest.fail.Exception, match="not written yet: gate.absent:nothing"):
        load_core("absent")
