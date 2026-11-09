"""Torch-dependent S4 harness tests (D-024).

CPU PyTorch is installed only with the non-default ``s4`` dependency group. Without it
these tests are skipped locally; with ``PHYSLOOP_REQUIRE_TORCH=1`` a missing torch fails
collection instead, so a verification run can never report a pass by skipping them.
"""

import os

import pytest

if os.environ.get("PHYSLOOP_REQUIRE_TORCH") == "1":
    import torch  # noqa: F401  (hard requirement in verification runs)
else:
    pytest.importorskip("torch", reason="install the s4 dependency group: uv sync --group s4")
