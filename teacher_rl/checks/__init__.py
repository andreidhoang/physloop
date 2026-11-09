"""S4 learning-signal, semantics and audit checks (build-spec A9; D-024).

Serves S4.1-S4.4/S4.6 [C] (G9, G7) as engineering preparation. Agent-authored
evaluator engineering under D-018: these checks *verify the shipped upstream behaviour*
(Cosmos-RL GRPO advantage, the AlpaGym replay trainer, AlpaSim metric aggregation)
against captured evidence. They never propose, select or tune a reward, loss,
estimator, tolerance or pass criterion; every such value is a required owner (Huy)
input with no default, and its absence yields ``blocked`` naming the field while the
facts are still computed.

Submodules (tickets from ``protocol-design.md`` section 4):

- ``signal``    - E06 recomputation, E07 equal-reward control, E08 ordering, E19
  float32 group advantage.
- ``steps``     - E20 optimizer-step accounting, E07 skip semantics, pinned Cosmos
  ``warmup_stable_decay`` learning-rate lambda.
- ``semantics`` - E10-E14 dense metrics vs online aggregate vs offline modifiers.
- ``audit``     - E05 version fence, E16 train/gate adapter audit, E17 uncertainty
  axes, E21 KL status and checkpoint-retention simulation.
- ``replay``    - E04 old/new log-density pairing (with the G3-06 trainer weight-version
  cross-check), E01 two-run repeatability.

This package module holds only the shared artifact envelope (build-spec section 2).
Nothing here imports torch, reads files other than module sources (for producer
identity), performs network or subprocess calls, or imports protected inference.
A green CPU result is never ``HOST_QUALIFIED``, admission or rental readiness.
"""

import hashlib
import math
from pathlib import Path

from gate.s2_identity import identity_hash

SCHEMA_VERSION = 1
SCOPES = ("synthetic", "public_source_fixture", "actual")
STATUSES = (
    "passed",
    "blocked",
    "not_run",
    "not_estimable",
    "unverified",
    "not_applicable",
    "qualify_on_host",
)
SEVERITIES = ("error", "warning", "info")


def source_sha256(module_file):
    """Return the SHA-256 of a module's own source bytes (producer identity)."""
    return hashlib.sha256(Path(module_file).read_bytes()).hexdigest()


def require(condition, message):
    """Raise ``ValueError`` (fail closed) when a structural precondition fails."""
    if not condition:
        raise ValueError(message)


def is_text(value):
    return type(value) is str and bool(value.strip())


def is_count(value):
    return type(value) is int and value >= 0


def is_real(value):
    """True for a finite int/float (booleans excluded)."""
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def check_scope(scope):
    require(scope in SCOPES, f"scope must be one of {SCOPES} (required, no default)")
    return scope


class Issues:
    """Collector for build-spec section 2 issues ``{code, path, severity, ...}``."""

    def __init__(self):
        self.items = []

    def add(self, code, path, message, severity="error", refs=()):
        require(severity in SEVERITIES, f"unknown severity {severity!r}")
        self.items.append(
            {
                "code": code,
                "path": path,
                "severity": severity,
                "evidence_refs": sorted(refs),
                "message": message,
            }
        )

    def has(self, *severities):
        return any(item["severity"] in severities for item in self.items)

    def codes(self):
        return sorted({item["code"] for item in self.items})

    def sorted(self):
        return sorted(self.items, key=lambda i: (i["path"], i["code"], i["message"]))


def artifact(kind, *, module, module_file, scope, inputs, units, status, issues, body):
    """Return the shared JSON artifact envelope with ``body`` merged in.

    ``inputs`` maps an input name to a JSON-native value (or ``None``); each is
    bound by its canonical identity hash. ``status`` must be one of ``STATUSES``.
    """
    require(status in STATUSES, f"unknown status {status!r}")
    check_scope(scope)
    envelope = {
        "artifact_kind": kind,
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "producer": {"module": module, "source_sha256": source_sha256(module_file)},
        "input_hashes": {
            name: None if value is None else identity_hash(value)
            for name, value in sorted(inputs.items())
        },
        "units_of_measure": units,
        "status": status,
        "issues": issues.sorted(),
    }
    overlap = set(envelope) & set(body)
    require(not overlap, f"artifact body reuses envelope keys {sorted(overlap)}")
    return {**envelope, **body}
