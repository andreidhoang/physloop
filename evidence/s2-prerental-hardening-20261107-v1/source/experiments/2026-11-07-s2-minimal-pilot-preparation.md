# Minimal S2 pilot preparation — engineering acceptance

Agent-authored, 2026-11-07. Huy authorized execution of the reduced CPU scope after reviewing the fourteen-package backlog. Serves S2.1/S2.2/S2.3/S2.7 [C], G1/G2/G3·R1/G7 under D-017/D-018. No scientific prediction is supplied or experiment admitted. Cost: $0 cloud; no model, GPU, gated asset or provider operation.

## Scope and deterministic expected outcomes

- Reuse incoming W0–W3 ingestion, with five invented scenes × off/on × one replicate: exactly ten jobs and completed attempts, two batches, ten scores of 0.8 and ten retained deviations of 4.2 m. This is the synthetic capture contract, not AlpaSim ASL or scorer replay.
- Removing the final job's required artifacts rejects the whole batch. Changed export bytes fail hash verification. Existing output is never overwritten.
- Reuse the exact-ID watchdog with fake clock/provider responses; durable arming precedes its stop, and structured stopped state is verified. No provisioning/destroy framework is added.
- USD usage/liability/reserve checks reject unknown inputs, exhaustion of any nested scope and a next unit that reaches the original cutoff before cleanup. No affordability result supplies action authority.
- Monitor scripts require a staged immutable image digest; exporter refuses host-engine conflicts without stopping services. CPU fake-command controls cover these boundaries.
- Both the scientific pilot draft and new technical setup draft remain unadmitted. Empty real scene lists, missing runtime pins/launch arguments and owner/action fields remain explicit.
- Rehearsal suite runs from a frozen source copy and records direct installed dependency versus lock identity separately. A stale lock blocks CPU handoff readiness even if current-environment tests pass.

## Test-first record

Root pilot-helper tests first failed at collection because `gate.s2_pilot` did not exist, then 20 passed. Rehearsal tests first failed at collection because `scripts.rehearse_s2_pilot` did not exist, then two passed. The host-fix author recorded 16 failing and four passing controls before implementation, then 22 passing tests including prior profiler-environment tests. Ten-job fixture tests likewise began with the missing module and then four passed. Full final results belong in the generated verification bundle.

## Environment limitation

Incoming ingestion added PyArrow to pyproject.toml while the existing uv.lock omitted it. Installed PyArrow was 26.0.0. Available uv 0.8.17 failed before resolution: its normal cache path was inaccessible, and writable-cache online/offline invocations panicked in macOS system-configuration; shell PyPI DNS also failed. No lock entries/hashes were fabricated. Record whether this remains unresolved in the final generated report; current-interpreter tests are not a successful frozen rebuild.

## Reproduction

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B scripts/rehearse_s2_pilot.py \
  --output evidence/s2-pilot-cpu-NEW-UNIQUE-ID
```

Expected: exact fixture controls and source-snapshot tests pass; exit 2 if a lock mismatch persists. Review `verification.json`, `demo/demo.json`, test/lint logs and source/checksum manifests. No rental, publication or commit follows from this check.

## Recorded outcome

`evidence/s2-pilot-cpu-20261107-v1/` contains the frozen source, file manifests, exact commands and captured outputs. The complete copied-source suite passed **1,046 tests in 35.35 s**, Ruff passed, and the ten-job demo passed. Snapshot bytes and protected S1 sources remained unchanged. Root whitespace checks passed separately.

The command correctly exited **2**: `cpu_checks_passed=true`, `cpu_handoff_ready=false`, `runtime_qualified=false`. The unresolved CPU blocker is the PyArrow declaration/lock mismatch; a clean frozen sync is also unverified. Actual scenes/launcher/runtime assets, owner scientific/action records and AlpaSim/host qualification remain separate pending work, not failed synthetic checks.

Independent review corrections incorporated before the frozen run: no implicit Docker pulls; cleanup bound to exact recorded container ID/name/ownership label; unknown service state blocks; exact exported file-set checking and destination symlink rejection; specific missing-artifact issue asserted; money overflow blocks; project/dev requirement graphs and Python constraints checked in addition to installed versions. No protected edits, rental, funding, gated download, publication or commit occurred.
