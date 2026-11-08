# S2 preparation and host-check handoff — updated 2026-11-01

**Plan v5 scope (D-021):** S2 qualifies the teacher/re-export/same-policy controls for the S4 expert-RL study; full TransFuser/FP8 comparisons and later student/device stages are deferred. The five-scene pilot, full 441-scene protocol, ≥3 unseeded full replicates, nested caps and pending scientific/license/rental approvals remain. The closed D-019 rental does not authorize another rental.

**Completed:** S2's free agent-owned preparation and the bounded public-software/synthetic Vast host checks authorized by D-019. **Not completed:** actual AlpaSim instrument qualification, the licensed-data pilot or any model experiment. The original 2026-10-31 handoff's $0/no-rental statement is historical; [the host-validation report](research/2026-11-01-vast-host-validation.md) owns actual cost and cleanup evidence. Public software, including CUDA, is subject to its applicable terms; no gated-asset acceptance/download occurred in the host check.

## What to read

1. [Engineering specification](S2_ENGINEERING_SPEC.md): interfaces, invariants, source pins, scientific assumptions and acceptance obligations.
2. [Implementation/budget plan](S2_IMPLEMENTATION_PLAN.md): exact work packages, pilot/full protocol, adopted nested cash ceilings and execution order.
3. [Research operating model](RESEARCH_OPERATING_MODEL.md): Huy's model-research focus, adopted evaluator delegation, data separation and teach-back.
4. [Upstream audit](research/2026-10-31-s2-upstream.md) and [models/prices audit](research/2026-10-31-s2-models-budget.md): dated primary-source evidence.
5. [Actual Vast host validation](research/2026-11-01-vast-host-validation.md): delivered resources, synthetic controls, profiler/counter evidence, remaining limitations and rental lifecycle. The earlier [VM-fit note](research/2026-11-01-vast-vm-fit.md) remains a dated public-source snapshot.

## Implemented and verified

- Mechanical schema-v2 preflight, including phase-specific compute/rate/all-in caps, frozen predictions and post-pilot full-sweep release; no launch/publication authorization.
- Exact-source AlpaSim summary adapter with expected-job mapping, provenance checks, quarantine and complete-batch admission; no scientific validity estimator is substituted for trusted raw observations.
- Local CLI for preflight, normalization and historical cost arithmetic.
- 16 tests executing the exact pre-fix/fixed public geometry helpers: original cases give 5 m before the fix and <0.00001 m after it. Full evaluator/runtime remains unverified.
- Expanded source snapshots retain supporting experiment/fixture files and their provenance/licenses.

The 2026-10-31 preparation snapshot records **723 passing tests** (498 S1-era tests plus 225 additions), with Ruff and whitespace checks passing; this is not a count of subsequent host-check tests. Protected S1 inference files match the previously qualified source hashes. The locked CPU environment includes Shapely 2.1.2; it is not the AlpaSim GPU environment.

Primary software evidence: [`verification.json`](../evidence/s2-20261031-offline-preparation-v2/verification.json), [`tests.log`](../evidence/s2-20261031-offline-preparation-v2/tests.log), source archive and checksums in the same directory. Independent locked-checkout reproduction: [`verification.json`](../evidence/s2-20261031-snapshot-replay-v2/verification.json), with **723 passing tests** again.

The archive must be overlaid on the recorded Git base, `cfce036f87be71fcf9c729fd2b46140ee6394621`: tests inspect Git metadata and the protected manifest supplied by that checkout. An earlier archive-only probe failed those two infrastructure checks (721 passed); its evidence remains in `s2-20261031-snapshot-replay-v1`. No test was skipped to obtain the subsequent pass. The first preparation bundle is also retained; v2 is the final preparation bundle.

Reproduce from the repository checkout:

```sh
uv sync --frozen
uv run --frozen python -B -m pytest -p no:cacheprovider -q
uv run --frozen python -B scripts/verify_s2.py --output evidence/s2-NEW-UNIQUE-ID
```

## Measured host capabilities, 2026-11-01

On Vast instance `58628722`, the delivered KVM Ubuntu 22.04 guest provided root and an RTX PRO 6000 Blackwell Workstation Edition with 97,887 MiB reported VRAM and driver 580.95.05. Guest RAM was 96.996 GiB, versus the offer's 123.438 GiB; use delivered resources for sizing.

- CUDA integer results matched the CPU oracle and repeated bitwise, while deliberate corruption was detected. The VRAM test verified approximately 70.82 GiB in a GPU pass plus 1,024 CPU samples; it does not cover every VRAM byte or establish long-duration reliability.
- Actual buildx and Compose execution verified DNS/network, shared bind mounts, guest-host PID namespace and GPU reservations.
- Nsight Compute read actual kernel counters; Nsight Systems captured three CUDA kernels. This establishes those measurement paths, not an AlpaSim timing profile.
- DCGM 4.7.1 tensor-active changed from idle zero to approximately 0.576 during the exact-answer cuBLAS control and returned to zero. Separately, exporter 4.8.4 with its bundled DCGM 4.6.1 embedded engine served responsive loopback-only scrapes. Keep those version pairs and records separate.
- DCGM XID was N/A and absent from exporter samples. NVML returned supported-event mask `0xf19c` and successful registration, then a short wait timed out. Subscription availability is verified; error-event delivery and alerting remain unverified. A timeout is not a zero-error observation.

Evidence is retained under [`s2-vast-host-20261101-v1`](../evidence/s2-vast-host-20261101-v1/); the [report](research/2026-11-01-vast-host-validation.md) links exact checks and limitations. This qualifies tested capabilities on one delivered host. Vast is a tested host route; Nebius remains an untested fallback. Neither has been adopted for licensed data, and simultaneous teacher/renderer/Harmonizer memory fit remains unmeasured.

## Decisions, separated from technical fields

The pre-adoption v2 evidence snapshot reported 31 pilot blockers and 38 full-sweep blockers. After D-018 and the prepared rental fields, the current drafts report 24 and 37 respectively. These counts are **not a list of questions for Huy**. Most are runtime pins, host facts and licensed metadata that agents obtain after the relevant admission.

D-018 records P-005 and adopts P-030. D-019 already authorized the bounded Vast host rental/test/cleanup; no repeat approval is needed within that scope. Before the scientific pilot, the remaining concrete action package is:

1. P-005 tier/caps are **adopted: $2,500 project / $1,200 S2 / $100 pilot**, nested rather than additive. D-019 uses at most $10 within the pilot cap for its host check. The scientific pilot's proposed ≤8 GPU-hours / ≤$2.50 hourly envelope, specific new order and any new funding still await approval; see `S2_RENTAL_APPROVAL.md`.
2. Pilot protocol and written prediction: five scenes × one unseeded BF16 replicate × Harmonizer off/on, graph off, batch one, background traffic replay with policy-controlled ego, fixed analysis; prediction and stop rule written before results.
3. Asset license acceptance for the internal experiment. Public numbers and derived weights/labels remain separate KC-L/KC-L2 gates.

Agents must then verify actual challenge closure/S2 start, release status, the selected pilot host's current quote and access permissions, timeout/retention, full runtime identities and source metadata. The measured host facts above are scoped evidence, not a blanket host-compatibility attestation. A filled manifest or quoted public rate is not proof of the remaining facts.

Before the full sweeps, Huy freezes the measured-cost-informed Harmonizer choice, road population/strata, contamination/validity conventions, margin, reliability/precision rule, stabilization criterion and scientific predictions, then explicitly releases the remaining stage cap against the pilot forecast. Pilot observations and score-driven redesign do not enter confirmation unnoticed.

P-030 is now adopted by D-018: evaluator-core implementation and plumbing are agent-owned; scientific decisions and model-training math remain Huy's. Both contracts are updated. Listed file locks remain until Huy unlocks them; no protected-file unlock has occurred. The software evidence above is the dated pre-adoption snapshot, not a claim that later documentation has identical hashes.

## Remaining real-environment obligations

Actual raw-log extraction and expected/clipped-horizon semantics; source split/road mapping; complete evaluator/scorer integration; renderer/policy controls and deterministic diagnostic port; simultaneous runtime memory fit; reliability/variance/validity statistics under Huy's definitions; FP8 correctness ladder; full-pipeline profiling/cost and four real sweeps. XID delivery/alerting needs qualification or an explicit acceptance amendment. These are S2 execution work; synthetic GPU success does not complete them.

**Tóm tắt:** chuẩn bị miễn phí đã xong; một Vast KVM host đã chạy kiểm tra GPU/deployment/profiling tổng hợp theo D-019. Máy đo AlpaSim, quyền dùng dữ liệu trên host và scientific pilot vẫn chưa được admission; không xem kết quả host là full-pipeline proof. Duyệt pilot nhỏ, đo chi phí và hoàn thiện protocol rồi mới mở full sweep. P-030/D-018 giao evaluator engineering cho agent, đồng thời giữ câu hỏi, tiêu chí và diễn giải khoa học ở Huy.
