# S2 implementation and spend plan

Updated 2026-11-01 under D-021. The D-019 synthetic host check is closed with $0.60 recorded usage; no scientific S2 run or gated-asset acceptance/download has occurred. Detailed contract: `S2_ENGINEERING_SPEC.md`. This runbook implements the current S2 card; optional Slurm/fault-injection/batching/traffic experiments remain gated proposals.

## A. Already executable at zero cloud cost

From the repository root, use the resolved CPU environment:

```sh
uv sync --frozen
uv run --frozen python -B -m pytest -p no:cacheprovider -q
uv run --frozen ruff check gate tests scripts
uv run --frozen python -B -m gate.s2 preflight experiments/S2_PILOT_DRAFT.json
uv run --frozen python -B -m gate.s2 preflight experiments/S2_FULL_DRAFT.json
uv run --frozen python -B -m gate.s2 cost-table
```

Both draft checks intentionally return exit 2 and named blockers. That is the expected result, not a request to insert fictional approvals. `gate.s2 normalize` accepts local summary, identity and observation JSON files; `--help` describes arguments. It neither downloads nor starts a service. Unit tests use invented records derived from public source, with no licensed output.

November 7 local interfaces, with the declared CPU dependencies installed:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m gate.s2 matrix --replicates 3
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m gate.s2 schedule REQUEST.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m gate.s2 ingest REQUEST.json --root DIR
```

`matrix` reports the active 10-job pilot and 2,646-job full schedule at three replicates; `cost-table` remains historical arithmetic. `schedule` requires explicit scene/road/stratum/asset and protocol inputs. `ingest` verifies retained receipts and attempt accounting, reads real Parquet files containing synthetic metrics, and feeds qualified synthetic observations into the existing adapter. The PhysLoop capture format is not an actual AlpaSim ASL decoder. Pilot/full ingestion retains and replays `schedule_request`; only tiny synthetic fixtures use null. All retries and unsupported terminal clipping remain blocked. See [S2_INGESTION](S2_INGESTION.md) for request schemas and exact limits. Ingestion requires PyArrow; frozen-lock setup and complete verification for this addition must be recorded separately, rather than inferred from these usage commands.

```sh
uv run --frozen python -B scripts/verify_s2.py --output evidence/s2-NEW-UNIQUE-ID
```

Verification uses exclusive output creation, runs the test/lint suite, writes draft readiness reports and historical price arithmetic, and hashes the source/drafts it used. Generated evidence is append-only. Tests are software known answers, not Huy's scientific predictions.

## B. Adopted ceilings and actions still requiring approval

| Decision | Recommended prepared choice | Who records it / why pending |
|---|---|---|
| P-005 project tier | **Adopted D-018:** `up_to_2500`, cloud ceiling $2,500 | Concrete funding/rental actions remain pending |
| S2 envelope | **Adopted:** $1,200 all-in, inclusive of pilot | Full-sweep release still needs pilot evidence and forecast |
| Pilot ceiling | **Adopted:** $100 all-in; **proposed phase limits:** ≤8 GPU-hours and ≤$2.50/GPU-hour | Approve concrete host/payment/limits before any charge |
| Pilot protocol | Five scenes × one BF16 unseeded replicate × Harmonizer off/on; batch one, graph off, background traffic replay with policy-controlled ego, fixed analysis | Huy freezes prediction and protocol; agent resolves actual source/host fields |
| Default execution | One compatible full host, Docker Compose, same topology/settings for comparisons | Quote must confirm root/host access, NVIDIA runtime, renderer compatibility, memory and disk |
| Assets | Huy accepts needed terms; internal use only until separate public gates pass | Agent cannot accept licenses; first-download date/expiry are recorded |
| Full study target | Primary score, maximum tolerable degradation, target population/roads/strata, reliability/precision rule and scientific prediction | Huy's research decisions; can be finalized after cost pilot but before full outcomes |

The first compute cap is **GPU-hours, not node-hours**: two GPUs allow at most four billed node-hours, including setup, warm-up and idle time. At the price ceiling that is at most $20 compute; the other $80 is contingency for storage, transfer, taxes and other approved charges. Unused allowance is not a spending target. No automatic extension is implied if setup consumes the allowance.

Protocol choices above remain recommendations, not agent-filled Huy predictions. D-018 adopts P-030: agents own evaluator implementation; Huy retains scientific definitions, thresholds and interpretation. `AGENTS.md` and `CLAUDE.md` have been updated together; the protected manifest is unchanged. The exact rental/license request is `S2_RENTAL_APPROVAL.md`.

## C. Concrete cost basis

The actual rental packet in `S2_RENTAL_APPROVAL.md` now includes a tested Vast KVM host and an untested Nebius fallback; re-price and approve the specific next rental. Neither provider is automatically selected for licensed assets or full runtime fit. Ordinary Runpod Pods do not support the adopted Compose topology. The following historical price arithmetic is retained for comparison.

The dated [price audit](research/2026-10-31-s2-models-budget.md) found public RTX PRO 6000 rates of $1.80–2.49 per GPU-hour; edition/region/full-host availability are not a booking. Plan v5 defers the full TransFuser/FP8 comparisons and retains two active full sweeps. Scaling the older per-sweep scenario gives approximately 68–170 GPU-hours, before pilots/profiling and all other costs:

| Two-sweep GPU-hours (rounded scenario) | $1.80/GPU-hour | $2.49/GPU-hour |
|---:|---:|---:|
| 68 | $122.40 | $169.32 |
| 170 | $306.00 | $423.30 |

This is a historical-throughput scenario, **not measured S2 demand**. Around 2,900 GB on Runpod's checked network-volume tiers is $165/month; temporary copies and retained storage add costs. Different provider storage products are not interchangeable. Keep a retention date and a provisioned-volume ledger, including charges after GPU teardown.

Illustrative $1,200 envelope: $100 pilot, up to $850 subsequent compute, $200 subsequent storage and $50 subsequent transfer/tax/other. Reallocate only inside a recorded revised forecast. If measured cost plus contingency exceeds the available balance, stop before full sweeps and revise scope/funding explicitly. A claimed $1,200 all-in cap must include tax; do not treat tax as an unbounded extra.

The $2,500 initial tier leaves at most $1,300 after a fully spent S2. It does not finance an unspecified frontier RL/distillation program. At the audited eight-H100 Lambda price, $31.92/node-hour makes 24 hours $766.08 compute-only. S4 requires a separately approved bounded correctness/measurement pilot, followed by a study forecast that includes additional confirmation training after development selection where required. S5 is deferred. Any larger tier requires a new explicit decision; current project cash remains capped at $2,500. Do not rent an eight-GPU node merely because a recipe lists one.

## D. Execution sequence after decisions

1. **Recheck dated gates.** Confirm the challenge has actually closed and the planned S2 start (2026-11-02) is reached; inspect the latest release and offered upstream work. Record evidence/time, not just today's date. If settings remain withheld, label results provisional.
2. **Verify quote before provisioning.** Confirm GPU edition/VRAM, required GPU count, renderer support, container privileges, disk/storage arrangement, network cost, minimum billing, availability and shutdown interface. Generic container-only Pods may not permit the needed Docker/DCGM host access. Select a suitable VM/full host or reject the quote. No credential or license acceptance is automated by this repository.
3. **Set financial controls first.** Record remaining balance, provider timeout, shutdown owner and retention/deletion date. Reserve tax/storage liability. Record the exact priced host and cap. Stop at the earliest compute-hour, total-dollar or expiry limit.
4. **Bootstrap inside the pilot cap.** Resolve runtime locks and images; save hashes/configs/GPU information. Obtain the minimal licensed subset and required assets after Huy's acceptance. Verify checksum and source metadata. If the packaging requires a larger corpus than the cap permits, stop; do not silently download the full release.
5. **Run compatibility and data-admission checks.** Exercise the pinned scorer's stationary-tail case and source-defined boundaries; verify map metrics, timestamps, recording clipping and `_complete`/retry interpretation. Populate actual-file extractor fixtures, run the adapter end to end on one real rollout, and reconcile normalized output to raw logs. A failed check blocks the scientific pilot.
6. **Run the registered Harmonizer cost pilot.** Capture all ten scheduled outcomes and every attempt. Record renderer/driver timing, raw validity, frames, paid wall time, cost and OOM/NaN retries. Investigate any performance beyond the written prediction or Amdahl bound. Do not choose a mode by its higher five-scene score.
7. **Freeze the full protocol.** Exclude pilot observations from full analysis; do not choose full scenes, strata or margins from pilot policy scores. Huy records Harmonizer, inference target, tolerances, strata, reliability/stabilization rules and predictions. Agent prepares the measured forecast, resolves all missing manifest fields, and requests release of only the remaining approved stage cap. Preflight must pass and references must be inspected.
8. **Qualify controls and capture throughput.** Full FP8 comparison is deferred by D-021. If separately reinstated, its P-009 E0a–E1 ladder and paired E2 verdict remain mandatory; no precision change enters the active teacher/re-export controls silently. Keep the unseeded sweep and deterministic diagnostic protocols distinct. Execute same-input/renderer probes, the required deterministic negative control once its separate port is qualified, re-export control, and graph check if enabling it. Stabilize for the prescribed 30–45 minute window and Huy's additional criterion; include the paid warm-up in cost. GPU timing is after warm-up with synchronization; pipeline throughput also uses complete wall-clock timestamps.
9. **Run the two registered full sweeps.** Preserve complete planned road/stratum coverage, identical comparison settings and ≥3 replicates. Report all attempt IDs; choose a retry result only by a frozen outcome-independent rule. The current adapter blocks unresolved retries. Never pick the best score or silently replace a hard scene with an easier one.
10. **Build the instrument report.** Reconcile every planned job and paid GPU-second. Produce reliability CIs and paired uncertainty, stratum sizes, validity/deviation dose response, contamination joins, noise decomposition, profile/goodput/energy, both score conventions and G3 power reforecast. Absent telemetry is unavailable, not zero. Huy supplies the scientific diagnosis and prediction reconciliation.

The source audit includes documented upstream launch skeletons. Resolve and validate those against the selected host after admission; this repository intentionally does not ship a guessed, auto-renting launcher. The separate D-019 host evidence verifies specific root/CUDA/Compose capabilities on that delivered host. It does not verify licensed formats, real renderer behavior or model fit.

## E. Data contracts for the remaining implementation

Create one immutable directory per run with `planned-manifest.json`, `realized-manifest.json`, exact config/lock, schedule, attempt ledger, original summary/metric paths and their hashes, normalized records, profiler/telemetry exports, cost ledger and analysis outputs. Preserve the earliest preregistration; amendments create new records. Actual step counts, retries, measured GPU-hours and prices paid belong in realized evidence, never backfilled as predictions.

The attempt ledger needs logical job ID, attempt ID, launch/finish times, reason for retry, source filenames/checksums, parent attempt, requested/expected/actual horizon and selected-for-analysis reason. A recorder must preserve failed attempts before rerun. A trusted probe must establish offroad/map coverage and full, pre-truncation deviation. A path string or `required_metrics_verified=true` alone is not proof.

The telemetry collector should reuse AlpaSim Prometheus, host DCGM and Nsight/NVTX. Capture XID, power/energy, clocks, temperatures and PCIe replays alongside available per-service metrics. Do not invent exported queue depth or retry counters; compute retries from the attempt ledger. Keep raw telemetry so agents can implement Huy's frozen stabilization/goodput and variance definitions under D-018, with independent verification and Huy's scientific interpretation.

## F. Exit criteria

Free preparation ends with passing software tests, an independently reviewed source audit, populated draft technical choices and explicit remaining owner/host fields. It does **not** end S2.

S2 ends only when every mandatory acceptance item in the adopted stage card is supported by actual evidence and Huy's frozen definitions, with no unresolved admission failure. S4 is the next active research stage, through separately priced engineering admission and then scientific-study admission under plan v5. S3 and KC-E are deferred rather than prerequisites for the teacher study; an unreleased model or open PR never becomes an available artifact by assumption.

**Tóm tắt quyết định tiền:** Huy đã chấp thuận các trần $100 pilot, $1,200 S2 và $2,500 cloud tại D-018; lệnh nạp tiền/thuê máy và license cụ thể vẫn chờ duyệt. Chạy pilot xong mới mở phần còn lại bằng dự báo đo thật. Huấn luyện RL/distillation có ngân sách riêng dựa trên thử nghiệm đo throughput, không mặc định nằm gọn trong số tiền này.
