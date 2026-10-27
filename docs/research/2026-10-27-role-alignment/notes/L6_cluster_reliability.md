# L6 — Cluster and reliability evidence inside PhysLoop's existing stages (JR2010863 D4, Q5, X1, scale half of D1)

**As of 2026-10-27.** Research only: nothing was rented, signed into, downloaded to disk, or posted. Every source below was read on 2026-10-27 unless another date is given.

**How this file is laid out.** The assignment asks for six parts, so they are the top-level sections. The researcher template's blocks (Takeaway / Cited Findings / Inferences / Gaps) sit inside part 2, one block per key question (Q1–Q6).

**Ground rules applied to every row.**
- ★ = Huy's work (instruments, predictions, tolerances, diagnoses). Prediction slots are written `____`, and kill thresholds Huy has to set are written `____ (Huy)`. No values are proposed for either.
- Hours are **Huy's focused hours**, with agent hours shown separately. Dollar figures carry the price and its date.
- Each row changes a stage card, so adopting it means a `docs/DECISIONS.md` proposal (the scope-creep rule in CLAUDE.md). This file is the evidence for that proposal, not the decision.

**Price caveat.** On 2026-10-27 at 08:29 UTC, vast.ai's public `bundles` endpoint returned **HTTP 403** on three host/method variants. A fresh pull would have needed a sign-in, which is out of scope. vast.ai prices therefore come from the plan's 2026-10-25 snapshot files (`docs/research/2026-10-25-teacher-to-thor/notes/vast_2026-10-25/`), re-analysed for this question. Lambda, RunPod and Nebius were re-read on 2026-10-27.

---

## 1. Findings table

Price key (per GPU-hour unless noted):
- **V25** = vast.ai snapshot, 2026-10-25, on-demand.
- **N27** = Nebius prices page, read 2026-10-27; the page carries no date.
- **L27** = Lambda pricing page, read 2026-10-27.
- **R08** = RunPod pricing page, which says "Updated October 8, 2026".

Node rates used:
- 8× RTX PRO 6000 VM: $18.78/h (V25, the only 8× VM offer, $2.35/GPU) or $14.40/h (N27, $1.80/GPU; an 8-GPU preset is unverified).
- 8× H100 SXM: $15.46/h (V25 cheapest, Docker) to $31.92/h (L27).

| ID | JR2010863 line(s) | Stage · deliverable | Kind | What changes in the stage card | GPU-h and $ delta (price, date) | Hours delta | Gate | Owner | Sources |
|---|---|---|---|---|---|---|---|---|---|
| **L6-1** Run S2 through AlpaSim's own Slurm path on one node | Q5, D4, D2 | S2 · G1, G7 | **a** if a Slurm-capable VM ≤ $1.60/GPU is available at S2's start; otherwise **c** under P-005 | **Build:** every S2 sweep runs via `wizard.run_method=SLURM_ENROOT`, on one rented VM with single-node Slurm (26.05.x) + Enroot 4.2.1. Sweeps are submitted as Slurm array jobs through AlpaSim's `submit.sh` (`--requeue`, `--signal=B:USR1@300`, autoresume on restart). All S2 sweeps use the same run method and node type, so the run method is not confounded with policy. **Manifest +=** Slurm/Enroot versions, job and array IDs, an `sacct` export (State, ExitCode, Elapsed, restarts), node type (VM), driver/CUDA, and the `resume.sh`/`reeval.sh` AlpaSim writes. **★:** none new. **Acceptance +=** every sweep has its `sacct` record and resume/reeval scripts under `evidence/`. **Prediction:** none (no new measured claim). **Kill:** if Slurm + Enroot will not start a 1-GPU smoke run within `____ (Huy)` hours on the VM, revert to docker-compose and record the failure mode as a negative result. **Cost:** see the next column | GPU-h: **0** (same sweeps). $: **$0** if a Slurm-capable VM ≤ $1.60/GPU is available at S2's start (V25 had 1–2× RTX PRO 6000 Server VMs at $1.24). That variant is not free in time: 136–340 GPU-h on 2 GPUs is 68–170 h of continuous running (≈3–7 days of S2's 14-day window), on an unverified host with reliability 0.93; **+$68–109** at N27 $1.80 (S2 becomes $245–612 vs the plan's $136–544); **+$184–255** at V25 $2.35 ($320–799). Plus about $15–21 of bootstrap time (≈3 h on a 1-GPU VM at $1.24–2.35, then ≈0.75 h on the 8-GPU VM at $14.40–18.78/h) | Huy +2–3; agent +6–8 (bootstrap script for Slurm, slurmdbd, munge, Enroot and FUSE; sqsh image builds; submit wrapper; manifest fields) | P-005 if the price is > $1.60/GPU | agent (plumbing); Huy reviews | AS1–AS6, AS8, PR1, PR3, NV3 |
| **L6-2** Measure S2 throughput the way NVIDIA's sim team does | D4, D2, Q5 | S2 · G7 (the plan's "first measured throughput"), G1 | **a** | **Build:** (i) follow AlpaSim's shipped `optimize-slurm-topology` loop and its Markdown + JSON experiment-record template for the Alpamayo 1.5 sweeps. The primary metric is the stabilized `alpasim:simulation_seconds_per_rollout:avg`, taken only after the 30–45 min stabilization the skill prescribes. The bottleneck signal is `alpasim:rpc_queue_depth_at_start_latest:max/min`. (ii) **One variable:** `wizard.enable_mps` on vs off, on the `8gpu_alpamayo1_5_eval` topology (2 driver replicas per GPU). MPS works only on Slurm run methods. (iii) **Invariance check:** re-run a 50-scene × 1-seed subset under the other arm, and under docker-compose vs SLURM_ENROOT, then compare with the gate's own paired equivalence test (TOST). **★:** the numeric stabilization criterion, the invariance margin, and the bottleneck diagnosis behind each topology decision. **Predictions:** stabilized seconds per rollout, MPS-on ÷ MPS-off: `____`; bottleneck service at the best topology: `____`; steady-state GPU-h per 1,000 Alpamayo 1.5 BF16 rollouts: `____`. **Kill:** if MPS or the run method breaks invariance beyond Huy's margin, run verdict sweeps with that setting off and report the breakage. **Cost:** see the next column | Topology candidates produce real sweep rollouts, so the only overhead is startup plus congestion: ≈4 candidates × ≈20 min = 1.3 node-h, which is **+$19–25** (N27 $14.40/h – V25 $18.78/h). The invariance subset is 50 × 200 steps × 0.463–1.15 s = 1.3–3.2 GPU-h single-stream, **+$2–8**. Likely absorbed, because the plan's 34–85 GPU-h per sweep is a single-stream upper bound (02 §5) | Huy +3–4; agent +3–4 | none (P-005 only if S2 leaves its band) | ★ Huy (criterion, margin, diagnosis); agent runs candidates and fills the record | AS6, AS9, AS12, AS13 |
| **L6-3** Monitoring: DCGM + AlpaSim + goodput dashboard | D4, X1 (stand-in) | S2 · G7, G1 (and S4 through the merged telemetry) | **a** | **Build:** (i) add `DCGM_FI_DEV_XID_ERRORS`, `POWER_USAGE`, `TOTAL_ENERGY_CONSUMPTION`, `SM_CLOCK`, `GPU_TEMP` and `PCIE_REPLAY_COUNTER` (all in dcgm-exporter's default CSV) to AlpaSim's `dcgm-counters.csv`, a one-file config change. (ii) Recording rules: sweep **goodput** (useful ÷ paid GPU-seconds, with paid time taken from provision and teardown timestamps in the manifest), energy per completed rollout (J), and retry and NaN-retry rate per policy, computed post-hoc from `results-summary.json` and driver logs, since AlpaSim exports no retry counter. (iii) One Grafana dashboard JSON that joins `alpasim_gpu_workload_info{gpu}` to DCGM. (iv) Alert rules: XID ≠ 0, queue-depth starvation, and divergence of p95 rollout time across workers. (v) A per-sweep goodput **waterfall**: provisioning and image staging, startup, congestion, steady-state idle, retries, requeue loss, tail drain, teardown. **★:** the goodput definition (what counts as useful) and the alert thresholds. **Acceptance:** the waterfall components sum to paid GPU-h within `____ (Huy)`. **Predictions:** S2 sweep goodput: `____`; largest loss component: `____`. **Kill:** none (instrumentation). **Cost:** $0 | **$0**; it runs alongside the sweeps. The PROF_* fields need SYS_ADMIN. On L6-1's VM they are feasible only if dcgm-exporter (or nv-hostengine) runs on the host and Prometheus scrapes it there, because AlpaSim's sidecar runs inside an Enroot container whose SYS_ADMIN lives in a user namespace. Verify with `dcgmi dmon` on the profiling fields before relying on them | Huy +2–3; agent +6–8 | none | ★ Huy (definition, tolerances); agent (CSV, rules, dashboard, reconciliation script) | AS7, AS10–AS12, NV4, NV5 |
| **L6-4** Failure injection and idempotent resume | D4, Q5 | S2 · G1 (reliability), G7 | **a** | **Build:** on one sweep chunk, inject four faults: (i) `USR1` / `scontrol requeue` mid-sweep, so `runtime.enable_autoresume` engages; (ii) SIGKILL one driver replica, so `runtime.max_rollout_retries` (default 2) engages; (iii) SIGSTOP the renderer for N s, as a hang to be caught by the queue-depth and p95 rules; (iv) the job hitting its time limit, so `USR1@300` triggers a requeue. For each, measure time to detect, time to recover, lost rollouts and duplicated rollouts (`_complete` markers vs `results-summary.json`), and paired score equality on the affected scenes against an uninterrupted reference. This reuses S2's negative-control logic. **★:** the equality tolerance and the diagnosis of any anomaly. **Predictions:** rollouts lost or duplicated after requeue + autoresume: `____`; time to recover from a requeue: `____`. **Kill:** if autoresume loses or duplicates rollouts, file an AlpaSim issue with logs and run S2 without automatic requeue. The bug report is itself evidence. **Cost:** see the next column | 0.25–0.5 node-h of lost in-flight work and restarts, **+$4–9** (N27–V25) | Huy +2–3; agent +3–4 (injection runbook, audit script) | none | ★ Huy (tolerance, diagnosis); agent (runbook) | AS4, AS6, AS8, AS12 |
| **L6-5** S4: measure the RL balance with AlpaGym's own presets | D1 (scale), D4, Q3, Q5, X1 (stand-in) | S4 · G9 | (i)–(ii) **a**; (iii) **c** | **Build:** (i) $0, config only: `perf.enabled=true` in every S4 run. Record AlpaGym's perf JSON plus the merged AlpaSim "Simulator internals", and report trainer, rollout and sim idle fractions. This **reconciles the preserved 22-Sep prediction E1 / headline 1** (the stock 2-GPU AlpaGym loop leaves the trainer idle ≥ 40%), already in 04 §7, so no new slot is needed. (ii) $0, env only: set `TORCH_NCCL_TRACE_BUFFER_SIZE=2000` and `TORCH_NCCL_DUMP_ON_TIMEOUT=true` (Flight Recorder), and capture `ncclras` status on any stall. (iii) **Gated pilot:** AlpaGym's shipped single-node Slurm presets — `local_disaggregated_2gpu`, then `slurm_partial_node_1_2_1` (4 GPUs: 1 policy / 2 rollout / 1 AlpaSim), then `slurm_full_node_1_3_4` (8 GPUs: 1/3/4). Run them on L6-1's Slurm VM, which also needs Pyxis 0.24.0. Measure GPU-h per GRPO step, rollouts/s and trainer idle. Label the result "RTX PRO 6000, not H100". **★:** the balance derivation (N_gen·g ≈ N_train·t, 04 §4) and the diagnosis. **Predictions:** E1 (existing); GPU-h per GRPO step, 8-GPU preset ÷ 2-GPU preset: `____`. **Kill:** if a Slurm preset will not launch within `____ (Huy)` h, keep the 2-GPU local preset. **Cost:** see the next column | (i)–(ii): **$0**. (iii): 3 presets × 1.5 h = 4.5 node-h = 36 GPU-h, **+$65–85** (N27 $14.40/h – V25 $18.78/h) | (i)–(ii): Huy +1–2, agent +1–2. (iii): Huy +3–4, agent +4–6 | (iii) P-005 | ★ Huy (balance math, diagnosis); agent (configs, runs) | AG1, AG2, AG5–AG8, PT1, NV9, L1 §1, L5 §7 |
| **L6-6** S5: turn "pilot first" into a scaling and resiliency ladder | D1 (scale), D4, D5, X1 (stand-in) | S5 · G5 | **a** | **Build:** (i) DDP and FSDP2 (`fully_shard`) at 1, 2, 4 and 8 GPUs on S5's 8×H100 node. Run strong scaling (fixed global batch) and weak scaling (fixed per-GPU batch), each repeated 3× after warm-up, timed with device sync. (ii) `nccl-tests all_reduce_perf` on the node as the communication ceiling: busbw = algbw · 2(n−1)/n. (iii) MFU from Huy's FLOPs-per-sample derivation. (iv) Train under NVRx (section-based hang detection with coordinated in-job restart, straggler detection, async checkpoints), with Flight Recorder on. Inject one hang (SIGSTOP a rank) and one crash (SIGKILL), then measure detection latency, restart time and lost steps. **★:** FLOPs per sample, the efficiency definitions and floor, and the diagnoses. **Predictions:** strong-scaling efficiency at 8 GPUs (DDP / FSDP2): `____`; MFU at 8 GPUs: `____`; NVRx hang-detection latency: `____`. **Kill:** if 8-GPU efficiency < `____ (Huy)`, train S5 at the GPU count with the lowest $ per useful step and report it. **Cost:** see the next column | 2–3 node-h = 16–24 H100-h, **$31–96** (V25 $15.46/h – L27 $31.92/h). **Absorbed**, because S5's card already says "Run-hours are unmeasured; pilot first" | Huy +4–6; agent +6–8 (harness, nccl-tests, NVRx and Flight Recorder wiring) | none | ★ Huy (FLOPs, definitions, diagnosis); agent (harness) | NV7, NV8, NV10, PT1, L3 S5 |
| **L6-7** One 2-node (16-GPU) step | D1 (scale), Q5, D4 | S5 · G5 | **c** | **Proposal only.** A 4–6 h pilot on 2 × 8 H100 with InfiniBand, run after L6-6. (i) Inter-node `nccl-tests`. (ii) The S5 FSDP2/DDP step at 8 vs 16 GPUs, strong and weak scaling. (iii) A multi-node Slurm launch (Nebius managed Soperator with Pyxis/Enroot, or RunPod managed Slurm). (iv) NCCL RAS and Flight Recorder on an injected cross-node hang. (v) Optional: two AlpaSim array chunks under one central Prometheus via file-SD. **★:** the scaling predictions, the busbw floor and the diagnosis. **Predictions:** inter-node all-reduce busbw: `____`; 16-GPU vs 8-GPU strong-scaling efficiency: `____`. **Kill:** stop if inter-node busbw < `____ (Huy)` within the first 45 min, or if spend reaches `____ (Huy)`. Run only if L6-6 shows S5 training is not communication-bound at 8 GPUs by Huy's criterion | 64–96 H100-h. Nebius N27 2 × 8 × $4.50 = **$72/h, so $288–432**. RunPod Instant Cluster, if H200 SXM $6.35 (R08) is per GPU: **$406–610**. Add ~$10–30 for data staging (inference). **Range ≈ $300–640** | Huy +8–12; agent +8–10 | **P-005** (needs headroom in a tier; the ≤ $6k tier already triggers the cap rule) | ★ Huy (design, predictions, diagnosis); agent (launch, harness) | PR5–PR11, NV9, NV10 |
| **L6-8** Upstream contributions | X1 (stand-in), D4 | S2 · G7 | **a** | **Build:** (i) an AlpaSim PR adding the XID, power, energy and clock fields from L6-3 plus a J-per-rollout recording rule; (ii) issues for any autoresume or retry defect found in L6-4; (iii) a short doc PR or issue on "single-node Slurm + Enroot on a rented VM". `submit.sh` hard-codes NVIDIA accounts and partitions; `sbatch` flags override them. Opening a PR or issue is Huy's outward action. **★:** none. **Acceptance:** links recorded. Merging is not a criterion, because AlpaSim has had no external merge since 2026-08-17 | **$0** | Huy +1–2; agent +2–3 | none | Huy (outward action); agent (drafts) | AS4, AS10, AS14, L3 |

**The smallest credible package is L6-1, L6-2, L6-3, L6-4, L6-5 (i)–(ii), L6-6 and L6-8** (inference, from the arithmetic above).
- **Cost:** about +$20 to +$320 in total. Most of it is L6-1's VM premium (+$0–255) and its bootstrap ($15–21); the rest is L6-2's overhead (+$21–33, likely absorbed) and L6-4's (+$4–9), with L6-6 absorbed.
- **Hours:** Huy about **+15–23 h** (≈ 5–7% of ~320 h); agent about +27–37 h.
- **Optional:** L6-5 (iii), +$65–85.
- **The one multi-node purchase:** L6-7, $300–640, gated.

**Coverage after the package** (inference against the earlier analysis):

| Line | Before | After the package |
|---|---|---|
| D4 | Partial | Primary on one node: measured throughput, goodput, fault detection and recovery |
| Q5 | None | Partial: Slurm, Enroot, Pyxis, MPS, array jobs and requeue on one node. Multi-node only with L6-7. Kubernetes still none, by design (§5) |
| Scale half of D1 | Partial | Partial+: a single-node scaling ladder, the measured RL generation/training balance, and AlpaGym's multi-node presets read and explained. Inter-node evidence only with L6-7 |
| X1 | None | Stand-in only (§4, L6-8) |

---

## 2. Facts ledger

### Source index

Every "Fact" row below cites one of these sources. Abbreviations: P = primary, S = secondary. Unless noted, sources were read on 2026-10-27.

**AlpaSim** — `main` @ `affc2eab`, 2026-08-18. Base URL `https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/`:

| ID | File | Type |
|---|---|---|
| AS1 | `src/wizard/alpasim_wizard/schema.py` | P |
| AS2 | `src/wizard/alpasim_wizard/deployment/slurm.py` | P |
| AS3 | `src/wizard/alpasim_wizard/deployment/slurm_enroot.py` | P |
| AS4 | `src/tools/run-on-slurm/submit.sh` | P |
| AS5 | `src/tools/run-on-slurm/resume_slurm_job.sh` | P |
| AS6 | `CHANGELOG.md` | P |
| AS7 | `docs/TELEMETRY.md` | P |
| AS8 | `docs/TUTORIAL.md` | P |
| AS9 | `docs/OPERATIONS.md` | P |
| AS10 | `src/wizard/alpasim_wizard/telemetry/resources/dcgm-counters.csv` | P |
| AS11 | `src/wizard/alpasim_wizard/telemetry/resources/prometheus_sidecar.sh` | P |
| AS12 | `.agents/skills/optimize-slurm-topology/` (SKILL.md, references/metrics.md, topology-knobs.md, experiment-record.md) | P |
| AS13 | `src/wizard/configs/topology/8gpu_alpamayo1_5_eval.yaml` | P |
| AS14 | Repo metadata via api.github.com (`pushed_at` 2026-10-20; `main` HEAD `affc2eab`), plus the git trees of `main` and `e2e_challenge` | P |

**AlpaGym** — `main` @ `972d160`, 2026-07-21. Base URL `https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/`:

| ID | File | Type |
|---|---|---|
| AG1 | `conf/topology/slurm_full_node_1_3_4.yaml` | P |
| AG2 | `conf/topology/slurm_partial_node_1_2_1.yaml` | P |
| AG3 | `conf/topology/slurm_distributed_1_1_1.yaml` | P |
| AG4 | `conf/topology/slurm_distributed_1_2_2.yaml` | P |
| AG5 | `conf/topology/slurm_distributed_shared_cosmos_2_3.yaml` | P |
| AG6 | `conf/topology/local_disaggregated_2gpu.yaml` | P |
| AG7 | `conf/deploy/slurm.yaml` | P |
| AG8 | `slurm.py` | P |
| AG9 | `conf/run/public_2507_1epoch.yaml` | P |

**NVIDIA and the cluster stack:**

| ID | Source | Type |
|---|---|---|
| NV1 | NVIDIA blog, "NVIDIA Acquires Open-Source Workload Management Provider SchedMD", https://blogs.nvidia.com/blog/nvidia-acquires-schedmd/ | P, read via a domain-restricted search summary; page not fetched in full |
| NV2 | SchedMD, GTC 2026 slides, https://slurm.schedmd.com/MISC26/Slurm_GTC_2026.pdf | P, via search summary |
| NV3 | GitHub release feeds (`/releases.atom`) for SchedMD/slurm, NVIDIA/enroot, NVIDIA/pyxis, kai-scheduler/KAI-scheduler, NVIDIA/gpu-operator, NVIDIA/k8s-device-plugin, NVIDIA/DCGM, NVIDIA/dcgm-exporter, NVIDIA/NVSentinel, NVIDIA/nvidia-resiliency-ext, NVIDIA/nccl, NVIDIA/nccl-tests, nebius/soperator, SlinkyProject/slurm-operator, skypilot-org/skypilot | P |
| NV4 | dcgm-exporter `etc/default-counters.csv`, https://github.com/NVIDIA/dcgm-exporter/blob/main/etc/default-counters.csv | P |
| NV5 | NGC catalog, dcgm-exporter 4.8.4, https://catalog.ngc.nvidia.com/orgs/nvidia/k8s/containers/dcgm-exporter/4.8.4 | P, via search summary |
| NV6 | NVSentinel README, https://github.com/NVIDIA/NVSentinel, and docs, https://docs.nvidia.com/nvsentinel | P |
| NV7 | NVRx README, https://github.com/NVIDIA/nvidia-resiliency-ext | P |
| NV8 | Megatron Bridge / NeMo resiliency docs, https://docs.nvidia.com/nemo/megatron-bridge/0.4.2/training/resiliency.html | P, via search summary |
| NV9 | NCCL RAS docs, https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html | P |
| NV10 | nccl-tests `doc/PERFORMANCE.md`, https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md | P |
| NV11 | KAI Scheduler README, https://github.com/kai-scheduler/KAI-scheduler (appears to have moved from NVIDIA/KAI-Scheduler; inferred from README badges and an empty release feed at the old path) | P |
| NV12 | NVIDIA Mission Control docs, "What's included", https://docs.nvidia.com/included-with-mission-control/index.html; BCM licensing, https://docs.nvidia.com/base-command-manager/bcm-licensing-and-support/ | P, via search summary |
| NV13 | DGX Cloud Lepton coverage, https://thenewstack.io/developers-can-now-uber-gpus-with-nvidias-lepton-platform/ | S |

**PyTorch:**

| ID | Source | Type |
|---|---|---|
| PT1 | Flight Recorder tutorial, https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html | P |

**Providers:**

| ID | Source | Type |
|---|---|---|
| PR1 | vast.ai snapshot files, 2026-10-25, in `docs/research/2026-10-25-teacher-to-thor/notes/vast_2026-10-25/` | P data; re-analysed 2026-10-27 |
| PR2 | vast.ai `console.vast.ai` / `cloud.vast.ai` `/api/v0/bundles/`: HTTP 403 at 2026-10-27 08:29 UTC | P observation |
| PR3 | https://docs.vast.ai/instances/virtual-machines | P, undated |
| PR4 | https://docs.vast.ai/create-your-own-multi-node-cluster | P, undated |
| PR5 | https://lambda.ai/pricing and https://lambda.ai/1-click-clusters (page manifest `last_updated` 2026-10-26) | P |
| PR6 | https://www.runpod.io/pricing ("Updated October 8, 2026") | P |
| PR7 | https://docs.runpod.io/instant-clusters | P, undated |
| PR8 | https://nebius.com/prices | P, undated |
| PR9 | https://docs.nebius.com/compute/clusters/gpu | P |
| PR10 | Soperator README, https://github.com/nebius/soperator | P |
| PR11 | https://docs.nebius.com/slurm-soperator | P, thin |
| PR12 | SkyPilot README and PyPI, https://github.com/skypilot-org/skypilot | P, via search summary |

**Earlier ledger** (repo files):

| ID | File |
|---|---|
| L1 | `docs/research/2026-10-03-verification/notes/nvidia_tooling_production.md` |
| L2 | `docs/research/2026-10-03-verification/notes/deployment_hardware_compute.md` |
| L3 | `docs/02_PHYSLOOP_PLAN.md` (v4) |
| L4 | `docs/03_MASTERY_PLAN.md` §6 |
| L5 | `docs/04_CORE_PLAN_TECHNICAL.md` §4, §5, §7 |

### Q1 — AlpaSim run methods, telemetry, and how rollouts parallelize

#### Takeaway
AlpaSim ships three working run methods and no Kubernetes path.
- `SLURM` launches services through Slurm + Pyxis + Enroot.
- `SLURM_ENROOT` launches them through Slurm + Enroot directly, with no Pyxis but with FUSE tools.
- `DOCKER_COMPOSE` launches them through Docker.

Slurm unlocks three features that docker-compose cannot use: CUDA MPS, automatic requeue, and GPU-ownership joins for DCGM. Within a node, rollouts run concurrently across per-GPU service replicas. Across nodes, nothing is shared: each Slurm array task runs a full single-node stack on its own chunk of the config. NVIDIA ships an agent skill that defines how its team tunes topology for throughput, and that skill is the measurement protocol L6-2 borrows.

#### Cited Findings

| # | Fact | Source | Primary/secondary | Confidence |
|---|---|---|---|---|
| 1 | `RunMethod` has four values: SLURM, SLURM_ENROOT, DOCKER_COMPOSE, NONE. `is_slurm()` is true for the first two | [AS1](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/wizard/alpasim_wizard/schema.py) | P | High |
| 2 | No Kubernetes, Helm or k8s files exist in the `main` tree (747 paths) or the `e2e_challenge` tree (1,040 paths) | [AS14](https://api.github.com/repos/NVlabs/alpasim/git/trees/affc2eab209fa43bdfa2f26c0f8d437922d78a68?recursive=1) | P | High (absence verified by path search) |
| 3 | `SLURM` deploys each service as a containerized Slurm job step: `srun --overlap --nodes=1 --ntasks=1 --nodelist=<wizard node> … --container-image=<sqsh>` (the Pyxis flag). Child steps are pinned to the wizard's node, so services are co-located even inside a multi-node allocation. Steps are cancelled with `scancel --name` | [AS2](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/wizard/alpasim_wizard/deployment/slurm.py) | P | High |
| 4 | `SLURM_ENROOT` is a "Direct Enroot deployment inside a Slurm allocation": `enroot start --rw [--root] --mount … <sqsh> bash -c …`, with no Pyxis. It requires `wizard.fuse_dir`, where the wizard installs FUSE tools. Images can optionally be staged to a node-local sqsh cache (`node_local_sqsh_cache_max_gib` defaults to 500) with digest checks | [AS3](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/wizard/alpasim_wizard/deployment/slurm_enroot.py); [AS1](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/wizard/alpasim_wizard/schema.py) | P | High |
| 5 | The 2026-08-11 sync added SLURM_ENROOT ("squash images with private writable overlays and optional node-local image caching") and cleared inherited `srun` step variables | [AS6](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/CHANGELOG.md) | P | High |
| 6 | `wizard.enable_mps=true` starts a per-job CUDA MPS daemon so that co-located GPU services can run kernels concurrently. It "fails fast for non-Slurm run methods"; `docker_compose.py` has no MPS branch | [AS6](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/CHANGELOG.md); [AS2](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/wizard/alpasim_wizard/deployment/slurm.py) | P | High |
| 7 | `submit.sh` defaults: `--nodes=1 --gpus-per-node=8 --exclusive --requeue --signal=B:USR1@300`. USR1 triggers `scontrol requeue`, capped by `ALPASIM_MAX_REQUEUES` (default 20). On restart (`SLURM_RESTART_COUNT > 0`) it passes `runtime.enable_autoresume=true`. Array tasks keep stable `task-<id>` directories across requeues. It writes `resume.sh` (from `sacct` submitline) and `reeval.sh`. Internal targets appear as `deploy=ord|ipp5|iad`, account `av_alpamayo_sim`, partitions polar/grizzly | [AS4](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/tools/run-on-slurm/submit.sh) | P | High |
| 8 | `resume_slurm_job.sh` resumes single jobs or arrays (`--array=0-<n>`) from the original job directory | [AS5](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/tools/run-on-slurm/resume_slurm_job.sh) | P | High |
| 9 | Autoresume tracks completion with a per-rollout `_complete` marker and removes incomplete rollout directories on restart. `generated-user-config-{ARRAY_ID}.yaml` holds the config, "possibly split into chunks" for multi-node simulation | [AS8](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/docs/TUTORIAL.md) | P | High |
| 10 | Failed rollouts are retried up to `runtime.max_rollout_retries` times (default 2). Renderer startup retries transient NRE init failures. Scene-affine renderer dispatch bounds how far scenes are replicated | [AS6](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/CHANGELOG.md) | P | High |
| 11 | Capacity = nr_gpus × replicas_per_container × n_concurrent_rollouts. NRE may run one process per container with `--max-workers` | [AS9](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/docs/OPERATIONS.md) | P | High |
| 12 | Topology `8gpu_alpamayo1_5_eval`: 2 driver replicas on each of GPUs 0–6 (14 rollout slots), with renderer and physics sharing GPU 7. `nr_workers` is 4; cache sizes are 20 (NRE) and 16 (physics) | [AS13](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/wizard/configs/topology/8gpu_alpamayo1_5_eval.yaml) | P | High |
| 13 | Prometheus is on by default (since 2026-06-30). Targets are published by file-SD under the run directory, and a central discovery file (`<run_uuid>.json`) can be shared. On Slurm, `alpasim_gpu_workload_info{gpu}` lets dashboards join AlpaSim processes to DCGM by run, node and GPU | [AS7](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/docs/TELEMETRY.md); [AS6](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/CHANGELOG.md) | P | High |
| 14 | There are four scrape jobs: `alpasim-runtime-worker`, `alpasim-node` (node_exporter), `alpasim-process` (process-exporter) and `alpasim-dcgm` (dcgm-exporter, "when available"). The sidecar starts `dcgm-exporter -f dcgm-counters.csv` only if the binary exists | [AS12 metrics.md](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/.agents/skills/optimize-slurm-topology/references/metrics.md); [AS11](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/wizard/alpasim_wizard/telemetry/resources/prometheus_sidecar.sh) | P | High |
| 15 | AlpaSim's DCGM CSV replaces the exporter's defaults. It lists GPU_UTIL, FB_TOTAL/USED/FREE/RESERVED, PROF_SM_ACTIVE, PROF_SM_OCCUPANCY, PROF_PIPE_TENSOR_ACTIVE and PROF_DRAM_ACTIVE. It omits XID, power, energy, clocks, temperature, NVLink and PCIe | [AS10](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/wizard/alpasim_wizard/telemetry/resources/dcgm-counters.csv) | P | High |
| 16 | Raw runtime metrics: `alpasim_rpc_duration_seconds`, `alpasim_rpc_blocking_seconds`, `alpasim_rpc_queue_depth_at_start_latest`, `alpasim_rollout_duration_seconds`, `alpasim_step_duration_seconds`, `alpasim_simulation_total_seconds`, `alpasim_simulation_rollout_count`, `alpasim_simulation_seconds_per_rollout`, event-loop idle/poll/work, and GC metrics. Recording rules include `alpasim:simulation_seconds_per_rollout:avg` / `:rate5m`, `alpasim:rpc_queue_depth_at_start_latest:max/min`, `alpasim:rollout_duration_seconds:p95_by_worker`, `alpasim:event_loop_idle_fraction:ratio` and GPU memory pressure. No retry counter is listed | [AS12 metrics.md](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/.agents/skills/optimize-slurm-topology/references/metrics.md) | P | High |
| 17 | The NVIDIA skill `optimize-slurm-topology` (synced 2026-07-01) sets the protocol. Optimize full 20 s rollout throughput. Expect ≈5 min startup and 10–15 min congestion, and use the full-run seconds per rollout only once it has stabilized, typically after 30–45 min. Treat GPU memory > 90% as a hard constraint and aim for 70–80% runtime idle. An NRE scene load takes ≈10 s. Evaluate at most 3 candidates in parallel. Keep a Markdown + JSON experiment record | [AS12](https://github.com/NVlabs/alpasim/tree/affc2eab209fa43bdfa2f26c0f8d437922d78a68/.agents/skills/optimize-slurm-topology); history via api.github.com | P | High |
| 18 | `deploy=external_video_model` lets the renderer (OmniDreams gRPC) live on another machine. This is the only documented cross-machine service placement | [AS8](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/docs/TUTORIAL.md) | P | High |
| 19 | `main` HEAD is still #154 (2026-08-18). The repo's `pushed_at` is 2026-10-20, which is consistent with challenge-branch activity | [AS14](https://api.github.com/repos/NVlabs/alpasim) | P | High |

#### Inferences
- **Why single-node Slurm matters.** On one node, Slurm is not ceremony. It is the only way to reach MPS (an evaluable throughput knob), automatic requeue with autoresume (an evaluable reliability path), and the DCGM GPU-ownership join.
- **SLURM_ENROOT is the right method for a rented single node.** It drops the Pyxis plugin requirement, needing only Enroot, FUSE and a writable `fuse_dir`. Pyxis is still needed for AlpaGym's Slurm backend (AG8), so a single install should include it.
- **`submit.sh` likely needs accounting storage.** It calls `sacct` to build `resume.sh`, so a single-node install should include slurmdbd and a database. Unverified on a node.

#### Gaps
- Whether Enroot (user namespaces + FUSE) runs inside an unprivileged vast.ai Docker instance was not tested. See §6.
- Whether `force_determinism` exists for the Alpamayo 1.5 driver was not read. Only the Alpamayo 1 driver's seed derivation is documented (AS6).

### Q2 — NVIDIA's 2026 cluster software stack (open source, usable on a rented node)

#### Takeaway
NVIDIA now owns Slurm's developer.
- **Usable on one rented VM:** the Slurm + Enroot + Pyxis path, DCGM/dcgm-exporter, NCCL RAS, nccl-tests, PyTorch Flight Recorder and NVRx. All are open and current as of October 2026.
- **Need Kubernetes:** KAI Scheduler, NVSentinel and the GPU Operator. On one node none of them yields a measurement Slurm cannot.
- **Neither:** Base Command Manager, Mission Control and DGX Cloud Lepton are a free-but-heavy cluster manager, a paid enterprise product and a marketplace respectively. None is an evidence path for a solo project.

#### Cited Findings

| # | Fact | Source | P/S | Confidence |
|---|---|---|---|---|
| 1 | NVIDIA announced the acquisition of SchedMD, Slurm's developer, on **2025-12-15**, committing to keep Slurm open-source and vendor-neutral. Coverage repeats that Slurm runs on more than half of the TOP500's top-10 and top-100 systems. (Some outlets date the news 12-16.) | [NV1](https://blogs.nvidia.com/blog/nvidia-acquires-schedmd/) | P (via search summary) | High |
| 2 | Since the acquisition, Slurm releases are published on GitHub and GitHub PRs are enabled. NVIDIA sells Slurm and Slinky support | [NV2](https://slurm.schedmd.com/MISC26/Slurm_GTC_2026.pdf); [nvidia.com/en-us/software/slurm-slinky-support](https://www.nvidia.com/en-us/software/slurm-slinky-support/) | P (via search summary) | Medium–high |
| 3 | Current versions on 2026-10-27, from release feeds: Slurm **26.05.5** (2026-10-13); Enroot **4.2.1** (2026-06-09); Pyxis **0.24.0** (2026-05-12); KAI Scheduler **v0.16.14** (2026-10-26) and v0.18.6 (2026-10-22); GPU Operator **26.7.2** (2026-10-14); k8s-device-plugin v0.20.1 (2026-09-22); DCGM **4.7.0** (2026-10-21); dcgm-exporter **4.8.4** (2026-09-18); NVSentinel **v1.29.0** (2026-10-26); NVRx **v0.7.0** (2026-09-16); NCCL **2.32.3-1** (2026-09-21); nccl-tests **2.21.1** (2026-10-05); Soperator **4.1.15** (2026-10-22); Slinky slurm-operator v1.2.3 (2026-09-30); SkyPilot nightly 1.0.0.dev20261027 | [NV3](https://github.com/SchedMD/slurm/releases) (and the other `/releases.atom` feeds listed in the index) | P | High |
| 4 | dcgm-exporter's default CSV includes SM_CLOCK, MEM_CLOCK, GPU_TEMP, MEMORY_TEMP, POWER_USAGE, TOTAL_ENERGY_CONSUMPTION (mJ counter), PCIE_REPLAY_COUNTER, **XID_ERRORS** (value of the last XID), NVLINK_BANDWIDTH_TOTAL, ROW_REMAP_FAILURE, PROF_GR_ENGINE_ACTIVE, PROF_PIPE_TENSOR_ACTIVE, PROF_DRAM_ACTIVE and PROF_PCIE_TX/RX_BYTES | [NV4](https://github.com/NVIDIA/dcgm-exporter/blob/main/etc/default-counters.csv) | P | High |
| 5 | The `DCGM_FI_PROF_*` fields need GPU support and the **SYS_ADMIN** capability. NGC shows `--cap-add SYS_ADMIN` | [NV5](https://catalog.ngc.nvidia.com/orgs/nvidia/k8s/containers/dcgm-exporter/4.8.4) | P (via search summary) | High |
| 6 | NVSentinel exists. It is Apache-2.0 and detects GPU, NIC and system faults through DCGM, syslog and cloud maintenance events, then cordons, drains, resets the GPU or reboots. It **requires Kubernetes 1.34+, Helm 3, the GPU Operator (DCGM standalone), cert-manager ≥ 1.21 and persistent storage for a database**. NVIDIA reports clusters of 1,100+ nodes and ≈40k GPUs (vendor claim) | [NV6](https://github.com/NVIDIA/NVSentinel); [docs](https://docs.nvidia.com/nvsentinel/getting-started/overview/) | P | High (scale claim: vendor-reported) |
| 7 | NVRx is experimental. It offers section-based hang detection with coordinated worker restarts inside a running job, async and local checkpointing, and straggler detection that monitors GPU/CPU performance per rank. It requires PyTorch ≥ 2.8.0, CUDA ≥ 12.8 and NVML driver ≥ 535 (570 for the GPU health check) | [NV7](https://github.com/NVIDIA/nvidia-resiliency-ext) | P | High |
| 8 | NVIDIA's Megatron Bridge docs say fault tolerance needs Slurm and `ft_launcher` (not torchrun) and that straggler detection works on any cluster type | [NV8](https://docs.nvidia.com/nemo/megatron-bridge/0.4.2/training/resiliency.html) | P (via search summary) | Medium |
| 9 | NCCL RAS was introduced in NCCL 2.24 and is on by default (`NCCL_RAS_ENABLE`). It listens on localhost:28028 (`NCCL_RAS_ADDR`) and is queried with the `ncclras` client. It reports per-rank collective progress, missing ranks and DEAD peers (no contact for 60 s). Optional diagnostics (`NCCL_RUN_RAS_DIAGNOSTICS=1`) cover Xid/SXid, ECC and NVLink. JSON output arrived in 2.28.7+ and monitor mode in 2.29+. It does not exercise data paths | [NV9](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html) | P | High |
| 10 | nccl-tests reports algbw = S/t, and for all-reduce busbw = algbw · 2(n−1)/n, which is comparable with hardware peak | [NV10](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) | P | High |
| 11 | PyTorch Flight Recorder: `TORCH_NCCL_TRACE_BUFFER_SIZE` (2000 recommended), `TORCH_NCCL_DUMP_ON_TIMEOUT=true`, dump prefix `TORCH_FR_DUMP_TEMP_FILE` (the tutorial also shows `TORCH_NCCL_DEBUG_INFO_TEMP_FILE`, so the page is inconsistent), and the `fr_trace` / `torchfrtrace` analyzer. Requires PyTorch ≥ 2.5 | [PT1](https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html) | P | High |
| 12 | KAI Scheduler is an Apache-2.0 Kubernetes scheduler (gang scheduling, hierarchical queues, DRF fairness, topology-aware scheduling since v0.10.0 in 2025-10). The repo appears to live now at `kai-scheduler/KAI-scheduler` (inferred) | [NV11](https://github.com/kai-scheduler/KAI-scheduler) | P | High |
| 13 | The free BCM license covers up to 8 accelerators per system, with no enterprise support. Mission Control is the paid tier (BCM plus support, Run:ai, UFM, NetQ) | [NV12](https://docs.nvidia.com/included-with-mission-control/index.html) | P (via search summary) | Medium |
| 14 | DGX Cloud Lepton is NVIDIA's marketplace brokering capacity from cloud partners (Lambda, Nebius, CoreWeave and others). No 2026 pricing for individuals was found | [NV13](https://thenewstack.io/developers-can-now-uber-gpus-with-nvidias-lepton-platform/) | S | Medium |

#### Inferences
- **On one rented VM the stack is complete.** Slurm 26.05 + Enroot 4.2.1 + Pyxis 0.24 + DCGM 4.7 / dcgm-exporter 4.8.4 + NCCL RAS + Flight Recorder + NVRx 0.7 covers scheduler, containers, telemetry, collective forensics and training resiliency, with no Kubernetes.
- **The Kubernetes-only tools need something one node cannot give.** NVSentinel's cordon/drain/remediate needs other nodes to move work to. KAI's fairness needs competing tenants.

#### Gaps
- Nsight Systems and Nsight Compute 2026 versions, and their GPU-counter permissions on rented hosts, were not verified this pass.
- CoreWeave SUNK terms for individuals were not checked.
- Whether NVRx fault tolerance strictly requires Slurm outside Megatron Bridge is not settled; see §6.

### Q3 — Cheapest credible ways for one person to exercise Slurm, Kubernetes and multi-node

#### Takeaway
- **Single-node Slurm + Enroot needs a VM,** or an equivalent root environment. vast.ai VM offers are rare but exist for RTX PRO 6000 (V25: 1–2× at $1.24/GPU, 8× at $2.35/GPU; none for H100). Nebius ($1.80 RTX PRO 6000, $4.50 H100) sells VMs. Lambda ($3.99, 8× H100) is commonly used as a VM provider, but its pricing page does not state VM or root access (unverified this pass).
- **Self-service multi-node** is realistic only on Nebius (InfiniBand GPU clusters, managed Soperator free as software) or RunPod Instant Clusters (2–8 nodes, managed Slurm, H100 price "contact sales").
- **Lambda 1-Click Clusters** have a 16-GPU, 2-week minimum: ≥ $33k for H100.
- **vast.ai multi-node** is Docker-only Ethernet overlay; InfiniBand is by request, and no H100 offer carried a `cluster_id` on 10-25.

#### Cited Findings

| # | Fact | Source | P/S | Confidence |
|---|---|---|---|---|
| 1 | vast.ai VMs give "a complete Linux environment" with an init manager (systemd), ptrace and nested containerization (Docker/Kubernetes). They are selected with `vms_enabled=true`. Limits: slower boot, higher disk overhead, fewer machines, SSH-only access | [PR3](https://docs.vast.ai/instances/virtual-machines) | P (undated) | High |
| 2 | vast.ai multi-node uses overlay networks within a "physical cluster" (an `eth0` virtual LAN, all ports open). The NCCL example uses TCP rendezvous and `NCCL_SOCKET_IFNAME=eth0`. Private networking is **not available for VM templates**. RoCE/InfiniBand needs a "cluster request" and is limited to A100/H100/H200 | [PR4](https://docs.vast.ai/create-your-own-multi-node-cluster) | P (undated) | High |
| 3 | V25, 8-GPU on-demand offers: H100 SXM n=2, at **$15.46/h** ($1.93/GPU; India; unverified host; `cuda_max_good` 12.6, driver 560) and $19.20/h ($2.40; Japan). RTX PRO 6000 Server n=7, from **$10.67/h** ($1.33; unverified hosts) to $13.87 ($1.73, verified). RTX PRO 6000 WS n=4, from $10.67/h ($1.33, verified). **VM-enabled 8×: one offer only**, RTX PRO 6000 WS at **$18.78/h ($2.35/GPU**, Czechia, verified). VM-enabled 1–2× RTX PRO 6000 Server at **$1.24/GPU** (Quebec, unverified, reliability 0.93). **No VM-enabled H100 offer**, and **no H100 offer with a `cluster_id`** | PR1 (repo files `/Users/danghuyhoang/Desktop/physloop/docs/research/2026-10-25-teacher-to-thor/notes/vast_2026-10-25/od_*.json`; re-analysed 2026-10-27 with a local script) | P data | High for 10-25; prices move (02 §8) |
| 4 | RTX PRO 6000 offers report `bw_nvlink` 0.0, i.e. PCIe-only multi-GPU (reported `pcie_bw` 13–55 GB/s). The cheaper H100 8× host reports `bw_nvlink` 478 GB/s; the other reports 0.0, which is probably not measured | PR1 | P data | Medium (vast's measured fields) |
| 5 | The vast.ai bundles endpoint returned HTTP 403 for unauthenticated GET and POST on 2026-10-27 08:29 UTC, so no 10-27 vast.ai re-price was possible without signing in | PR2 | P observation | High |
| 6 | Lambda on-demand (read 10-27, page undated): 8× H100 SXM **$3.99**, 1× $4.29, 8× B200 $6.69, 1× B200 $6.99. No RTX PRO 6000 | [PR5](https://lambda.ai/pricing) | P | High |
| 7 | Lambda 1-Click Clusters: **minimum 16 GPUs, 2 weeks – 1 year**. H100 is $6.16/GPU-h at 16 GPUs ($5.85 at 64, $5.54 at 256); B200 is $9.86 at 16. Quantum-2 InfiniBand with SHARP; managed Kubernetes or Slurm. Page manifest `last_updated` 2026-10-26. A 16 × 336 h × $6.16 minimum is **$33,116** (arithmetic) | [PR5](https://lambda.ai/1-click-clusters) | P | High |
| 8 | RunPod (updated 2026-10-08): H100 SXM community **$2.69** / secure **$3.99**; RTX Pro 6000 $1.69 / $2.49; B200 $5.98 / $7.99. Instant Clusters: **H100 SXM "Contact sales"**, H200 SXM **$6.35/hr**, A100 SXM $2.15/hr. Whether these are per GPU is not stated | [PR6](https://www.runpod.io/pricing) | P | High (unit ambiguity noted) |
| 9 | RunPod Instant Clusters run **2–8 nodes (16–64 GPUs)** with H100/H200/B200 at 3,200 Gbps between nodes. "Managed Slurm" is offered. Larger clusters go through sales; default spending limits apply | [PR7](https://docs.runpod.io/instant-clusters) | P (undated) | High |
| 10 | Nebius (page undated): H100 **$4.50** on-demand, preemptible from $0.79; RTX PRO 6000 **$1.80** on-demand, from $0.79 preemptible; B200 $8.50; B300 $9.50. **Managed Soperator: "Free"** (the software; nodes are billed by consumption). Managed Kubernetes: free. No minimum commitment beyond a $25 first payment | [PR8](https://nebius.com/prices) | P | High |
| 11 | Nebius GPU clusters are self-service (console, CLI, SDK); a VM is assigned to a cluster at creation. Example preset `gpu-h100-sxm` `8gpu-128vcpu-1600gb`; InfiniBand is **400 Gbps per GPU, 3.2 Tbps per node**. Quotas and on-demand eligibility are not stated | [PR9](https://docs.nebius.com/compute/clusters/gpu) | P | High (eligibility unverified) |
| 12 | Soperator (4.1.15) is an open-source Kubernetes operator for Slurm. It provides a shared-root "jail", active and passive health checks with automatic drain and replacement, InfiniBand topology awareness, preinstalled NCCL and nccl-tests, and **Pyxis/Enroot**. It requires NCCL ≥ 2.28 and Kubernetes ≥ 1.35 | [PR10](https://github.com/nebius/soperator) | P | High |
| 13 | SkyPilot v0.12 (March 2026) added Slurm support and "Job Groups for RL". It supports vast.ai, RunPod, Nebius, Lambda, Kubernetes and CoreWeave, and runs inside the user's own accounts | [PR12](https://github.com/skypilot-org/skypilot) | P (via search summary) | Medium |
| 14 | AlpaGym's Slurm backend launches Cosmos tasks with `srun … --container-image=<image>` (Pyxis) and converts registry references to Enroot URIs with registry credentials | [AG8](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/slurm.py) | P | High |

#### Inferences
- **(i) Single-node Slurm with Enroot/Pyxis, running S2 sweeps as array jobs through AlpaSim's own run method, is feasible on any root VM.**
  - The cheapest verified VM rate is $1.24/GPU at 1–2 GPUs on V25, which is inside the plan's band.
  - 8-GPU VMs cost $1.80 (Nebius, if an 8-GPU RTX PRO 6000 preset exists) to $2.35 (vast) per GPU.
  - Inside an unprivileged Docker rental, Enroot probably fails. Docker's default seccomp profile blocks unprivileged namespace creation, and there may be no `/dev/fuse`. This is unverified and testable for under $1 (§6).
- **(ii) Kubernetes (k3s + GPU Operator + KAI) is feasible on a vast.ai VM or Nebius, and probably Lambda (VM/root unverified),** but it produces no PhysLoop measurement. Rejected in §5.
- **(iii) Multi-node:**
  - The only self-service, priced, InfiniBand option found is Nebius at $72/h for 2 × 8 H100.
  - RunPod is a likely second, but its H100 cluster price is not public.
  - Lambda 1-Click Clusters are priced out by the 2-week minimum.
  - vast.ai offers Ethernet only. NCCL over sockets would show nothing about InfiniBand, and no same-cluster H100 pair existed on 10-25.

#### Gaps
- Nebius 8-GPU RTX PRO 6000 preset availability, and new-account GPU quotas, are unverified.
- RunPod's per-GPU H100 cluster price is unknown.
- 10-27 vast.ai prices could not be read (403).
- Lambda: VM/root access is not stated on its pricing page (unverified this pass).
- RunPod: whether it sells anything other than container pods (VMs or bare metal) for single-node Slurm + Enroot was not established; only its managed-Slurm Instant Clusters (2+ nodes) were verified.

### Q4 — Checkable monitoring and debugging evidence, cheap inside S2 and S4

#### Takeaway
Most of the machinery already exists upstream. What PhysLoop adds is the **measurement of it**:
- stabilized throughput per topology;
- fault → detection → recovery timings;
- lost and duplicated rollout audits;
- a goodput waterfall;
- XID, power and energy fields.

All of this runs inside sweeps that happen anyway (L6-2, L6-3, L6-4) or as config in S4 (L6-5 (i)–(ii)).

#### Cited Findings

| # | Fact | Source | P/S | Confidence |
|---|---|---|---|---|
| 1 | AlpaSim already provides retries, requeue, autoresume, per-worker p95 (straggler signal) and queue-depth min/max (starvation and hang signal) | AS4, AS6, AS8, AS12 (Q1 rows 7–10, 16) | P | High |
| 2 | AlpaGym's opt-in `perf/` module writes per-role perf JSON and auto-discovers AlpaSim's Prometheus files as "Simulator internals" | L1 §1 (`/Users/danghuyhoang/Desktop/physloop/docs/research/2026-10-03-verification/notes/nvidia_tooling_production.md`), citing [perf/README](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/perf/README.md) | P (earlier ledger) | High |
| 3 | The preserved 22-Sep prediction E1 / headline 1 says the stock 2-GPU AlpaGym loop leaves the trainer idle ≥ 40% | L5 §7 (`/Users/danghuyhoang/Desktop/physloop/docs/04_CORE_PLAN_TECHNICAL.md`) | P (repo) | High |
| 4 | The plan already requires every NaN-trajectory retry to be logged and the retry rate reported per policy | L3 S2 Acceptance; L5 §6 "Retry audit" | P (repo) | High |
| 5 | NVIDIA's own preset comment reports a "per-node render bottleneck" and a "0.42 baseline" (unit not stated) when AlpaSim nodes are under-provisioned | [AG5](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/topology/slurm_distributed_shared_cosmos_2_3.yaml) | P | High (value's meaning unclear) |
| 6 | NVRx describes its aim as maximizing goodput through health checks, fast fault detection and automatic resume | [NV7](https://github.com/NVIDIA/nvidia-resiliency-ext) | P | High |

#### Inferences
- **Cheap inside S2:** topology and MPS throughput, invariance, the goodput waterfall, XID/energy fields and the failure-injection audit.
- **Cheap inside S4, at config or env level:** AlpaGym perf (which reconciles E1), Flight Recorder and RAS snapshots.
- **Not cheap and not available:** hardware-fault injection, i.e. real XIDs. DCGM-side injection was not verified (§6).
- **A goodput number needs a written definition first.** That is ★ Huy's, because it is a measurement instrument.

#### Gaps
- No AlpaSim retry counter metric is exported (AS12's metric list), so retry rates have to be computed post-hoc.

### Q5 — Distributed-training evidence on one node

#### Takeaway
- **S5 (own code)** gives a clean 1 → 2 → 4 → 8 DDP/FSDP2 ladder, with nccl-tests as the communication ceiling and MFU from Huy's FLOPs derivation.
- **S4 (AlpaGym)** scales only through shipped topology presets.
  - `dp_shard_size` and the replica counts are config-level. Prefetch requires `dp_shard_size == 1`, so a ladder must hold one of the two fixed.
  - The natural single-node scale variable is the preset's policy / rollout / AlpaSim split: 1/0/1 local (2 GPUs) → 1/2/1 (4) → 1/3/4 (8). It measures the RL generation/training balance rather than gradient all-reduce.

#### Cited Findings

| # | Fact | Source | P/S | Confidence |
|---|---|---|---|---|
| 1 | `slurm_full_node_1_3_4`: one 8-GPU node split 1 policy / 3 rollout / 4 AlpaSim GPUs; NCCL transport; disaggregated Cosmos; AlpaSim topology `alpagym_4gpu` | [AG1](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/topology/slurm_full_node_1_3_4.yaml) | P | High |
| 2 | `slurm_partial_node_1_2_1`: 4 GPUs, 1/2/1, non-exclusive, 64G memory | [AG2](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/topology/slurm_partial_node_1_2_1.yaml) | P | High |
| 3 | `local_disaggregated_2gpu`: `execution.backend: local_process`, 1 AlpaSim GPU. These are the only non-Slurm presets besides the 1-GPU colocated one | [AG6](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/topology/local_disaggregated_2gpu.yaml) | P | High |
| 4 | Multi-node presets. `slurm_distributed_1_1_1`: **3 nodes** (2 Cosmos + 1 AlpaSim), policy 4 / rollout 8 replicas, `dp_shard_size: 2`. `slurm_distributed_1_2_2`: **5 nodes**, policy 4 / rollout 16, `dp_shard_size: 2`; cross-node tensors travel over NCCL, replacing the default `transport: disk` | [AG3](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/topology/slurm_distributed_1_1_1.yaml); [AG4](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/topology/slurm_distributed_1_2_2.yaml) | P | High |
| 5 | `slurm_distributed_shared_cosmos_2_3`: 5 nodes rebalanced toward AlpaSim. Five coupled settings ("drop any one and the third AlpaSim node starves") include `n_generation` 2, rollout batch 2, policy 4 / rollout 12, `dp_shard_size` 1, and 30 sim steps at 0.2 s control, which give 22 replay rows | [AG5](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/topology/slurm_distributed_shared_cosmos_2_3.yaml) | P | High |
| 6 | `public_2507_1epoch`: `n_generation` 2, batch 1, `train_batch_per_replica` 8, mini_batch 4, 1 epoch, 1,800 s simulation timeout | [AG9](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/run/public_2507_1epoch.yaml) | P | High |
| 7 | `prefetch_rollout: true` "requires dp_shard_size == 1"; defaults are `n_generation` 1, batch 1, `expected_valid_steps` 8. AlpaGym needs 2 GPUs with ≥ 40 GB | L1 §1 (`/Users/danghuyhoang/Desktop/physloop/docs/research/2026-10-03-verification/notes/nvidia_tooling_production.md`), citing default.yaml and ONBOARDING | P (earlier ledger) | High |
| 8 | busbw = algbw · 2(n−1)/n for all-reduce | [NV10](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) | P | High |
| 9 | H100 SXM is 989 TFLOPS BF16 dense; RTX PRO 6000 WS is ≈ 500 (derived) | L5 §3 (`/Users/danghuyhoang/Desktop/physloop/docs/04_CORE_PLAN_TECHNICAL.md`) | P (repo) | High |

#### Inferences
- **Report scaling honestly:**
  - strong efficiency E(n) = T(1) / (n·T(n)), and weak efficiency at a fixed per-GPU batch;
  - exclude warm-up, synchronize the device, repeat ≥ 3×, and give CIs;
  - state the communication ceiling from busbw;
  - state the interconnect: RTX PRO 6000 nodes are PCIe-only, while H100 SXM nodes have NVLink, so the same code scales differently on the two;
  - never quote an emulated or roofline number as a measurement.
- **MFU** follows the standard model-FLOPs-utilization definition (PaLM, arXiv 2204.02311, App. B; not re-fetched this pass). The numerator for a frozen-VLM-plus-trained-expert model is a judgment call, so the FLOPs per sample are ★ Huy's.
- **AlpaSim rollouts should not get an MFU.** They are a render/inference service graph, so stabilized seconds per rollout and goodput are the honest metrics there.

#### Gaps
- PyTorch FSDP2 docs were not re-fetched; `fully_shard` is assumed available in the pinned torch 2.8.0.
- The NCCL version bundled with the pinned torch 2.8.0 is unverified; read it from S2's resolved lockfile. RAS needs ≥ 2.24.

### Q6 — Sizing the one multi-node step

#### Takeaway
Prefer **S5 on 2 × 8 H100 with InfiniBand for 4–6 h**: 64–96 H100-h, about $300–640, with Nebius cheapest at $72/h. S5 is PhysLoop's own code, so inter-node all-reduce and resiliency can be shown without touching code-frozen AlpaGym. AlpaGym's own multi-node presets start at 3 nodes, 24 GPUs.

#### Cited Findings
- Nebius H100 on-demand is $4.50/GPU-h, with InfiniBand at 400 Gbps per GPU and self-service clusters ([PR8](https://nebius.com/prices), [PR9](https://docs.nebius.com/compute/clusters/gpu)). Managed Soperator is free as software and supports Pyxis/Enroot ([PR8](https://nebius.com/prices), [PR10](https://github.com/nebius/soperator)).
- RunPod Instant Clusters span 2–8 nodes with managed Slurm; the H100 price is "Contact sales" and H200 SXM is $6.35/hr ([PR6](https://www.runpod.io/pricing), [PR7](https://docs.runpod.io/instant-clusters)).
- Lambda 1-Click Clusters require ≥ 16 GPUs for ≥ 2 weeks ([PR5](https://lambda.ai/1-click-clusters)).
- AlpaGym's smallest multi-node preset is 3 nodes ([AG3](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/topology/slurm_distributed_1_1_1.yaml)).

#### Inferences (arithmetic shown)
- **Nebius:** 16 GPUs × $4.50 = $72.00/h. Over 4–6 h that is $288–432.
- **RunPod H200**, if $6.35 is per GPU: 16 × $6.35 = $101.60/h. Over 4–6 h that is $406–610.
- **Data staging:** about $10–30 more (inference).

**What the step demonstrates:**
- inter-node NCCL over InfiniBand (busbw against the 400 Gb/s ≈ 50 GB/s per-GPU NIC);
- cross-node FSDP2/DDP strong and weak efficiency on the real S5 model;
- multi-node Slurm launch with Pyxis/Enroot;
- RAS and Flight Recorder on a cross-node hang;
- central Prometheus discovery across nodes.

**What it cannot demonstrate:**
- failure statistics at 100–1,000+ GPU scale (MTBF-driven restarts, congestion, stragglers at scale);
- AlpaGym's multi-node presets at their designed scale;
- production scheduling (fairshare, preemption, multi-tenancy);
- NVL72 rack-scale NVLink;
- industry experience (X1).

**Risks:**
- new-account quota or on-demand unavailability for InfiniBand clusters (unverified);
- InfiniBand misconfiguration burning paid hours, mitigated by running nccl-tests first and the 45-minute kill rule;
- S5 data must stay internal (PAI-AV/NuRec licence §1 / §4.6), and no weights are published before KC-L2;
- scope creep: it is one gated pilot, not a stage.

#### Gaps
- RunPod's H100 cluster rate and Nebius quotas, as above.

---

## 3. Deltas against the earlier ledger (dated)

| # | Earlier ledger said (date) | Delta found 2026-10-27 | Source |
|---|---|---|---|
| D-1 | AlpaSim has "Slurm, enroot, MPS": CHANGELOG lines only (L1, 2026-10-03) | **Detail added; no code change.** `main` HEAD is still `affc2eab` (2026-08-18). New detail: `SLURM` = Pyxis `--container-image`; `SLURM_ENROOT` = direct `enroot start` with no Pyxis, needing `fuse_dir`; MPS is Slurm-only; `submit.sh` uses requeue, USR1@300, a requeue cap of 20, autoresume and array chunking | AS1–AS6, AS8, AS14 |
| D-2 | Telemetry: Prometheus default, `metrics_plot.png`, Grafana throughput panels (L1, 2026-10-03) | **Detail added:** DCGM counters CSV with PROF_SM_ACTIVE, TENSOR_ACTIVE and DRAM_ACTIVE (synced 2026-08-11; no XID, power or energy); the `alpasim_gpu_workload_info{gpu}` join; four scrape jobs; central file-SD | AS7, AS10–AS12 |
| D-3 | Not recorded (L1) | **New:** NVIDIA's `optimize-slurm-topology` agent skill (synced 2026-07-01) defines the team's throughput protocol and its experiment-record schema | AS12 |
| D-4 | Mastery plan: "Kubernetes/SLURM optional via `SLURM_ENROOT`" (L4, 2026-10-25) | **Refined:** AlpaSim has **no Kubernetes path**, so `SLURM_ENROOT` is the only cluster path. It also needs a root VM (or a test showing Enroot works in Docker rentals) | AS1, AS14, PR3 |
| D-5 | Gap: "did not read `conf/topology/*` or `run/*`" (L1 §1, 2026-10-03) | **Closed:** AlpaGym ships single-node 4- and 8-GPU Slurm presets (1/2/1, 1/3/4) and 3- and 5-node presets (`dp_shard_size` 2, NCCL transport, the "0.42 baseline" comment), plus `public_2507_1epoch`. Its Slurm backend uses Pyxis and Enroot. AlpaGym HEAD is unchanged (`972d160`, 2026-07-21) | AG1–AG9 |
| D-6 | Mastery plan: "multi-node distributed training: uncovered" (L4, 2026-10-25) | Still uncovered in the plan. **The cheapest self-service priced option is now identified:** Nebius 2 × 8 H100 InfiniBand at $72/h (L6-7, c). Lambda 1-Click Clusters are ruled out by a 16-GPU / 2-week minimum (≥ $33k), and vast.ai by Docker-only Ethernet overlays | PR4–PR10 |
| D-7 | Lambda H100 $3.99–4.29 (02 §5, 2026-10-25) | **Unchanged** on 10-27 (8× $3.99, 1× $4.29; B200 $6.69 / $6.99) | PR5 |
| D-8 | RunPod H100 $2.69–3.99 (02 §5, 10-25); the 10-03 ledger had secure H100 at $3.49 and secure RTX PRO 6000 at $2.09 | **Consistent with 10-25.** Against the 10-03 ledger: secure H100 +$0.50 (to $3.99) and secure RTX Pro 6000 +$0.40 (to $2.49), page dated 10-08. New: Instant Clusters H100 is "Contact sales" and H200 SXM is $6.35/hr | PR6 |
| D-9 | Nebius H100 $4.50, RTX PRO 6000 $1.80 (L2, 2026-10-03) | **Unchanged** on 10-27. New: managed Soperator and managed Kubernetes are listed "Free" (software); GPU clusters are self-service with 400 Gbps InfiniBand per GPU | PR8, PR9 |
| D-10 | vast.ai 10-25 basis: RTX PRO 6000 WS $1.00 / $1.60; H100 SXM $1.93 / $2.40 (02 §5) | **Not re-priced** (403 on 10-27). New readings of the 10-25 data: VM-enabled offers are rare (RTX PRO 6000 only; one 8× VM at $2.35/GPU; 1–2× at $1.24); no H100 VM; no H100 `cluster_id`; RTX PRO 6000 is PCIe-only; the cheapest 8× H100 host maxes out at CUDA 12.6 (driver 560), which may not run the plan's CUDA builds (unverified) | PR1, PR2 |
| D-11 | Not in the ledger | **New facts:** NVIDIA acquired SchedMD (2025-12-15); NVSentinel exists (v1.29.0, 2026-10-26, Kubernetes-only); and the October 2026 versions of Slurm, Enroot, Pyxis, KAI, GPU Operator, DCGM, dcgm-exporter, NVRx, NCCL, nccl-tests and Soperator | NV1–NV12 |

---

## 4. First-principles rationale per finding

**L6-1: Slurm path for S2.**
- **Mechanism.** A reviewer on the AV Simulation team runs AlpaSim through `submit.sh` on ORD/IAD (AS4). Running PhysLoop's real sweeps through the same entry point produces artifacts in their own idiom: `sacct` records, array task directories, resume and reeval scripts, requeue counts. Anyone can check them against AlpaSim's code.
- **Why it beats the alternatives.**
  - A toy Slurm cluster with nothing on it produces no measurement, which the rules ban.
  - Kubernetes has no AlpaSim backend.
  - docker-compose cannot use MPS or automatic requeue (AS6).
- **What it costs.** A price premium only where a cheap VM is unavailable: $0 to +$255 on S2. Running every S2 sweep on one run method keeps the policy comparisons unconfounded; mixing run methods would break the one-variable rule.

**L6-2: Throughput, measured the way NVIDIA measures it.**
- **Mechanism.** The skill's protocol turns "we measured throughput" into a falsifiable record: a stabilized metric with a defined warm-up exclusion, a named bottleneck signal, and one knob per candidate.
- **Invariance.** The check uses PhysLoop's own instrument, the gate's paired TOST, to show that MPS, topology and run method do not change scores. That makes reliability and throughput one story. A faster simulator that silently changes results is a D4 failure.
- **Why it beats the alternative.** An ad-hoc wall-clock number is not comparable with anything NVIDIA's team produces.
- **What it costs.** Mostly absorbed: the candidates' rollouts are real sweep rollouts.

**L6-3: Monitoring.**
- **Mechanism.** D4 asks for "monitoring and debugging tools". Three gaps get closed:
  - AlpaSim's CSV omits XID, power and energy;
  - AlpaSim exports no retry counter;
  - nobody publishes goodput for closed-loop simulation.
- **What the reviewer can check.** Prometheus TSDB snapshots are already saved per run (AS8), so a reviewer can load them and see the same panels.
- **Why the VM matters.** The PROF_* fields need SYS_ADMIN (NV5). On a VM you are root, so you can run dcgm-exporter (or nv-hostengine) on the host and scrape it there, bypassing the containerized sidecar. That is feasible but unverified; check with `dcgmi dmon` on the profiling fields. On Docker rentals the fields are likely unavailable (inference).
- **What it costs.** $0.

**L6-4: Failure injection.**
- **Mechanism.** Reliability is a claim about failure behaviour, and the only checkable form is to inject a fault and measure detection time, recovery time, lost or duplicated work, and result equality.
- **What already exists.** Upstream has the mechanisms (`_complete` markers, retries, requeue); PhysLoop supplies the evidence that they work, or the bug report if they don't.
- **What it costs.** ≈ $4–9.

**L6-5: S4 balance.**
- **Mechanism.** RL infrastructure quality at scale is mostly about keeping generation and training balanced (04 §4: N_gen·g ≈ N_train·t). NVIDIA's own preset comments name the render bottleneck (AG5).
- **What it delivers.** (i) reconciles a prediction written on 2026-09-22 at $0. (iii) measures how the balance shifts from 2 to 4 to 8 GPUs using only shipped configs, which respects the AlpaGym code freeze.
- **Why it beats the alternatives.** Integrating NVRx into AlpaGym would mean modifying frozen internals, which the rules ban.

**L6-6: S5 ladder.**
- **Mechanism.** On PhysLoop's own training code, the textbook distributed-training evidence is cheap and turns the plan's unmeasured S5 run-hours into a measured cost model:
  - a scaling curve with a communication ceiling (busbw);
  - MFU;
  - a resiliency drill (NVRx hang detection and restart, Flight Recorder).
- **Why it beats the alternatives.** A synthetic DDP benchmark proves nothing about the real model.
- **What it costs.** Absorbed in "pilot first".

**L6-7: Two nodes.**
- **Mechanism.** It is the minimum configuration where inter-node collectives, multi-node launch and cross-node hang forensics exist at all.
- **Why S5 and not S4.** S5 is unfrozen code; AlpaGym's multi-node presets start at 24 GPUs.
- **Why Nebius.** It is the cheapest priced self-service InfiniBand option found.
- **Why it is gated.** It adds $300–640 and is worth buying only if the L6-6 ladder shows S5 would actually benefit.

**L6-8: Upstream.**
- **Mechanism.** A PR or issue on NVIDIA's own repo is third-party-verifiable evidence of engineering standards. The XID/energy CSV and the retry metric are small and useful.
- **Caveat.** AlpaSim's external-merge drought since 2026-08-17 makes merging uncertain, so acceptance is "filed", not "merged".

**X1, stated honestly.** No solo project supplies industry experience. The strongest partial stand-in is:
1. Operating NVIDIA's own production-shaped paths, with measured throughput, goodput and recovery numbers: AlpaSim's Slurm, Enroot, MPS, requeue and telemetry, plus AlpaGym's Slurm presets.
2. A training loop with production-grade resiliency and failure drills (S5 with NVRx, Flight Recorder and RAS).
3. Public upstream contributions.

All of it must be labelled "solo reproduction on rented single nodes (plus one 2-node pilot)", never "industry infrastructure".

---

## 5. Rejected options, with reasons

| Option | Reason rejected |
|---|---|
| **Kubernetes on a rented VM** (k3s or kind + GPU Operator 26.7 + KAI v0.16/0.18) | No measurement it yields that Slurm doesn't. AlpaSim has no Kubernetes run method (AS1, AS14), and AlpaGym's backends are `slurm` / `local_process` (AG1–AG8), so a Kubernetes path means writing a new deployment backend, which is code, not evidence. KAI's fairness and gang scheduling need competing jobs or tenants. Q5 reads "Kubernetes/SLURM", and Slurm satisfies it. Setup would take ~4–8 h for zero measurements (inference) |
| **NVSentinel** | Requires Kubernetes 1.34+, GPU Operator with DCGM standalone, cert-manager and a database (NV6). Cordon, drain and remediate are meaningless on one node with nowhere to move work. Cite it as the production analogue of L6-3's XID alerts instead |
| **NVRx inside AlpaGym (S4)** | Needs code integration (section wrapping, `ft_launcher`) in frozen AlpaGym internals, which the rules ban. Use NVRx in S5 (L6-6) |
| **Base Command Manager (free ≤ 8 accelerators) / Mission Control** | BCM is a bare-metal cluster manager, a poor fit for rented VMs. Mission Control is paid enterprise. No measurement either way (NV12) |
| **DGX Cloud Lepton** | A brokering marketplace (NV13); no 2026 individual pricing found. It is a possible future source of capacity, not evidence. Re-check only if L6-7's providers fail |
| **Lambda 1-Click Clusters** | 16-GPU, 2-week minimum: 16 × 336 h × $6.16 = **$33,116** (PR5) |
| **vast.ai multi-node** | Docker-only Ethernet overlay; InfiniBand by "cluster request"; no H100 offer had a `cluster_id` on 10-25 (PR1, PR4). NCCL over sockets says nothing about InfiniBand |
| **AlpaGym multi-node presets** (3- or 5-node) | 24–40 H100s: at least 24 × $4.50 = $108/h on Nebius before any useful run. It would use NVIDIA's topology without new insight beyond L6-5 (iii) plus L6-7 |
| **CoreWeave SUNK** | Enterprise Slurm-on-Kubernetes; individual access and terms not verified. Out of reach for a solo, unaffiliated project (inference) |
| **SkyPilot as the evidence** | Useful for reproducible provisioning across vast.ai, RunPod, Nebius and Lambda (PR12), but it adds an orchestration layer without a measurement. It could serve as optional agent plumbing for L6-7 only |
| **Mixing run methods across S2 policies** (cheap Docker for some sweeps, Slurm for others) | Confounds run method with policy and breaks the one-variable rule. L6-2's invariance test is the only licence to mix later (for S3) |
| **Slurm inside a vast.ai Docker rental** (to dodge the VM premium) | Likely infeasible (Enroot needs user namespaces + FUSE; Docker seccomp defaults). Unverified, so kept as an open question with a < $1 test, not as a plan |
| **MFU for AlpaSim rollouts** | Rollouts are an inference and render service graph. Stabilized seconds per rollout and goodput are the honest metrics |
| **Hardware-fault injection (real XIDs) on rented GPUs** | Not safely possible on rented hardware. DCGM-side field injection is unverified (§6) |

---

## 6. Open questions and what would resolve them

1. **Can Enroot run inside a vast.ai Docker rental?** If yes, L6-1's VM premium disappears. *Resolve:* rent a 1-GPU instance for 10 minutes (< $1) and run `unshare -Ur true`, `ls -l /dev/fuse`, and an `enroot import` + `enroot start` smoke test. This needs P-005, since it is spend, and is Huy's call.
2. **Does Nebius offer an 8-GPU RTX PRO 6000 VM preset on demand, and what are new-account GPU quotas,** including for InfiniBand H100 clusters (L6-7)? *Resolve:* Nebius console or quota page, which is Huy's sign-in.
3. **RunPod Instant Cluster per-GPU H100 price, and whether $6.35 for H200 SXM is per GPU.** *Resolve:* the RunPod console cluster quote (Huy's sign-in) or sales.
4. **vast.ai prices on 10-27 and later** (the endpoint returned 403). *Resolve:* re-pull from an authenticated session at S2's start (Huy's action), as the plan's re-price rule already requires.
5. **Does NVRx fault tolerance (`ft_launcher`, in-job restart) need Slurm outside Megatron Bridge** (NV8 says it does)? *Resolve:* read the NVRx fault-tolerance docs (https://nvidia.github.io/nvidia-resiliency-ext/) or test on the S5 node. If it does, run S5 on the same Slurm VM image (an H100 VM premium, P-005) or use the in-process restart mode.
6. **Which NCCL ships in the pinned torch 2.8.0 build?** RAS needs ≥ 2.24, JSON output 2.28.7+. *Resolve:* read the S2/S4 resolved lockfile (`nvidia-nccl-*` pin), which the manifest already records.
7. **Does the cheapest 8× H100 vast.ai host (CUDA ≤ 12.6, driver 560) run the plan's torch build?** *Resolve:* compare the lockfile's CUDA wheel tag with the host's `cuda_max_good` before renting; filter on `cuda_max_good` at S4/S5 start.
8. **Does `submit.sh` work with a single-node Slurm that has no slurmdbd?** It calls `sacct`. *Resolve:* L6-1's bootstrap smoke test; include slurmdbd if it doesn't.
9. **Does AlpaGym run on RTX PRO 6000 (Blackwell, sm_120)** for L6-5 (iii)? It is tested on RTX 6000 Ada and L40S (L1). *Resolve:* a 1-step smoke run on L6-1's VM.
10. **Is the Alpamayo 1.5 driver deterministic per seed?** This decides whether L6-2/L6-4 equality can be bitwise or only TOST-equivalent. *Resolve:* read the 1.5 driver config (AlpaSim `driver=alpamayo1_5`) and run a 10-scene replay twice.
11. **Can DCGM inject synthetic XID or field values for alert testing?** *Resolve:* the DCGM 4.7 docs (`dcgmi test --inject`). If unavailable, alerts are tested only by threshold replay.
12. **What does the "0.42 baseline" in AG5 measure?** It could be a utilization or throughput ratio; the unit is unstated. *Resolve:* ask on an AlpaSim or AlpaGym issue only if it matters for L6-5. Otherwise cite it as an NVIDIA comment without interpreting it.
13. **Nsight Systems/Compute versions and counter permissions on the rented VM,** for the S5 profiler diagnosis. *Resolve:* run `ncu --version` / `nsys --version` and a counter-permission check on the L6-1/L6-6 VMs.

14. **Do DCGM profiling fields work on the rented VM under AlpaSim's SLURM_ENROOT path?** *Resolve:* on the L6-1 VM, run `dcgmi dmon` on the PROF fields as root on the host; if they work there but not inside the Enroot sidecar, run dcgm-exporter on the host and point the `alpasim-dcgm` scrape at it.
