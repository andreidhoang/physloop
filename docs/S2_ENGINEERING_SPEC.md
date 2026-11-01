# S2 engineering specification — updated 2026-11-01

**Status: offline preparation verified; D-019's bounded synthetic Vast host checks executed; S2 scientific execution is not admitted.** This document refines the adopted S2 card in `02_PHYSLOOP_PLAN.md`, serving G1, G2, G3·R1 and G7. D-018 delegates evaluator engineering and adopts the cash ceilings; D-019 authorizes the specific public-software host rental/test/cleanup. Statistical methods, scientific thresholds, license gates and listed file locks remain unchanged. Read with `S2_IMPLEMENTATION_PLAN.md`, `RESEARCH_OPERATING_MODEL.md` and the [host-validation report](research/2026-11-01-vast-host-validation.md).

**Tóm tắt:** S1 kiểm tra bộ ra quyết định. S2 kiểm tra dữ liệu mà bộ đó sẽ nhận: cùng model chạy lại dao động bao nhiêu, nguồn dao động ở đâu, chi phí để phân biệt hai model là bao nhiêu. Đã có kiểm tra GPU/deployment/profiling tổng hợp trên một Vast VM; chưa có bằng chứng AlpaSim/model thực tế hoặc scientific pilot được admission.

## 1. Question, estimand and evidence boundary

S2 must distinguish policy differences from renderer, driver, evaluator and sampling differences. Its primary comparison input is the continuous challenge score in [0,1], higher is better. The writer calls this field `score`; project prose previously calls it `scene_score`. The separate card-style km-between-events statistic is diagnostic, with its own exposure and exclusion convention. Neither is a real-world safety rate.

The operational S1 method is D-016's conservative **equal-road**, then equal-scene, then equal-replicate estimator. A scene-weighted public benchmark mean may be reported separately with that name. Neither should be passed off as the other. The S1 theorem requires independent complete roads; a plausible number of clips is not evidence of independence. Road identity and the inference population need an explicit sampling justification. If the dataset cannot supply defensible road grouping, descriptive reliability is possible but calibrated release claims remain blocked.

Inferential strata must be compatible with the current gate: a road belongs to exactly one stratum. Multi-label descriptive tags may overlap. Huy must resolve road-level partitioning before outcomes are viewed, or propose a statistical amendment; do not split one road into fake independent units. No importance weighting or IRT-selected subset is silently accepted by the equal-road gate.

Four layers have different acceptance meanings:

| Layer | Evidence needed | What it establishes |
|---|---|---|
| Offline preparation | Known-answer synthetic fixtures, schema checks, source audit, reproducible test logs | Plumbing behaves as specified on tested inputs |
| Host engineering check | Delivered VM identities, synthetic CUDA/VRAM controls, actual Compose execution, profiler and telemetry probes | The tested capabilities on that host; no licensed-data or simulation admission |
| Instrument qualification | Actual pinned logs, negative controls, full validity/retry coverage, reliability CIs, measured cost profile | The particular simulation protocol can be characterized |
| Candidate decision | Qualified instrument, frozen comparison/margin/strata, complete paired road data and applicable S1 assumptions | GO / NO-GO / INSUFFICIENT for that registered comparison only |

No number of unit tests proves the simulator valid for physical driving. No pilot result is a model verdict. S1's finite-sample limits can remain too wide even when S2's score looks stable.

## 2. Source lock and compatibility policy

Plan v5/D-021 narrows the active policy matrix in §3, not the sampling or inference contract. The November 1 [RL stack review](research/2026-11-01-rl-direction-review/nvidia-stack.md) additionally distinguishes Gym's dependency graph from this S2 scoring graph; qualification does not transfer automatically between them.

The [upstream audit](research/2026-10-31-s2-upstream.md) records the checked public source and exact writer behavior. The [model/budget audit](research/2026-10-31-s2-models-budget.md) records public model revisions, recipe status and prices.

| Identity | Prepared value | Recheck before running |
|---|---|---|
| Scoring branch | `e2e_challenge` | Post-challenge release/sync and #224 |
| Simulator/evaluator commit | `e121e59141dc391f6c295513d6dadc840a419149` | Re-audit adapter if changed |
| Ground-truth scorer Git blob | `6dce0f50c79350f6c392ff7f0f028bc5c47de93b` | Hash installed file as well |
| Scorer file SHA256 | `5bd239cbe30fee33aeff44c86cf43a05da0ec5095068c86f2ef14f4c102c76c4` | Compare actual bytes |
| Teacher model revision | `7aba8293c09993f2e125c6819df05d7fa3e873ea` | Resolve model/code dependencies and terms |
| Renderer image, driver dependency, runtime lock | Unresolved | Resolve actual digest/commit/lock; never fabricate |

The local CPU `uv.lock` qualifies the gate tooling and pins Shapely for the public geometry-helper tests. It is **not** the AlpaSim/driver environment lock. The selected upstream lock, containers, effective configuration and GPU edition must be captured separately. Same JSON schema version across branches does not establish compatibility. A new branch/release needs a source diff and fixture refresh before ingestion. No branch mixing within a comparison.

## 3. Two registrations, not a circular protocol

**Pilot registration:** five predetermined scenes, one unseeded replicate each, BF16, Harmonizer off/on, renderer timing enabled; ten policy rollouts total plus explicitly budgeted setup/probes. Select by metadata coverage before viewing policy outcomes. Alternate or pre-randomize off/on order and save that order. Everything except Harmonizer remains fixed. One observation per condition cannot estimate a reliable model effect; it supplies compatibility/cost information. Record Huy's runtime/overhead prediction and the hard stop before launch.

**Full registration (scope amended by D-021):** after the pilot, freeze Harmonizer and the forecast. The two active full sweeps are BF16 teacher and its bit-identical re-export control; each is 441 scenes × at least three replicates. At three replicates this is 2,646 policy rollouts, before separately scheduled diagnostics and retries. Full TransFuser/FP8 comparisons are deferred; required pipeline sanity and same-policy/negative controls remain. This scope amendment does not populate or freeze the scientific registration. Preserve the complete schedule, including failed attempts and missing results. Pilot observations are excluded from full-sweep analysis; rerun the registered schedule with fresh replicates. Pilot policy scores must not choose scenes, strata or margins. If outcomes inform redesign, label that work exploratory and justify independent confirmation. The full protocol additionally freezes margin, reliability rule, noise prediction, stabilization criterion, strata, exclusions, target population and analysis plan.

Recommended defaults, pending Huy's record: background traffic replay with ego controlled by the evaluated policy (not the upstream GT replay driver), batch size one, expert graph off, one run method/topology, fixed-sample analysis and unseeded challenge sweeps. No efficacy peeking. Health/cost monitoring may stop a run; that yields incomplete evidence, never a favorable early verdict. If sequential inference is later chosen, use only the S1-qualified finite-look schedule and complete randomized road batches, including its multiplicity/futility rules.

Harmonizer selection is a protocol choice informed by cost and intended rendering conditions, **not** selection of whichever five-scene score looks higher. Unmerged future defaults do not retrospectively rewrite this experiment. Until the intended post-challenge settings are published and matched, report results as provisional.

## 4. Normalization contract

Implemented plumbing: `gate/alpasim_adapter.py`. It consumes three JSON objects supplied by a caller, does no I/O and returns records only when the entire batch passes.

| Input | Required meaning |
|---|---|
| Summary | Pinned schema-1 writer, scoring enabled, per-rollout `score`, raw metrics/status/failure provenance |
| Identity map / expected schedule | Every expected `(run_uuid, clipgt_id, rollout_id)` mapped explicitly to policy, scene, road, stratum and integer replicate index |
| Observation ledger | Requested/actual steps, verified completion, **full-trajectory** maximum deviation, raw required-metric/map completeness and provenance reference |

The identity map is the expected schedule, not a map constructed only from successful outputs. The orchestrator binds unknown rollout UUIDs to declared jobs via launch records; if that binding is unavailable, preserve the expected logical job as missing rather than invent a successful ID. No IDs are guessed from filenames or row order. Replicate indices mean paired schedule positions, not matched seeds. A seeded adapter is not implemented by assigning arbitrary seed values to these records.

Invariants checked on synthetic fixtures:

1. Exact supported evaluator identity is required, beyond schema version.
2. Every expected row is present once; unmapped rows, duplicate attempts and unmatched ledgers block the batch.
3. Nested source identities and duplicated scoring metrics agree. Required scalar metrics are finite; hard failure has a zero score and the writer's exact failure reason. A zero-progress pass remains a valid zero score.
4. Presence of `metrics.error`, including null or empty values, quarantines infrastructure failure. A completed collision/offroad/corridor outcome remains a scored observation.
5. Road/scene/stratum identity is explicit and stable. The downstream pairing layer still checks comparison protocol identity and both arms.
6. Admission requires verified full-horizon observation and full deviation. Values ≥4 m are retained and tallied for the existing validity path; they are never silently deleted or converted to ordinary in-envelope evidence.
7. An incomplete batch returns `records=[]`, issues and per-row diagnostics. Counts describe normalization before whole-batch admission, not a valid analysis denominator. Raw input files remain the audit authority.

**Known v0 boundary:** exact requested/actual horizon equality is currently required. Upstream can legitimately clip a completed recording to available ground truth. Those recordings receive `horizon_unverified`, not an invented infrastructure/driving failure. Real-run admission requires Huy's terminal-segment semantics and a tested extractor for expected clipped horizons, or an explicitly approved alternative protocol. Never silently shrink or replace the required 441-scene schedule with easier exact-horizon recordings. Do not relabel a shortened run as complete to pass the check.

The adapter trusts declared sidecars; it does not authenticate them or compute physical validity. Real-data extractors must establish `_complete`, actual executed steps/timestamps, original untruncated metric arrays, map coverage, black-frame evidence and retry history. The summary may truncate at a collision, offroad or ordinary deviation threshold, and may fill absent offroad data with zero. Summary maxima cannot establish full-trajectory validity.

## 5. Preflight and approval boundary

`gate/s2_preflight.py` validates schema-v2 **logical planning consistency**. It distinguishes planned inputs from realized outputs, pilot from full sweep, public-number approval from derived-weight/label approval, and quote arithmetic from budget authorization. `cloud_launch_authorized` and `publication_authorized` always remain false. References and human attestations are not authenticated.

Prepared templates: `experiments/S2_PILOT_DRAFT.json` and `experiments/S2_FULL_DRAFT.json`. Nulls and pending records are intentional. They must fail readiness today. The older S2 manifest is retained as a historical schema-v1 draft.

The scientific pilot has two operational admissions:

1. **Rental/bootstrap admission:** adopted P-005 tier/caps plus explicit approval of the concrete funding/rental/license package, challenge closure verified and S2 start reached, Huy's license decision, real compatible host quote, planned timeout/retention and pilot registration. Some runtime identities can only resolve during this capped setup.
2. **Scientific-run admission:** resolved environment/config/image identities, complete scene/schedule mapping and frozen applicable protocol; run the offline check, then inspect the referenced evidence. Before a full sweep also require pilot evidence, forecast and explicit release of the remaining S2 cap.

D-019 separately authorized the November 1 public-software/synthetic host check before scientific admission, within its $10/60-minute bounds. That authority has been exercised and does not need to be requested again. It does not authorize another rental, new account funding, gated-asset acceptance/download or the scientific pilot. Public CUDA and other software terms still apply; public availability is not an absence of licensing. Actual task charges and cleanup verification belong to the [host report](research/2026-11-01-vast-host-validation.md).

Do not ask Huy to invent hashes or a renderer digest. Agents obtain those. Do not label structural readiness a launch permit. Other costs include tax, setup and non-compute charges; paid idle/startup time belongs inside billed GPU-hours. Quote availability, cumulative spend and actual provider shutdown require operational verification.

**Host terminology and evidence, 2026-11-01:** `host_compatibility_verified.full_host_access` means administrator access to the Linux Docker host, which may be a **VM guest**, sufficient to manage the required services/runtime and establish monitoring permissions. It does not require physical-server or hypervisor ownership. The boolean is an attestation, not a hardware probe. Actual KVM guest root, CUDA execution, buildx and Compose DNS/network/shared-mount/guest-host-PID/GPU operation were tested on Vast instance `58628722`; ordinary unprivileged GPU containers remain a different route. See the [source assessment](research/2026-11-01-vast-vm-fit.md) and [measured report](research/2026-11-01-vast-host-validation.md). No complete draft host attestation is promoted by these partial checks.

The delivered RTX PRO 6000 Blackwell Workstation Edition reported 97,887 MiB VRAM and driver 580.95.05; guest RAM was 96.996 GiB rather than the listing's 123.438 GiB. The CUDA integer oracle/repeat/negative control and approximately 70.82 GiB VRAM verification passed their known-answer checks. Nsight Compute read actual kernel counters, Nsight Systems captured three CUDA kernels, and DCGM 4.7.1 tensor-active responded to the exact-answer cuBLAS control (idle → approximately 0.576 → idle). Separately, exporter 4.8.4/DCGM 4.6.1 embedded loopback scrapes responded to load. These establish neither AlpaSim performance nor simultaneous teacher/renderer/Harmonizer memory fit. Recheck a newly rented host; no permanent provider adoption follows.

**Monitoring gap:** XID was N/A in DCGM and absent from exporter samples. NVML supported-event mask `0xf19c` and successful registration establish an available subscription API; the short wait returned TIMEOUT, so event delivery/alerting remains unverified. Do not convert N/A, missing samples or timeout to zero errors. Required monitoring still needs complete qualification or an explicit acceptance amendment.

## 6. Qualification work packages and acceptance

| Package | Outputs and acceptance | Ownership / current status |
|---|---|---|
| Source/schema audit | Immutable pins, sample contracts, compatibility differences | Agent; completed offline |
| CPU plumbing | Preflight, adapter, CLI, adversarial fixture tests | Agent; see generated verification |
| Actual environment | Resolved lock/config/images; clean setup transcript; host GPU/driver/edition identity | Public/synthetic Vast guest checks measured under D-019; licensed AlpaSim bootstrap, complete runtime pins and simultaneous memory fit remain outstanding |
| Raw observation extraction | Exact logical-job binding, full metric streams, expected/actual horizon, retry and validity provenance | Agent plumbing under Huy's validity semantics; actual-file integration outstanding |
| #198 scorer control | Original minimal reproduction: stationary tail's false ~5 m lateral error becomes ~0 on the pinned scorer; endpoint/boundary controls | Actual public geometry helpers tested on CPU; complete evaluator/runtime execution remains unverified |
| Instrument decomposition | Fixed-pose renderer repeats per Harmonizer; identical policy inputs; first divergence tick; replay-traffic control | Agent implements decomposition/capture and technical diagnosis; Huy owns scientific interpretation |
| Reliability | Registered repeatability/ICC and event-noise uncertainty, noise-floor and negative-control rate, road-cluster uncertainty, base rates/discordance/paired SD; Kendall W/policy-panel ranking deferred until a separately admitted meaningful panel | Agent-owned evaluator implementation under D-018, with independent review and Huy's scientific definitions; no fabricated results |
| Validity and contamination | Deviation bins, lateral corridor separately, overshoot/black frames, quality/time/speed joins, source split for all 441+1,761 scenes | Huy defines semantics/rule; agent joins/audit |
| Performance/cost | Stabilized throughput, service/driver split, goodput, mean paid time, energy, retry ledger, hard-stop forecast | Synthetic profiler/counter paths measured; XID delivery and full-pipeline profile remain outstanding. Huy freezes criterion/cost interpretation; agent telemetry |
| Handoff | G1 report, G2 records, G3·R1 measured power inputs, G7 monitoring with linked immutable evidence | Requires all mandatory S2 acceptance items |

A same-policy same-seed negative control cannot be claimed on the unchanged challenge driver: it ignores the session seed. Keep unseeded test–retest distinct. To complete the adopted deterministic control, prepare and qualify a separately pinned per-tick reseed diagnostic port, including deterministic runtime settings, and identify its protocol separately. Alternatively record an explicit amendment to that acceptance item. The default full sweeps remain unseeded; diagnostic results do not prove their determinism. Graph stays off until the adopted bitwise/negative-control checks pass on the pinned driver.

ICC alone is not a decision threshold: large between-scene variability can yield high ICC while paired uncertainty is still too large for the chosen margin. Reliability acceptance must include the absolute uncertainty needed by the research question. Normal/ICC planning estimates do not replace the conservative operational S1 bound. More repeats reduce conditional rollout noise; they do not create additional independent roads.

## 7. Proof obligations and limits

Offline correctness claims are local: no silent admission after a checked issue; no conversion of an infrastructure row into a driving outcome; no fabricated launch/publication authorization; budget arithmetic obeys declared nested caps; known malformed inputs return structured blockers. Tests exercise counterexamples and a source-faithful valid fixture. Independent review found and corrected malformed-input crashes and contradictory score admission before packaging.

The S1 theorem is linked, not re-proved for an unknown dataset. Unverified obligations include actual raw-log interpretation, sidecar authenticity, dataset sampling/road independence, scorer execution, model/renderer runtime reproducibility, simultaneous memory fit, complete monitoring, dataset licenses and the actual host access arrangement. The synthetic GPU measurements above are narrow host evidence. No end-to-end simulator correctness, scientific reliability, long-duration uptime or universal correctness is claimed.

**Giải thích ngắn:** nhiều lần chạy trên cùng đường giúp đo nhiễu, nhưng không tạo thêm đường độc lập. GO cần dữ liệu đủ tốt *và* đủ mạnh để chứng minh tiêu chí đã chốt. Máy đo chưa đủ tin cậy hoặc mẫu quá ít thì kết quả hợp lệ là INSUFFICIENT.
