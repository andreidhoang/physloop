# Track E: an audit of the PhysLoop plan for an inference-led capstone (as of 2026-10-28)

**Scope.** This note audits plan v4.1 against three capstone paths: (1) a research paper, (2) exceptional engineering on Jetson AGX Thor, (3) shadow mode in HCMC. It does not score or pick a path. Every prediction, tolerance, margin, latency model, kernel design and diagnosis stays Huy's (★), and no `____` slot is filled.

**Labels.**
- `repo X:n` is a file and line in `/Users/danghuyhoang/Desktop/physloop`.
- `code path:n@commit` is read from source at a named commit.
- `reported` is an external primary source, with link and read date.
- `inferred` shows its arithmetic.
- `unverified` is stated but not checked.

**Abbreviations.**
- Repo documents:
  - `02` = `docs/02_PHYSLOOP_PLAN.md`
  - `01` = `docs/01_SYSTEM_DESIGN.md`
  - `03` = `docs/03_MASTERY_PLAN.md`
  - `04` = `docs/04_CORE_PLAN_TECHNICAL.md`
  - `05` = `docs/05_ROLE_MAPPING.md`
  - `DEC` = `docs/DECISIONS.md`
  - `nano` = `nano_alpamayo/README.md`
- Research notes:
  - `L1`, `L5`, `L7` = `docs/research/2026-10-27-role-alignment/notes/L1_upstream_delta.md`, `L5_student_data.md`, `L7_thor_performance.md`
  - `rep` = `…/2026-10-27-role-alignment/report.md`
  - `ver27` = `…/2026-10-27-role-alignment/verification.md`
  - `ver25` = `docs/research/2026-10-25-teacher-to-thor/verification.md`
  - `rev` = `…/2026-10-25-teacher-to-thor/review.md`
- AlpaSim code:
  - `@E` = AlpaSim `e2e_challenge` @ `e121e591` (2026-10-20); `@M` = AlpaSim `main` @ `affc2eab` (2026-08-18). Both were still the branch heads on 2026-10-28 (GitHub `list_commits`).
  - `RT/` = `src/runtime/alpasim_runtime/`; `CTL/` = `src/controller/alpasim_controller/`; `WZ/` = `src/wizard/configs/`; `DRV/` = `src/driver/src/alpasim_driver/`.
  - Local clones: `scratchpad/repo_e2e/alpasim` and `scratchpad/repo_main/alpasim`.

**Scratch arithmetic** (reproduce with `python3 -I <file>`):
- `/private/tmp/claude-501/-Users-danghuyhoang-Desktop-physloop/6a1c1084-894a-4933-a900-90a396bc7621/scratchpad/capstone_scratch/E/delay_ka.py` (output in `delay_ka.out.txt`)
- `/private/tmp/claude-501/-Users-danghuyhoang-Desktop-physloop/6a1c1084-894a-4933-a900-90a396bc7621/scratchpad/capstone_scratch/E/calc.py` (output in `calc.out.txt`)

---

## 1. Takeaways

1. **S5(c)'s score axis needs no Thor; only its latency axis does.**
   - **Scores carry no latency by default.** An AlpaSim score includes latency only if `planner_delay_us` is set. Its default is 0, and the runtime asserts zero decision delay (`code WZ/base_config.yaml:288,293@E`; `RT/events/policy.py:72-74,169-188@E`).
   - **What S6 must supply.** Latency reaches the score only as ⌈L/100 ms⌉ control ticks (§3), so S5(c) needs from S6 just an n → latency map good enough to place each arm in a 100-ms band.
   - **Why that is still hard.** The plausible 2B-student latencies straddle the 100-ms boundary. The decision-doc roofline floors are 53–59 ms without CoC and 139–145 ms with 20 CoC tokens (`repo rev:51-57`). NVIDIA's measured 171 ms FP8 is on **DRIVE** AGX Thor (`repo 04:167`).
   - **What the plan lacks.** No item states this decomposition, and S5(c)'s build list has no latency-injection step (`repo 02:623-633`).
2. **AlpaSim rounds latency up to whole ticks, from code.** The delayed plan is read once per 100-ms control tick (`code RT/events/controller.py:123-127@E`, identical @M).
   - **Rounding.** Any latency in (100, 200] ms is simulated as 200 ms of plan staleness. So 139 ms puts the car 3.0 m down the road on a stale plan at 15 m/s, not 2.085 m. Any non-zero latency costs at least one full tick (`code`, re-typed buffer run in `delay_ka.out.txt`).
   - **How stale plans are used.** They stay anchored in the world and are indexed by time, and the MPC ignores the next 1.0 s of the plan (`code CTL/system.py:332-359`; `CTL/mpc_impl/linear_mpc.py:177-190`; `CTL/mpc_controller.py:53`).
   - **Warm-up.** The initial ⌈L/Δt⌉ closed-loop ticks track force-GT references (`code RT/events/controller.py:75-93,123`).
3. **Lockstep is real, and it is the challenge's own production path.**
   - The competition preset runs the driver through `driver_source: external_static` (`code WZ/e2e_challenge/ec2.yaml:16,27-29@E`).
   - The runtime awaits `drive()` with no per-call deadline (`code RT/services/driver_service.py:262-264@E`; `RT/telemetry/rpc_wrapper.py:96-125@E`).
   - So a slow Thor stalls simulation time but never enters it.
4. **Path (2) has no like-for-like public baseline on Jetson AGX Thor for a 2B Alpamayo student.** Re-checked 2026-10-28:
   - **TensorRT-Edge-LLM** exports Alpamayo-R1-10B in FP16 only. It has had no commit since 10-20, and PR #260 is still open and blocked (`reported` [PR #260](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/260)).
   - **Recipes PR #45** targets DRIVE AGX Thor on TensorRT 11.2/11.3 (`repo L5:55`); Jetson runs 10.16.2 (`repo 04:52`). Its body says the forked Edge-LLM is used only "while the PR lands" upstream (`reported` [PR #45](https://github.com/NVlabs/alpamayo-recipes/pull/45), read 2026-10-28). That is a race signal.
   - **FlashRT's** Jetson numbers are for the 10B teacher: 811 ms at best for a full request (`repo 04:173`).
5. **NVIDIA already ships most of S6's vision-side and precision items.**
   - Edge-LLM ships media-aware KV reuse (v0.10.0), FP8 ViT attention and a ViT-embedding cache (v0.10.1), and Thor NVFP4-A16 kernels (v0.11.0) (`repo L7:95-97`; `ver27:109`).
   - ModelOpt's Alpamayo example targets **Alpamayo 1 (R1)**. Its QAD trains only the VLM, against PAI-AV clips (`reported` [ModelOpt examples/alpamayo README @b6b0ff52](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/alpamayo), read 2026-10-28).
6. **The cut order exposes path (2) most.**
   - §0.1 cuts S8 before anything else, then [O] items, then [C] role items in reverse order of the per-stage lists, and never the corrections (`repo 02:79`; quoted verbatim in §5).
   - On the literal reading, the "S3 and S6–S7, integration and C++" list is cut ahead of the other [C] role items (`repo 02:74-77`). That list is the batched student driver, the TensorRT-11-ready C++ and the production C++ driver server.
   - §8 appends "then S7" (`repo 02:872`), which §0.1 and D-015 do not (`repo DEC:249`).
7. **The calendar is tight.**
   - A 12-week window from 11-02 ends on 2027-01-25, which is KC-7's date (`inferred`, `calc.out.txt`).
   - At 25–30 h/week the window holds 300–360 h, against 306–340 h of S2–S7 [C] work. That leaves −40 to +54 h of slack (`inferred`).
   - The inference-led content already scheduled is S5–S7 [C], 150–164 h. P-002 must be decided before S6's 12-07 start, and P-005 before S2's spend (`repo 02:184,806`).
8. **Publication gates are set by the data, not the path.**
   - Every closed-loop score any path would cite is computed on NuRec 26.01/26.04 reconstructions of NVIDIA Physical AI dataset clips. KC-L covers it (`code e2e_challenge/local_evaluation/README.md:11-17,31-36@E`; `reported` NuRec card).
   - The licence's §3 counts "regression or performance data relating to the Dataset" as confidential (`reported` HF NuRec gate text, read 2026-10-28).
   - Latency and bitwise checks on synthetic or own data fall outside the dataset licence (`inferred`).
   - P-004 is open, with no record that it was sent (`repo DEC:306-313`; no `journal/` exists).
9. **The KC-E fallback breaks §1's framing.** If Edge is not public, or has no runtime-CoC switch, S5(c) runs on the released 10B teacher (`repo 02:639`; `nano:46`). No 10B configuration reaches 10 Hz on Jetson AGX Thor (`repo 01:176-177`), so "a fixed on-device latency budget" (`repo 02:118`) has no meaning there.
10. **S5(c) conflicts with its own prerequisite.**
    - The card fixes the checkpoint and the architecture (`repo 02:626,630`).
    - Yet it requires history-frame prefill reuse (`repo 02:631`), which FlashDrive needs expert fine-tuning for: 0.96 vs 0.77 minADE6 without vs with (`repo rep:66`; `01:165`).
    - The quoted +2.7% is the fine-tuned cost (`repo rev:62`).

---

## 2. Path → plan-item map

### 2.1 Path (1), a research paper

**Which gaps are open** (dated evidence):
- **Score vs CoC budget n on the deployable student, paired:** open.
  - TacitVLA (NVIDIA, 2026-10-13) reports only three unpaired endpoints, 46.4 / 52.3 / 52.5, with H100 latency and no token-budget sweep (`repo ver27:97`; `L1:229`).
  - No NVIDIA sweep was found (`repo L5:116`). TRUST (non-NVIDIA) finds that gains in reasoning correctness on 1.5 in AlpaSim do not automatically become driving gains (`repo L1:239`).
- **Why the deployed student drives worse than its teacher, paired and stratified:** open.
  - There is no paired student-vs-teacher closed-loop verdict (`repo 02:101`); PR #42 and #45 are open-loop or latency only (`repo 02:406`).
  - It is blocked on Edge's weights (KC-E, 10-31; HF still lists no Edge model on 2026-10-28, `reported` [HF API](https://huggingface.co/api/models?author=nvidia&search=alpamayo)).
- **Closed-loop score vs flow steps K:** open for Alpamayo.
  - NVIDIA deploys K = 5 with fixed noise (PR #45; `repo L5:55`).
  - Few-step evidence comes from other architectures and open-loop metrics, and the known order holds only for ODE error, not for closed-loop score (`repo L5:95,110`).
- **Closed-loop score vs precision on real engines:** open on Thor.
  - ModelOpt publishes open-loop minADE only (`repo 01:166`), and FlashDrive's W4A8 gives 1.16× (`repo 04:79`).
  - A generic closed-loop compression precedent exists in Duckietown (arXiv 2609.00718; `repo L1:240`).
- **Score lost to latency:** the mechanism is NVIDIA's (a 200 ms planner delay in its configs until 2026-03-24, `repo ver27:61`). No published latency-cost curve for an Alpamayo-family policy turned up in the plan's sources (`unverified`: not searched beyond them).

**Adjacent gap with no item** (`inferred`): no plan item composes the separate levers into one closed-loop score vs on-device latency frontier on a common protocol. Those levers are n (S5(c)), K (S5(b)) and precision (S6 G6), with latency injection (S7). Each lever is a separate one-variable experiment (`repo 02:619-633,688,719-723`). A frontier built as a set of one-variable arms would still respect the contract's one-variable rule (`repo CLAUDE.md`, "How we work").

| Plan item (repo) | What it would prove | What it cannot prove | Gates |
|---|---|---|---|
| S3 paired verdict, strata, controller split into plan geometry / controller tracking / decisions (`02:411-436`) | A paired, stratified NI verdict (or INSUFFICIENT) for Edge vs 1.5 under protocol P; where the student loses; whether a difference is geometry or decisions (ECO-style) | Anything if Edge's weights are not public; real-world validity; decidability if σ_d ≳ 0.59 for a 7 pp margin (`03:77`) | KC-E, P-005, KC-L |
| S3 KC-E fallback: "deployment levers" on the released teacher (`02:461-464`) | Known-order closed-loop pairs on the teacher: K = 5 with fixed noise vs K = 10; CoC capped vs uncapped; FP8 (emulated) vs BF16. The same axes as path (1), inside S3's 3–4 sweeps | Transfer to the 2B student; any on-device latency claim | P-005, KC-L |
| S5(b) flow-step ladder, K ∈ {10, 5, 4, 2} with common noise (`02:619-622`) | Closed-loop order across K on one checkpoint, paired by noise file | That the closed-loop order is "known": only the ODE-error order is (`L5:110`; `rep:153`); Thor latency per K | P-005 (+2 sweeps, $68–265, `rep:153`), KC-L, KC-E |
| S5(c) CoC-budget curve (`02:623-633`; `nano:25`) | Score(n) in lockstep at fixed architecture, with realized length per tick and paired caps | "At a fixed on-device latency" without S6's n → latency map and S7-style injection (§3.6); separating truncation shift from information value unless the optional arm runs (`02:633`; `rep:159`) | KC-E (runtime switch), P-005, KC-L; P-002 for device latency |
| S6 G6 per-format verdicts on real engines (`02:149,688`) | NVFP4 vs FP8 vs BF16 closed-loop verdicts for Jetson engines | DRIVE AGX Thor behaviour; a cloud FP8 engine standing in for the Thor engine (`DEC:488`; `L7:189`) | P-002, P-005, KC-L |
| S7 constant-delay injection (`02:719-723`) | The score change at ⌈L/100 ms⌉ ticks of plan staleness | Differences within a band (e.g., 139 vs 171 ms); sub-tick latency; transport or actuation delay as separate terms (§3) | P-002, KC-L |
| P-018, a few-step expert (`DEC:482-485`) | A trained few-step alternative, if S5(b) shows a cliff | Novelty if Edge is already few-step; publication of weights without KC-L2 | P-005, KC-L2 |
| P-015, a batched driver (`DEC:465-470`) | That batching leaves the verdict unchanged (a known null the gate must not flag), plus throughput | Bitwise identity: batching changes numerics and RNG (`L7:250`) | P-005, KC-L |

### 2.2 Path (2), exceptional engineering (2B student at 10 Hz on Jetson AGX Thor; a win that does not change behaviour)

**Which gaps are open:**
- **10 Hz:** no 2B student has been shown at 10 Hz on Jetson AGX Thor (`repo 02:103,714`).
- **NVIDIA's own 2B numbers:** 171.2 ms median in FP8 on **DRIVE** AGX Thor, about 5.8 Hz, through a forked Edge-LLM (`repo 04:167`; `ver25:34`). Edge's "10 Hz" is stated for DRIVE AGX Thor with no camera, reasoning or precision configuration (`repo L7:103-104,113`).
- **Baselines** (`inferred` from `04:56,173`; `L5:55`):
  - Edge-LLM: R1-10B FP16 only;
  - PR #45: DRIVE and TensorRT 11.x;
  - FlashRT: the 10B teacher.

  None is a Jetson 2B baseline, so "a measured win over" them needs a stated common model, device and input. No item names one: the S6 Why lists the numbers but no baseline (`repo 02:675-680`).
- **Precedents for "does not change behaviour":**
  - PR #36's 160/160 bitwise expert evaluations (`repo ver27:60`);
  - Edge-LLM's pi0.5 parity procedure, which injects the reference initial noise and thresholds cosine similarity and max absolute error (`repo L7:183`);
  - FlashRT's 2026-10-27 GR00T port, which claims projections bitwise equal to torch (`reported` [FlashRT commit f4ddc906](https://github.com/flashrt-project/FlashRT/commit/f4ddc9060ba952ad767598ee79df27f3a8dd2024), read 2026-10-28).

| Plan item (repo) | What it would prove | What it cannot prove | Gates |
|---|---|---|---|
| Real NVFP4/FP8 engines and FP8 prefill (`02:682`) | Real-engine latency on Jetson (labelled "real engine") | DRIVE numbers; that behaviour is preserved without P-009's ladder and a verdict | P-002 |
| Static KV cache and CUDA graphs (`02:683`) | Removal of launch overhead. CUDA graphs over the same kernels are bitwise; static padded KV is numerics-changing (`L7:247-251`) | Speed-ups on Thor from the PyTorch-path graph (PR #36 is PyTorch, B ≤ 4) | P-002 |
| Fused flow-step kernel (`02:684`) | A ★ kernel, if S6a's diagnosis calls for one | Anything until P-007/P-008 decide; overlaps PR #260's SM110 fused gate/up plugin (open) | P-002, P-007 |
| Camera-batched ViT; FP8 ViT; ViT cache (`02:685`) | Little: NVIDIA ships these (`ver27:109`) | Novelty; building them races NVIDIA (`DEC:291`) | — |
| Streaming reuse of history frames (`02:686`) | Prefill cut (≈ 4× by `rev:62`) | That it is a serving-only change: it needs expert fine-tuning (`DEC:356-358`; `rep:66`) | P-002; KC-L2 if the fine-tuned weights are published |
| p50/p99 per stage; a sustained 30-minute 10 Hz run with thermals; energy (`02:687`; owner targets `02:131-132`) | The owner's device targets, named per device | DRIVE equivalence (`02:828`) | P-002; P-009 (iii) clocks |
| Expert-efficiency diagnosis (`02:689`; P-007) | Where the expert's time goes on sm_120, then on sm_110 | That the "~20% of bound" figure is measured: it is derived (`DEC:328-335`) | P-005 (a few GPU-h), KC-1 |
| TensorRT-11-ready C++: strongly typed, `IPluginV3`, CI on 10.16 and 11.x (`02:691-694,705`) | Portability; no speed claim | — | P-002 for the device part |
| Production C++ driver server (`02:724-731,737`) | Q4 evidence: sanitizer-clean runs, golden bitwise replay, p99.9 histograms (`05:86`) | Anything without the device, unless KC-6's RTX PRO 6000 fallback is applied | P-002 |
| P-019, one runtime for two targets (`DEC:486-490`) | The same runtime serves AlpaSim student sweeps on SM120 through `external_static`, then runs on Thor. This joins paths (1) and (2) | A Thor-engine verdict from a cloud run (`DEC:488`) | P-007, P-005, KC-L |
| P-009 (i) contention, (ii) glass-to-action, (iii) pinned clocks (`DEC:362-374`) | Latency under shared bandwidth; an end-to-end delay; reproducible clocks | Sub-tick score effects in AlpaSim (§3) | P-002 |
| P-010, the GEMV exercise (`DEC:375-381`) | The H2 (starvation) discriminator for P-007 (`L7:330`) | Loop-level evidence on its own | P-005 (≈ 1 GPU-h) |

### 2.3 Path (3), shadow mode (S8)

| Plan item (repo) | What it would prove | What it cannot prove | Gates |
|---|---|---|---|
| S8 rig, disagreement mining, OOD set, within-rig contrast (`02:743-761`) | An open-loop disagreement rate over Y hours of HCMC traffic; an OOD set (PAI-AV has no Asian country, `02:744`; `ver25:58`); real-sensor latency on the device (`01:184`) | Driving quality, i.e. validation (`01:183-186`; `02:827`); closed-loop effects, since covariate shift is invisible (`rev:130`) | P-006, KC-8, privacy plan. **P-002 implicitly:** the rig is built on the Jetson dev kit (`02:751`), though §9 lists P-002 only for S6–S7 (`02:884`) |
| P-006 (`DEC:317-323`) | The decision to go: rig budget, local rules, privacy | — | Huy's decision; engineering reading only, not legal advice (`02:749`) |

**Joint point with path (2)** (`inferred`): S8's real-sensor glass-to-action on the device is the same quantity as P-009 (ii). Shadow mode is the only path in the plan that measures it with real cameras rather than through AlpaSim's injected constant.

---

## 3. AlpaSim delay and lockstep mechanics (from code)

All of this section is `code` unless it is labelled otherwise. The delay-buffer logic is identical on @E and @M: `RT/delay_buffer.py` is identical, and the delay lines 1–149 of `RT/events/controller.py` are identical, with the files differing only at lines 150–160, on force-GT dynamics (`diff`, 2026-10-28).

### 3.1 Where `planner_delay_us` is applied
- **The field.** `SimulationConfig.planner_delay_us = 0`. Its comment says it models delay from image capture to planner output to the controller (`RT/config.py:312-314@E`; `:340@M`). The wizard default is also 0 (`WZ/base_config.yaml:288@E`; `:333@M`).
- **The buffer.** One `DelayBuffer(planner_delay_us)` is built per rollout (`RT/event_loop.py:232@E`; `:171@M`) and put in the service bundle (`RT/event_loop.py:425@E`).
- **Its one consumer.** `ControllerEvent._run_controller` adds the current reference plan stamped `step_start_us`, then reads `.at(step_start_us)` (`RT/events/controller.py:123-127@E/@M`). No other event references the buffer; a grep of `RT/events/traffic.py`, `physics.py`, `step.py` and `camera.py` finds none.
- **When it runs.**
  - `ControllerEvent` is a `RecurringEvent` with `interval_us = control_timestep_us` (`RT/events/controller.py:24-32`; `RT/events/base.py:139-176`).
  - It is scheduled at the rollout's initial policy timestamp with the same `dt` as the Policy, Physics, Traffic and Step events (`RT/event_loop.py:464-517@E`).
  - At equal timestamps, Policy (priority 20) runs before Controller (40) (`RT/events/base.py:21-50`).
- **No interpolation.** `item_at` pops the head while the *next* item's age is ≥ the delay, then returns a whole stored plan (`RT/delay_buffer.py:35-49@E`).
- **Intermediate poses do not help.** `pose_reporting_interval_us` (default 0, `WZ/base_config.yaml:285@E`) only adds intermediate poses inside the controller service. It never queries the buffer again (`CTL/system.py:164-174`; `RT/events/controller.py:140-141`), so finer reporting does not undo the quantization.
- **Conclusion.** The delayed plan is read only at control ticks, and nothing interpolates between plans. This resolves "check it" at `02:723` and `03:81`, L7's gap (`L7:161-163`) and L7's open question 9 (`L7:545`).

### 3.2 Quantization: known-answer run (the buffer logic re-typed, not imported; `delay_ka.out.txt`)

| Injected delay | Ticks late at Δt = 100 ms | Applied staleness | Distance on the stale plan at 15 m/s: v·d vs what is simulated |
|---|---|---|---|
| 0 ms | 0 | 0 ms | 0 vs 0 m |
| 1–100 ms | 1 | 100 ms | 0.015–1.5 m vs 1.5 m |
| 101–200 ms (139, 171, 200) | 2 | 200 ms | 2.085 / 2.565 / 3.000 m vs 3.0 m for all three |
| 201–300 ms | 3 | 300 ms | 3.015–4.5 m vs 4.5 m |

- **The rule.** Lag = ⌈L/Δt⌉ ticks, so applied staleness = ⌈L/Δt⌉·Δt ≥ L. The simulator **rounds latency up**: 139 ms is simulated as 200 ms (`code`, from the run).
- **Warm-up.** During force-GT, the reference added to the buffer is the ground-truth reference (`RT/events/controller.py:75-93,123`), and `force_gt_duration_us` is 1.7 s (`WZ/base_config.yaml:277@E`). So with a delay, the initial ⌈L/Δt⌉ closed-loop ticks track GT references. In the run, at 139 and 200 ms, ticks 17–18 apply GT15 and GT16. Whether to discard or flag those ticks is a ★ design input for Huy; the note proposes no rule.
- **A finer resolution is not just a config change** (`inferred` from code).
  - Lowering `control_timestep_us` also changes the policy's query rate, because Policy shares `dt` (`RT/event_loop.py:464-480`).
  - Camera clocks are per-camera, and the render factory discards `control_timestep_us` (`RT/events/camera.py:196-210`).
  - `assert_zero_decision_delay: true` (`WZ/base_config.yaml:293@E`) requires fresh frames at every decision. At 50 ms control steps with 10 Hz cameras it would fail.
  - So sub-tick latency needs code: the plan's "trace-driven delay … as an AlpaSim PR" (`02:722`).

### 3.3 Control step and simulation step
- **Control step.** Δt = 100 ms (`RT/config.py:289@E`; `WZ/base_config.yaml:284@E`; `:329@M`).
- **Simulation step = control step.** `n_sim_steps` counts control steps; the end time is the initial policy timestamp + `n_sim_steps`·Δt, clipped with a log line when the scene is too short (`RT/unbound_rollout.py:145-181@E`). That clipping is #208. The default is 200 steps, 20 s (`WZ/base_config.yaml:275@E`).
- **Policy rate.** One `drive()` per control step (`RT/event_loop.py:468-480`).
- **Inside the controller service, each tick:**
  - the MPC uses `dt_mpc = 0.1` s and `n_horizon = 20`, a 2.0 s horizon (`CTL/mpc_controller.py:19-20`; `WZ/controller/default.yaml:1-3@E`);
  - with Δt = `dt_mpc` it solves once per tick, and the command is held over the tick (`CTL/system.py:176-205`);
  - the vehicle model, a dynamic bicycle model that switches to kinematic at low speed (`src/controller/README.md:37-41@E`), integrates by RK4 in substeps of ≤ 10 ms (`CTL/vehicle_model.py:157,162-168`).
- **Which MPC.** `e2e_challenge` defaults to the **nonlinear** MPC (`WZ/base_config.yaml:11@E`; `WZ/controller/nonlinear.yaml:1-5` inherits the default gains); `main` defaults to the **linear** one (`WZ/base_config.yaml:11@M`). S2's manifest already records "the controller and trajectory post-processing identity" (`repo 02:347`).

### 3.4 How a 64-waypoint plan is tracked while the next one waits
- **Stamping.** The driver's plan is the current pose followed by N waypoints stamped `time_now_us + k·(1/output_frequency_hz)` (`DRV/main.py:963,1006-1008@E`). These are absolute simulation times from the tick that produced the plan.
- **Storage.** The runtime stores the plan in the rig frame of the producing tick (`RT/events/controller.py:123-126`).
- **Re-anchoring.** The controller re-expresses the plan in the current rig frame using the ego pose interpolated at the plan's earliest timestamp (`CTL/system.py:332-359`). So a stale plan stays fixed in the world, with odometry compensated. It is **not** re-anchored to the current pose.
- **Sampling.** The MPC samples the plan at the *current* absolute times t + k·0.1 s, k = 0…20, clipped to the plan's range (`CTL/mpc_impl/linear_mpc.py:177-190`; `nonlinear_mpc.py:319-330`). The stale plan is not re-timed; the MPC reads its later waypoints. A 6.4 s plan covers the 2.0 s horizon for any delay under 4.4 s (`inferred`: 6.4 − 2.0).
- **Tracking window.** The tracking cost applies only from horizon index 10, i.e. 1.0 s ahead (`CTL/mpc_controller.py:44,53`; `WZ/controller/default.yaml:11`; `linear_mpc.py:376-380`; `nonlinear_mpc.py:317`). The controller never tries to follow the next 1.0 s of any plan, so latency acts through how stale the plan's *inputs* are, not through tracking lag (`inferred`).
- **Docs.** The controller README says reference commands are "possibly delayed" (`src/controller/README.md:31-35@E`).

### 3.5 `external_static` and lockstep
- **The mode.**
  - The wizard does not launch the driver, and the runtime keeps a driver endpoint (`WZ/driver_source/external_static.yaml:1-12@E`).
  - The address comes from `wizard.external_services.driver` (docs: `docs/OPERATIONS.md:406-411@E`).
  - `external_dynamic` takes addresses per request instead (`WZ/driver_source/external_dynamic.yaml`; `OPERATIONS.md:409-411`).
  - **The challenge's own EC2 competition preset uses `external_static`** (`WZ/e2e_challenge/ec2.yaml:16,27-29@E`).
- **Lockstep.**
  - `PolicyEvent` awaits `driver.drive(time_now_us, time_query_us)` (`RT/events/policy.py:136-140`) before that tick's `ControllerEvent` runs, and the event heap is popped in sequence (`RT/events/base.py:66-101`).
  - The `drive` RPC carries no deadline argument (`RT/services/driver_service.py:262-264`; `RT/telemetry/rpc_wrapper.py:96-125,146-174`); the only timeout found is the 240 s startup timeout (`RT/config.py:349`).
  - So an external driver's wall-clock latency stalls the simulation and never enters simulation time (`code` plus `inferred`).
- **Zero decision delay is enforced.** `assert_zero_decision_delay` checks that the egopose and every camera frame are current at decision time (`RT/events/policy.py:72-74,169-188`).
- **Docs-only claims.** "Real-time execution is a non-goal" comes from DESIGN.md via `repo L7:141`. On-device time has to be logged by the driver itself, for example in `DriveResponse` debug bytes (`repo L7:140,148`; `inferred`).

### 3.6 What this implies (`inferred`)
- **S7.** The constant-delay arms measure the score at 0 / 100 / 200 / 300 … ms of staleness. p50 and p99 inside one band are indistinguishable, and every latency is rounded up. KC-6's 150 ms threshold (`repo 02:711`) falls mid-band.
- **S5(c) at a fixed budget.** At budget B, the curve is score(n | delay = ⌈B/Δt⌉ ticks) over the set {n : latency(n) ≤ B}.
  - The score part can be measured on rented GPUs.
  - Only the membership set needs device latency.
  - Varying n together with an n-dependent delay would be two variables in one experiment.
- **Thor in the loop.** One Jetson serving every call serially spends 264,600 calls × 100–171 ms ≈ 7.3–12.6 h of device time per 441 × 3 sweep (`calc.out.txt`). No stage estimate states this.
- **WAN reachability.** The cloud runtime is the gRPC client, so the Thor must be reachable as a server; NAT or a tunnel is not discussed in the plan (`unverified` operational risk).

---

## 4. Item table: keep, re-cut, merge and drop options, with evidence (unscored)

| Item | Reusable now or already NVIDIA's (dated) | Races NVIDIA? | Stale or wrong in the plan text | Options, with evidence |
|---|---|---|---|---|
| **S5(b)** ladder | PR #45 ships fixed noise, `action-noise.bin` (`L5:55`); FlashDrive velocity reuse; the ODE-error known answer (`03:79`) | No. NVIDIA deploys K = 5 but publishes no closed-loop ladder (`L5:95`) | "Ladders ordered as predicted" (`02:650`) vs only the ODE order being known (`L5:110`) | **Keep**, with the closed-loop order reported, not assumed (C-44, `rep:153`). **Merge**: K is already a fallback lever in S3 (`02:462`) |
| **S5(c)** CoC curve | PR #45: 16 drafted tokens, greedy, ≈ 8.3 accepted (`L5:55`); its engine appends the trajectory-start token when needed (`L5:18`); PR #42 evaluates with a 256-token cap (`L5:128`); TacitVLA endpoints (`ver27:97`) | Partly. The tacit-student line is NVIDIA's (`L1:352`); a budget sweep is a natural NVIDIA follow-up (`inferred`) | "Latency comes from S6" vs the calendar (§7.1); the +2.7% figure and same-checkpoint condition (§7.5); the "3–20 tokens" derivation (§7.9); no injection step (§3.6) | **Keep as written** (lockstep curve) and state the latency set separately. **Re-cut** it as score(n \| fixed delay) plus a device latency map (§3.6). **Merge** with S6/P-019 into one device-backed artifact. **Re-scope** if KC-E finds no runtime switch: the teacher fallback cannot meet §1's budget (§7.11) |
| **S6** build list | PR #36 expert graph (PyTorch, bitwise 160/160); Edge-LLM media-aware KV reuse (v0.10.0), FP8 ViT attention and ViT cache (v0.10.1), NVFP4-A16 Thor kernels (v0.11.0) (`L7:95-97`); ModelOpt `--real-quant` NVFP4/FP8 and R1 QAD (`reported` README @b6b0ff52); TensorRT 10.16 FP8/FP4 on sm_110 (`ver25:82`) | Yes for the ViT, cache and KV-reuse items (`DEC:291`); an Alpamayo 1.5 exporter for Edge-LLM (issue #130 promised; `L7:525`); the SM110 fused gate/up plugin (PR #260 open, unchanged 10-28); PR #45's fork intends to land upstream (`reported` PR #45 body) | The fused flow-step kernel and camera-batched ViT are still unconditional (`02:684-685`) vs P-008 (`DEC:349-361`); streaming reuse appears in both S5 and S6 (`02:631,686`); "~20%" is unlabelled in `01:165` and `04:74` | **Keep** P-008's four items. **Re-cut**: the kernel only if S6a's diagnosis says so (P-007). **Drop**: the vision-side items NVIDIA ships. **Merge**: history reuse into S5 as a model change |
| **S7** HIL and injection | `external_static` (the challenge preset's own path); `DelayBuffer` on both branches; `egodriver.proto`; zero-decision-delay assertion (§3) | No, apart from a possible trace-driven delay PR (none seen upstream, `unverified`) | "Check it" is resolved (§3.1); 2.1 m vs 3.0 m simulated (§7.4); the TOST comparator is unspecified (§7.13) | **Keep** (saving V5, `02:57`). **Re-cut**: the injection resolves only 100-ms bands, so sub-tick work needs an AlpaSim PR (`02:722`). **Merge**: the C++ server is shared with P-019 |
| **S8** shadow mode | HSB 2.6.0 supports JetPack 7.2 on AGX Thor (`L7:144`); Holoscan 5.0 is new (`L7:143`); no NVIDIA shadow-mode trigger tooling found | No. HCMC data is absent from PAI-AV (`02:744`) | HSB stock and rig prices date from 10-25 (`02:760`); P-002 is not listed as blocking S8 (§7.16) | **Keep optional**, cut ahead of everything else (`02:79,761`). **Drop** for a 12-week window: its window starts 01-25 (`02:186`). **Merge** its real-sensor latency with P-009 (ii) |
| **P-006** | — | — | Prices and stock as of 10-25 | **Keep** as the gate. Its decision has no effect inside a window that ends 01-25 unless S8 moves earlier, which would displace S5–S7 |
| **P-007** (S6a on RTX PRO 6000) | PR #36 already removes CPU launch overhead (H1a) on the PyTorch path (`L7:327`) | No (a diagnosis) | "~20%" is derived (`DEC:328-335`) | **Keep and run early**, a few GPU-h. **Merge** into P-019's runtime. Whether vast.ai exposes `ncu` counters is unverified (`L7:544`) |
| **P-008** (narrow S6) | Same as S6 | Reduces the race | — | **Adopt or keep open.** The evidence strengthened on 10-27 (`DEC:291`) and is unchanged on 10-28 (no Edge-LLM commit since 10-20) |
| **P-009** | (iv) partly adopted as the driver-change protocol (`DEC:292`); (ii) maps to `planner_delay_us` | No | (ii) "glass-to-action" scope vs the AlpaSim delay (§7.14) | **Keep** (i)–(iii) for S6–S7. **Re-cut** (ii) into a device-measured term and a simulated term quantized to ticks |
| **P-010** | — | No | — | **Merge** into P-007 as the H2 test (`L7:330`); on its own it is weak (`DEC:293`) |
| **P-015** (batched driver) | — (no Alpamayo adapter batches, `ver27:60`) | No | The saving holds only if non-Alpamayo analogs transfer (`DEC:469`) | **Keep gated.** Its known-null verdict doubles as a validity test of the gate (`L7:467`) |
| **P-018** (few-step expert) | — | Possibly: Edge may already be few-step (`DEC:485`) | — | **Keep conditional** on S5(b)'s cliff and on Edge's release-day facts; KC-L2 applies to the weights |
| **P-019** (one C++ driver, two targets) | Edge-LLM lists x86 SM120 as a development platform (`L7:99`) | No | Depends on P-007; whether PR #45's fork builds for SM120 or Jetson TRT 10.16 is unknown (`L7:312`; `L5:55`) | **Merge point** for paths (1) and (2). Acceptance is the ladder plus a verdict, and a cloud FP8 verdict never stands in for the Thor verdict (`DEC:488`) |
| P-002 (context) | — | — | Lead time and stock unknown (`L7:107,549`) | Must be decided before 12-07 for S6's window (`02:184`) |

---

## 5. Calendar, budget and cut-order constraints for a 12-week capstone from 11-02

**Dates** (`repo 02:177-186,842-853`; `00:49`):

| Date | Event |
|---|---|
| 10-31 | KC-E; the challenge closes |
| 11-02 | KC-1; S2 starts. P-005 is needed before any spend (`02:806`) |
| 11-16 | KC-2; S3 starts |
| 11-30 | KC-3; S4 starts |
| 12-07 | S6 starts (needs P-002) |
| 12-21 | KC-4; S5 starts |
| 01-11 | KC-5 |
| 01-18 | KC-6; S7 starts |
| 01-25 | KC-7; S8 starts; **the 12-week window ends** (`inferred`) |
| 02-08 | S8 ends |

**Hours** (`inferred`, `calc.out.txt`; stage rows `repo 02:179-186`):
- **All [C] work, S1–S8:** 370–404 h, i.e. 12.3–16.2 weeks at 25–30 h/week. This matches `02:188`.
- **S2–S7 [C]:** 306–340 h against 300–360 h of capacity, so the slack is −40 to +54 h.
  - **11-02 to 12-21 (7 weeks):** S2–S4 = 156–176 h against 175–210 h of capacity (−1 to +54 h).
  - **12-21 to 01-25 (5 weeks):** S5–S7 = 150–164 h against 125–150 h (−39 to 0 h). The plan relies on S6 running alongside S4–S5 (`02:188`).
- **The inference-led content already scheduled:**
  - S5 70–78 h (includes S5(b) and S5(c));
  - S6 60–62 h;
  - S7 20–24 h;
  - in all 150–164 h. S6 and S7 alone are 80–86 h.

**What fits without displacing the corrections or the S3/S4 verdicts** (`inferred`):
- **(a) S5–S7 [C] as scheduled.** This needs P-002 by 12-07 and the ≤ $6k tier.
- **(b) Cheap items run early, in parallel:**
  - S3's [C] batch-curve driver: timing-only, $4–13 (`02:417-423,469`);
  - P-007's E-K0: a few GPU-h, after KC-1 and P-005 (`DEC:347`). It is not in the [C] hours, so it uses slack.
- **(c) Anything else** is displacement. The S3/S4 verdicts are protected: "the role items are cut before the S3/S4 verdicts slip" (`02:867`). The corrections are never cut (`02:79`).

**What it would displace, by the cut order.** §0.1, verbatim: "**Cut order if hours run short:** S8 first, then [O] items, then [C] role items in reverse order of the per-stage lists. The corrections are never cut." (`02:79`).
1. **S8, 40 h.** Its window (01-25 → 02-08) lies outside the 12 weeks, so cutting it frees no hours inside the window (`inferred`).
2. **[O] items, up to about 40 h.** These run only if KC-1 allows (`02:79,188`); S5's [O] items are at `02:615-618`.
3. **[C] role items in reverse order.** Two readings:
   - **(R1) Lists reversed.** §0.1's lists run S2, S4, S5, then "S3 and S6–S7, integration and C++" (`02:60-77`). Reversed, the batched student driver, the TensorRT-11-ready C++ and the production C++ driver server go before S5's pipeline and scaling items, and those before S4's and S2's.
   - **(R2) Within each stage.** Each stage's own [C] list is reversed.

   Under R1, path (2)'s C++ artifacts are among the earliest cuts.
4. **The §8 variant** adds "then S7" (`02:872`); D-015 has only §0.1's three steps (`DEC:249`).

**Budget** (`repo 02:792-800,805`):
- **≤ $2.5k:** S2 + S3 + an S4 pilot, ≈ $0.6–2.3k. S5 is excluded, and with it S5(b) and S5(c).
- **≤ $6k:** S2–S5 plus S7's cloud side, ≈ $2.2–6.3k after the expert-graph saving. The upper end exceeds the tier, so the $4k cap rule applies.
- **Hardware:** the Jetson AGX Thor costs $5,499 (P-002); the rig ~$5–8k (P-006).
- **Per path, from plan rows** (`inferred`):
  - path (1): S5 $1,038–2,783, plus S6/S7 if it uses device latency;
  - path (2): $5,499 plus S6 sweeps $102–395 plus S7 ~$200;
  - path (3): rig $5–8k, plus the Jetson.

**Calendar risks:**
- **P-011, the hackathon** (open). It would take about 20–30 h inside 10-26 → 11-04 and could slip KC-1 by 0–3 days (`repo DEC:418-419`), which would push back S2 and the window.
- **P-003, weekly hours,** is unrecorded (`repo DEC:304-305`).

---

## 6. Publication gates per path

**Source facts:**
- **Where the 441 scenes come from.** The 441 are `nurec_curated_val`, the 20% side of an 80/20 split curated from the public **NuRec 26.01 and 26.04** releases. It includes 2,202 of 2,523 public scenes (`code e2e_challenge/local_evaluation/README.md:9-17,31-36@E`).
- **The NuRec card.** It sources its scenes from clips of NVIDIA's Physical AI dataset (26.04: 1,607 clips; "1500+" usdz files) without naming PAI-AV (`reported` [NuRec card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec), read 2026-10-28).
  - That the 441 are NuRec reconstructions of PAI-AV clips is therefore strongly implied (`inferred`), and the plan assumes it (`02:289`; `nano:20`).
- **The licence** (NuRec gate text in the HF API, read 2026-10-28; PAI-AV unchanged in substance per `repo research/2026-10-25-teacher-to-thor/notes/lane_A_models_licenses.md:24`):
  - §1: internal AV development only;
  - §3: confidential information includes "regression or performance data relating to the Dataset";
  - §4.6: no derivative works and no distribution;
  - §8: expires 12 months after initial download.
- **How these licence texts were read.** The NuRec §1/§3/§4.6/§8, OpenMDW-1.1 and NVIDIA Open Model License readings came back through a fetch tool's summary, not raw text. The §3 phrase was returned verbatim as a quotation, and it is the one that decides the gate.
- **The plan's conservative reading** gates "throughput measured on NuRec scenes" too (`repo 05:116`).
- **P-004 status: open, not recorded as sent.**
  - It asks about §3 aggregates, the manifest fallback and §4.6 weights and labels (`repo DEC:306-313`).
  - Day 7 (11-01) says "send the license letter if not yet sent" (`repo 02:778`).
  - No `journal/` directory and no decision entry records it being sent (`repo`, file tree 2026-10-28).
- **Model licences:**
  - **Alpamayo 1.5 weights** are under OpenMDW-1.1 (`repo 04:18`). The licence file imposes no restriction or obligation on how outputs are used (`reported` [HF LICENSE](https://huggingface.co/nvidia/Alpamayo-1.5-10B/blob/main/LICENSE), read 2026-10-28). The 1.5 card also carries a non-commercial line, with commercial licensing on request, which conflicts (`repo ver25:53`).
  - **Cosmos-Reason2-2B** is under the NVIDIA Open Model License (`repo ver27:96`). NVIDIA claims no ownership of outputs, commercial use is allowed, and derivatives must carry the agreement and an attribution, with "Built on NVIDIA Cosmos" for Cosmos models (`reported` [NVIDIA OML](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/), version 2025-10-24, read 2026-10-28).
  - **Edge's licence** is unknown until KC-E (`repo L1:102`).
- **ModelOpt's Alpamayo example** calibrates on 16 AV clips, and its QAD trains the VLM on user-supplied PAI-AV clips, which needs gated access (`reported` README @b6b0ff52).

| Path | Number or artifact | Derived from PAI-AV or NuRec? | Gate |
|---|---|---|---|
| (1) | Every closed-loop score: S3, S5(b), S5(c), S6 G6, S7 injected | Yes (NuRec scenes) | KC-L. The fallback is code, scene-ID manifests and synthetic results (`02:390`), and whether even that is allowed is P-004's open question 2 (`DEC:309-310`) |
| (1) | The latency axis on synthetic inputs (E-K0 uses synthetic inputs, `DEC:337`) | No (`inferred`) | The model licence only; name the device (`02:818`) |
| (1) | Latency logged during HIL on NuRec frames | "Performance data relating to the Dataset" on the conservative reading | KC-L (`05:116`) |
| (1) | S5(a) corrected weights; the budget-randomized expert (C-56); the P-018 expert; any QAD'd checkpoint | Yes (trained on derived states or labels) | KC-L2 |
| (2) | p50/p99/p99.9, thermals, energy on synthetic or own inputs | No (`inferred`) | The model licence; device, JetPack and TensorRT named; "real engine" or "emulated" |
| (2) | **Bitwise identity on non-PAI-AV inputs** | No (`inferred`) | Publishable as a numerics claim about a code path. It carries over to dataset inputs only where kernels and shapes do not depend on the input (fixed-shape graphs); paths that depend on the data (sampling, realized CoC length) need the ladder. It **does not** license publishing any closed-loop score or per-format verdict |
| (2) | Engines quantized with ModelOpt calibration or QAD on PAI-AV clips | Arguably derived (`inferred`, conservative) | KC-L2 if weights or engines are published; calibrating on own data would avoid the question (`inferred`) |
| (2) | Comparisons with FlashRT and PR #45 | Their numbers are public | PR #45's 171 ms is DRIVE AGX Thor and is never a Jetson baseline (`02:828`) |
| (3) | Disagreement rates and clips from own HCMC footage | No data licence involved (`inferred`) | Model licence: OpenMDW-1.1 places no restriction on outputs, against the card's non-commercial line; Edge unknown. Privacy: Vietnam PDPL 91/2025 Art. 32 notice and anonymization before publication (`02:747`; `ver25:129`; engineering reading, not legal advice). Claim wording: open-loop only (`02:819,827`) |
| (3) | Outputs of a student corrected on PAI-AV/NuRec-derived data | Its weights are KC-L2 material | Whether its *outputs* are derivative works is not among P-004's questions (`DEC:306-313`): open |

---

## 7. Inconsistencies, stale numbers and unverified derived numbers (S5–S8)

1. **Where S5(c)'s latency comes from.**
   - The dependency graph draws S5 → S6 (`repo 02:157-158`), while S5(c) says "latency comes from S6" (`repo 02:629`) and A6 needs latency "measured on Jetson AGX Thor (S6)" (`repo nano:25`).
   - KC-5 (2027-01-11) precedes KC-6 (01-18) (`repo 02:849-850`), so A6, an S5 acceptance criterion (`repo 02:648`), cannot close on time. The same step appears in L5-4 step (6) (`repo L5:18`) and C-37 step (6) (`repo rep:130`).
2. **Which checkpoint S6 serves initially is unstated.** S6 starts 12-07, before S5 (12-21) and during S4 (`repo 02:182-184`). If KC-E fails, no 2B weights exist; PR #42 ships none (`repo nano:5`). S6 would then be timing-only, like S3's driver (`repo 02:418`).
3. **The cut order is stated three ways.** §0.1 (`02:79`) and D-015 (`DEC:249`) vs §8 (`02:872`, which adds "then S7"). "Reverse order of the per-stage lists" is ambiguous (§5).
4. **S7's text vs the simulator.**
   - "Check it" (`02:723`; `03:81`) is resolved by code (§3.1).
   - `02:717` says 139 ms "lands 2 ticks late … travels 2.1 m", but AlpaSim simulates 200 ms of staleness, 3.0 m at 15 m/s (§3.2).
   - `03:81`'s 2.085 / 2.565 / 3.000 m are v·d. The simulator applies 3.000 m to all three.
5. **S5(c)'s fixed checkpoint vs its prerequisite.** The card requires the same checkpoint and a fixed architecture (`02:626,630`), but its prerequisite, history-frame prefill reuse (`02:631`), needs expert fine-tuning (`01:165`; `DEC:356-358`; `rep:66`). The quoted +2.7% minADE6 is the fine-tuned figure; without fine-tuning it is 0.96 vs 0.77 (`rev:62`; `rep:66`).
6. **03 §4 S6's "≈ 10.5 tokens" uses 2,560 vision tokens.** That is AR1's 448×280 setting. Alpamayo 1.5's contract is 2,880 (`01:93`; `04:14`), which gives **9.0** (`inferred`: prefill 55.7 ms, `calc.out.txt`). The review prints both (`rev:51-57`), but 03 prints only 10.5 (`03:80`).
7. **The latency floors assume a 0.5B FP8 expert** (`rev:55`; `03:80`). PR #42's compact expert is 28 layers × 384 wide (`L5:21`), about 50M parameters (`inferred`: 12·d²·L). The expert term is likely mis-sized, and its KV-read term is not modelled (`unverified`).
8. **"~20% of the bandwidth bound" is printed as if measured** at `01:165` and `04:74`, and used as an efficiency at `01:178`. P-007 (`DEC:328-335`) and `02:689` label it derived.
9. **The CoC-budget figures drift.** "3–20 tokens" (`02:594`), "about 3" (`01:178`) and "≈ 2.6" (`rev:61`) all come from Alpamayo 1.5 efficiencies on an RTX PRO 6000 (FlashDrive), not from the 2B student on Thor (`unverified` for the student).
10. **Contamination.**
    - `nano:20` says the 441 come from clips labelled `alpasim`.
    - But the curated set has 2,202 scenes (`code` local_evaluation README:16-17,31@E), against 1,536 `alpasim`-labelled clips (`repo ver27:93`).
    - So at least 666 curated scenes must come from clips with another label, or none (`inferred`). That pre-empts S2's contamination report (`02:289`).
11. **The KC-E fallback vs §1's question.** The fallback runs S5(b)–(c) on the 10B teacher (`02:639`; `nano:46`). §1's question fixes an on-device latency (`02:118,125`) that no 10B configuration meets on Jetson (`01:176-177`; `04:173`).
12. **S6 G6's per-format closed-loop verdicts** (`02:149,688`) need either the device in the loop (S7's `external_static`) or a cloud engine, which cannot stand in (`DEC:488`). S6's verdict cost is priced as RTX PRO 6000 sweeps (`02:712`); the 7–13 h of device time per sweep (§3.6) appears nowhere.
13. **S7's equivalence test has no stated comparator.** S7 compares the Thor-in-loop score with "the cloud score" (`02:735`). If the cloud arm is the BF16 PyTorch driver, the TOST mixes lockstep semantics with engine numerics (`inferred`). The design is ★ Huy's.
14. **P-009 (ii)'s scope vs `planner_delay_us`.**
    - P-009 (ii) runs from the "oldest frame's exposure" until the "command takes effect" (`DEC:365-366`). `planner_delay_us` covers capture → planner output → controller (`code RT/config.py:312-314@E`), and D-015 equates the two (`DEC:292`).
    - Actuation sits in the vehicle model, not in the delay.
    - With a 4-frame, 0.4 s history (`01:93`), "oldest frame" could add about 300 ms of history by definition (`inferred`). The definition is ★ Huy's.
15. **S7's hours.** V5's −4 to −8 h (`02:57`) sits beside S7 growing from 20 to 20–24 h (`02:185`). The C++ server (+6–10 h, `L7:47`) is "built during S6" (`02:695,724`), while S6 grows only +0–2 h (`02:184`), so where those hours are counted is unclear.
16. **S8 needs the Jetson** (`02:751`), but §9 lists P-002 as blocking only S6–S7 (`02:884`).
17. **S8's role evidence is stated two ways:** "weak (D2)" (`02:761`; `05:129`) vs "carries no JR2010863 line" (`rep:384`). The live documents win.
18. **Unqualified superlatives.** Novelty superlatives at `02:169,399,584` lack the release-day qualifier that `02:472` carries; §6 allows them only after a dated check (`02:832`).
19. **The camera contract.** The owner targets 4 cameras (`02:131`). Edge's camera contract is unstated, and AD-Memo's 2B Alpamayo 2 is single-camera-focused (`L1:92,99`). Unverified until KC-E.
20. **KC-6's 150 ms threshold** (`02:711`) falls in the middle of AlpaSim's (100, 200] ms band (§3.6).
21. **S6's build list** still lists the fused kernel and camera-batched ViT unconditionally (`02:684-685`), while P-008 is open (`DEC:349-361`). `02:685` already says to consume NVIDIA's items, so P-008 is partly applied.
22. **Running PR #45 on Jetson would need porting.** The fork targets TensorRT 11.2/11.3 on DRIVE (`L5:55`) and Jetson runs 10.16.2 (`04:52`). S6's Why (`02:677`) does not note it (`inferred` gap).
23. **ModelOpt's Alpamayo example targets R1, not 1.5.** G6 cites "ModelOpt PTQ including the expert (0.47 current)" (`02:149`). The example is R1-only, and its QAD trains only the VLM (`reported` README @b6b0ff52), which settles L7's inference (`L7:180`). No QAD path for 1.5 or the 2B student ships.

---

## 8. Source list

**Repo** (read 2026-10-28):
- `DOC_AUTHORITY.md`, `CLAUDE.md`, `docs/00_START_HERE.md`, `01`, `02` (all), `03`, `04`, `05`, `DEC` (all), `nano`;
- `research/2026-10-27-role-alignment/{report.md, verification.md, notes/L1, L5, L7}`;
- `research/2026-10-25-teacher-to-thor/{verification.md, review.md, notes/lane_A_models_licenses.md}`.

**Code at commits** (local clones; heads re-confirmed 2026-10-28):
- **AlpaSim @E / @M**, via the `RT/`, `CTL/`, `WZ/` and `DRV/` prefixes defined at the top:
  - `RT/delay_buffer.py`, `RT/event_loop.py`, `RT/config.py`, `RT/unbound_rollout.py`;
  - `RT/events/{base,policy,controller,camera}.py`;
  - `RT/services/driver_service.py`, `RT/telemetry/rpc_wrapper.py`;
  - `CTL/{system,mpc_controller,vehicle_model}.py`, `CTL/mpc_impl/{linear_mpc,nonlinear_mpc}.py`;
  - `src/controller/README.md`, `DRV/main.py`;
  - `WZ/{base_config.yaml, controller/*.yaml, driver_source/*.yaml, e2e_challenge/ec2.yaml}`;
  - `e2e_challenge/local_evaluation/README.md`, `docs/OPERATIONS.md`.
- **ModelOpt** `examples/alpamayo/README.md` @ `b6b0ff52`, through the GitHub MCP.

**Upstream, read-only, all on 2026-10-28:**
- **GitHub commits:**
  - AlpaSim: no commit since 10-20 on `e2e_challenge` (head `e121e591`) or since 08-18 on `main` (head `affc2eab`);
  - TensorRT-Edge-LLM `main`: last commit 10-20 (CI);
  - alpamayo1.5: last commit 09-09 (`36aeb4c5`, #38);
  - FlashRT: commits on 10-27 (GR00T, X-VLA, AMD), none on Alpamayo.
- **Pull requests:**
  - [Edge-LLM PR #260](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/260): open, blocked, updated 10-21;
  - [recipes PR #45](https://github.com/NVlabs/alpamayo-recipes/pull/45): open, blocked, updated 10-21. Its body says "~5Hz on Thor" and uses a fork while its PR lands in Edge-LLM.
- **Hugging Face:**
  - [HF models API](https://huggingface.co/api/models?author=nvidia&search=alpamayo): three models (R1-10B, 1.5-10B, 2-Super) and no Edge;
  - [NuRec dataset API](https://huggingface.co/api/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec): gate text §1/§3/§4.6/§8, lastModified 2026-07-30;
  - [NuRec card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec): clips from the Physical AI dataset (1,607, 26.04); the raw README is gated (401);
  - [Alpamayo-1.5-10B LICENSE](https://huggingface.co/nvidia/Alpamayo-1.5-10B/blob/main/LICENSE): OpenMDW-1.1.
- **NVIDIA:** [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/), version 2025-10-24.

**Scratch:** `/private/tmp/claude-501/-Users-danghuyhoang-Desktop-physloop/6a1c1084-894a-4933-a900-90a396bc7621/scratchpad/capstone_scratch/E/{delay_ka.py, delay_ka.out.txt, calc.py, calc.out.txt}`.

**Gaps:**
- Edge's runtime-CoC switch, licence and camera contract: the KC-E check on 10-31.
- Jetson dev-kit lead time to Vietnam.
- Whether the PR #45 fork builds against TensorRT 10.16 or for SM120.
- Whether vast.ai exposes `ncu` counters.
- Whether the outputs of PAI-AV-derived weights count as derivative works.
- What the PAI-AV `alpamayo_aligned` labels are for the 441 source clips (S2 measures this).

**Suspicious content:** none found. No fetched page or file in this track contained text addressed to an AI.
