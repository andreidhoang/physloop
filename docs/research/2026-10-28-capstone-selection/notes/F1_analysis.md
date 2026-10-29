# F1 — PhysLoop capstone selection: core analysis (as of 2026-10-28)

**Status.** Draft for the red team. Analysis date is the brief's, 2026-10-28; the web novelty checks and the two fact re-checks in §4.4 and §9 ran on 2026-10-29 (UTC+7) and are dated so. Nothing in this document is measured by PhysLoop. Labels: **reported** [ID] (a source states it), **inferred** (arithmetic in `capstone_scratch/F/arith_F.py`, output `arith_F.out.txt`, rows F0–F11), **unverified** (what was tried is stated). Citations: [A-…] Track A, [B-…] Track B, [C-…] Track C, [D-…] Track D, [E-§…] Track E, repo paths as `02 §3`, `L7 2.3`, `DEC P-007`. Every prediction, tolerance, margin, latency-model, kernel-design, precision-verdict, trigger-logic and diagnosis slot is Huy's (★) and is left as `____`. Jetson AGX Thor and DRIVE AGX Thor are always named; no Jetson number is a DRIVE number. "First" is never claimed; novelty is "no public X found on <date> (searched: …)".

---

## §0 One-page verdict

**#1 (primary; FOCUS path 1, path 2 supplying its x-axis): "Score per tick" — the tick-quantized closed-loop cost of on-device latency for a trajectory-output driving VLA under plan-tracking control, paired and stratified with CIs on AlpaSim, with the deployment levers (CoC budget n, flow steps K, precision) read off the same curve.** It re-cuts S7's injection into the cloud from week 2 and folds S5(b)/(c) in as lever arms. AlpaSim applies a delayed plan once per control tick, so 139, 171 and 200 ms are all simulated as 200 ms of staleness (3.0 m at 15 m/s), not 2.085 m [E-§3.2; F1]. Every published AlpaSim score is lockstep and scores latency as zero [E-§3.5]; NVIDIA ran a 200 ms planner delay until 2026-03-24 [L7 2.3]. No public latency-injection study on AlpaSim/NuRec, or on a plan-tracked VLA, with replicates and CIs, was found on 2026-10-28 or 10-29 (§4.4). At measured Jetson proxies the no-CoC 2B student sits near 90 ms, a CoC token costs 7.35 ms and an expert step ≤ 1.7 ms at bound, so n = 2…14 share one 2-tick band [A-Q1.10; F3]: the CoC budget, not the expert kernel, decides the tick. Cheapest decisive experiment: 100 scenes × 3 replicates × {1, 2 ticks} paired to S2's 0-delay arm, $15–56 (DR1) [F5]. Serves JR2010863 X2, D3, I2, D5; JR2025038 (2), (3), (5).

**#2: a bitwise-invariant batched AlpaSim driver (P-015 re-cut).** Two arms: invariant kernels with per-session generators, accepted only if B = k is bitwise identical to B = 1; and the non-invariant batch under P-009's ladder plus one known-null paired verdict that also tests the gate. Net 14–54 % per sweep after the invariance tax [F7]; branch-proof; one sweep ($34–124); hits I2-perf, X3, D2, D3, D5. No such study exists for flow policies in closed loop [C-44].

**#3: the Jetson AGX Thor measurement artifact plus one C++ driver (S6, P-008, P-019 merged).** Per-stage p50/p99/p99.9, glass-to-action, J/tick, a 30-minute sustained run at pinned clocks, real engines labelled, and the C++ `EgodriverService` (golden bitwise replay, sanitizers) that serves S7's HIL and #1's device point. None is public for any Alpamayo on either Thor [A-§4.1 #6–7]. Not a kernel win: the Jetson teacher expert is at 66–79 % of bound [A-Q4.6], no 2B weights or Jetson 2B baseline exist [A-§4.1 #1–2], and FlashRT plans the behaviour-preserving 1.5 Jetson reference [B-R03]. Needs P-002; the x86 half starts first.

**Why #1.** #2 is cheaper and branch-proof but is about the simulator, not the car. #3 needs a $5,499 purchase with an unverified lead time and no like-for-like baseline until Edge ships. #1 is decisive on one rented GPU, legible in one chart (score per stratum vs ticks of staleness, device p99 band overlaid), and alone answers whether a speed-up made the car drive worse, by construction. Equal-weight scores 42 / 40 / 33 [F10].

**Branches (KC-E 2026-10-31 × P-002).**

| | Thor bought | Not bought |
|---|---|---|
| **Edge public, fine-tunable** | #1 on the student (n, K, precision arms), Jetson device point, HIL on a 100-scene subset (≈ 2.9 h device time, taking PR #45's DRIVE 171 ms only as an illustrative stand-in; the Jetson value is #3's measurement [F5]); #2 on the student driver; #3 full | #1 as left, device point "not measured on Thor"; #2 unchanged; #3 = x86 runtime + labelled latency model |
| **Edge not public / not evaluable** | #1 on the 10B teacher (ticks 0–3 (+5); K, CoC-cap, FP8 arms), Jetson point at 9 ticks [F4]; #2 unchanged; #3 = methodology contributed to FlashRT; buy only when a 2B student exists (P-024) | #1 on the teacher, cloud only; #2 unchanged; #3 = x86 C++ runtime, "not measured on Thor"; P-002 deferred |

Edge public but not fine-tunable: #1 keeps every inference-time lever, S5(a) dies. Edge tacit: the n arm is gone; the curve is ticks × {K, precision}.

---

## §1 Candidate pool and scoring

### 1.1 The six tensions, reconciled first (they set several scores)

| # | Tension | Winner and why | Consequence |
|---|---|---|---|
| 1 | Expert ≈ 20 % of bound (10-03 ledger, FlashDrive on RTX PRO 6000, BF16, 8 steps; P-007's premise) vs 66–79 % (FlashRT FP8 expert on Jetson) | **Both inferred; for the Thor question Track A wins** (stage times on the target device; one clip, seed 0, FlashRT's own kernels, MAXN locked clocks [A-Q4.3, Q4.6]). The 20 % is the ledger's own arithmetic on a different device, precision and kernel path [DEC P-007; F0] | P-007 becomes a sweep-cost diagnosis on sm_120, not a Thor lever; the 2B's expert step is ≤ 1.7 ms at bound [F3], so no expert kernel moves a tick |
| 2 | 139 ms ⇒ 2.085 m (S7 card, `03` §4) vs 139 ms simulated as 200 ms ⇒ 3.0 m (code) | **Track E wins**: `RT/events/controller.py:123-127` reads the delayed plan once per control tick, `delay_buffer.py:35-49` does not interpolate, identical on both branches [E-§3.1] | The curve's x-axis is applied staleness in ticks; 2.085 m is the physical distance during 139 ms, 3.0 m is what AlpaSim applies; KC-6's 150 ms sits mid-band |
| 3 | C11's ≈ 4.1 ms/token NVFP4 bound vs Edge-LLM's 7.35 ms/token measured | **Both hold**: 4.12 is 2.0B at 4.5 bpp with no head split and no KV; with KV and the head the bound is 4.4–6.4 ms; 7.35 is 56–88 % of bound depending on the unstated LM-head precision [A-Q1.10, A-X.2; F2] | Plan with the measured figure (labelled: Qwen3-VL-2B proxy, power mode unstated); quote bounds as "at 100 % of peak" |
| 4 | S6 plans to consume Edge-LLM's media-aware KV reuse vs the support matrix | **Track A wins**: the matrix rejects an action runner together with context reuse and bypasses reuse for image-bearing speculative requests [A-Q1.5] | Streaming reuse on the Alpamayo path is unavailable upstream *and* model-changing (needs expert fine-tuning [C-20]); drop it from S6 |
| 5 | S5(c) "same checkpoint" vs its history-frame-reuse prerequisite | **Track E wins** [E-§7.5]: the +2.7 % minADE6 is the fine-tuned figure; without fine-tuning it is 0.96 vs 0.77 m | Run the CoC arms at full prefill on one checkpoint; reuse becomes a separate model-change arm (KC-L2 if weights are published), as P-008 already moves it |
| 6 | Prices: vast.ai 10-28 vs `02` §5 (10-27) | **Track D wins** (dated raw snapshot, 16:15 UTC [D-DR1]): RTX PRO 6000 WS $1.00 / $1.47, Server $1.24 / $1.38, H100 SXM $1.93 / $3.14 (+31 % median in a day), B200 $7.50 / $8.75 | Every cost below is at DR1; 15 of 35 WS offers cap at 350–550 W, so the manifest records the power cap [D-F6.2] |

### 1.2 Pool (20 candidates; existing items marked ●, new ○; "new" items carry a dated search in §4.4 / §5)

| ID | Candidate | Path | Existing item |
|---|---|---|---|
| K1 | Score-per-tick latency-cost curve (S7 injection re-cut; S5(b)/(c) as lever arms) | 1 (+2 for the axis) | ● S7, S5(b), S5(c) |
| K2 | Bitwise-invariant batched AlpaSim driver, two arms | 2-sim (SCOPE 3) | ● P-015 |
| K3 | Jetson AGX Thor measurement artifact + one C++ driver | 2 | ● S6 items, P-008, P-019 |
| K4 | S5(c) CoC-budget curve as written (lockstep, Edge, reuse prerequisite) | 1 | ● |
| K5 | S5(b) flow-step ladder as written | 1 | ● |
| K6 | S6 as written: 2B student at 10 Hz on Jetson with a behaviour-preserving win | 2 | ● |
| K7 | S6 G6 per-format closed-loop verdict on real Jetson engines | 1/2 | ● |
| K8 | P-007 expert-efficiency diagnosis (E-K0 on RTX PRO 6000) | 2 | ● |
| K9 | P-018 few-step expert distillation | 1 | ● |
| K10 | P-019 one C++ driver for two targets, as written | 2 | ● |
| K11 | S7 as written (constant arms + HIL + C++ server; needs P-002) | 1/2 | ● |
| K12 | S8 HCMC shadow-mode OOD set | 3 | ● |
| K13 | sm_110 flow-expert kernel (tcgen05/TMEM-resident accumulators across Euler steps) | 2 | ○ |
| K14 | AlpaSim upstream PR: sub-tick / trace-driven `DelayBuffer` + command-FIFO option | 1 (MISSION a) | ○ |
| K15 | Per-lever paired closed-loop re-run of FlashDrive's public MIT bundle on Alpamayo 1.5 | 1 | ○ |
| K16 | DART visual-token pruning closed-loop verdict | 1 | ○ |
| K17 | Edge-LLM action-runner + KV-reuse runtime | 2 | ○ |
| K18 | Proxy-to-Thor transfer study (why an sm_120 fix does not transfer to 20-SM sm_110) | 1/2 | ○ |
| K19 | Speculative-CoC paired closed-loop verdict on the student | 1 | ○ |
| K20 | Previous-tick warm start of the flow sampler | 1 | ○ |

### 1.3 Scores (1–5; criteria: AV relevance, Gap, Decisive at small scale, Win vs strongest baseline, Behaviour preservation, Legibility, Upstreamability, Role evidence, Cost/TTFS). Totals from `arith_F.py` F10. Equal weights decide; the weighted variant (Gap ×2; Small, Behaviour, Role, Cost ×1.5) is a robustness check and keeps the same top three after merges.

| ID | AV | Gap | Small | Win | Beh | Leg | Up | Role | Cost | Equal | Wtd |
|---|---|---|---|---|---|---|---|---|---|---|---|
| K1 | 5 | 4 | 5 | 4 | 5 | 5 | 4 | 5 | 5 | **42** | 56.0 |
| K2 | 4 | 4 | 5 | 4 | 5 | 4 | 5 | 5 | 4 | **40** | 53.5 |
| K14 | 4 | 4 | 5 | 3 | 5 | 4 | 5 | 4 | 5 | 39 | 52.5 |
| K11 | 5 | 4 | 3 | 3 | 5 | 5 | 4 | 5 | 3 | 37 | 49.0 |
| K5 | 4 | 3 | 5 | 3 | 5 | 4 | 3 | 3 | 5 | 35 | 47.0 |
| K4 | 5 | 4 | 3 | 3 | 4 | 5 | 3 | 4 | 3 | 34 | 45.0 |
| K15 | 4 | 3 | 5 | 3 | 4 | 4 | 3 | 4 | 4 | 34 | 45.5 |
| K3 | 4 | 4 | 3 | 3 | 5 | 4 | 3 | 4 | 3 | **33** | 44.5 |
| K7 | 4 | 4 | 2 | 3 | 5 | 4 | 3 | 4 | 2 | 31 | 41.5 |
| K8 | 3 | 2 | 4 | 3 | 5 | 3 | 3 | 3 | 5 | 31 | 41.5 |
| K16 | 3 | 3 | 4 | 3 | 3 | 4 | 3 | 3 | 4 | 30 | 40.0 |
| K10 | 4 | 3 | 3 | 3 | 3 | 3 | 3 | 4 | 3 | 29 | 38.5 |
| K18 | 3 | 4 | 2 | 3 | 5 | 4 | 3 | 3 | 2 | 29 | 39.0 |
| K13 | 4 | 3 | 2 | 2 | 4 | 3 | 3 | 3 | 2 | 26 | 34.5 |
| K19 | 4 | 2 | 3 | 3 | 3 | 3 | 2 | 3 | 3 | 26 | 34.0 |
| K6 | 5 | 2 | 2 | 2 | 3 | 4 | 2 | 3 | 2 | 25 | 32.0 |
| K9 | 3 | 2 | 3 | 3 | 2 | 3 | 2 | 3 | 3 | 24 | 31.5 |
| K12 | 3 | 4 | 1 | 2 | 5 | 3 | 2 | 2 | 1 | 23 | 31.5 |
| K20 | 3 | 3 | 3 | 2 | 2 | 3 | 2 | 2 | 3 | 23 | 31.0 |
| K17 | 4 | 1 | 2 | 2 | 3 | 3 | 1 | 3 | 2 | 21 | 27.0 |

**Merges, read from the plan's own text.** K11 and K14 are #1's mechanism and its upstream deliverable (S7 Build: "a per-step, trace-driven delay is added as an AlpaSim PR only if Huy's rule says the constant arms are insufficient", `02` S7). K4 and K5 are #1's lever arms (S5(c): "latency comes from S6", `02` S5; E-§3.6: the latency enters the score only as ⌈L/Δt⌉). K15 is #1's lever set on the KC-E-fails branch (S3's fallback: "deployment levers on the released teacher", `02` S3). K10 and K7's device half fold into K3 (S6 Build: "the C++ driver server for S7, built here"; P-019: the S6a runtime "serves AlpaSim's student sweeps"), with K8 as K3's week-1 step. After merges the order is K1 > K2 > K3 under both weightings. Every item the brief required is scored on its own row before any merge: K11 is S7 as written (37), K4 and K5 are S5(c) and S5(b) as written, K10 is P-019 as written, K6 is S6 as written, K8 is P-007, K9 is P-018, K12 is S8; K2's non-invariant arm is P-015 as written, and its row scores identically except Behaviour 5 → 3 (38 equal-weight).

### 1.4 Reasons and source IDs per score (one line per criterion)

- **K1.** AV 5: the 100 ms deadline and PR #45's 171 ms = 2 ticks vs a 10 Hz announcement [A-Q2.2, A-Q7.1]. Gap 4: no AlpaSim/NuRec latency curve with CIs [C-35, C D10]; NVIDIA could publish (K6) but no artifact is announced. Small 5: `planner_delay_us` is a config value on both branches [E-§3.1]; pilot $15–56 [F5]. Win 4: vs lockstep 0-delay reporting (FlashDrive, TacitVLA, PR #45 [A-§4.3 e]) and Bench2Drive-Robust (3 levels, no CIs, command delay [C-30]). Beh 5: one checkpoint, nothing changed; the expert graph is bitwise [L7 4.8]. Leg 5: score per stratum vs ticks with the device band. Up 4: AlpaSim takes outside PRs [B-C08]; KC-L gates the numbers [E-§6]. Role 5: X2, D3, I2-accuracy, D5 (`05` §3); JR2025038 (2),(3),(5) [B-J02]. Cost 5: first signal week 2–3.
- **K2.** AV 4: simulation throughput is SCOPE 3, not the vehicle. Gap 4: no batch-invariance study for flow policies in closed loop [C-44]; PR #26's per-row generators exist upstream, unmerged [A-Q3.3]. Small 5: one equivalence sweep [DEC P-015]. Win 4: s ≈ 2.0–2.6 Amdahl-bounded [L7 4 Q4]; net 14–54 % with the invariance tax [F7]. Beh 5: bitwise arm by construction; known-null verdict for the other. Leg 4: GPU-h per rollout vs B with the verdict flag. Up 5: AlpaSim driver `predict_batch` override, no adapter batches today [L7 4.4]; consume PR #26, never duplicate it. Role 5: I2-perf, X3, D2, D3, D5 [`05` §3]. Cost 4: $34–124 + 16–22 h [F9].
- **K3.** AV 4: deployment measurement; JR2014515 (4) methodology [B-J03]. Gap 4: no public glass-to-action, sustained, thermal or J/tick for Alpamayo on either Thor [A-§4.1 #6–7]; no NVIDIA Jetson latency for any Alpamayo [B-R09]. Small 3: the x86 SM120 half runs now (Edge-LLM "Developer" platform [L7 1.11]); the device half needs P-002, and no Thor is rentable [D-F6.7]. Win 3: methodology beyond FlashRT's model-time [A-Q4.2]; no 2B Jetson baseline [E-§1.4]. Beh 5: golden bitwise replay; engines via the ladder. Leg 4: per-stage p99.9 + glass-to-action timeline. Up 3: FlashRT merges in days [B-R03]; Edge-LLM 0/33 outside merges [B-C02]. Role 4: Q4 Full-if, X3, I2-perf, D5; JR2025038 (2)–(4), JR2016777 (1),(4). Cost 3: $5,499 + lead time [D-F6.8].
- **K4.** AV 5: the §1 research question [`02` §1]. Gap 4: no cap sweep on a fixed driving checkpoint [C D11]; TacitVLA endpoints are NVIDIA's [C-19]. Small 3: needs a student with a decodable CoC (KC-E) and conflicts with its own prerequisite [E-§7.5]. Win 3: vs TacitVLA's three unpaired endpoints. Beh 4: caps on one checkpoint; realized length confounds [`02` S5]. Leg 5. Up 3: KC-L, Edge licence. Role 4: I1, X2, D3. Cost 3: inside S5's $272–1,055.
- **K5.** AV 4: K = 5 deployed [A-Q2.2]. Gap 3: raced (K = 5 shipped; driving planners at 1–2 steps; Edge's K unknown) [C K4]. Small 5: paired by noise file, +2 sweeps. Win 3: the closed-loop order is the result, no baseline [E-§4]. Beh 5. Leg 4. Up 3: KC-L. Role 3: X3, R7 ladder. Cost 5.
- **K6.** AV 5. Gap 2: FlashRT's planned official-1.5 Jetson reference [B-R03]; Edge-LLM's 1.5 support promised and NVIDIA-owned [B-E01]; PR #45's C++ runtime "planned" [A-Q2.2]. Small 2: no 2B weights [A-Q7.3], no Thor rental [D-F6.7]. Win 2: no like-for-like Jetson 2B baseline [E-§1.4]. Beh 3: ladder needed for engines. Leg 4. Up 2: Edge-LLM merges none [B-C02]. Role 3: Q4/X3; JR2010863 has no on-device line [B-J01]. Cost 2.
- **K7.** AV 4. Gap 4: no driving-VLA closed-loop per-format verdict [C D14]. Small 2: device in the loop costs 7–13 h per sweep [E-§3.6]; Edge-LLM exports Alpamayo FP16 only [A-Q1.7]; NVFP4 on sm_110 is fragile [B-E04]. Win 3. Beh 5. Leg 4. Up 3. Role 4: X2, D3, Q4. Cost 2.
- **K8.** AV 3: sm_120 sweep cost. Gap 2: FlashRT publishes per-phase kernel tables [A-Q4.3]; the Thor expert is near bound [A-Q4.6]. Small 4: a few GPU-h [DEC P-007]. Win 3: any fix is Amdahl-bounded at 11–25 % [L7 4 Q4]. Beh 5: graphs are bitwise. Leg 3. Up 3: alpamayo1.5 accepts contributor PRs [B-C04]. Role 3: X3, I2-perf. Cost 5.
- **K9.** AV 3. Gap 2: K = 5 deployed, Edge may be few-step, MISTY 1-step, GR00T one-step −38 pp [C-28, C K4]. Small 3. Win 3. Beh 2: changes the model; KC-L2. Leg 3. Up 2. Role 3. Cost 3.
- **K10.** AV 4. Gap 3: PR #45's C++ runtime planned [A-Q2.2]. Small 3: x86 SM120 build; equivalence sweep; needs P-007. Win 3: unmeasured [DEC P-019]. Beh 3: ladder + verdict; a cloud FP8 verdict never stands in for Thor [DEC P-019]. Leg 3. Up 3. Role 4: Q4, D5, X3, I2. Cost 3.
- **K11.** As K1 except Small 3 and Cost 3 (HIL needs P-002; 7–13 h device time per sweep [E-§3.6]).
- **K12.** AV 3: real distribution shift matters, but open-loop only [`01` C12]. Gap 4: PAI-AV has no Asian country [A8 in ver25]. Small 1: rig $5–8k with the 9295 prices unverified, the deserializer PSU backordered and HSB stock unverified [D-F7.1–F7.4]; P-002 implicit [E-§7.16]; cross-border dossier under Decree 356 Arts. 17(1)(c), 18, 19 [D-F8.3–F8.5]. Win 2: no baseline. Beh 5. Leg 3. Up 2. Role 2: D2 weak [`02` S8]; JR2025038 (6). Cost 1: window starts 01-25 [E-§5].
- **K13.** AV 4: JR2014515 (3) flow-matching decoders on SoCs [B-J03]. Gap 3: no public sm_110 flow-expert kernel [D-F5.10], but CAKE automation ships SM110 decode routes weekly [B-K03–K05] and PR #260 holds the gate/up plugin. Small 2: Thor not rentable; B200 matches the ISA but has 3.7× the bandwidth and 148 SMs [D-F4.9]. Win 2: Jetson teacher expert at 66–79 % of bound on a 14 % stage (126 of 897 ms) [A-Q4.3]; the 2B expert step ≤ 1.7 ms [F3]. Beh 4: bitwise vs reference or the ladder. Leg 3. Up 3: FlashInfer wants before/after numbers [B-C06]. Role 3. Cost 2.
- **K14.** AV 4. Gap 4: sub-tick resolution needs code [E-§3.2]; no such PR exists [E-§4 S7]. Small 5: CPU-testable + 100-scene arms. Win 3: an instrument change. Beh 5. Leg 4. Up 5: AlpaSim accepts outside PRs; not #208's field [B-C08, ver27 V21]. Role 4: I2 "improve the simulation", D3, X2. Cost 5.
- **K15.** AV 4. Gap 3: FlashDrive's closed-loop table is bundle-only [C-26]; z-lab/FlashRT could do it. Small 5: MIT code and weights public [C-01]. Win 3. Beh 4: streaming uses fine-tuned weights (a different checkpoint). Leg 4. Up 3. Role 4. Cost 4.
- **K16.** AV 3. Gap 3: Edge-LLM ships DART with Thor speed-ups on a 4B VLM; no closed-loop verdict [A-Q1.11]. Small 4. Win 3. Beh 3. Leg 4. Up 3. Role 3. Cost 4.
- **K17.** Gap 1: the support matrix rejects it [A-Q1.5]; it races NVIDIA; Up 1: 0 of 33 outside PRs merged [B-C02]. Killed (§2).
- **K18.** Gap 4: documented nowhere [D-§5 path 1]. Small 2: every Thor number is a model without P-002 [D-F6.7]. Role 3: JR2025038 (3) dev-vs-target discrepancy [B-J02]. Folds into #3's report.
- **K19.** Gap 2: crowded; NVIDIA ships drafters (PR #43) and Domino in PR #45 [C-09, C K2]. Beh 3: exact acceptance preserves the distribution in exact arithmetic, not bitwise [C D2]. Killed.
- **K20.** Gap 3: none in driving [C-29] but crowded in robotics (RTC, SEAM). Beh 2: behaviour-changing with no closed-loop precedent. Killed.

---

## §2 Kills

Applied in the brief's order: scoop risk first, then the six kill tests.

| Candidate | Kill reason | Evidence |
|---|---|---|
| K6 S6 as written (2B at 10 Hz on Jetson with a behaviour-preserving win) | **Scoop + no baseline + no model.** FlashRT's maintainer stated (10-12) a plan for an official-1.5, behaviour-preserving Jetson Thor reference; Edge-LLM's Alpamayo 1.5 support is promised and NVIDIA-owned; PR #45's C++ runtime is "planned"; no 2B weights exist and no Jetson 2B number exists to beat. What survives is the measurement artifact (#3) | [B-R03], [B-E01], [A-Q2.2], [A-Q7.3], [E-§1.4] |
| K17 Edge-LLM action-runner + KV reuse | **Races NVIDIA and is rejected upstream.** The support matrix rejects the action runner with context reuse; Edge-LLM has merged 0 of 33 outside PRs since 08-01 | [A-Q1.5], [B-C02] |
| K19 Speculative-CoC paired verdict | **Scoop.** NVIDIA ships Domino in PR #45 and drafters in PR #43; ≥ 15 speculative-VLA papers; at most a paired verdict remains, which #1's n arm subsumes | [A-Q2.2], [C-09], [C-10b] |
| K20 Previous-tick warm start | **Behaviour-changing with a crowded robotics literature** (RTC, SEAM) and no AlpaSim closed-loop precedent; needs the full ladder for a lever whose value is unquantified | [C-03], [C-29] |
| K9 P-018 few-step expert | **Raced (K4) and out of window.** K = 5 is deployed; Edge may be few-step; a one-step head cut LIBERO-Long to 26 %; weights are KC-L2 material. Kept conditional, outside the 12 weeks (P-026) | [A-Q2.2], [C-28i], [DEC P-018] |
| K12 S8 shadow mode | **Out of window and gated three ways.** Its window starts 01-25, the day the 12 weeks end; P-006 and P-002 are unrecorded; Decree 356 Art. 17(1)(c) counts processing on foreign GPU hosts as a transfer (Art. 18 dossier within 60 days, no research exemption); 9295 camera prices unverified, deserializer PSU backordered. Deferred (P-027), never "validation" | [E-§5], [D-F8.3–F8.5], [D-F7.1–F7.4] |
| K13 sm_110 flow-expert kernel, standalone | **Non-target device without a transfer argument** until P-002: not rentable; B200 shares the ISA but not the bandwidth or SM count; on Jetson the teacher's expert is already at 66–79 % of bound on a 14 % stage, and the 2B expert step is ≤ 1.7 ms at bound, so the headroom is at most a few ms per tick. Kept only as a conditional ★ item inside #3 if the diagnosis says so (P-023) | [D-F6.7], [D-F4.9], [A-Q4.6], [F3] |
| K16 DART closed-loop verdict | **Weak lever, shipped upstream, model-dependent.** Edge-LLM already ships DART; its speed-ups are on Qwen3-VL-4B VQA benchmarks; it needs a student and a KC-L number. Not a capstone | [A-Q1.11] |
| Any Jetson comparison against the official eager BF16 build (3,322 ms) | **Weak baseline.** The 10B baseline is FlashRT `fast` (811 ms, real kernels, MAXN locked clocks); the 2B baseline is PR #45's configuration ported to Jetson, labelled; "emulated" rows never count as a speed-up | [A-§4.3 a], [`02` §6] |
| Any score derived from the FP8 ModelOpt PyTorch checkpoint presented as a Thor speed-up | **Fake-quant.** It is emulated and slower on Thor; it is a policy row, not a lever | [B-R04], [B-F01], [L7 §5] |
| Any real-car claim beyond open-loop shadow mode | Forbidden by the plan and the brief | [`02` §6] |
| S6 vision-side items (camera-batched ViT, FP8 ViT, ViT cache, streaming reuse) | **Race**: Edge-LLM ships FP8 ViT attention, the embedding cache and media-aware KV reuse; reuse needs fine-tuning anyway (tensions 4–5) | [A-Q1.1], [A-Q1.5], [C-20] |

Kept, not killed: K8 (P-007) as a week-1 diagnostic inside #3; K7's device half inside #3 on a scene subset; K18 as a section of #3's report; K14, K15, K4, K5, K11 inside #1.

---

## §3 FOCUS-path scorecard

Scores from `arith_F.py` F11 (same nine criteria).

| Path | Equal | Wtd | Proves to an NVIDIA AV hiring manager (`05` §3 line IDs; other JRs from [B-J02–J04]) | Cannot prove | Cost and gates | Time to first signal |
|---|---|---|---|---|---|---|
| **(1) Research: the tick-cost curve with lever arms** | 42 | 56.0 | JR2010863 **X2** (closed-loop evaluation of E2E models: a paired, stratified latency verdict), **D3** (measure simulation quality: lockstep hides latency; staleness is a simulator knob NVIDIA once set to 200 ms), **I2-accuracy** (the delay-buffer semantics and warm-up contamination, upstreamed), **D5** (the student architecture served through the batched driver), **X3** via the lever arms. JR2025038 (2) timing contracts, (3) why behaviour differs between dev and target, (5) driving-quality regression metrics. JR2014515 (3) latency feasibility of flow decoders, partly | On-device latency itself (that is #3); real-world validity; C++ depth unless #3's server runs; anything public before KC-L (every closed-loop score is NuRec-derived [E-§6]) | Sweeps only: $6–23 smoke, $15–56 pilot, $102–373 full curve, lever arms inside S5's $272–1,055 [F5]. Gates: P-005, KC-L (publication), KC-E (student arms only), none for the teacher | Week 2–3 (pilot after S2's pinned protocol) |
| **(2) Engineering on Jetson AGX Thor, as the measurement artifact** | 30 | 39.0 | JR2010863 **Q4** (C++: `EgodriverService`, golden bitwise replay, sanitizers, p99.9), **X3**, **I2-perf**, **D5**. JR2025038 (2),(3),(4),(7); JR2014515 (1)–(5): operator-level review, kernel traces, SoC feasibility of flow decoders and multi-camera tokenizers, MLPerf-Edge-style methodology, TensorRT on Jetson; JR2016777 (1),(4) | A win over a like-for-like baseline (none public for a 2B on Jetson; FlashRT owns the 10B); DRIVE numbers; that the served student is NVIDIA's (no weights); the 10 Hz claim if p99 > 100 ms | $5,499 + import/lead time ____ (unverified; 30–45 days quoted by one AU reseller [D-F6.8]) + $10–30 x86 dev + optional $75–88 B200 ISA check [F8]; device-in-loop sweeps on subsets. Gates: P-002, P-005, P-009 (i)(iii), KC-E for a student, KC-L if NuRec frames feed latency logs | x86 half: 2–3 weeks; device: ≥ 4–8 weeks after P-002 |
| **(3) Shadow mode in HCMC (open-loop)** | 23 | 31.5 | JR2010863 **D2** (weak: a data engine); JR2025038 (6) in-vehicle data analysis; JR2022179 data flywheel | Driving quality, closed-loop effects, validation (covariate shift invisible [`01` C12]); any "on-road testing" wording | Rig $5–8k (partly unverified), Jetson, two legal dossiers (Decree 356 Arts. 18, 19) and a privacy plan; gates P-006, P-002, KC-8; model-licence terms for outputs; Edge's licence unknown | ≥ 12 weeks (window 01-25 → 02-08) |

**Recommendation.** Primary: path (1). Order: (1) → (2) → (3). Path (2)'s row (30) scores the device-gated path as a whole (Small 2, Cost 2); K3 (33) counts its x86 half, which runs before P-002. Path (2) is kept as the artifact that supplies (1)'s x-axis and the Q4 evidence, and it starts on x86 SM120 before P-002. Path (3) stays optional and outside the window.

**Combinations verified from the plan's text.** One C++ driver serves both (1) and (2): `02` S6 Build says the C++ driver server for S7 is "built here", and P-019 says the S6a runtime "serves AlpaSim's student sweeps through `driver_source=external_static`" [DEC P-019]. The batched driver (#2) is also (1)'s sweep engine: `02` S3 says "the same driver is S5's collection driver". No other combination is claimed.

---

## §4 The #1 deep dive: "score per tick"

### 4.1 The production incident

An OEM's OTA candidate is NVIDIA's 2B student. NVIDIA's own 2B students run at 171.2 ms median in FP8 on DRIVE AGX Thor (DRIVE OS 7.0.5, TensorRT 11.2, B = 1, 6 candidates × 64 steps, K = 5, greedy Domino with 8.30 accepted tokens per round, a 3,006-row prefill every request, timer starting at token handoff) [A-Q2.2, Q2.4], while Alpamayo 2 Edge is announced at 10 Hz for the same device with no camera, reasoning or precision configuration [A-Q7.1]. At Δt = 100 ms, 171 ms is 2 ticks late and would be simulated as 200 ms of plan staleness [F1]. Every closed-loop score anyone has published for an Alpamayo-family model, including the challenge leaderboard and TacitVLA, was produced in lockstep: AlpaSim's runtime awaits `drive()` with no deadline, so wall-clock latency stalls simulation time and never enters the score [E-§3.5]. NVIDIA's legacy Alpamayo configs ran a 200 ms planner delay until 2026-03-24, when the default became 0 and the file was removed [L7 2.3]. So the release question "does the deployable model drive worse, and where" is answered today at a latency the car will never have. The incident is a model that passes the gate at 0 ms and lands 2 ticks late on the road.

### 4.2 Why now

- Edge's weights are due by 10-31 (KC-E) [A-Q7.1]; whether they ship or not, the curve runs on the released 1.5 (S3's fallback levers) and the instrument is the same.
- AlpaSim already contains the mechanism (`planner_delay_us`, `DelayBuffer`, external-driver mode) on both branches, so the experiment is a config value plus a CPU test, not a fork [E-§3.1, L7 2.1–2.4b].
- The challenge closes 10-31 and NVIDIA ties the withheld simulation choices to its next release [ver27 V7]; S2 pins the protocol right after.
- Bench2Drive-Robust (05-18) showed that the strongest clean CARLA model is the most delay-sensitive (SimLingo DS 85.94 → 28.45 at 100 ms of command delay) with no CIs and no replication [C-30, C-30i]; nobody has tested the plan-plus-MPC interface that AlpaSim uses.
- FlashDrive's code and weights are public (MIT), so the teacher's deployment levers are reproducible [C-01].
- The Jetson proxy numbers (Qwen3-VL-2B at 7.35 ms/token) put the 2B student's CoC budget on the tick boundary [F3].

### 4.3 Falsifiable hypothesis, mechanism and prediction slots

**Mechanism (derived from code).** The driver's plan is 64 waypoints stamped at absolute simulation times; the runtime stores it in the producing tick's rig frame; the controller re-anchors it with odometry, so a stale plan stays fixed in the world; the MPC samples the plan at the *current* absolute times and its tracking cost starts at horizon index 10, i.e. 1.0 s ahead (`CTL/mpc_controller.py:53`); a 6.4 s plan covers the 2.0 s horizon for any delay under 4.4 s [E-§3.4]. So staleness does not act through tracking lag. It acts through stale *inputs*: a plan computed d·Δt earlier reacts d·Δt later to anything that changed in the scene. Loss therefore concentrates where the world changes inside the staleness window, in dynamic-actor strata (a lead vehicle braking or stationary in path, a cut-in, a crossing actor), and should be near zero in geometry strata (lane keeping, curvature) until the delay exceeds the horizon's slack. Bench2Drive-Robust delays the command instead, which loses state feedback for τ; the two perturbations are not equivalent [C-31].

**Hypothesis H1.** On AlpaSim's plan-tracking interface, the paired per-tick loss in `scene_score` is a decreasing step function of applied staleness with a stratum-dependent knee: ≈ 0 within the replicate noise floor on geometry strata through 2 ticks, and above the floor on dynamic-actor strata from 1 tick. **H2.** The loss per tick on the plan interface is smaller than on a command-FIFO interface at matched applied staleness (the within-AlpaSim version of C-31). **H3.** At a fixed tick band, the CoC budget n inside the band buys score (or does not): score(n | d = ⌈L(n)/Δt⌉) is non-decreasing in n within a band and drops at the band boundary. **H4.** Flow steps K within a band do not change the verdict (the ODE-error order is known; the closed-loop order is the result [`03` §4 S5]).

**Prediction slots (★ Huy, written before the run):** loss at d = 1 / 2 / 3 ticks, overall ____ / ____ / ____ pp; on the stationary-lead stratum ____; on geometry strata ____; plan vs command-FIFO difference at d = 2 ____; curve shape over n at d = 2 ____; the K below which the score drops at fixed d ____; the non-inferiority margin per stratum ____; the force-GT warm-up handling rule (ticks 17–18 apply GT references at d = 2 [E-§3.2]) ____. A result that beats its prediction is investigated as a bug first (`CLAUDE.md`).

### 4.4 Dated novelty check

"No public latency-injection study on AlpaSim or NuRec, or for a trajectory-output VLA whose stale plan is tracked by a controller, with replicates and CIs, found on 2026-10-28 (Track C, arXiv API: `abs:latency AND abs:closed-loop AND abs:driving` [77 hits, 38 read]; `(inference delay | control delay | action delay | latency-aware) AND (driving | CARLA | nuPlan | NAVSIM)` [21]) [C-35, C D10]; re-checked 2026-10-29 by web search (`AlpaSim "planner_delay_us" OR "planner delay" latency injection closed-loop driving score Alpamayo`; `inference latency delayed trajectory tracking MPC vs delayed control command closed-loop end-to-end driving simulation staleness tick 2026`), which returned no AlpaSim latency study." Closest neighbours: Bench2Drive-Robust (CARLA, command-delay FIFO, 100/200/500 ms, 220 routes, no repeats or CIs; repo Thinklab-SJTU/Bench2Drive-Robust) [C-30]; "Think at 5 Hz, Act at 20 Hz" (CARLA, freshness ablation 82.1 → 94.0 RC) [C-08]; arXiv 2601.07393 (CARLA framework that skips decision updates and holds the last command when inference overruns a step; web 10-29, abstract-level, unverified); RTC's delay-in-controller-steps framing (robotics) [C-03, A-Q8.8]; NVIDIA's own removed 200 ms planner-delay config [L7 2.3]; FlashDrive's unpaired 100-clip closed-loop table [A-Q5.6]. NVIDIA authors are active on adjacent questions (TacitVLA 10-13, "Perceives but Doesn't Brake" 10-20) [C K6], so the novelty wording stays "no public … found on <date>" and is re-checked on release day.

### 4.5 Per-stage roofline, bytes and FLOPs per tick (inferred, at 100 % of peak; device constants from Track D; shapes from the repo and notes)

Device: Jetson AGX Thor T5000, 273 GB/s shared LPDDR5X [D-DT1, DT7], dense FP8 517 / BF16 258.5 / FP4 1,035 TFLOPS at MAXN [D-DT2], ×0.880 in the default 120 W mode [D-DT3], 20 SMs and 32 MiB L2 (user runtime query) [D-DT9]. Request: 4 cams × 4 frames at 576×320, newest frame encoded, 3,006 prompt rows, 6 × 64 action rows, K = 5 [A-Q2.2]; 180 vision tokens per image [`04` §2]. Student: Qwen3-VL-2B backbone proxy (1.409e9 non-embedding + 0.311e9 embedding, 28 layers, 8 KV heads × 128) [A-X.2]; compact expert 28 layers × 384 wide ≈ 0.116e9 params, cross-attending the VLM KV at 114,688 B/token ⇒ 0.345 GB for 3,006 rows [L5 1.3, 1.3i]. All rows: `arith_F.py` F2–F3.

| Stage | Bytes per tick | FLOPs per tick | Bound (MAXN; 120 W) | Measured proxy on Jetson (reported; not an Alpamayo student; power mode unstated) |
|---|---|---|---|---|
| ViT, newest 4 frames (720 tokens) | ViT weights (size not in the notes) + activations | not derivable from the notes | — | 31.0 ms by linear scaling of Edge-LLM's 265-token 11.4 ms (FP16 ViT) [A-Q1.10]; all 16 frames: 123.9 ms |
| Prefill, 3,006 rows | weights 0.79 GB NVFP4; KV write 0.345 GB | 8.47 TFLOP linear + 2.07 TFLOP attention | FP8 20.4 ms; 23.2 ms at 120 W; NVFP4-linear 12.2 ms | 41.1 ms (28.0 ms per 2,048 rows, linear) [A-Q1.10] |
| Decode, per CoC token | 1.12–1.76 GB (body 0.79 GB at 4.5 bpp; head 0.175 NVFP4 or 0.622 FP16; KV 0.235–0.345) | 2.8 GFLOP | 4.12–6.44 ms/token | 7.35 ms/token (136 tok/s at past KV 2,048) ⇒ 56–88 % of bound depending on head precision [A-Q1.10, A-X.2] |
| Expert, per Euler step (M = 384 rows) | 0.116 GB FP8 weights + 0.172–0.345 GB KV read | 0.35 TFLOP | 1.06–1.69 ms/step (FP8 w); K = 5: 5.3–8.4 ms; weights at 111 MiB exceed the 32 MiB L2, so they are re-read every step | none for the compact expert; teacher's FP8 expert on Jetson 12.6 ms/step at 66–79 % of bound [A-Q4.3, Q4.6] |
| Host | — | — | — | ≈ 10 ms feeding the paged pool (PR #45, DRIVE) [A-Q2.2] |

### 4.6 The 100 ms budget and ⌈L/Δt⌉

L(n) = ViT + prefill + n·decode + K·expert + host. Compositions [F3]:

| Composition | L0 at n = 0 | per token | L crosses 100 ms at n ≈ | crosses 200 ms at n ≈ | n = 16 |
|---|---|---|---|---|---|
| Bound: ViT cached, prefill FP8 bound, expert K = 5 bound, host 10 | 38.8 ms (1 tick) | 4.81 | 12.7 | 33.5 | 116 ms, 2 ticks |
| Proxy: ViT newest-4, prefill proxy, expert bound, host 10 | 90.5 ms (1 tick) | 7.35 | 1.3 | 14.9 | 208 ms, 3 ticks |
| Proxy with expert at 2× bound | 98.9 ms (1 tick) | 7.35 | 0.1 | 13.7 | 217 ms, 3 ticks |
| Proxy with all 16 frames encoded | 183.4 ms (2 ticks) | 7.35 | 0 | 2.3 | 301 ms, 4 ticks |

Three readings. (i) The marginal cost of a CoC token (7.35 ms measured; 4.1–6.4 bound) is 4–7× the marginal cost of an expert step (≤ 1.7 ms bound), so n decides the tick band and K or an expert kernel do not; the "expert-efficiency diagnosis" is the wrong lever for the 2B. (ii) The tick boundary n* lies anywhere in [1, 13] depending on ViT caching, prefill efficiency and host overhead; that is a ★ latency-model question and the reason #3 exists. (iii) Within a band, n is free in tick terms: at the proxy row, n = 2…14 all cost 2 ticks, so "what does reasoning buy" must be asked at a fixed d, which is H3. PR #45's 171 ms on DRIVE needs a 71 ms (41.6 %) cut to reach 1 tick [F3]; its incumbent counter-lever is Domino speculation, which preserves the greedy output in exact arithmetic, not bitwise [C D2]. For the 10B teacher on Jetson the levers move whole ticks only between 8 and 9 (FlashRT `fast` 811 ms; no-CoC ≈ 478 ms = 5 ticks; K = 5 ≈ 748 ms = 8 ticks) [F4], which is why the teacher branch of #1 reports the curve far from the deployment regime and says so.

**Tick quantization, grounded in code (Track E).** `SimulationConfig.planner_delay_us = 0` and the wizard default 0 (`RT/config.py:312-314`, `WZ/base_config.yaml:288`); one `DelayBuffer` per rollout (`RT/event_loop.py:232`); its only consumer is `ControllerEvent._run_controller`, which adds the current plan stamped `step_start_us` and reads `.at(step_start_us)` once per control tick (`RT/events/controller.py:123-127`, a `RecurringEvent` with `interval_us = control_timestep_us`); `item_at` pops while the next item's age ≥ delay and returns a whole plan, no interpolation (`RT/delay_buffer.py:35-49`); `pose_reporting_interval_us` never re-queries the buffer; Δt = 100 ms (`RT/config.py:289`); `assert_zero_decision_delay: true` would fail at 50 ms control steps with 10 Hz cameras, so sub-tick resolution needs code [E-§3.1–3.3]. Applied staleness is ⌈L/Δt⌉·Δt ≥ L [F1]. During the 1.7 s force-GT warm-up the buffer holds GT references, so at d = 2 the first two closed-loop ticks apply GT15 and GT16 [E-§3.2]; whether to discard or flag them is a ★ rule. Bench2Drive-Robust uses ⌊τ·20/1000⌋ at 20 Hz, so its "100 ms" is 100 ms applied and AlpaSim's "139 ms" is 200 ms applied: comparisons are made in applied-staleness seconds, never nominal milliseconds [F1].

### 4.7 Baselines

- **The 0-delay lockstep score of the same checkpoint on the same pinned protocol** (S2's arm): the comparator every published number implicitly uses [E-§3.5].
- **Bench2Drive-Robust** (CARLA, command delay, 4 policies, 220 routes, no CIs): the strongest published latency→closed-loop result; compared in applied seconds and by interface, never as a number to beat [C-30].
- **FlashDrive's unpaired 100-clip AlpaSim table** (collision 0.19 → 0.15, wrong-lane 0.45 → 0.51; no replicates, pairing, commit or GPU) as the quality-protocol baseline the curve improves on [A-Q5.6, C-47].
- **TacitVLA's three unpaired endpoints** (46.4 / 52.3 / 52.5 on 441 scenes, ± SD over 3 rollouts) for the n arm [C-19, ver27 V8].
- For the device point: FlashRT `fast` on Jetson (811 ms) for the 10B; PR #45's configuration for the 2B, labelled DRIVE until ported [A-§4.3].

### 4.8 Metrics

Produced by #1 itself: paired Δ `scene_score` per stratum per tick with a one-sided 95 % bound, cluster-adjusted by road, against Huy's margin ____; INSUFFICIENT where power is lacking (it blocks GO); ≥ 4 m exits tallied separately per arm; both score metrics (challenge `scene_score`; card-style km-between-events as a diagnostic); the per-tick regression flags with Holm; replicate test–retest beside every Δ; the realized CoC length per tick for the n arm; the deviation-bin validity record per rollout; GPU-hours per rollout from the mean wall clock [`02` S2]. Supplied by #3 (or labelled "not measured on Thor"): per-stage p50/p99/p99.9, deadline-miss rate at 100 ms, glass-to-action, J/tick, % of bound per stage, the P-009 ladder rows for any engine used. MDE at 441 scenes, one-sided, 80 % power: 3.55 pp at σ_d = 0.3 and 5.92 pp at 0.5; 3.04 / 5.07 pp with 3 replicates at ρ = 0.6 [F6]. Seed mode is recorded; repeats are unseeded replicates unless the manifest says otherwise [`02` §0.1 K1].

### 4.9 Experiment plan (one variable per experiment; the prediction file precedes every run)

| ID | Variable | Fixed | Arms | Size and cost (DR1) | Decision rule |
|---|---|---|---|---|---|
| E0 | none (CPU) | — | `DelayBuffer` known-answer tests: ⌈L/Δt⌉ table [F1]; the warm-up GT ticks; an AlpaSim issue with the findings | $0 | The tests reproduce E-§3.2 or the mechanism is wrong |
| E1 smoke | applied staleness d | checkpoint (1.5 BF16, expert graph on, bitwise-tested), protocol P, 20 scenes × 3 replicates | d ∈ {0, 1, 2, 3} ticks | 6–15 GPU-h, $6–23 [F5] | Harness works; confirms that `planner_delay_us > 0` coexists with `assert_zero_decision_delay: true` on `e2e_challenge` (unverified until E1; the two act at different events, policy vs controller [E-§3.1], and NVIDIA's 200 ms config ran [L7 2.3]); distributions plotted |
| E2 pilot (the cheapest decisive experiment; it needs KC-2's pinned protocol and noise floor, so it starts in the back half of week 2 on S2's partial 0-delay arm if S2 pins early, else in week 3) | d | as E1, 100 scenes × 3 replicates, paired to S2's 0-delay arm | d ∈ {1, 2} | 15–38 GPU-h, $15–56 | Go to E3 if the paired loss at d = 2 exceeds the replicate floor on any pre-registered stratum (MDE ≈ 6–12 pp at n = 100); if not, add d ∈ {3, 5} on the same 100 scenes before judging; a null through d = 5 is reported as a latency-tolerance result |
| E3 full curve | d | 441 scenes × 3 replicates | d ∈ {1, 2, 3} (+5) | 102–254 GPU-h, $102–373 (91–219 GPU-h with the graph) | The paper's headline; interim looks only by S1's group-sequential rule |
| E4 interface | plan-buffer vs command-FIFO (ã_t = a_{t−τ} on the MPC's outputs) at matched applied staleness; command-hold only as an optional third arm (its mean staleness is (d − 1)/2 ticks) | d = 2; 100 scenes | 2 | ≈ 15–38 GPU-h | Needs K14's AlpaSim change; tests H2 |
| E5 sub-tick | L within a band | the K14 trace-driven/sub-tick mode; 100 scenes | L ∈ {139, 171} vs 200 | ≈ 15–38 GPU-h | Only if Huy's rule says the constant arms are insufficient (`02` S7) |
| E6 n arm | CoC cap n at fixed d | student (or teacher) checkpoint; d = ⌈L(n)/Δt⌉ pinned per n from #3's map or the roofline, labelled | n ∈ {0, 4, 8, 16} inside one band + the boundary pair | inside S5's $272–1,055 | H3; natural length measured first; n = 0 labelled by architecture [`02` S5(c)] |
| E7 K arm | K at fixed d | common noise file per scene and replicate | K ∈ {10, 5, 4, 2} + velocity-reuse schedule | +2 sweeps | H4; closed-loop order reported, not assumed |
| E8 precision | BF16 vs FP8 (emulated on sm_120; real engine only on Jetson) at fixed d | S2's rows | 2 | S2 | A policy row, never a speed-up |
| E9 device point | none | #3's measured L on Jetson for the chosen configuration | 1 | #3 | Places the real system on the curve; HIL on a 100-scene subset (≈ 2.9 h device time if the per-call time were PR #45's DRIVE 171 ms, an illustrative stand-in and never a Jetson number; the Jetson value is T3's) with TOST against the cloud arm (Huy's margin) |

Ablation discipline: E4 and E5 change the instrument and are reported as instrument results; E6–E8 change the model's deployment setting and are reported at a fixed d; nothing is combined before each single-variable arm has run.

---

## §5 #2 and #3 in brief

### 5.1 #2 — the bitwise-invariant batched AlpaSim driver (P-015 re-cut)

- **Production incident.** The stock Alpamayo driver every S2–S5 sweep runs is eager PyTorch at `max_batch_size: 1`; no Alpamayo adapter overrides `predict_batch`; NVIDIA's own 8-GPU evaluation topology gives the driver 7 of 8 GPUs [L7 4.3–4.4, 4.11]. The policy is 73–87.5 % of each rollout step by inference, so 26–27 sweeps cost 884–2,295 RTX PRO 6000 GPU-h [L7 §1]. Every GPU-hour of #1's curve goes through this driver.
- **Why now.** NVIDIA's bitwise expert graph (PR #36) is merged but never enabled in AlpaSim [L7 4.7–4.10]; per-row generators for batched flow sampling exist as an open NVIDIA PR (alpamayo1.5 #26) [A-Q3.3]; vLLM's batch-invariant mode validates exactly the Qwen3-VL-2B/4B backbone family [C-38]; the cost of invariance is now published (34 % SGLang, 61.5 % Thinking Machines, > 2× CoRun) [C-36i, C-37, C-39]; vLLM-Omni's two-stage Alpamayo path is datacenter-only, BF16, eager [B-R02].
- **Hypothesis.** H5: a batched driver with per-session generators and batch-invariant GEMM/RMSNorm/attention reduction order gives outputs bitwise identical to the B = 1 path on every tick (E0a passes at B = k), at an invariance tax of ____× on the driver and a net sweep saving of ____ %; H6: the non-invariant batch passes the P-009 ladder and the known-null paired verdict within Huy's margin ____ (if the gate flags it, either the batching or the gate is wrong, which is itself a D3 result). Amdahl brackets: 36.5–53.8 % per sweep without the tax; 13.9–42.4 % with a 1.34–1.62× tax [F7].
- **Novelty.** "No public determinism or batch-invariance study for VLA or flow-matching action policies in closed-loop simulation found on 2026-10-28 (Track C arXiv API: batch-invariant / batch invariance / deterministic inference / nondeterminism in LLM [69 hits, 30 read]) [C-44]; re-checked 2026-10-29 (web: `"batch-invariant" OR "batch invariant" OR "bitwise" batched rollout flow matching action expert closed-loop simulation vision-language-action driving`), nothing found." Closest: vLLM batch invariance (LLM serving) [C-38]; PhyAI's cross-environment batching (throughput only) [C-02]; alpamayo1.5 PR #26 [A-Q3.3]; PR #36's 160/160 bitwise precedent [L7 4.8].
- **Roofline.** At B = 1 on RTX PRO 6000 the teacher's decode (8.7 ms/token bound) and expert (2.79 ms/step bound) are memory-bound; at B = 32 decode is 0.51 ms per sample per token and the expert crosses B* ≈ 4.4 into its ≈ 7 ms compute floor; prefill (~229–252 ms) stays per-sample, so the per-sample bound is ≈ 253–276 ms against 513–717 ms, s ≈ 2.0–2.6 [`04` §4, L7 4 Q4]. For the student architecture the knee is a ★ prediction (`02` S3 batch curve).
- **Baselines.** The B = 1 eager driver with the expert graph on (bitwise, the strongest behaviour-preserving baseline); FlashDrive's system-optimized path (1.397×) as a non-batched reference [A-Q5.4]; published batching analogues PhyAI 2.26× and GR00T 2.84× [L7 4.14].
- **Metrics.** E0a bitwise equality count per tick at B ∈ {1, 2, 4, 8, 16}; GPU-hours per rollout and rollouts per GPU-hour (stabilized, mean wall clock) [`02` S2]; the invariance tax as a ratio; p50/p99 per call and VRAM vs B; the known-null verdict (GO/NO-GO/INSUFFICIENT) for the non-invariant arm; the gate's flag rate under the null.
- **Experiment plan.** B1 timing-only batch curve on the PR #42 architecture and 1.5 FP8 (4–8 GPU-h, $4–13) [`02` S3]; B2 invariant arm: bitwise test on a 20-scene subset at B = k vs B = 1, synthetic inputs first (publishable now), then NuRec inputs; B3 non-invariant arm: ladder E0a–E1 then one equivalence sweep ($34–124) [F7]; B4 adopt the faster passing arm for #1's E3 and the S3–S5 sweeps. Upstream: a driver `predict_batch` override behind a flag, filed after re-checking NVIDIA's post-10-31 sync; a comment on PR #26 with the bitwise results rather than a competing PR.
- **Cannot prove.** Anything about the car; a speed-up for FP8 rows (a different policy); multi-node scale.

### 5.2 #3 — the Jetson AGX Thor measurement artifact and one C++ driver (S6, P-008, P-019 merged)

- **Production incident.** The only NVIDIA 2B number is DRIVE-only and measures model time from token handoff [A-Q2.4]; FlashRT reports model time with host video decode outside it, at MAXN locked clocks [A-Q4.2]; nobody publishes glass-to-action, a sustained run, thermals or energy for any Alpamayo model on either Thor [A-§4.1 #6–7]; the default power mode runs the GPU at 88 % of MAXN, MAXN is "experimental" and throttles, and over-current throttling has been seen under multi-image VLM load [D-F1.12, F2.7]. A 10 Hz claim measured for 20 requests at MAXN is not a deployment claim.
- **Why now.** TensorRT 11 removes weak typing and deprecates `IPluginV2DynamicExt`; Jetson's 11.x is announced and undated [A-Q6.2, L7 1.4–1.5]; Edge-LLM lists x86 SM120 as a developer platform with TRT 10/11 wheels, so the C++ can start before the device [L7 1.11]; FlashRT merges Thor ports in 1–3 days and lacks exactly this methodology [B-R03].
- **Hypothesis.** H7: for the served configuration, glass-to-action exceeds model time by ____ ms and the 30-minute p99 at the default 120 W mode exceeds the 20-request MAXN p99 by ____ ms; H8: the C++ `EgodriverService` reproduces the Python path bitwise on a golden replay (E0a) and runs with zero sanitizer findings; H9: J/tick at 10 Hz sustained is ____ J against a 12–13 J upper bound at 120–130 W module power [F8].
- **Novelty.** "No public glass-to-action, 30-minute sustained, thermal or J/tick figure for any Alpamayo model on Jetson AGX Thor or DRIVE AGX Thor found on 2026-10-28 [A-§4.1 #6–7]; re-checked 2026-10-29 (web: `"glass-to-action" OR "glass to action" OR "camera-to-trajectory" latency Jetson Thor driving VLA Alpamayo`), nothing found." Closest: FlashRT's model-time per window [A-Q4.2]; ZETIC's Jetson VLA dashboard (p50, tegrastats power, no TensorRT, no Alpamayo) [B-F08]; NVIDIA's GR00T N1.7 and π0.5 Thor tutorials (medians of 20 runs, MAXN) [A-Q8.6, Q8.7]; the MLPerf Edge Agentic methodology [B-F09].
- **Roofline.** §4.5's table is the device half of this artifact; the x86 half's numbers are "not Thor" by ISA (sm_120 `mma.sync`, no `tcgen05`; ridge 558 vs 1,894 FLOP/B) [D-F3.9–F3.11, F4.2]. The proxy-transfer section (K18) explains why a fix on 188 SMs may regress on 20 [D-F4.1, F4.3].
- **Baselines.** FlashRT `fast`/`balanced` on Jetson for the 10B [A-Q4.3]; PR #45's configuration ported to Jetson for the 2B, labelled; Edge-LLM's Qwen3-VL-2B rows as the backbone proxy [A-Q1.10]; never the eager BF16 build.
- **Metrics.** Per-stage p50/p99/p99.9 (CUDA events) and glass-to-action (host monotonic clock: oldest-frame exposure → command available; the definition is ★); deadline-miss rate at 100 ms; J/tick from tegrastats; 30-minute sustained histogram with clocks and temperatures logged; % of bound per stage; the P-009 ladder for every engine; golden bitwise replay; sanitizer and clang-tidy results; TRT 10.16 and 11.x CI.
- **Experiment plan.** T1 (week 1, x86): P-007's E-K0 re-cut as a sweep-cost diagnosis with graphs on/off (a few GPU-h) [DEC P-007 note]; T2 (weeks 6–9, x86 SM120): the C++ runtime and `EgodriverService` against the Python driver (ladder with injected reference `x_0`, as Edge-LLM's pi0.5 parity does [A-Q1.8]), TRT-11-ready; T3 (if P-002): engines on Jetson, labelled real; T4: the measurement protocol at 120 W default and MAXN pinned with `jetson_clocks`, plus P-009's contention arm; T5: HIL on a 100-scene subset for #1's E9. A kernel is written only if T1/T3's profile puts one stage above its bound by a margin Huy sets ____ (P-023).
- **Cannot prove.** A win over a 2B Jetson baseline (none public); DRIVE numbers; the 10 Hz claim if p99 > 100 ms (then report 5 Hz honestly, `02` KC-6).

---

## §6 Stage-gated 12-week plan (Monday 2026-11-02 → Monday 2027-01-25)

**Frame.** The plan's own calendar is kept: S2 11-02→11-16 (KC-2), S3 11-16→11-30 (KC-3), S4 11-30→12-21 (KC-4), S6 12-07→01-18 (KC-6), S5 12-21→01-11 (KC-5), S7 01-18→01-25 (KC-7); S8 starts 01-25, outside the window [`02` §3, E-§5]. Capacity 300–360 h at the unconfirmed 25–30 h/week vs 306–340 h of S2–S7 [C] work: slack −40 to +54 h [F9]. The capstone adds no stage (D-015 binding); it re-cuts items inside S2–S7. All prices DR1 (vast.ai 2026-10-28); every GPU-hour is re-priced at each stage start. Gates respected: no S2 run before 10-31 (S2 starts 11-02); no PAI-AV/NuRec download before S2 (first download 11-02 at the earliest; the 12-month clock starts then); no spend before P-005 (needed on 11-02); nothing on a car before P-006 (S8 is outside the window). P-004 (the licence letter) goes out in week 1 (Huy's action, $0): every closed-loop number below is NuRec-derived and KC-L-gated.

| Week | Dates | Stage | Milestones (capstone items in bold) | GPU type × hours × $ (DR1) | Go / kill |
|---|---|---|---|---|---|
| 1 | 11-02 → 11-08 | S1 close, S2 start | KC-1; P-005 recorded; P-004 sent; KC-E result captured (release-day list); **P-002 decided under P-024's trigger (order now if a public, evaluable 2B student exists; at the quoted 30–45-day lead time [D-F6.8] only an order by 12-07 still lands the device before T4)**; S2 pre-registration (branch, seed mode, Harmonizer, run method); re-read NVIDIA's post-10-31 AlpaSim release; **E0: `DelayBuffer` CPU known-answer tests + AlpaSim issue on tick quantization and warm-up GT ticks**; **T1: P-007 E-K0 sweep-cost profile, graphs on/off** | RTX PRO 6000 WS 4–8 h × $1.00–1.47 = $4–12 (E-K0, Harmonizer pilot) | KC-1: `verdict()` holds false-GO ≤ α or no GPU spend. KC-E branch chosen (§0 table) |
| 2 | 11-09 → 11-15 | S2 | S2 sweeps (TransFuser; 1.5 BF16; 1.5 FP8 emulated row), expert-graph bitwise test, rollout profile, DCGM, noise floor; **E1 smoke: 20 scenes × 3 × {0,1,2,3 ticks}; E2 pilot starts in the back half of the week on S2's partial 0-delay arm if the protocol is already pinned** | S2: 136–340 GPU-h × $1.00–1.47 = $136–500 (+ storage $40–216/mo); **E1 6–15 GPU-h, $6–23** | KC-2 (11-16): reliability ≥ floor; noise floor below the effects. E1: all arms run; distributions plotted |
| 3 | 11-16 → 11-22 | S3 | **E2 pilot completes: 100 scenes × 3 × {1, 2 ticks} paired to S2's 0-delay arm (the cheapest decisive experiment; it depends on KC-2's pinned protocol and noise floor, so it starts in week 2 only if S2 pins early)**; S3 verdict starts (student if KC-E yes, else the teacher's deployment levers); **B1 batch curve, timing-only** | **E2 15–38 GPU-h, $15–56**; B1 4–8 GPU-h, $4–13; S3 sweeps $102–500 | **Go/kill for #1:** go to E3 if the paired loss at 2 ticks exceeds the replicate floor on any pre-registered stratum; else add {3, 5} ticks on the same 100 scenes (another $15–56) before judging; a null through 5 ticks becomes the latency-tolerance result and #2 becomes primary |
| 4 | 11-23 → 11-29 | S3 | **E3 full curve arms {1, 2, 3} ticks on 441 × 3** (shared with S3's sweeps where the checkpoint matches); **B2 invariant arm bitwise test on 20 scenes, synthetic inputs first**; interim look only by S1's rule | **E3 102–254 GPU-h, $102–373 (91–219 GPU-h with the graph)** | KC-3 (11-30): the student verdict is the result; batch-curve numbers labelled timing-only |
| 5 | 11-30 → 12-06 | S4 start | S4 (GRPO, `n_generation ≥ 2`, estimator audit); **B3 non-invariant arm: ladder + one known-null equivalence sweep**; **K14 draft: sub-tick/trace-driven `DelayBuffer` + command-FIFO option, only if Huy's rule says the constant arms are insufficient** | **B3 34–85 GPU-h, $34–124**; S4 training on H100 SXM at $1.93–3.14: 225–450 H100-h = $434–1,413 (repriced; the plan's $403–1,080 used $1.79–2.40) | B2/B3: adopt the faster passing arm for every later sweep; if neither passes, stay at B = 1 and record the negative result |
| 6 | 12-07 → 12-13 | S4; S6 start | **P-002 last-possible order date if it was not decided in week 1 (P-024); a device ordered later arrives 01-06 → 01-21 at the quoted lead time, and T4 (weeks 10–11) and T5/E9 (week 12) collapse to the x86 half plus a labelled latency model**; **T2 start: x86 SM120 C++ runtime + `EgodriverService`, TRT-11-ready, mock-engine tests** | x86 dev 10–20 GPU-h, $10–29; optional B200 ISA check 10 h, $75–88; Jetson $5,499 + import ____ if P-002 yes | P-002 recorded either way; without it every Thor line below is "not measured on Thor" |
| 7 | 12-14 → 12-20 | S4; S6 | S4 arms and scaling model; **E4 interface arm (plan-buffer vs command-FIFO at 2 ticks, 100 scenes) if K14 landed locally**; T2 continues | E4 15–38 GPU-h, $15–56 | KC-4 (12-21): seed-SD > 0.1 → open-loop fallback; parity failure blocks training |
| 8 | 12-21 → 12-27 | S5 start; S6 | S5(a) correction pilot (only if a fine-tunable student exists); **E6 n arm at fixed d** (natural CoC length first; caps by budget forcing; n = 0 labelled by architecture); S6 T2 ladder against the Python driver with injected reference `x_0` | S5 node: 8 × H100 24–48 h at list $31.92–36.00/h = $766–1,728 (vast.ai's one 8-GPU offer at $29.33/h [D-F6.3] gives $704–1,408 [F12]); E6 inside S5's $272–1,055 | S5 prerequisites: licence permits fine-tuning; A5 decoder test passes; a student exists, else E6/E7 run on the teacher and say so |
| 9 | 12-28 → 01-03 | S5; S6 | **E7 K arm at fixed d (common noise file)**; **E5 sub-tick arms {139, 171} vs 200 ms on 100 scenes if K14 exists**; T3: engines on Jetson if the device arrived (real, labelled) | E7 +2 sweeps $68–248; E5 $15–56; Jetson engines: device time only | — |
| 10 | 01-04 → 01-10 | S5; S6 | S5 scaling/resiliency ladder (the first [C] item displaced if hours run short, P-029); **report v1 of #1: harness, tick known answers, mechanism, latency axis on synthetic inputs (publishable now); curves withheld until KC-L**; **T4: Jetson measurement protocol (120 W default and MAXN pinned; 30-min sustained; J/tick; contention arm)** | T4: device time only | KC-5 (01-11): no paired correction gain after three ablation rounds → negative result with mechanism (S5(a) only; #1 does not depend on it) |
| 11 | 01-11 → 01-17 | S6 | **T4 complete; the device point E9 placed on the curve; #2 upstream PR filed (driver `predict_batch` flag) after re-checking NVIDIA's release; K14 PR filed if Huy's rule triggered** | — | KC-6 (01-18), re-cut to tick bands (P-023): p99 ≤ 100 ms = 1 tick; (100, 200] = 2 ticks; > 150 ms as written → no CoC, K = 2, smaller expert; still over → report 5 Hz honestly |
| 12 | 01-18 → 01-24 | S7 | **T5/E9 HIL: Thor serves `EgodriverService` through `external_static` on a 100-scene subset (≈ 2.9 h device time at PR #45's DRIVE 171 ms per call, an illustrative stand-in for the Jetson value T3 measures; 13.5 h at FlashRT's 811 ms for the 10B)**; TOST vs the cloud arm (Huy's margin); golden bitwise replay; sanitizer runs; the technical report with the Thor latency overlay; the FlashRT methodology contribution | S7 cloud ≈ $200 (plan) | KC-7 (01-25): lockstep alters policy semantics → open-loop replay only. S8 not started (KC-8 untouched) |

**Totals added by the capstone (cloud, DR1):** E1 $6–23; E2 $15–56 (+$15–56 if extended); E3 $102–373; E4 $15–56; E5 $15–56; B1 $4–13; B3 $34–124; x86 dev $10–29; optional B200 $75–88: **$201–730 without the optional B200 check and $276–818 with it [F12], on top of the plan's S2–S7 cloud lines**, which the ≤ $6k tier can hold only if the plan's own upper end ($6.3k) does not materialize; the $4k cap rule (`02` §5) applies. Hardware lines: Jetson AGX Thor $5,499 MSRP + import to Vietnam ____ (unverified; SabrePC $6,268.86, out of stock; AU reseller quotes 30–45 days) [D-F6.8]; no rig in the window. Repricing note: the H100 SXM median moved $2.40 → $3.14 in one day; S4 and S5 are re-priced at their starts.

**Hours added and what is displaced (cut order `02` §0.1: S8 first, then [O], then [C] role items in reverse list order; corrections are never cut).** New hours: E0–E5 harness, tests and the K14 PR +14–20 h; #2's driver and invariance arm +16–22 h; #3's x86 runtime +12–20 h (the gated L7-6 item, now core): **+42–62 h** [F9]. S8 (40 h) is outside the window and frees nothing inside it. The [O] pool (≈ 40 h: AlpaSim's Slurm path, failure injection, replicas per GPU, the capacity document, successive halving, anytime monitoring, the RoaD control arm, token-level on-policy, the decode micro-benchmark) absorbs most of it; the residual 2–22 h must come from [C] role items. Read literally, the cut order removes "S3 and S6–S7, integration and C++" first, which is exactly #2 and #3; P-029 asks Huy to re-cut the order so that P-020/P-022/P-023's artifacts are protected and the displaced [C] items are, in order, S5's 1→8-GPU scaling and resiliency ladder (D4) and S4's multi-configuration scaling sweep beyond the two local topologies (D1). The S3/S4 verdicts are not displaced (`02` §8).

---

## §7 PhysLoop decisions (paste-ready for `docs/DECISIONS.md`; new proposals from P-020; no existing proposal is rewritten)

| Item | Decision | Carried by |
|---|---|---|
| S5(c) CoC-budget curve | **Re-cut**: a lockstep curve at a fixed tick band, reuse prerequisite dropped | P-020 |
| S6 build list | **Re-cut**: measurement artifact + one C++ driver; vision-side items dropped; kernel conditional | P-023 |
| S7 HIL + latency injection | **Re-cut**: injection moves to the cloud window (weeks 2–4); HIL stays in week 12 on a subset; sub-tick mode as an upstream PR | P-020, P-021 |
| S8 shadow mode | **Drop from the 12-week window; keep optional** | P-027 |
| P-006 | **Keep** as the gate; checklist extended | P-027 |
| P-007 | **Re-cut**: sweep-cost diagnosis on sm_120; Thor premise dropped | P-025 |
| P-008 | **Merge** into the S6 re-cut (adopt its four items) | P-023 |
| P-009 | **Keep** (i), (iii), (iv); **re-cut** (ii) into a device term and a tick term | P-028 |
| P-010 | **Merge** into P-007's re-cut as the H2 test | P-025 |
| P-015 | **Re-cut**: two arms, bitwise acceptance for the invariant arm | P-022 |
| P-018 | **Keep conditional, out of window** | P-026 |
| P-019 | **Merge** into the S6 re-cut | P-023 |
| P-002 | **Keep**; add a trigger rule | P-024 |
| §0.1 cut order | **Re-cut** for the capstone window | P-029 |

- P-020 · *(proposed 2026-10-28)* **Capstone #1: "score per tick", the tick-quantized latency-cost curve, with S5(b)/(c) as its lever arms (re-cuts S7's injection and S5(c)).**
  - **What.** S7's constant-delay injection runs in the cloud from week 2 on the pinned S2 protocol: E0 CPU known-answer tests, E1 smoke (20 scenes × 3 × {0,1,2,3} ticks), E2 pilot (100 scenes × 3 × {1,2} ticks paired to S2's 0-delay arm), E3 full curve (441 × 3 × {1,2,3} (+5)). S5(c) becomes score(n | d) at a fixed tick band on one checkpoint at full prefill (the history-frame-reuse prerequisite is dropped; reuse is a separate model-change arm, KC-L2). S5(b) runs at the same fixed d with a common noise file. The x-axis is applied staleness in ticks; the device point comes from P-023 or is labelled "not measured on Thor".
  - **Why.** AlpaSim applies a delayed plan once per control tick, so 139/171/200 ms are all 2 ticks = 200 ms of staleness [E-§3.1–3.2]; every published AlpaSim score is lockstep [E-§3.5]; NVIDIA ran 200 ms until 03-24 [L7 2.3]; no public AlpaSim latency curve with CIs exists [C-35]; the CoC token (7.35 ms measured) decides the tick band, not the expert step (≤ 1.7 ms bound) [A-Q1.10; F3]; S5(c)'s prerequisite conflicts with its fixed checkpoint [E-§7.5]; KC-5 (01-11) precedes KC-6 (01-18), so S5(c)'s latency axis could not have come from S6 in time [E-§7.1].
  - **Carries over.** S7's constant arms, lockstep semantics and TOST (★); S5(b)'s K set and noise pairing; S5(c)'s natural-length-first rule, budget forcing and the n = 0 architecture label; S3's teacher-lever fallback on the KC-E-fails branch.
  - **Gates.** P-005 before any run; KC-L before any public curve (code, synthetic known answers, manifests and the latency axis on synthetic inputs are publishable now); KC-E for student arms only; interim looks by S1's group-sequential rule.
  - **Cost.** E1 $6–23; E2 $15–56; E3 $102–373 (91–219 GPU-h with the expert graph); lever arms inside S5's $272–1,055 (DR1, 10-28) [F5]. Hours +14–20 including P-021's harness.
  - **Predictions (★).** loss at 1/2/3 ticks overall ____ / ____ / ____; on the stationary-lead stratum ____; on geometry strata ____; curve shape over n at d = 2 ____; the K below which the score drops at fixed d ____; margins per stratum ____; the force-GT warm-up handling rule ____.

- P-021 · *(proposed 2026-10-28)* **AlpaSim upstream: a sub-tick / trace-driven `DelayBuffer` mode and a command-FIFO option (makes S7's conditional PR a deliverable).**
  - **What.** (a) A per-step delay source (trace or distribution) applied inside the controller service so that L in (100, 200] ms resolves below one control tick without changing the policy's query rate or the camera clocks; (b) a command-FIFO option, ã_t = a_{t−τ}: the same `DelayBuffer` logic applied to the MPC's output commands (Bench2Drive-Robust's main mode [C-30]), so the two interfaces compare at matched applied staleness (E4); command-hold (repeat the last command and refresh every d ticks: B2D-Robust's dynamic mode and arXiv 2601.07393's semantics) only as an optional third arm, because its mean staleness is (d − 1)/2 ticks, not d. Filed as an issue first (week 1, with the CPU known answers and the warm-up GT finding), as a PR only after re-checking NVIDIA's post-10-31 release and only if Huy's rule says the constant arms are insufficient (`02` S7).
  - **Why.** Finer resolution is not a config change: policy and controller share `dt`, camera clocks are separate, and `assert_zero_decision_delay` fails at 50 ms steps [E-§3.2]; this is MISSION (a) on AlpaSim's driver path; AlpaSim accepts outside PRs [B-C08]; it is not #208's field [ver27 V21]; C-31's plan-vs-command hypothesis has no measurement anywhere [C-31].
  - **Carries over.** Nothing is forked; S7's "trace-driven delay as an AlpaSim PR" text.
  - **Gates.** Re-check the next AlpaSim release; never compete with an offered PR; KC-L for any score shown in the PR (use synthetic scenes or scene-ID manifests).
  - **Cost.** +8–12 h; E4/E5 arms $15–56 each.
  - **Predictions (★).** plan-buffer minus command-FIFO loss at d = 2 ____; loss difference between 139 and 200 ms applied ____.

- P-022 · *(proposed 2026-10-28)* **Capstone #2: a bitwise-invariant batched Alpamayo driver (amends P-015).**
  - **What.** Two one-variable arms of a GPU-batched `predict_batch` for the Alpamayo adapters: (i) invariant arm — per-session generators (consume alpamayo1.5 PR #26's per-row `torch.Generator`) plus batch-invariant GEMM/RMSNorm/attention reduction order, accepted only if B = k is bitwise identical to B = 1 on every compared tick (E0a); (ii) non-invariant arm — P-009's ladder E0a–E1 plus one known-null paired closed-loop verdict. Report the invariance tax and GPU-hours per rollout vs B; adopt the faster passing arm for S3–S5 and P-020's sweeps.
  - **Why.** The stock driver is eager at B = 1 and the policy is ≈ 73–87.5 % of step time [L7 4.3–4.4, 4 Q4]; batching changes numerics and RNG streams [L7 4 Q4]; invariance costs 34 %–> 2× on LLM decode [C-37, C-39], so the net is 14–54 % per sweep [F7]; no such study exists for flow policies in closed loop [C-44]; the known-null verdict is a validity test of the gate (D3).
  - **Carries over.** P-015's build, acceptance and the conditional static-KV sub-option; L7-4's cost lines.
  - **Gates.** P-005 (one extra sweep within the tier); KC-L for the verdict's numbers; a comment on PR #26 rather than a duplicate PR; re-check NVIDIA's post-10-31 sync before the AlpaSim PR.
  - **Cost.** B1 $4–13; B3 one equivalence sweep $34–124; +16–22 h. Saving on S3–S5: 90–234 GPU-h (invariant, worst) to 348–910 GPU-h (non-invariant, best) [F7].
  - **Predictions (★).** invariance tax ____×; per-sample speed-up at B = 8 / 16 ____ / ____; E2 verdict for the non-invariant arm ____; the gate's false-flag rate under the known null ____.

- P-023 · *(proposed 2026-10-28)* **Capstone #3: the Jetson AGX Thor measurement artifact and one C++ driver for two targets (re-cuts S6; merges P-008 and P-019; amends P-009 (iii)).**
  - **What.** S6's build list becomes: (1) a real engine with a per-stage timeline, labelled real or emulated, device and versions named; (2) one diagnosed bottleneck and one fix, a kernel only if the profile puts a stage above its bound by Huy's margin ____; (3) FP8 vs NVFP4 as policy rows with the ladder; (4) the measurement protocol: per-stage p50/p99/p99.9, glass-to-action (P-028), J/tick, a 30-minute sustained run at the default 120 W mode and at MAXN pinned, P-009's contention arm, all on synthetic or own inputs (publishable under the model licence); (5) the C++ `EgodriverService` built first on x86 SM120 and then on Jetson (strongly typed TensorRT, `IPluginV3`, CI on 10.16 and 11.x, golden bitwise replay, sanitizers, p99.9 histograms). Dropped: camera-batched ViT, FP8 ViT, ViT cache, streaming reuse. Device-in-loop verdicts run on a 100-scene subset only.
  - **Why.** Edge-LLM ships the vision-side items and its matrix rejects action-runner KV reuse [A-Q1.1, Q1.5]; on Jetson the teacher's expert is at 66–79 % of bound and the 2B's expert step is ≤ 1.7 ms, so there is no expert-kernel headroom worth a stage [A-Q4.6; F3]; no glass-to-action, sustained or J/tick figure exists for Alpamayo on either Thor [A-§4.1 #6–7]; a full sweep with the device in the loop costs 7–13 h of device time [E-§3.6]; KC-6's 150 ms sits mid-band [E-§7.20]; TRT 11 API changes are known [L7 1.5].
  - **Carries over.** P-008's four items; P-019's acceptance (ladder with injected reference `x_0`, paired verdict, a cloud FP8 verdict never stands in for Thor); S6's owner targets; the TRT-11-ready C++ (L7-7) and the C++ server spec (L7-9); K18's proxy-transfer section.
  - **Gates.** P-002 for every device line (P-024); P-005; P-009 (i)(iii) with a recorded-clocks fallback; KC-L if NuRec frames are used for any published latency; KC-6 re-read in tick bands.
  - **Cost.** x86 10–20 GPU-h ($10–29); optional B200 ISA check 10 h ($75–88); Jetson $5,499 + import ____; +12–20 h for the runtime (the gated L7-6 item becomes core).
  - **Predictions (★).** p99 per stage, no-CoC and CoC arms ____; glass-to-action minus model time ____; 30-minute p99 at 120 W minus 20-request MAXN p99 ____; J/tick ____; expert % of bound ____; NVFP4 vs FP8 verdict ____.

- P-024 · *(proposed 2026-10-28)* **A trigger rule for P-002 (amends P-002, Huy's decision).**
  - **What.** Buy the Jetson AGX Thor dev kit when KC-E yields a public 2B student whose licence permits evaluation (then #3 has a model and #1 has a device point in the 1–2-tick regime). Decide in week 1 (11-02 → 11-08), right after KC-E; at the quoted 30–45-day lead time [D-F6.8], 12-07 is the last order date that lands the device before T4 (weeks 10–11), and a later arrival collapses T4 and T5/E9 to the x86 half plus a labelled latency model. If KC-E fails, defer: the x86 half of P-023 runs, #1 runs on the teacher, and the purchase is re-decided at Edge's release.
  - **Why.** No Thor is rentable [D-F6.7]; the dev kit is out of stock at one reseller with a 30–45-day quote at another [D-F6.8]; for the 10B teacher the Jetson point is 9 ticks and FlashRT owns the behaviour-preserving reference [F4; B-R03]; S6's window starts 12-07 [`02` §3].
  - **Carries over.** P-002's price and the Jetson-vs-DRIVE warning.
  - **Gates.** P-005 tier recorded; the purchase line is separate from the cloud tier (`02` §5).
  - **Cost.** $5,499 + import ____; lead time ____.
  - **Predictions (★).** none; this is a decision rule.

- P-025 · *(proposed 2026-10-28)* **P-007 re-cut as a sweep-cost diagnosis on sm_120, with P-010 merged as its H2 test (amends P-007 and P-010).**
  - **What.** E-K0 on the rented RTX PRO 6000 in week 1: `nsys` over the tick, `ncu` over one expert step (if counters are exposed; else recorded clocks and CUDA events), graphs on and off, with the P-010 GEMV/skinny-GEMM curve as the H2 discriminator. Its output decides whether any sweep-side fix exists and feeds P-022; it makes no Thor claim.
  - **Why.** The "~20 % of bound" is the ledger's derivation on sm_120 [DEC P-007]; on Jetson the teacher's expert runs at 66–79 % of its bound [A-Q4.6], so the Thor premise is gone; the 2B's expert step is ≤ 1.7 ms [F3]; PR #36 already removed CPU launch overhead on the PyTorch path [L7 4.7]; any sm_120 fix is Amdahl-bounded at 11–25 % of a sweep [L7 4 Q4].
  - **Carries over.** E-K0's setup; H1a/H1b/H2/H3 discriminators; P-010's known answer (56–90 KB in flight per CTA at 16 CTAs) [L7 §4 KA-7].
  - **Gates.** KC-1 passed; P-005.
  - **Cost.** A few GPU-h ($4–12); 2–4 h.
  - **Predictions (★).** expert % of bound on sm_120 with graphs on ____; the top kernel class ____.

- P-026 · *(proposed 2026-10-28)* **P-018 stays conditional and outside the 12-week window.**
  - **What.** Re-decide at Edge's release and after E7 shows a cliff above the K that the tick band allows; no few-step training before 01-25.
  - **Why.** K = 5 with fixed noise is deployed [A-Q2.2]; Edge may be few-step [DEC P-018]; a one-step head cut LIBERO-Long to 26 % [C-28i]; weights are KC-L2 material; the K lever does not move a tick for the 2B [F3].
  - **Carries over.** P-018's method and cost lines.
  - **Gates.** KC-L2; P-005.
  - **Cost.** Deferred.
  - **Predictions (★).** ____ (none until E7).

- P-027 · *(proposed 2026-10-28)* **S8 deferred beyond the window; P-006's checklist extended.**
  - **What.** No rig work before 01-25. P-006 adds: the Decree 356/2025 Art. 18 cross-border dossier (Form 01b, within 60 days of any transfer) and Art. 19 processing dossier if identifiable footage is processed on foreign GPU hosts (Art. 17(1)(c)); on-device anonymization before upload as the design lever; the smart-vehicle definition check; the modification question for external cameras; re-verified rig prices (the 9295 cameras and HSB stock are unverified; the deserializer PSU is backordered). S8's real-sensor glass-to-action moves to P-023/P-028's measurement list.
  - **Why.** S8's window starts the day the 12 weeks end [E-§5]; P-006 and P-002 are unrecorded [E-§7.16]; the legal reading is an engineering reading, not legal advice [D-F8.3–F8.11]; shadow mode stays open-loop and is never validation [`01` C12].
  - **Carries over.** S8's rig design, the within-rig contrast and the trigger-classifier spec (★).
  - **Gates.** P-006, P-002, KC-8.
  - **Cost.** $0 in the window.
  - **Predictions (★).** disagreement rate, dense mixed traffic vs arterials ____ (unchanged slot).

- P-028 · *(proposed 2026-10-28)* **P-009 (ii) glass-to-action split into a device term and a simulated term (amends P-009; keeps (i), (iii), (iv)).**
  - **What.** Device term: oldest-frame exposure → command available, measured on the device by the C++ server's host monotonic clock and CUDA events (the frame-age convention is ★: with a 0.4 s, 4-frame history the "oldest frame" adds ≈ 300 ms by definition [E-§7.14]). Simulated term: ⌈L/Δt⌉ ticks in AlpaSim, where L is either model time or the device term, as two arms. (iii) keeps pinned clocks with a recorded-clocks fallback on rented hosts; (i) and (iv) stand as adopted by D-015.
  - **Why.** `planner_delay_us` covers capture → planner output → controller and is applied in whole ticks [E-§3.1]; actuation lives in the vehicle model, not in the delay [E-§7.14]; D-015 equated (ii) with the simulated delay [DEC D-015 notes].
  - **Carries over.** P-009's ladder as the gate's protocol for every driver change.
  - **Gates.** P-002 for the device term.
  - **Cost.** 0 beyond P-023.
  - **Predictions (★).** device term minus model time ____.

- P-029 · *(proposed 2026-10-28)* **Cut order for the capstone window (amends §0.1's cut order for 11-02 → 01-25 only).**
  - **What.** If hours run short: S8 (already outside the window), then the [O] items, then — before the "S3 and S6–S7, integration and C++" list, which the literal reverse order would cut first — S5's 1→8-GPU scaling and resiliency ladder, then S4's multi-configuration scaling sweep beyond the two local topologies. P-020, P-022 and P-023's artifacts are cut last among role items. The corrections and the S3/S4 verdicts are never cut.
  - **Why.** The capstone adds +42–62 h against a −40 to +54 h slack [F9]; the literal order removes the batched driver, the TRT-11 C++ and the C++ server first [E-§5]; §8 and §0.1 already state the order in three ways [E-§7.3].
  - **Carries over.** §0.1's three tiers and its "corrections are never cut" rule.
  - **Gates.** KC-1's re-forecast from S1's actual hours (P-003).
  - **Cost.** 0.
  - **Predictions (★).** none.

---

## §8 Proof-of-value plan

**Order of shipment, with what KC-L and KC-L2 allow at each step.**

| # | Week | Artifact | KC-L / KC-L2 status | Reaches whom, where |
|---|---|---|---|---|
| 1 | 1 | `physloop-gate` v0 (S1) and the AlpaSim issue proposing the paired layer (`02` S1) | $0; no dataset data | **Simulation team**: NVlabs/alpasim issue (the maintainers who answer #74, #208, #224) |
| 2 | 1–3 | **AlpaSim issue: latency is quantized to control ticks** — `DelayBuffer` semantics, the force-GT warm-up ticks, the ⌈L/Δt⌉ CPU known-answer tests, and the sub-tick / command-FIFO proposal (P-021) | Publishable now: code and synthetic tests only; no NuRec number | **Simulation team**: NVlabs/alpasim; cross-reference #74 (navigation text), #208 (`n_sim_steps`); never compete with an offered PR |
| 3 | 3–5 | **#2's bitwise test and batch curve on synthetic inputs** (B1, B2), then the known-null verdict | Code, bitwise counts and timing-only numbers now; the verdict's NuRec-derived numbers after KC-L | **Simulation + research teams**: AlpaSim driver PR behind a flag (after re-checking the post-10-31 sync); a results comment on alpamayo1.5 PR #26 (Karkus) instead of a duplicate PR |
| 4 | 10–12 | **The "score per tick" technical report / arXiv paper** (CoRL/ICRA/MLSys-grade claims only with KC-L): mechanism, interface comparison, the tick-quantized curve per stratum with one-sided bounds, the lever arms at fixed d, the device point or its "not measured on Thor" label | Without KC-L: method, code, scene-ID manifests, synthetic results and the latency axis on synthetic inputs; with KC-L: the curves. No weights (none produced; S5(a) weights are KC-L2) | **AV research**: arXiv + AlpaSim issue linking the report; cite TacitVLA and "Perceives but Doesn't Brake" authors' questions; the Bench2Drive-Robust repo issue tracker for the interface comparison |
| 5 | 6–12 | **The Jetson AGX Thor measurement artifact + C++ driver** (P-023): per-stage p50/p99/p99.9, glass-to-action, J/tick, 30-minute sustained at 120 W and MAXN pinned, real-engine labels, TRT 10.16/11.x CI, golden bitwise replay; the proxy-transfer section | Publishable under the model licence when inputs are synthetic or own; NuRec-frame latency logs after KC-L; never a DRIVE number | **Inference teams**: FlashRT (methodology/benchmark harness PR; the maintainer merges Thor ports in days; issue #223 thread); TensorRT-Edge-LLM issues for what breaks (no merge expected: 0 of 33 outside PRs; DCO sign-off); the NVIDIA Jetson Thor forum thread on Alpamayo quantization (382409) with the real-engine numbers |
| 6 | after 01-25 | Shadow-mode OOD report (only if P-006) | Own footage; privacy plan; model-licence terms; never "validation" | Data-engine lines (JR2025038 (6)); not in this window |

**Posting lines each artifact answers** (paraphrased IDs; `05` §3 and [B-J02–J04]): #1 → JR2010863 X2, D3, I2-accuracy, D5; JR2025038 (2), (3), (5); JR2014515 (3). #2 → JR2010863 I2-perf, X3, D2, D3, D5. #3 → JR2010863 Q4, X3, I2-perf, D5; JR2025038 (2), (3), (4), (7); JR2014515 (1)–(5); JR2016777 (1), (4). Every public number states n, CI, replicates and seed mode, manifest, AlpaSim SHA and lockfile, scene list, score definition, deviation convention, device with JetPack/TensorRT, "real engine" or "emulated", one node, solo reproduction (`02` §6).

---

## §9 Risks, what would change the verdict, and my two weakest points

**Risks and responses.**
- **A null curve.** If the paired loss stays within the replicate floor through 5 ticks on every stratum, #1 becomes a latency-tolerance result for the plan-plus-MPC interface (still citable against Bench2Drive-Robust's command-delay collapse), and #2 becomes the primary artifact. The pilot's MDE is 6–12 pp [F6], so a pilot null is not evidence of zero; E3 decides.
- **NVIDIA publishes first.** The next AlpaSim release discloses withheld simulation choices [ver27 V7]; NVIDIA authors are active on adjacent questions [C K6]. Response: the novelty wording is dated and narrow (pairing, CIs, strata, interface); the mechanism and the upstream PR keep value.
- **KC-L.** Every closed-loop score is NuRec-derived; P-004 is unsent [E-§6]. Response: send it in week 1; ship code, synthetic results and manifests first (§8).
- **KC-E outcomes.** Edge not public: #1 runs on the teacher at 8–9 Jetson ticks, far from the deployment regime; Edge public but tacit: the n arm disappears; Edge not fine-tunable: S5(a) dies, #1 unaffected.
- **P-002 timing.** No Thor is rentable; stock and lead time are unverified [D-F6.7–F6.8]. Response: P-024's trigger; the x86 half first; "not measured on Thor" labels.
- **Power mode and clocks on both sides.** 15 of 35 rented WS offers cap at 350–550 W [D-F6.2]; Thor's default mode is 88 % of MAXN with DVFS on [D-F1.12]. Response: P-009 (iii), recorded in every manifest.
- **The instrument.** NuRec may vary run to run (#128, a hedged recollection) [ver27 V26]; the evaluator changed mid-challenge (#198/#215) [ver27 §A]; evaluator artifacts can inflate scores [C-54]. Response: S2's noise floor and scorer known-answer test precede every arm.
- **Scope creep.** +42–62 h against a −40 to +54 h slack [F9]. Response: P-029; the S3/S4 verdicts and the corrections are never cut.
- **Config compatibility.** Whether `planner_delay_us > 0` runs with `assert_zero_decision_delay: true` on `e2e_challenge` is unverified until E1; the two checks act at different events (policy vs controller) [E-§3.1] and NVIDIA's 200 ms config ran [L7 2.3], so the risk is low, but #1 rests on it.

**What would change the verdict.**
- Edge ships by 10-31 with weights, a decodable CoC and an evaluation licence, and a Thor arrives by mid-December: #3 rises (a model, a device, a 1–2-tick regime) and #1's student arms become the headline; the order stays 1 > 2 > 3 but #3's score moves from 33 toward 38 (Small 3 → 4, Win 3 → 4).
- A FlashRT or NVIDIA release of a Jetson 2B runtime with published per-stage, sustained and energy numbers: #3's gap closes; keep only the C++ server and the HIL.
- An AlpaSim release that adds a trace-driven delay or an asynchronous driver mode: P-021 is withdrawn; #1 consumes it.
- An invariance tax above ≈ 2× on the Alpamayo driver (CoRun's regime [C-39]): #2's invariant arm loses its net saving; keep the non-invariant arm with the known-null verdict.
- P-005 at ≤ $2.5k: #1's pilot and curve and #2's sweep still fit ($201–730 added [F12]); #3's device line does not.

**My two weakest points (for the red team).**
1. **The effect may be small where it matters.** The mechanism predicts loss only in dynamic-actor strata, which are rare in the 441 (a 9-scene stratum has ≈ 0.2 % power, `01` C7); the whole-curve MDE is 3–6 pp and the per-stratum verdicts may return INSUFFICIENT at every tick. Then #1 is a step function of 3–4 points plus a mechanism, and a critic can call it methodology rather than a result. The sub-tick PR (P-021) and the interface arm (E4) are the only ways to add information, and both depend on an upstream change and Huy's rule.
2. **#1 is a promotion of existing plan items, and its headline numbers are licence-gated.** The curve re-cuts S7 and S5(b)/(c); its novelty rests on pairing, CIs, strata and the plan-vs-command mechanism, not on a new lever; NVIDIA, which once ran the 200 ms config, could publish the same curve from internal data at any time; and until KC-L passes, the public artifact is the harness, the known answers and the latency axis on synthetic inputs, not the curve. A third, smaller point: without P-002 the "ms" axis is a roofline-plus-proxy model, and "score per tick" with a modelled tick is weaker than with a measured one.

---

## §10 Source index (ID → URL or path)

Dates of access are those stated in each track's note (2026-10-28) unless marked. Track notes: `research_notes/PhysLoop AV capstone selection/{A_state_of_the_art, B_demand_signals, C_research_frontier, D_hardware_reality, E_physloop_audit}.md`.

**Track A (state of the art)**
- A-Q1.1 TensorRT-Edge-LLM releases — https://github.com/NVIDIA/TensorRT-Edge-LLM/releases (re-checked 2026-10-29: newest v0.11.0, 29 Sep)
- A-Q1.3 issue #130 — https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/130
- A-Q1.4 PR #260 — https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/260
- A-Q1.5, Q1.6 KV-cache reuse doc — https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/features/kv-cache-reuse.md
- A-Q1.7 Alpamayo example — https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/examples/vla/alpamayo.md
- A-Q1.8 pi0.5 example — https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/examples/vla/pi05.md
- A-Q1.10, A-J6 performance benchmarks — https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/performance/performance-benchmarks.md
- A-Q1.11 visual-token pruning — https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/features/visual-token-pruning.md
- A-Q2.1 recipes PR #45 — https://github.com/NVlabs/alpamayo-recipes/pull/45
- A-Q2.2, Q2.3 PR #45 README @ fbed0a7 — https://github.com/NVlabs/alpamayo-recipes/blob/fbed0a70d5452bc50527e1c11ae6855fccbea58c/recipes/alpamayo1_x_edgellm/README.md
- A-Q2.4 PR #45 run.py @ fbed0a7 — https://github.com/NVlabs/alpamayo-recipes/blob/fbed0a70d5452bc50527e1c11ae6855fccbea58c/recipes/alpamayo1_x_edgellm/run.py
- A-Q2.5 .gitmodules @ fbed0a7 — https://github.com/NVlabs/alpamayo-recipes/blob/fbed0a70d5452bc50527e1c11ae6855fccbea58c/.gitmodules
- A-Q3.3 alpamayo1.5 PR #26 — https://github.com/NVlabs/alpamayo1.5/pull/26
- A-Q4.1 FlashRT PR #229 — https://github.com/flashrt-project/FlashRT/pull/229
- A-Q4.2–Q4.6 FlashRT Alpamayo guide — https://github.com/flashrt-project/FlashRT/blob/main/docs/alpamayo15.md ; arithmetic `capstone_scratch/A/arith_A.py`
- A-Q4.8 FlashRT-PAI — https://arxiv.org/abs/2610.24349
- A-Q5.1–Q5.6 FlashDrive — https://arxiv.org/abs/2608.12932 ; https://arxiv.org/html/2608.12932v1 ; code https://github.com/z-lab/flashdrive
- A-Q6.1 JetPack archive — https://developer.nvidia.com/embedded/jetpack-archive
- A-Q6.2 TensorRT 10→11 Jetson migration — https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-jetson.html
- A-Q7.1 Edge press release — https://nvidianews.nvidia.com/news/european-mobility-leaders-build-safe-vehicles-and-robotaxi-fleets-on-nvidia-hyperion ; blog https://blogs.nvidia.com/blog/open-models-europe-industry/
- A-Q7.3 HF Alpamayo search — https://huggingface.co/api/models?author=nvidia&search=alpamayo (re-checked 2026-10-29: 3 models, no Edge)
- A-Q7.7 ModelOpt examples/alpamayo — https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/alpamayo
- A-Q8.6 GR00T N1.7 on Thor — https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/
- A-Q8.7 openpi π0.5 on Thor — https://www.jetson-ai-lab.com/tutorials/openpi_on_thor
- A-Q8.8 RTC — https://arxiv.org/abs/2506.07339
- A-X.2 Qwen3-VL-2B config — https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/main/config.json
- A-§4.1, §4.3, C1–C4 — sections of `A_state_of_the_art.md`

**Track B (demand signals)**
- B-E01 Edge-LLM #130 — https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/130
- B-E04 #216, PR #225 — https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/216 ; https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/225
- B-C02 merged PRs — https://github.com/NVIDIA/TensorRT-Edge-LLM/pulls?q=is%3Apr+is%3Amerged
- B-C04 alpamayo1.5 CONTRIBUTING — https://github.com/NVlabs/alpamayo1.5/blob/main/CONTRIBUTING.md
- B-C06 FlashInfer CONTRIBUTING — https://github.com/flashinfer-ai/flashinfer/blob/main/CONTRIBUTING.md
- B-C08 AlpaSim PR #228 — https://github.com/NVlabs/alpasim/pull/228
- B-K03–K05 FlashInfer CAKE trackers — https://github.com/flashinfer-ai/flashinfer/issues/4254 ; https://github.com/flashinfer-ai/flashinfer/issues/6594
- B-F01 forum 382409 — https://forums.developer.nvidia.com/t/382409
- B-F08 ZETIC dashboard thread — https://forums.developer.nvidia.com/t/387191
- B-F09 MLPerf Edge Agentic blog — https://developer.nvidia.com/blog/tensorrt-edge-llm-completes-the-mlperf-edge-agentic-benchmark-6-4x-faster-on-jetson-agx-thor/
- B-J01 JR2010863 — https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Senior-Machine-Learning-and-Simulation-Engineer---Autonomous-Vehicles_JR2010863
- B-J02 JR2025038 — https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Senior-Integration-Engineer--End-to-End-Model---Autonomous-Vehicles_JR2025038
- B-J03 JR2014515 — https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Senior-Systems-Software-Engineer---Deep-Learning-Solutions_JR2014515
- B-J04 JR2016777 — https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Senior-Software-Performance-Engineer---AV-Platform_JR2016777
- B-R02 vLLM-Omni PR #9133 — https://github.com/vllm-project/vllm-omni/pull/9133
- B-R03 FlashRT #223 — https://github.com/flashrt-project/FlashRT/issues/223
- B-R04 ModelOpt #1895 — https://github.com/NVIDIA/Model-Optimizer/issues/1895
- B-R09 absence search — `B_demand_signals.md` row R09

**Track C (research frontier)**
- C-01 version checks and FlashDrive code — https://export.arxiv.org/api/query ; https://github.com/z-lab/flashdrive ; https://tacitvla.github.io
- C-02 PhyAI — https://arxiv.org/abs/2608.03682
- C-03 RTC — https://arxiv.org/html/2506.07339
- C-08 Think at 5 Hz — https://arxiv.org/abs/2607.15621
- C-09 FlashDrive DFlash; recipes PR #43 — https://arxiv.org/html/2608.12932 ; https://github.com/NVlabs/alpamayo-recipes/pull/43
- C-10b speculative VLA query — `C_research_frontier.md` row C-10b
- C-19 TacitVLA — https://arxiv.org/abs/2610.17540
- C-20 FlashDrive streaming — https://arxiv.org/html/2608.12932
- C-26 quantization-in-closed-loop search — `C_research_frontier.md` row C-26
- C-28, C-28i few-step, GR00T one-step — https://arxiv.org/abs/2609.18108 ; https://arxiv.org/abs/2604.21489
- C-29 warm-start search — `C_research_frontier.md` row C-29
- C-30, C-30i, C-31 Bench2Drive-Robust — https://arxiv.org/html/2605.18059v1 ; https://github.com/Thinklab-SJTU/Bench2Drive-Robust ; AlpaSim `delay_buffer.py` https://github.com/NVlabs/alpasim/blob/main/src/runtime/alpasim_runtime/delay_buffer.py
- C-35 search record — `C_research_frontier.md` row C-35
- C-36, C-36i Thinking Machines — https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- C-37 SGLang deterministic — https://lmsys.org/blog/2025-09-22-sglang-deterministic/
- C-38 vLLM batch invariance — https://docs.vllm.ai/en/latest/features/batch_invariance.html
- C-39 CoRun — https://arxiv.org/abs/2608.14376
- C-44 batch-invariance search record — `C_research_frontier.md` row C-44
- C-47 within-paper disagreements — https://arxiv.org/html/2608.12932 ; https://arxiv.org/html/2609.20756
- C-54 Bench2Drive #203 — https://github.com/Thinklab-SJTU/Bench2Drive/issues/203
- C D2, D10, D11, D14, K2, K4, K6 — `C_research_frontier.md` §3–§4
- New (web, 2026-10-29, abstract-level, unverified): arXiv 2601.07393 — https://arxiv.org/pdf/2601.07393

**Track D (hardware reality)**
- D-DT1 Jetson Thor module datasheet — https://dam-cdn.nvd.orangelogic.com/AssetLink/x20v2h2y15uh0ne54rpb73xh4x7s0m1s.pdf
- D-DT2 Jetson Thor blog — https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/
- D-DT3 Thor power and performance (r39.2.1) — https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonThor.html
- D-DT5 Jetson FAQ — https://developer.nvidia.com/embedded/faq
- D-DT7 CUDA for Tegra — https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html
- D-DT9 forum 356355 — https://forums.developer.nvidia.com/t/clarification-on-cuda-core-and-tensor-core-counts-for-jetson-agx-thor/356355
- D-DC1 CUDA programming guide, compute capabilities — https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
- D-DC2 PTX ISA 9.4 — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- D-DP1 RTX PRO 6000 WS datasheet — https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf
- D-DR1 vast.ai bundles (snapshot 2026-10-28 16:15:50 UTC) — https://console.vast.ai/api/v0/bundles/ ; `capstone_scratch/D/vast_2026-10-28/`
- D-DR3 RunPod pricing — https://www.runpod.io/pricing
- D-DR6 SabrePC — https://www.sabrepc.com/945-14070-0080-000-NVIDIA-S104023743
- D-DR7 Thor rental search (negative) — https://www.jetson-ai-lab.com/tutorials/gtc26/setup ; https://nvidia.com/en-au/data-center/launchpad
- D-DG1–DG4 rig parts — https://leopardimaging.com/gmsl2-cameras-on-thor/ ; https://leopardimaging.com/product/accessories/adapters-carrier-boards/for-holoscan/li-gmsl2-deser-holoscan/ ; https://www.sparkfun.com/sparkfun-allband-gnss-rtk-breakout-zed-x20p-qwiic.html ; https://docs.nvidia.com/holoscan/sensor-bridge/latest/ptp.html
- D-DV1 Decree 356/2025/NĐ-CP (signed scan) — https://hcc.nghean.gov.vn/laws/detail/Nghi-dinh-so-356-2025-ND-CP-cua-Chinh-phu-Quy-dinh-chi-tiet-mot-so-dieu-va-bien-phap-thi-hanh-Luat-Bao-ve-du-lieu-ca-nhan-2058/
- D-F1.12, F2.7, F3.9–F3.11, F4.1–F4.10, F5.10, F6.1–F6.8, F7.1–F7.8, F8.3–F8.11 — rows of `D_hardware_reality.md`; arithmetic `capstone_scratch/D/scripts/d_arith.py`

**Track E (plan audit)** — `E_physloop_audit.md` §1–§7; AlpaSim code at `e2e_challenge` @ e121e591 and `main` @ affc2eab (`RT/delay_buffer.py`, `RT/events/controller.py`, `RT/events/policy.py`, `RT/config.py`, `WZ/base_config.yaml`, `CTL/mpc_controller.py`, `CTL/system.py`, `CTL/mpc_impl/*.py`, `RT/services/driver_service.py`); arithmetic `capstone_scratch/E/{delay_ka.py, calc.py}`.

**Repo (read-only)** — `DOC_AUTHORITY.md`; `docs/00_START_HERE.md`; `docs/01_SYSTEM_DESIGN.md` C1, C3, C5, C7, C10–C13; `docs/02_PHYSLOOP_PLAN.md` §0.1, §1, §2, §3 (S1–S8), §5–§9; `docs/03_MASTERY_PLAN.md` §4; `docs/04_CORE_PLAN_TECHNICAL.md` §2–§4; `docs/05_ROLE_MAPPING.md` §1, §3; `docs/DECISIONS.md` D-012, D-015, P-002…P-019; `nano_alpamayo/README.md`; `docs/research/2026-10-27-role-alignment/verification.md` (ver27 V4, V5, V7, V8, V21, V26); `docs/research/2026-10-27-role-alignment/notes/L7_thor_performance.md` (rows 1.4–1.11, 2.1–2.7, 4.3–4.14, §4 KA-1…KA-7, §5); `docs/research/2026-10-27-role-alignment/notes/L5_student_data.md` (rows 1.3, 1.3i, 1.13); `docs/research/2026-10-25-teacher-to-thor/verification.md` (A8, B4, C3, C7); `docs/research/2026-10-25-teacher-to-thor/review.md` §3.1–§3.3.

**This analysis** — `capstone_scratch/F/arith_F.py` and `arith_F.out.txt` (rows F0–F11). Web novelty checks run 2026-10-29 (UTC+7) with the query strings quoted in §4.4 and §5; the two fact re-checks (HF Alpamayo list; Edge-LLM releases) ran the same day and found no change against the 10-28 notes.
