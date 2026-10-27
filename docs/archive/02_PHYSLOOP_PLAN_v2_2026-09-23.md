> **ARCHIVED — historical, superseded on 2026-10-03 (DECISIONS D-010, D-011). Never act on this file; the live version is listed in `DOC_AUTHORITY.md`.**

# 02 — PhysLoop Final Build Plan v2 (23 September 2026)

**Status:** authoritative (see `../DOC_AUTHORITY.md`). Replaces v1 of the same day.
Mirror of `claude/physloop-final-plan-2026-09-23.md` in the claude.ai PhysicalAI project.

## 0. What changed and why

**Correction (our error).** v1 and the core-plan doc claimed NVIDIA's open stack never
trains Alpamayo's flow-matching action expert. That is true only of the *open-loop* recipe.
**AlpaGym (NVlabs, first commit 2026-06-16) already runs closed-loop GRPO on the action
expert** with SDE log-probs (`int_method: sde`, `sde_type ∈ {sde, cps}`), off-policy
staleness up to 4 steps, rollout prefetch and a `perf/` instrumentation module. v1's D5
("RL through the deployed expert") is therefore largely done by NVIDIA, and D4's "async
engine" has a live NVIDIA baseline. Both are demoted.

**Other verified facts that reshape the plan**
- **Production architecture.** DRIVE AV (Mercedes CLA MB.DRIVE ASSIST PRO, the L4 S-Class,
  the Uber program) runs an end-to-end stack *in parallel with a classical safety stack*,
  under Halos guardrails. Alpamayo is positioned as a **teacher distilled into compact
  runtime models on DRIVE AGX Thor**; a distillation script is "planned", no public procedure.
- **Evaluation already exists in part.** The AlpaSim E2E Closed-Loop Challenge 2026
  (NVIDIA AV Research, KE:SAI, OpenDriveLab) ships a leakage-aware 1,761/441 NuRec split
  and **Drive-IRT**, an item-response-theory capability score with 95% intervals.
  Deadline Oct 31 2026 (4-page report, NeurIPS presentation); submissions run in
  **16 GiB VRAM at 0.1 s model-time per call**; registration requires an organization email.
- **Quantization is partly taken.** FlashDrive (CoRL 2026): Alpamayo 1.5 from 717 to
  151 ms with W4A8 + speculative decoding, action expert kept in BF16, closed-loop in sim.
  NVIDIA's PTQ recipe is open-loop only. TensorRT-Edge-LLM runs Alpamayo-R1 on Thor in FP16
  only, with an unanswered OOM report.
- **What nobody has done.** No study correlates AlpaSim or Drive-IRT scores with any
  held-out or independent outcome. No published end-to-end RL throughput profile. No
  go/no-go release rule. No distillation procedure. No evaluation of policy + guardrail.
- **How the industry gates releases** (Waymo 2025 approval guidelines, NATM, UL 4600,
  ISO/PAS 8800, UN ADS GTR adopted June 2026, UN R185 from Jan 2027): no single numeric
  gate; paired regression vs the previous release, stratified scenario groups,
  validity-checked simulation, staged rollout. Best published sim-metric correlation is
  rank ρ ≈ 0.9 between cheap and closed-loop *simulation*. **No peer-reviewed coefficient
  exists between neural-reconstruction closed-loop scores and on-road outcomes.**

**Net:** the project becomes what production lacks in the open — a credible release gate,
validated as a measuring instrument, applied to the teacher→Thor-student path NVIDIA ships.

## 1. The one thing we build

**An open release gate for the Alpamayo teacher→student path:** stratified,
validity-checked, paired-regression closed-loop evaluation on AlpaSim, whose own predictive
validity is measured; used to answer *"did distillation / quantization / RL make the
deployable model worse, and where?"*

## 2. Deliverables, in build order

| # | Deliverable | Reuse (do not rebuild) | New work | Done when |
|---|---|---|---|---|
| G1 | **Gate v1** | AlpaSim scorers, `RolloutSpec` seeds, e2e_challenge split, Drive-IRT, CAT-K traffic | Paired regression rule (per-scenario McNemar + log-clustered paired bootstrap); stratification by scenario type/ODD tag; failure taxonomy; explicit **allowed claims** list | Two checkpoints compared with a go/no-go verdict, per-stratum bounds and seed test–retest variance |
| G2 | **Validity envelope** | NuRec, OmniDreams renderer, Instant NuRec | Rendering-fidelity vs ego-deviation curve → deviation cap; renderer agreement; on-rails vs CAT-K agreement; invalid-rollout rate | Every rollout carries a validity record; out-of-envelope results reported separately |
| G3 | **Gate validity study (headline)** | Alpamayo 1/1.5/2, VaVAM, Transfuser, quantized and community variants | Rank correlation (Spearman/Kendall, bootstrap CI) of gate scores vs independent references across ≥ 10 policies; rank inversions published | First published predictive-validity number for AlpaSim, with limits stated |
| G4 | **Guardrail-in-the-loop** | AlpaSim controller | A Halos-analog rule-based monitor (TTC / drivable-area override, written in C++); gate scores policy alone vs policy+guardrail; intervention rate | Shows which failures the classical layer catches and which it cannot |
| G5 | **Teacher→student distillation** | Alpamayo 2 Super / 1.5 as teachers; AlpaGym; OPTED/RoaD ideas | Distill to a ≤ 3B student fitting 16 GiB and ≤ 0.1 s; closed-loop SFT/DAgger with teacher labels; compare with AlpaGym RL fine-tune | Student vs teacher on G1 with paired verdict and failure-mode diff |
| G6 | **Deployment precision** | ModelOpt recipe, FlashDrive | FP8/NVFP4 including the action expert; closed-loop evaluation on G1 with intervals | Per-format verdict + per-layer sensitivity |
| G7 | **AlpaGym throughput profile** | AlpaGym `perf/`, Prometheus merge | First public end-to-end RL profile; targeted upstream PRs (render-at-model-resolution, batching) instead of an RLinf fork | Numbers published; ≥ 1 PR merged or reviewed |
| G8 | **Flywheel interface** (spec only) | CoC autolabeler, Instant NuRec | Diagram + config: new failure logs → reconstruction → gate stratum → training set | Document |

**Dropped from v1:** RLinf fork as core engine; "first RL through the expert" claim;
GR00T horizontal. **Demoted:** DSRL vs AlpaGym-SDE comparison becomes an ablation in G5.

## 3. First 12 weeks

| Week | Dates | Work | Output |
|---|---|---|---|
| 1 | Sep 23–29 | Gated access (PAI-AV, NuRec 26.04, OmniDreams weights, Cosmos-Reason2). Check challenge registration (org email). AlpaSim `e2e_challenge` branch: Alpamayo 1.5 FP8 in 16 GiB, one scored scenario. AlpaGym 2-GPU smoke with `perf/` on | One scored rollout; first perf trace |
| 2 | Sep 30–Oct 6 | G1 skeleton: 441-scene val × 3 seeds for Alpamayo 1 / 1.5 / FP8; paired rule; taxonomy | First go/no-go table with test–retest variance |
| 3–4 | Oct 7–20 | G2 envelope; optional challenge submission of a gate-vetted FP8 policy (deadline Oct 31) | Validity records; optional Drive-IRT external score |
| 5–6 | Oct 21–Nov 3 | G3 policy zoo ≥ 10 checkpoints; correlation study | Validity number + inversions |
| 7 | Nov 4–10 | G4 guardrail (C++); GTC 2027 poster submission by Nov 10 with G1–G3 | Poster; intervention-rate results |
| 8–10 | Nov 11–Dec 1 | G5 distillation (8×H100 bursts) | Student on the gate |
| 11 | Dec 2–8 | G6 precision incl. action expert | Precision verdicts |
| 12 | Dec 9–14 | G7 profile + PRs; write-up; **kill check Dec 14** | Preprint draft |

**Kill check (Dec 14):** if G3 finds gate rankings do not agree with independent
references (ρ < 0.6 on ≥ 10 policies), the gate is not a valid instrument — publish that
negative result and stop building on it. If G3 holds but G5 cannot produce a ≤ 3B student
within 10% of the teacher on G1, publish G1–G4 + G6 and stop G5.

## 4. Budget (rented, Sep 2026 prices)

| Item | Estimate |
|---|---|
| Gate runs (RTX PRO 6000 renderers + H100 drivers), ~450 GPU-h | $900–1,300 |
| Policy zoo + correlation (G3), ~300 H100-h | $700–1,000 |
| Distillation (G5), 8×H100 spot ~200 h | $3,400 |
| Precision + profiling (G6–G7), ~150 H100-h | $350–500 |
| Storage (NuRec 26.04 ~2.9 TB + PAI subsets) | $150/month |
| Contingency 30% | ~$1,900 |
| **Total, 12 weeks** | **≈ $7,500–8,500** |

Prices: H100 ≈ $2.2–3.4/h (spot ≈ $0.9–2.2), RTX PRO 6000 ≈ $1.5–2.2/h, B200 ≈ $3.5–8/h.
Thor dev kit ($5,499) only if a collaborator needs on-device numbers.

## 5. Claims the gate may and may not make

**May:** "Checkpoint B is not worse than A on strata S at stated confidence (paired)";
"per-scenario failure rate in stratum S ≤ X at 95% (Clopper–Pearson; ≤ 0.3% for
0/1,000)"; "these failure modes were found" (adversarial set, reported as findings);
"gate rankings agree with reference R at ρ = … on n policies."
**Must not:** per-mile crash rates, human parity, "safe to deploy", compliance with
ISO 26262 / 21448 / PAS 8800 / UL 4600 / R157 / R171, or any on-road correlation not
actually measured.

## 6. Risks

| Risk | Check | Fallback |
|---|---|---|
| NVIDIA publishes its own gate/correlation study | Every AlpaSim/Alpamayo release | Contribute G2–G3 upstream; independent validation stays valuable |
| Challenge registration blocked (no org email) | Sep 26 | Use the public val split and Drive-IRT locally; skip submission |
| Renderer artifacts dominate at realistic deviations | Oct 13 | Tighten deviation cap; report coverage loss honestly |
| Too few diverse policies for G3 | Oct 27 | Add perturbed/quantized/early checkpoints; state n |
| Distillation compute overruns | Nov 20 | Smaller student, open-loop distillation + closed-loop DAgger only |

## 7. Key sources
- Alpamayo-R1: https://arxiv.org/abs/2511.00088 · Alpamayo 2 Super blog (Aug 4 2026):
  https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/
- AlpaGym: https://github.com/NVlabs/alpagym · AlpaSim: https://github.com/NVlabs/alpasim ·
  recipes: https://github.com/NVlabs/alpamayo-recipes · closed-loop blog:
  https://developer.nvidia.com/blog/how-to-post-train-autonomous-vehicle-models-in-closed-loop-with-nvidia-alpamayo/
- Challenge: https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026 ·
  Drive-IRT: https://github.com/kesai-labs/drive-irt
- FlashDrive: https://arxiv.org/abs/2608.12932 · OPTED: https://arxiv.org/abs/2609.20756 ·
  OmniDreams: https://arxiv.org/abs/2606.03159 · Instant NuRec: https://arxiv.org/abs/2607.14203
- Dual stack / Halos: https://blogs.nvidia.com/blog/drive-av-software-mercedes-benz-cla/ ·
  https://www.nvidia.com/en-us/ai-trust-center/halos/autonomous-vehicles/
- Waymo approval guidelines: https://arxiv.org/abs/2505.09880 · NATM:
  https://unece.org/sites/default/files/2022-05/WP.29-187-08e.pdf · nuPlan misalignment:
  https://arxiv.org/abs/2306.07962 · pseudo-sim: https://arxiv.org/html/2506.04218v2 ·
  Bosch correlation: https://arxiv.org/abs/2605.00066 · EUVS:
  https://en.papernotes.org/ICCV2025/autonomous_driving/extrapolated_urban_view_synthesis_benchmark/
- VLA-Perf: https://arxiv.org/html/2602.18397v1 · Jetson Thor specs:
  https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/
