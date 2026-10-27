> **Evidence record, 2026-10-03.** The claim ledger (§10) and the findings (§2–§9) are the evidence base for `docs/02_PHYSLOOP_PLAN.md`. Its §1 recommendation (keep the park; overlay on another program) and every reference to another program were **overridden by the owner on 2026-10-03 (DECISIONS D-009): PhysLoop is a standalone, active project.** Where this report and a live doc differ, the live doc wins.

# Rebuild PhysLoop around a falsifiable release gate

*Verified unified plan, state of the field as of 3 October 2026. Synthesized from the seven research notes in `research_notes/PhysLoop verified unified plan/`. This report proposes changes. It enacts nothing: `docs/02_PHYSLOOP_PLAN_v2.md` stays authoritative until Huy adopts changes, D-008 stays in force, and every owner-written prediction in the docs is preserved.*

PhysLoop's core thesis survives verification, but its headline result does not. **No open tool gives a paired, stratified go/no-go verdict between two Alpamayo checkpoints.** That gap is confirmed upstream, and AlpaSim already merges outside pull requests within days, so it is a real contribution worth making. **G3 as written, however, cannot be falsified.** Its implied reference, Drive-IRT, re-fits the same rollouts the gate scores. "Predictive validity" needs on-road outcomes, and only NVIDIA, Wayve and Waabi hold them. A ρ < 0.6 kill rule decides nothing at 10 policies. Five other findings reshape the plan:

- **The 2026 AlpaSim challenge is closed to Huy in practice.** Registration ends 2026-10-18 and requires an organization email.
- **NVIDIA's quantization recipe already covers the action expert**, so G6 has lost its novelty.
- **AlpaGym accepts no outside contributions**, so G7's planned pull requests have nowhere to go.
- **NVIDIA has promised a 10B→~2B distillation script three times.** If it ships, G5 is pre-empted.
- **The PAI-AV and NuRec licenses classify benchmarking and regression data as confidential.** Read literally, that could block publishing any gate number.

Restated honestly, G3 becomes a construct-validity study:

- reliability across seeds;
- agreement across renderer and traffic-model variants;
- dose–response degradation ladders;
- cross-paradigm convergence on 28–34 checkpoints;
- a discriminant check;
- a pre-registered PASS / KILL / INSUFFICIENT rule.

The proposed D-009 recommends **keeping the D-008 park until the 2026-11-20 re-entry test** and adopting this plan as the re-entry package. Three reasons:

- By D-008's own count, switching fields again would be at least the fourth pick in about a month.
- Ladders' application deadlines are already running.
- The time-critical AV options are mostly out of reach before 2026-11-20 anyway.

Any re-entry should be an overlay on Ladders' skills (kernels, FP8/NVFP4, decode, GRPO, evaluation statistics), not a replacement. The first public artifact is a synthetic-truth gate with a power analysis. It costs $0 and fits in seven days. The cost of the later stages depends on the budget tier:

- The gate, validity, precision and guardrail stages fit about $2.5k, but only at marketplace or interruptible RTX PRO 6000 rates.
- The cross-paradigm study and nanoAlpamayo need the $6k tier or a collaborator.

---

## 1. Decision D-009: keep the park, pre-build an overlay-shaped re-entry

The project is parked, not killed. D-008 (2026-09-24) froze Stage 1a and all G1–G8 work. It suspended the Dec 14 kill check and set a re-entry test for 2026-11-20: re-enter only if the six-month target flips to NVIDIA autonomous driving, or a collaborator brings data and compute (`docs/DECISIONS.md`, D-008). The live competing plan is Ladders. On 2026-10-02 Huy chose **"one program, 100% Ladders"** and fixed:

- frontier-lab apply-by dates of **2026-11-30 (target) and 2026-12-07 (hard deadline)**;
- combined compute caps of **$3,990** inside a **$6,000** planning envelope;
- **two wake-relative blocks of up to 3 active hours a day**, with 42 hours a week as a trial ceiling.

C6 adds that no funding approval is to be inferred, and line 851 of the Ladders plan records physloop as parked (`ladders/plan/PLAN.md`, header C1 and C6; line 851). Earlier AV work (Alpamayo 1.5 inference optimization, a bench harness, a FlowMatchRL spec) counts as prior experience only. None of that code exists on disk after the 2026-10-01 rebuild (`START HERE.txt`).

Verification changes the factual basis of D-008 in three places, and this matters for the decision:

- **Gated assets are no longer a blocker.** PAI-AV, the NuRec scenes, Cosmos-Reason2-2B/8B and the OmniDreams weights all use Hugging Face automatic approval. Access is immediate once the license is accepted ([PAI-AV card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles); [HF gating docs](https://huggingface.co/docs/hub/models-gated)).
- **The price depends on the provider mix.** Re-priced at verified October 2026 rates, the docs' own 12-week PhysLoop budget spans **$3.7k to $15k** (deployment notes Q5, from [RunPod](https://www.runpod.io/pricing), [Nebius](https://nebius.com/prices), [Lambda](https://lambda.ai/pricing) and [vast.ai](https://cloud.vast.ai/api/v0/bundles/) rates).
- **There is a new blocker.** The PAI-AV and NuRec licenses treat benchmarking, regression and performance data relating to the datasets as NVIDIA confidential information. They also forbid derivatives and expire 12 months after first download ([PAI-AV card and license prompt](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)).

### Proposed DECISIONS.md entry (not enacted)

> **PROPOSED — NOT ENACTED. D-009 · [date Huy adopts it] · Target, budget tier and weekly hours after the 2026-10-03 verification.**
>
> **Options (choose one).**
> - **(A) AV replaces Ladders.** Un-park PhysLoop now and reverse Ladders C1–C6.
> - **(B) AV as a domain overlay on Ladders.** Un-park a bounded, recorded slice that reuses Ladders' transferable skills: LLM/VLM training, GRPO/RL post-training, kernels, FP8/NVFP4, serving and decode, evaluation statistics. Fund it only from envelope headroom Huy explicitly approves, and only with hours Huy assigns.
> - **(C) Keep D-008's park until the 2026-11-20 re-entry test**, and adopt `reports/PhysLoop verified unified plan.md` as the re-entry package.
>
> **Recommendation: (C).** Any re-entry is B-shaped (an overlay), never A-shaped (a replacement). The choice is Huy's.
>
> **Why.**
> - Another switch costs more than waiting (§1 reasoning).
> - The AV opportunities that expire before 2026-11-20 are blocked anyway: the challenge needs an organization email and closes registration on 2026-10-18.
> - The plan's headline needed redesign regardless (G3 unfalsifiable, G6 novelty gone, G7 pull-request target wrong).
> - The licensing question must be answered before any gate number can be published.
>
> **What it changes if (C) is adopted.**
> - D-008 stays in force. `docs/02` stays authoritative. The Dec 14 kill check stays suspended.
> - This entry amends D-008's "what stays" list with three zero-dollar items. Huy adopts or rejects each one; none is in force until he does:
>   - **(a)** One written request to NVIDIA asking whether the confidentiality clause (§3) of the PAI-AV and NuRec licenses permits publishing aggregate regression results computed on those datasets. The same request should ask whether the fallback is allowed: publishing scene-ID manifests and aggregate statistics. No note adjudicates that, so the fallback is untested.
>   - **(b)** No PAI-AV or NuRec download before execution starts, so the 12-month license term (§8) does not begin early.
>   - **(c)** A sharpened re-entry test for 2026-11-20. Re-enter only if three things hold:
>     - D-008's own condition holds;
>     - either the §3 answer permits publication, or Huy accepts the fallback (publish code, scene-ID manifests and synthetic results only);
>     - a budget tier and a weekly-hours number are recorded.
> - Execution would start on D = 2026-12-08, the day after the Ladders hard deadline, unless Huy accepts overlap with the apply-by window.
>
> **Budget tier.** $0 until re-entry. At re-entry, the ≤$2.5k tier covers stages S1–S4, S6 and S8. S5 and S7 are gated on the ≤$6k tier or a collaborator.
>
> **Hours (this would close P-003 for the park period).** 0 h/week until re-entry. At re-entry Huy records a number.
>
> **Note.** If Huy prefers (B) now, the least-cost version is S1 alone: about 20 focused hours, $0, CPU only. That is still an un-park, and it must be recorded as one.

### Three options, priced in hours, dollars and switching cost

| | (A) AV replaces Ladders | (B) AV as Ladders overlay | (C) Keep park, adopt as re-entry package |
|---|---|---|---|
| D-008 status | Un-parked now; Ladders C1–C6 reversed | Partially un-parked; slice recorded in D-009 | In force; re-entry test 2026-11-20 |
| Weekly hours | 25–30 h, the unconfirmed assumption in `docs/03` §2 | 6–9 h (proposal: one Ladders-sized 3-hour block on 2–3 days a week) | 0 h until re-entry |
| Money | Repurposes the $6,000 envelope and displaces Ladders' $3,990 of caps | At most the $2,010 headroom (6,000 − 3,990); Huy's explicit approval required, since C6 says no funding approval is inferred | $0 until re-entry |
| What it buys | S1–S6 plus S8 in about 8 weeks at 25–30 h/week; S5 and S7 with the ≤$6k tier | S1, plus the S4b expert-kernel diagnosis that sits inside Ladders' kernel spine; S2–S4 slowly | A redesigned plan, a license answer, and no novelty work |
| What it costs | Orphans Ladders' 2026-11-30 and 2026-12-07 apply-by dates; at least the fourth field pick in about a month | Splits attention in Ladders' final 8 weeks; the license answer is still needed | Novelty erosion while parked (GTC Berlin 20–22 Oct 2026; monthly AlpaSim syncs); 2026 challenge forfeited, though it is already blocked |

### Why (C), reasoned from first principles

**The cost of another switch is lost compounding, not lost code.** D-001 picked autonomous driving on 2026-09-22. Two days later, D-008 parked it and cited "the third field pick in 21 days" among its reasons. Ladders became the one program on 2026-10-02, ten days after D-001 (`docs/DECISIONS.md`; `ladders/plan/PLAN.md` C1). Choosing (A) would reverse a decision made the day before this report. By D-008's own count it would be at least the fourth field change in about a month (inference). Neither folder holds code after the 2026-10-01 rebuild, so nothing is sunk. The real loss is that evidence compounds only inside one program. Each switch resets the public record to zero artifacts, while the Ladders apply-by dates, 58 and 65 days from today, keep running.

**Waiting is cheap because the expiring AV options are already out of reach.**
- **Challenge registration closes on 2026-10-18.** It requires the team captain's verified Hugging Face primary email to be an organization address, and registrations are reviewed ([challenge Space](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026)).
- **The challenge's official ranking runs on private held-out scenes.** Each team gets three official submissions per UTC month, shared across both tracks ([AlpaSim challenge README](https://github.com/NVlabs/alpasim/blob/e2e_challenge/e2e_challenge/README.md)).
- **The Nov 10 GTC 2027 poster date that the docs schedule against is UNVERIFIABLE.** No note verified it: it is tooling-notes ledger row 94, and the evaluation lane listed it as out of scope.

What waiting does risk is novelty erosion. The gate's niche (a paired verdict, stratification, seed test–retest) was NOT FOUND upstream as of 2026-10-03. AlpaSim syncs from NVIDIA's internal GitLab roughly monthly, so watching for a competing release costs minutes ([AlpaSim CONTRIBUTING](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/CONTRIBUTING.md)).

**The plan had to be redesigned regardless.** Section 2 lists the corrections. G3 cannot be falsified as written. G6's "including the action expert" is already in NVIDIA's recipe. G7's pull requests target a repository with zero external merges. G5 races an announced NVIDIA release. The park window is the cheapest time to absorb those changes and to get an answer on the license.

**Re-entry should be an overlay, because the skills transfer.**
- **The demand overlaps.** In 43 comparable NVIDIA AV postings, VLA/VLM/LLM experience appears in **49%** of qualification sections, CUDA/GPU kernels in **40%**, and RL or post-training in **23%** ([role notes, Q4, computed from NVIDIA Workday](https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite)). Those are Ladders' spines.
- **There is one concrete open AV problem inside Ladders' kernel spine.** After CUDA graphs and fusion, FlashDrive measured Alpamayo 1.5's flow-matching expert at **113.9 ms for 8 steps**, against a bandwidth floor of about **22.3 ms**. That is roughly 20% of the bound, and the cause is undiagnosed in public sources ([FlashDrive Table 1](https://arxiv.org/html/2608.12932); deployment notes Q3).
- **S1 transfers as well (inference).** The statistics core is domain-neutral: paired non-inferiority, clustered bootstrap, power. Any LLM evaluation suite needs the same machinery, and Ladders' own header lists post-training with an eval suite (V8). Whether Ladders admits that work is a Ladders decision.

**One asymmetry should be stated plainly.** The notes verify AV demand, but they contain no frontier-lab posting data. A comparison of hiring odds between the two targets is therefore not evidenced here. The recommendation rests on switching cost, option value and transfer, not on which target is "better".
- **Steelman for (A).** If the six-month target truly flips to NVIDIA AV (D-008's own criterion), (A) becomes rational. That flip should be decided at the 2026-11-20 test, not one day after C1–C6.
- **Steelman for (B) now.** S1 costs $0, transfers to Ladders, and produces a public artifact in a week.

### (ii) Budget tiers at verified October 2026 prices

Costs below are inferences. Verified prices are multiplied by GPU-hours derived from FlashDrive's measured AlpaSim per-step rollout latency: 463 ms optimized, 1,150 ms baseline, covering inference, trajectory optimization and rendering ([FlashDrive](https://arxiv.org/html/2608.12932)). The rollout count is 441 scenes × 3 seeds × 200 steps, the challenge contract ([local_evaluation README](https://github.com/NVlabs/alpasim/blob/e2e_challenge/e2e_challenge/local_evaluation/README.md)). One policy sweep therefore takes **34–85 GPU-hours single-stream**. That is an upper bound: the challenge preset runs several rollouts concurrently, and the first measured throughput in S2 replaces these figures.

| Tier | What it unlocks | Verified price basis | Fit with Ladders |
|---|---|---|---|
| **$0** | S1 (synthetic-truth gate plus power analysis); every curriculum exercise; corrections to the docs; an AlpaSim issue proposing the paired-verdict layer; the license question | CPU only | Fits every option |
| **≤ $2.5k** | S2, S3, S4, S6 and S8: about 24 policy sweeps, ≈817–2,029 GPU-hours. Estimated **$660–1,890 at interruptible RTX PRO 6000 medians ($0.81–0.93/h)**, $1,140–2,840 at the vast.ai on-demand median ($1.40/h), and $1,710–4,240 at RunPod secure ($2.09/h). Storage adds $40–216 a month for the ~2.9 TB NuRec release | RTX PRO 6000: vast.ai median $1.40 (Workstation) / $1.47 (Server); interruptible $0.93 / $0.81; RunPod $1.69 community / $2.09 secure; Nebius $1.80, preemptible from $0.79 ([deployment notes Q5](https://cloud.vast.ai/api/v0/bundles/); [RunPod](https://www.runpod.io/pricing); [Nebius](https://nebius.com/prices)). Storage: RunPod network storage above 1 TB $0.05/GB-month; Nebius object storage $0.0147/GiB-month; Nebius shared filesystem $0.08/GiB-month | The $2,010 headroom covers it only at interruptible rates, and only with Huy's approval |
| **≤ $6k** | Adds S5, the cross-paradigm study. Its reference bracket is the docs' policy-zoo line re-priced: $507 (vast.ai) to $1,350 (Nebius). Also adds an S7 pilot: an 8×H100 node costs $13.4/h (a single vast.ai offer) to $31.92/h (Lambda) to $36.00/h (Nebius), and Alpamayo 2 Super teacher labeling needs a full H100 per instance at $1.69–4.50/h | H100 on-demand: vast.ai median $1.69 (n = 5, thin supply), RunPod $2.69 / $3.49, Lambda $3.99–4.29, Nebius $4.50 since 2026-10-01 (was $3.85); spot floors $0.60–0.79 ([Lambda](https://lambda.ai/pricing); [Nebius](https://nebius.com/prices)) | Coherent only with (A), or after Ladders' spending concludes |
| **Collaborator compute** | Full G3, including private-set checks if a later challenge edition is announced (none is in the notes; registration needs an organization email); nanoAlpamayo SFT and RL; G5 at scale; possibly Thor hardware | NVIDIA Academic Grant: up to 30,000 H100 80 GB hours, or up to eight RTX PRO 6000 GPUs, for full-time faculty PIs only ([NVIDIA](https://www.nvidia.com/en-us/industries/higher-education-research/academic-grant-program/)). Nebius research credits need an academic or non-profit affiliation ([Nebius](https://nebius.com/nebius-research-grants)). Inception needs an incorporated company ([NVIDIA](https://www.nvidia.com/en-us/startups/)) | Independent of Ladders' caps |

The B200 is not needed at any tier, and its verified on-demand floor is $5.98/h, not the docs' $3.5. The Jetson AGX Thor developer kit costs **$5,499**, up from its $3,499 launch price ([NVIDIA Jetson FAQ](https://developer.nvidia.com/embedded/faq)). It would consume almost the entire ≤$6k tier, so it belongs only to the collaborator tier.

### (iii) Weekly hours: P-003 is still open

| Option | Weekly hours (proposal) | Basis | What it does to the schedule |
|---|---|---|---|
| (C) during the park | 0 | D-008 | Nothing moves until re-entry |
| (B) overlay | 6–9 | One 3-hour Ladders block on 2–3 days (`ladders/plan/PLAN.md` C1) | The §3 windows stretch by about 3–4.6× (27.5 ÷ 9 to 27.5 ÷ 6) |
| (A) or post-re-entry | 25–30 | `docs/03` §2 assumption, unconfirmed | The §3 windows apply as written |
| Ceiling | 42 | Ladders C1 trial ceiling for all work combined | Upper bound; Ladders reforecasts after three actual workdays |

The §3 schedule is expressed as D+k weeks at 25–30 h/week. Huy should record the real number in D-009 and re-baseline, mirroring Ladders' "reforecast after three actual workdays" rule. Every effort estimate in §3 is a planning inference, to be re-forecast after S1's actuals.

---

## 2. Verification overturned G3 and six other load-bearing assumptions

Seven research lanes re-checked every claim in docs 01–05 against primary sources accessed on 2026-10-03. The bibliography is sound: **every arXiv ID on the docs' reference shelf resolves to the claimed paper**, 13 of 13 plus EUVS ([curriculum notes, shelf check](https://arxiv.org/abs/2511.00088)). The worked statistics in C6 and C8 are arithmetically correct. What breaks is the framing of the headline, several upstream assumptions, and the economics. The corrections that change the plan follow.

| # | Finding | What it changes | Source |
|---|---|---|---|
| 1 | **G3 cannot be falsified as written.** No reference is named. Drive-IRT is fitted to the same local AlpaSim rollouts the gate scores, so ρ(gate, Drive-IRT) measures aggregation robustness, not validity. "Predictive validity" needs an on-road criterion that only NVIDIA, Wayve and Waabi hold | G3 is restated as construct validity (§4) | [AlpaSim local_evaluation README](https://github.com/NVlabs/alpasim/blob/e2e_challenge/e2e_challenge/local_evaluation/README.md); [Cronbach & Meehl 1955](https://doi.org/10.1037/h0040957) |
| 2 | **The kill rule cannot decide at n = 10.** An observed ρ = 0.8 has a 95% CI of about [0.24, 0.96]. When the true ρ is 0.8 the rule kills wrongly 15.8% of the time; when the true ρ is 0.4 it passes wrongly 24.3% of the time. A lower bound ≥ 0.6 at an observed 0.8 needs **n ≈ 28–34** | Replaced by a pre-registered three-way rule (§4) | [Bonett & Wright 2000](https://doi.org/10.1007/BF02294183); [Fieller, Hartley & Pearson 1957](https://doi.org/10.1093/biomet/44.3-4.470) |
| 3 | **"Nothing exists" is wrong, and the novelty claim must narrow.** Sim2Val (CoRL 2025) reports per-episode Pearson ρ > 0.9 between NVIDIA's neural-reconstruction simulator and on-road logs. That result covers one policy, two kinematic metrics, 138 + 75 paired episodes and an 8-second closed loop. NVIDIA's blog names the simulator as AlpaSim. Separately, OmniDreams kept the ranking of 4 policies when NuRec was swapped out, but gives no coefficient | "First predictive-validity number" becomes "first open, multi-policy reliability and convergent-validity study" | [Sim2Val](https://arxiv.org/abs/2506.20553); [NVIDIA blog](https://developer.nvidia.com/blog/building-autonomous-vehicles-that-reason-with-nvidia-alpamayo/); [OmniDreams §9.4.1](https://arxiv.org/html/2606.03159) |
| 4 | **Bosch's ρ = −0.36 does not reproduce.** It rests on n = 7 with p = 0.43, and recomputing from the paper's own Table 1 gives ρ = −0.143 (p = 0.76). Its ρ = 0.90 (n = 8) does reproduce. Pseudo-simulation's r = 0.89 is a Pearson sim-to-sim correlation over 83 privileged planners against nuPlan's reactive simulation | Cite as "no significant correlation (n = 7)". No cited coefficient links any simulator score to real-world outcomes | [Bosch 2605.00066](https://arxiv.org/html/2605.00066v1); [pseudo-sim](https://arxiv.org/html/2506.04218v3) |
| 5 | **The statistics need repair.** Exact McNemar ignores road clustering, which makes it anti-conservative. "≤0.3% for 0/1,000" is a **one-sided** bound (0.2991%); the two-sided bound is 0.3682%, and a two-sided ≤0.3% needs 1,228 failure-free scenes. At 441 scenes and 4% discordance, 80% power detects about **2.72 pp**. The docs' own 12-vs-6 example has power **0.22**. The smallest demonstrable non-inferiority margin is about **2.67 pp** | Use a cluster-adjusted paired non-inferiority test with a pre-registered margin, and label one-sided bounds as one-sided | [Durkalski et al. 2003](https://doi.org/10.1002/sim.1438); [Clopper & Pearson 1934](https://doi.org/10.1093/biomet/26.4.404); [Connor 1987](https://doi.org/10.2307/2531961) |
| 6 | **Stage 1a's false-GO target can be met only by abstaining.** A 2% stratum holds about 9 scenarios and has at most 0.2% power. Zero failures in 9 still leaves a 28.3% one-sided upper bound | INSUFFICIENT must block GO, and strata need minimum sizes | [McNemar 1947](https://doi.org/10.1007/BF02295996) (computed in the evaluation notes) |
| 7 | **The challenge is closed in practice.** Registration closes **2026-10-18** and requires an organization email, with affiliations reviewed. There are **3 official submissions per UTC month**, shared across two tracks (PAI-AV/NuRec and nuPlan/MTGS). The 441 scenes are a public local-validation split; the official ranking runs on private held-out scenes. The board closes 2026-10-31 | No challenge path for an unaffiliated solo engineer; at most 3 private-set data points | [challenge Space](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026); [KE:SAI blog](https://kesai.eu/blog/2026-09-08-common-task-framework/) |
| 8 | **The memory and latency limits are not symmetric.** **16 GiB is a hard per-instance limit; 0.1 s per Drive call is only a target**, policed through an unpublished per-track throughput budget. Images may be at most 40 GiB, with no outbound network and a read-only root filesystem. Alpamayo 1.5 in BF16 (20.6 GiB of weights) does not fit; FP8 (about 10.3 GiB) does | Distillation or precision work aims at a memory cap, not a latency gate | [challenge README](https://github.com/NVlabs/alpasim/blob/e2e_challenge/e2e_challenge/README.md) |
| 9 | **A lateral exit of 4 m or more scores as failure.** The scene score is 0 for an at-fault collision (front or lateral; rear collisions are never at fault), off-road driving, or `left_corridor_laterally` ≥ 4 m; otherwise it is min(progress/0.8, 1). NVIDIA's papers instead *exclude* events beyond 4 m. Close encounters have no scorer on the challenge branch, and stall, oscillation and red-light have no scorer at all | G2 must split validity exits out of policy failures and report both 2 m and 4 m; the taxonomy must be re-grounded | [AlpaSim e2e branch](https://github.com/NVlabs/alpasim/blob/e2e_challenge/e2e_challenge/README.md); [AR1 §6.1](https://arxiv.org/html/2511.00088v2) |
| 10 | **Drive-IRT intervals do not mean what the docs imply.** The model is a zero-and-one-inflated Beta IRT with per-item discrimination. Its 95% intervals come from a mean-field Gaussian variational posterior, which its own docs warn can under-report spread. Rank intervals are Monte Carlo draws. Seed and scene-sampling variance are excluded. Ability is relative to the panel, at least 6 subjects are needed, there is no external validation, and the repository has no license | PCS is a poor statistic for a paired B-vs-A decision. G1 complements Drive-IRT rather than duplicating it | [drive-irt](https://github.com/kesai-labs/drive-irt); [docs ch. 6](https://ln2697.github.io/navhard-leaderboard-2/docs/preliminary/06_parameter_uncertainty_quantification.html) |
| 11 | **AlpaGym is closed to outsiders.** Its first public commit was 2026-06-16. It trains only the expert, with the VLM frozen and gradients stopped. `sde_type="cps"` raises NotImplementedError. The expert-RL code sits on unmerged recipes PR #18. The mirror has been code-frozen since 2026-07-08. The default is an 8-policy-step smoke preset with the `distance_to_gt` reward. There have been zero external PRs, and contributions need DCO sign-off and two internal reviewers | No pull-request path. G7 becomes a measurement report; RL through the expert belongs to NVIDIA | [first commit](https://github.com/NVlabs/alpagym/commit/92ba5cea2234e5407d0977f7b96a7c854f32e039); [default.yaml](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/default.yaml); [PR #18](https://github.com/NVlabs/alpamayo-recipes/pull/18) |
| 12 | **The upstream gap is narrow but real.** AlpaSim ships per-rollout scorers, `results-summary.json`, seeds (re-added by external PR #149 on 2026-08-17), a leakage-aware 1,761/441 split (community PR #167) and Drive-IRT ranking. It has **no** paired verdict, no stratification (0 code hits), no seed test–retest, no renderer or traffic agreement, and no monitor implementation. It has about 11 external merges, mostly within 0–6 days, and no CLA or DCO | G1, G2 and G4 become thin layers that can be upstreamed to AlpaSim | [tooling gap table](https://github.com/NVlabs/alpasim/pulls?q=is%3Apr+is%3Amerged); [PR #149](https://github.com/NVlabs/alpasim/pull/149) |
| 13 | **G6 has lost its novelty.** NVIDIA's ModelOpt 0.43 recipe already quantizes expert layers: FP8 model-wide except `patch_embed`, with AutoQuant sensitivity measured by flow-matching MSE. It evaluates open-loop minADE only and publishes no accuracy or latency numbers | G6 becomes closed-loop per-format verdicts plus published sensitivity | [quant utils.py](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_5_quant/utils.py); [quant README](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_5_quant/README.md) |
| 14 | **G5 races NVIDIA.** There is no public distillation recipe or NVIDIA student checkpoint. NVIDIA has promised one three times: the Edge-LLM blog (2026-03-12), the closed-loop blog (2026-05-31), and AlpaGym's onboarding doc, which targets "e.g. 2B" on one GPU. GTC Berlin (20–22 Oct 2026) is a near-term risk | G5 merges into nanoAlpamayo, with a kill clause that pivots to evaluating NVIDIA's student | [AlpaGym ONBOARDING](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/docs/ONBOARDING.md); [NVIDIA blog](https://blogs.nvidia.com/blog/alpamayo-2-super/) |
| 15 | **Alpamayo 2 Super's size is reported inconsistently.** The card field says a 32B backbone plus a 2.3B expert. The card prose and blog say a 2B expert and 34B total. The safetensors hold 35.81B parameters. The 1 Jun press release said 32B, and one NVIDIA blog says 3× the 10B models. It peaks at 72,115 MiB on one H100 80 GB at batch 1 | Cost teacher labeling at a full H100 per instance; cite the card field and note the conflict | [A2S card](https://huggingface.co/nvidia/Alpamayo2-Super/blob/00554695e729a6ff0b6281fd2c81b18d06e33dbe/README.md); [A2S blog](https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/) |
| 16 | **Jetson Thor and DRIVE Thor are different devices.** Jetson AGX Thor T5000: 2,070 sparse FP4 TFLOPS, 128 GB, 273 GB/s. DRIVE AGX Thor dev kit: 1,000 INT8 TOPS / 2,000 FP4, 64 GB, 273 GB/s, DriveOS 7, designed for ASIL-D, with GPU memory set by carveout. FlashDrive's Thor numbers come from Jetson (943.6 ms per tick, fully optimized). The challenge's 16 GiB rule has nothing to do with Thor | Every Thor claim names the device | [Jetson Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/); [DRIVE AGX](https://developer.nvidia.com/drive/agx) |
| 17 | **The Thor toolchain story is mixed.** TensorRT-Edge-LLM supports only Alpamayo-R1-10B, with FP16 export. Alpamayo 1.5 support, promised for August, is still absent at v0.11.0. The "unanswered" OOM thread *was* answered by NVIDIA, and the user reported it working on 2026-10-01. TensorRT 11.3 lists FP8/FP4 on Thor (SM 11.0), but users report that ModelOpt checkpoints run as fake quantization on Thor | Thor precision claims are labeled as emulated unless a real engine is built | [Edge-LLM issue #130](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/130); [forum 382501](https://forums.developer.nvidia.com/t/drive-agx-thor-with-alpamayo/382501); [TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html) |
| 18 | **Prices have moved.** H100 on-demand now runs **$1.60–4.50/h**. B200 runs $5.98–9.39/h (no listing at $3.5). RTX PRO 6000 runs $1.40–2.19/h. The docs' $150/month storage figure matches only RunPod storage above 1 TB. Nebius raised H100 prices 17% on 2026-10-01 | Every stage is re-priced, with fallback rates | [deployment notes Q5](https://nebius.com/prices); [Lambda](https://lambda.ai/pricing) |
| 19 | **The cost model was wrong.** FlashDrive's 717 → 151 ms comes mostly from streaming KV reuse, speculative reasoning, adaptive steps and CUDA graphs; **W4A8 removes only 24.6 ms**. The docs' per-tick split is wrong in all three stages: Alpamayo 1.5 reasons about 16 tokens, not 40, and its expert runs 4–7× slower than its bound. At B = 32 the expert costs about 7 ms per sample, not 0.1 ms. Rollout throughput is about 4 ticks/s per GPU, not 6–9. VLA-Perf's numbers are roofline models, not measurements. AR1's 99 ms does not describe the released 10B model | The cost model is redone in S8, and budgets assume about 4 ticks/s until measured | [FlashDrive](https://arxiv.org/html/2608.12932); [VLA-Perf](https://arxiv.org/html/2602.18397v1) |
| 20 | **Instrument and target numbers are unstable.** The Alpamayo 1.5 card's AlpaSim score moved **0.81 → 1.37 on 2026-08-07 with no weight change**. The nanoAlpamayo targets come from two AR1 tables: Table 10 on public data, Table 9 on internal data. The 644- and 920-item lists are unpublished, and the 0.5B weights are unreleased | §5 re-scopes nanoAlpamayo; G3 re-runs every policy under one pinned protocol | [A1.5 card, earlier version](https://huggingface.co/nvidia/Alpamayo-1.5-10B/blob/22c33224a78a3d9343b954c8259a12d36cc276c8/README.md); [A1.5 card, current version](https://huggingface.co/nvidia/Alpamayo-1.5-10B/blob/7aba8293c09993f2e125c6819df05d7fa3e873ea/README.md); [AR1 Tables 9–10](https://arxiv.org/html/2511.00088v2) |
| 21 | **The licenses split in two directions.** Model weights have been under OpenMDW-1.1 since 2026-08-04. The PAI-AV and NuRec licenses are different: internal AV development using NVIDIA technology only (§1); benchmarking, regression and performance data treated as confidential (§3); no derivatives or distribution (§4.6); expiry 12 months after first download (§8). Public CoC labels cover 1,450 training clips (1,728 labels) and 290 validation clips (349 labels) | A publication gate before any number derived from PAI-AV or NuRec; downloads delayed; D-008's "requested, not granted" premise is outdated | [PAI-AV card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles); [NuRec card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) |
| 22 | **Regulation and industry practice are different things.** The ADS GTR and its companion regulation were adopted on 2026-06-24. Draft UN Regulation No. [185] is notified and enters into force around **2027-01-30** unless more than one-fifth of Contracting Parties object; it is not in force today. None of NATM, the GTR, R[185], R157 or R171 prescribes paired regression against the previous release, or staged rollout. Waymo uses release-vs-prior regression only as a diagnostic. Zoox gates releases on one combined quantitative risk estimate | Rewrite 02 §0. Paired regression is PhysLoop's own design choice. Add R[185] and the GTR to the no-compliance list | [NHTSA 91 FR 48485](https://www.govinfo.gov/content/pkg/FR-2026-07-31/html/2026-15483.htm); [C.N.288.2026](https://treaties.un.org/doc/Publication/CN/2026/CN.288.2026-Eng.pdf); [Waymo 2505.09880](https://arxiv.org/abs/2505.09880); [Zoox 2026](https://zoox.com/assets/1786550724-zoox-safety-case-framework-2026.pdf) |
| 23 | **The role mapping is stale.** Role A (JR2010863) is live: L5 $224,000–356,500, L6 $272,000–431,250, 12+ years. Its simulation-quality duty includes *measuring*, a word the mirror the docs used had dropped. Role B's text is live again as a Senior role in China (JR2019758: PhD + 2 years, or MS + 4 years). Role C is live again (JR2005596), without the Alpamayo or OPTED tags. No US Principal end-to-end posting is live | Re-map everything to live posting IDs (§8) | [JR2010863](https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Senior-Machine-Learning-and-Simulation-Engineer---Autonomous-Vehicles_JR2010863); [JR2019758](https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/China-Shanghai/Deep-Learning-Senior-Engineer--End-To-End-Autonomous-Driving_JR2019758); [JR2005596](https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Senior-Research-Scientist---Autonomous-Vehicles_JR2005596) |
| 24 | **Several algorithm citations are mis-framed.** OPTED trains a privileged *teacher* with RL, then supervises the camera student on-policy, using about 1,000× fewer simulator interactions than direct RL; it was tested on TransFuser and VaVAM. RoaD is closed-loop *supervised* fine-tuning (+41% driving score, −54% collisions). No source uses n = 12, and RLinf's 1.9× has no source. AR1's GRPO is unclipped and softmax-weighted, unlike AlpaGym's clipped objective | S7 tries RoaD/OPTED-style correction first and names its estimator | [OPTED](https://arxiv.org/abs/2609.20756); [RoaD](https://arxiv.org/abs/2512.01993); [AR1 Eq. 10](https://arxiv.org/html/2511.00088v2) |

**What survives intact.** The negative claims that matter all held on 2026-10-03:

- No open go/no-go release rule exists; AlpaSim aggregates descriptive statistics and Drive-IRT ranks a panel.
- No public distillation procedure exists.
- No stage-level RL throughput profile exists. Only coarse community wall-clock numbers do (2 × L40S, 10 steps in about 32 minutes).
- No published policy-plus-guardrail evaluation was found. The hook exists ([tooling notes §9](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/eval/src/eval/scorers/safety.py); [Classmethod](https://dev.classmethod.jp/articles/nvidia-alpagym-on-aws/)).

Several positive claims also held:

- NVIDIA positions Alpamayo as a teacher distilled into compact models on DRIVE AGX Thor ([A2S blog](https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/)).
- DRIVE AV pairs an end-to-end stack with a parallel classical safety stack, for the Level 2 CLA and the L4-ready S-Class ([CLA blog](https://blogs.nvidia.com/blog/drive-av-software-mercedes-benz-cla/); [S-Class blog](https://blogs.nvidia.com/blog/mercedes-benz-l4-s-class-drive-av-platform/)).
- The memory arithmetic holds: 22.16 GB in BF16, about 11 GB in FP8 ([quant recipe](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_5_quant/README.md)).
- The environment pins hold: torch 2.8.0, vLLM 0.11.0, transformers 4.57.1, ModelOpt 0.43.0 ([RL recipe pyproject](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_x_rl/pyproject.toml)).

The thesis "build the measuring instrument, not another model" is, if anything, stronger. The instrument problem is visible in NVIDIA's own model cards.

---

## 3. Eight stages, one dependency graph, re-baselined to the decision date

The unified plan merges three things into one graph:

- the PhysLoop deliverables (G1–G8, `docs/02` §2);
- the mastery stages (`docs/03` §3–§4);
- the corrections from §2.

Stage 0 (foundations) no longer stands alone. Its derivations are embedded in the stages that use them, as the brief asks. Mastery Stage 6 (the release note) folds into S4. Mastery Stages 5 and 7 (data flywheel, multi-node scale) move beyond this plan's horizon and survive as G8's specification and as named gaps (§8).

Time is counted from **D**, the date Huy adopts D-009, not from 23 September 2026. Windows assume the unconfirmed 25–30 h/week in `docs/03` §2 (P-003). Effort figures are planning inferences, to be re-forecast after S1.

```
S1 ──► S2 ──┬──► S3 ──┬──► S4 ──┬──► S5   G3 verdict + preprint (needs S2, S3, S4)
            │         │         └──► S7   nanoAlpamayo student (needs S2, S3, S4)
            │         └──► S6             guardrail in C++ (needs S2, S3)
            └──► S8                       throughput report (telemetry from S2–S4)

Publication gate between S2 and any public number derived from PAI-AV or NuRec:
written answer on license §3, or the fallback (code + scene-ID manifests + synthetic results).
```

| Stage | Serves | Window | Effort (focused h, inference) | Tier | Calendar if D = 2026-12-08 | Check at stage end |
|---|---|---|---|---|---|---|
| S1 Gate core on synthetic truth + power | G1 core; mastery 1a | D+0 → D+1 | ~20 | $0 | 2026-12-08 → 12-15 | KC-1 |
| S2 Gate v1 on AlpaSim, reliability first | G1; G3·R1; G2 records; mastery 1b | D+1 → D+3 | ~40 | ≤ $2.5k | 12-15 → 12-29 | KC-2; KC-L before the first publication |
| S3 Validity envelope + inter-rater agreement | G2; G3·R2–R4 | D+3 → D+5 | ~30 | ≤ $2.5k | 12-29 → 2027-01-12 | KC-3 |
| S8 Throughput report + flywheel spec | G7; G8 | D+3 → D+8 | ~15 | ≈ $0 marginal | 12-29 → 2027-02-02 | — |
| S4 Known-group ladders + precision verdicts | G3·R7–R9; G6; mastery 6 | D+4 → D+7 | ~40 (+~20 for S4b) | ≤ $2.5k | 2027-01-05 → 01-26 | KC-4 |
| S6 Guardrail-in-the-loop in C++ | G4 | D+6 → D+8 | ~35 | ≤ $2.5k | 2027-01-19 → 02-02 | KC-6 |
| S5 Cross-paradigm convergence + verdict | G3·R5, R6, R9b | D+8 → D+11 | ~60 | ≤ $6k or collaborator | 2027-02-02 → 02-23 | KC-5 |
| S7 nanoAlpamayo as the gate-measured student | G5 merged; mastery 2–3 | D+11 → D+20+ | ~150 | ≤ $6k or collaborator | 2027-02-23 → 04-27+ | KC-7 |

The calendar shifts in two cases:

- **Option (B) adopted now (D = 2026-10-05).** At 25–30 h/week, D+1 is 2026-10-12, D+8 is 2026-11-30 and D+11 is 2026-12-21. At the overlay pace of 6–9 h/week, multiply every duration by about 3–4.6.
- **The earliest Option (C) start (D = 2026-11-23).** This collides with Ladders' apply-by window (target 2026-11-30, hard deadline 2026-12-07).

**How to read the stage cards.**

- **The L5 symptom** comes from verified sources, not a hypothetical.
- **The descent** runs L4 system → L3 math → L2 numerics and memory → L1 silicon. Items marked ★ are the derivations and instruments Huy owns.
- **Prediction slots are deliberately blank.** Huy fills them before any run (`experiments/TEMPLATE.md`).
- **Cost** is verified price × derived hours. Every derived number is an inference.

### S1 — Gate decision core on synthetic truth, plus power analysis ($0, CPU)

| Field | Content |
|---|---|
| Depends on | Nothing. Needs no dataset license, so the PAI-AV clock does not start |
| L5 symptom (verified) | A release decision on 441 paired scenes cannot see small regressions. At 4% discordance, 80% power detects only about **2.72 pp**, and the docs' own 12-vs-6 example has power **0.22**. A regression confined to a 2% stratum (about 9 scenes, at most 0.2% power) dilutes from 2.1 pp to **0.042 pp** pooled, so it hides inside a passing aggregate. Computed in the evaluation notes ([Connor 1987](https://doi.org/10.2307/2531961); [McNemar 1947](https://doi.org/10.1007/BF02295996)) |
| L4 system | The gate as a decision function over manifests: an identity tuple, `is_pair()`, and `verdict()` returning GO, NO-GO or INSUFFICIENT. **INSUFFICIENT blocks GO** |
| L3 math ★ | Exact McNemar (b ~ Bin(b+c, ½)). Paired non-inferiority with a pre-registered margin ([Blackwelder 1982](https://doi.org/10.1016/0197-2456(82)90024-1); ICH E9; FDA's 2016 non-inferiority guidance, 81 FR 78605, cited in the evaluation notes without a link). Tango score CI, which handles zero discordant cells ([Tango 1998](https://doi.org/10.1002/(SICI)1097-0258(19980430)17:8<891::AID-SIM780>3.0.CO;2-B)). An intersection–union test across strata, so family-wise false-GO ≤ α without adjustment ([Berger 1982](https://doi.org/10.1080/00401706.1982.10487790)). Holm step-down for regression flags (the Holm 1979 text could not be verified online). Cluster-adjusted McNemar ([Durkalski 2003](https://doi.org/10.1002/sim.1438)) or a road-level paired bootstrap ([Field & Welsh 2007](https://doi.org/10.1111/j.1467-9868.2007.00593.x)). A wild cluster bootstrap-t when there are 5–30 clusters ([Cameron, Gelbach & Miller 2008](https://doi.org/10.1162/rest.90.3.414)). The zero-failure bound p_U = 1 − α^(1/n), one-sided |
| L2 numerics ★ | Exact binomial tails in integer arithmetic (31,180 / 262,144); bootstrap RNG seeding for deterministic verdicts; float64 tail sums |
| L1 silicon | CPU only. A 1,000-run Monte Carlo × bootstrap resamples, vectorized |
| Build artifact | `gate/`: the decision core; a synthetic generator with known effect sizes, road clusters and 441-scene strata; the Monte Carlo; power tables; a known-answer test suite (detailed in §6). It consumes per-rollout pass/fail and scene scores keyed by identical scene IDs, which is the content AlpaSim's `results-summary.json` provides ([AlpaSim OPERATIONS.md](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/docs/OPERATIONS.md)) |
| Acceptance test | The known-answer tests reproduce the verified values: one-sided 0.11894 and two-sided 0.23788 for 12 vs 6; 0.677% (one-sided) and 0.833% (two-sided) for 0 failures in 441; 0.2991% and 0.3682% for 0 in 1,000; MDE 1.94 / 2.72 / 3.90 pp at ψ = 2.1 / 4 / 8%; non-inferiority margins 2.67 / 3.77 pp; power 0.22 for the docs' example. The Monte Carlo shows (a) INSUFFICIENT, never GO, for a 9-scene stratum; (b) higher false-GO when stratification is removed; (c) false-GO ≤ α wherever power exists |
| Predictions (Huy, blank) | P-S1-1 false-GO rate for a −30% effect on a 2% stratum at 7% base failure: ____ · P-S1-2 share of INSUFFICIENT verdicts under the null: ____ · P-S1-3 change in false-GO when stratification is removed: ____ |
| Kill criterion | **KC-1, D+1 (2026-12-15 if D = 2026-12-08):** if `verdict()` cannot hold false-GO ≤ α on synthetic truth, fix the statistics before any rented GPU-hour (`docs/03` §5 kill, preserved). Time box: if S1 exceeds twice its estimate, ship the power-analysis half alone (inference) |
| Cost | $0 |
| Public contribution and why upstream values it | AlpaSim aggregates mean, standard deviation and quantiles, and Drive-IRT ranks a panel of six or more subjects. No paired two-checkpoint verdict, stratification or test–retest exists upstream ([tooling gap table G1](https://github.com/NVlabs/alpasim/blob/20235345d7ea9eed5d5156099cf219cd35b2e4dc/e2e_challenge/local_evaluation/README.md)). The challenge's own 95% rank intervals leave PAI teams #2–#5 statistically tied ([live leaderboard](https://nvidia-alpasime2eclosedloopchallenge2026.hf.space/gradio/)), so a published "what 441 scenes can certify" table is useful to every participant. Delivery: an AlpaSim issue first. External PRs merge in 0–6 days, with no CLA ([AlpaSim CONTRIBUTING](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/CONTRIBUTING.md)) |
| In-demand skill evidenced | Evaluation and metrics: 28% of NVIDIA qualification sections and 51% of responsibility sections; 39% and 57% at competitors ([role notes Q4](https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite)). Postings: JR2014651 (evaluation products; precision and recall of metrics); JR2024273 (closed-loop benchmarks; regression prevention); JR2025038 (tools for regression performance); [Zoox Simulation Evaluation & Validation](https://jobs.lever.co/zoox/2ef807a0-f80d-47ba-a1a1-578ebe195592) (applied statistics with CIs and p-values) |
| Curriculum | **Sources:** McNemar 1947; Clopper & Pearson 1934; [Hanley & Lippman-Hand 1983](https://doi.org/10.1001/jama.1983.03330370053031); [Efron 1979](https://doi.org/10.1214/aos/1176344552); Cameron, Gelbach & Miller 2008. **Derivation:** the exact McNemar null and the zero-failure bound (1 − p)ⁿ = α ⇒ p ≈ 3/n. **Exercise (known answer):** A fails 30, B fails 24, b = 12, c = 6 ⇒ Σ_{k≤6} C(18,k) = 31,180; one-sided p = 0.11894; two-sided p = 0.23788. **Trap:** the uncorrected χ² gives p = 0.157 and an unpaired two-proportion z-test gives p = 0.399. Calling 3/n a "95% CI" is also wrong: it is one-sided, and the two-sided upper limit at n = 441 is 0.833% ([curriculum notes P5](https://doi.org/10.1007/bf02295996)) |
| Ownership | ★ Huy: `verdict()` logic, margin choice, cluster bootstrap, power derivations, predictions. Agents: generator harness, Monte Carlo runner, CI, plots, packaging |

### S2 — Gate v1 on AlpaSim, reliability first

| Field | Content |
|---|---|
| Depends on | S1. The license publication gate (KC-L) sits between S2 and any public number |
| L5 symptom (verified) | NVIDIA's own Alpamayo 1.5 card changed its AlpaSim score from **0.81 ± 0.01 (910 scenarios) to 1.37 ± 0.10 (913) on 2026-08-07, with the weights unchanged**, and the card never defines "±" ([card at 22c3322](https://huggingface.co/nvidia/Alpamayo-1.5-10B/blob/22c33224a78a3d9343b954c8259a12d36cc276c8/README.md); [card at 7aba829](https://huggingface.co/nvidia/Alpamayo-1.5-10B/blob/7aba8293c09993f2e125c6819df05d7fa3e873ea/README.md)). An NVIDIA maintainer says NuRec has small run-to-run variation, and the traffic model ignores the seed ([AlpaSim issue #128](https://github.com/NVlabs/alpasim/issues/128)) |
| L4 system | AlpaSim gRPC microservices: runtime hub, renderer (default NRE-GA 26.04), driver, controller (linear MPC by default; the challenge uses nonlinear MPC), physics, and optional CAT-K traffic (disabled by default). Seeds live in `RolloutSpec.random_seed` on `main` (rollout k uses seed + k; re-added on 2026-08-17 by external PR #149) and in `DriveSessionRequest.random_seed` on the challenge branch. **Pin both SHAs** ([runtime.proto](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/grpc/alpasim_grpc/v0/runtime.proto); challenge-branch path `src/grpc/alpasim_grpc/v0/egodriver.proto`, per the evaluation notes) |
| L3 math ★ | The VIM (International Vocabulary of Metrology) distinction between repeatability (seeds), intermediate precision (GPU, driver and AlpaSim-version drift) and reproducibility (another site or renderer) ([JCGM 200:2012](https://www.bipm.org/documents/20126/2071204/JCGM_200_2012.pdf)). ICC(2,1) with a 95% CI, read against the Koo & Li bands: < 0.5 poor, 0.5–0.75 moderate, 0.75–0.9 good, > 0.9 excellent ([Shrout & Fleiss 1979](https://doi.org/10.1037/0033-2909.86.2.420); [Koo & Li 2016](https://doi.org/10.1016/j.jcm.2016.02.012)). Kendall's W, flagged when ≲ 0.8 ([Drive-IRT docs ch. 6](https://ln2697.github.io/navhard-leaderboard-2/docs/preliminary/06_parameter_uncertainty_quantification.html)). The design effect 1 + (m − 1)ρ. The scenario-level discordance noise floor |
| L2 numerics | Sources of nondeterminism: NuRec run-to-run variation, unseeded CAT-K, JPEG delivery at 1916/1920 × 1080. Alpamayo prefill intermittently emits an all-NaN trajectory (AlpaGym config comment), and retries can bias scores, so log every retry ([AlpaGym policy config](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/configs/policy/alpamayo_r1.yaml)) |
| L1 silicon | Record the RTX PRO 6000 edition: Workstation at 1,792 GB/s vs Server at 1,597 GB/s, which changes latency comparability ([Workstation datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf); [Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)) |
| Build artifact | A G1 v1 report on three released policies (Alpamayo 1, Alpamayo 1.5 BF16, Alpamayo 1.5 FP8 from the ModelOpt recipe), run on 441 `nurec_curated_val` scenes × ≥ 3 seeds. It contains: per-stratum paired verdicts and one-sided bounds; seed test–retest; a negative control (a bit-identical re-export must not be flagged); per-rollout validity records (deviation bins < 1, 1–2, 2–4 and ≥ 4 m, with `left_corridor_laterally` split out); a failure taxonomy that maps AlpaSim's native scorers (`collision_at_fault` = front ∨ lateral, `collision_rear`, `offroad`, `wrong_lane`, `min_distance_to_obstacle_m`, `plan_deviation`, `safety_monitor_triggered`) onto NHTSA pre-crash numbers; manifests (git hash, config, seeds, SHAs); and the **first measured throughput**, which re-baselines every cost in this plan. AlpaSim's Prometheus telemetry is saved for S8 |
| Acceptance test | ICC and W reported with CIs; the noise floor reported; the negative-control flag rate reported; every rollout carries a validity record; deviation reported at both 2 m (the docs' cap) and 4 m (NVIDIA's convention: `max_dist_to_gt_trajectory: 4.0` in the challenge-branch config `src/wizard/configs/e2e_challenge/ec2.yaml`; [AR1 §6.1](https://arxiv.org/html/2511.00088v2)) |
| Predictions (Huy, blank) | Preserved from `docs/03` Stage 1b: (i) test–retest discordance rate of Alpamayo 1.5 across seeds: ____; (ii) percentage of rollouts beyond the 2 m and 4 m caps, and whether it differs between good and bad policies: ____. New slot: the pre-registered reliability floor: ____ |
| Kill criterion | **KC-2, D+3:** if policy-level reliability falls below Huy's pre-registered floor, stop. Validity cannot exceed reliability (evaluation notes, G3 protocol step 1). **KC-L:** publish no number derived from PAI-AV or NuRec until there is a written answer on §3. The fallback is code, scene-ID manifests and synthetic results. Whether manifests and aggregates fall under §3 is unadjudicated (inference), so the same written request asks ([PAI-AV license prompt](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)). Budget: stop at the S2 line |
| Cost (inference) | 4 sweeps × 34–85 GPU-h ⇒ **$110–707** across the interruptible-to-RunPod-secure RTX PRO 6000 range, plus **$40–216/month** of storage for the ~2.9 TB NuRec 26.04 release. The size of the 441-scene subset is unpublished ([NuRec card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec)) |
| Public contribution and why upstream values it | The first public seed test–retest and reliability numbers for AlpaSim. Upstream averages repeated rollouts and reports no test–retest. Drive-IRT's intervals exclude seed and scene-sampling variance, and NVIDIA's own card score moved 70% with the protocol ([local_evaluation README](https://github.com/NVlabs/alpasim/blob/20235345d7ea9eed5d5156099cf219cd35b2e4dc/e2e_challenge/local_evaluation/README.md)) |
| In-demand skill evidenced | JR2010863, Role A: its simulation-quality duty includes measuring, and closed-loop evaluation of end-to-end models is preferred. JR2019461 (validity, reproducibility, dataset hygiene, leakage prevention, model comparison). JR2014651. [Waymo Staff TLM, Simulator Evaluation](https://careers.withwaymo.com/jobs?gh_jid=8209790). Simulation appears in 35% of NVIDIA qualification sections and 49% of responsibility sections |
| Curriculum | **Sources:** Shrout & Fleiss 1979; Koo & Li 2016; JCGM 200:2012; [NHTSA pre-crash typology, DOT HS 810 767](https://rosap.ntl.bts.gov/view/dot/6281) (37 scenarios). **Derivation:** Var(mean) = σ²/(nm) · [1 + (m − 1)ρ]. **Exercise (known answer):** m = 3 seeds and ρ = 0.6 give an inflation of 2.2; a simulation gives 2.236 ([curriculum notes P5](https://www.nber.org/papers/t0344)). **Trap:** counting seeds as independent scenes; ranking card scores from different protocols (0.73 / 1.37 / 1.50 on the cards versus 0.35 / 0.72 in the paper) |
| Ownership | ★ Huy: reliability statistics, stratum definitions, taxonomy mapping, validity-record semantics, predictions, diagnosis of every discordance. Agents: AlpaSim deployment, wizard configs, sweep orchestration, manifest writer, telemetry capture, plots |

### S3 — Validity envelope and inter-rater agreement

| Field | Content |
|---|---|
| Depends on | S2 |
| L5 symptom (verified) | NuRec degrades sharply as rollouts move away from the capture path (OmniDreams Fig. 14). The challenge *scores* a lateral exit ≥ 4 m as failure, while NVIDIA's papers *exclude* events beyond 4 m. So a wandering policy is penalized for renderer limits under one convention and silently excused under the other ([OmniDreams](https://arxiv.org/html/2606.03159); [AR1 §6.1](https://arxiv.org/html/2511.00088v2); [commit f012862](https://github.com/NVlabs/alpasim/commit/f012862749366a72f356fce73e22091fe5160e41)). The docs' own version of this is `docs/01` §4 Q2 |
| L4 system | One `renderer` endpoint that can serve NuRec (NRE) or OmniDreams (through FlashDreams). The NRE Harmonizer is off for initial runs and on for PAI final runs. Traffic is either log replay or CAT-K (`trafficsim=catk`, whose seed is unused) ([AlpaSim CHANGELOG](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/CHANGELOG.md)) |
| L3 math ★ | Bland–Altman limits of agreement, rather than correlation alone ([Bland & Altman 1986](https://doi.org/10.1016/S0140-6736(86)90837-8)). Cohen's κ per deviation bin. Kendall τ and pairwise agreement with CIs from a road-clustered bootstrap. Campbell & Fiske: a renderer swap on the same reconstruction measures *reliability*, not validity ([Campbell & Fiske 1959](https://doi.org/10.1037/h0046016)) |
| L2 numerics ★ | Why extrapolated views fail: the 3DGS EWA projection Σ′ = J W Σ Wᵀ Jᵀ and alpha compositing. On EUVS, translation-only extrapolation cuts PSNR 20.67 → 15.59 (−24.6%) and translation plus rotation 20.40 → 14.15 (−30.6%), averaged over 9 methods. NuRec was not among them ([EUVS, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Han_Extrapolated_Urban_View_Synthesis_Benchmark_ICCV_2025_paper.pdf)) |
| L1 silicon | NuRec renders 100 FPS at 1080p on an RTX 6000 Pro, but NVIDIA's NuRec page does not say whether that counts aggregate or per-camera frames, so the renderer manages 25 or 150 ticks/s at 4 cameras ([NuRec docs](https://docs.nvidia.com/nurec/best-practices/scaling-neural-reconstruction.html)). OmniDreams with 4 views runs at about 12 FPS per camera on one GB300 ([OmniDreams Table 3](https://arxiv.org/html/2606.03159v3)) |
| Build artifact | A deviation-binned validity report; renderer-agreement and traffic-agreement tables; out-of-envelope results reported separately, never silently scored or dropped. Upstream PR: **seed the CAT-K traffic service**, a contribution an NVIDIA maintainer explicitly invited ([issue #128](https://github.com/NVlabs/alpasim/issues/128); [PR #149](https://github.com/NVlabs/alpasim/pull/149) left the trafficsim seed unused) |
| Acceptance test | κ and τ with CIs per deviation bin; the validity-exit rate per policy; agreement reported both with and without rollouts ≥ 4 m |
| Predictions (Huy, blank) | τ between Harmonizer on and off: ____ · κ in the ≥ 4 m bin versus the < 1 m bin: ____ · share of rollouts whose failure label changes between replay and CAT-K: ____ |
| Kill criterion | **KC-3, D+5:** if the out-of-envelope rate leaves coverage below Huy's pre-registered floor, tighten the deviation cap and report the coverage loss honestly (`docs/02` §6 fallback, preserved). If renderer agreement fails, report the renderer as a dominant noise source and stop treating NuRec-only scores as policy evidence (inference) |
| Cost (inference) | 6 sweeps ⇒ **$165–1,060**. The OmniDreams slice is UNVERIFIED: it is GB300-class, no verified price exists, so it stays optional |
| Public contribution and why upstream values it | A fidelity-calibrated cap, separate reporting of out-of-envelope rollouts, renderer agreement, and on-rails-vs-CAT-K agreement are all NOT FOUND upstream; the only existing piece is a fixed 4 m corridor scored as failure ([tooling gap table G2](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/eval/src/eval/scorers/ground_truth.py)). Regulators define a virtual tool's usable *envelope* through credibility assessment ([NATM Annex III](https://web.archive.org/web/20240814155120/https://unece.org/sites/default/files/2022-05/WP.29-187-08e.pdf); [EU 2022/1426 Part 4 §1.4](http://publications.europa.eu/resource/celex/32022R1426)). PhysLoop can claim only an internally evidenced envelope, never regulatory credibility |
| In-demand skill evidenced | World models, generative models and neural reconstruction appear in 21% of NVIDIA and 13% of competitor qualifications. Postings: JR2010863; JR2020021 (verification and validation of world-model stacks); JR2019461; [Waymo Staff MLE, Simulation Realism](https://careers.withwaymo.com/jobs?gh_jid=8027348); [Nuro Senior/Staff MLE, Sensor Simulation](https://nuro.ai/careersitem?gh_jid=8049931) |
| Curriculum | **Sources:** [NeRF Eq. 3](https://arxiv.org/html/2003.08934); [3DGS Eq. 5](https://arxiv.org/html/2308.04079); EUVS ([arXiv 2412.05256](https://arxiv.org/abs/2412.05256)); Bland & Altman 1986; Campbell & Fiske 1959. **Derivation:** alpha compositing from dT/dt = −σT, and the EWA projection, under which an on-axis splat of std σ projects to fσ/z pixels. **Exercise (known answer):** σ = [1, 2], δ = [0.5, 0.5] ⇒ α = [0.3935, 0.6321], T = [1, 0.6065], weights [0.3935, 0.3834], total opacity 0.7769. **Trap:** reading agreement on the same reconstruction as validity. Also, when building any stratum dataset, a frame-level random split with k = 20 frames per location and f = 0.2 puts **98.85%** of locations in both splits ([curriculum notes P4](https://arxiv.org/html/2003.08934)) |
| Ownership | ★ Huy: agreement statistics, envelope semantics, the cap decision, predictions. Agents: renderer and traffic configs, the CAT-K seeding PR plumbing and tests, sweeps, plots |

### S4 — Known-group dose–response ladders, precision verdicts and the discriminant check

| Field | Content |
|---|---|
| Depends on | S2 (the gate) and S3 (validity records) |
| L5 symptom (verified) | FlashDrive ran a quantized, optimized Alpamayo 1.5 on 100 AlpaSim clips. Collision improved (0.19 → 0.15) and off-road improved (0.41 → 0.32), but **wrong-lane regressed (0.45 → 0.51)** ([FlashDrive Table 3](https://arxiv.org/html/2608.12932)). NVIDIA's own PTQ recipe evaluates open-loop minADE only and publishes no accuracy or latency ([quant README](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_5_quant/README.md)). One aggregate would have called this a win |
| L4 system | ModelOpt 0.43 recipe formats: `fp8` (model-wide except `patch_embed`); `nvfp4` (which also skips `action_in_proj` trunk.0); `w4a8_nvfp4_fp8`; AutoQuant (default budget 4.8 bits; the 6.5-bit example is about 9 GB; sensitivity from a teacher-forced flow-matching MSE) ([utils.py](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_5_quant/utils.py)). Degradation ladders: camera dropout 4 → 2 → 1 (the NVIDIA precedent), injected latency, action noise, quantization bit-width, early checkpoints |
| L3 math ★ | Pairwise accuracy over K pairs whose order is known, with a Wilson CI ([Kocmi et al. 2021](https://aclanthology.org/2021.wmt-1.57/)). Tie-aware accuracy ([Deutsch et al. 2023](https://doi.org/10.18653/v1/2023.emnlp-main.798)). A trend test per ladder. A MAD outlier screen at \|z\| > 2.5 ([Mathur et al. 2020](https://aclanthology.org/2020.acl-main.448/)). **The ladder assumption must be pre-registered:** a degradation cannot improve true driving, yet camera dropout might occasionally reduce distraction. **Ladders are reported per metric**, because quantization moved FlashDrive's metrics in opposite directions |
| L2 numerics ★ | E4M3's maximum is 448. NVFP4 costs 4.5 bits per element (16-element blocks, an E4M3 scale and an FP32 per-tensor scale). The roofline crossover is B* = b·F / (2·BW). After CUDA graphs, the expert runs at about 20% of its bandwidth bound (113.9 ms measured vs ≈ 22.3 ms bound, 8 steps), and the cause is undiagnosed publicly ([deployment notes Q3](https://arxiv.org/html/2608.12932)) |
| L1 silicon | Ridges: about 279 for RTX PRO 6000 Workstation BF16; 947 for Jetson AGX Thor. DRIVE AGX Thor has 64 GB, with GPU memory limited by carveout. TensorRT 11.3 lists FP8 and FP4 on Thor (SM 11.0), but users report that ModelOpt checkpoints run as fake quantization on Thor ([TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html); [forum 382409](https://forums.developer.nvidia.com/t/how-to-perform-alpamayo1-5-fp8-quantization-for-nvidia-thor/382409)) |
| Build artifact | A known-group report (pairwise accuracy, trends, per-metric ladders). Per-format paired verdicts with intervals (BF16, FP8, NVFP4, W4A8, AutoQuant), plus the published per-layer sensitivity (AutoQuant's logged `search_state`). The discriminant check: the gate against open-loop minADE, which should correlate only weakly. A release note for one checkpoint written in the corrected allowed-claims language (mastery Stage 6 folds in here). **S4b, optional and B-shaped:** an expert-kernel diagnosis explaining why the flow expert runs at about 20% of its bandwidth bound |
| Acceptance test | K known pairs with k/K ordered correctly and a Wilson CI; per-metric direction reported, never a single aggregate; per-format verdicts with intervals; the discriminant correlation with a CI |
| Predictions (Huy, blank) | Pairwise accuracy: ____ · metrics that move monotonically on each ladder: ____ · FP8 vs BF16 paired verdict: ____ · sensitivity of the expert's layers under NVFP4: ____. **Preserved:** `docs/04` E7 (precision), values as written. Premise note: FlashDrive measured W4A8 at 1.16× end-to-end once other optimizations were applied (prefill 1.25×, decode 1.27×) ([deployment notes Q3](https://arxiv.org/html/2608.12932)) |
| Kill criterion | **KC-4, D+7:** if the 95% upper bound on known-group pairwise accuracy falls below the value Huy pre-registers, downgrade the gate-validity claims. The evaluation notes use 0.75 only as an example; Huy sets the actual floor |
| Cost (inference) | At least 3 ladders × 4 levels = 12 or more sweeps ⇒ **$331–2,120**. Quantized and camera-dropout variants process fewer bytes and tokens, so the ceiling is conservative. S4b needs a few RTX PRO 6000 hours |
| Public contribution and why upstream values it | Closed-loop per-format verdicts with intervals, and published accuracy and sensitivity numbers, are NOT FOUND upstream. "Including the action expert" is NOT new ([tooling gap table G6](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_5_quant/quantize.py)). The ladders yield dozens of pairs whose order is known, without on-road data (evaluation notes, G3 protocol step 3) |
| In-demand skill evidenced | Distillation, quantization and TensorRT: 7% of NVIDIA and 16% of competitor qualifications. CUDA: 40% of NVIDIA qualifications. Postings: JR2025038 (quantization, compilation, profiling); JR2016768 (TensorRT inference toolchains); JR2016777 (latency and throughput); [Zoox Senior AI Inference Engineer](https://jobs.lever.co/zoox/c88c8b02-71b6-492c-a666-584458ac8c6e) (PTQ and QAT; INT8 / FP8 / FP4); [XPeng Staff MLE, LLM Quantization & Deployment](https://job-boards.greenhouse.io/xpengmotors/jobs/8710020002) |
| Curriculum | **Sources:** [FP8 formats](https://arxiv.org/html/2209.05433); [MX formats](https://arxiv.org/html/2310.10537); [NVIDIA NVFP4 blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/); [SmoothQuant](https://arxiv.org/html/2211.10438); [Roofline](https://doi.org/10.1145/1498765.1498785); [COSMIN](https://doi.org/10.1016/j.jclinepi.2010.02.006); [PMPP 5th ed. (2026)](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-443-43900-1). **Derivation:** B* = b·F/(2·BW), and NVFP4's effective bits (4 + 8/16). **Exercise (known answer):** NVFP4 saves 16/4.5 = 3.556× vs FP16 and 1.778× vs FP8 (NVIDIA states about 3.5× and 1.8×); E4M3's maximum is (1 + 6/8) · 2⁸ = 448. **Trap:** "4-bit is 4× smaller"; "E4M3 tops out at 240"; assuming every metric moves the same way under quantization |
| Ownership | ★ Huy: ladder design and its assumption, pairwise statistics, format verdicts, the S4b kernel diagnosis. Agents: ModelOpt runs, checkpoint export, sweeps, plots, the release-note template |

### S5 — Cross-paradigm convergent validity and the pre-registered verdict (G3 headline)

| Field | Content |
|---|---|
| Depends on | S2–S4: reliability, agreement and known-group evidence must exist first |
| L5 symptom (verified) | The challenge's own 95% rank intervals leave PAI teams #2–#5 statistically tied, with rank intervals 3–8, 3–9, 3–10 and 3–10. Rank also departs from PCS order: a PCS of 1386 ranks above 1424 ([evaluation notes Q1](https://nvidia-alpasime2eclosedloopchallenge2026.hf.space/gradio/)). At this scale a single aggregate rank is noise. A release gate needs evidence that its orderings survive a different paradigm |
| L4 system | The AlpaSim nuPlan/MTGS track, whose shipped NAVSIM-family adapters cover LTF, DiffusionDrive and GTRS. The NAVSIM navtest/navhard EPDMS. `py123d_garage` (Apache-2.0), which runs one policy in CARLA 2.0, NAVSIM v1 and AlpaSim ([py123d_garage](https://github.com/kesai-labs/py123d_garage); [challenge README](https://github.com/NVlabs/alpasim/blob/e2e_challenge/e2e_challenge/README.md)) |
| L3 math ★ | Spearman and Kendall with Fisher-z CIs, using the FHP variance 1.06/(n−3) or the Bonett–Wright variance (1 + ρ²/2)/(n−3). A lower bound ≥ 0.6 at an observed 0.8 needs **n ≈ 28–34**. **The three-way rule:** PASS if the one-sided 95% lower bound ≥ 0.6; KILL if the upper bound < 0.6; otherwise INSUFFICIENT. PASS thresholds are 0.87–0.89 at n = 10, 0.80–0.82 at n = 20 and 0.77–0.78 at n = 30; KILL thresholds are 0.05–0.07, 0.28 and 0.35 ([evaluation notes Q6](https://doi.org/10.1007/BF02294183)). Also pairwise accuracy, MMRV ([SIMPLER](https://arxiv.org/abs/2405.05941)), and within-family versus between-family correlation |
| L2 numerics | Shared-method variance inflates agreement: PDM-style metrics descend from the same family, and KE:SAI authors appear on both sides. Degraded variants of one parent widen the score range, which mechanically inflates ρ. Report n_families and n_checkpoints separately ([pseudo-sim](https://arxiv.org/html/2506.04218v3)) |
| L1 silicon | Per-checkpoint compute for the NAVSIM-family planners is UNMEASURED. Pilot one checkpoint before committing |
| Build artifact | A G3 preprint: *the first open, multi-policy reliability and convergent-validity study of AlpaSim closed-loop scores, with stated limits*. Plus the pre-registration file, data manifests, and Bean et al.'s construct-validity checklist ([arXiv 2511.04703](https://arxiv.org/abs/2511.04703)) |
| Acceptance test | The pre-registered rule is applied as written; n_families and n_checkpoints are stated; results are shown with and without outlier policies; the R5 private-set check runs only if registration was ever possible (≤ 3 policies, order preserved or not, no coefficient) |
| Predictions (Huy, blank) | Cross-paradigm ρ: ____ · pairwise accuracy: ____ · gap between within-family and between-family correlation: ____. **Preserved:** `docs/01` §4 Q3 ("what ρ between the gate and Drive-IRT on 10 policies?") stays as written. It is re-labelled as reference R0, an aggregation-robustness check rather than a validity check, because both statistics come from the same rollouts ([evaluation notes row 2.7](https://github.com/NVlabs/alpasim/blob/e2e_challenge/e2e_challenge/local_evaluation/README.md)) |
| Kill criterion | **KC-5, D+11:** apply the three-way rule. KILL ⇒ publish the negative result with its mechanism and stop building on the gate's validity. INSUFFICIENT ⇒ report the interval and make no validity claim |
| Cost (inference) | Unmeasured per checkpoint. The reference bracket is the docs' own policy-zoo line (~300 H100-hours) re-priced: **$507** (vast.ai), $1,047 (RunPod secure), $1,287 (Lambda 1×), $1,350 (Nebius) ([deployment notes Q5](https://lambda.ai/pricing)). Whether 300 H100-hours covers 28–34 checkpoints is unknown |
| Public contribution and why upstream values it | Sim2Val covers a single policy with per-episode kinematics, and OmniDreams covers n = 4 with no coefficient. No published multi-policy study correlates AlpaSim/NuRec scores or rankings with an independent reference, and Drive-IRT has no external validation ([Sim2Val](https://arxiv.org/abs/2506.20553); [Drive-IRT docs ch. 1](https://ln2697.github.io/navhard-leaderboard-2/docs/preliminary/01_motivation.html)) |
| In-demand skill evidenced | Publications: 33% of NVIDIA and 24% of competitor qualifications. JR2005596 accepts publications *or* research artifacts such as software projects. JR2019461 sets a scientific roadmap for evaluation and validity standards. JR2010863 prefers closed-loop evaluation of end-to-end models. [Waymo TLM, Prediction ML Evaluation](https://careers.withwaymo.com/jobs?gh_jid=7963516) |
| Curriculum | **Sources:** [Cronbach & Meehl 1955](https://doi.org/10.1037/h0040957); [Messick 1995](https://doi.org/10.1037/0003-066X.50.9.741); [Kane 2013](https://doi.org/10.1111/jedm.12000); [Rutjes et al. 2007](https://doi.org/10.3310/hta11500); FHP 1957; Bonett & Wright 2000; [Kadian et al.](https://arxiv.org/abs/1912.06321); [PolaRiS](https://arxiv.org/abs/2512.16881); [BenchBench](https://arxiv.org/abs/2407.13696). **Derivation:** the Fisher-z CI for Spearman's ρ and the n it implies. **Exercise (known answer):** gate order [1, 2, 3, 4, 5] against reference order [2, 1, 3, 5, 4] ⇒ Σd² = 4, ρ = 0.8; exact P(ρ ≥ 0.8 \| H0, n = 5) = 8/120 = 0.067; at n = 10 the 95% CI for an observed 0.8 is about [0.24, 0.96] (Bonett–Wright). **Trap:** adding degraded variants of one parent as extra n. Also tuning the simulator, the cap or the policies on the validation set: Kadian's tuned SRCC was in-sample, and PolaRiS co-finetuned the policies it evaluated |
| Ownership | ★ Huy: the pre-registration, the statistics, the verdict, the paper's argument. Agents: adapters, NAVSIM and `py123d_garage` runs, sweeps, plots, LaTeX |

### S6 — Guardrail-in-the-loop in C++

| Field | Content |
|---|---|
| Depends on | S2 and S3 |
| L5 symptom (verified architecture) | DRIVE AV runs its end-to-end stack alongside a parallel classical safety stack built on Halos, both in the Level 2 CLA (2026-01-05) and in the L4-ready S-Class (2026-01-29). Scoring the bare model therefore scores something that never ships ([CLA blog](https://blogs.nvidia.com/blog/drive-av-software-mercedes-benz-cla/); [S-Class blog](https://blogs.nvidia.com/blog/mercedes-benz-l4-s-class-drive-av-platform/)). The masking concern, that a guardrail can hide a model regression, is the docs' own (`docs/01` §4 Q4) |
| L4 system | The AlpaSim hook: a driver response may carry `safety_monitor_safe`, and the `SafetyScorer` turns it into `safety_monitor_triggered`. Only consumers and tests exist upstream; there is no monitor implementation. The override path runs into the controller ([safety.py](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/eval/src/eval/scorers/safety.py)) |
| L3 math ★ | Time-to-collision and the RSS safe distance from Lemma 2 ([Shalev-Shwartz et al., arXiv 1708.06374](https://arxiv.org/html/1708.06374)). The intervention rate as an early regression signal. A paired comparison of the policy alone against the policy plus monitor |
| L2 numerics | C++ with fixed memory and deterministic timing. This is the DriveOS idiom: C++ runtimes, and per the DriveOS 6 documentation the safety runtime runs only on QNX with kSAFETY engines ([DriveOS 6.0.10 TensorRT notes](https://developer.nvidia.com/docs/drive/drive-os/6.0.10/public/drive-os-tensorrt/release-notes/index.html)) |
| L1 silicon | Not applicable; the monitor runs on CPU |
| Build artifact | A rule-based TTC and drivable-area monitor in C++, labelled as **PhysLoop's own design, not "Halos"**, since no public NVIDIA source specifies the Halos runtime-monitor logic ([Halos page](https://www.nvidia.com/en-us/ai-trust-center/halos/autonomous-vehicles/)). Paired verdicts for policy versus policy plus monitor; an intervention-rate report; an upstream issue or PR implementing a monitor behind the existing hook |
| Acceptance test | C++ unit tests for the monitor; the paired comparison; the intervention rate with a CI; a list of the failures the monitor catches and the ones it misses |
| Predictions (Huy, blank) | Intervention rate per policy: ____ · share of at-fault collisions the monitor prevents: ____ |
| Kill criterion | **KC-6, D+8:** if the monitor cannot sit in the loop without changing the semantics of the policy under test, report it as an offline monitor only (inference) |
| Cost (inference) | 2 sweeps ⇒ **$55–353** |
| Public contribution and why upstream values it | The hook exists; the monitor, the override path, the policy-versus-policy-plus-monitor scoring and the intervention-rate reporting do not ([tooling gap table G4](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/eval/src/eval/scorers/safety.py)) |
| In-demand skill evidenced | C++ appears in **67%** of NVIDIA and 58% of competitor qualifications, nearly always as a requirement. Safety, V&V and standards appear in 19% of both. Postings: JR2023529 (classical and learned systems in one production architecture); JR2024273 (fallback and degraded modes; C/C++); JR2025038 (C++, QNX); [Kodiak Staff SWE, Learned & Hybrid Behavior Planning](https://job-boards.greenhouse.io/kodiak/jobs/4294898009) (validation, fallback and monitoring criteria for learned planners) |
| Curriculum | **Sources:** RSS Lemma 2; the Halos page; UL 4600 Ed. 3 ([UL](https://www.ul.com/news/ul-4600-edition-3-updates-incorporate-autonomous-trucking)); ISO/PAS 8800:2024 metadata ([ISO via Wayback](https://web.archive.org/web/20260813162702/https://www.iso.org/standard/83303.html)); NHTSA Standing General Order fields, which include no fault field ([SGO dictionary](https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Data_Element_Definitions.pdf)). **Derivation:** the worst-case kinematics behind RSS d_min. **Exercise (known answer):** v_r = v_f = 20 m/s, ρ = 0.5 s, a_max,accel = 3, a_min,brake = 4, a_max,brake = 8 m/s² ⇒ **d_min = 43.16 m**. **Trap:** assuming the lead car brakes no harder than you (a_max,brake = 4) gives 18.16 m; and calling the monitor "Halos" |
| Ownership | ★ Huy: the monitor logic (the verifier), thresholds, intervention-rate statistics, predictions. Agents: the C++ build, gRPC plumbing, tests, sweeps, plots |

### S7 — nanoAlpamayo as the gate-measured student (G5 merged)

| Field | Content |
|---|---|
| Depends on | S2 (closed-loop numbers come only from the gate), S3 (RL claims need G2), and S4 (the choice of deployment format). This preserves the dependency rule in `docs/03` §2 |
| L5 symptom (verified) | In AR1, a reasoning-only reward raised the critic's reasoning grade from 3.1 to 4.5, while consistency fell from 0.62 to 0.53 and ADE worsened to 2.19 m. The proxy improved while behaviour degraded ([AR1 Table 9](https://arxiv.org/html/2511.00088v2)). RoaD shows the covariate shift that appears when open-loop behaviour cloning is deployed in closed loop ([RoaD](https://arxiv.org/abs/2512.01993)) |
| L4 system | A data pipeline on a PAI-AV subset. The license allows internal use only and forbids derivatives, so publish code, clip-UUID manifests and aggregates only. Splits are grouped by location. A small VLM backbone: AR1's 0.5B is DINOv2 plus Qwen2.5-0.5B, and Cosmos3-Edge 4B is only a size-class candidate. 128 discrete trajectory tokens, and a flow expert that reads the VLM KV cache. Teacher labels from Alpamayo 1.5 (the cheap teacher NVIDIA recommends) or Alpamayo 2 Super (one H100 per instance) ([NVIDIA blog](https://blogs.nvidia.com/blog/alpamayo-2-super/); [AR1 §6.5.1](https://arxiv.org/html/2511.00088v2)) |
| L3 math ★ | The AR1 Eq. 5 decoder, which integrates heading exactly and position by the trapezoid rule. BC's T²ε bound versus DAgger's uTε. The CFM target and its marginal field. The Flow-GRPO ODE→SDE conversion. GRPO advantages and the k3 KL estimator; PPO clipping. AR1's GRPO variant (unclipped, softmax-weighted, mean-centred) versus AlpaGym's clipped objective. GKD on-policy distillation. OPTED- and RoaD-style closed-loop supervision |
| L2 numerics | 180 tokens per image (Qwen3-VL: patch 16 × merge 2), 2,880 tokens for 16 images, and 147,456 B of KV cache per token ([curriculum notes P2](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/raw/main/config.json)). At 273 GB/s, the Thor floor is 22 ms per token for a 3B BF16 student and 11 ms in FP8. A 0.1 s budget therefore allows only a few reasoning tokens (inference from [deployment notes Q3](https://developer.nvidia.com/drive/agx)) |
| L1 silicon | One 8×H100 node for SFT |
| Build artifact | A reproducible training repository (code plus manifests). v0: imitation. v1: CoC SFT on auto-labels, GRPO on tokens, and Flow-SDE through the expert. A paired student-versus-teacher verdict on the gate, with a failure-mode diff |
| Acceptance test | Re-based in §5: minADE6 on a *published* public subset; an RL row measured on a public metric with an open judge and paired seeds; a closed-loop paired non-inferiority verdict against a released checkpoint under one pinned protocol |
| Predictions (Huy, blank) | minADE6 on the defined subset: ____ · relative gain from RL: ____ · paired student–teacher margin: ____. **Preserved:** `docs/04` E5 (Flow-SDE close-encounter change within 200 GPU-hours), values as written. Premise note: AlpaGym trains only the expert (VLM frozen), samples 5 SDE steps, and clips a trajectory-level ratio; RoaD and OPTED both report that direct RL in AlpaSim is sample-hungry ([e2e notes B19–B24](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/default.yaml)) |
| Kill criterion | **KC-7.** (a) If NVIDIA releases its 10B→~2B distillation script, pivot S7 to evaluating NVIDIA's student on the gate. (b) If the student misses Huy's paired non-inferiority margin after three ablation rounds, shrink the subset and fix the pipeline. The margin can be no tighter than the gate can certify: about 2.7 pp at ψ = 4% on 441 scenes. (c) If the minADE target is missed after three rounds, apply the same response (the `docs/03` §5 analogue) |
| Cost (inference) | An 8×H100 node costs **$13.4/h** (a single, thin vast.ai offer), $31.92/h (Lambda) or $36.00/h (Nebius), so a 24-hour run costs $322–864. Alpamayo 2 Super teacher labeling costs $1.69–4.50 per H100-hour (spot from $0.60–0.79). Run-hours are UNMEASURED. The docs' G5 line (1,600 GPU-hours) re-prices to **$960–7,200** ([deployment notes Q5](https://cloud.vast.ai/api/v0/bundles/)) |
| Public contribution and why upstream values it | No public distillation recipe and no NVIDIA student checkpoint exist (NOT FOUND, despite three promises). A gate-measured open student would fill that gap. But NVIDIA has announced the same thing, so the durable value is the **measured comparison**, not the student itself ([tooling gap table G5](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/docs/ONBOARDING.md)) |
| In-demand skill evidenced | VLA/VLM/LLM experience appears in **49%** of NVIDIA and 27% of competitor qualifications, the largest AI-specific cluster. RL appears in 23% and 16%; imitation and end-to-end driving in 23% and 7%. Postings: JR2019758 (building LLMs, VLMs or VLAs from scratch; C++ deployment); JR2022179 (flow-matching and diffusion policies; training on the policy's own rollouts; data flywheels); JR2005597 (RL fine-tuning of multimodal LLMs); [Waymo Staff MLE, Foundation Models](https://careers.withwaymo.com/jobs?gh_jid=8152556); [XPeng Senior Staff MLE](https://job-boards.greenhouse.io/xpengmotors/jobs/8465375002) |
| Curriculum | **Sources:** AR1 §3.2.2; [Ross & Bagnell 2010](https://proceedings.mlr.press/v9/ross10a.html); [DAgger](https://arxiv.org/html/1011.0686); [Flow Matching](https://arxiv.org/html/2210.02747) and [its guide](https://arxiv.org/abs/2412.06264); [Flow-GRPO](https://arxiv.org/html/2505.05470); [π_RL](https://arxiv.org/html/2510.25889); [PPO](https://arxiv.org/pdf/1707.06347); [DeepSeekMath](https://arxiv.org/html/2402.03300); [Schulman's KL estimators](http://joschu.net/blog/kl-approx.html); [GKD](https://arxiv.org/html/2306.13649); [OPTED](https://arxiv.org/abs/2609.20756). **Derivation:** the exact heading update Δθ = κ(vΔT + aΔT²/2); the CFM marginal β(t); the ODE→SDE drift. **Exercise (known answer):** a constant arc with v = 10 m/s, κ = 0.02 m⁻¹, T = 6.4 s. Forward Euler gives an endpoint error of **0.5972 m**; AR1 Eq. 5 gives **0.00199 m**. (Alternate: a 1-D Gaussian flow with 10 Euler steps gives std 0.43078 and a relative MSE of 1.92%.) **Trap:** a forward-Euler decoder adds a 0.60 m systematic error, the same order as the 0.913 m target. Also: "straight conditional paths, so a few steps suffice"; an inverted k3 ratio; the 14-pixel rule (231 tokens per image); computing KV with all 32 heads (1.70 GB) |
| Ownership | ★ Huy: RL loss math, reward and verifier logic, decoder math, the data split and leakage audit, predictions, diagnoses. Agents: data loaders, training and eval harnesses, launch scripts, plots |

### S8 — Throughput measurement report and flywheel specification

| Field | Content |
|---|---|
| Depends on | Telemetry from S2–S4 |
| L5 symptom (verified) | The docs' own cost model was off by up to about 2×. Rollout throughput is about 4 ticks/s per GPU, not 6–9, and the expert costs about 7 ms per sample at B = 32, not 0.1 ms. AlpaGym's default is an 8-step smoke preset. Budgets built on unmeasured throughput fail ([deployment notes Q3](https://arxiv.org/html/2608.12932); [AlpaGym default.yaml](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/default.yaml)) |
| L4 system | AlpaSim Prometheus telemetry (on by default since 2026-06-30) and `metrics_plot.png`, which shows RPC durations, simulated seconds per rollout and runtime idle fraction. AlpaGym's opt-in `perf/` module parses AlpaSim's telemetry files ([TELEMETRY.md](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/docs/TELEMETRY.md); [perf README](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/perf/README.md)) |
| L3 math ★ | The balance equation N_gen · g ≈ N_train · t (`docs/04`); the batching crossover; the fact that KV-cache reads grow with batch size |
| L2 numerics | Render resolution: 1080p carries 11.25× the pixels of 320×576, and Alpamayo drivers reject images smaller than 320×576 ([AlpaSim CHANGELOG](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/CHANGELOG.md)) |
| L1 silicon | NuRec's 100 FPS may mean 25 or 150 ticks/s, depending on whether it counts aggregate or per-camera frames |
| Build artifact | A stage-level AlpaSim profile (renderer, policy, controller, synchronization) at stated presets. The render-resolution experiment, run as an AlpaSim camera-config change rather than a PR. A batch-scaling curve. Optionally, an AlpaGym trainer profile at a stated preset. The G8 flywheel specification (document only) |
| Acceptance test | Every number carries its preset, SHA and GPU edition; the preserved prediction (iii) is reconciled |
| Predictions (Huy, blank) | Preserved from `docs/03` Stage 1b: (iii) scenarios per second per RTX PRO 6000 at 1080p versus 576×320: ____. **Preserved:** `docs/04` E1, E2 and E4, values as written. Premise notes: E1's 200-tick backprop argument does not match AlpaGym's 8-policy-step default, and E4 is an AlpaSim camera-config change, not a pull request ([tooling notes §1–§2](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/wizard/configs/cameras/2cam_2168.yaml)) |
| Kill criterion | None beyond the budget line. If the telemetry is insufficient, skip the report |
| Cost (inference) | About $0 marginal, since the telemetry comes from S2–S4. An optional AlpaGym smoke profile on two GPUs with at least 40 GB each costs tens of dollars; Classmethod reported 10 steps in about 32 minutes on 2 × L40S ([Classmethod](https://dev.classmethod.jp/articles/nvidia-alpagym-on-aws/)) |
| Public contribution and why upstream values it | The instrumentation is already upstream, but no stage-level profile has been published. AlpaSim accepts external PRs; AlpaGym does not. NVIDIA names closed-loop evaluation throughput as a product metric ([JR2026365](https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Director--AV-Simulation-Product-Management_JR2026365)) |
| In-demand skill evidenced | Role A JR2010863, duties ① and ④ (RL training frameworks; monitoring and debugging GPU training). JR2005597 (simulation infrastructure at scale). JR2016777 (latency and throughput). Kubernetes, SLURM and Ray appear in 9% of NVIDIA and 12% of competitor qualifications, partly coverable through AlpaSim's `SLURM_ENROOT` run method |
| Curriculum | **Sources:** AlpaSim TELEMETRY.md and OPERATIONS.md; the NuRec scaling page; [PhyAI](https://arxiv.org/html/2608.03682v3); VLA-Perf, which gives *modeled* bounds. **Derivation:** the batching crossover with KV reads included. **Exercise (known answer):** Alpamayo 1.5 BF16 reads 15.17 GB per token, so 15.17 GB ÷ 273 GB/s = **55.6 ms per token**; a 3B FP8 student reads 3 GB, giving **11.0 ms** ([deployment notes Q3](https://developer.nvidia.com/drive/agx)). **Trap:** reading NuRec's 100 FPS as per-camera; profiling AlpaGym's 8-step smoke preset as if it were production; quoting VLA-Perf's modeled Thor numbers as measurements |
| Ownership | ★ Huy: the cost model and the bottleneck diagnosis. Agents: telemetry extraction, profiling runs, plots |

---

## 4. G1–G8 re-scoped: keep the gate, restate G3, merge G5 into the student

Every deliverable was checked against what upstream already ships as of 2026-10-03. The verdicts follow the tooling notes' gap table ([AlpaSim](https://github.com/NVlabs/alpasim/tree/affc2eab209fa43bdfa2f26c0f8d437922d78a68); [alpamayo-recipes](https://github.com/NVlabs/alpamayo-recipes/tree/c29dc61b7318169d2439f3e9d88e547242e301de); [AlpaGym](https://github.com/NVlabs/alpagym/tree/972d160eed0e23d388497851504a3a233fec5879)).

| G | Upstream already ships | Still open | Verdict | Reason | Stage |
|---|---|---|---|---|---|
| G1 Gate v1 | Per-rollout scorers; `results-summary.json` (2026-06-05); seeds (PR #149, 2026-08-17); the curated 1,761/441 split (PR #167, 2026-08-31); Drive-IRT ranking with ≥ 6 subjects | A paired two-checkpoint verdict; stratification; seed test–retest; per-stratum bounds; an extended taxonomy; an allowed-claims list | **KEEP, as a thin AlpaSim layer** | The highest-value gap, and the inputs already exist. Margins must be at least the certifiable ≈ 2.7 pp at ψ = 4% | S1, S2 |
| G2 Validity envelope | `dist_to_gt_*` deviation metrics; the 4 m corridor scored as failure (#163, 2026-08-24); one renderer endpoint serving NuRec or OmniDreams; OmniDreams Fig. 14 (an FVD-vs-deviation precedent) | A calibrated cap; separate reporting of out-of-envelope rollouts; renderer agreement; replay-vs-CAT-K agreement; an envelope-level invalid rate | **KEEP, narrowed**; the agreement parts merge into G3's reliability legs | Partly done upstream. The open part is calibration and separate reporting | S2, S3 |
| G3 Gate validity | Sim2Val (one policy, per-episode); OmniDreams (n = 4, qualitative) | A multi-policy reliability and convergent-validity study | **KEEP, RESTATED as construct validity** (below) | As written it cannot be falsified, and "predictive validity" cannot be attained | S2–S5 |
| G4 Guardrail | The `safety_monitor_safe` hook; MPC controllers; the public dual-stack architecture | The monitor; the override path; paired scoring; the intervention rate | **KEEP** | The hook exists and the monitor does not. Label the mechanism as PhysLoop's design, not "Halos" | S6 |
| G5 Distillation | Teacher positioning; the Alpamayo 2 Super SFT recipe (2026-09-30); AlpaGym expert RL | Any public distillation procedure; a student-vs-teacher paired verdict | **MERGE into nanoAlpamayo** (S7), with a pivot-to-evaluation kill | NVIDIA has announced a 10B→~2B script three times | S7 |
| G6 Precision | ModelOpt 0.43 PTQ including expert layers (2026-06-29); open-loop minADE eval; Edge-LLM supports Alpamayo-R1-10B only | Closed-loop per-format verdicts with intervals; published accuracy, latency and sensitivity | **KEEP, RE-SCOPED and merged** into S4 as the quantization rung of the known-group ladder | The expert-quantization novelty is gone; the closed-loop evidence is not | S4 |
| G7 Throughput | AlpaGym `perf/`; AlpaSim Prometheus (default since 2026-06-30); Grafana panels; batched inference; coarse community wall-clock | A stage-level public profile at a realistic preset; a batch-scaling curve; the render-resolution experiment | **RE-SCOPE to a measurement report**; drop "≥ 1 PR merged in AlpaGym" | AlpaGym has zero external PRs and a frozen mirror. Render resolution is an AlpaSim camera config change. PRs go to AlpaSim | S8 |
| G8 Flywheel spec | The CoC autolabeler repository (since 2026-05-24); Instant NuRec | A diagram and config only | **KEEP as a document**, low priority | Cheap; data redistribution is limited by license §4.6 | S8 |

### G3 restated honestly: a construct-validity protocol with a pre-registered verdict

Without an accepted criterion, measurement science prescribes an argument built from independent kinds of evidence, not predictive validity ([Cronbach & Meehl 1955](https://doi.org/10.1037/h0040957); [Messick 1995](https://doi.org/10.1037/0003-066X.50.9.741); [Kane 2013](https://doi.org/10.1111/jedm.12000)). The references are ranked by independence, from the same rollouts re-aggregated (R0, Drive-IRT) to on-road outcomes (R10, attainable only inside NVIDIA, Wayve or Waabi) ([evaluation notes, G3 §2](https://arxiv.org/abs/2506.20553)). The protocol below follows the evaluation notes. Every prediction in it is Huy's, written into `experiments/` before the run.

1. **Reliability (R1).** At least 10 policies × 441 `nurec_curated_val` scenes × at least 3 seeds. Under VIM conditions, seeds measure repeatability; GPU, driver and AlpaSim-version drift measure intermediate precision; another site or renderer measures reproducibility ([JCGM 200:2012](https://www.bipm.org/documents/20126/2071204/JCGM_200_2012.pdf)). Report scenario-level seed discordance, policy-level ICC with a CI, and Kendall's W, flagged when ≲ 0.8. Negative controls (no-op changes) estimate the false-block rate. **If reliability is poor, stop: validity cannot exceed reliability.** The S4 ladder checkpoints can serve as the ≥ 10-policy reliability panel (inference). They cannot add n to the convergent ρ (evaluation notes row 4.17).
2. **Renderer and traffic agreement (R2 Harmonizer on/off; R4 replay vs CAT-K; R3 OmniDreams if accessible).** Report Kendall τ and pairwise agreement with CIs from a road-clustered bootstrap, plus Cohen's κ on failure labels by deviation bin (< 1, 1–2, 2–4, ≥ 4 m). Under Campbell & Fiske, this is reliability evidence: agreement between maximally similar methods ([Campbell & Fiske 1959](https://doi.org/10.1037/h0046016)).
3. **Dose–response degradation ladders (R7, R9).** At least 3 degradation types × at least 4 levels, with monotone decline pre-registered per metric. Report pairwise accuracy over the K known-ordered pairs with a Wilson CI, a trend test per ladder, tie-aware accuracy, accuracy stratified by expected difference, and a MAD outlier screen ([Kocmi 2021](https://aclanthology.org/2021.wmt-1.57/); [Deutsch 2023](https://doi.org/10.18653/v1/2023.emnlp-main.798); [Mathur 2020](https://aclanthology.org/2020.acl-main.448/)). This produces dozens of ground-truth-ordered pairs without on-road data.
4. **Cross-paradigm agreement (R6, R9b) with at least 28–34 checkpoints.** NAVSIM-family policies on the AlpaSim nuPlan/MTGS track, compared against NAVSIM EPDMS, with CARLA via `py123d_garage` as a third paradigm. Report Spearman and Kendall with Bonett–Wright CIs, pairwise accuracy and MMRV, and analyse within-family and between-family correlation separately ([SIMPLER](https://arxiv.org/abs/2405.05941); [py123d_garage](https://github.com/kesai-labs/py123d_garage)).
   - **4b. Hold-out discipline.** Never tune the cap, renderer or traffic model on the validation policies, and never adapt policies to the evaluator. Kadian's tuned SRCC was in-sample, and PolaRiS co-finetuned the policies it evaluated ([Kadian](https://arxiv.org/abs/1912.06321); [PolaRiS](https://arxiv.org/abs/2512.16881)).
5. **Discriminant check (R8).** The gate should correlate only weakly with open-loop minADE6 on PAI-AV validation clips. AR1 used 644 such samples, but that list is unpublished, so Huy defines a public subset. A strong correlation would mean the gate measures log imitation, not closed-loop competence ([nuPlan misalignment](https://arxiv.org/abs/2306.07962)).
6. **At most 3 private-set submissions (R5).** This needs registration before 2026-10-18 and an organization email. For Huy it applies only through a collaborator, or in a later challenge edition if one is ever announced (none is in the notes). At n ≤ 3 it can show whether order is preserved, but it yields no coefficient ([challenge Space](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026)).
7. **Pre-registered PASS / KILL / INSUFFICIENT rule.** Kill or downgrade only on decidable evidence:
   - (a) policy-level reliability below a floor Huy sets;
   - (b) a 95% upper bound on known-group pairwise accuracy below a value Huy sets (the notes use 0.75 only as an example);
   - (c) for cross-paradigm ρ: PASS if the one-sided 95% lower bound is ≥ 0.6, KILL if the upper bound is < 0.6, otherwise INSUFFICIENT. At n = 30, PASS needs an observed ρ ≳ 0.77–0.78 and KILL ≲ 0.35; at n = 10, PASS needs ≳ 0.87–0.89 and KILL ≲ 0.05–0.07 ([evaluation notes Q6 row 6.10](https://doi.org/10.1007/BF02294183)). The 0.6 cut is carried over from the docs; Huy may re-set it before pre-registration.

**The bottom line in measurement-science terms.** Without on-road outcomes, the strongest conclusion this protocol can license is "a reliable, renderer- and traffic-robust instrument that responds correctly to known degradations and converges with independent simulation paradigms." That is construct-validity evidence. It is not predictive validity ([AERA/APA/NCME Standards](https://testingstandards.net/uploads/7/6/6/4/76643089/standards_2014edition.pdf)).

### Allowed claims, corrected (proposed replacement for `docs/02` §5, not enacted)

**May claim**, each with n, CI, seeds, manifest, AlpaSim SHA, scenario list and deviation convention stated:

- "Checkpoint B is not worse than A by more than Δ on every pre-specified stratum, at one-sided 95% confidence, paired and cluster-adjusted by road." Δ is pre-registered, and no tighter than ≈ 2.7 pp at ψ = 4% on 441 scenes. Strata without power are reported INSUFFICIENT, and INSUFFICIENT blocks GO ([Durkalski 2003](https://doi.org/10.1002/sim.1438); [Berger 1982](https://doi.org/10.1080/00401706.1982.10487790)).
- "The per-scenario failure rate in stratum S is ≤ X, by a **one-sided** 95% Clopper–Pearson bound." For 0 failures in 1,000 that bound is ≤ 0.30%; the two-sided bound is 0.37%. n counts roads or logs when scene independence is doubtful ([Clopper & Pearson 1934](https://doi.org/10.1093/biomet/26.4.404)).
- "These failure modes were found." This applies to an adversarial set and is reported as findings.
- The G3 claims the protocol licenses:
  - "gate rankings are reliable across seeds (ICC = …)";
  - "robust to renderer post-processing and traffic model (τ = …)";
  - "responsive to controlled degradations (k/K known pairs ordered correctly)";
  - "converge with NAVSIM pseudo-simulation on n NAVSIM-family policies (ρ = … [CI]; n_families, n_checkpoints)";
  - "diverge from open-loop minADE as expected";
  - "the order of ≤ 3 policies is preserved on the challenge's private set", only if that data is ever obtained.
- "Gate rankings agree with reference R at ρ = … [CI] on n policies", always naming R's independence level (R0–R10).

**Must not claim:**

- Per-mile crash rates, human parity, or "safe" / "safe to deploy". RAND's arithmetic shows 275 million failure-free miles are needed to bound a fatality rate at 1.09 per 100 million miles with 95% confidence ([RAND RR-1478](https://www.rand.org/pubs/research_reports/RR1478.html)).
- Compliance with ISO 26262:2018, ISO 21448:2022, **ISO/PAS 8800:2024**, UL 4600 Ed. 3, UN R157 or UN R171.
- Compliance with **UN R[185]**. Its number is provisional, and it is not in force before about 2027-01-30, and only if no blocking objection arises ([C.N.288.2026](https://treaties.un.org/doc/Publication/CN/2026/CN.288.2026-Eng.pdf)).
- Compliance with **the UN ADS GTR**, adopted on 2026-06-24 ([NHTSA 91 FR 48485](https://www.govinfo.gov/content/pkg/FR-2026-07-31/html/2026-15483.htm)). "NATM compliance" is not a meaningful claim either, because NATM is guidance.
- "Predictive validity", "predicts on-road outcomes" or "validated against the real world", unless a real criterion (paired on-road outcomes) is actually obtained.
- Regulatory credibility of AlpaSim as a virtual toolchain. EU 2022/1426 and NATM Annex III require correlation with physical tests, so PhysLoop may claim only an internally evidenced envelope ([EU 2022/1426](http://publications.europa.eu/resource/celex/32022R1426)).
- Any on-road correlation that was not measured.
- Counting out-of-envelope (≥ 4 m) rollouts as policy failures without reporting them separately.
- Generalizing NAVSIM-family agreement to PAI-track VLA policies without R1–R3 evidence on PAI.
- Calling the PhysLoop monitor "Halos".
- Presenting paired release-to-release regression as "what Waymo does". Waymo's primary comparators are human-derived benchmarks; regression against prior releases is one diagnostic input ([Waymo 2505.09880](https://arxiv.org/abs/2505.09880)).

**Novelty wording fixes.** These replace sentences in `docs/02` §0 and §2 and in `docs/01` §0 and C8:

- **Replace** "first published predictive-validity number for AlpaSim" **with** "first open, multi-policy reliability and convergent-validity study of AlpaSim closed-loop scores, with stated limits".
- **Replace** "No peer-reviewed coefficient exists between neural-reconstruction closed-loop scores and on-road outcomes" **with** "No published multi-policy study correlates neural-reconstruction closed-loop policy scores or rankings with on-road outcomes. The only peer-reviewed coefficient (Sim2Val, ρ > 0.9) is per-episode, single-policy, for kinematic metrics over an 8-s closed-loop horizon" ([Sim2Val](https://arxiv.org/abs/2506.20553)).
- **Rewrite** `docs/02` §0's "how the industry gates releases" so that each source is credited with what it actually requires:
  - **regulators:** scenario-based validation, credibility-assessed simulation, in-service monitoring, and a human-driver baseline;
  - **Waymo:** twelve criteria combined qualitatively, human-derived benchmarks, regression against prior releases as a diagnostic, and phased rollout;
  - **Zoox:** each safety-relevant release gated on one combined quantitative risk estimate.

  Paired regression then becomes PhysLoop's own engineering choice ([Zoox 2026](https://zoox.com/assets/1786550724-zoox-safety-case-framework-2026.pdf)).

### The kill check, corrected

| | `docs/02` §3 as written (suspended by D-008) | Corrected (proposed; dated from D) |
|---|---|---|
| Gate validity | Dec 14: if ρ < 0.6 on ≥ 10 policies, publish the negative result and stop | KC-2 (D+3): reliability floor. KC-4 (D+7): known-group upper bound. KC-5 (D+11): the three-way ρ rule with n ≈ 28–34. Each threshold is Huy's, pre-registered |
| Student | If G5 cannot produce a ≤ 3B student within 10% of the teacher on G1, publish G1–G4 + G6 and stop G5 | KC-7: the student fails a paired non-inferiority margin Δ, chosen by Huy, after three ablation rounds; or NVIDIA ships its distillation script. Either way, pivot to evaluating NVIDIA's student. "Within 10%" becomes a paired margin, because a 10% relative gap on an unknown base rate can be smaller than the ≈ 2.7 pp that 441 scenes can certify (inference from [evaluation notes Q6](https://doi.org/10.2307/2531961)) |
| Date | 2026-12-14, fixed | Relative to D, plus fixed external dates (§9) |

---

## 5. nanoAlpamayo shrinks to a gate-measured student with public targets

The feasibility notes conclude that **the architecture is feasible and the data plan is the binding risk** ([e2e notes KQ7](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)). The current acceptance table mixes two kinds of target:

- **public, reproducible numbers**, from AR1 Table 10;
- **numbers that cannot be reproduced**, from AR1 Table 9, which uses NVIDIA's internal Chain-of-Causation (CoC) data and an internal LRM critic ([AR1 v2](https://arxiv.org/html/2511.00088v2)).

The table below says what AR1 actually reports, for which model, and on which set.

| Metric | AR1 reports | Model | Evaluation set | Public? | Use in the re-scoped plan |
|---|---|---|---|---|---|
| minADE6 at 6.4 s | **0.913 m** (10B: 0.849 m) | `Alpamayo-R1-0.5B`, described only as "a smaller backbone"; weights unreleased | 644 examples from the PAI-AV evaluation set; the list is unpublished, and the repos ship 1,181 undocumented clip IDs | Public data, unpublished list | Set a target only after Huy defines and publishes a subset. Compare with a released checkpoint re-run on the same subset ([alpamayo1.5 notebooks](https://github.com/NVlabs/alpamayo1.5/tree/36aeb4c5938cbc2eb2aed33b22434773da4ab639/notebooks)) |
| Closed loop: close-encounter / off-road / AlpaSim score | **9.0 ± 1.0% / 19.0 ± 0.0% / 0.35 ± 0.01** (10B: 4.0 / 16.0 / 0.72); at-fault; the score is km between events | `Alpamayo-R1-0.5B` | 920 PAI-AV NuRec scenarios through AlpaSim; list and AlpaSim version unpublished; > 4 m events excluded | Cannot be re-run, since the 0.5B weights are not public | Reference only. Re-baseline a released checkpoint under the gate with one pinned protocol |
| Most-likely-mode ADE (the top of 6 samples, **not** minADE6) | SFT **2.12 m → 1.92 m** with RL on r_reason + r_consistency (−9.4%); no seeds or error bars | "A 0.5B AR1 model fine-tuned on CoC data"; never stated to be Table 10's model | NVIDIA's internal CoC dataset (geo-fenced) | No | Replace with a public metric measured over paired seeds |
| Reasoning grade / consistency | 3.1 → 4.5 (LRM critic, 0–5 rubric) / 0.62 → 0.85 (a rule-based binary match). Close-encounter on the cited row falls only from 6.9% to 6.2%; adding r_safety gives 3.7% at ADE 1.94 m; r_reason alone gives ADE 2.19 m and consistency 0.53 | Same | Internal | No | Replace with an open judge on the 349-label OOD validation set |
| Data scaling (0.5B) | 100k segments overfit (1.111 m without early stopping, 1.016 m with); 500k → 0.880 m; 2M → 0.874 m | 0.5B | Internal D_overall | No | Treat as a risk signal: 100–300 h may sit in the overfitting regime (a hypothesis, since the test sets differ) |
| Model-card numbers, for comparison only | Alpamayo 1.5: minADE6 0.916 m (1,434 samples); Alpamayo 2 Super: 0.911 m (1,434); AlpaSim 1.37 ± 0.10 and 1.50 ± 0.13 (913 scenes) | Released 10B and 34B | The new protocol | Cards only | **Not comparable** with 0.913 m (644 samples) ([A2S card](https://huggingface.co/nvidia/Alpamayo2-Super/blob/00554695e729a6ff0b6281fd2c81b18d06e33dbe/README.md)) |

**Feasibility constraints, all verified:**

- **Data scale.** 100–300 h is 0.1–0.4% of AR1's 80,000 h, or 18,000–54,000 twenty-second clips ([AR1 §6.1](https://arxiv.org/html/2511.00088v2)).
- **CoC labels.** Public labels cover **1,450 training clips (1,728 labels) and 290 validation clips (349 labels)**, an out-of-distribution subset only. Training-scale CoC labels must be auto-labeled.
- **License.** The PAI-AV license forbids derivatives and distribution and expires 12 months after first download ([PAI-AV card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)).
- **Teacher cost.** Alpamayo 2 Super, the auto-labeler NVIDIA advertises, needs a full H100 80 GB per instance at batch 1. Alpamayo 1.5 at 22 GB is the cheaper teacher NVIDIA itself recommends ([A2S card](https://huggingface.co/nvidia/Alpamayo2-Super/blob/00554695e729a6ff0b6281fd2c81b18d06e33dbe/README.md); [NVIDIA blog](https://blogs.nvidia.com/blog/alpamayo-2-super/)).
- **Compute.** No verified GPU-hour figure for SFT or GRPO on 0.5–3B VLAs exists in the notes; the delegated survey did not return. The docs' $4–6k for Stages 2–3 is therefore unverified.

**Re-scoped acceptance (a proposal; Huy sets every number):**

- **(A1)** minADE6 on a *published* PAI-AV subset: a clip-UUID manifest with location-grouped splits. Target: ____ (Huy, after the subset is defined). The comparison anchor is a released checkpoint re-evaluated on the same subset.
- **(A2)** The RL gain is measured as public-validation minADE6 or most-likely-mode ADE over at least 3 paired seeds, plus a CoC-consistency score from an open judge on the 349-label set. Target: ____.
- **(A3)** Reasoning moves in the same direction, with a CI, on public stand-ins only. Publishing auto-labels is limited by license §4.6.
- **(A4)** Closed loop: a paired non-inferiority verdict on the gate against a released checkpoint under one pinned protocol (scene list, AlpaSim SHA, seeds, the 4 m convention). This replaces "within 20% of 0.35".
- **(A5)** The decoder reproduces AR1 Eq. 5. The known-answer arc gives 0.00199 m, against 0.5972 m for forward Euler ([curriculum notes P1](https://arxiv.org/html/2511.00088)).

**Order.** The data pipeline comes first, then the overfit-one-batch and loss-at-init checks preserved from `docs/03`. Next come SFT, CoC SFT on auto-labels, and closed-loop RoaD/OPTED-style correction. GRPO on tokens follows, and Flow-SDE through the expert runs last as an ablation, with the expert-drift monitor. The order is justified because RoaD and OPTED report orders-of-magnitude better sample efficiency than direct RL in AlpaSim ([RoaD](https://arxiv.org/abs/2512.01993); [OPTED](https://arxiv.org/abs/2609.20756)). nanoAlpamayo *is* the G5 student.

---

## 6. The first public artifact ships in seven days for $0

**Artifact: `physloop-gate` v0.** A paired release-gate core validated on synthetic truth, with a 441-scene power analysis. It needs no dataset license, so the PAI-AV clock never starts and the §3 confidentiality question does not arise. It runs on CPU only. Its value upstream is that it is exactly the missing layer, a paired two-checkpoint verdict with stratification and an INSUFFICIENT state. It is designed to consume what AlpaSim's challenge evaluator already emits: per-rollout pass/fail and scene scores on identical scene IDs ([local_evaluation README](https://github.com/NVlabs/alpasim/blob/20235345d7ea9eed5d5156099cf219cd35b2e4dc/e2e_challenge/local_evaluation/README.md)). Its power table tells every challenge participant what 441 scenes can and cannot certify.

The schedule is sized at about 21 focused hours: one Ladders-sized 3-hour block per day. Under Option (C) it runs Days 1–7 from D. At 6–9 h/week, ship Days 1, 2, 5 and 7 in week 1 (the known-answer tests and power tables) and the Monte Carlo in week 2.

| Day | Huy (★ core) | Agents | Output |
|---|---|---|---|
| 1 | ★ Write `experiments/<date>-stage1a-synthetic-truth.md` from the template: hypothesis, mechanism, scope, kill, and decision rule. Fill predictions P-S1-1 to P-S1-3 (left blank here). ★ Derive the exact McNemar null and the zero-failure bound by hand | Scaffold the repository, CI and test harness | The experiment file with Huy's predictions |
| 2 | ★ Implement the `verdict()` core: paired non-inferiority with a Tango score CI, the intersection–union rule across strata, and the INSUFFICIENT state | Encode the known-answer tests from §3 S1: 31,180/262,144; 7,795/32,768; 0.677% and 0.833%; 0.2991% and 0.3682%; Spearman 0.8 and 8/120 | Core plus failing-then-passing tests |
| 3 | ★ Implement the road-clustered paired bootstrap and the Durkalski variance adjustment, and specify the generator (stratum sizes, road clusters, effect sizes) | Build the synthetic generator to Huy's specification | Generator with known truth |
| 4 | ★ Diagnose every Monte Carlo deviation from expectation | Run 1,000-run Monte Carlo sweeps (with and without stratification; null and −30% effects; 2% and larger strata); plots | False-GO, false-NO-GO and INSUFFICIENT rates |
| 5 | ★ Power analysis: MDE (1.94 / 2.72 / 3.90 pp), the non-inferiority margin (2.67 / 3.77 pp), stratum power (≤ 0.2% at 9 scenes), and the G3 sample size (n ≈ 28–34; PASS and KILL thresholds at n = 10, 20, 30) | Tables and figures | The "what 441 scenes can certify" table |
| 6 | ★ Reconcile predictions against measurements and explain every miss causally. Write the allowed-claims section of the README | Packaging, documentation build | Reconciliation recorded in the experiment file |
| 7 | Publish the repository (Huy's action). Open an AlpaSim issue proposing the paired-verdict layer for `e2e_challenge/local_evaluation`. Write the three-sentence journal entry | Release notes | **Public artifact + upstream issue** |

**Acceptance** is S1's. **Cost** is $0. The ownership rule holds throughout: agents never fill prediction slots, and every ★ core is Huy's.

---

## 7. Huy owns every instrument, statistic, loss and prediction

The split in `CLAUDE.md` is restated per stage. "Owner proof" for each ★ core borrows Ladders' C5 rule: a representative cold reconstruction, or the diagnosis of a deliberately planted defect, establishes ownership without redoing every delegated line (`ladders/plan/PLAN.md`, C5).

| Stage | ★ Huy writes (instruments, statistics, RL loss math, verifier logic, predictions, diagnoses) | Agents write (plumbing, harnesses, scaffolds, tests, sweeps, plots, reviews) |
|---|---|---|
| S1 | `verdict()`, the margin, the cluster bootstrap, the power derivations, the predictions | Generator harness, Monte Carlo runner, CI, plots, packaging |
| S2 | Reliability statistics, strata, taxonomy mapping, validity-record semantics, discordance diagnosis | AlpaSim deployment, configs, sweep orchestration, manifests, telemetry |
| S3 | Agreement statistics, envelope semantics, the cap decision | Renderer and traffic configs, CAT-K seeding PR plumbing and tests, sweeps |
| S4 | Ladder design and assumption, pairwise statistics, format verdicts, the S4b kernel diagnosis | ModelOpt runs, exports, sweeps, release-note template |
| S5 | Pre-registration, statistics, the verdict, the paper's argument | Adapters, NAVSIM and `py123d_garage` runs, plots, LaTeX |
| S6 | Monitor logic (the verifier), thresholds, intervention-rate statistics | C++ build, gRPC plumbing, tests, sweeps |
| S7 | RL loss math, reward and verifier logic, decoder math, leakage audit, diagnoses | Data loaders, training and eval harnesses, launch scripts |
| S8 | The cost model and the bottleneck diagnosis | Telemetry extraction, profiling runs, plots |

**The learning evidence supports the split.**

- A randomized Anthropic study found that AI assistance cut quiz mastery by **17% (Cohen's d = 0.738, p = 0.010)**, with no significant speed gain. Three of the six interaction patterns, the cognitively engaged ones, preserved learning ([Shen & Tamkin, arXiv 2601.20245](https://arxiv.org/abs/2601.20245)).
- In a high-school field study, students given unguarded GPT access did 17% worse once access was removed ([Bastani et al. 2025, PNAS](https://pubmed.ncbi.nlm.nih.gov/40560616/)).
- Retrieval practice beats restudy on delayed tests, even though restudy raises confidence ([Roediger & Karpicke 2006](https://pubmed.ncbi.nlm.nih.gov/16507066/)).

The protocol follows from this. Huy writes the prediction or derivation first, and AI comparison comes second.

---

## 8. Skill coverage: evaluation and VLA work are covered; C++ and multi-node training are thin

The frequencies below are the shares of qualification sections that mention each cluster, across 43 comparable NVIDIA AV postings and 212 postings from 14 competitors. The source is primary job-board feeds read on 2026-10-03, and the notes advise ±5–10 points ([role notes Q4](https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite)). ● marks primary evidence; ◐ marks partial evidence.

| Skill cluster | NVIDIA | Competitors | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | Posting IDs this evidences |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C++ | 67% | 58% | | | | | | ● | ◐ | | JR2025038, JR2024273, JR2019758, JR2010863 |
| Python | 65% | 64% | ● | ● | ● | ● | ● | ◐ | ● | ● | JR2014651, JR2010863 |
| PyTorch / JAX | 35% | 40% | | | | ● | | | ● | | XPeng 8465375002 |
| VLA / VLM / LLM / foundation models | 49% | 27% | | | | ◐ | | | ● | | JR2019758, JR2005597, Waymo 8152556 |
| Evaluation / metrics / benchmarks | 28% (51% of responsibilities) | 39% (57%) | ● | ● | ● | ● | ● | ● | ◐ | | JR2014651, JR2024273, JR2025038, JR2019461 |
| Simulation | 35% | 24% | | ● | ● | ● | ● | ◐ | ◐ | ◐ | JR2010863, JR2005597 |
| Motion planning / behavior | 30% | 29% | | | | | ◐ | ◐ | ◐ | | JR2024273, Kodiak 4294898009 |
| RL / post-training / reward | 23% | 16% | | | | | | | ● | ◐ | JR2010863, JR2022179, JR2005597 |
| Imitation / end-to-end driving | 23% | 7% | | | | | | | ● | | JR2022179, JR2019758 |
| World models / generative / neural reconstruction | 21% | 13% | | ◐ | ● | | | | ◐ | | JR2020021, Waymo 8027348 |
| Data pipelines / data engine | 23% | 17% | ◐ | ◐ | | | | | ● | ◐ | JR2022179, Waymo 7593522 |
| CUDA / GPU kernels | 40% | 13% | | | | ◐ (S4b) | | | | ◐ | JR2005597, JR2005596 |
| Real-time / embedded | 35% | 18% | | | | ◐ | | ◐ | ◐ | ◐ | JR2019758, JR2025038 |
| Distillation / quantization / TensorRT | 7% (12% of responsibilities) | 16% | | | | ● | | | ● | | JR2025038, JR2016768, Zoox inference |
| Large-scale distributed training | 14% | 18% | | | | | | | ◐ (one node) | | JR2005597, XPeng 8465375002 |
| Kubernetes / SLURM / Ray | 9% | 12% | | ◐ | | | | | | ◐ | JR2010863 |
| Safety / V&V / standards | 19% | 19% | ◐ | ◐ | ◐ | ◐ | | ● | | | JR2014137, JR2023529, Wayve 0620aa63 |
| Publications | 33% | 24% | | | | | ● | | ◐ | | JR2005596, Waymo 8075587 |
| Open-source contributions | 14% | 4% | ● | ● | ● | ◐ | ◐ | ◐ | ◐ | ◐ | JR2019758, JR2005596 |
| Shipped production | 28% | 24% | | | | | | | | | Not addressable by a solo project |

**Uncovered or thin gaps, named honestly:**

- **C++.** It is the most common hard requirement, and only S6 covers it fully. The old C++ TensorRT runtime from `docs/03` Stage 4 is not in this plan. A second C++ artifact would be the AlpaSim monitor service, or a later DriveOS-style runtime.
- **Large-scale and multi-node distributed training (14% / 18%).** Uncovered here. It overlaps Ladders V9 ("parallelism on one node and across two"), which is the strongest argument for the overlay shape (`ladders/plan/PLAN.md` header).
- **Real-time and embedded work on actual Thor hardware.** Without hardware this is only partial. The Jetson AGX Thor kit costs $5,499, and DRIVE AGX Thor runs DriveOS with GPU-memory carveouts ([DRIVE AGX](https://developer.nvidia.com/drive/agx)).
- **Kubernetes and SLURM.** Optional, through AlpaSim's `SLURM_ENROOT` method.
- **Shipped production and years of experience.** Not addressable. Thirteen of the 43 NVIDIA postings (30%) require 12+ years; none of the 212 competitor postings does. The postings name public evidence as a complement, never as a substitute ([role notes Q5](https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite)).
- **Competition wins.** Named only in NVIDIA's end-to-end posting, and blocked for 2026 by the organization-email rule.

---

## 9. Risks and dated kill checks

**Dated kill checks.** Calendar dates assume D = 2026-12-08. Shift them for any other D.

| Check | When | Rule |
|---|---|---|
| KC-1 | D+1 (2026-12-15) | `verdict()` controls false-GO on synthetic truth; otherwise fix before any GPU spend |
| KC-L | Before the first public number derived from PAI-AV or NuRec (target D+3) | Written §3 answer, or the fallback (code + manifests + synthetic results). The fallback is itself untested (inference): the same written request asks whether scene-ID manifests and aggregate statistics fall under §3 |
| KC-2 | D+3 (2026-12-29) | Reliability ≥ Huy's pre-registered floor; otherwise stop |
| KC-3 | D+5 (2027-01-12) | Coverage inside the envelope ≥ Huy's floor; otherwise tighten the cap and report the loss |
| KC-4 | D+7 (2027-01-26) | Known-group upper bound ≥ Huy's value; otherwise downgrade validity claims |
| KC-6 | D+8 (2027-02-02) | The monitor fits in the loop without changing policy semantics; otherwise report it offline only |
| KC-5 | D+11 (2027-02-23) | The three-way ρ rule: PASS / KILL / INSUFFICIENT |
| KC-7 | D+20 (2027-04-27), or any NVIDIA distillation release | Student non-inferiority margin, or pivot to evaluating NVIDIA's student |

**Risks.**

| Risk | Check (date) | Trigger → response | Source |
|---|---|---|---|
| License §3 blocks publication of gate results | KC-L | No permission → publish code, scene-ID manifests and synthetic results only. Whether manifests and aggregates are themselves covered by §3 is unadjudicated, so ask in the same request | [PAI-AV license prompt](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles); [NuRec card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) |
| The 12-month license term starts at first download | No download before S2 begins; expiry = first download + 12 months | Expiry inside S7 → finish data-dependent work first, or re-accept | same |
| NVIDIA ships the 10B→~2B distillation script | GTC Berlin 2026-10-20 to 2026-10-22; every recipes, AlpaGym and Edge-LLM release | Release → KC-7(a) pivot | [AlpaGym ONBOARDING](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/docs/ONBOARDING.md); [NVIDIA blog](https://blogs.nvidia.com/blog/alpamayo-2-super/) |
| NVIDIA ships a paired gate or correlation study | Each AlpaSim sync (the last `main` sync, 2026-08-11, is overdue); the recipes training-efficiency "Methodology" chapter, announced as covering same-window paired runs and dose–response ladders but scoped to training performance | Release → contribute G2–G3 evidence upstream (the `docs/02` §6 fallback) | [AlpaSim commits](https://github.com/NVlabs/alpasim/commits/main/src/grpc/alpasim_grpc/v0/runtime.proto); [training-efficiency overview](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/docs/training_efficiency/00_Overview.md) |
| Too few independent policies for G3 | Pilot at D+8; KC-5 | State n_families and n_checkpoints; never count ladder variants as convergent n; INSUFFICIENT is an acceptable outcome | [pseudo-sim](https://arxiv.org/html/2506.04218v3) |
| AlpaGym is fragile: unmerged PR #18 pin, frozen mirror | Every recipes or AlpaGym change | Build nothing on AlpaGym internals; its profile in S8 stays optional | [PR #18](https://github.com/NVlabs/alpamayo-recipes/pull/18) |
| Seeds and nondeterminism (NuRec run-to-run variation; unseeded CAT-K) | S2 | Report them as noise sources, or run with CAT-K disabled (the default) | [issue #128](https://github.com/NVlabs/alpasim/issues/128) |
| Prices and supply: Nebius H100 +17% on 2026-10-01; thin vast.ai H100 supply (n = 5) | Each stage start | Re-price; fall back to interruptible RTX PRO 6000 | [Nebius](https://nebius.com/prices); [vast.ai](https://vast.ai/pricing) |
| Budget overrun | Every stage boundary | Stop at the boundary; never mid-stage | `docs/02` §4 practice |
| Collision with the Ladders apply-by window | 2026-11-30 and 2026-12-07 | Under (C), AV hours stay at 0 until 2026-12-07 unless Huy accepts overlap | `ladders/plan/PLAN.md` C6 |
| Challenge | 2026-10-18 (registration); 2026-10-31 (board closes) | Skip without an organization email (D-008) | [challenge Space](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026) |
| Regulatory wording drifts | ≈ 2027-01-30, when R[185] enters into force absent objection | Update the allowed-claims wording; never claim compliance | [C.N.288.2026](https://treaties.un.org/doc/Publication/CN/2026/CN.288.2026-Eng.pdf) |
| Hours assumption wrong (P-003) | After S1 actuals (D+1) | Re-forecast every window | Ladders C1 |
| Ownership drift: agents writing ★ cores | Every stage close | Owner proof by cold reconstruction or planted-defect diagnosis; otherwise redo the core | [Shen & Tamkin](https://arxiv.org/abs/2601.20245) |

---

## Conclusion

The verification's most useful finding is a confirmation, not a flaw. NVIDIA's own model card scored the same Alpamayo 1.5 weights at 0.81 and then 1.37. Its challenge cannot separate teams #2–#5. Its papers exclude the large-deviation events (beyond 4 m) that its challenge scores as failures. The instrument problem PhysLoop set out to attack is real and visible in primary sources. What changes is the size of the honest claim: from "the first predictive-validity number" to "a reliable, renderer- and traffic-robust instrument that responds to known degradations and converges across paradigms." That claim is smaller, decidable, publishable, and upstreamable into the one repository (AlpaSim) that accepts outside work.

The strategic implication runs against the instinct to switch fields again. Every AV deliverable that survived verification (the paired gate, construct validity, closed-loop precision verdicts, the expert-kernel diagnosis) is built from skills Ladders is already training: statistics for evaluation, kernels and rooflines, FP8/NVFP4, RL post-training. The artifacts that did not survive (AlpaGym pull requests, expert-quantization novelty, a racing distillation recipe) were the ones that depended on outrunning NVIDIA rather than measuring it. So the right shape for PhysLoop is an overlay that measures what NVIDIA ships, started after Huy's Ladders deadlines or at a recorded, bounded un-park, with its first $0 artifact ready on day one.

---

## 10. Appendix: merged claim ledger

This ledger merges the seven research notes, all accessed on 2026-10-03. It includes every row marked CORRECTED, NOT FOUND or UNVERIFIABLE, plus the VERIFIED rows the plan depends on.

**ID prefixes** name the source note:

- **EV**: evaluation and release gating
- **AM**: Alpamayo models and data
- **TL**: NVIDIA tooling and production
- **DP**: deployment, hardware and compute
- **E2E**: driving SOTA and post-training
- **CU**: first-principles curriculum
- **RS**: role and skill demand

**Rules for reading the table:**

- Where two notes disagree, part L records the resolution.
- "Doc §" refers to the planning documents under `docs/`.
- A row in one lane marked UNVERIFIABLE "(out of scope)" that a sibling note verified is listed with its resolution.

### A. Challenge, scoring and Drive-IRT

| ID | Claim as written (doc §) | Status | Correct value / note | Source |
|---|---|---|---|---|
| EV-1.1 | Challenge run by NVIDIA AV Research, KE:SAI, OpenDriveLab (02 §0) | VERIFIED | NVIDIA AV Research Group, KE:SAI, HKU (OpenDriveLab); KE:SAI's blog names the NVIDIA partner "ASPIRE Group" | [Space](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026); [KE:SAI](https://kesai.eu/blog/2026-09-08-common-task-framework/) |
| EV-1.3 / TL-23 | "Leakage-aware 1,761/441 NuRec split" (02 §0, G1) | VERIFIED (qualified) | Curated from NuRec 26.01 and 26.04 (2,202 of 2,523 public scenes); road-sharing scenes grouped; a local development split, not the official test set; community PR #167, merged 2026-08-31 | [local_evaluation README](https://github.com/NVlabs/alpasim/blob/e2e_challenge/e2e_challenge/local_evaluation/README.md); [PR #167](https://github.com/NVlabs/alpasim/pull/167) |
| EV-1.4 / TL-24 | Drive-IRT "capability score with 95% intervals" (02 §0) | VERIFIED (method); resolves TL-24's UNVERIFIABLE | Confidence level 0.95 by default; intervals from the posterior std of a mean-field variational fit; rank intervals from Monte Carlo; no bootstrap; no seed or scene-sampling variance | [evaluate.py](https://github.com/NVlabs/alpasim/blob/e2e_challenge/e2e_challenge/local_evaluation/evaluate.py) |
| EV-1.5 | Deadline Oct 31 2026 (02 §0, §3) | VERIFIED | Board closes 2026-10-31. Also: registration cutoff 2026-10-18; rules frozen 2026-09-15; final results targeted for 2026-11-15 | [Space](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026) |
| EV-1.6 | "4-page report" (02 §0) | VERIFIED | 4 pages, excluding references and appendix; not a NeurIPS paper and must not imply acceptance; not anonymous; latency, memory and limitations sections required | [Space](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026) |
| EV-1.7 | "NeurIPS presentation" (02 §0) | VERIFIED | A results presentation, with no workshop named; prizes are two DGX Spark per track (about USD 4,700 each) | [Space](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026) |
| EV-1.8 / TL-25 / DP-Q2 | "16 GiB VRAM at 0.1 s model-time per call" (02 §0; 01 C2) | CORRECTED (the 0.1 s part) | 16 GiB is a hard per-instance limit. 0.1 s is a target, enforced through an unpublished per-track throughput budget. The EC2 preset runs 16 replicas on GPUs 4–7 with 2 concurrent rollouts each. Image ≤ 40 GiB; no outbound network; read-only root filesystem | [challenge README](https://github.com/NVlabs/alpasim/blob/e2e_challenge/e2e_challenge/README.md) |
| EV-1.9 / DP-Q2 | Registration requires an organization email (02 §0, §6) | VERIFIED | The captain's verified HF primary email must be an organization address; teams must be affiliated and are reviewed; the binding date is 2026-10-18 | [Space](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026) |
| EV-1.10 | Hidden test set (question) | VERIFIED | Private held-out scenes. KE:SAI says thousands, from 10 countries. The nuPlan track uses a private dataset | [challenge README](https://github.com/NVlabs/alpasim/blob/e2e_challenge/e2e_challenge/README.md); [KE:SAI](https://kesai.eu/blog/2026-09-08-common-task-framework/) |
| EV-1.11 | Challenge treated as one NuRec benchmark (02 §0, G1) | CORRECTED | Two tracks: Physical AI AV, and nuPlan (MTGS rendering, vehicles only) | [challenge README](https://github.com/NVlabs/alpasim/blob/e2e_challenge/e2e_challenge/README.md) |
| EV-1.12 | Optional submission by Oct 31 (02 §3) | VERIFIED; constraints missing from the docs | 3 official submissions per team per UTC month, shared across tracks; failed evaluations still count; warmups use 32 private scenes, 3 per rolling 30 days | [Space](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026) |
| EV-1.13 | Leaderboard state (question) | VERIFIED (snapshot) | PAI: 25 shown of 73 submissions. #1 PCS 2240 (2186.36–2293.58), average scene score 0.665. #2–#5 PCS 2101–2128 with overlapping rank intervals. Host entry 2000. nuPlan #1 PCS 1784 | [live board](https://nvidia-alpasime2eclosedloopchallenge2026.hf.space/gradio/) |
| EV-1.14 / TL-76 | No go/no-go rule comparing B with A (01 §0; 02 §0) | VERIFIED (absence) | Panel ranking only; zoib needs at least 6 subjects | [evaluate.py](https://github.com/NVlabs/alpasim/blob/e2e_challenge/e2e_challenge/local_evaluation/evaluate.py) |
| EV-1.15 / TL-46 | "`RolloutSpec` seeds" (02 G1) | VERIFIED (branch-dependent) | Challenge branch: the seed sits in `DriveSessionRequest` (egodriver.proto, field 2). `main`: `RolloutSpec.random_seed = 5`, where rollout k uses seed + k; removed 2026-03-10, re-added 2026-08-17 by external PR #149; trafficsim ignores it | [runtime.proto](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/grpc/alpasim_grpc/v0/runtime.proto); [PR #149](https://github.com/NVlabs/alpasim/pull/149) |
| EV-1.16 / TL-48 | "CAT-K traffic" (02 G1; 01 C5) | VERIFIED (main; off by default) | `trafficsim=disabled` by default; challenge presets set trafficsim to null; whether official runs use reactive traffic is UNVERIFIABLE | [trafficsim README](https://github.com/NVlabs/alpasim/blob/main/src/trafficsim/README.md); [CHANGELOG](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/CHANGELOG.md) |
| EV-1.17 / TL-45 | "AlpaSim scorers" (02 G1) | VERIFIED | Collision, OffRoad, MinDistanceToObstacle, GroundTruth, MinADE, PlanDeviation, Image, Safety, plus OpenLoopCollision on `main` | [scorers](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/eval/src/eval/scorers/__init__.py) |
| EV-1.18 | Policy zoo "VaVAM, Transfuser" (02 G3) | VERIFIED | Ships LTF, DiffusionDrive and GTRS adapters plus a VAVAM sample; LTF is NAVSIM-trained (nuPlan track) | [challenge README](https://github.com/NVlabs/alpasim/blob/e2e_challenge/e2e_challenge/README.md) |
| EV-1.20 / TL-77 | "No evaluation of policy + guardrail" (02 §0) | NOT FOUND (absence) | The hook exists (`safety_monitor_safe` → `safety_monitor_triggered`); there is no implementation and no published study | [safety.py](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/eval/src/eval/scorers/safety.py) |
| EV-1.21 / TL-78 | Risk: "NVIDIA publishes its own gate/correlation study" (02 §6) | CORRECTED (partly materialized) | Sim2Val (a sim–real correlation for one policy) and OmniDreams §9.4.1 (cross-renderer rank preservation, n = 4) are published. No release gate is published, and none was found in the tooling lane | [Sim2Val](https://arxiv.org/abs/2506.20553); [OmniDreams](https://arxiv.org/html/2606.03159) |
| EV-1.22 | "RAND Driving to Safety (RR1478)" (03 §7) | VERIFIED | Hundreds of millions to hundreds of billions of miles; supports the ban on per-mile claims | [RAND](https://www.rand.org/pubs/research_reports/RR1478.html) |
| EV-1.23 | Cross-simulator tooling (new fact) | VERIFIED | `py123d_garage` covers CARLA 2.0, NAVSIM v1 and AlpaSim; the NAVSIM v2 navhard leaderboard bundled with Drive-IRT has 115 submissions from 24 teams | [py123d_garage](https://github.com/kesai-labs/py123d_garage) |
| EV-1.24 | "% rollouts beyond a 2 m deviation cap" (03 1b) | CORRECTED (context) | NVIDIA's convention is 4 m: `max_dist_to_gt_trajectory: 4.0`, and AR1 excludes events beyond 4 m. Report both 2 m and 4 m | Config path `src/wizard/configs/e2e_challenge/ec2.yaml` (challenge branch); [AR1](https://arxiv.org/abs/2511.00088) |
| EV-score | How the official score is built (context) | VERIFIED | The scene score is 0 on `collision_at_fault`, then `offroad`, then `left_corridor_laterally` (≥ 4 m); otherwise min(progress/0.8, 1); clips with ground-truth travel < 5 m get full score. At-fault means front ∨ lateral. zoib is fitted for 1,000 epochs with seed 42, pinned to 82ddd5a4. Ranking uses the upper bound of the 95% rank interval. Local references: 8 subjects × 441 scenes × 3 rollouts, on an arbitrary local scale | [challenge README](https://github.com/NVlabs/alpasim/blob/e2e_challenge/e2e_challenge/README.md); [local_evaluation README](https://github.com/NVlabs/alpasim/blob/e2e_challenge/e2e_challenge/local_evaluation/README.md) |
| EV-2.1 | Drive-IRT repository (02 §7) | VERIFIED | Created 2026-03-18; last push 2026-08-12; **no license file** (analysis only, no redistribution) | [drive-irt](https://github.com/kesai-labs/drive-irt) |
| EV-2.2 | Which IRT model (02 §0) | VERIFIED | The challenge uses `BetaIrt` (zoib, 2PL-type, per-item discrimination); the library's default config is 1PL on binarized NavHard data | [drive-irt](https://github.com/kesai-labs/drive-irt); [evaluate.py](https://github.com/NVlabs/alpasim/blob/e2e_challenge/e2e_challenge/local_evaluation/evaluate.py) |
| EV-2.4 | How the 95% intervals are computed | VERIFIED | loc ± 1.96·scale from mean-field Normal guides, which can under-report spread; rank intervals from 100,000 Monte Carlo draws; the optional `BootstrapIRT` resamples items i.i.d. and is unused | [docs ch. 6](https://ln2697.github.io/navhard-leaderboard-2/docs/preliminary/06_parameter_uncertainty_quantification.html) |
| EV-2.5 | External validation of Drive-IRT | VERIFIED (none exists) | Only internal checks: self_test and seed rank-spread; no held-out ground-truth ranking exists | [docs ch. 1](https://ln2697.github.io/navhard-leaderboard-2/docs/preliminary/01_motivation.html) |
| EV-2.7 | Teach-back Q3: ρ between the gate and Drive-IRT on 10 policies (01 §4) | CORRECTED (framing) | Both are fitted to the same local rollouts, so ρ measures aggregation robustness, not validity. The prediction slot is preserved and re-labelled as R0 | [local_evaluation README](https://github.com/NVlabs/alpasim/blob/e2e_challenge/e2e_challenge/local_evaluation/README.md) |
| TL-26 / AM-10.8 | Deadline, report and registration rules (02 §0) | UNVERIFIABLE in two lanes → resolved VERIFIED by EV-1.5 and EV-1.9 | Also: the leaderboard was wiped after maintenance from 2026-08-31 to 09-06, and the terms were under legal review on 2026-09-07 | [issue #170](https://github.com/NVlabs/alpasim/issues/170) |
| TL-79 | Risk "registration blocked (no org email)" (02 §6) | UNVERIFIABLE in the tooling lane → resolved by EV-1.9 | The binding date is 2026-10-18 | [Space](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026) |

### B. Simulator validity evidence and measurement science

| ID | Claim as written (doc §) | Status | Correct value / note | Source |
|---|---|---|---|---|
| EV-3.1 | nuPlan "fundamentally misaligned" (01 C3) | VERIFIED | The evidence is a trade-off table; no coefficient is reported | [arXiv 2306.07962](https://arxiv.org/abs/2306.07962) |
| EV-3.2 / EV-4.7 | Pseudo-sim r = 0.89 over 83 planners (01 C8) | VERIFIED (reference added) | Pearson r against nuPlan's *reactive* closed-loop simulation; 83 *privileged* planners from 5 families; the 3DGS renderer is never exercised; R² = 0.8 | [arXiv 2506.04218v3](https://arxiv.org/html/2506.04218v3) |
| EV-3.3 | Bosch ρ = −0.36 between L2 and Bench2Drive DS (01 C3; 03 P1) | VERIFIED as stated; **NOT REPRODUCIBLE** | n = 7 (Table 1). Recomputed: ρ = −0.143 (p = 0.76) and Pearson −0.60 (p = 0.15). No 5- or 6-method subset gives −0.36. Cite as "no significant correlation (n = 7)" | [arXiv 2605.00066v1](https://arxiv.org/html/2605.00066v1) |
| EV-3.4 | NAVSIM↔Bench2Drive ρ = 0.90 over 8; "rank ρ ≈ 0.9" (01 C8; 02 §0) | VERIFIED (reproduced) | ρ = 0.905, p = 0.0020; τ = 0.786. NAVSIM **v1** PDMS. 02 §0 mixes Pearson (pseudo-sim) with Spearman (Bosch) | [arXiv 2605.00066v1](https://arxiv.org/html/2605.00066v1) |
| EV-3.5 / EV-3.6 | EUVS: −24.6% PSNR translation; −30.6% with rotation; all below 15 (01 C5) | VERIFIED (primary) | Table 1: 20.67 → 15.59 and 20.40 → 14.15; all 9 methods below 15; NuRec not evaluated | [EUVS (CVF)](https://openaccess.thecvf.com/content/ICCV2025/papers/Han_Extrapolated_Urban_View_Synthesis_Benchmark_ICCV_2025_paper.pdf) |
| EV-3.7 | EUVS cited via papernotes.org (02 §7) | CORRECTED (source quality) | Use the CVF PDF or arXiv 2412.05256 | [arXiv 2412.05256](https://arxiv.org/abs/2412.05256) |
| EV-3.8 | Waymo approval guidelines (01 §2; 02 §0) | VERIFIED | Twelve criteria aggregated qualitatively; no quantitative targets in the paper | [arXiv 2505.09880](https://arxiv.org/abs/2505.09880) |
| EV-4.1 | "No published correlation between a neural-reconstruction simulator's score and any independent outcome" (01 §0) | CORRECTED | Sim2Val: per-episode ρ > 0.9 for 2 kinematic metrics; one policy; n = 138 + 75 episodes; 8-s closed loop; up to 82.9% variance reduction | [Sim2Val](https://arxiv.org/abs/2506.20553); [PMLR](https://proceedings.mlr.press/v305/luo25a.html) |
| EV-4.2 | "No study correlates AlpaSim or Drive-IRT scores with any outcome" (02 §0) | CORRECTED (AlpaSim) / VERIFIED (Drive-IRT) | NVIDIA's blog credits Sim2Val with AlpaSim-based validation; the paper redacts the simulator's name | [NVIDIA blog](https://developer.nvidia.com/blog/building-autonomous-vehicles-that-reason-with-nvidia-alpamayo/) |
| EV-4.3 | "No peer-reviewed coefficient exists…" (02 §0; 01 C8) | CORRECTED (narrowed) | Use the narrowed wording in §4 | [Sim2Val](https://arxiv.org/abs/2506.20553) |
| EV-4.4 | "Nothing exists for AlpaSim"; renderer agreement treated as new work (01 C8, C5) | CORRECTED (precedent exists) | OmniDreams §9.4.1: ranking preserved from NuRec to OmniDreams for 4 policies; incidents counted only within 4 m; no coefficient | [OmniDreams](https://arxiv.org/html/2606.03159) |
| EV-4.5 | Fidelity-vs-deviation curve as new G2 work (02 G2) | CORRECTED (partly done) | OmniDreams Fig. 14 plots 4-camera FVD against distance from the recorded trajectory | [OmniDreams](https://arxiv.org/html/2606.03159) |
| EV-4.6 | NVIDIA's deviation handling | VERIFIED | NVIDIA's papers *exclude* events beyond 4 m; the challenge *scores* them as failure | [AR1](https://arxiv.org/abs/2511.00088) |
| EV-4.8 | CARLA sim-to-real | VERIFIED (negative precedent) | 10 models, 9 real scenarios: simulation predicted real performance poorly; no coefficient | [Osiński](https://arxiv.org/abs/1911.12905) |
| EV-4.9 | Offline vs closed-loop precedents | VERIFIED | Codevilla (45 models); 2510.08571 (82 models) | [Codevilla](https://arxiv.org/abs/1809.04843); [2510.08571](https://arxiv.org/abs/2510.08571) |
| EV-4.10 | Other neural-reconstruction simulators (HUGSIM, Bench2Drive-R, RAD) | NOT FOUND (no external coefficient) | — | [RAD](https://arxiv.org/abs/2502.13144) |
| EV-4.11 | Cam2Sim | VERIFIED (single scenario) | 3DGS in CARLA vs a real vehicle; no ranking coefficient | [Cam2Sim](https://arxiv.org/abs/2607.04770) |
| EV-4.12 | Waymo scaling and WOSAC | VERIFIED (sim-to-sim only) | — | [arXiv 2506.08228](https://arxiv.org/abs/2506.08228) |
| EV-4.13 | Wayve's correlation studies | UNVERIFIABLE | No coefficient, n or method published | [GAIA-3](https://wayve.ai/thinking/gaia-3/); [GAIA-4](https://wayve.ai/thinking/gaia-4/) |
| EV-4.14 | Waabi Realism Score 99.7% | UNVERIFIABLE (company claim) | Scenario count not given | [Waabi](https://waabi.ai/simulator-realism-the-new-safety-standard-for-the-av-industry/) |
| EV-4.15 / EV-5.3 | PolaRiS r = 0.9 | VERIFIED (caveat) | Holds only after co-finetuning the evaluated policies on about 350 simulated demos | [PolaRiS](https://arxiv.org/abs/2512.16881) |
| EV-4.16 | Combining few real with many simulated trials | VERIFIED | Control variates (Sim2Val); prediction-powered inference (SureSim) | [SCAPE](https://arxiv.org/abs/2608.19425); [SureSim](https://arxiv.org/abs/2510.04354) |
| EV-4.17 | Fallback "add perturbed/quantized/early checkpoints" (02 §6) | CORRECTED (design caveat) | Variants are not independent policies and inflate ρ. Report within-family and between-family separately, and count ladders as known-group evidence | [SIMPLER](https://arxiv.org/abs/2405.05941) |
| EV-gap | Sim2Val simulator identity | UNVERIFIABLE | Redacted in arXiv v2 and PMLR; only the blog names AlpaSim. Whether it was the open stack or an internal superset is unknown | [Sim2Val](https://arxiv.org/abs/2506.20553) |
| EV-5.1 | Kadian SRCC | VERIFIED (caveat) | SRCC on success rose from 0.18 to 0.844 after tuning, but the tuning was in-sample on the same 9 models | [Kadian](https://arxiv.org/abs/1912.06321) |
| EV-5.2 | SIMPLER and MMRV | VERIFIED | Pearson penalizes non-linear recovery; the Bridge results use n = 3 | [SIMPLER](https://arxiv.org/abs/2405.05941) |
| EV-5.21 / EV-5.34 | Pairwise accuracy (WMT) | VERIFIED | COMET 83.4% vs BLEU 74.6% over 3,344 pairs | [Kocmi 2021](https://aclanthology.org/2021.wmt-1.57/) |
| EV-5.25 | Reliability vs validity | VERIFIED | Reliability is agreement between maximally similar methods; validity is agreement between maximally different ones | [Campbell & Fiske](https://doi.org/10.1037/h0046016) |
| EV-5.27 | VIM precision conditions | VERIFIED | Repeatability, intermediate precision, reproducibility | [JCGM 200:2012](https://www.bipm.org/documents/20126/2071204/JCGM_200_2012.pdf) |
| EV-5.28 | ASME V&V 40 | VERIFIED (scope page only) | Credibility scales with model influence; the standard is paywalled | [ASME](https://www.asme.org/codes-standards/find-codes-standards/assessing-credibility-of-computational-modeling-through-verification-and-validation-application-to-medical-devices) |
| EV-5.29 | NASA-STD-7009B | VERIFIED | Without a real-system referent, outcome validity stays at the low levels | [NASA](https://standards.nasa.gov/sites/default/files/standards/NASA/B/1/NASA-STD-7009B-Final-3-5-2024.pdf) |
| EV-5.33 | Real-is-Sim; Duckietown (robotics sim–real precedents) | VERIFIED (papers); sim–real coefficient NOT FOUND | Real-is-Sim reports only that ordering was preserved (n = 4 checkpoints, one task, 60 real trials each); Duckietown gives no sim–real coefficient. Small-n "ordering preserved" claims are common and weak | [Real-is-Sim](https://arxiv.org/abs/2504.03597); [Duckietown](https://arxiv.org/abs/2009.04362) |
| EV-5.39 | BenchBench | VERIFIED as preprint; venue UNVERIFIABLE | Agreement τ ≈ 0.65–0.99, depending on n | [arXiv 2407.13696](https://arxiv.org/abs/2407.13696) |
| EV-5.41 | Krippendorff's α thresholds | Bibliographic only; content UNVERIFIABLE | — | [doi](https://doi.org/10.1080/19312450709336664) |

### C. Statistics

| ID | Claim as written (doc §) | Status | Correct value / note | Source |
|---|---|---|---|---|
| EV-6.1 | A fails 30, B fails 24; b = 12, c = 6 (01 C6) | VERIFIED | 18 both fail / 12 A only / 6 B only / 405 both pass; ψ = 4.08% | arithmetic (evaluation notes) |
| EV-6.2 | P(X ≤ 6 \| 18, ½) = 31,180/262,144 ≈ 0.119 | VERIFIED | 0.118942 | [McNemar 1947](https://doi.org/10.1007/BF02295996) |
| EV-6.3 | Two-sided p ≈ 0.24 | VERIFIED | 0.237885 exact; mid-p 0.167; χ² without correction gives p = 0.157 | same |
| EV-6.4 | "Only discordant pairs carry information" | VERIFIED (for the test) | The effect estimate and its CI still use N (Tango) | [Tango 1998](https://doi.org/10.1002/(SICI)1097-0258(19980430)17:8<891::AID-SIM780>3.0.CO;2-B) |
| EV-6.5 | Exact McNemar alongside "bootstrap over logs" (01 C6) | CORRECTED (inconsistent design) | Road clustering makes McNemar anti-conservative. Use cluster-adjusted McNemar or a road-level bootstrap, and the wild cluster bootstrap-t for 5–30 clusters | [Durkalski 2003](https://doi.org/10.1002/sim.1438); [Field & Welsh 2007](https://doi.org/10.1111/j.1467-9868.2007.00593.x); [CGM 2008](https://doi.org/10.1162/rest.90.3.414) |
| EV-6.6 | Rule of three: 3/441 ≈ 0.68% (01 C6) | VERIFIED | Exact one-sided 0.6770%; two-sided Clopper–Pearson upper 0.8330% | [Hanley & Lippman-Hand](https://doi.org/10.1001/jama.1983.03330370053031) |
| EV-6.7 | "≤ 0.3% for 0/1,000" (02 §5) | CORRECTED (sidedness) | One-sided 0.2991%; two-sided 0.3682%; a two-sided ≤ 0.3% needs 1,228 scenes (998 one-sided) | [Clopper & Pearson](https://doi.org/10.1093/biomet/26.4.404) |
| EV-6.8 | ρ = 0.8 worked example (01 C8) | VERIFIED | Not significant: P = 8/120 = 0.067 | arithmetic |
| EV-6.9 | "Hence ≥ 10 policies" (01 C8) | CORRECTED | At n = 10, an observed 0.8 has a CI of [0.32, 0.95] (FHP) or [0.24, 0.96] (BW). A lower bound ≥ 0.6 needs n ≈ 28–34 | [FHP 1957](https://doi.org/10.1093/biomet/44.3-4.470); [BW 2000](https://doi.org/10.1007/BF02294183) |
| EV-6.10 | Kill rule "ρ < 0.6 on ≥ 10 policies" (02 §3; 03 §5) | CORRECTED (computable but not decidable) | Wrong kill 15.8% when the true ρ is 0.8, and wrong pass 24.3% when it is 0.4, at n = 10. Use the three-way rule and its thresholds (§4) | computed (evaluation notes) |
| EV-6.11 | 441 scenes suffice for G1 (implicit) | CORRECTED (power) | MDE 1.94 / 2.72 / 3.90 pp at ψ = 2.1 / 4 / 8%; the example has power 0.22; about 1,728 scenes are needed for 80% power on it; non-inferiority margins 2.67 / 3.77 pp | [Connor 1987](https://doi.org/10.2307/2531961) |
| EV-6.12 | Stage 1a: false-GO ≤ 5% at −30% on a 2% stratum (03) | CORRECTED (achievable only by abstention) | About 9 scenes; power ≤ 0.2%; zero failures in 9 still gives a 28.3% bound; INSUFFICIENT must block GO | computed |
| EV-6.13 / 6.14 | Removing stratification raises false-GO; INSUFFICIENT appears below the noise floor | VERIFIED (in principle) | 2.1 pp dilutes to 0.042 pp pooled | computed |
| EV-Holm | Holm 1979 step-down | UNVERIFIABLE online | No DOI; JSTOR is bot-protected | research note `evaluation_release_gating.md` (no primary URL) |
| EV-FHP/BW forms | Variance forms 1.06/(n−3) and (1+ρ²/2)/(n−3) | Confirmed only through secondary sources | The primary papers are paywalled | research note `evaluation_release_gating.md` (no primary URL) |
| EV-Connor forms | Connor/Miettinen formulas | Taken from SAS documentation (secondary) | The exact-McNemar power figures were computed independently | research note `evaluation_release_gating.md` (no primary URL) |
| EV-AV practice | Paired tests in AV evaluation | VERIFIED (sparse) | One Wilcoxon test on 1,000 paired WOMD scenarios, unclustered (Keswani & Bhattacharyya); no road-cluster bootstrap paper found | [arXiv 2512.17586](https://arxiv.org/abs/2512.17586) (as cited in the evaluation notes) |
| CU-P5-Wald | SPRT boundaries A ≈ (1−β)/α, B ≈ β/(1−α) | UNVERIFIABLE (not text-verified) | Wald 1945 verified only as a bibliographic record | [Wald 1945](https://doi.org/10.1214/aoms/1177731118) |
| CU-P5-constructs | "Variance decomposition" and "data contracts" as framed in 03 P5 | NOT FOUND (no canonical source) | PhysLoop's own constructs | research note `first_principles_curriculum.md` (no primary URL) |

### D. Regulation, standards and industry gating

| ID | Claim as written (doc §) | Status | Correct value / note | Source |
|---|---|---|---|---|
| EV-7.1 | NATM; "simulator credibility (NATM)" (02 §0; 03 1b) | VERIFIED (guidance, not binding) | Annex III: credibility defines the envelope; partly superseded by ECE/TRANS/WP.29/2024/39; the live URL in 02 §7 is blocked, so use Wayback | [NATM via Wayback](https://web.archive.org/web/20240814155120/https://unece.org/sites/default/files/2022-05/WP.29-187-08e.pdf) |
| EV-7.2 | "UN ADS GTR adopted June 2026" (02 §0) | VERIFIED (date); GTR number UNVERIFIABLE | Adopted 2026-06-24 per NHTSA; established by AC.3 at WP.29's 199th session; "No. 26" appears only in secondary sources | [NHTSA 91 FR 48485](https://www.govinfo.gov/content/pkg/FR-2026-07-31/html/2026-15483.htm) |
| EV-7.3 | "UN R185 from Jan 2027" (02 §0) | VERIFIED (conditional) | Draft No. [185], notified 2026-07-30; in force about 2027-01-30 unless more than one-fifth object; bracketed number; not in force on 2026-10-03 | [C.N.288.2026](https://treaties.un.org/doc/Publication/CN/2026/CN.288.2026-Eng.pdf) |
| EV-7.4 | No-compliance list: ISO 26262 / 21448 / PAS 8800 / UL 4600 / R157 / R171 (02 §5) | CORRECTED (incomplete) | Add R[185] and the ADS GTR; write ISO/PAS 8800:2024; "NATM compliance" is not meaningful | rows below |
| EV-7.5 | UN R157 (02 §5) | VERIFIED | No ML requirements; simulation needs validated scope and correlation with physical tests | [C.N.344.2025](https://treaties.un.org/doc/Publication/CN/2025/CN.344.2025-Eng.pdf) |
| EV-7.6 | R171, and "what R171 asks of ML validation" (03 P7) | VERIFIED; CORRECTED (no ML content) | Technology-neutral; virtual testing optional, with an Annex 5 credibility assessment if used; in-service monitoring with at least yearly reports | [C.N.539.2025](https://treaties.un.org/doc/Publication/CN/2025/CN.539.2025-Eng.pdf) |
| EV-7.7 | ADS GTR / R[185] content | VERIFIED (drafts) | No ML-specific requirements; assumes no online self-modifying learning; baseline is a competent and careful human driver; update V&V is not a paired comparison | [NHTSA FR](https://www.govinfo.gov/content/pkg/FR-2026-07-31/html/2026-15483.htm) |
| EV-7.8 | ISO/PAS 8800 (02 §0, §5; 03 P7) | VERIFIED (metadata); clause content UNVERIFIABLE; the "UL summary" is UNVERIFIABLE | Published 2024-12-13; 172 pp.; ML-focused | [ISO via Wayback](https://web.archive.org/web/20260813162702/https://www.iso.org/standard/83303.html) |
| EV-7.9 | ISO 21448:2022 | VERIFIED (metadata); content UNVERIFIABLE | Marked "to be revised" since 2025-10-14 | [ISO via Wayback](https://web.archive.org/web/20261001154448/https://www.iso.org/standard/77490.html) |
| EV-7.10 | UL 4600 | VERIFIED (edition); content UNVERIFIABLE | Edition 3, 2023-03-17 | [UL store](https://www.shopulstandards.com/ProductDetail.aspx?productId=UL4600_3_S_20230317) |
| EV-7.11 | ISO 26262:2018 | VERIFIED (metadata); applying clause 11 tool confidence to simulators is UNVERIFIABLE | The 3rd edition is at DIS stage | [ISO via Wayback](https://web.archive.org/web/20260928113321/https://www.iso.org/standard/68390.html) |
| EV-7.12 / EV-5.31 | EU 2022/1426 | VERIFIED | Part 4 §1.4: credibility defines the envelope; simulation needs correlation with physical tests | [EU 2022/1426](http://publications.europa.eu/resource/celex/32022R1426) |
| EV-7.13 | "How the industry gates releases: paired regression, staged rollout…" (02 §0) | CORRECTED (conflation) | In the regulatory texts, paired regression against the previous release is NOT FOUND and staged rollout is NOT FOUND | rows 7.1–7.7 |
| EV-8.1 | "Waymo publishes its approval framework, not its model" (01 §2) | VERIFIED (framework); "not its model" UNVERIFIABLE | — | [arXiv 2505.09880](https://arxiv.org/abs/2505.09880) |
| EV-8.2 | "No single numeric gate" at Waymo | VERIFIED (caveat) | Each criterion has its own targets, such as minimum scores per scenario group | same |
| EV-8.3 | "No single numeric gate" across industry | CORRECTED | Zoox gates on one combined quantitative risk estimate (CIF) | [Zoox 2026](https://zoox.com/assets/1786550724-zoox-safety-case-framework-2026.pdf) |
| EV-8.4 | "Paired regression vs the previous release" | CORRECTED | At Waymo it is a diagnostic sub-input; the comparators are human-derived benchmarks. In CAT, a regression in one pedestrian group was accepted | [arXiv 2505.09880](https://arxiv.org/abs/2505.09880); [CAT](https://arxiv.org/abs/2212.08148) |
| EV-8.5 | "Stratified scenario groups" | VERIFIED | — | same |
| EV-8.6 | "Validity-checked simulation" | NOT FOUND in W25; VERIFIED in other Waymo papers | Waymo 2020; CAT | [arXiv 2011.00054](https://arxiv.org/abs/2011.00054) |
| EV-8.7 | "Staged rollout" | VERIFIED (Waymo) | — | [arXiv 2505.09880](https://arxiv.org/abs/2505.09880) |
| EV-8.8 | Aurora | VERIFIED | Holistic launch decision by a Safety Review Board; no release-to-release regression described | [Aurora 2025](https://downloads.ctfassets.net/9i0s3p5vkth9/Gx5t3M6aWWUk8yEsXIZ8N/bf9767dfeb33360114438df17739d8be/Aurora_Driverless_Safety_Report_2025_with_Addendum.pdf) |
| EV-8.9 | Mobileye RSS | VERIFIED (title corrected) | "On a Formal Model of Safe and Scalable Self-driving Cars" | [arXiv 1708.06374](https://arxiv.org/abs/1708.06374) |
| EV-8.10 | Other developers | VERIFIED (Motional) / UNVERIFIABLE (Tesla, Cruise, Wayve) | Motional: a safety board makes go/no-go decisions, with about 15 years-equivalent of simulated driving per release. The NVIDIA safety report contains none of the five elements | [Motional VSSA](https://motional.com/sites/default/files/inline-files/Motional_Voluntary_Safety_Self-Assessment_2026.pdf); [NVIDIA safety report](https://docs.nvidia.com/self-driving-cars/autonomous-driving-safety-report/) |
| CU-P7 | RAND figures | VERIFIED | 275 million failure-free miles (about 12.5 years for 100 cars); 8.8 billion to estimate within 20%; 11 billion to show 20% better | [RAND PDF](https://www.rand.org/content/dam/rand/pubs/research_reports/RR1400/RR1478/RAND_RR1478.pdf) |

### E. Failure taxonomy

| ID | Claim as written (doc §) | Status | Correct value / note | Source |
|---|---|---|---|---|
| EV-C7 | C7's seven categories (01 C7) | CORRECTED (not a standard) | At-fault maps only to NAVSIM and AlpaSim; oscillation has no crash-taxonomy counterpart; missing are VRUs, intersection geometry, stop signs, wrong-way, backing and severity tiers | [NHTSA typology](https://rosap.ntl.bts.gov/view/dot/6281) |
| EV-9.1 | NHTSA pre-crash typology | VERIFIED | 37 scenarios covering about 5,942,000 crashes | [DOT HS 810 767](https://rosap.ntl.bts.gov/view/dot/6281) |
| EV-9.2 | NHTSA SGO data elements | VERIFIED | No fault or responsibility field | [SGO dictionary](https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Data_Element_Definitions.pdf) |
| EV-9.3 | Waymo crash-rate papers | VERIFIED; journal DOI of the 56.7M-mile paper UNVERIFIABLE | 11 crash-type groups | [arXiv 2505.01515](https://arxiv.org/abs/2505.01515) |
| EV-9.6 | NAVSIM v1 / v2 | VERIFIED | DDC and TLC exist only in v2 | [arXiv 2406.15349](https://arxiv.org/abs/2406.15349) |
| EV-9.7 | CARLA Leaderboard 2.1 infractions | VERIFIED | — | [CARLA](https://leaderboard.carla.org/evaluation_v2_1/) |
| EV-9.8 / TL-51 | AlpaSim native metrics; failure taxonomy (02 G1) | VERIFIED; taxonomy PARTLY upstream | No red-light, stall or oscillation scorer; close-encounter is in AR1 but not in the challenge score | [scorers](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/eval/src/eval/scorers/__init__.py) |

### F. Alpamayo models, data and licenses

| ID | Claim as written (doc §) | Status | Correct value / note | Source |
|---|---|---|---|---|
| AM-1.1 / 1.2 | Alpamayo-R1, arXiv 2511.00088 (01 C1; 02 §7) | VERIFIED | NVIDIA, 42 authors; v1 2025-10-30, v2 2026-01-07; Table 14 in v2 is Table 13 in v1 | [arXiv 2511.00088](https://arxiv.org/abs/2511.00088) |
| AM-1.3 | "~8.2B VLM (Cosmos-Reason2)" (01 C1) | VERIFIED (card) | Base model Cosmos-Reason2-8B (8,767,123,696 parameters), itself based on Qwen3-VL-8B; the paper names only "Cosmos-Reason" | [A1.5 card](https://huggingface.co/nvidia/Alpamayo-1.5-10B/blob/7aba8293c09993f2e125c6819df05d7fa3e873ea/README.md) |
| AM-1.4 | "~2.3B flow-matching expert" (01 C1) | VERIFIED (cards); the paper says 2B | Cite the card for the release | [A1 card](https://huggingface.co/nvidia/Alpamayo-R1-10B/blob/dd4a24cacefc9a6477a6dfc7354de2443401409d/README.md) |
| AM-1.5 / 1.6 / 1.7 | Expert reads the VLM KV cache; 64 waypoints over 6.4 s at 10 Hz; (accel, curvature) unicycle | VERIFIED | Stop-gradient on the KV cache; accel bounds ±9.8, curvature bounds ±0.33, dt 0.1 | [AR1 v2](https://arxiv.org/html/2511.00088v2) |
| AM-1.8 / 1.9 | "~10 Euler steps", a ← a + 0.1·v (01 C1; 04) | VERIFIED (paper default); the paper is inconsistent | Table 14 and the decoding ablation use 5 steps; Alpamayo 1.5's code default is 10 (see DP); AlpaGym uses 5 SDE steps; FlashDrive uses 8 | [AR1 v2 §5.1, §6.6](https://arxiv.org/html/2511.00088v2) |
| AM-1.11 | 128 trajectory tokens (03 Stage 2) | VERIFIED | Used for training only; inference decodes through the flow expert | [AR1 v2 §5.1](https://arxiv.org/html/2511.00088v2) |
| AM-1.12 / 1.13 | 180 tokens per image at 320×576; 2,880 tokens; [B, 2880, 4096] (03 P2; 04) | VERIFIED (derived, for the release only); not from the paper | The paper uses 448×280, 160 tokens and two front cameras. Residual risk: if frames are packed as video (temporal patch 2), the count halves to 1,440 | [A1.5 card](https://huggingface.co/nvidia/Alpamayo-1.5-10B/blob/7aba8293c09993f2e125c6819df05d7fa3e873ea/README.md); [Qwen3-VL config](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json) |
| AM-1.15 | Tensor `egomotion [B,16,6]` (04) | CORRECTED | 16 × (3 translation + 9 rotation) floats, tokenized into 48 history tokens | [A1.5 card](https://huggingface.co/nvidia/Alpamayo-1.5-10B/blob/7aba8293c09993f2e125c6819df05d7fa3e873ea/README.md) |
| AM-1.17 | `coc_tokens [B,≤200]`; 40–200 reasoning tokens per tick (04) | UNVERIFIABLE (upper bound) | Table 14 uses 40; FlashDrive reports about 16 on average | [FlashDrive](https://arxiv.org/html/2608.12932v1) |
| AM-1.19 | "1080p wastes ~11× the pixels" (04) | VERIFIED | 11.25× | [A1.5 card](https://huggingface.co/nvidia/Alpamayo-1.5-10B/blob/7aba8293c09993f2e125c6819df05d7fa3e873ea/README.md) |
| AM-2.1–2.3 | 99 ms on an RTX PRO 6000; per-stage cells (04) | VERIFIED | Size, precision and token count are not stated | [AR1 v2 Table 14](https://arxiv.org/html/2511.00088v2) |
| AM-2.6 | "99 ms is a smaller or quantized configuration" (04) | UNVERIFIABLE (NVIDIA has not published the configuration) | FlashDrive measures the released A1.5-10B at 716.9 ms on the same GPU class | [FlashDrive Table 1](https://arxiv.org/html/2608.12932v1) |
| AM-3.1 | minADE6 0.913 m on 644 PAI samples (03 §2) | VERIFIED | Table 10 (v2 only); the 10B scores 0.849 | [AR1 v2 Table 10](https://arxiv.org/html/2511.00088v2) |
| AM-3.2 / E2E-A15 | RL: ADE 2.12 → 1.92 m (−9.4%) (03 §2) | VERIFIED (number); framing CORRECTED | Most-likely-mode ADE, not minADE6; internal CoC data; "a 0.5B AR1 fine-tuned on CoC"; no seeds | [AR1 v2 Table 9](https://arxiv.org/html/2511.00088v2) |
| AM-3.3 / 3.4 / E2E-A16 | Reasoning 3.1 → 4.5; consistency 0.62 → 0.85 | VERIFIED (numbers); framing CORRECTED | Graded by an LRM critic on internal data; the reasoning reward alone lowers consistency to 0.53 | same |
| AM-3.5 | Table 9 close-encounter (omitted in the docs) | INFO | 6.9% → 6.2% on the cited row; 3.7% with r_safety | same |
| AM-3.6–3.8 / E2E-A17 | Closed loop on 920 scenarios: 9.0% / 19% / 0.35 (03 §2) | VERIFIED | At-fault metrics; the AlpaSim score is km between events; the 10B scores 0.72 | [AR1 v2 Table 10](https://arxiv.org/html/2511.00088v2) |
| AM-3.9 / E2E-A19 | Table header "0.5B, public val" (03 §2) | CORRECTED | Only Table 10's rows use public data | same |
| AM-3.10 / E2E-A12 | Small pretrained backbone "as NVIDIA does" (03 §2) | VERIFIED (with note) | The 0.5B in the scaling ablation is DINOv2 + Qwen2.5-0.5B; Table 10's 0.5B is unspecified | same |
| AM-3.11 | "minADE6 ≤ 1.0 m on the 644-sample val" (03 Stage 2) | UNVERIFIABLE | The 644 IDs are unpublished; the repos ship 1,181 undocumented IDs | [alpamayo1.5 notebooks](https://github.com/NVlabs/alpamayo1.5/tree/36aeb4c5938cbc2eb2aed33b22434773da4ab639/notebooks) |
| AM-3.12 / AM-7.7 / E2E-A13 | "A PhysicalAI-AV subset with public CoC labels" (03 §2) | CORRECTED (scale) | 1,450 training clips / 1,728 labels; 290 validation clips / 349 labels; test held out; OOD only | [PAI-AV card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) |
| AM-4.1 | Alpamayo 1, 1.5 and 2 available (02 G3) | VERIFIED | Exactly 3 NVIDIA models on HF | [HF API](https://huggingface.co/api/models?author=nvidia&search=alpamayo) |
| AM-4.2 / 4.4 | Release dates | VERIFIED | A1 on HF 2025-12-03, renamed at CES on 2026-01-05; A1.5 on HF 2026-03-19 | [CES release](https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development) |
| AM-4.3 | A1 card numbers | INFO (old protocol) | AlpaSim 0.73 ± 0.01 (910 scenarios); minADE6 1.22 m (937) | [A1 card](https://huggingface.co/nvidia/Alpamayo-R1-10B/blob/dd4a24cacefc9a6477a6dfc7354de2443401409d/README.md) |
| AM-4.5 / 4.18 | "8.2B + 2.3B"; "GRPO update on 10.5B parameters" (04) | VERIFIED / CORRECTED (minor) | The checkpoint holds 11,078,526,194 parameters; use 11.08B in compute math | [HF API](https://huggingface.co/api/models/nvidia/Alpamayo-1.5-10B) |
| AM-4.7 | "OpenMDW-1.1" (04) | VERIFIED today; history differs | OpenMDW-1.1 since 2026-08-04; before that an NVIDIA non-commercial license; the cards still carry stale "non-commercial" text | [old license](https://huggingface.co/nvidia/Alpamayo-1.5-10B/blob/5dc9c5f26ea4bf995728df628e49966bb016bc4e/LICENSE); [new license](https://huggingface.co/nvidia/Alpamayo-1.5-10B/blob/912a0f74846149e1f2aa45630b88b44ae34573c1/LICENSE) |
| AM-4.8 / 4.10 / 4.19 | BF16 ≈ 22 GB; FP8 ≈ 11 GB; 3B student ≈ 6 GB (01 C2) | VERIFIED | 22,157,316,944 bytes | [quant recipe](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_5_quant/README.md) |
| AM-4.9 | "24 GB" (04) | CORRECTED | 24 GB is the minimum VRAM. Measured on H100: about 24 GB for 1 sample, 40 GB for 16, 60 GB for 16 with CFG | [alpamayo1.5 README](https://github.com/NVlabs/alpamayo1.5/blob/36aeb4c5938cbc2eb2aed33b22434773da4ab639/README.md) |
| AM-4.11 | ModelOpt 0.43: FP8, NVFP4, AutoQuant 6.5 bits, 22 → 9 GB; no accuracy published (04) | VERIFIED | AutoQuant's default budget is 4.8 bits | [quant recipe](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_5_quant/README.md) |
| AM-4.12 | Noise floor "AlpaSim 1.37 ± 0.10 on 913" (04) | VERIFIED today, but **unstable** | It was 0.81 ± 0.01 on 910 until 2026-08-07, with the weights unchanged; "±" is never defined | [card at 22c3322](https://huggingface.co/nvidia/Alpamayo-1.5-10B/blob/22c33224a78a3d9343b954c8259a12d36cc276c8/README.md) |
| AM-4.13 | A1.5 minADE | INFO | 0.916 m on 1,434 (new protocol); it was 1.11 m on 937 | [A1.5 card](https://huggingface.co/nvidia/Alpamayo-1.5-10B/blob/7aba8293c09993f2e125c6819df05d7fa3e873ea/README.md) |
| AM-4.14 / 4.15 / 4.16 | "1.5 Nano"; public GRPO and NVFP4 recipes; 4 policy + 1 vLLM GPU; 512 : 128 split (04) | VERIFIED | `alpamayo2_sft` also exists (2026-09-30) | [RL recipe](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_x_rl/README.md) |
| AM-5.1 / 5.2 | A2S blog dated Aug 4 2026; "34B" (01 §2; 02 §7) | VERIFIED | The 2026-06-01 press release said 32B; safetensors total 35.81B | [A2S blog](https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/) |
| AM-5.3 / E2E-B25 | "32B Cosmos 3 Reasoner + 2.3B expert" (04) | VERIFIED (card fields); sources disagree | The card prose and blog say a 2B expert; the exact size is unresolved | [A2S card](https://huggingface.co/nvidia/Alpamayo2-Super/blob/00554695e729a6ff0b6281fd2c81b18d06e33dbe/README.md) |
| AM-5.4 | "1×H100 80 GB" (04) | VERIFIED | Peak 72,115 MiB with 7 cameras, BF16, 10 steps, batch 1; the navigation-CFG demo needs 2×80 GB | same |
| AM-5.5 / 5.6 | AlpaSim 1.50; minADE6 0.911 m (04) | VERIFIED | 1.50 ± 0.13 on 913 scenes; 0.911 m on 1,434 samples. **Not comparable** with 0.913 m on 644 | same |
| AM-5.7 / 5.10 | Teacher and auto-labeler purpose; CoC auto-labels (03 P4) | VERIFIED | CoC similarity 0.652 vs 0.450 for Qwen3-VL 32B. License §4.6 may limit publishing auto-labels of PAI-AV clips | [A2S blog](https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/) |
| AM-6.1–6.4 / TL-21 / E2E-C1 | Alpamayo positioned as a teacher distilled for DRIVE AGX Thor (02 §0; 01 C2) | VERIFIED | CES (2026-01-05), Computex (2026-06-01), the A2S blog and the NVIDIA blog (2026-08-04) | [NVIDIA blog](https://blogs.nvidia.com/blog/alpamayo-2-super/) |
| AM-6.5 / TL-22 | Distillation script "planned"; no procedure (01 C2; 02 §0) | VERIFIED | The planned script targets a smaller single-GPU AlpaGym checkpoint, not a Thor student. No "distill" path exists in recipes or AlpaGym | [closed-loop blog](https://developer.nvidia.com/blog/how-to-post-train-autonomous-vehicle-models-in-closed-loop-with-nvidia-alpamayo/) |
| AM-7.1 / 7.2 / 7.5 | PAI-AV 1,700 h (03 P4) | VERIFIED | 306,152 clips × 20 s; 25 countries; 7 cameras; 133 TB; version 26.09 | [PAI-AV card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) |
| AM-7.8 | "Gated assets must be requested by Huy" (CLAUDE.md; 02 §3) | VERIFIED (gated), with **automatic approval** | Access is immediate; there is no review queue | [HF gating docs](https://huggingface.co/docs/hub/models-gated) |
| AM-7.9 | License terms (decision-relevant) | INFO | §1 internal AV development using NVIDIA technology; §3 benchmarking, regression and performance data are confidential; §4.6 no derivatives or distribution; §8 expiry 12 months after first download. NVIDIA's reading of §3 is NOT FOUND | [PAI-AV card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) |
| AM-8.1 / 8.2 / 8.3 | NuRec 26.04, ~2.9 TB (02 §3–§4) | VERIFIED | 1,607 scenes, 2.87 TB, same license; reconstruction supported only for Hyperion 8.1 | [NuRec card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) |
| AM-8.5 | NVIDIA's evaluation scene counts | INFO | 920 / 910 / 913, none equal to 1,607; the lists are unpublished | sources above |
| AM-8.7 | Instant NuRec (02 §7) | VERIFIED | About 1.5 s per scene; +2.01 dB PSNR; weights ungated; code license not checked | [arXiv 2607.14203](https://arxiv.org/abs/2607.14203) |
| AM-8.8 / 8.9 / 8.10 | OmniDreams, 68 FPS single view, gated weights (02; 04) | VERIFIED | v3 dated 2026-09-23; 4-view at 12 FPS on 1 GB300 and 105 on 16; automatic gating | [OmniDreams v3](https://arxiv.org/html/2606.03159v3) |
| AM-9.1–9.6 | Cosmos-Reason2 gating; Cosmos 3 report and sizes (04) | VERIFIED | 2B and 8B gated automatically, 32B ungated; Edge 4B, Nano 16B, Super 64B; reasoner-only sizes unknown | [Cosmos 3](https://arxiv.org/abs/2606.02800) |
| AM-10.2 | Any NVIDIA student, distilled or edge Alpamayo | NOT FOUND | Only third-party derivatives exist | [HF search](https://huggingface.co/api/models?search=alpamayo) |
| AM-10.3 | "TensorRT-Edge-LLM runs Alpamayo-R1 on Thor in FP16 only" (02 §0) | VERIFIED (example page) | See part L, item 3 | [Edge-LLM Alpamayo doc](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/95515c2f87fba8982db5a519f9022277667b3cc9/docs/source/user_guide/examples/vla/alpamayo.md) |
| AM-10.4 | "…with an unanswered OOM report" (02 §0) | NOT FOUND in GitHub issues → resolved CORRECTED by TL-35 | The forum thread was answered and the setup works | [forum 382501](https://forums.developer.nvidia.com/t/drive-agx-thor-with-alpamayo/382501) |
| AM-10.5 | "Not available for AGX Thor" (04) | UNVERIFIABLE in this lane → resolved by TL-36 | — | [forum 382409](https://forums.developer.nvidia.com/t/how-to-perform-alpamayo1-5-fp8-quantization-for-nvidia-thor/382409) |
| AM-10.6 | FlashDrive 717 → 151 ms, expert in BF16 (02 §0) | VERIFIED | — | [FlashDrive](https://arxiv.org/abs/2608.12932) |
| AM-10.7 | "Thor bandwidth 273 GB/s" (01 C2) | VERIFIED (Jetson); DP verifies DRIVE as well | — | [Jetson Thor blog](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/) |
| AM-10.9 / 10.10 | AlpaGym reward preset; first commit (04; 02 §0) | UNVERIFIABLE in this lane → resolved by TL-11 and TL-1 | — | [AlpaGym commit](https://github.com/NVlabs/alpagym/commit/92ba5cea2234e5407d0977f7b96a7c854f32e039) |

### G. Open tooling and the production stack

| ID | Claim as written (doc §) | Status | Correct value / note | Source |
|---|---|---|---|---|
| TL-1 | "AlpaGym, first commit 2026-06-16" (02 §0) | VERIFIED | Commit 92ba5ce; repository created 2026-05-19; announced 2026-05-31 | [commit](https://github.com/NVlabs/alpagym/commit/92ba5cea2234e5407d0977f7b96a7c854f32e039) |
| TL-2 | "Already runs closed-loop GRPO on the action expert" (02 §0; 01 C10) | VERIFIED (nuanced) | Expert only, with the VLM frozen and gradients stopped; the code sits on unmerged recipes PR #18 | [bundle.py](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/bundle.py); [config.py](https://github.com/NVlabs/alpamayo-recipes/blob/1d99fc50370c96637157455da386323a44663c9d/recipes/alpamayo1_x_rl/models/expert_model/config.py) |
| TL-3 | "`int_method: sde`" (02 §0) | VERIFIED | `noise_level: 0.2`, `inference_step: 5` | [policy yaml](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/configs/policy/alpamayo_r1.yaml) |
| TL-4 | "`sde_type ∈ {sde, cps}`" (02 §0) | CORRECTED | `cps` raises NotImplementedError; AlpaGym does not expose the option | [flow_matching.py](https://github.com/NVlabs/alpamayo-recipes/blob/1d99fc50370c96637157455da386323a44663c9d/recipes/alpamayo1_x_rl/diffusion/flow_matching.py) |
| TL-5 / 6 / 9 | Staleness 4; rollout prefetch; async baseline (02 §0; 04) | VERIFIED | `allowed_outdated_steps: 4`; `prefetch_rollout: true` (requires `dp_shard_size == 1`); the default smoke preset is 1 policy replica + 1 rollout replica | [default.yaml](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/default.yaml) |
| TL-7 / 8 | `perf/` module; "Prometheus merge" (02 G7) | VERIFIED (nuanced) | Opt-in; parses AlpaSim's Prometheus files rather than exporting its own | [perf README](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/perf/README.md) |
| TL-10 | "2 GPUs ≥ 40 GB" (04) | VERIFIED | Tested on 2 × RTX 6000 Ada | [ONBOARDING](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/docs/ONBOARDING.md) |
| TL-11 / E2E-B22 | Reward preset: progress 1.0, collision −10, off-road −5, distance −0.01 (04) | VERIFIED (nuanced) | That is the `progress_safety` preset; the **default** is `distance_to_gt` alone | [progress_safety.yaml](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/reward/progress_safety.yaml) |
| TL-12 | Tick subsampling undocumented (04) | VERIFIED | Code structure suggests no subsampling (inference) | [replay_objective.py](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/cosmos/replay_objective.py) |
| TL-13 | "Backprop over a 200-tick episode costs ~106 GPU-s" (04) | CORRECTED (premise) | AlpaGym's default is 8 policy steps plus 17 warmup steps; 200 steps is the challenge's evaluation contract | [default.yaml](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/default.yaml) |
| TL-14 | VLM-only RL shifts the KV cache the expert reads (04) | VERIFIED (clarified) | The hazard applies to the open-loop recipe; AlpaGym's default never updates the VLM | [RL recipe FAQ](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_x_rl/README.md) |
| TL-15 | "AlpaGym publishes no throughput" (01 C10; 02 §0) | CORRECTED (partly) | Classmethod measured 2 × L40S, 10 steps in about 32 minutes; no stage-level profile exists | [Classmethod](https://dev.classmethod.jp/articles/nvidia-alpagym-on-aws/) |
| TL-16 | Profile AlpaGym and send PRs; render-resolution PR (01 C10; 04) | CORRECTED (target) | AlpaGym has 8 PRs, all from NVIDIA, reviewed by two internal reviewers; the profiler already exists; render resolution is an AlpaSim config setting | [AlpaGym CONTRIBUTING](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/CONTRIBUTING.md) |
| TL-17 / 18 | The CLA and S-Class run end-to-end plus classical stacks under Halos (02 §0; 01 C9) | VERIFIED | The CLA is enhanced Level 2; the S-Class is L4-ready | [CLA](https://blogs.nvidia.com/blog/drive-av-software-mercedes-benz-cla/); [S-Class](https://blogs.nvidia.com/blog/mercedes-benz-l4-s-class-drive-av-platform/) |
| TL-19 | "The Uber program" uses the same architecture (02 §0) | VERIFIED (program); architecture wording UNVERIFIABLE | 100,000 vehicles from 2027; 28 cities by 2028, starting in LA and SF in H1 2027 | [Uber press release](https://nvidianews.nvidia.com/news/nvidia-uber-robotaxi); [Uber IR](https://investor.uber.com/news-events/news/press-release-details/2026/NVIDIA-to-Launch-L4-Software-Driven-Robotaxis-on-Uber-Across-28-Cities-by-2028/default.aspx) |
| TL-20 | DRIVE Hyperion 10 | VERIFIED; name superseded | 2 × DRIVE AGX Thor, 14 cameras, 9 radars, 1 lidar, 12 ultrasonics; renamed "NVIDIA Hyperion" in September 2026 | [rename note](https://nvidianews.nvidia.com/news/nvidia-drive-hyperion-becomes-the-global-platform-for-a-robotaxi-ready-world) |
| TL-27 / 28 / 29 | PTQ recipe open-loop only; ModelOpt 0.43; no accuracy or latency (02 §0; 04) | VERIFIED | `eval.py` reports minADE only | [quant README](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_5_quant/README.md) |
| TL-30 / 31 / 32 | Public GRPO recipe; 4 + 1 GPUs; 512 : 128 (04) | VERIFIED; the doc's "1.9 vs 13 samples/s" is UNVERIFIABLE | The GRPO recipe trains the VLM only, open-loop; the README calls "rollout too fast" the most common case | [RL recipe](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_x_rl/README.md) |
| TL-33 | Pins: torch 2.8.0, vLLM 0.11.0, transformers 4.57.1, ModelOpt 0.43.0 (CLAUDE.md) | VERIFIED | AlpaGym pins cosmos-rl d2a2c57; the recipe pins 747d1bd | [RL pyproject](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_x_rl/pyproject.toml) |
| TL-34 | "FP16 only" on Thor (02 §0) | CORRECTED (partly) / UNVERIFIABLE ("only") in this lane | Only Alpamayo-R1-10B is supported; Alpamayo 1.5 is absent through v0.11.0. See part L, item 3 | [supported models 0.11.0](https://nvidia.github.io/TensorRT-Edge-LLM/0.11.0/user_guide/getting_started/supported-models.html); [issue #130](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/130) |
| TL-35 / DP-T2 | "…with an unanswered OOM report" (02 §0) | CORRECTED | Thread 382501 got NVIDIA replies from 09-07 to 09-18 (a hugepages / carveout fix) and was working on 10-01; cross-post 382500 got only a community reply | [382501](https://forums.developer.nvidia.com/t/drive-agx-thor-with-alpamayo/382501); [382500](https://forums.developer.nvidia.com/t/alpamayo-r1-10b-tensorrt-engine-oom-on-drive-agx-thor/382500) |
| TL-36 / DP-T3 | "Alpamayo is not available for AGX Thor currently" (04) | VERIFIED (quote, 2026-09-07); superseded in practice | The context was the NIM container on Jetson; a community member ran A1.5 on a Jetson Thor at about 14.5 GB with AutoQuant 6.5 | [382409](https://forums.developer.nvidia.com/t/how-to-perform-alpamayo1-5-fp8-quantization-for-nvidia-thor/382409) |
| TL-37 / 38 / AM-8.6 | NuRec 100 FPS at 1080p × 6 cameras; "~25 ticks/s" (04) | VERIFIED (source); 25 ticks/s UNVERIFIABLE | The FPS definition is ambiguous: 25 or 150 ticks/s | [NuRec docs](https://docs.nvidia.com/nurec/best-practices/scaling-neural-reconstruction.html) |
| TL-39 / TL-84 | "Resizes inputs to 320×576"; 2,880 vision tokens (04) | UNVERIFIABLE (conflict) | The card says 320×576; AlpaGym's config uses `[320, 512]` (giving 2,560 tokens); AlpaSim rejects images below 320×576 | [AlpaSim CHANGELOG](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/CHANGELOG.md) |
| TL-41 / DP-W9 | "AlpaSim throughput unpublished" (04) | CORRECTED (third party) | FlashDrive reports AlpaSim per-step latency of 1,150 → 463 ms; NVIDIA publishes no benchmark | [FlashDrive](https://arxiv.org/html/2608.12932) |
| TL-42 | Render at model resolution as an upstream PR (02 G7) | CORRECTED (mechanism) | Per-camera `height`/`width` config, with a floor of 320×576 | [camera config](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/wizard/configs/cameras/2cam_2168.yaml) |
| TL-43 | "NuRec rebuilds scenes with 3DGS" (01 C4) | UNVERIFIABLE (internals) | The NuRec card says the USDZ holds a 3DGUT checkpoint; NVIDIA's GTC 2026 release mentions 3D Gaussian Splatting | [NuRec card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) |
| TL-44 | AlpaSim wires renderer, traffic, controller and policy over gRPC (01 C4) | VERIFIED | DESIGN.md is stale (it still says trafficsim is "coming soon") | [DESIGN.md](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/docs/DESIGN.md) |
| TL-49 / 50 / 52 / 53 | Paired rule; stratification; allowed claims; test–retest (02 G1) | NOT FOUND upstream | "stratif" has 0 code hits; repeats are averaged | [OPERATIONS.md](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/docs/OPERATIONS.md) |
| TL-56 | Instant NuRec as G2 reuse (02 G2) | UNVERIFIABLE (outside the lane) | — | research note `nvidia_tooling_production.md` (no primary URL) |
| TL-57 | Fidelity-vs-deviation curve and cap (02 G2) | PARTLY upstream | A fixed 4 m corridor scored as failure (#163); no calibration | [commit f012862](https://github.com/NVlabs/alpasim/commit/f012862749366a72f356fce73e22091fe5160e41) |
| TL-58 / 59 | Renderer agreement; replay-vs-CAT-K agreement (02 G2) | NOT FOUND upstream | — | [scorers](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/eval/src/eval/scorers/__init__.py) |
| TL-60 | Invalid-rollout rate (02 G2) | PARTLY upstream | Failed-rollout rows; invalid-scene error codes; `img_is_black` | [image.py](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/eval/src/eval/scorers/image.py) |
| TL-62 | Halos-analog monitor in C++ (02 G4) | NOT FOUND upstream (hook exists) | No public NVIDIA specification of TTC or drivable-area logic | [safety.py](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/eval/src/eval/scorers/safety.py) |
| TL-64 | AlpaGym as G5 reuse (02 G5) | VERIFIED (fragile) | Pinned to unmerged PR #18; the mirror has been frozen since 2026-07-08 | [PR #18](https://github.com/NVlabs/alpamayo-recipes/pull/18) |
| TL-65 / TL-68 | OPTED/RoaD ideas; FlashDrive (02 G5, G6) | UNVERIFIABLE in this lane → resolved by E2E-C2 and DP | — | [OPTED](https://arxiv.org/abs/2609.20756) |
| TL-66 | Distill to a ≤ 3B student fitting 16 GiB and ≤ 0.1 s (02 G5) | NOT FOUND upstream; NVIDIA plans it | The target is "e.g. 2B" on one GPU | [ONBOARDING](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/docs/ONBOARDING.md) |
| TL-69 | "FP8/NVFP4 including the action expert" (02 G6) | CORRECTED (not new) | FP8 is applied model-wide except `patch_embed`; AutoQuant disables `action_out_proj` and `lm_head` | [utils.py](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_5_quant/utils.py) |
| TL-70 / 71 | Closed-loop per-format evaluation; per-layer sensitivity (02 G6) | NOT FOUND / PARTLY (computed internally, not published) | — | [quantize.py](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_5_quant/quantize.py) |
| TL-72 / 73 / 74 | First public RL profile; batching PRs; "≥ 1 PR merged" (02 G7) | PARTLY open / CORRECTED (batching exists) / realistic for AlpaSim only | `max_batch_size: 4`; "batched deterministic rollouts" (06-26 sync) | [AlpaGym README](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/README.md) |
| TL-75 | "No study correlates AlpaSim or Drive-IRT scores…" (02 §0) | VERIFIED (absence, in this lane); see EV-4.2 for the Sim2Val correction | — | [HF blog](https://huggingface.co/blog/drmapavone/nvidia-alpamayo-2) |
| TL-80 / 81 | Model lines for A1.5 and A2S (04) | UNVERIFIABLE in this lane → resolved in part F | — | [recipes README](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/README.md) |
| TL-82 / E2E-B7 | "Cosmos-RL … maintenance mode" (04) | UNVERIFIABLE in this lane → **VERIFIED** by E2E | No longer under active development; limited maintenance only; commits still landing | [cosmos-rl](https://github.com/nvidia-cosmos/cosmos-rl) |
| TL-83 | Other 04 state-of-the-art items (GR00T, π0.5, DreamDojo, Newton, etc.) | UNVERIFIABLE in this lane; partly verified in DP and E2E | — | research note `nvidia_tooling_production.md` (no primary URL) |
| TL-85 / 86 | `egomotion` and `a_t` shapes (04) | PARTLY VERIFIED | See AM-1.15 | [policy yaml](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/configs/policy/alpamayo_r1.yaml) |
| TL-87 / E2E-B23 | "10 Euler steps δt = 0.1" (04) | CORRECTED (for AlpaGym) | AlpaGym rollouts use 5 SDE steps (δt = 0.2 in flow time) | same |
| TL-88 | `eps_t [10,B,64,2]` stored per step (04) | CORRECTED | The sample chain is stored, and log-probs are recomputed at train time | [inference_model.py](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/inference_model.py) |
| TL-89 | `logp_t [10,B]` fp32 (04) | CORRECTED | One log-prob per sample, [B], the mean over steps and dimensions | [flow_matching.py](https://github.com/NVlabs/alpamayo-recipes/blob/1d99fc50370c96637157455da386323a44663c9d/recipes/alpamayo1_x_rl/diffusion/flow_matching.py) |
| TL-90 | `policy_version [B]` (04) | UNVERIFIABLE | — | research note `nvidia_tooling_production.md` (no primary URL) |
| TL-91 / 92 | RLinf issue #1595; the challenge as a path (04) | VERIFIED | — | [RLinf #1595](https://github.com/RLinf/RLinf/issues/1595) |
| TL-93 | "Lucid, JLR, Uber, Berkeley DeepDrive are named Alpamayo endorsers" (04) | CORRECTED (wording) | They are "showing interest", not endorsing | [CES release](https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development) |
| TL-94 | Nebius hackathon; **GTC 2027 poster Nov 10**; arXiv/RSS/CoRL/MLSys; Inception (04) | UNVERIFIABLE | Do not schedule against these dates | research note `nvidia_tooling_production.md` (no primary URL) |
| TL-§8 | Can an outsider land upstream PRs? | VERIFIED | AlpaSim: about 11 external merges, mostly within 0–6 days, no CLA or DCO. AlpaGym: 0 external PRs. Recipes: 0 external merges, issue-first, DCO required | [AlpaSim PRs](https://github.com/NVlabs/alpasim/pulls?q=is%3Apr+is%3Amerged); [recipes CONTRIBUTING](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/CONTRIBUTING.md) |
| TL-§3 | Recipes training-efficiency docs | VERIFIED | Chapters "05 Measurement" and "06 Methodology" are stubs; the latter is announced as covering same-window paired runs, bitwise-invariance gates and dose–response ladders, scoped to SFT training performance | [overview](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/docs/training_efficiency/00_Overview.md) |
| TL-§4 | NVIDIA's distillation promises | VERIFIED | Edge-LLM blog (2026-03-12); closed-loop blog (2026-05-31); AlpaGym ONBOARDING. None shipped by 2026-10-03; none in Edge-LLM up to v0.11.0 | [Edge-LLM blog](https://developer.nvidia.com/blog/build-next-gen-physical-ai-with-edge%E2%80%91first-llms-for-autonomous-vehicles-and-robotics/); [releases](https://github.com/NVIDIA/TensorRT-Edge-LLM/releases) |

### H. Deployment, hardware, prices and compute programs

| ID | Claim as written (doc §) | Status | Correct value / note | Source |
|---|---|---|---|---|
| DP-1 | FlashDrive is arXiv 2608.12932, CoRL 2026 (02 §0) | VERIFIED; the venue appears only on the project page | Submitted 2026-08-13 | [arXiv](https://arxiv.org/abs/2608.12932); [project page](https://z-lab.ai/projects/flashdrive) |
| DP-2 | "717 → 151 ms with W4A8 + speculative decoding" (02 §0) | CORRECTED (attribution) | CUDA graphs and fusion (1.40×), streaming KV reuse, DFlash speculative reasoning and adaptive steps account for most of the gain; W4A8 removes only 24.6 ms (176.0 → 151.4) | [FlashDrive Table 1](https://arxiv.org/html/2608.12932) |
| DP-3 | Expert kept in BF16 (02 §0; 01 C10) | VERIFIED | VLM in W4A8 via ParoQuant/Marlin; about 18.3 GB | same |
| DP-4 | Closed-loop evaluation in simulation (02 §0) | VERIFIED | 100 AlpaSim clips: collision 0.19 → 0.15, off-road 0.41 → 0.32, wrong-lane 0.45 → **0.51** | [FlashDrive Table 3](https://arxiv.org/html/2608.12932) |
| DP-5 | FlashDrive hardware | VERIFIED | **Jetson** Thor: 3,770.3 → 943.6 ms at 1 sample; 14,596.5 → 1,522.6 ms at 6 samples | [FlashDrive Table 2](https://arxiv.org/html/2608.12932) |
| DP-6 / 7 | VLA-Perf; "π0 52.6 ms on Thor, 3.2 ms on B100 (measured)" (04) | VERIFIED (identity) / CORRECTED (modeled, not measured) | A roofline model; real systems reach 73.3–82.6% of roofline; it assumes Thor at 400 TFLOPS | [VLA-Perf](https://arxiv.org/html/2602.18397v1) |
| DP-8 / 9 | Expert OI ≈ 54, memory-bound; 10 → 50 steps = 2.15× (04; 01 C10) | VERIFIED (π0's expert; modeled) | Not Alpamayo's expert | same |
| DP-10 / 11 | π0.5 at 44 → 100 samples/s at B = 32; expert share 57% → 13.5% (04) | VERIFIED (qualified) | Measured on a "Hopper-series GPU". On Thor: 7.43 → 16.25 samples/s at B = 16, peaking at about 2.2× | [PhyAI](https://arxiv.org/html/2608.03682v3) |
| DP-12 | Jetson-PI, Aug 2026 (04) | CORRECTED (date) | v1 was July 2026 (2026-07-14) | [arXiv 2607.12659](https://arxiv.org/abs/2607.12659) |
| DP-13 | π0.5 458 ms naive on Thor; 7.6 Hz with async (04) | VERIFIED (qualified) | 7.59 Hz also needs graph reuse and buffering | same |
| DP-14 | CUDA graphs + fusion: 106.5 → 27.3 ms (04) | VERIFIED (qualified baseline) | 3.9× against naive PyTorch, but 1.97× against openpi/JAX | [realtime-vla](https://arxiv.org/html/2510.26742) |
| DP-15 | SnapFlow 274 → 83 ms (04) | VERIFIED | Measured on an A800 | [SnapFlow](https://arxiv.org/abs/2604.05656) |
| DP-16 | FlashVLA "up to 20×" (04) | CORRECTED | That applies to per-step action decoding; end-to-end is ≤ 2.43× | [FlashVLA](https://arxiv.org/html/2608.27384) |
| DP-17 | HoloQ-VLA 98.0% vs 97.1% (04) | VERIFIED | Reports **no latency** | [HoloQ](https://arxiv.org/abs/2605.28803) |
| DP-18 | FoldQuantVLA W4A4 1.25–1.52× (04) | CORRECTED | 1.25–1.52× is the desktop result; Orin gets 1.20–1.33×; no Thor result | [FoldQuant](https://arxiv.org/abs/2609.24433) |
| DP-19 | GR00T: 27.9 ms on H100, 80.4 ms on Thor; TensorRT 3.08× (04) | VERIFIED (qualified) | The 3.08× is H100 only; Thor gets 1.40×; BF16 only, static batch 1 | [GR00T deployment](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md) |
| DP-20 / 21 / 22 | SemiAnalysis idle times; world-action models 3–4× slower; RoboWorld r = 0.989 (04) | VERIFIED | — | [SemiAnalysis](https://newsletter.semianalysis.com/p/rl-systems-mind-the-gap-matching); [WAM blog](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/); [RoboWorld](https://arxiv.org/abs/2607.01060) |
| DP-23 | Chelsea Finn, YC podcast: 1M trajectories = 700 robot-days (04) | UNVERIFIABLE | No transcript found; the arithmetic gives 694 robot-days | research note `deployment_hardware_compute.md` (no primary URL) |
| DP-24 | DreamZero 5.7 s → 150 ms, 38× (04) | VERIFIED (qualified) | The 38× is on GB200 only, using 2-GPU CFG parallelism | [DreamZero](https://arxiv.org/abs/2602.15922) |
| DP-H1–H4 | H100 989 TFLOPS / 3.35 TB/s; H200 ridges; B200 2.25 PFLOPS / 8 TB/s (04) | VERIFIED | Dense values are half the sparse figures | [H100](https://www.nvidia.com/en-us/data-center/h100/); [HGX](https://www.nvidia.com/en-us/data-center/hgx/) |
| DP-H5 | RTX PRO 6000 504 TFLOPS BF16 (04) | CORRECTED | Not printed by NVIDIA; about 500 dense is derived | [Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/) |
| DP-H6 | RTX PRO 6000 1.79 TB/s (04) | VERIFIED (Workstation) / CORRECTED (Server: 1.597 TB/s) | Clouds list both editions | [Workstation datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf) |
| DP-H7 / H8 | Jetson Thor: 258 TFLOPS BF16; 2,070 sparse FP4; 128 GB; 273 GB/s | VERIFIED (BF16 derived) | No BF16 figure is printed; 130 W | [Jetson Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/) |
| DP-H9 / H10 | "Thor 273 GB/s"; DRIVE AGX Thor | VERIFIED (both devices) | DRIVE dev kit: 1,000 INT8 TOPS / 2,000 FP4 (sparsity unstated), 64 GB, DriveOS 7, designed for ASIL-D | [DRIVE AGX](https://developer.nvidia.com/drive/agx) |
| DP-H11 | Ridge points (04) | VERIFIED (with scope) | 947 / 1,894 is Jetson T5000; DRIVE is about 916 / 1,832 if its FP4 figure is sparse; the RTX PRO 6000 Server is about 313 / 626 | computed (deployment notes) |
| DP-W1 | Prefill ≈ 4.9e13 FLOP, 98–150 ms (04) | CORRECTED | About 5.5e13 FLOP with ViT and attention included; 110–169 ms modeled; **229–252 ms measured** | [FlashDrive](https://arxiv.org/html/2608.12932) |
| DP-W2 | Decode 9.2 ms/token × 40 tokens = 366 ms (04) | CORRECTED | 8.7 ms/token (Workstation) or 9.8 ms (Server); about 16 tokens; 171–272 ms measured | same |
| DP-W3 | Expert 2.3B × 10 steps → 26 ms (04) | CORRECTED | The bound is 27.9 ms; measured 113.9–192.9 ms for 8 steps, **4–7× slower** | same |
| DP-W4 | Total ≈ 500 ms per tick (04) | VERIFIED (total) / CORRECTED (split) | Matches the optimized 513.3 ms only by coincidence | same |
| DP-W5 / W6 / W7 | B = 32: decode 11 ms per sample; expert 0.1 ms per sample; 6–9 ticks/s per GPU (04) | CORRECTED | Decode ≈ 20.5 ms (40 tokens) or 8.2 ms (16 tokens), because of KV reads; expert ≈ **7.0 ms per sample** (compute floor); **≈ 4 ticks/s**, about 0.02 episodes/s | computed (deployment notes) |
| DP-W10 / W11 | Trainer 0.53 s/tick; vLLM ~13 samples/s (04) | VERIFIED and refined (0.40–0.47 s) / UNVERIFIABLE | — | computed |
| DP-W14 / W15 / W16 | Levers: KV reuse ≤ 1% minADE; CUDA graphs 3–4×; FP8/NVFP4 1.4–1.8× (04) | CORRECTED | KV reuse costs +2.7% minADE6 (0.770 → 0.791); CUDA graphs give 1.40× on A1.5; W4A8 gives 1.16× end-to-end (prefill 1.25×, decode 1.27×) | [FlashDrive](https://arxiv.org/html/2608.12932) |
| DP-W17 | Cross-scenario batching 2.3–2.8× (04) | VERIFIED (approximately) | π0.5 2.26×; GR00T 2.84×; Thor peaks at 2.19×; unmeasured for Alpamayo | [PhyAI](https://arxiv.org/html/2608.03682v3) |
| DP-W18 | Backprop "2–3× more than generating" (04) | CORRECTED (inconsistent) | The docs' own numbers give 3.3–4.8× | computed |
| DP-T4 | Alpamayo 1.5 in TensorRT-Edge-LLM | NOT FOUND (pending) | Promised for August; still missing on 2026-09-15; a quantized-export request (#134) is open | [issue #130](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/130) |
| DP-T5 / T6 / T7 | TensorRT and Edge-LLM precision support on Thor | VERIFIED | TensorRT 11.3 lists FP8, FP4 and BF16 for SM 11.0; Edge-LLM supports DRIVE Thor on DriveOS 7.2; pi0.5 is experimental with `--cudaGraph` | [support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html); [Edge-LLM matrix](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/getting_started/support-matrix.md) |
| DP-T9 | ModelOpt runtime on Thor | User-reported gap | Fake quantization, no real quantized GEMM, so no speed-up (users, 2026-09-15) | [forum 382409](https://forums.developer.nvidia.com/t/how-to-perform-alpamayo1-5-fp8-quantization-for-nvidia-thor/382409) |
| DP-T10 / T11 / T13 | DriveOS 7 stack; safety runtime; memory carveouts | VERIFIED (DriveOS 7 safety specifics NOT FOUND) | The DriveOS 6 safety runtime is QNX-only with kSAFETY engines; one board showed 6.0 GiB of GPU-visible memory by default | [DRIVE kit blog](https://developer.nvidia.com/blog/accelerate-autonomous-vehicle-development-with-the-nvidia-drive-agx-thor-developer-kit/); [DriveOS 6.0.10](https://developer.nvidia.com/docs/drive/drive-os/6.0.10/public/drive-os-tensorrt/release-notes/index.html) |
| DP-P1 | H100 ≈ $2.2–3.4/h (02 §4) | CORRECTED | $1.60–4.50: vast.ai median $1.69 (n = 5); RunPod $2.69 / $3.49; Lambda $3.99–4.29; Nebius $4.50 since 2026-10-01 | [vast.ai](https://cloud.vast.ai/api/v0/bundles/); [RunPod](https://www.runpod.io/pricing); [Lambda](https://lambda.ai/pricing); [Nebius](https://nebius.com/prices) |
| DP-P2 | H100 spot ≈ $0.9–2.2 (02 §4) | CORRECTED (floor) | $0.60–0.79 floor; thin availability | same |
| DP-P3 | RTX PRO 6000 ≈ $1.5–2.2 (02 §4) | VERIFIED (approximately) | Medians $1.40–1.47 (vast.ai); RunPod $1.69–2.19; Nebius $1.80 | same |
| DP-P4 | B200 ≈ $3.5–8 (02 §4) | CORRECTED | $5.98–9.39 on-demand; below that only as preemptible capacity | same |
| DP-P5 | Storage $150/month (02 §4) | VERIFIED (one provider) | RunPod above 1 TB ≈ $145; Nebius shared filesystem ≈ $216; Nebius object storage ≈ $40 | same |
| DP-P6 | Thor dev kit $5,499 (02 §4) | VERIFIED | Launched at $3,499; raised around 2026-07-21 | [Jetson FAQ](https://developer.nvidia.com/embedded/faq); [CNX](https://www.cnx-software.com/2026/07/22/nvidia-increases-the-price-of-jetson-modules-and-devkits-by-up-to-101/) |
| DP-P7 | 12-week total ≈ $7,500–8,500 (02 §4) | CORRECTED (holds only for one mix) | $3.7k to $15k by provider mix; distillation alone costs $6.4–7.2k on Lambda or Nebius | computed (deployment notes) |
| DP-C | Free or discounted compute for a solo researcher | VERIFIED (eligibility) | Inception needs incorporation; the Academic Grant needs a faculty PI; Nebius research needs an academic affiliation; Nebius startups need ≥ $5M VC; TRC offers TPUs only; ZeroGPU covers Spaces demos only; Lambda grant eligibility is UNVERIFIABLE | [Inception](https://www.nvidia.com/en-us/startups/); [Academic Grant](https://www.nvidia.com/en-us/industries/higher-education-research/academic-grant-program/); [Lambda research](https://lambda.ai/research) |

### I. Learning algorithms and driving post-training

| ID | Claim as written (doc §) | Status | Correct value / note | Source |
|---|---|---|---|---|
| E2E-A2 | Bosch ρ = −0.36 (03 P1) | VERIFIED (caveat) | p = 0.43; not significant; see EV-3.3 | [arXiv 2605.00066](https://arxiv.org/html/2605.00066) |
| E2E-A3 | 180 tokens per image (03 P2) | VERIFIED by derivation (Qwen3-VL-class encoders) | AR1's 28-px scheme would give about 231 tokens | [A2S config](https://huggingface.co/nvidia/Alpamayo2-Super/blob/main/config.json) |
| E2E-A5 / A22 | "Privileged-teacher RL (OPTED)" (03 P3, Stage 4) | CORRECTED | RL trains the render-free teacher; the camera student is supervised on-policy (closed-loop distillation). Tested on TransFuser (×1.6 driving score) and VaVAM (×9.5), with about 1,000× fewer simulator interactions; not a VLA recipe | [OPTED](https://arxiv.org/abs/2609.20756) |
| E2E-A11 | "Targeted synthetic variants (Cosmos Transfer)" (03 P4) | UNVERIFIABLE (not checked) | — | research note `e2e_driving_sota_post_training.md` (no primary URL) |
| E2E-A23 | Roofline: OI 54 vs ridge ≈ 281 (03 Stage 4) | UNVERIFIABLE in this lane → resolved by DP-8 and DP-H11 | — | [VLA-Perf](https://arxiv.org/abs/2602.18397) |
| E2E-A24 | FlashDrive as a competitor on the expert (03 Stage 4) | VERIFIED | — | [FlashDrive](https://arxiv.org/abs/2608.12932) |
| E2E-A29 / CU-A6 | RoaD listed under "RL" (03 §7) | VERIFIED (identity); categorization CORRECTED | Closed-loop *supervised* fine-tuning: +41% driving score, −54% collisions in AlpaSim | [RoaD](https://arxiv.org/abs/2512.01993) |
| E2E-B1 | "RLinf v0.3 (hybrid pipelining 1.9–2.4×)" (04) | CORRECTED | 1.61–1.88× comes from RLinf-VLA, and "up to 2.434×" is the README's v0.1 claim; no 1.9× figure exists; v0.3's headline is DreamZero SFT at nearly 4× | [RLinf-VLA](https://arxiv.org/abs/2510.06710); [RLinf v0.3](https://github.com/RLinf/RLinf/releases/tag/v0.3) |
| E2E-B2–B6 | RLinf: Flow-SDE, DSRL, RECAP, GR00T N1.7, Isaac Lab (04) | VERIFIED | — | [RLinf v0.3](https://github.com/RLinf/RLinf/releases/tag/v0.3) |
| E2E-B8 | "RL-VLA³ (+85%, no community)" (04) | VERIFIED (+85.2%) / "no community" UNVERIFIABLE | COLM 2026 | [arXiv 2602.05765](https://arxiv.org/abs/2602.05765) |
| E2E-B9 / B10 | verl rollout correction; "bounds 0.5/5.0" (04) | VERIFIED (0.5/5.0 is an option, not the default) | It is the IcePop bound; the default TIS clamp is 2.0 | [verl](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html) |
| E2E-B11 | AReaL 2.77× (04) | VERIFIED | Measured on LLM math and code RL, not on VLAs | [AReaL](https://arxiv.org/abs/2505.24298) |
| E2E-B12 | π_RL π0.5 ManiSkill 40.1 → 90.9% (04) | VERIFIED (depends on version) | The v2 abstract says 90.8%; v1 says 40.0 → 84.8% | [π_RL v3](https://arxiv.org/html/2510.25889v3) |
| E2E-B13 / B14 / B15 | ReinFlow; DSRL; DiffusionNFT (04) | VERIFIED | DiffusionNFT's results are on image generation only | [DSRL](https://arxiv.org/abs/2506.15799); [DiffusionNFT](https://arxiv.org/abs/2509.16117) |
| E2E-B16 | "RECAP (closed)" (04) | VERIFIED (method published) | Only PI's model is closed; RLinf re-implements the method | [arXiv 2511.14759](https://arxiv.org/abs/2511.14759) |
| E2E-B17 | π*0.6: 2× throughput, failures halved (04) | VERIFIED (scope-limited) | Holds for the hardest tasks only | same |
| E2E-B18 | PI ran RL on real robots "because simulators miss long-tail contact failures" (04) | Real-robot RL VERIFIED; the stated reason NOT FOUND | The paper never mentions simulation | same |
| E2E-B19 | Per-step ratios r_t exist under Flow-SDE (04) | VERIFIED (concept) | AlpaGym clips a **trajectory-level** ratio | [default.yaml](https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/conf/default.yaml) |
| E2E-B20 | GRPO with "n = 12 rollouts" (04) | CORRECTED (n = 12 NOT FOUND) | AlpaGym presets use groups of 1, 2 or 6. AR1's GRPO is softmax-weighted and unclipped, with mean-centred advantages; its K is unstated | [AR1 Eq. 10](https://arxiv.org/html/2511.00088v2) |
| E2E-B21 | Staleness dropping plus sequence-level TIS (04) | VERIFIED (the building blocks exist) | — | [verl](https://github.com/volcengine/verl/blob/main/docs/algo/rollout_corr.md) |
| E2E-B24 | E5: Flow-SDE −15% close-encounters in 200 GPU-h (04) | UNVERIFIABLE (it is a prediction; preserved as written) | Context: RoaD and OPTED find direct RL sample-hungry | [RoaD](https://arxiv.org/abs/2512.01993) |
| E2E-D2 | "≥ 5% ADE gain from RL over 3 seeds" (nano README) | CORRECTED (basis) | NVIDIA's −9.4% is open-loop, on internal data, most-likely-mode ADE, from a single run | [AR1 Table 9](https://arxiv.org/html/2511.00088v2) |
| E2E-D3 | "Closed-loop within 20% relative on the gate" (nano README) | VERIFIED (references exist); needs a re-baselined anchor | The gate's 441 scenes differ from the 920 behind the reference numbers | [AR1 Table 10](https://arxiv.org/html/2511.00088v2) |
| E2E-E2 | RAD (Horizon Robotics) | VERIFIED | NeurIPS 2025; 3× lower collision rate than imitation learning | [RAD](https://arxiv.org/abs/2502.13144) |
| E2E-KQ2 / KQ7 | VLA and planner landscape; small-backbone licenses; SFT and GRPO GPU-hours | UNVERIFIABLE (the delegated survey did not return) | No dollar estimate for nanoAlpamayo can be traced to the notes | research note `e2e_driving_sota_post_training.md` (no primary URL) |

### J. Curriculum shelf and learning evidence

| ID | Claim as written (doc §) | Status | Correct value / note | Source |
|---|---|---|---|---|
| CU-shelf | Every arXiv ID on 03 §7 and 01 §3 | VERIFIED (13 of 13, plus EUVS) | The classical texts (LaValle, Thrun et al., Sutton & Barto, DAgger) are now verified as well | [AR1](https://arxiv.org/abs/2511.00088); [LaValle](https://lavalle.pl/planning/); [Sutton & Barto](http://incompleteideas.net/book/the-book-2nd.html); [DAgger](https://proceedings.mlr.press/v15/ross11a.html) |
| CU-A14 | "EUVS novel-view limits (ICCV 2025)" with no ID | CORRECTED (ID added) | arXiv 2412.05256 | [EUVS project](https://ai4ce.github.io/EUVS-Benchmark/) |
| CU-B30 | "A standard text such as Embretson & Reise" | CORRECTED (edition) | 2nd edition, Routledge, 2025-10-17, with a new title | [doi](https://doi.org/10.4324/9781315726557) |
| CU-B37 | OCP Microscaling specification | UNVERIFIABLE (not accessed) | Parameters taken from Table 1 of the MX paper | [MX paper](https://arxiv.org/abs/2310.10537) |
| CU-B44 | "PMPP, latest edition" | CORRECTED (edition) | 5th edition, Hwu, Kirk & El Hajj, 2026 | [Elsevier](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-443-43900-1) |
| CU-D5 / D8 | CMU 16-831; the year of CMU 16-745 | UNVERIFIABLE | — | [16-745](https://optimalcontrol.ri.cmu.edu/) |
| CU-P1 | "Euler discretization" in AR1 Eq. 5 | CORRECTED (interpretation, computed) | Heading is integrated exactly and position by the trapezoid rule. Forward Euler adds 0.60 m of endpoint error over 6.4 s | [AR1](https://arxiv.org/html/2511.00088) |
| CU-P1-text | Kong et al. 2015 and Werling et al. 2010 | Bibliographic records only; equations not re-read | — | [Kong](https://doi.org/10.1109/IVS.2015.7225830) |
| CU-P2 | 2,880 tokens assumes per-frame image tokenization | UNVERIFIED in the A1.5 code | Video mode with temporal patch 2 would give 1,440 | [Qwen3-VL config](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/raw/main/config.json) |
| CU-P2-config | The Cosmos-Reason2-8B config (gated) | An assumption: the Qwen3-VL-8B text config is used in its place | — | [cosmos-reason2 README](https://raw.githubusercontent.com/nvidia-cosmos/cosmos-reason2/main/README.md) |
| CU-P3 | Stage 0 acceptance: "toy flow within 2% MSE of the analytic OT path" (03 §4) | Ambiguous (inference) | 10 Euler steps alone consume 1.92%. Name the coupling, the step count, and whether the path is conditional or marginal | [Flow Matching](https://arxiv.org/html/2210.02747) |
| CU-learning | Learning-science basis for the mastery gates | VERIFIED | Testing effect; Dunlosky's "high utility" rating; generation effect d ≈ 0.40; Shen & Tamkin (17%, d = 0.738); Bastani 2025. Karpicke & Blunt is contested | [Shen & Tamkin](https://arxiv.org/abs/2601.20245); [Dunlosky](https://pubmed.ncbi.nlm.nih.gov/26173288/); [Bertsch](https://pubmed.ncbi.nlm.nih.gov/17645161/) |

### K. Roles and skill demand

| ID | Claim as written (doc §) | Status | Correct value / note | Source |
|---|---|---|---|---|
| RS-A1–A8 | Role A is live; AV Simulation team; Santa Clara; L5 / L6 pay; equity; closed-loop RL mission (05) | VERIFIED | JR2010863, Workday start 2026-08-29. Exact pay: L5 224,000–356,500 USD; L6 272,000–431,250 USD. The posting names Alpamayo R1 | [JR2010863](https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Senior-Machine-Learning-and-Simulation-Engineer---Autonomous-Vehicles_JR2010863) |
| RS-A9 | Source is a job-board mirror (05) | VERIFIED (secondary; stale) | The primary Workday source is now available | [mirror](https://www.dreamworkhq.com/job/90345157-b49f-4ea3-a830-0d2e1e6dcf69) |
| RS-A12 | Duty ③ "Improve sim quality & reward functions" (05) | CORRECTED (wording) | The primary text includes *measuring* simulation quality | [JR2010863](https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Senior-Machine-Learning-and-Simulation-Engineer---Autonomous-Vehicles_JR2010863) |
| RS-A15 / A29 | Mapping of duties to G1–G8; the Fit column (05) | UNVERIFIABLE (internal self-assessment) | — | research note `role_skill_demand.md` (no primary URL) |
| RS-A16–A28 | Requirements R1–R9 and preferences P1–P4 (05) | VERIFIED | PPO and GRPO; reward design; C++ and Python; Kubernetes and SLURM; 12+ years | same |
| RS-A30 / A31 / A32 | Requirement list complete; stack and seniority signals (05) | CORRECTED (omissions) | Add the BS-or-equivalent line; NuRec, Traffic Models and Cosmos World Model; technical leadership and a productization track record | same |
| RS-B1 / B2 / B3 | Role B closed Oct 2025; title; $272k–425.5k (05) | VERIFIED (secondary) | Listed on builtin 2025-10-03 → 10-10. That was the 2025 band; the 2026 L6 band is 272,000–431,250 | [builtin 6878054](https://builtin.com/job/deep-learning-engineer-end-end-autonomous-driving/6878054) |
| RS-B8 | Implied: no live version of Role B (05) | CORRECTED | The same text is live as a Senior role, JR2019758 (Shanghai and Beijing), asking PhD + 2 years or MS + 4 years | [JR2019758](https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/China-Shanghai/Deep-Learning-Senior-Engineer--End-To-End-Autonomous-Driving_JR2019758) |
| RS-B9 | Role B's requisition number | NOT FOUND | — | research note `role_skill_demand.md` (no primary URL) |
| RS-B10 | A live US Principal end-to-end posting | NOT FOUND | NVIDIA's US end-to-end and VLA hiring sits at Senior L3–L6 | [NVIDIA Workday](https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite) |
| RS-B11 | Mirror listings titled "Principal Deep Learning Senior Engineer…" | UNVERIFIABLE (HTTP 410) | — | [jobgether](https://jobgether.com/offer/69cd8e955bd3b40540a53f3b-principal-deep-learning-senior-engineer-end-to-end-autonomous-driving) |
| RS-B12 | Role B credential lines: "12 years, PhD" (03 header) | VERIFIED (secondary) | A PhD is not mandatory; "or equivalent experience" is accepted | [builtin 6878054](https://builtin.com/job/deep-learning-engineer-end-end-autonomous-driving/6878054) |
| RS-B13 / C6 / H1 / H2 | Duty-to-deliverable mappings; matching artifacts; hiring-manager checks; "apply after week 7" (05) | UNVERIFIABLE (internal or opinion) | — | research note `role_skill_demand.md` (no primary URL) |
| RS-C1 | Role C "closed Apr 2026" (05) | CORRECTED | Live again as JR2005596 (Workday start 2026-08-05); the builtin record ran only 2026-04-22 → 04-23 | [JR2005596](https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Senior-Research-Scientist---Autonomous-Vehicles_JR2005596) |
| RS-C4 | Role C tagged "(Alpamayo, OPTED)" (05) | NOT FOUND | Neither term appears in the posting | same |
| RS-C5 | "Needs publications" (05) | VERIFIED | Also accepts other research artifacts, such as software projects | same |
| RS-C8 / C9 | Role C pay and requirements omitted (05) | CORRECTED (omissions) | L4 192,000–304,750; L5 224,000–356,500 USD; PhD or equivalent; 6+ years; Python, C++, CUDA | same |
| RS-03§0 | "AI 1.0 → AI 2.0 is NVIDIA's framing"; "NVIDIA chips in production vehicles" (03 §0) | VERIFIED | Matches the live posting JR2019758 | [JR2019758](https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/China-Shanghai/Deep-Learning-Senior-Engineer--End-To-End-Autonomous-Driving_JR2019758) |
| RS-Q3 | What "Principal" means at NVIDIA | VERIFIED (bands); the level mapping is inference | Principal titles carry the L6 band (272,000–431,250 USD) and 12–15+ years; Senior titles span L3–L6 | [JR2017016](https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Principal-High-Performance-LLM-Training-Engineer_JR2017016) |
| RS-Q4 | Skill-cluster frequencies | VERIFIED (computed, ±5–10 points) | See §8 | [NVIDIA Workday](https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite) |
| RS-Q5 | Evidence asks: years, degrees, publications, open source, competitions | VERIFIED (computed) | 13 of 43 NVIDIA postings require 12+ years; none of 212 competitor postings does. A PhD is never mandatory without an alternative at NVIDIA. Competition wins appear only in JR2019758 | same |
| RS-Q6 | Alpamayo 2 Super parameter count | Conflicting sources | "3× the 10B models" (blog), 34B (card), 32B (press release and a secondary wiki); treat as about 30B-class | [NVIDIA blog](https://blogs.nvidia.com/blog/alpamayo-2-super-open-model-now-available/) |

### L. Cross-note conflicts and how they were resolved

| # | Conflict | Resolution | Basis |
|---|---|---|---|
| 1 | Bosch ρ = −0.36: n = 8 (curriculum notes, P1 trap) vs n = 7 (evaluation notes) | **n = 7.** The evaluation notes recomputed from the paper's Table 1, which lists 7 methods, and got ρ = −0.143 | [arXiv 2605.00066v1](https://arxiv.org/html/2605.00066v1) |
| 2 | Drive-IRT "95%": UNVERIFIABLE (tooling notes) vs VERIFIED (evaluation notes) | **VERIFIED.** Default 0.95, computed from the mean-field posterior std | [evaluate.py](https://github.com/NVlabs/alpasim/blob/e2e_challenge/e2e_challenge/local_evaluation/evaluate.py) |
| 3 | Edge-LLM "FP16 only": partly corrected / UNVERIFIABLE (tooling notes, which read the 0.11.0 supported-models page; the example page returned 404) vs VERIFIED (Alpamayo and deployment notes, which read the example page at commit 95515c2 and on `main`) | **VERIFIED** that the Alpamayo example page states FP16-only export; only Alpamayo-R1-10B is supported; Alpamayo 1.5 is absent through v0.11.0 | [Edge-LLM Alpamayo doc](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/95515c2f87fba8982db5a519f9022277667b3cc9/docs/source/user_guide/examples/vla/alpamayo.md) |
| 4 | The "unanswered OOM report": NOT FOUND (Alpamayo notes, GitHub issues) vs forum threads (tooling and deployment notes) | **CORRECTED.** Answered by NVIDIA in 382501 and working on 2026-10-01 | [382501](https://forums.developer.nvidia.com/t/drive-agx-thor-with-alpamayo/382501) |
| 5 | "Not available for AGX Thor": UNVERIFIABLE (Alpamayo notes) vs VERIFIED (tooling and deployment notes) | **VERIFIED quote**, in a NIM-container / Jetson context; superseded in practice | [382409](https://forums.developer.nvidia.com/t/how-to-perform-alpamayo1-5-fp8-quantization-for-nvidia-thor/382409) |
| 6 | Cosmos-RL "maintenance mode": UNVERIFIABLE (tooling notes) vs VERIFIED (e2e notes) | **VERIFIED** | [cosmos-rl](https://github.com/nvidia-cosmos/cosmos-rl) |
| 7 | Alpamayo 1.5's default flow steps: "not stated" (Alpamayo notes) vs a code default of 10 (deployment notes) | **10 Euler steps in code.** AlpaGym uses 5 SDE steps, FlashDrive 8, AR1 Table 14 5 | [alpamayo1.5 flow_matching.py](https://github.com/NVlabs/alpamayo1.5/blob/main/src/alpamayo1_5/diffusion/flow_matching.py) |
| 8 | Seed location: `DriveSessionRequest` (evaluation notes) vs `RolloutSpec` (tooling notes) | **Both, by branch:** `DriveSessionRequest` on the challenge branch, `RolloutSpec` on `main` | [runtime.proto](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/grpc/alpasim_grpc/v0/runtime.proto) |
| 9 | Input size: 320×576 (card) vs `[320, 512]` (AlpaGym config) | **Unresolved.** Pin the configuration per run; token counts differ (2,880 vs 2,560) | [policy yaml](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/configs/policy/alpamayo_r1.yaml) |
| 10 | Challenge split of 2,202 scenes vs 1,607 public NuRec 26.04 scenes (Alpamayo notes flag) | **Resolved.** The split draws on 26.01 and 26.04: 2,202 of 2,523 public scenes | [local_evaluation README](https://github.com/NVlabs/alpasim/blob/e2e_challenge/e2e_challenge/local_evaluation/README.md) |
| 11 | EUVS 24.6% / 30.6%: not located (curriculum notes keyword pass) vs VERIFIED (evaluation notes, primary Table 1) | **VERIFIED** | [EUVS (CVF)](https://openaccess.thecvf.com/content/ICCV2025/papers/Han_Extrapolated_Urban_View_Synthesis_Benchmark_ICCV_2025_paper.pdf) |
| 12 | Thor at 273 GB/s: Jetson only (Alpamayo notes) vs both Thors (deployment notes) | **VERIFIED for both** Jetson AGX Thor and DRIVE AGX Thor | [DRIVE AGX](https://developer.nvidia.com/drive/agx) |
| 13 | AlpaGym "first commit 2026-06-16" vs repository created 2026-05-19 | **Both are true.** First public commit 2026-06-16; repository created 2026-05-19 | [commit](https://github.com/NVlabs/alpagym/commit/92ba5cea2234e5407d0977f7b96a7c854f32e039) |
| 14 | Alpamayo 2 Super expert: 2.3B (card field) vs 2B (card prose, blog, e2e notes) | **Unresolved.** Cite both | [A2S card](https://huggingface.co/nvidia/Alpamayo2-Super/blob/00554695e729a6ff0b6281fd2c81b18d06e33dbe/README.md) |
| 15 | AlpaSim throughput "unpublished": VERIFIED (tooling notes) vs CORRECTED (deployment notes) | **NVIDIA publishes none**; a third party (FlashDrive) reports per-step latency | [FlashDrive](https://arxiv.org/html/2608.12932) |
