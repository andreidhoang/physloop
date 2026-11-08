# 2026-10-31 — s1-synthetic-truth

**Serves:** stage S1 · deliverable G1 (`docs/02_PHYSLOOP_PLAN.md` §3)
**Owner of the prediction:** Huy
**Curriculum exercise done** (`docs/03_MASTERY_PLAN.md` §4): yes / no

## Hypothesis (one sentence)
`verdict()` holds false-GO ≤ α on synthetic truth, at fixed n and across interim looks, and returns INSUFFICIENT, never GO, where 441 scenes lack the power to decide.

## Mechanism (why we expect it; derived, with numbers, L5 → L1)
From `02` S1 and `research/2026-10-03-verification/report.md` §3 S1:
- **L5 symptom.** On 441 paired scenes at ψ = 4%, 80% power detects only about 2.72 pp (two-sided 0.05; exact McNemar power averaged over D ~ Bin(441, ψ)). The 12-vs-6 example has power 0.22.
- **L4 system.** A −30% effect on a 2% stratum at 7% base failure is 2.1 pp on about 9 scenes (0.02 × 441 = 8.82). It dilutes to 0.042 pp pooled, so it hides inside a passing aggregate. That stratum has at most 0.2% power, and zero failures in 9 still leaves a 28.3% one-sided upper bound. A gate that demands per-stratum evidence can stay honest there only by returning INSUFFICIENT.
- **L3 math.** Intersection–union across road-disjoint strata needs no multiplicity adjustment, but overall GO power is the product of per-stratum powers: four strata at 0.80 give 0.4096. Exact McNemar assumes independent pairs, so road clustering makes it anti-conservative. Repeated looks inflate false-GO unless a spending function pays for them.
- **Huy's hand derivations (`02` §4 Day 1).**
  - Exact McNemar null: ____
  - Zero-failure bound: ____

## Falsifiable prediction (written BEFORE the run)
Fill these before the first Monte Carlo run. The scaffold and the known-answer tests can exist without them; a run cannot.
- Metric and predicted value (`02` S1):
  - P-S1-1, false-GO for a −30% effect on a 2% stratum at 7% base failure: ____
  - P-S1-2, share of INSUFFICIENT under the null: ____
  - P-S1-3, change in false-GO without stratification: ____
  - P-S1-4, the base at-fault rate at which a 20% relative cut becomes decidable on 441 scenes × 3 replicates: ____
  - P-S1-5, number of S3 strata that return INSUFFICIENT under the planned allocation: ____
- Result that would make us abandon this: ____

## Pre-registered thresholds (if this experiment feeds a kill check)
- Feeds KC-1. False-GO ≤ α is fixed by `02` §7. The choices below are ★ and still open:
  - margin Δ used in the synthetic runs: ____
  - spending function: ____
  - K: ____
  - futility rule: ____
  - stage-wise-ordering bounds after early stopping: ____
  - allocation objective across strata: ____
- Decision rule (PASS / KILL / INSUFFICIENT): KC-1, below.

## Scope: what this experiment is NOT testing
- AlpaSim, any policy, any checkpoint, or any real scene. Everything is synthetic.
- Real base rates, discordance, σ_d or road-cluster sizes. S2 measures those; here they are generator inputs.
- Replicate reliability on real rollouts, validity envelopes or out-of-envelope rollouts (S2).
- Whether the gate's verdict means anything on a real road (`02` §6).

## Setup
- Scenes / strata / replicates: synthetic generator with known effects, road clusters and strata, built by agents to Huy's specification (`02` §4 Day 3). Generator specification: ____
- Model / checkpoint and precision: none.
- AlpaSim, renderer, traffic, deviation convention, score definition: none. Outputs will use the shape of AlpaSim's per-rollout results keyed by scene ID.
- Run method and topology: local CPU.
- Contamination: not applicable (no dataset).
- Interim looks: as pre-registered above.
- Hardware: CPU only. GPU-hours 0, cost $0.
- Environment: Python 3.12, `uv.lock`.
- Command(s):
  - `uv run pytest -m "not star"` for plumbing;
  - `uv run pytest -m star` for the known answers on Huy's cores;
  - Monte Carlo command: added with the generator (`02` §4 Day 4).

## Gates
- Budget tier (P-005): not needed, the $0 tier. Hardware (P-002, P-006): none. Spend stays within the stage line: yes.
- Public number derived from PAI-AV or NuRec: no. The results are synthetic, so KC-L does not apply.
- Weights or labels trained on PAI-AV/NuRec-derived data: none. KC-L2 does not apply.

## Kill criterion (wall-clock or cost past which we stop regardless)
- **KC-1, due 2026-11-02.** If `verdict()` cannot hold false-GO ≤ α on synthetic truth, sequentially as well as at fixed n, fix the statistics before any rented GPU-hour.
- No interim look is used in S3–S5 until the sequential Monte Carlo holds false-GO ≤ α.
- If S1 runs past twice its estimate (about 24 h, so about 48 h), ship the power-analysis half alone.

## Metric-to-decision rule
- If false-GO ≤ α holds at fixed n and sequentially, the known-answer tests pass and the S3/S4 power tables exist, then KC-1 passes and S2 may be pre-registered; otherwise fix the statistics first.
- At KC-1, re-forecast the windows from S1's actual hours (P-003) and decide which [O] items run.

## Result (filled AFTER the run)
- Measured:
- Prediction vs measurement, and the causal explanation of any gap:
- Statistics used (paired? cluster-adjusted? one- or two-sided?):
- Evidence paths (`evidence/...`), git hash, config, manifest:

## Decision taken
