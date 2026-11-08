# Pricing the S4 improvement claim: strata and guard margins (2026-11-08)

Agent-authored for Dang Huy. Serves S2.6 [C] (operational power) for the S4/G9 improvement question.

**This prices options; it registers nothing.** Strata, margins, δ_min, gating and the claim design remain Huy's.

**Inputs are illustrative, not measurements:**
- at most 441 independent roads, which is an upper bound;
- a hypothetical 8% teacher failure rate per road;
- one scene and one binary outcome per road, unless stated;
- α = 0.05.

**b** is the share of roads where the candidate is worse: true regressions plus noise discordance. **Δ** is the net improvement, P(better) − P(worse).

**Method.**
- `gate.superiority_power.design_power` scores every design on the same simulated rows in each trial, through the locked gate and the candidate rule (`gate/superiority.py`).
- Exact values enumerate the improved and worsened road counts through the locked bound.

**Evidence.**
- `evidence/strata-pricing-20261108-v1/`: first grid, with `CORRECTIONS.txt`.
- `evidence/strata-pricing-20261108-v2/`: joint-guard and gating re-pricing after independent review.

## Four findings

### 1. Do not register "improved in every stratum"

Every stratum must clear its own bound with only its own roads, so power collapses (exact).

| Net improvement Δ, b = 0 | 1 stratum (441) | 2 strata (220/221) | 4 strata (110 each) | 300/100/41 |
|---|---|---|---|---|
| 3 pp | 0.853 | 0.017 | 0.000 | 0.000 |
| 5 pp | 0.999 | 0.446 | 0.000 | 0.000 |

### 2. Claim pooled improvement, with per-stratum non-inferiority guards

**The design:** improvement on the pooled population, plus non-inferiority of the same endpoint in each guarded stratum.
- It is a valid intersection-union claim at α.
- The pooled estimand is the equal-road average over the registered roads, so strata weigh by their road count (the 300-road stratum is 68% of 300/100/41).
- The pooled claim keeps the single-stratum power: about 0.85 at Δ = 3 pp, b = 0, in every configuration.

Without a guard, the pooled claim can pass while a stratum is harmed (finding 3).

### 3. The guard must pass in every gating stratum at once

**Correction.** An earlier version of this note gave per-stratum pass rates as if they were the guard's. The guard needs every gating stratum GO together. With independent strata, the joint rate is the product of the per-stratum rates.

**Joint pass rate with no effect anywhere** (exact, guard margin m):

| Gating strata | m = 0.05, b = 2% | m = 0.10, b = 0 | m = 0.10, b = 2% | m = 0.10, b = 5% |
|---|---|---|---|---|
| 1 × 441 | 0.904 | 1.000 | 1.000 | 0.998 |
| 2 × 220/221 | 0.208 | 1.000 | 0.997 | 0.676 |
| 4 × 110 | 0.000 | 1.000 | 0.495 | 0.017 |
| 300/100/41 | 0.000 | 1.000 | 0.343 | 0.036 |
| 300/100 (41 reported) | 0.096 | 1.000 | 0.785 | 0.293 |

**Sizing.** For a joint pass rate of at least 0.8 with S gating strata, each must pass with at least 0.8^(1/S): about 0.89 for two strata and 0.95 for four. At b = 2% and m = 0.10 that is about 120 and 150 roads per stratum. The exact per-stratum rates are:

| Roads | 110 | 120 | 125 | 135 | 140 | 150 |
|---|---|---|---|---|---|---|
| Pass rate | 0.840 | 0.908 | 0.900 | 0.947 | 0.943 | 0.970 |

The rate is not monotone in size because the counts are discrete.

**Margin 0.05** works only when discordance is near zero or the strata are very large.

**What the guard costs the claim when an effect exists** (simulation, 1,000 trials, m = 0.10, values in brackets at m = 0.05). Improving strata pass far more easily than flat ones, so the cost sits in flat or small strata.

| Scenario | Pooled only | Guard every stratum | Guard 300/100 only, 41 reported |
|---|---|---|---|
| 4 × 110, all +5 pp, b = 2% | 0.468 | 0.445 [0.031] | — |
| 4 × 110, one stratum flat, b = 2% | 0.193 | 0.179 [0.008] | — |
| 300/100/41, all +5 pp, b = 2% | 0.451 | 0.239 [0.013] | 0.437 [0.247] |
| 300/100/41, rare flat, b = 2% | 0.356 | 0.183 [0.000] | 0.347 [0.208] |
| 300/100/41, all +5 pp, b = 0 | 0.999 | 0.999 [0.019] | 0.999 [0.999] |
| 300/100/41, **rare harmed +10 pp**, b = 0 | 0.333 | **0.016** [0.000] | 0.333 [0.333] |

**The trade-off for a small stratum, which is Huy's decision:**
- **Guard it.** The claim is protected against harm of at least the margin in that stratum (0.333 → 0.016), at about half the power when b = 2%.
- **Report it.** Power is kept (0.437), but harm in that stratum is only visible in the descriptive report. The gate's regression flag does not catch a 10 pp harm on 41 roads.
- **Enlarge it.** More independent roads in that stratum reduce the cost.

### 4. The decisive unknown is discordance, and replicates do not fix it

**Pooled power at Δ = 3 pp (exact):**

| b | Power |
|---|---|
| 0 | 0.853 |
| 2% | 0.089 |
| 5% | 0.025 |

**Replicates do not lower the detectable effect.** The locked bound depends on the road-level means of the positive and negative differences, not on their variance. Three replicates shrink the spread of the upper bound while leaving its median unchanged:

| Δ | b | Median upper bound | Power, 1 replicate | Power, 3 replicates |
|---|---|---|---|---|
| 3 pp | 0 | −0.004 | 0.867 | 0.973 |
| 2 pp | 0 | +0.001 | 0.437 | 0.323 |
| 3 pp | 2% | about +0.015 | 0.097 | 0.020 |

So replicates help only when the median is already below zero. This explains P-032's finding that replicates are a weak lever.

**The strong lever is common random numbers.** Running both arms on the same seed removes noise discordance, so it lowers b itself. It needs the seeded driver path that the `e2e_challenge` driver lacks (P-032).

## Run aggregation

With "all three runs must pass", the chance that all pass lies between p³ (if run outcomes were independent) and p (if they were identical). Two examples:
- at p = 0.85, it lies between 0.61 and 0.85;
- at p = 0.09, it lies between 0.0007 and 0.09.

## Recommended registration (Huy decides)

1. **Primary claim:** improvement of the primary endpoint on the pooled registered population, decided by the candidate rule at one fixed look. Register one design and one margin before any data; choosing among the priced designs after seeing results would be selection.
2. **Guards:** per-stratum non-inferiority of the same endpoint at margin 0.10, plus the other guard endpoints priced the same way.
   - Gate only strata whose size fits the measured b by the joint sizing above.
   - For each smaller stratum, choose: guard it and accept the power cost, report it descriptively (with its ≥4 m and harm rates), or add roads to it.
3. **Before S4:** S2 measures b on the registered path with and without seed coupling. If coupling cannot be qualified and b stays near 2% or more, the expected S4 outcome is INSUFFICIENT at any plausible effect. That should be known before training is bought.

## Not established

- **Model simplifications:** one binary outcome per road; strata and roads drawn independently. The model cannot price dependence between roads, an assumption whose failure would cost the test's validity, not only its power.
- **Assumed values:** the 8% base rate and the 441-road count.
- **Not modelled:** other guard endpoints, so every rate is an upper bound for a claim that adds them; sequential analysis (all results are fixed-sample).
- **Monte Carlo precision:** cells use 1,000 trials, roughly ±0.03 near 0.5. They agree with exact values to about two standard errors; the largest gap is 0.109 simulated against 0.089 exact.
- **Scope:** none of this is an AlpaSim measurement.
