# S1 engineering handoff

S1/G1's autonomous software qualification passed on 2026-10-31 under D-016. Cloud cost was $0. This is an agent-authored implementation and diagnosis, not Huy's independent mastery record. No commit, publication, upstream message, dataset download or rented run was performed.

## What shipped locally

- Strict versioned input admission, identity pairing, explicit missing strata, separate roads/scenes/replicates, and >=4 m accounting.
- Paired bounded-road GO/NO-GO/INSUFFICIENT with an explicit equal-road estimand and Holm regression flags.
- A replayable finite-look state machine with efficacy closure, configured futility, fixed error spending, duplicate/reused-road rejection and immutable saved checkpoint copies.
- A seeded clustered generator, preregistered qualification runner, append-only evidence capture, CLI, nine power/reference table families, CSVs and an SVG outcome plot.
- All original known answers, Tango reference intervals, road-bootstrap diagnostics, and K=2/3 canonical Gaussian O'Brien-Fleming/Pocock/Lan-DeMets planning tables.

The operational decision method deliberately differs from the older plan: its authorization uses bounded-road Chernoff limits and finite-look union-bound spending. Tango/bootstrap and Gaussian planning are not substituted for this guarantee. Spent-alpha bounds are not stage-wise-ordering intervals. D-016 records this amendment. The statistical target is equal-road, not silently scene-weighted. A small detected positive regression can trigger NO-GO even inside the NI margin.

## Qualification results

Primary evidence: `../evidence/s1-20261031-autonomous-v1/`. There are 10,000 generated trials, each evaluated at fixed n and sequentially, plus one pooled ablation: 21,000 decisions. Dependence between those decisions is preserved; they are not described as 21,000 independent experiments.

The simultaneous exact-binomial audit covers 20 registered calibration cells. The worst observed false-GO frequency is 3/1,000 in the independent one-bad-stratum fixed design; its one-sided upper bound is 0.01183453445, below alpha=0.05. The sequential version has 2/1,000 and upper bound 0.01008366331. These bounds have simultaneous 95% coverage under the trial-generation assumptions; zero observed errors are not reported as zero error probability.

Identical-policy controls returned GO in all 1,000 trials for both schedules. Gross-regression controls returned NO-GO in 1,000 fixed and 999 sequential trials. Rare-stratum harm returned INSUFFICIENT in every trial; removing stratification returned pooled GO in 999/1,000, illustrating a different and inadequate target for the all-strata claim.

Noisy equivalent policies returned INSUFFICIENT in 1,000 fixed and 999 sequential trials. This is a consequential power limitation, not evidence of policy regression. S2 must measure variability and dependence before the operational gate is used to price a real-model comparison. Normal planning tables do not override measured operational power.

All preregistered numeric prediction thresholds were met. The P-A4 planning baselines were 0.11840356 for one replicate and 0.10139476 under the illustrative three-replicate/rho=.6 calculation, matching the pre-run approximations. These are synthetic planning assumptions, not measured AlpaSim base rates or ICC.

## Independent checks and diagnosis

498 tests passed locally and in an isolated checkout reconstructed from the evidence source archive with the locked environment. Wheel and source distribution builds succeeded. Exact small-support tests sum binomial and multinomial probabilities rather than estimating them by Monte Carlo. The proof is conditional on the sampling contract; no formal verification of all floating-point executions is claimed.

An independent audit verified every artifact checksum, all 28 archived source hashes, the config byte hash, all 10,000 trial seeds, raw counts, truth labels and simultaneous binomial bounds. Independent reviewers found and fixed tiny-probability root bracketing, cross-look scene reuse, checkpoint aliasing and shared data/order random streams before qualification.

The five false-GO decisions across three NI-boundary trials were reconstructed and their bounds independently inverted. They are permitted false positives within the registered error budget, not erased results. Equivalent trial 194 closes the highway stratum at look 2 with R=148, zero positive and eight negative differences, upper=0.01826325. The fixed analysis includes later roads and has upper=0.03216324. The sequential GO therefore uses different, legitimately frozen evidence. Gross-regression trial 367 has final p-values 0.0181499 and 0.0267851: fixed Holm rejects, while the smaller final sequential budget does not. Reduced sequential regression sensitivity is the consequence of spending that budget across looks.

The three fixed false-GOs were unusually frequent for that particular scenario. Independent enumeration of all 24,531 multinomial count states gives fixed false-GO probability 0.000267346, or 0.267 expected per 1,000 trials; the chance of at least three is approximately 0.00260331. All corresponding generated outcomes and distinct seeds were checked; aggregate discordances matched the registered marginals and no defect was identified. This post hoc diagnostic is retained, not used to change the seed or acceptance rule.

The independent verification record is `../evidence/s1-20261031-verification-v2/verification.json`. A complete second 10,000-trial run from the archived code in an isolated locked checkout produced bitwise-identical raw trials, configuration, power tables and SVG output. The installed wheel CLI passed using Python isolated mode. The first packaging verification attempt is preserved as an incomplete directory: its smoke test imported an adjacent extracted source tree, so it did not qualify the wheel. Running with `-I` excluded that path and verified the actual installed package. No scientific run or seed was replaced.

## Reproduce

From the repository:

```sh
uv sync --locked
uv run --frozen pytest -q
uv run --frozen python -m gate qualify --output evidence/s1-local-reproduction
```

Use a new output path. For the exact measured source, extract the primary run's `source.zip` over a fresh checkout of the manifest's base git revision, then run the same commands. The run was made on dirty worktree base `cfce036f87be71fcf9c729fd2b46140ee6394621`; source hashes and the archive identify the actual implemented code. The base git hash alone is insufficient to reproduce this run.

The primary run records Python 3.12.11, NumPy 2.5.4 and SciPy 1.18.1, its resolved lockfile, separate data/order seeds, raw decisions and looks. Simulator-specific fields explicitly say not applicable.

An independent post-run verifier is `../scripts/verify_s1.py`; it checks hashes, recounts trials, compares a separate reproduction, enumerates the unusual boundary count, records test/lint output, and copies the built distributions into a new verification directory. Wheel and source distribution are in `../evidence/s1-20261031-verification-v2/`.

## S2 admission

S1 no longer blocks S2 on software calibration. The draft manifest is `../experiments/S2_MANIFEST_DRAFT.json`; it is deliberately not a completed preregistration. Before a rented run: record P-005, re-price compute, pin the actual upstream branch/SHAs and evaluator, decide seed mode/Harmonizer/run method, and define the road population and strata. The proof does not establish independence of AlpaSim roads. Apply all existing dataset/license gates. P-003's actual human hours remain unrecorded; no delivery speed is imputed to Huy.

The repository and upstream issue are prepared locally, not published. The user instruction covered S1 engineering; no outward message or license acceptance was inferred.

VI: S1 đã có bằng chứng kiểm tra được cho phiên bản gate bảo thủ này. Có thể chuyển sang chuẩn bị S2, nhưng chưa thể thuê GPU chỉ vì test đã xanh. Cần chốt ngân sách, protocol và cách lấy mẫu road; điều đó kiểm tra các giả định mà chứng minh đang dựa vào.
