# S1 autonomous synthetic qualification

Stage S1, deliverable G1, CPU only, cloud cost $0. Agent-authored design and predictions under the user's explicit autonomous S1 instruction. This is not Huy's written prediction or evidence of his mastery. The earlier human worksheet is preserved.

## Mechanism before prediction

With independent roads and bounded paired road differences, exponential moment bounds control each directional error. Splitting a one-sided tail across positive and negative components gives a valid difference bound. Intersection-union GO needs all strata; a harmful stratum alone bounds false-GO. Across registered looks, the allocated tails sum to alpha. Holm and a separately spent family budget control false regression flags.

For identical road differences, the GO upper bound is 1-(alpha/2)^(1/R). At R=220, alpha=.05 this is about .01663, below a .05 margin. At R=9 it is about .3363, above either .02 or .05: the rare nonnegative-difference stratum cannot GO. Removing strata changes the target; a .021 loss confined to 9/441 scenes is only .000429 pooled. With almost no discordance outside that group the pooled bound can fall below .02. This is an invalid substitute for the per-stratum claim, not a Type-I error for the pooled estimand itself.

## Frozen configuration and predictions before Monte Carlo

Master seed 20261031. Each cell has 1,000 trials; independent trial streams derive from (master seed, cell index, trial index), not call order. Each trial spawns separate data and road-order streams; both seeds are recorded. Synthetic RNG seeds are not AlpaSim session seeds.

- Alpha .05; default NI margin .02, except positive-control identical outcomes at .05. All intervals explicitly one-sided.
- Four joint-pair probabilities in order (neither fails, A-only, B-only, both) are (1-pA-qplus, qminus, qplus, pA-qminus), where qplus-qminus=delta. Validity is checked before sampling.
- Road dependence: each scene copies its road's categorical latent outcome with probability road_correlation, otherwise draws independently. Replicate zero is the scene latent; later replicates copy it with probability replicate_correlation, otherwise draw independently. Marginal pair probabilities remain unchanged. Both named parameters are copy probabilities in [0,1], not calibrated ICCs. Complete roads arrive in a seeded predeclared random order.
- Cases: equivalent (pA=.07,qplus=qminus=.02); identical (pA=.07,qplus=qminus=0,margin=.05); NI boundary (pA=.07,qplus=.04,qminus=.02); harmful (pA=.07,qplus=.06,qminus=.02); gross regression (pA=.07,qplus=.27,qminus=.02); rare harm (432 common scenes with qplus=qminus=0 and 9 rare scenes with qplus=.021,qminus=0,pA=.07).
- Independent cells: one road per scene, one replicate, 441 scenes (two strata 220/221 except rare 432/9). Clustered cells: 147 roads x 3 scenes x 3 replicates, road_correlation=.8, replicate_correlation=.6, two strata of 73/74 complete roads. Boundary and harmful configurations run both fixed and sequential, plus a boundary cell with one bad and one identical stratum to challenge the IUT.
- Sequential: K=3, weights (.2,.3,.5); complete roads are split as evenly as possible into 3 batches within each stratum. Efficacy closes a stratum; no early futility, last undecided look is INSUFFICIENT. NO-GO has precedence. The same generated data feed fixed/sequential comparisons; only the decision schedule changes.
- P-A1: rare-stratum false-GO = 0, by the nonnegative-difference bound above; its observed INSUFFICIENT share >=.99.
- P-A2: equivalent independent .04-discordance trials have INSUFFICIENT share >=.80 (conservative split-tail width exceeds .02); this is a prediction, not an acceptance requirement.
- P-A3: removing stratification on the rare-harm cell raises invalid-for-the-stratified-claim GO by >=.80. No pooled error-control claim is made.
- P-A4: independent known-variance planning for a 20% relative improvement at psi=.04 and n=441 starts near baseline .118; with m=3 and rho=.6, illustrative effective n=601.36 lowers that to about .101. These are normal planning approximations, not operational gate certification, and road dependence can erase this gain.
- P-A5: the two-stratum equivalent .02-margin operational gate is predicted to leave at least one stratum INSUFFICIENT in >=.80 of trials. The real S3 allocation is not yet measured; no synthetic prediction is attributed to it.
- P-A6: identical fixed and sequential controls GO in every trial; gross independent regression NO-GO >=.95.
- P-A7: each registered harmful/boundary cell has false-GO <=.01 empirically; the mathematical guarantee is <=.05, not .01.

## Calibration acceptance and stopping

For one family of M=20 preregistered configuration-and-schedule calibration cells, compute exact one-sided Clopper-Pearson upper bounds at tail .05/M. Harmful/boundary cells test false-GO; equivalent and identical cells test false-NO-GO. Acceptance requires every upper bound <=.05, simultaneously across both types. This can be inconclusive even for a valid method; do not change seed, cells, trial count or rule after observing results. The pooled ablation is outside this family because it has a different estimand. Report all counts, denominators and bounds. If calibration fails, keep evidence, diagnose, preregister a new version before rerunning changed code.

All existing known answers must pass. Require identical-control GO=1, gross-regression NO-GO>=.95, rare-stratum GO=0 and pooling demonstration >=.80. Record missed predictions without rewriting them. CPU run wall-clock cap 30 minutes; no paid resources. This experiment qualifies the amended conservative operational gate under docs/S1_IMPLEMENTATION.md, not canonical Gaussian efficacy stopping on arbitrary cluster data.

## Results

Completed 2026-10-31. The preregistration above is unchanged after the run; its pre-run full-file hash and bytes are preserved in the primary manifest/source archive.

- Primary evidence: `evidence/s1-20261031-autonomous-v1/`. 10,000 trials, 20 registered calibration cells plus the pooled ablation. All registered calibration/control rules passed.
- Worst false-GO: 3/1,000 fixed and 2/1,000 sequential for one bad stratum; simultaneous one-sided upper bounds 0.01183453 and 0.01008366, below .05. Rare harm: zero GO for both schedules. Pooled ablation: 999 GO, illustrating the different estimand.
- Equivalent cases: 1,000 fixed INSUFFICIENT and 999 sequential INSUFFICIENT. Identical controls: 1,000 GO for each schedule. Gross regression: 1,000 fixed NO-GO, 999 sequential NO-GO. Conservative power is a limitation, not a hidden negative result.
- All numerical prediction thresholds were met. P-A4's planning values are .11840356 and .10139476, consistent with the registered approximations. They do not estimate actual S3/S4 base rates.
- Diagnostics and independently reconstructed exceptional trials: `docs/S1_HANDOFF.md`. The three fixed boundary errors are an unusual draw relative to exact enumeration (probability of >=3 approximately .00260331). No implementation/generation defect was found; seed and results are retained.
- Validation: 498 tests pass; a separate source-archive reproduction gives bitwise-identical raw trials, config, power tables and plot. The built wheel passes an isolated CLI smoke test. Records: `evidence/s1-20261031-verification-v2/`.
- Decision: amended conservative S1 engineering qualification passes; S2 preparation is unblocked. S2 execution still needs P-005 and its registered sampling/runtime protocol. Agent delivery is not human mastery evidence and is not a claim of unconditional correctness.
