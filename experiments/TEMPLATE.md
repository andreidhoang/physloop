# <date> — <slug>

**Serves:** stage S? · deliverable G? (`docs/02_PHYSLOOP_PLAN.md` §3)
**Owner of the prediction:** Huy
**Curriculum exercise done** (`docs/03_MASTERY_PLAN.md` §4): yes / no

## Hypothesis (one sentence)

## Mechanism (why we expect it; derived, with numbers, L5 → L1)

## Falsifiable prediction (written BEFORE the run)
- Metric:
- Predicted value or range:
- Result that would make us abandon this:

## Pre-registered thresholds (if this experiment feeds a kill check)
- Floor or margin (reliability / coverage / known-group accuracy / Δ / ρ cut):
- Decision rule (PASS / KILL / INSUFFICIENT):

## Scope: what this experiment is NOT testing

## Setup
- Scenes / strata / replicates (scene list file, minimum stratum size; replicates per scene; seed mode: seeded with `force_determinism`, or unseeded replicates):
- Model / checkpoint (hash) and precision:
- AlpaSim branch and SHA (`main` and `e2e_challenge`), evaluator SHA, **and resolved lockfile**, `n_sim_steps` requested and actual, renderer image and Harmonizer state, traffic model (replay / CAT-K), deviation convention (2 m and 4 m), score definition (challenge `scene_score` or card-style km-between-events):
- Run method and topology (docker-compose / SLURM; replicas; concurrency); driver settings (expert CUDA graph on/off, `max_batch_size`, resolved `alpamayo1_5` commit); controller and post-processing identity:
- Contamination: the `splits/alpamayo_aligned.parquet` label (train / test / alpasim) of every scene's source clip:
- Interim looks (if any): spending function, K, futility rule (`02` S1):
- Hardware (GPU and edition; for Thor, name Jetson AGX or DRIVE AGX plus JetPack/DriveOS and TensorRT versions; real engine or emulated), GPU count and scale (one node / multi-node), GPU-hours computed from the **mean** wall clock, price (date), cost estimate:
- Command(s):

## Gates
- Budget tier recorded (P-005)? Hardware decision recorded (P-002 Thor, P-006 rig)? Spend stays within the stage line?
- Will any public number derive from PAI-AV or NuRec? If so, has KC-L passed (P-004)?
- Will any weights or labels trained on PAI-AV/NuRec-derived data be published? If so, has KC-L2 passed?

## Kill criterion (wall-clock or cost past which we stop regardless)

## Metric-to-decision rule
- If … then …; otherwise …

## Result (filled AFTER the run)
- Measured:
- Prediction vs measurement, and the causal explanation of any gap:
- Statistics used (paired? cluster-adjusted? one- or two-sided?):
- Evidence paths (`evidence/...`), git hash, config, manifest:

## Decision taken
