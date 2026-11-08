# Researcher C: experimental validity and transfer (2026-11-05)

This review is agent-authored and read-only. It serves S2 (G1, G2, G3·R1, G7), with S4.6 as the downstream consumer. The lane covers the population, randomness, deviation, missingness, rendering validity and power.

What was not done:
- No repository file was edited.
- No license was accepted and no gated data or Git LFS object was fetched.
- Nothing was rented, posted or run on a GPU.

What was done:
- Only public AlpaSim catalog CSVs and source files were downloaded, into the scratchpad.
- The gate's functions were imported, never modified, and run on the CPU.
- The untracked folders `docs/research/2026-11-05-contribution-review/` and `evidence/contribution-review-20261105-v1/` appeared during this session. They were not read, to keep this lane independent. A grep incidentally showed one line of them.

**Evidence labels:**
- **source-inspected:** a pinned public file was read.
- **doc-stated:** taken from vendor or maintainer text.
- **derived:** obtained by algebra or counting.
- **executed-CPU:** a script was run in the scratchpad.
- **hypothesis:** not yet established.

Margins and failure rates in §6 are illustrative and are not Huy's.

## 0. Upstream state since 2026-11-01

| Ref | Head on 2026-11-05 | Change since 11-01 | Consequence |
|---|---|---|---|
| AlpaSim `e2e_challenge` | `297b4d00cfdd2c4f636b9d9bcee7a85c699174c0` (#228, merged 11-03 by `mwatson-nvidia`; outside author) | **Moved.** All 39 changed files are under `e2e_challenge/local_evaluation/`: README, `test_evaluate.py`, and the nuPlan reference manifest plus 36 LFS score files. No `src/` change. The scorer blob `6dce0f50…` and writer blob `2abd4b09…` are re-hashed and unchanged. | The reviewed pin `e121e591` remains the scoring content. Record that the HEAD differs from the pin. Re-check before S2 admission. |
| AlpaSim `main` | `affc2eab…` | None | — |
| AlpaGym `main` | `972d160e…` | None | Gym's AlpaSim pin `46b401be…` is an immutable commit. |
| AlpaSim releases | v2026.5 (2026-06-04) is the latest | None | The post-challenge release promised in #224 has not appeared. |
| PAI-AV card | v26.11: 306,151 clips (one removed); `alpamayo_aligned` train 276,298 / test 28,317 / alpasim 1,536; new `challenging_scenarios` split | The card moved since the 10-27 notes (306,152 / 276,299). | Pin the dataset revision and parquet hashes at S2. |
| NuRec card | `main` = 26.10 (1,601 scenes, 7 cameras); the 26.04 branch reports "All 1,607 USDZ files were replaced on 2026-07-30" | Current release ≠ the release the 441 scenes resolve to | See §1 and candidate 2. |

Offered upstream work that must not be competed with (§8): PR #234, issue #237, PR #235 and PR #238. The nuPlan deadline is extended to 11-16 (#225). The PAI final-rerun selection was answered in #232.

## 1. Population and hierarchy

**What "441" is (source-inspected and executed-CPU).** It is AlpaSim's community-contributed suite `nurec_curated_val`. PR #167 (`85d5cb13`, 2026-08-31, outside contributor) added it with `sim_suites_curated.csv`, which has exactly one commit in its history and is unchanged at the new HEAD. The SHA-256 of the CSV fetched at `297b4d00` is `326cb0eb…4936`. My scan of the public CSVs found:
- val 441 and train 1,761, with zero overlap;
- val drawn as 177 scenes from the 26.01 catalog (NRE `26.1.112-59673f88`, `hf_revision` "26.01") and 264 from the 26.04 catalog (NRE `26.4.96-91b06fb8`, `hf_revision` "26.04").

The two reconstruction versions are a nuisance factor inside the population; record them per scene as metadata. "441" is therefore:
- a registered historical list on `e2e_challenge` only;
- absent from `main` (its `data/scenes/` has no curated suites and a different 606 KB catalog);
- not equal to NuRec's current 26.10 release.

It is not the 1,761/441 PAI-AV split. `alpamayo_aligned` is a different partition (train/test/alpasim).

**Road and geography identifiers (source-inspected and doc-stated).**
- The AlpaSim catalogs carry only `uuid, scene_id (= clipgt-<clip uuid>), nre_version_string, path, last_modified, artifact_repository, hf_revision`.
- PR #167 and the README say road-sharing scenes "are grouped and assigned to the same split". The method, group IDs and road count are unpublished, and "Different roads or driving directions at the same location may occur across splits".
- The grouping prevents train/val leakage. It does **not** give the clustering *within* val.
- The PAI-AV card lists `metadata/data_collection.parquet` (country, month, time of day, `platform_class`) and `clip_index` (`clip_is_valid`, `chunk`; a chunk is a storage block of up to 100 clips, not a drive). Both are gated. No GPS, city, road, drive or session ID is listed.
- The NuRec card lists `clip_ratings` (quality score, time of day, ego speed; 26.04 branch, gated) and `labels.json` (25.05 only).

**What is knowable without accepting a license:**
- scene and clip UUIDs;
- release and NRE-version membership;
- curated split membership;
- the fact that an undisclosed road grouping exists.

**What is not knowable:** the number of independent roads, their identities and any geographic or temporal clustering. The count lies somewhere between an unknown lower bound and 441.

The preflight's `road_list` and `road_independence_justification` cannot be populated from public data. After license acceptance, country/month/time-of-day can only coarse-cluster.

*Hypothesis to test at S2 bootstrap:* the USDZ map or egomotion files may carry a geodetic origin, which would allow geometry-based road grouping. If they do not, road identity needs NVIDIA's grouping or Huy's documented scientific amendment (S2.1 already says to stop in that case).

**Contamination counting (derived).** The public catalogs hold 2,364 unique scene IDs (916 + 1,448, zero overlap), and only 1,536 clips carry the `alpasim` label. Assume every NuRec scene UUID is a PAI-AV `clip_id` carrying exactly one label (the cards say NuRec scenes are PAI-AV clips). Then **at least 828 public NuRec scenes, and at least 666 of the 2,202 curated scenes, are labelled `train` or `test`**, not `alpasim`.

Whether any of the 441 is a teacher-training clip cannot be verified without the gated parquet. The S2.5 audit stays mandatory.

**Public reference runs.** The bundle of 8 subjects × 441 scenes × 3 rollouts under the challenge contract is in Git LFS and was not fetched. The 10-27 L4-10 proposal already covers it. It would give the within-scene replicate variance v̄ that §6 needs. Whether fetching it falls under the no-NuRec-download/KC-L rules is Huy's decision.

## 2. Randomness and coupling

| Source | `e2e_challenge` (`e121e591` / `297b4d00`) | `main` (`affc2eab`) | Evidence |
|---|---|---|---|
| Policy sampling: Alpamayo CoC top-p 0.98 / T 0.6, then flow noise | Global torch RNG, never reseeded; the session seed is stored and ignored | `torch.manual_seed(inference_seed)` plus CUDA reseed **only if** `force_determinism` (false in shipped configs). `inference_seed = session.seed + inference_count` (driver `main.py` L713–714; `alpamayo_base.py` L612–615) | source-inspected (challenge established 10-31; main re-read) |
| Session-seed origin | n/a | Daemon `engine.py`: `spec.random_seed + rollout_idx` if set, else 0. `event_loop` passes `session_seed or None`, and `driver_service` falls back to `random.randint(0, 2**32-1)` (L76–78). **The stock `simulate/__main__.py` builds `RolloutSpec` without `random_seed` (L186–192), and no wizard config sets it.** The same seed goes to the driver and traffic sessions. | source-inspected |
| GPU numeric nondeterminism | No deterministic configuration | `configure_deterministic_runtime()` (cuDNN deterministic, `use_deterministic_algorithms`, CUBLAS workspace) runs only with `force_determinism` | source-inspected |
| Cross-rollout RNG state | One process-global stream shared by every rollout the driver process serves, so a rollout's draws depend on the schedule, retries and concurrency. PyTorch's default generator starts from a fixed seed, so fresh processes with identical schedules could partly repeat draws. | Removed by per-tick reseeding when enabled | Shared stream: derived. Default seed: hypothesis (PyTorch default, not checked in this driver) |
| Renderer (NuRec, `nre-ga:26.04`) | Maintainer: small run-to-run variation possible (#128, closed 09-10). Not bit-identical across NRE versions or GPU families (26.08 and 26.09 notes) | same | doc-stated |
| Harmonizer | The base config passes the stateless `--enable-harmonizer` to `nre-ga:26.04`. It is a diffusion enhancer (CosmosPredict2 0.6B); **no seed or determinism statement exists**. Caveat: the release notes date that flag name to the 26.06 rename and temporal mode to 26.08, so the mapping from image tag to NuRec release is unverified. "Only stateless mode is reachable" is conditional on that mapping. | No flag on `main` (10-27 notes) | doc-stated + config; determinism is a hypothesis for the fixed-pose probe |
| Traffic | Log replay (no CAT-K) | CAT-K's session seed is only logged (#149 note) | doc-stated / source-inspected |
| Controller / physics | Nonlinear MPC (base `controller: nonlinear`, per the local-evaluation README), `physics_update_mode: EGO_ONLY`; no RNG found in the files read | Not re-inspected | config/doc-inspected; solver determinism is a hypothesis |
| Timing | `planner_delay_us: 0`, `assert_zero_decision_delay: true`, driver `n_concurrent_rollouts: 1` | same defaults | config-inspected |
| Retries | Only `RetryableRendererInfrastructureError`, at most once (`max_retry_attempts=1`), with a **fresh session UUID**. The failed attempt is only logged (`scheduler.py` L186–213; `worker/main.py` L163–186) | **Any failure except `INVALID_SCENE`**, up to 2 retries (`replace(pending_job, …)`, L843–875). The **seed is always preserved**. The UUID is reused, and the failed directory **deleted** (`_clear_retry_rollout_dir`), only when the request supplied session UUIDs (`engine.py` L208). The stock path supplies none, so the retry gets a fresh runtime UUID and the failed directory persists. | source-inspected |

**Do two different policies with the same session seed get identical noise on `main`?**
- **Same seed is never reached by default:** the stock wizard path sends no seed, so each session draws `random.randint`.
- **With a seed set:** if a seed is injected through a direct daemon `SimulationRequest` and `force_determinism` is true, both arms reseed to `s + n` at inference n. The draws are identical while both arms consume the RNG identically.
- **S4 case:** for S4's frozen-VLM, expert-only checkpoints, identical observations give identical CoC tokens and therefore identical flow noise. Coupling holds up to the first tick where the actions, and so the rendered inputs, differ. After that, CoC length can differ and the flow-noise offset shifts.
- **Per-tick resynchronization:** each tick restarts from `s + n`.
- **Alignment requirement:** this also needs the same `inference_count` alignment (same `skip_driver_during_force_gt`) and batch size 1.

These statements are derived; none is measured.

**Determinism-related upstream issues and PRs:**
- #128 (closed 2026-09-10);
- #149 (merged 2026-08-17; it added `RolloutSpec.random_seed` and deliberately left trafficsim and video-model sampling out of scope);
- no open seed or determinism PR found on 2026-11-05.

## 3. Deviation and validity

**Scorer definitions, `GroundTruthScorer` blob `6dce0f50` (source-inspected):**
- `dist_to_gt_trajectory` is the 2-D distance from the **ego polygon centroid** to the **whole recorded GT polyline**. It is time-free: lagging along the path counts as zero. It is computed at each simulation timestamp from index 2 on, with indices 0–1 hard-coded to 0.0, and reduced by `MAX`.
- `dist_to_gt_location` is the time-aligned counterpart: the distance to the GT point at the same timestamp.
- `lateral_dist_to_gt_trajectory` takes the minimum of the path distance and the perpendicular distance to the forward endpoint ray, so overshoot is not lateral departure.
- `left_corridor_laterally` is `lateral ≥ max_dist_to_gt_trajectory`. That value is 4.0 in the base, dev and EC2 presets, and the time reduction is MAX.

**Summary support, after the offline modifiers:**
- Timesteps before `eval_relevant` are removed. These are the pre-evaluation steps; that this equals the 1.7 s force-GT warm-up is inferred from the 10-31 audit, not re-verified.
- Timesteps after the first `collision_any|offroad` are removed.
- Timesteps after the first `dist_to_gt_trajectory ≥ 4` are removed; the triggering step is kept.

The summary maximum is therefore a maximum over this truncated support (already recorded 10-31). Within the GT path span, ordinary ≥ 4 m implies lateral ≥ 4 m. **Most ≥ 4 m exits are therefore scored 0 by the challenge score itself.** The validity exit is folded into the outcome; only an overshoot past the endpoint can separate the two.

**The plan's bins (`<1`, `1–2`, `2–4`, `≥4 m`).** The draft convention says "full-trajectory ordinary distance". This must be computed from the untruncated per-timestep `metrics.parquet`, which the adapter ledger's `full_max_deviation_m` asks for. The definition still to be registered covers:
- the time support: whether warm-up is included;
- whether time ends at the clip end or the requested horizon;
- the reference point: polygon centroid versus rig.

**Why policy-dependent bins cannot be comparison strata (derived).**
- Deviation D is a post-treatment outcome: D_B depends on policy B and its noise. Stratifying on D_B, or on max(D_A, D_B), conditions on a consequence of treatment.
- Within a bin, the arms are compared on scene subsets selected by their own behaviour. A policy that deviates less moves hard scenes into lower bins, which changes the composition of each bin.
- The within-bin contrast is then a principal-stratum effect, identifiable only under untestable assumptions (for example that D does not depend on the policy).
- Even stratifying on D_A alone (a reference-run outcome) induces regression to the mean under unseeded replicates. The scene's bin is itself random, so scenes selected for high D_A will show B "improving" with identical policies.
- A row-level bin also breaks the gate's contract: a road must belong to exactly one stratum, fixed before outcomes.

**Legitimate alternatives:**
- pre-treatment strata from scene metadata registered before outcomes;
- a teacher-envelope class estimated from the S2 teacher sweep and frozen before S4 runs, used as a covariate and never reused as comparison-arm data;
- deviation bins reported as an **outcome**: a paired 4×4 (bin_A × bin_B) table per stratum, with exits bounded as in §4.

**What the gate does with deviation (source-inspected, `gate/verdict.py`, read only).** `a_deviation_m` and `b_deviation_m` are optional. If present, they must be finite and non-negative. Each arm adds `int(dev ≥ 4)` to a per-stratum counter (`out_of_envelope_a/b`).
- The rows stay in d; nothing is excluded, re-stratified or bounded.
- The counter is unpaired, so the table of who exited where is lost.
- Upstream, the adapter copies `full_max_deviation_m` from the trusted ledger into `deviation_m`. It does not compute it.

## 4. Missingness and failure paths

| Class | How it appears on the challenge pin | PhysLoop handling today (`gate/alpasim_adapter.py`) | Gap |
|---|---|---|---|
| Infrastructure (renderer RPC, transport) | A renderer error is retried once with a fresh session; a second failure gives a failed-rollout row with `score=0.0`, a `metrics.error` key and null score metrics. The first attempt survives only in the log and an orphan directory. | `infrastructure_error` quarantine on key presence. The whole batch is not admitted (`records=[]`). | Retry lineage needs the runtime log or orphan directory; the summary alone hides the first attempt. |
| Renderer / simulator validity (black or smeared frames) | `ImageScorer` is registered (`scorers/__init__.py` blob `553c7ffe`) and computes `img_is_black`: one camera, all pixels ≤ 10, MAX. When the scorer ran, the summary's per-rollout `metrics` holds its value, maximized over the truncated support (`processing.py` L291–299). The score ignores it. **The challenge `DEFAULT_MODIFIERS` lacks `RemoveTrajectoryWithEvent(img_is_black)`, which #109 removed on 2026-06-16.** `main` and Gym's `46b401be` still drop such trajectories (line 64). `ScorerGroup` swallows scorer exceptions (`base.py` L33–45). The NuRec 26.10 notes list "nonfinite sensor pose → badly smeared render at status OK". | The adapter does not read `img_is_black`; no ledger field | Challenge: render failures are scored as driving outcomes, and per-tick coverage needs the parquet. Main/Gym: silent row loss. Candidate 1. |
| Policy-induced invalid state | A driver termination or exception becomes a failed row (`fail_on_driver_termination: true`). Driver OOM is not retried on the challenge pin (#222 plus source). A ≥ 4 m exit is a corridor failure with score 0. | An error row is quarantined as infrastructure; ≥ 4 m is counted only | A policy-caused crash is mislabelled as infrastructure, and policy-dependent missingness is hidden. |
| Adverse driving outcome | Collision, offroad or corridor exit gives score 0, in the reason order at-fault → offroad → corridor. The multi-actor collision class depends on query order (#237, same loop on the pin at `collision.py` L45–56). | Scored | Sensitivity: rescore the logs locally with the OR-over-actors loop. Do not open a competing PR (#234). |
| Missing artifact or metric | `offroad=0.0` is inserted when the column is absent (`processing.py` L752). A crashed scorer silently drops its columns. A clipped available recording leads to `horizon_unverified`. | `metrics_unverified`, `horizon_unverified`, `missing_expected_rollout` | Correct fail-closed behaviour; needs the extractor. |

**Bounds instead of complete-case analysis (executed-CPU, `missing_sensitivity.py`, illustrative model μ = 0.1, κ = 0.5, 441 scenes × 3 replicates, margin 0.10).** The sensitivity rule: impute failed or missing logical observations at the extreme adverse to GO (missing B → 0, missing A → 1). Then pass the complete rows through the **unmodified** `verdict`, and claim GO only if the adverse analysis is GO. Report per-arm failure counts as a separate outcome.

Under missing-completely-at-random draws, P(GO) was:

| roads | missing per arm | adverse | complete-case |
|---|---|---|---|
| 147 | 0% | 0.89 | 0.89 |
| 147 | 0.5% | 0.80 | 0.93 |
| 147 | 1% | 0.60 | 0.90 |
| 147 | 2% | 0.16 | 0.87 |
| 147 | 5% | 0.00 | 0.73 |
| 441 | 5% | 0.11 | 1.00 |

- The worst-case shift in mean d is at most 2 × the missing rate, because |d| ≤ 1 per rollout. **Infrastructure failure at 1–2% per arm consumes most of a 0.10 margin.**
- Complete-case analysis keeps the GO only because these simulated failures are random. Driver OOM and render failures plausibly correlate with scene difficulty (hypothesis), which would bias the complete-case result.
- Horizons: register a common-support rule that scores both arms on min(actual horizon_A, actual horizon_B) per scene, or carry clipped scenes as a separate stratum. Do not drop them.

## 5. Rendering validity (NuRec documentation)

**Harmonizer** ([page](https://docs.nvidia.com/nurec/nurec/harmonizer.html), last updated 2026-10-30). It is a diffusion enhancer on CosmosPredict2 0.6B and runs in two exclusive modes:
- **Non-temporal** (`--enable-harmonizer`): stateless per frame; compatible with batch rendering and multi-process serving.
- **Temporal** (`--enable-harmonizer-temporal`, since 26.08): conditions on previously enhanced frames of a session and requires create/end session calls with frames in trajectory order. The page warns that reusing a session across sequences "corrupts the temporal conditioning".

The weights come from gated Hugging Face Cosmos repositories (a separate license), and the page makes **no determinism or seed statement**.

**Release notes:**
- 26.09 changed the default temporal checkpoint, so its URL must be pinned.
- 26.10 adds per-request `apply_harmonizer`, automatic temporal sessions and a history reset.
- 26.08 and 26.09 renders are not bit-identical to the prior releases.
- Neither page documents how far from the recorded path reconstructions remain valid. The 4 m corridor is an AlpaSim challenge convention (#163: "NuRec corridor (4m tunnel)"), not a NuRec-stated envelope.
- The 26.10 notes add "ego-path protection for the camera ground mesh", which suggests degradation away from the ego path (hypothesis).

**Consequence:** the challenge config requests only the stateless mode, so no session state should leak across rollouts. That conclusion is conditional on the unverified mapping from image tag to NuRec release: the image `nre-ga:26.04` already accepts a flag name the notes date to 26.06. A move to 26.08 or later changes protocol, state semantics and checkpoint, and must be a registered change. The determinism of Harmonizer outputs is unknown; S2's fixed-pose renderer probe should repeat the same pose with Harmonizer off and on.

## 6. Power with the gate's own functions

`power_tables.py` imports `bounded_mean_bounds`, `paired_road_bounds` and `verdict`. Every constructed case is cross-checked against `verdict()` for a single stratum (executed-CPU).

**Mechanism (derived).** For an equivalent pair (true mean d = 0) with m₊ = mean(d⁺) = mean(d⁻), the operational bound is approximately UB ≈ 2√(2 m₊ ln(2/α)/R) + O(ln(2/α)/R). GO at margin δ therefore needs roughly R ≳ 29.5 m₊/δ² at α = 0.05.

The KL bound uses the Bernoulli variance proxy m₊(1−m₊). It is insensitive to how small the continuous road differences actually are.
- **Check:** with R = 441 and m₊ = 0.0541 the formula gives 0.060; the gate gives 0.0589.
- **Rule of thumb:** for roughly symmetric d, m₊ ≈ 0.4·SD_road, so GO needs SD_road ≲ δ²R/11.8.
- **With no true policy-by-scene interaction:** SD_road = √(2v̄/(R_rep·S)), where v̄ is the mean within-scene per-rollout score variance and S the scenes per road. Then UB ∝ (v̄/(R_rep·N))^{1/4}·R^{-1/4}.
- **Replicates are a weak lever:** doubling them shrinks UB only by 2^{-1/4} ≈ 0.84.
- **Unseeded noise is the problem:** the "same-policy" control can GO only if v̄ is small. Coupling or seeding (§2) attacks v̄ directly.

**Table B (executed-CPU): GO feasibility for an equivalent pair, α = 0.05.** The last five columns give the largest m₊ = mean(d⁺) = mean(d⁻) at which the upper bound still falls below the margin. A dash means GO is impossible even if every road difference is zero.

| Roads R | All-zero floor = 1−(α/2)^{1/R} | Holm single-stratum flag needs mean d ≥ | margin 0.01 | margin 0.02 | margin 0.05 | margin 0.07 | margin 0.10 |
|---|---|---|---|---|---|---|---|
| 30 | 0.1157 | 0.447 | — | — | — | — | — |
| 60 | 0.0596 | 0.316 | — | — | — | 0.0025 | 0.0145 |
| 110 | 0.0330 | 0.233 | — | — | 0.0054 | 0.0153 | 0.0365 |
| 147 | 0.0248 | 0.202 | — | — | 0.0099 | 0.0229 | 0.0514 |
| 220 | 0.0166 | 0.165 | — | 0.0008 | 0.0174 | 0.0369 | 0.0811 |
| 441 (every scene its own road; an upper bound the public files cannot justify) | 0.0083 | 0.117 | 0.0004 | 0.0051 | 0.0383 | 0.0795 | 0.1843 |

The noise-driven Holm flag is not the binding constraint; the width is. With k registered strata each stratum has roughly R/k roads. For example, 4 strata out of 441 roads leave about 110 each, so margin 0.05 tolerates only m₊ ≤ 0.0054.

**Table A excerpt (executed-CPU): constructed equivalent road vectors.** A fraction f of roads is discordant, half at +|d| and half at −|d|.
- R = 147, f = 0.1: |d| = 0.1 gives UB 0.039 (GO at 0.05); |d| = 0.25 gives 0.054 (GO at 0.07); |d| = 1 gives 0.097 (GO only at 0.10).
- R = 147, f = 0.5: |d| = 0.25 gives UB 0.108 (no GO up to 0.10).
- R = 441, f = 0.5: |d| = 0.25 gives UB 0.063.

**Table C (executed-CPU, 300 trials per cell): illustrative unseeded rollout model.** Each rollout fails with probability p_s ~ Beta(μκ, (1−μ)κ) and scores 0. Otherwise it scores 1 with probability 0.85, else U(0.3, 1). The design is 441 scenes × 3 replicates, split equally over R roads. Arms are A and B of an identical policy.

| μ | κ | R | median m₊ | median UB | P(GO@0.05) | P(GO@0.07) | P(GO@0.10) |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.5 | 60 | 0.025 | 0.120 | 0 | 0 | 0.01 |
| 0.1 | 0.5 | 147 | 0.038 | 0.086 | 0 | 0.04 | 0.96 |
| 0.1 | 0.5 | 441 | 0.054 | 0.059 | 0.14 | 0.88 | 1.00 |
| 0.3 | 2 | 147 | 0.069 | 0.114 | 0 | 0.00 | 0.16 |
| 0.3 | 2 | 441 | 0.109 | 0.080 | 0.01 | 0.23 | 0.93 |
| 0.5 | 10 | 147 | 0.088 | 0.129 | 0 | 0.00 | 0.08 |
| 0.5 | 10 | 441 | 0.146 | 0.092 | 0.01 | 0.11 | 0.69 |

Margins 0.01 and 0.02 gave P(GO) = 0 in every cell. The full grid is in `power_out/power_tables.md`.

**Diagnostic comparators (not authorization methods):**
- A normal-oracle UB (invalid at finite sample) is 0.013–0.032, about 4–9× tighter than the operational bound.
- A Maurer–Pontil empirical-Bernstein bound is no better at R ≤ 441, because its range term 14 ln(2/α)/(3(R−1)) is 0.039 at 441 and 0.118 at 147.

A tighter valid method, for example betting, would be an inference-method amendment. Under D-016 that is a scope decision for Huy. It is noted here, not proposed.

**Implication, under the illustrative noise model:** the unseeded same-policy control can be expected to GO only at margins of about 0.07–0.10, and only if the independent road count is well above 147. S2 must measure v̄, the road count and v̄'s within-road correlation before the S4 claim is priced.

## 7. Candidates (at most two)

### Candidate 1: render-validity status in S2 extraction (instrumentation; fits S2.3/S2.5)

- **Beneficiary and workload:** the S2 full sweeps (2 × 441 × ≥ 3 = 2,646 rollouts), the five-scene pilot, the S4.6 held-out evaluation and the G2 records.

- **Existing implementation and problem (source-inspected).**
  - The challenge evaluator computes `img_is_black` but deliberately dropped the trajectory-removal modifier (#109, "img black failure that was throwing off r…").
  - As a result, a black frame that leads to a collision, offroad or corridor exit is scored as a policy failure.
  - The summary's per-rollout `metrics` carries `img_is_black` only when the scorer ran, and only over the truncated support. The score ignores it, and the PhysLoop adapter does not read it.
  - `main` and Gym's pin instead delete such trajectories silently.
  - `ScorerGroup` swallows `ImageScorer` exceptions, so the metric can be absent without notice.
  - NuRec 26.10 documents smeared renders from nonfinite poses at status OK.
  - The PhysLoop ledger has no render field, although the S2.5 plan requires per-rollout render failures.

- **Strongest baseline:** upstream `img_is_black` with drop (main/Gym) or with ignore (challenge). Both lose the paired accounting.

- **Narrow change:** in the agent-owned extractor and `alpasim_adapter.py` (not locked):
  - require `img_is_black` rows covering every driver tick, else `render_status_unverified`;
  - record any black tick and the first black tick;
  - check finiteness of the ego and sensor poses;
  - set the ledger field `render_status ∈ {clean, black_at_t, nonfinite_pose_at_t, unverified}`;
  - quarantine invalid rows as non-driving outcomes and report a paired (A valid, B valid) 2×2 table per stratum.

  Huy decides the analysis rule: exclude, adverse-impute (§4) or separate outcome.

- **Mechanism.** Let V be the validity indicator. Then E[d_naive] − E[d | V_A = V_B = 1] is bounded in absolute value by P(V_A = 0 ∨ V_B = 0), and the bias is non-zero when ρ_A ≠ ρ_B. Renderer failure rates can differ by policy because policies visit different poses (hypothesis).

- **Objective:**
  - render-invalid rate per arm, with an exact binomial bound;
  - the share of score-0 rollouts whose first black or nonfinite tick precedes the failure tick;
  - the target is no row admitted without verified status.

- **Smallest disproof experiment:**
  - Now, on CPU: a synthetic `metrics.parquet` / ledger fixture with `img_is_black = 1` at tick t and a collision at t + 3 must be quarantined; one with `img_is_black` absent must give `unverified`.
  - At the already-planned pilot: count black and nonfinite ticks, at no extra rollout cost.
  - Disproof: all ticks are present, there are zero invalid ticks across the full sweep in both arms, and the status was verified. The check then stays as a cheap guard.
  - What it misses: partial corruption, Harmonizer hallucination, and dark-scene false positives at the ≤ 10/255 threshold.

- **Failure conditions:**
  - the evaluator receives no cameras (accumulator only gets driver camera images);
  - night or tunnel false positives, a plausible reason for #109 (hypothesis);
  - failure modes that are not black frames.

- **Adopting component:** PhysLoop adapter/extractor only. Do not open an upstream PR that re-adds the modifier: NVIDIA removed it intentionally on the challenge branch, and #235 (invalid-timestep aggregation) is a related offered PR that must not be competed with.

### Candidate 2: content-addressed scene identity (instrumentation; fits S2.1/S2.3)

- **Beneficiary and workload:** every paired comparison made at different times or on different hosts: the S2 teacher sweep against the S4 checkpoint evaluation, and the S4.6 reference arm. PhysLoop rents, then destroys, hosts per unit, so each unit re-downloads its assets.

- **Existing implementation and problem.**
  - The catalog `hf_revision` values are branch names ("26.01", "26.04") passed to `hf_hub_download(revision=…)` (`sceneset.py` L459–465).
  - The only check is the artifact UUID in `metadata.yaml` (L472).
  - A cached file is reused with no hash (L499).
  - The NuRec card states that every 26.04 USDZ was replaced on 2026-07-30; the 26.04 branch tree shows commit `5fe7a149…`, "3 months ago".
  - Renders are not bit-identical across NRE versions.
  - So the same AlpaSim SHA and the same 441 IDs can resolve to different bytes on different dates. The draft manifest stores only `scene_list`.

- **Strongest baseline:** pin `hf_revision` to a commit SHA, which `hf_hub_download` accepts; record the Hugging Face LFS `oid`/SHA-256 per file (DVC-style content addressing).

- **Narrow change:** at the licensed bootstrap:
  - resolve each branch to its commit SHA;
  - write a frozen copy of the catalog with commit-SHA revisions (config-only, through `scenes.scenes_csv`);
  - record the SHA-256 and the NRE version of each USDZ in the manifest and observation ledger;
  - have the preflight require them;
  - verify cache hashes before every unit;
  - block pairing on any mismatch.

  An optional AlpaSim PR could add a `sha256` column check. No offered PR overlaps (#56 is trajdata loading). Re-check before filing.

- **Mechanism.** d_s = f_B(x_s) − f_A(x_s′). If x_s′ ≠ x_s, then E[d] = Δ_policy + Δ_asset, and the two terms cannot be separated. Hash equality forces Δ_asset = 0 by construction. Renderer nondeterminism stays a separate probe.

- **Objective:** a per-scene hash match rate of 1.0 between registration and every evaluation unit; any mismatch blocks the comparison.

- **Smallest disproof experiment:** after license acceptance, list Hugging Face tree metadata (LFS oids) for the 441 paths at S2 admission and again at S4 admission, with no GPU and no asset re-download. If nothing changes and the caches persist, the candidate adds bookkeeping only. It covers asset identity; renderer and Harmonizer stochasticity stay outside it.

- **Failure conditions:**
  - Hugging Face history rewrite (older commits unavailable);
  - the 12-month license clock (§8) blocking re-download;
  - asset retention adding storage cost.

- **Adopting component:** `gate/s2_preflight.py` and the extractor (agent-owned, not locked), plus a frozen catalog file. Upstream is optional.

**Not proposed:** a change to the gate's bound (locked file, D-016 scope); a seeded comparison port (already a separately qualified diagnostic in S2.4).

## 8. Offered upstream work (record; do not compete)

| # | State (11-05) | Base | Subject | Relevance |
|---|---|---|---|---|
| 234 | open | main | Collision flags reset per actor, so the union is lost | Same loop on the challenge pin. Use it as a local rescoring sensitivity only. |
| 237 | open issue | — | The same bug; it reports a pass/fail flip | Validity sensitivity |
| 235 | open | e2e_challenge | Invalid timesteps leak into the run summary | Affects diagnostics such as `min_distance_to_obstacle_m`, not the score inputs |
| 238 | open | main | Commits `uv.lock`; pins `alpamayo1-5 @ 36aeb4c` and others; closes #207 | Environment pin. Do not file our own. |
| 228 | merged 11-03 | e2e_challenge | nuPlan reference scores | The HEAD moved; no scoring change |
| 149 | merged 08-17 | main | `RolloutSpec.random_seed` | The seed exists in the protocol but is not wired into the stock simulate path |

## 9. Reproduce (read-only, from the repository root)

```sh
SP=/private/tmp/claude-501/-Users-danghuyhoang-Desktop-physloop/0fe84b24-06c3-443d-87b9-6d50134535e0/scratchpad
uv run --frozen python $SP/scripts/power_tables.py $SP/power_out        # sha256 b543eee4...
uv run --frozen python $SP/scripts/missing_sensitivity.py $SP/power_out # sha256 8afee577...
python3 -I $SP/scripts/catalog_audit.py $SP/upstream_csv                # sha256 eb5edfbe...
```

The environment was Python 3.12.11, NumPy 2.5.4 and SciPy 1.18.1, with Monte Carlo seeds 20261105 and 20261106. The outputs are `power_out/power_tables.json` (sha256 131257b9…), `power_out/power_tables.md` and `power_out/missing_sensitivity.json` (86c7802e…). The public CSV copies are in `upstream_csv/`.

## 10. Evidence log

| Claim | Source (URL @ SHA/version) | Read | Type | Applicability | Uncertainty | Consequence |
|---|---|---|---|---|---|---|
| The `e2e_challenge` HEAD moved to 297b4d00; all 39 changed files are under `e2e_challenge/local_evaluation/` | https://github.com/NVlabs/alpasim/pull/228 (files pages 1–2); https://github.com/NVlabs/alpasim/commit/297b4d00cfdd2c4f636b9d9bcee7a85c699174c0 | 11-05 | source-inspected | S2.1 | Low | The pin stays valid for scoring; record HEAD ≠ pin |
| `main` (affc2eab) and AlpaGym (972d160e) are unchanged; the latest release is v2026.5 | https://api.github.com/repos/NVlabs/alpasim/releases ; https://github.com/NVlabs/alpasim/commits/main ; https://github.com/NVlabs/alpagym/commits/main | 11-05 | source-inspected | S2.1 | Low | No re-audit needed |
| Scorer blob 6dce0f50 (sha256 5bd239cb…) and writer blob 2abd4b09 are unchanged at the new HEAD | https://github.com/NVlabs/alpasim/blob/297b4d00cfdd2c4f636b9d9bcee7a85c699174c0/src/eval/src/eval/scorers/ground_truth.py ; https://github.com/NVlabs/alpasim/blob/297b4d00cfdd2c4f636b9d9bcee7a85c699174c0/src/eval/src/eval/aggregation/processing.py | 11-05 | executed-CPU (`git hash-object`) | S2.3 | Low | Existing fixtures remain valid |
| 441 = `nurec_curated_val`: 177 from 26.01 (NRE 26.1.112) and 264 from 26.04 (NRE 26.4.96); train/val overlap 0; one commit in the file's history | https://github.com/NVlabs/alpasim/blob/297b4d00cfdd2c4f636b9d9bcee7a85c699174c0/data/scenes/sim_suites_curated.csv (sha256 326cb0eb…); sim_scenes.csv (b1cc2d2f…); sim_scenes_2604.csv (cf0df2b7…); https://github.com/NVlabs/alpasim/pull/167 | 11-05 | executed-CPU | S2.1/S2.5 | Low | Register the population as a historical list, with NRE version per scene |
| `main` has no curated suites; its catalog differs | https://github.com/NVlabs/alpasim/tree/affc2eab209fa43bdfa2f26c0f8d437922d78a68/data/scenes | 11-05 | source-inspected | S2.1 | Low | "441" belongs to the challenge branch only |
| No road, drive, session, geographic or time IDs in the public AlpaSim files; the grouping method and groups are unpublished | https://github.com/NVlabs/alpasim/pull/167 ; https://github.com/NVlabs/alpasim/blob/297b4d00cfdd2c4f636b9d9bcee7a85c699174c0/e2e_challenge/local_evaluation/README.md | 11-05 | source-inspected | S2.1/S2.6 | Low | The independent road count cannot be known publicly; `road_list` is blocked |
| PAI-AV metadata (country, month, time of day, platform_class; `clip_index` chunk) is gated; no GPS or road IDs are listed; v26.11 counts 306,151 clips and 276,298/28,317/1,536 | https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles (card, v26.11) | 11-05 | doc-stated | S2.5 | Medium (card only) | Coarse clustering at best after the license |
| At least 828 of 2,364 public NuRec scenes and at least 666 of 2,202 curated scenes are not `alpasim`-labelled | Derived from the CSV counts and the PAI-AV card | 11-05 | derived | S2.5 | Conditional: UUID = clip_id with one label each | The contamination audit is mandatory |
| `hf_revision` is a branch name; download verifies only the `metadata.yaml` UUID; cached files are reused without a hash | https://github.com/NVlabs/alpasim/blob/297b4d00cfdd2c4f636b9d9bcee7a85c699174c0/src/wizard/alpasim_wizard/scenes/sceneset.py#L445-L505 | 11-05 | source-inspected | S2.1, S4.6 | Low | Candidate 2 |
| All 26.04 USDZ files were replaced on 2026-07-30; the 26.04 tree shows commit 5fe7a149; `main` = 26.10 with 1,601 scenes | https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec (card); …/tree/26.04 | 11-05 | doc-stated | S2.1 | Medium | Candidate 2 |
| The stock `main` simulate path sets no `RolloutSpec.random_seed`, so the session seed is `random.randint`; the same seed goes to driver and traffic | https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/runtime/alpasim_runtime/simulate/__main__.py#L145-L195 ; https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/runtime/alpasim_runtime/services/driver_service.py#L76-L78 ; https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/runtime/alpasim_runtime/daemon/engine.py#L200-L213 | 11-05 | source-inspected | S2.4 | Low | `force_determinism` alone does not seed a stock run |
| Reseeding to session.seed + inference_count only under `force_determinism` | https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/driver/src/alpasim_driver/main.py#L713-L714 ; https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/driver/src/alpasim_driver/models/alpamayo_base.py#L612-L615 | 11-05 | source-inspected | S2.4 | Low | Common random numbers are possible only through a custom request |
| Two same-seed checkpoints share draws until their inputs diverge | Derived from the two rows above | 11-05 | derived | S4.6 | Medium | Coupling reduces v̄ (see §6) |
| Challenge retries only renderer infrastructure errors, once, with a fresh UUID | https://github.com/NVlabs/alpasim/blob/297b4d00cfdd2c4f636b9d9bcee7a85c699174c0/src/runtime/alpasim_runtime/daemon/scheduler.py#L186-L213 ; https://github.com/NVlabs/alpasim/blob/297b4d00cfdd2c4f636b9d9bcee7a85c699174c0/src/runtime/alpasim_runtime/worker/main.py#L163-L186 ; https://github.com/NVlabs/alpasim/blob/297b4d00cfdd2c4f636b9d9bcee7a85c699174c0/src/runtime/alpasim_runtime/worker/ipc.py#L47-L50 | 11-05 | source-inspected | S2.3 | Low | The first attempt is visible only in the logs |
| `main` retries everything except INVALID_SCENE, up to 2 times, keeping the seed; it reuses the UUID and deletes the failed directory only when the request supplied UUIDs | https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/runtime/alpasim_runtime/daemon/scheduler.py#L43-L70 ; https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/runtime/alpasim_runtime/daemon/scheduler.py#L843-L875 | 11-05 | source-inspected | S2.3, S4.3 | Low | Retried policy crashes can disappear from the summary |
| Challenge `DEFAULT_MODIFIERS` lacks the black-image trajectory drop (removed by #109); `main` and Gym's pin keep it | https://github.com/NVlabs/alpasim/blob/297b4d00cfdd2c4f636b9d9bcee7a85c699174c0/src/eval/src/eval/aggregation/processing.py#L37-L64 ; https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/eval/src/eval/aggregation/processing.py#L64 ; https://github.com/NVlabs/alpasim/blob/46b401beaec27977cbb86826df8e39cec2d49e49/src/eval/src/eval/aggregation/processing.py#L64 ; https://github.com/NVlabs/alpasim/pull/109 | 11-05 | source-inspected | S2.5, S4.3 | Low | Candidate 1 |
| `ImageScorer` is registered; `ScorerGroup` swallows scorer exceptions; the per-rollout `metrics` holds every time-aggregated column | https://github.com/NVlabs/alpasim/blob/297b4d00cfdd2c4f636b9d9bcee7a85c699174c0/src/eval/src/eval/scorers/__init__.py ; https://github.com/NVlabs/alpasim/blob/297b4d00cfdd2c4f636b9d9bcee7a85c699174c0/src/eval/src/eval/scorers/base.py#L26-L45 ; https://github.com/NVlabs/alpasim/blob/297b4d00cfdd2c4f636b9d9bcee7a85c699174c0/src/eval/src/eval/scorers/image.py ; https://github.com/NVlabs/alpasim/blob/297b4d00cfdd2c4f636b9d9bcee7a85c699174c0/src/eval/src/eval/aggregation/processing.py#L291-L299 | 11-05 | source-inspected | S2.3 | Low | Absent metrics are possible; fail closed |
| Absent offroad is filled with 0.0 | https://github.com/NVlabs/alpasim/blob/297b4d00cfdd2c4f636b9d9bcee7a85c699174c0/src/eval/src/eval/aggregation/processing.py#L750-L759 | 11-05 | source-inspected | S2.3 | Low | Keep `required_metrics_verified` |
| Collision class is kept from the last actor only (the same loop is on the pin) | https://github.com/NVlabs/alpasim/blob/297b4d00cfdd2c4f636b9d9bcee7a85c699174c0/src/eval/src/eval/scorers/collision.py#L45-L56 ; https://github.com/NVlabs/alpasim/pull/234 ; https://github.com/NVlabs/alpasim/issues/237 | 11-05 | source-inspected / doc-stated | S2.5 | Low (rate unknown) | Local rescoring sensitivity; no competing PR |
| Deviation definitions, MAX reductions, the corridor at 4.0, the three truncations | https://github.com/NVlabs/alpasim/blob/297b4d00cfdd2c4f636b9d9bcee7a85c699174c0/src/eval/src/eval/scorers/ground_truth.py ; https://github.com/NVlabs/alpasim/blob/297b4d00cfdd2c4f636b9d9bcee7a85c699174c0/src/eval/src/eval/aggregation/main.py#L155-L160 ; https://github.com/NVlabs/alpasim/blob/297b4d00cfdd2c4f636b9d9bcee7a85c699174c0/src/wizard/configs/base_config.yaml#L354-L356 | 11-05 | source-inspected | S2.5 | Low | The bins need an untruncated extractor and a registered time support |
| The gate counts dev ≥ 4 per arm and keeps the rows in d | Local `gate/verdict.py` (locked; read only) | 11-05 | source-inspected | S2/S4 | Low | Report a paired exit table separately |
| Harmonizer: diffusion model with stateless and temporal modes; no determinism statement; temporal mode since 26.08; flag rename in 26.06 | https://docs.nvidia.com/nurec/nurec/harmonizer.html (updated 2026-10-30); https://docs.nvidia.com/nurec/release-notes/index.html | 11-05 | doc-stated | S2.2/S2.4 | Medium; the tag-to-release mapping is unverified | The fixed-pose probe must test Harmonizer repeatability |
| No NuRec statement of novel-view validity distance; renders not bit-identical across versions; a nonfinite pose gives a smeared render at status OK (26.10 known issue) | https://docs.nvidia.com/nurec/release-notes/index.html | 11-05 | doc-stated | S2.5 | Medium | The 4 m corridor is a convention; pin the renderer digest |
| Equivalent-pair UB ≈ 2√(2m₊ ln(2/α)/R); Tables A–C | Local gate functions via `scripts/power_tables.py` → `power_out/power_tables.json` | 11-05 | executed-CPU + derived | S2.6 | Model inputs are illustrative | The unseeded control is likely INSUFFICIENT at δ ≤ 0.05 |
| Adverse imputation consumes about 2 × the missing rate of the margin | `scripts/missing_sensitivity.py` → `power_out/missing_sensitivity.json` | 11-05 | executed-CPU | S2.3/S2.6 | The missingness model is illustrative | Register a missing-data rule before outcomes |
| Driver OOM can become a rollout failure; renderer OOM is retried | https://github.com/NVlabs/alpasim/issues/222#issuecomment-6342124367 plus the challenge worker source | 11-05 | doc-stated + source-inspected | S2.3 | Low | Policy-dependent missingness |
| Offered PRs and issues not to compete with: #234, #235, #237, #238 | https://github.com/NVlabs/alpasim/pull/234 ; https://github.com/NVlabs/alpasim/pull/235 ; https://github.com/NVlabs/alpasim/issues/237 ; https://github.com/NVlabs/alpasim/pull/238 | 11-05 | source-inspected | Upstream policy | Low | Record only |
| The default PyTorch generator seed may make "unseeded" streams partly repeatable | PyTorch default-generator behaviour (not checked in this driver) | — | hypothesis | S2.4 | High | Test with the fixed-input probe |
