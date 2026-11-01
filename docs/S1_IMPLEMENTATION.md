# S1 gate implementation contract

This is the agent-authored S1/G1 implementation specification under Huy's 2026-10-31 instruction to complete S1 autonomously. It does not establish Huy's independent mastery or authorize S2 spending. The original human prediction worksheet is preserved; the autonomous experiment has its own preregistration.

## Decision and estimand

Positive difference means candidate B is worse than reference A. Binary failure is 0/1. Continuous outcomes must be bounded in [0,1] with a declared direction. Within a scene, average paired replicate differences; within a road, average scene differences; within a stratum, average road differences. The estimand is therefore an equal-road, equal-scene-within-road, equal-replicate-within-scene expected difference. It is not a scene-weighted pooled rate when road sizes differ.

Roads are independent sampling units with a stable population mean within each stratum. Dependence within a road is unrestricted. Road identities, strata, metric, margin, replicate schedule and look order are fixed before outcomes. A road cannot cross strata. New replicates cannot be appended to a road after it has entered a sequential look. No selection by observed outcomes is allowed.

GO requires a one-sided upper bound strictly below the positive non-inferiority margin in every required stratum. The null for false-GO is that at least one true stratum difference is greater than or equal to the margin. Equal policies with a positive margin are an acceptable alternative, not that null. NO-GO requires a Holm-adjusted positive-regression flag against difference zero. Regression flags take precedence, then missing or undecided strata produce INSUFFICIENT, otherwise GO. A small statistically detected regression can therefore be NO-GO even if inside the NI margin; these are two deliberately separate requirements.

## Conservative authorization bound

For road difference d in [-1,1], let X=max(d,0), Y=max(-d,0). Invert the bounded-variable Bernoulli KL Chernoff inequality separately for EX and EY at tail alpha/2. Udelta=UX-LY and Ldelta=LX-UY each have one-sided error at most alpha. Their pair is not a two-sided 1-alpha confidence interval. Add an outward numerical guard of 1e-12; never use a normal approximation in this authorization bound.

Regression uses the valid Hoeffding p-value min(1, exp(-R*max(mean(d),0)^2/2)) for R independent roads and the null E[d]<=0. Holm applies to the full registered stratum family, including inactive strata with p=1. Both error guarantees are separate: the false-GO budget is alpha and the regression-family budget is alpha, not a combined alpha guarantee for all error types.

This conservative verifier is an explicit S1 engineering amendment to the planned Tango/bootstrap authorization path. Tango paired-score intervals remain an independent-pairs reference; road bootstrap intervals remain diagnostics. Neither overrides the finite-sample bound. The price is reduced power, which must be reported rather than disguised. The proof and restrictions are in S1_CORRECTNESS.md.

## Input admission

Versioned protocol identity includes seed mode, configuration identity, evaluator, score definition and deviation convention. Checkpoint identity may differ. Duplicate observations, mismatched protocols, unmatched arms, incomplete rollouts, nonfinite outcomes, missing identifiers, inconsistent scene metadata, mixed outcome representations and crossed roads are errors before inference. Required absent strata are retained as INSUFFICIENT. Pairing by scene/replicate does not imply common random numbers. Seed coupling may be claimed only in an explicitly seeded protocol.

The trusted low-level verdict function accepts already-paired rows; the public CLI admits raw arm records through the contract layer. Counts distinguish rows, scenes and roads. Every reported difference names its direction and estimand. Deviations >=4 m are counted separately by arm; their outcomes remain included under the declared include-and-report convention. This is bookkeeping on synthetic data, not evidence of simulator validity.

## Sequential execution

The operational gate uses predeclared finite looks with nonnegative alpha weights summing to one. Default K=3 weights are (0.2,0.3,0.5); K=2 is (0.4,0.6). These are union-bound alpha spending, not canonical Gaussian O'Brien-Fleming boundaries. Each look uses cumulative complete roads in its registered batches and its allocated alpha. A stratum crossing efficacy stops receiving observations. Every new batch covers each still-active stratum. Terminal verdicts and closed strata reject further data. A futility schedule is predeclared; the registered experiment uses nonbinding no-early-futility, with undecided strata INSUFFICIENT at the last look. A configured futility crossing ends the whole decision as INSUFFICIENT unless a regression flag already gives NO-GO.

Per-look spent-alpha bounds remain valid after stopping by a union bound. They are explicitly not stage-wise-ordering intervals. Classical K=2/3 O'Brien-Fleming, Pocock and Lan-DeMets tables are separate Gaussian planning references; their expected sample sizes do not forecast the conservative gate. This distinction must appear in every power report.

## Acceptance

1. All original known answers, input contracts and adversarial decision/state tests pass without loosening expected values.
2. Exhaustive small-support probability checks support the finite-sample derivation; they are regression tests, not a substitute for the proof.
3. Each preregistered calibration cell has 1,000 trials, raw records, and exact binomial uncertainty. The simultaneous Monte Carlo acceptance rule is in the experiment file.
4. Both fixed and sequential harm-boundary cells control false-GO; equivalent and gross-regression controls show the instrument is not always INSUFFICIENT. Removing stratification exposes a rare-stratum loss hidden by a pooled question.
5. Power tables cover binary, continuous and rate-ratio planning, replicate correlation, IUT, and all prescribed canonical sequential designs. Operational gate power is separately measured.
6. An append-only evidence bundle records source hashes, lockfile, config, preregistration, trials and results. The package and report reproduce on CPU. No dataset, model, rented resource, publication or upstream message is part of execution.

VI: Mỗi road là một đơn vị bằng chứng độc lập; nhiều scene cùng road không làm tăng số đơn vị độc lập. Chứng minh áp dụng cho mô hình lấy mẫu này. Monte Carlo kiểm tra triển khai trên các trường hợp đã đăng ký, không chứng minh mọi dữ liệu tương lai.
