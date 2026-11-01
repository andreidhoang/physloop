# Paired road clustered release verdict layer for AlpaSim

Draft only, not submitted. Re-check NVIDIA's current release and offered work before posting. No novelty or real-policy performance claim is made.

We have a CPU-only prototype that admits paired rollout records and returns GO, NO-GO or INSUFFICIENT across predeclared strata. It preserves road clusters and unseeded replicate identities, records out-of-envelope observations separately, and emits reproducible synthetic qualification artifacts.

The authorization method uses conservative bounded-road inference with a predeclared finite-look error budget. The target is an equal-road population mean; independence between roads remains an explicit sampling assumption. Tango, road-bootstrap and Gaussian sequential calculations are separate diagnostic/planning paths. We have not yet run this layer against AlpaSim policy outputs.

Would a thin adapter around the current per-rollout evaluation records be useful, and which current evaluator schema and road/group identity should it target? We would like to agree on a compatible seam before proposing code. The adapter would preserve scorer definitions and avoid duplicating any offered upstream implementation.

Synthetic tests and configuration/source manifests can be shared after repository publication. No licensed dataset-derived results are included in this draft.
