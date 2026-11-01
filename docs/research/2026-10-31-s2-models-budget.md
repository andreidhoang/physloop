# S2 preparation: public models, research ownership and budget choices

Checked 2026-10-31. This is a research note and recommendation, not an adopted budget, protocol, model change or expansion of ownership. Current spend authorization remains **$0**. Only public pages and metadata were read; no account, license acceptance, asset download or rental action was taken. The task serves S2/G1, G2, G3·R1 and G7 preparation; S4–S5 references below describe the existing research path, not work started today.

## Public artifacts: released weights and proposed recipes are different inputs

| Artifact | Observed state on 2026-10-31 | Implication for PhysLoop |
|---|---|---|
| [Alpamayo 1.5 10B](https://huggingface.co/nvidia/Alpamayo-1.5-10B) | Public model repository; model card identifies OpenMDW-1.1 for weights and Apache 2.0 for code. Metadata revision `7aba8293c09993f2e125c6819df05d7fa3e873ea`. | Keep the adopted S2 teacher. Record the chosen revision and actual terms in the run manifest. Model terms do not settle the separate PAI-AV/NuRec publication gates. |
| [Alpamayo 2 Super](https://huggingface.co/nvidia/Alpamayo2-Super) | Public 34B model; card gives release date 2026-08-04. Metadata revision `00554695e729a6ff0b6281fd2c81b18d06e33dbe`, OpenMDW-1.1. | A newer public model exists; replacing the S2 teacher would change the experiment and is not adopted here. |
| Alpamayo 2 Edge | NVIDIA's [product page](https://www.nvidia.com/en-us/solutions/autonomous-vehicles/alpamayo/) names a 2B Edge model. The [public NVIDIA model catalog query](https://huggingface.co/api/models?author=nvidia&search=Alpamayo&limit=100&full=true) returned only 1.5-10B, R1-10B and Alpamayo2-Super. | **Public Edge weights and an evaluation-permitting license were not verified.** This is an observation about the checked sources, not proof that no artifact exists elsewhere. KC-E is not passed by the marketing announcement; repeat the check at S3 admission. |
| [Recipes PR #42](https://github.com/NVlabs/alpamayo-recipes/pull/42) | Open, unmerged; head `5ade7d0982610db7b8c6fd886df2ed6206ef6b60`, updated 2026-10-20. Distillation, SFT and RL recipe material for 2B models. | An available recipe is not released NVIDIA student weights. Preserve the adopted KC-E fallback; the pilot student remains P-017, not an automatic substitute. |
| [Recipes PR #44](https://github.com/NVlabs/alpamayo-recipes/pull/44) | Open, unmerged; head `4ba4dd5a5c55fda9491fc26aeef465c262e0f092`, updated 2026-10-29. Training measurement guidance. | Useful definitions for mean step time, paid GPU-hours and utilization; proposed documentation is not independently measured PhysLoop performance. |
| [Recipes PR #45](https://github.com/NVlabs/alpamayo-recipes/pull/45) | Open, unmerged; head `fbed0a70d5452bc50527e1c11ae6855fccbea58c`, updated 2026-10-21. 2B inference recipe with a dependency fork. | Record compatibility and device identity before any later reproduction. Its performance claim is not a PhysLoop result. |

The recipes [main commit](https://github.com/NVlabs/alpamayo-recipes/commit/2e907dab31e5353f42b2c8a2f79d7e39b6e0d42c) is dated 2026-10-16. [AlpaGym main](https://github.com/NVlabs/alpagym/commit/972d160eed0e23d388497851504a3a233fec5879) is dated 2026-07-21. Both repositories returned no GitHub Releases in the public releases API at this check. Repository `pushed_at` or `updated_at` timestamps must not be mistaken for a new main-branch release.

AlpaGym's [README at the checked commit](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/README.md) describes an early-development AlpaSim/Cosmos-RL harness, supports Alpamayo 1.5 10B, and provides a two-GPU smoke preset. Its Slurm examples need environment customization. Reuse this in the existing S4 stage; no S4 training or new model integration is part of S2 prep. NVIDIA's [expert graph PR #36](https://github.com/NVlabs/alpamayo1.5/pull/36) is merged; S2 still needs its own bitwise acceptance check before enabling it.

## Public prices: planning inputs, not booked offers

USD, excluding unverified tax, availability and ancillary charges. Edition, host permissions, region and storage compatibility still need a concrete quote.

| Provider/source | Displayed on-demand GPU price | Node arithmetic |
|---|---:|---:|
| [Runpod pricing](https://www.runpod.io/pricing), page dated 2026-10-08 | RTX PRO 6000 96GB: $2.49/GPU-h; H100 SXM 80GB: $3.99/GPU-h | Eight H100s would be $31.92/h arithmetically; no eight-GPU offer was verified. The RTX edition and selected cloud tier are not established by the extracted card. |
| [Nebius prices](https://nebius.com/prices) | RTX PRO 6000: $1.80/GPU-h; H100 HGX: $4.50/GPU-h | Eight H100s imply $36/h before extras; not a verified reservation. RTX Workstation versus Server edition is unspecified. |
| [Lambda pricing](https://lambda.ai/pricing), eight-GPU tab | H100 SXM 80GB: $3.99/GPU-h | Listed eight-GPU instance: $31.92/node-h. Included local SSD must not be assumed to be durable retained storage. |
| [Vast pricing](https://vast.ai/pricing) | Dynamic marketplace; no numeric live offer independently verified in this pass | Keep 2026-10-27 prices explicitly historical. Do not silently reuse them as today's quote. |

[Runpod network-volume documentation](https://docs.runpod.io/storage/network-volumes), modified 2026-10-30, charges the first 1,000 GB at $0.07/GB-month and the remainder at $0.05. Thus 2,900 GB costs approximately **$165/month**, before any working-copy storage. Volume charges continue while compute is stopped; Pod use requires a compatible Secure Cloud datacenter.

Nebius lists object storage at $0.0147/GiB-month plus $0.015/GiB object egress, non-replicated block at $0.053/GiB-month, and shared filesystem at $0.08/GiB-month. An illustrative 3,000 GiB costs $44.10, $159 or $240/month respectively; these are different storage services, not interchangeable quotes. Allocate working copies and retention separately. Spot minima are not reliable budget inputs. [Source: Nebius prices](https://nebius.com/prices).

## Proposed P-005 structure

The adopted S2 estimate is **136–340 GPU-hours** for four sweeps, before profiling and pilots (`02`, S2 Cost). Applying the newly checked $1.80–2.49 rates yields **$244.80–846.60 GPU-only**. This is an inference from the old throughput model, not a measured cost forecast. A full-release persistent volume can add roughly $165/month on the Runpod example above. Download time, setup, idle time, retries, storage duplicates, taxes and egress can materially increase the total.

Recommend preparing the following decision for Huy, with **no authorization inferred today**:

1. Project cloud tier: **at most $2,500**, with hardware separately decided. This is a ceiling, not a forecast that every later stage fits.
2. S2 all-in envelope: **at most $1,200**, included inside that project tier. Include compute, storage, egress, tax and retained-storage liability. Stop or redesign if the pilot forecast exceeds this envelope.
3. First release: **at most $100 all-in**, nested inside the $1,200 S2 and $2,500 project ceilings. Compute remains bounded by **8 GPU-hours at at most $2.50/GPU-hour ($20)**. The remaining **$80 is an upper contingency for setup, storage, transfer and tax**, because full-host requirements and data startup may dominate the small pilot. Unused allowance is not a spending target and cannot automatically extend compute. Use the smallest licensed subset needed for the declared pilot, and obtain a concrete compatible-host quote before renting. If no provider meets these boundaries, revise the decision before booking.
4. Keep the rest of S2 unreleased until the pilot supplies measured seconds/step, completed-rollout goodput, retries, paid idle time and storage needs. Calculate remaining GPU-hours from the actual accepted scene list and replica count. Record a hard timeout, shutdown owner, storage deletion/retention date and remaining project balance.

The arithmetic is a proposed spending boundary, not an experimental prediction or a tolerance. All prediction slots and the S2 admission protocol remain separate decisions. H100 nodes are future S4/S5 options; no H100 purchase or rental follows from listing their prices. The $6,000 tier, Slurm demonstration, fault injection, batching and other optional proposals are not adopted by this note.

## Research ownership and essential teach-back

For the requested AV model-research focus, Huy should lead the causal question, model intervention, training objective, RL/distillation/correction loss, ablation choice, prediction and interpretation. The existing S4 teacher-RL and S5 on-policy student correction stages are the natural place for that work. Preserve S2 as the instrument that can distinguish an intervention from simulator noise. Do not replace it with a succession of model demos or start training before it qualifies.

Agents can carry the repeatable engineering load within explicit authorization: manifests and lineage, data joins and leakage audits, run orchestration, retry accounting, deterministic replay tests, known-answer fixtures, plots, cost accounting and independent statistical/code review. Core ownership remains governed by the current contract until a specific amendment is recorded. Agent-authored derivations or code are labeled as such and do not establish Huy's mastery.

Before approving model experiments, Huy should be able to explain these using one concrete PhysLoop example each:

- The inference target and unit: scene-weighted and equal-road results answer different questions; many replicates do not create many independent roads.
- Why an observed improvement, a non-inferiority GO and INSUFFICIENT differ; why a finite-scene or correlated sample may not satisfy the S1 theorem.
- What the seed actually controls, how pairing is established, and how ≥4 m rollouts, retries and missing outcomes enter the estimand.
- Why on-policy state visitation changes distillation, which term optimizes the desired behavior, and which reward-hacking failure an ablation is designed to expose.
- Which evidence could falsify the hypothesis and what result would justify the next spend.

The result is attributable independent research with clear human and agent contributions. It does not establish employment at NVIDIA, production-scale ownership or real-world driving validity.
