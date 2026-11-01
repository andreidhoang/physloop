# S2 pilot: license and access approval package — 2026-10-31

This public-source check serves S2/G1 preparation. Huy has adopted the project operating model and the $100 pilot / $1,200 S2 / $2,500 project caps; those caps do not themselves accept licenses, order a host or authorize a download. The rental proposal is in [the companion note](2026-10-31-s2-rental-approval.md). Root owns the live contract/manifest updates.

Only public license pages, documentation and small public source files were read. No account was used or created, no acceptance was submitted, no gated metadata/scene/weight/container was downloaded, and nobody was contacted.

## 1. Minimal pilot asset inventory

The pilot is five predetermined NuRec scenes, BF16 Alpamayo 1.5, Harmonizer off/on. It does not need TransFuser/LTF, FP8, raw PAI camera/lidar collections, Harmonizer training data or a second driving model.

| Asset and official access page | Owner action after approval | What access is for / what to record |
|---|---|---|
| [PAI-AV](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) | Sign in with Huy's HF account; read the displayed dataset agreement; use **“I accept the terms of the NVIDIA Autonomous Vehicle Dataset License Agreement”**. Public page data identifies automatic approval. | Metadata/split checks when required by the pilot selection/contamination plan; not an entire raw-data download. Record account, agreement hash, acceptance time and first actual delivery/download. |
| [PAI-AV NuRec](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) | Read its separate agreement; use **“I accept the terms of the NVIDIA Autonomous Vehicles NuRec Dataset License Agreement”**. Public page data identifies automatic approval. | Only the five selected scene artifacts and necessary metadata. Record exact scene/artifact IDs, paths, repository revisions, hashes and every storage location. |
| [Alpamayo 1.5 10B](https://huggingface.co/nvidia/Alpamayo-1.5-10B) | Review [the LICENSE at the planned revision](https://huggingface.co/nvidia/Alpamayo-1.5-10B/raw/7aba8293c09993f2e125c6819df05d7fa3e873ea/LICENSE); record Huy's acceptance of use terms. Current public page metadata says **not gated**, so no HF acceptance button is established. | BF16 teacher weights/config. OpenMDW-1.1 is verified in the actual LICENSE. Record revision `7aba8293c09993f2e125c6819df05d7fa3e873ea`, artifact hashes and source dependency commit. The card still has a non-commercial-use sentence; this internal research pilot does not test commercial rights. |
| [NRE renderer catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nre/containers/nre-ga) | Review the catalog's governing terms; use a free individual NGC account, then an expiring Catalog-access key. Do not select a paid enterprise subscription merely to obtain the public container. | Exact renderer `nvcr.io/nvidia/nre/nre-ga:26.04`; record accepted terms, tag and actual resolved digest. Catalog digest candidate is below; it has not been pulled. |
| [Harmonizer](https://huggingface.co/nvidia/Harmonizer) | Review its [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). Public page metadata says **not gated**; absence of a button does not remove use terms. | The renderer's on arm uses `harmonizer_nontemporal.pt`. Record immutable revision/hash and whether it was fetched or already cached. Do not download its training dataset. |
| Teacher processor/config dependencies | Resolve the selected driver/checkpoint's actual requests before approving extra assets. Current [helper source](https://github.com/NVlabs/alpamayo1.5/blob/36aeb4c5938cbc2eb2aed33b22434773da4ab639/src/alpamayo1_5/helper.py#L20-L27) uses [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) processor assets (Apache-2.0). | Processor/tokenizer/config files are distinct from full model weights. The [base-model loader](https://github.com/NVlabs/alpamayo1.5/blob/36aeb4c5938cbc2eb2aed33b22434773da4ab639/src/alpamayo1_5/models/base_model.py#L336-L390) also resolves the checkpoint's `vlm_name_or_path`. Its final dependency set is not yet frozen. |

NuRec's exact repository is established by [AlpaSim's constant](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/alpasim_wizard/scenes/csv_utils.py#L18) and [artifact loader](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/alpasim_wizard/scenes/sceneset.py#L444-L475). The loader uses a selected artifact path and revision, then validates its internal UUID. The [curated preset](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/configs/nurec_scenes/curated_val.yaml) is the whole 441-scene suite: do not run it as an implicit five-scene allowlist. A separate PAI raw-data corpus is not shown as a renderer input.

### Conditional Cosmos access — do not conflate two models

The [26.04 server documentation](https://docs.nvidia.com/nurec/archives/26.04/api/grpc_api_guide.html#command-line-parameters) specifies the default Harmonizer checkpoint URL as `https://huggingface.co/nvidia/Harmonizer/resolve/main/harmonizer_nontemporal.pt`. This documents a possible network fetch; it does not prove the asset is bundled in the selected image. Freeze that asset before the measured on/off comparison.

Harmonizer's backbone is **Cosmos-Predict2 0.6B**, not Cosmos-Reason2. The [standalone Harmonizer instructions](https://github.com/NVIDIA/harmonizer#download-pretrained-checkpoint) require [Cosmos-Predict2-0.6B-Text2Image](https://huggingface.co/nvidia/Cosmos-Predict2-0.6B-Text2Image). Whether its files are separately required by the existing NRE container remains unverified. Do not install the standalone pipeline just to satisfy an inferred dependency.

[Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B) remains gated. Access is relevant only if the resolved teacher loader actually requests its assets. The old AlpaSim tutorial recommends downloading both models, but reviewed loading code does not establish that a second full 8B weight download is necessary. For either conditional Cosmos gate, Huy reviews its displayed agreement, checks the age/authority acceptance box, then clicks **Submit**. Public page metadata identifies automatic approval. Add only the dependency actually needed to the approved allowlist.

## 2. What the current agreements establish

**Individual eligibility and purpose.** Both dataset agreements expressly include individuals; the accepting person must be an adult with authority. Their grant is for internal AV/ADAS development using NVIDIA technology. Metadata and annotations are included. No organizational-email requirement was found in these inspected asset gates; challenge registration is a separate process. [PAI-AV terms](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)

**Confidentiality and derivatives.** Dataset outputs and benchmark/regression/performance results fall within the confidentiality clause. Section 4.6 restricts dataset derivatives and making the dataset available to others. An open model license does not override these dataset restrictions. Keep KC-L and KC-L2 separate; acceptance alone does not pass either. [NuRec terms, §§3–4.6](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec)

**The clock.** Section 8 expires the agreement twelve months after initial delivery or download and requires stopping use/destruction on termination. Record acceptance separately from first delivery; a gated metadata download is still dataset delivery. Track each dataset's first date, copies and deletion deadline. [PAI-AV terms, §8](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)

**Private cloud is a concrete access question, not an automatic ban or permission.** NuRec §4.6 includes **“host the Dataset (in whole or in part)”** among transfer/distribution restrictions; §2 describes authorized-user access through a secure network, and §3 restricts disclosure. The text does not expressly resolve a solo researcher's private rented GPU VM. Assess the actual arrangement: Huy-controlled private storage, provider/support access, backups, retention and anyone receiving licensed inputs/outputs. Do not equate private computation with a public dataset-hosting service, or assume any cloud host is authorized. If the proposed access/use cannot be reconciled with the terms, obtain clarification before that transfer; this note does not invent a blanket written-approval requirement for every private VM. [NuRec agreement](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec)

Practical consequence for agent work: P-030 is project authorization, not NVIDIA permission to disclose confidential material. Keep licensed frames, source metadata and benchmark outputs out of public repositories, public tickets and external chat/model services unless the relevant access/disclosure arrangement is permitted. Local/private tooling can retain full evidence; status reports need not expose the protected payload.

**Renderer has separate terms.** The NRE catalog links the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/) (modified 2026-05-07) and the former Omniverse terms, now [AI Product Terms](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/) (2026-04-15). These recognize individual customers and owned/leased infrastructure. Software §8.9 separately restricts disclosure of performance/benchmark results; the linked [exception list](https://docs.nvidia.com/nvidia-containers-benchmarking.pdf) does not list NRE. Do not treat dataset KC-L clearance as renderer clearance automatically.

The current [Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) permits model use and derivatives subject to its conditions, including applicable notices/attribution. Conditional Cosmos gate texts display different version dates; archive the actual accepted text rather than substituting a newer global page. This pilot does not propose distributing models or derived datasets.

## 3. Click/order sequence for Huy

1. **Review this asset list and the rental proposal first.** Confirm accepting identity, private host/access plan, exact five-scene scope, retention and which conditional dependency is needed. Budget caps are already adopted; a concrete host/order and license acceptances remain separate approvals.
2. **Sign in to HF as Huy.** On each required gated page in the table, read and accept its own terms. Do not reuse another person's token. HF grants access to individual users; its [dataset access instructions](https://huggingface.co/docs/hub/datasets-gated#access-gated-datasets-as-a-user) require the browser request. Save the acceptance record without downloading data yet.
3. **Create a limited read credential** at [HF token settings](https://huggingface.co/settings/tokens), following [HF's token guidance](https://huggingface.co/docs/hub/security-tokens). Record scope/expiry only; put the value in the designated secret store, not chat, manifests or source control.
4. **Set up NGC access.** At [NGC](https://catalog.ngc.nvidia.com/), use the free individual-account flow. Under account settings → API Keys → Generate Personal Key, choose a short expiry and Catalog access. [Official NGC instructions](https://docs.nvidia.com/ngc/latest/ngc-user-guide.html#generating-a-personal-api-key) establish this flow. It is not a cloud GPU purchase.
Before paid GPU time, a small local metadata download may support scene selection only after Huy's relevant acceptance, the S2 start/date gates and permission for that delivery. Record its first-delivery clock; this is not permission for a full dataset download.

5. **After the host-order approval**, provision within the recorded cap. Authenticate the host to `nvcr.io` with Docker's documented `$oauthtoken` username and the secret key; do not expose the value. [Version-matched setup instructions](https://docs.nvidia.com/nurec/archives/26.04/basics/software.html) describe authentication.
6. **At admitted S2 bootstrap**, resolve/pin the runtime, pull the approved renderer, then fetch only the approved model/processor/Harmonizer assets and five scene artifacts. Record downloaded bytes, exact revisions, licenses and hashes. A newly discovered dependency gets reviewed before fetching; it is not implicitly covered by an unrelated acceptance.
7. **Before any licensed dataset transfer**, verify the actual host/access arrangement and applicable start/date gates. Record first delivery immediately. Complete the source/scene manifests and preflight before the ten measured policy rollouts. No publication follows automatically.

No steps above were performed in this audit.

## 4. Source identities available for the approval record

The HF hashes below are SHA-256 of the UTF-8 `extra_gated_prompt` string extracted from the public, logged-out page's JSON props on 2026-10-31. They identify the displayed legal text, not a dataset revision or an accepted agreement.

| Public text / image | Observed identity |
|---|---|
| PAI-AV displayed agreement | `146dbf7fb1dfc6774b3aab23b38061d04990131f8a61b28dbb4c1f42afdb7609` |
| NuRec displayed agreement | `f78c47117197259b801bcf7a1851cf5ff3ae3d04f77458d9fc8edfbe2a0973ae` |
| Alpamayo LICENSE at planned model revision | 2,619 bytes; `2ab44b68365473c112f5092211a38f231cb23e50de68b75a13369adbd76a74df`; matches the [license upload commit](https://huggingface.co/nvidia/Alpamayo-1.5-10B/commit/912a0f74846149e1f2aa45630b88b44ae34573c1) |
| Conditional Cosmos-Reason2 displayed agreement | 2025-09-23; `d4fb382bcb7ef4c775cca5f56b0ca494d4ebd13669067b9bb5f22aa3188d142d` |
| Conditional Cosmos-Predict2 displayed agreement | 2025-04-30; `898f690ea28cfce2493460958577fa78ddbf877cccd3a69079c401419a026ca0` |
| Harmonizer linked Open Model License | Global page version 2025-10-24; archive actual delivered component terms at bootstrap |
| NRE `26.04` catalog manifest candidate | Linux/amd64, `sha256:97f43e7130c5636ce3e80ea3184d97f56a87fdd989b05cce42230881dbdea284`; [public tag listing](https://catalog.ngc.nvidia.com/orgs/nvidia/nre/containers/nre-ga/-/tags) also assigns this digest to `26.04.01`. Not a pulled/verified local image. |

[NuRec 26.10](https://docs.nvidia.com/nurec/release-notes/index.html) introduces open-source code and renamed images. That newer release does not silently change the reviewed 26.04 protocol, bundled-component terms or dataset licenses.

The approval record should contain: accepting individual/entity, exact URL and terms/hash, acceptance timestamp/status, approved asset allowlist, private host and authorized-access description, secret-store references, first-delivery dates, locations/copies, expiry/deletion dates, and separate KC-L/KC-L2/renderer-publication status. Agent-prepared technical values are not a human acceptance receipt.

## 5. Recommended approval wording

Present the user with separate decisions, rather than one ambiguous authorization:

> Approve the listed internal S2 pilot asset/access plan for the five selected scenes and named components. I will personally accept the required provider/model/dataset terms under my own identity. Keep conditional model access excluded until its dependency is established. No dataset-derived results or artifacts may be published under this approval.

The rental decision should separately name the provider, exact host/region, first payment or deposit, rate and shutdown/retention caps from the companion proposal. Approval of the $100 ceiling alone is not approval of an unspecified order. Any clarification to NVIDIA is a separate draft/send decision; none has been sent.

Remaining unresolved items are concrete: the selected host's access/retention arrangement, actual user acceptances, five scene paths/revisions, teacher processor dependency resolution, NRE-bundled versus separately fetched Harmonizer dependencies, actual image/component bytes, and publication permissions. There is no verified organization-email obstacle for the listed asset gates.
