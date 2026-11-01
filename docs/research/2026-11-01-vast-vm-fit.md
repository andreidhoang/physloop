# Vast.ai and the control S2 actually needs

**Dated supersession — 2026-11-01, after the public-source assessment below.** D-019 subsequently authorized and executed a bounded public-software/synthetic test on Vast instance `58628722`. The original no-rental/$0/unverified-host status below is historical; actual charges and cleanup verification belong to [the measured host-validation report](2026-11-01-vast-host-validation.md), with raw records in [`evidence/s2-vast-host-20261101-v1/`](../../evidence/s2-vast-host-20261101-v1/). The original offer snapshot and pricing remain unchanged.

The delivered KVM Ubuntu 22.04 guest provided root, RTX PRO 6000 Blackwell Workstation Edition (97,887 MiB VRAM), and driver 580.95.05. Guest RAM was 96.996 GiB versus the listing's 123.438 GiB. Actual CUDA integer/repeat/negative controls, approximately 70.82 GiB VRAM verification and buildx/Compose network/mount/guest-host-PID/GPU checks passed. Nsight Compute read kernel counters, Nsight Systems captured three CUDA kernels, and DCGM 4.7.1 tensor-active responded to the exact-answer cuBLAS control (idle → approximately 0.576 → idle). Exporter 4.8.4 with its separate bundled DCGM 4.6.1 embedded engine produced responsive loopback-only scrapes.

DCGM XID remained N/A/absent. NVML reported event mask `0xf19c`, accepted subscription, and timed out waiting; event delivery/alerting remains unverified, not zero errors. This establishes a tested Vast host route only. Nebius is an untested fallback; no permanent licensed-provider adoption follows. Scientific S2, full renderer/model/Harmonizer integration and simultaneous memory fit, future offer availability, sustained reliability and dataset/provider access remain unadmitted or unverified. Public software terms, including CUDA's, applied; no gated-asset acceptance/download occurred.

**Original assessment follows (public snapshot at 13:29 ICT, before D-019 and the paid host check).** Its earlier recommendation and unverified list are superseded only by the scoped observations above.

Checked 2026-11-01, Asia/Ho_Chi_Minh. Agent-authored S2 preparation for G1, G2, G3·R1 and G7: `02_PHYSLOOP_PLAN.md` S2 [C] run method/topology, runtime profile and monitoring (lines 278–315). Public-source analysis only; no scientific experiment, rental, account funding, license acceptance or asset download. D-018 approvals and the blocked pilot/full drafts are unchanged.

**Recommendation: Vast.ai is eligible through its full-VM offering. For the first S2 pilot, retain the prepared Nebius VM as the reference option and compare an actual Vast VM offer against it. Do not choose an ordinary Vast container just because its GPU is cheaper.** A public search found five matching VM-enabled 96 GB Workstation offers at 13:29 ICT; none has passed our host/license/runtime checks. A Secure Cloud/VM intersection and account checkout remain unverified. This is a provider-fit assessment, not admission.

## 1. There are two different Vast products

Vast's dedicated [VM guide](https://docs.vast.ai/guides/instances/virtual-machines) documents full Linux instances with systemd, Docker and process tracing. Select its VM template and `vms_enabled=true`; VM images use `docker.io/vastai/kvm`. SSH keys must be registered before launch. This route has fewer eligible hosts, longer startup and more disk overhead. Its [host guide](https://docs.vast.ai/host/vms) establishes KVM and explains that VM support depends on hardware/IOMMU configuration.

The ordinary [Docker offering](https://docs.vast.ai/guides/instances/docker-environment) provides a container on the provider's kernel. Its allocated GPU is exclusive while running, but CPU/RAM allocations can share the physical machine. Container root does not confer control of the host Docker daemon or kernel. Renting every GPU does not itself change this privilege boundary.

Vast's [technical FAQ](https://docs.vast.ai/guides/reference/faq/technical) explicitly disables Docker-in-Docker for the ordinary container offering. A custom model image is possible there; the current multi-container simulator still needs its orchestration environment.

There is stale contradictory text: the generic [instance-management FAQ](https://docs.vast.ai/guides/instances/manage-instances) still says VMs are planned. The dedicated VM pages, concrete templates and [official December 2024 launch](https://vast.ai/article/announcing-virtual-machine-rental-on-vast-ai) establish an implemented VM product. This is not grounds to assert that every GPU listing supports it. No current alpha label was established from the dedicated pages inspected.

## 2. Control follows the workload

| Work | Needed control | S2 decision |
|---|---|---|
| Local gate, manifest checks, synthetic tests | Python environment | Already CPU-only; no rental |
| A standalone model forward/backward job | GPU-enabled process/container with adequate memory and libraries | A normal GPU container can be sufficient; this does not qualify the simulator |
| Current AlpaSim deployment | Docker Engine/Compose/buildx, NVIDIA runtime, networks/mounts and service lifecycle | Guest-admin Linux VM is the practical baseline |
| Nsight/DCGM measurement | Actual tracing/counter support, permissions and GPU/driver compatibility | Probe separately; a VM label is insufficient |
| One-node Slurm/Pyxis | Scheduler/services and required namespace/runtime support | Optional S2 item; not required for this pilot |
| Physical host firmware/hypervisor administration | Provider-level control | Not required for the current S2 design |

The pinned [AlpaSim onboarding](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/docs/ONBOARDING.md) requires Docker with Compose/buildx, NVIDIA Container Toolkit and a CUDA-12.8-compatible driver (570.x or later). The simulator coordinates renderer, policy driver and other services. A working PyTorch notebook proves none of that integration. The renderer's [26.04 hardware requirements](https://docs.nvidia.com/nurec/archives/26.04/basics/hardware.html) are more specific: **R580 or later for Blackwell**. Its supported list also includes A100/H100, so do not invent a mandatory Vulkan/display/RTX-core requirement for NuRec. This assessment keeps the proposed RTX GPU class; it does not approve a substitute topology.

The pinned [Compose generator](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/alpasim_wizard/deployment/docker_compose.py#L195-L213) assigns the runtime `pid: host`. In a VM this means the guest Docker host, not visibility into the provider's physical hypervisor.

**Meaning of full control here:** administrator access to the guest operating system, its Docker daemon, service configuration, mounts and supported driver settings. This does not mean control of the physical host, firmware, power delivery or the provider's hypervisor. Bare metal is an alternative, not an S2 prerequisite. Nor is a VM a mathematical prerequisite for all possible AlpaSim deployments: a provider-managed compatible orchestration service could suffice. Repackaging services into a single unprivileged container or rebuilding orchestration would add unqualified engineering and is not the proposed pilot.

GPU counters have another boundary. NVIDIA documents [profiling permissions](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters); container use may need permissions granted by the host. [DCGM profiling](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html) requires supported fields and privileges, and can conflict with developer profilers. Inspect the delivered GPU's supported metrics and run a small probe. Record unavailable counters as unavailable, never zero. Required missing measurement must be fixed or explicitly amended before claiming S2 complete. Guest root cannot manufacture an unsupported GPU feature.

## 3. Vast offer admission, before and after payment

Before presenting a Vast order for approval, attach:

1. **Identity:** offer, host and machine IDs, exact region/operator, VM template/image identity and VM capability; one RTX PRO 6000 Blackwell **96 GB**, with Workstation/Max-Q/Server edition resolved. The old 48 GB RTX 6000 Ada is not an equivalent offer.
2. **Resources:** guaranteed guest CPU/RAM, disk capacity/throughput, network limits and contract duration. The existing 500 GiB pilot allowance is an estimate, not a requirement to download the whole dataset. CPU/RAM minimums must follow the selected topology and smoke measurement; do not invent a proven minimum from a listing's peak capacity.
3. **Operation:** on-demand for initial qualification, sufficient uninterrupted contract time, external stop monitor, evidence export and storage deletion plan. No spot/interruptible or reserved commitment is proposed for this first pilot.
4. **Trust and terms:** named operator and access/retention arrangement. Prefer a verifiable datacenter/Secure Cloud offer for licensed data; a performance-verification badge is not proof of contractual confidentiality. Review the actual provider terms and dataset permissions before transfer.
5. **Cost:** itemized compute, disk while running/stopped, actual VM transfer tariff, tax/FX, credit purchase and automatic-payment behavior. Keep the adopted $100 all-in pilot ceiling; ≤8 GPU-hours and ≤$2.50/GPU-hour remain proposed operational limits requiring the concrete order approval.

Only inside an approved bootstrap allocation, establish guest administration, GPU identity/exclusivity, driver and CUDA operation, GPU-enabled Compose service communication, required telemetry/profiling access, private mounts, cleanup/cutoff operation, and the real renderer/model memory fit. Authorized renderer/data checks follow their license and stage gates. **One 96 GB GPU fitting BF16 teacher + renderer + Harmonizer remains unmeasured on both providers.** Stop and revise the quote/topology on failure; do not silently lower precision or add GPUs.

Vast explicitly says [host security varies](https://docs.vast.ai/guides/reference/faq/security). VM isolation addresses other tenants; it is not proof that a provider cannot access the guest or stored data. The [NuRec agreement](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) imposes confidentiality, authorized-access and hosting/distribution restrictions. Neither Vast's Secure Cloud badge nor Nebius's brand resolves their application to our private rented environment. Review actual access, support, copies and retention; clarify unresolved permission before transfer. Encryption at rest alone does not protect decrypted computation from a privileged host. This does not introduce a blanket ban on rented GPUs.

## 4. Compare cost per usable result

Use `all-in cost / completed, valid, protocol-matching rollouts`, with failures/retries retained in the report. This is a procurement comparison, not permission to exclude invalid rollouts from scientific analysis. Include setup, idle time, retries and migration/requalification in the forecast. A cheaper GPU-hour can buy fewer usable results if CPU, disk, memory, availability or instrumentation constrains the pipeline.

The unauthenticated [public offer search](https://docs.vast.ai/api-reference/hello-world#3-search-for-gpus) at **2026-11-01 06:29:56 UTC / 13:29:56 ICT** returned five verified/rentable, one-GPU `RTX PRO 6000 WS` offers with `vms_enabled=true`, requesting 500 GB storage; `truncated=false`. The [saved response](../../evidence/s2-vast-public-offers-20261101T062955Z/response.json), query, retrieval metadata, hashes and [derived arithmetic](../../evidence/s2-vast-public-offers-20261101T062955Z/derived_costs.json) preserve the snapshot. The search is not a reservation or GPU runtime test.

| Candidate | Compute/hour | Allocated storage/hour | 8h compute + 8h storage | 8h compute + 168h storage |
|---|---:|---:|---:|---:|
| Vast South Korea, offer 56403068 / machine 140630, Workstation 96 GB, 500 GB disk | $1.4013 | $0.1389 | $12.32 | $34.54 |
| Vast Czechia, offer 58291364 / machine 155673, Workstation 96 GB, 500 GB disk | $2.4000 | $0.1389 | $20.31 | $42.53 |
| Prepared Nebius Server Edition 96 GB, 500 GiB disk, public tariff | $1.8000 | $0.0486 | $14.79 | $22.57 |

These are base-cost scenarios, **not all-in quotes or identical hardware/storage configurations**: Vast reports GB and Nebius GiB; GPU editions and CPU/RAM differ. Excludes transfer, taxes, FX and other unverified fees; assumes each quoted disk tariff persists during retention. Stopped-storage tariffs must be confirmed. Nebius CPU/RAM inclusion and capacity remain account-quote checks. [Nebius pricing](https://docs.nebius.com/compute/resources/pricing).

The Korean listing reports 16 effective CPUs, 126,400 MiB system RAM and driver 580.178.04; these are listing claims, not verified guest configuration. The Czech candidate also appeared in the reviewer's `datacenter=true` search, but that does not establish Secure Cloud certification or our license/access requirements. Its reported reliability value is 0.9391864; do not interpret a marketplace score as the probability that our pilot will finish. Both have nonzero transfer-rate fields. No quoted offer is approved for rental.

The cheapest snapshot is only about **$2.47 less for eight hours with immediate cleanup**, before differing fees; with seven-day retention its base scenario is about **$11.97 more**. Over longer runs the compute difference can matter, but throughput, availability and repeatability must be measured before choosing a full-sweep host.

Vast [prices are offer-specific](https://docs.vast.ai/guides/instances/pricing). Its host VM page says hosts do not earn VM bandwidth revenue, while generic billing describes transfer fees: obtain the VM-specific checkout tariff rather than assuming either universal free transfer or universal transfer charges. Do not reuse the plan's historical marketplace GPU range as a current VM quote.

[Vast billing](https://docs.vast.ai/guides/reference/billing) requires prepaid credits; storage continues after stop, and saved cards/automatic payments can cover balances. A prepaid balance is not a sufficient hard cap. Provider-side stop must be verified; export and verify allowed evidence before destroying storage. Preserve billing records before destroying the instance. Never use credit exhaustion as the stop mechanism.

For the first short pilot, choose the compatible, accountable offer that minimizes time to trustworthy measurements. If Vast passes the documented checks and has a meaningfully lower all-in forecast, use Vast. If the eligible host or access conditions cannot be established, use the prepared Nebius option after the same real runtime checks. No platform is qualified merely by its documentation.

## 5. Research implications

A model's training math generally does not require administering an entire VM. The reason S2 benefits from it is the surrounding simulation and measurement stack. Huy's model/loss/reward work should remain portable while agents own evaluator operations under D-018. There is no need to build Kubernetes, become a cloud-platform engineer or rent a large cluster for this pilot.

After S2, a standalone training job may use a cheaper container offering, but closed-loop RL includes simulator interactions and must be sized accordingly when that stage is admitted. Single-GPU success cannot establish multi-GPU suitability: Vast's host VM documentation warns of GPU peer-communication constraints under IOMMU. Later training needs its own measured topology/communication check.

For comparative S2 runs, keep the registered environment/hardware and run method consistent across policy arms. Host changes can change timing and unseeded request order. A migration therefore needs recorded environment differences and repeatability/drift checks before reusing old results; never pool unlike-device latency as if it were one measurement.

**Unverified:** availability after the snapshot and account checkout, operator/license arrangement, delivered GPU edition/passthrough, guest resource allocation, driver/rendering/counter support, simultaneous memory fit, runtime reliability and measured cost. This note establishes a feasible purchasing route and a qualification plan, not GPU test results.
