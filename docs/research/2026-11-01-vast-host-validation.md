# Vast S2 host validation — actual rental, tests and deletion

**Result: Vast's full KVM VM passed the tested S2 host requirements: GPU containers, Compose and GPU profiling/telemetry. The tested instance was stopped and destroyed; recorded usage was $0.60 from existing credit. Scientific S2 remains unadmitted.** This is an agent-authored engineering result under D-019, serving S2/G1, G2, G3·R1 and G7 (plan S2 [C], lines 278–315), using public software and synthetic inputs only.

The [acceptance contract](../../experiments/2026-11-01-vast-host-check.md) was written before the dependent checks. [Raw evidence](../../evidence/s2-vast-host-20261101-v1/) includes failed setup attempts, commands, sources, image identities, traces, metrics, billing and deletion receipts. No PAI-AV, NuRec, renderer, model weights or scientific rollout was downloaded/run. Public software terms, including the CUDA container's displayed license, apply; no project-asset license acceptance was performed.

## Selected offer and delivered resources

| Item | Observed value |
|---|---|
| Rental | Instance **58628722**, label `physloop-s2-host-20261101-v1` |
| Offer / machine / host | **56500949 / 140620 / 395870**, South Korea |
| GPU | **RTX PRO 6000 Blackwell Workstation Edition**, NVML 97,887 MiB; 600 W power limit |
| Virtualization / access | **KVM**, Ubuntu 22.04.5, kernel 6.8.0-87, guest UID 0 |
| Delivered CPU / RAM | **15 visible vCPUs; 104,149,004,288 bytes RAM = 96.996 GiB** |
| Instance disk | Requested numeric 160; filesystem 166,304,071,680 bytes; no separate volume |
| Delivered driver | **580.95.05**, versus listing 580.178.04 |
| Hourly quote | $1.406667 compute + $0.044444 disk = **$1.451111/hour**, before listed transfer/other charges |
| Lifetime | Created 06:51:15 UTC; early stop requested around 07:20:25; [destruction confirmed **07:21:57 UTC**](../../evidence/s2-vast-host-20261101-v1/destruction-check-0.json), 2026-11-01 |

The [offer search](../../evidence/s2-vast-host-20261101-v1/offers.json) filtered verified, available, VM-enabled single-GPU 96 GB Workstation offers, direct SSH, reliability score ≥0.99 and sufficient advertised RAM/disk. This candidate cost $0.018667/hour more than the cheapest matching Korean offer and advertised faster download plus a slightly higher reliability score. Those are selection heuristics, not measured uptime probabilities. It is the **tested economical candidate from that filtered search**, not a demonstrated global optimum or a reservation for future use.

The listing's 16 effective CPU threads and 123.438 GiB RAM did **not** equal guest resources: the guest exposed 15 CPUs and about 21.42% less RAM. Use delivered resources for sizing. The 160 disk allocation was for this check; it does not establish capacity for the licensed pilot. The host/operator's licensed-data access arrangement remains unresolved.

Creation used the explicit official image `docker.io/vastai/kvm:ubuntu_cli_22.04-2025-11-21`, SSH/direct mode and `--cancel-unavail`, with no community-template startup script. The [exact order](../../evidence/s2-vast-host-20261101-v1/order-request.json) and public registry metadata are retained. The registry digest is a metadata observation, not a guest-side attestation of Vast's underlying VM package. Guest OS, driver and resources were measured separately. [Vast's VM documentation](https://docs.vast.ai/guides/instances/virtual-machines) describes this guest-admin route; the actual checks below establish its operation on this allocation.

## What passed, and what each result establishes

| Check | Actual evidence and boundary |
|---|---|
| CUDA correctness | 1,048,576 integer outputs match an independently expressed CPU reference; three launches are bitwise identical; deliberate one-element corruption is detected exactly once. Repeated in the Compose GPU client. |
| VRAM integrity | First run allocated **76,039,815,168 bytes = 70.8176 GiB**: every 64-bit word checked in a GPU pass, plus 1,024 distributed CPU samples; zero mismatches and allocation released. The configured ceiling was 72 GiB, not the amount actually tested. One pattern and a short run do not prove sustained reliability or all advertised memory. |
| Deployment | Actual Docker buildx build; two Compose services; DNS and exact request/response; shared bind mount; client PID namespace equals the **guest** Docker host's; GPU reservation executes the CUDA oracle. |
| Nsight Compute | Version 2025.3.1 collected real basic kernel counter tables with guest root while `RmProfilingAdminOnly=1`. No physical-host administration was needed. |
| Nsight Systems | Version 2025.3.2 produced a trace and a parsed summary containing **three CUDA kernel instances**. Explicit NVTX markers, representative AV latency and CPU branch tracing were not established. |
| Synthetic matrix control | cuBLAS FP16 ones × ones with FP32 accumulation/output: all **4,194,304 values exactly 2,048**, before and after a five-second control window; deliberate corruption detected. This is a monitoring load and finite known answer, not a generic GEMM proof, model test or throughput benchmark. |
| DCGM CLI | DCGM 4.7.1: tensor activity goes idle 0 → positive (maximum sampled 0.576) → 0, with power, energy, clocks and PCIe readback. |
| Exporter | Public exporter **4.8.4 / bundled DCGM 4.6.1**, embedded mode, fourteen timestamped scrapes and a separate matrix control. Actual listener is **127.0.0.1:9400**. Tensor activity reaches 0.576561 and returns to zero. Power, energy and PCIe series are present. No Prometheus/alerting deployment is claimed. |
| XID events | NVML advertised mask `0xf19c`, accepted XID subscription and returned timeout after 100 ms. Raw DCGM field 230 returned **NO_DATA (-14)** and its blank sentinel, not NOT_SUPPORTED and not zero errors. No fault was injected; end-to-end fault delivery remains unverified. |

Direct records: [CUDA/memory/Compose and profiler logs](../../evidence/s2-vast-host-20261101-v1/probe-v2/), [matrix/DCGM control](../../evidence/s2-vast-host-20261101-v1/monitor-v1/), [exporter scrapes](../../evidence/s2-vast-host-20261101-v1/exporter-v1/), [derived exporter values](../../evidence/s2-vast-host-20261101-v1/exporter-metrics-summary.json), [XID raw state](../../evidence/s2-vast-host-20261101-v1/xid-v1/dcgm-status-v3.log).

CUDA compilation/execution used immutable local image ID and RepoDigest `sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97` (CUDA 12.8.1 devel). Exporter amd64 digest: `sha256:804c17f67ab63e8a4dd5301546beefaa104129f833dae4a68f5ff7c4819f471b`. Its bundled engine was used after stopping the separately installed 4.7.1 service; the older exporter client was not attached to the newer engine. [NVIDIA's exporter instructions](https://docs.nvidia.com/datacenter/dcgm/latest/installation/install-dcgm-exporter.html) explain version compatibility and embedded/remote modes.

Energy baselines differ between the separate collection phases. Treat energy as segment-local and record/reset-check collector and driver lifecycles; do not concatenate these values into an experiment energy total. Missing initial profiler samples and event-only XID data remain missing. [NVIDIA's NVML event API](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlEvents.html) distinguishes subscription and timeout from an observed event.

## Failures retained and lifecycle proof

The image initially lacked host profiling tools; this was recorded as unverified, then public packages were installed and the main probe repeated. Two raw-DCGM Python attempts failed because the binding loader was not initialized and the suggested function did not exist in installed 4.7.1; the corrected installed API succeeded. All three sources/logs remain in `xid-v1`. Neither failure was relabelled as unsupported hardware.

The live early-stop test found a real watchdog integration defect: Vast reported `actual_status=exited`, `intended_status=stopped`, `next_state=stopped`, while its free-text message still said running. The stop **succeeded**, but the old monitor retried because it recognized only `stopped`. The final predicate also recognizes `exited` with explicit stop intent and no queued restart. A failing real-shape regression and two queued-start controls were added; **26 watchdog tests pass**, with independent review. The correction was tested locally against the observed response shape, without renting another GPU. The exact old runtime source is preserved separately.

The CLI can return prose, or even exit 0 after an API error. Therefore stdout must be parsed and resource state re-read; exit status alone is not completion. The old billing command failed, while `show invoices-v1 --charges` returned task-specific charges successfully. No credential was printed or copied into the VM.

Before stop, **161 guest files (888,496 bytes) were copied and their SHA-256 values matched the guest**, with zero mismatches. [Export verification](../../evidence/s2-vast-host-20261101-v1/guest-export-verification.json). The timer's actual [stop events](../../evidence/s2-vast-host-20261101-v1/watchdog-live-stop-events.jsonl), [structured stopped state](../../evidence/s2-vast-host-20261101-v1/stop-state-072043.json), [destruction request](../../evidence/s2-vast-host-20261101-v1/destruction-request.json) and [post-delete empty instance/volume inventory](../../evidence/s2-vast-host-20261101-v1/post-delete-inventory.json) establish the lifecycle. A guest shutdown alone was never treated as provider cleanup. The external watchdog depends on the control machine/process/network remaining available; it is not a provider-enforced monetary cap.

## Cost and next admission

The [post-delete provider ledger](../../evidence/s2-vast-host-20261101-v1/charges-after-delete.json) records **$0.600**: $0.577 GPU (displayed 0.410 hours), $0.023 disk (displayed 0.516 hours), and $0 transfer. These are rounded provider values at 07:23 UTC; settlement may lag. **New cash deposited: $0.** Existing credit after the check was $22.193260. This usage is inside the task's $10 allowance and the nested $100/$1,200/$2,500 caps. No VM, separate volume, reserved commitment or automatic retry remains.

**Recommendation:** use the full KVM route for the current Compose-based S2 bootstrap; guest administration was sufficient for the tested stack. There is no demonstrated need to pay for bare metal or hypervisor control. Retain this machine/configuration as a measured candidate, re-price and re-qualify any future allocation, and keep Nebius as an untested alternative if CPU/RAM, availability or the licensed-data access arrangement is better there. A cheaper hourly rate is not yet evidence of cheaper valid AlpaSim rollouts.

The next paid scientific pilot still needs the project's asset-license/access decisions, dated admission/release checks, five-scene registration and Huy's written prediction. Then measure the actual pinned teacher + NRE + Harmonizer combination, simultaneous CPU/VRAM/disk fit, raw-log validity and cost per usable rollout within its approved pilot allocation. Full-study funding remains conditional on that measured forecast. No current draft attestation is promoted to full S2 qualification by these host tests.

The final local suite has **754 passing tests**, recorded with the completed lint/syntax checks in [verification](../../evidence/s2-vast-host-20261101-v1/verification-complete.json). Source is uncommitted over Git base `cfce036f87be71fcf9c729fd2b46140ee6394621`; the evidence bundle includes exact task sources, hashes and the resolved CPU lock. These are engineering checks with deterministic synthetic inputs, not model replicates or simulator seeds; AlpaSim SHA/scene list/renderer/model protocol fields are not applicable to this host-only result.

The [task source overlay](../../evidence/s2-vast-host-20261101-v1/task-source-overlay.zip) and [source manifest](../../evidence/s2-vast-host-20261101-v1/source-manifest.json) preserve new/changed task sources and documents. For a fresh reconstruction, check out the recorded Git base, overlay the prior S2 preparation `source.zip` identified by hash in that manifest, then this task overlay. Protected inference sources are verified unchanged against the prior archive; the overlay does not replace them. The actual pre-fix watchdog and guest-executed diagnostic versions remain separately preserved in raw evidence.

To reproduce the CPU checks in the resulting checkout, provide a C++17 compiler for the fake-NVML tests, run `uv sync --frozen`, then `uv run --frozen python -B -m pytest -p no:cacheprovider -q`. The current checkout passed 754 tests; a second reconstruction was not executed for this task. For an independently admitted future host, use the scripts and recorded command/config files in the source bundle; the historical offer ID and price must not be treated as current availability or new rental authority.
