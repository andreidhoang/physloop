# S2 rental choice prepared for final approval

Public-source check: **2026-10-31**. The user has adopted the **$100 pilot / $1,200 S2 / $2,500 project** ceilings. These nested ceilings are not permission to charge a card, rent a VM, accept licenses or download assets. None of those actions was taken in this preparation. The proposed first-run limits remain **8 GPU-hours and at most $2.50 per GPU-hour**; confirm them in the final operational approval.

## Recommended configuration

**Primary: one Nebius regular on-demand VM in `us-central1` (Kansas City, Missouri), platform `gpu-rtx6000`, preset `1gpu-24vcpu-218gb`.** This is **one NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB**, with 24 vCPUs and 218 GiB RAM. The published alternative RTX preset has eight GPUs; do not accidentally select it. **Fallback: the same one-GPU preset in `eu-south1` (Madrid), platform `gpu-rtx6000-a`.** Both are public regional offerings, not verified live capacity or reserved offers. [Nebius platform/preset table](https://docs.nebius.com/compute/virtual-machines/types), [region locations](https://docs.nebius.com/overview/regions).

Choose the stable **`ubuntu24.04-cuda13.0`** GPU image. Nebius recommends it for both RTX platforms and documents a 580.x driver, NVIDIA Container Toolkit and DCGM. Resolve the actual image ID and component versions after provisioning; the family name alone is not an immutable pin. [Boot image compatibility and components](https://docs.nebius.com/compute/storage/boot-disk-images).

This is a VM with administrator access inside its operating system, not a container-only Pod or a claim of control over the provider's physical hypervisor. Nebius documents unrestricted `sudo` for the configured VM user; direct SSH usernames `root` and `admin` are reserved. [VM access configuration](https://docs.nebius.com/compute/virtual-machines/create).

The adopted [AlpaSim source audit](2026-10-31-s2-upstream.md) requires Docker Engine with Compose/buildx, NVIDIA Container Toolkit and a driver compatible with CUDA 12.8 (at least 570.x), plus authorized assets. The VM/image above is a credible bootstrap candidate. **Successful simultaneous teacher 1.5 BF16, NuRec rendering and Harmonizer operation on this one GPU is not yet measured.** Install/verify Docker and plugins, test GPU visibility, renderer startup, memory headroom and required DCGM counters within the approved bootstrap allowance. If co-location fails, stop and present a revised topology/quote; do not silently rent another VM or select eight GPUs.

**Rejected fallback: ordinary Runpod Pods.** Runpod explicitly states that users cannot run their own Docker instance or Docker Compose on Pods. The previously checked $2.49 GPU card therefore does not establish a compatible host for this protocol. A different deployment architecture would require a separate design, not an improvised substitution. [Runpod limitations](https://docs.runpod.io/pods/overview#limitations).

## Price, storage and payment boundary

Nebius publishes **$1.80 per GPU-hour** for both selected RTX platforms, with per-second compute billing. Its RTX table does not list additional vCPU/RAM charges, unlike its L40S table; interpreting $1.80 as the selected one-GPU preset's compute total is a **planning inference pending an itemized console quote**. Require the quote to confirm CPU/RAM inclusion. Running-but-idle time is billable; stopped compute is not. Prices exclude applicable tax. [Compute pricing](https://docs.nebius.com/compute/resources/pricing).

Proposed storage: **one 500 GiB VM-managed Network SSD boot/work disk**, encrypted, with no extra disks, filesystem, snapshots or object bucket. This size is a planning allowance for images, model weights, the minimum scene subset and evidence; it is not a measured requirement. Network SSD supports this size. Do not choose a similarly named non-replicated disk: that product requires multiples of 93 GiB and has different durability. Managed disks are deleted with the VM. [Storage types and lifecycle](https://docs.nebius.com/compute/storage/types).

At the current Network SSD rate of **$0.071/GiB per 730 hours**, 500 GiB retained for seven days costs **$8.17**. Eight hours on the selected one-GPU VM costs **$14.40**; their baseline sum is **$22.57**, excluding account-specific taxes/fees and unquoted services. Storage bills by provisioned size and elapsed existence, including after compute stops. These are calculated scenarios, not a checkout quote. [Compute/storage billing](https://docs.nebius.com/compute/resources/pricing).

The [public price page](https://nebius.com/prices) lists ordinary network ingress/egress and public IPs as free. This proposal creates no object bucket; object-storage egress is a separate tariff and must not be confused with VM networking. Verify the selected services in the console before booking.

For the final approval, retain the **$100 all-in ceiling**, reserve at most **$20 for the first compute allocation**, and use at most **$80 for storage/transfer/setup/tax and other explicitly quoted charges**. Unused contingency cannot extend GPU-hours. Target deletion within seven days; a longer retention period requires an explicit revised forecast inside the same approved envelope. Stop before any package expansion or full-corpus download that would exceed space, time or money limits.

**Adding a payment card is already a paid action.** Nebius requires a **$25 first payment**, credited to the account balance; subsequent manual top-ups also have a $25 minimum. PAYG can automatically charge the card when its negative-balance threshold is reached or at month start. The balance is not a hard spending limit. Keep separate records of cash deposited and service charges: the initial $25 credit is not an extra $25 of compute cost. Request only the $25 initial credit; no further manual recharge or upsizing without approval. Do not assume automatic charging is disabled. [Card payments](https://docs.nebius.com/signup-billing/payments/card), [PAYG billing](https://docs.nebius.com/signup-billing/billing-models/payg).

Tax depends on the actual payer type and billing residence. Nebius's current table says it adds no tax for unlisted locations; Vietnam is not listed. That is the provider's invoicing statement, not a conclusion about Huy's tax obligations. Confirm the checkout/invoice treatment and any card FX fee; reserve them inside the cap. [Provider tax policy](https://docs.nebius.com/signup-billing/payments/taxes).

## Console sequence, with the approval boundary explicit

The following steps are instructions for the eventual approved operation; none was executed here.

1. **Before any payment:** Huy reviews the provider terms and the [exact asset/license packet](2026-10-31-s2-license-approval.md). If an existing account is available, inspect its project/region, quota and billing state without provisioning. A new signup entails the provider's terms, and billing setup can trigger the $25 charge. Stop before submitting card details until final payment approval. [Signup and billing flow](https://docs.nebius.com/signup-billing/sign-up).
2. **Check actual availability:** open **Administration → Capacity dashboard**; inspect the regular, one-GPU RTX preset in the primary region, then the named fallback if necessary. Save its timestamp, quota and capacity result. Even the dashboard does not guarantee capacity at creation. If this requires account activation/payment, report that dependency rather than calling the public listing an available offer. [Capacity advisor](https://docs.nebius.com/compute/virtual-machines/capacity-advisor).
3. **Prepare the VM review:** **Compute → Virtual machines → Create resource → Virtual machine**. Select **With GPUs → Regular**, the region/platform/preset above, and name `physloop-s2-pilot`. Choose the stated image and managed disk. Use the regional subnet, dynamic public IP, and a named SSH user with sudo. Configure only the user's public key, never a private key. Reach **Review**, capture the itemized price/configuration, and stop before **Create VM**. [Creation wizard](https://docs.nebius.com/compute/virtual-machines/create).
4. **Freeze the concrete approval record:** attach the quote, payment amount, exact GPU count/edition, image choice, disk lifecycle, billing assumptions, license decisions, pilot registration and cutoff plan below. Only after that final approval may Huy submit payment/license forms and the approved VM be created. New region/SKU/count, commitment or spot pricing is not an automatic fallback.
5. **After creation:** start the billed-time ledger immediately, install/verify prerequisites and resolve immutable environment identities. Admit the scientific pilot only after compatibility, data and protocol checks pass. Setup, image downloads, warm-up and idle time consume the same GPU-hour allocation. No full sweep is released by the pilot approval.

A provider email budget is useful only as an alert. Nebius states budgets **do not cap usage**, can update hours late and exclude VAT. Do not treat an alert setting as the $100 enforcement mechanism. [Budget limitations](https://docs.nebius.com/signup-billing/budgets).

## Stop and cleanup plan that must accompany approval

Proposed operational policy: one VM only; record creation/start time and cumulative billed GPU-hours across every start. **Prepare the external control-plane stop monitor/timer and designate its operator before creation.** Bind the returned VM ID immediately, verify the monitor is armed before bootstrap or scientific work, and stop the VM if that cannot be established. A proposed cutoff at 7.5 billed hours leaves 30 minutes for shutdown confirmation inside the eight-hour ceiling; the operator must verify the provider reports Stopped. The monitor must include forecast storage/tax liability and available funded balance, not only GPU utilization. No external watchdog was implemented or activated during this preparation. The public documentation inspected here does not establish a native hard dollar cap or a configured automatic lifetime for this VM.

Use **Compute → Virtual machines → VM menu → Stop**, confirm, and verify status **Stopped**. The documented API/CLI operation is `nebius compute instance stop --id <VM_ID>`. **Do not rely on Linux `shutdown`/`halt`: Nebius may interpret guest shutdown as failure, restart the VM and continue billing.** [Provider stop procedure](https://docs.nebius.com/compute/virtual-machines/stop-start).

Stopping preserves billable storage. After copying allowed evidence to the designated private location and verifying its checksums, delete the VM through its **Settings → Delete virtual machine** flow. Its managed disk is deleted with it; separately check for any standalone disks, snapshots, filesystems or buckets and remove those explicitly if they were created. Preserve proof of stop/deletion and inspect subsequent usage entries. Never delete the only evidence copy. [VM deletion and remaining-volume charges](https://docs.nebius.com/compute/virtual-machines/delete).

For an unexpected standalone disk/filesystem, first detach it from the VM. Use **Storage → Disks** (or **Shared filesystems**) → the volume → **Settings → Delete disk** (or **Delete filesystem**) → confirm. This is a separate cleanup action; deleting a VM does not remove standalone storage. [Documented volume deletion](https://docs.nebius.com/compute/storage/manage#how-to-delete-a-volume).

## Concrete review record still required

| Field | Prepared value / remaining verification |
|---|---|
| Money authority | Adopted nested $100 / $1,200 / $2,500 ceilings; actual payment/rental approval pending |
| Primary / fallback | Nebius Kansas City `gpu-rtx6000` / Madrid `gpu-rtx6000-a`; same one-GPU preset |
| Physical GPU identity | RTX PRO 6000 Blackwell **Server Edition**, 96 GB; confirm delivered identity |
| Compute | Proposed ≤8 cumulative GPU-hours; unit rate ≤$2.50 including any separate CPU/RAM charge; public scenario $1.80 |
| Storage | Proposed 500 GiB managed Network SSD; export and delete by recorded timestamp within seven days |
| Initial payment | $25 credit if new card setup is needed; quote tax/fees; no larger automatic deposit requested |
| Live quote / quota | Not inspected in an account; capture exact configuration, availability timestamp and itemized charge before creation |
| Technical fit | VM sudo/image components documented; actual Docker/renderer/DCGM and simultaneous VRAM fit remain bootstrap checks |
| License action | Huy reviews and accepts the exact asset terms; no agent acceptance or asset download performed |
| Stop mechanism | External timer/API stop monitor prepared before creation and bound/armed before work; named operator, time margin, verification of Stopped and storage cleanup date |
| Scientific permission | Pilot registration, release check/closure evidence, predictions and full manifest remain required; no full study release |

**Recommendation:** approve this specific configuration only after the unresolved checkout/account details are attached. If those details cannot be viewed without the first $25 payment, request a separately bounded **account-activation payment** first; do not imply that payment guarantees the GPU or authorizes the VM.
