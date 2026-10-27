# Alpamayo-Grade AV Engineering: #1 Project Decision (Oct 2026)

Oct 25, 2026 · @Dang Huy Hoang

## Verdict

**Don't pretrain a model, and don't let Alpamayo steer a car.** Do what NVIDIA itself does with Alpamayo: post-train the open teacher in closed loop, distill it into a small real-time student, and serve that student on Thor hardware. Then run it in shadow mode in a real car in Ho Chi Minh City, where it predicts the trajectory but never actuates.

> *VN:* Kết luận ngắn: **không pretrain, không cho Alpamayo điều khiển xe thật.** Hãy làm đúng việc NVIDIA đang làm nội bộ: (1) RL post-train teacher trong closed loop, (2) distill xuống student nhỏ chạy real-time, (3) serve trên Thor, (4) chạy *shadow mode* trên xe thật ở TP.HCM — model dự đoán trajectory song song với người lái nhưng không bao giờ được actuate. Đây là chuỗi giá trị đúng chuẩn "teacher trên cloud → student trên xe" mà NVIDIA tự mô tả.

Five facts force this decision:

1. **The released checkpoints are cloud teachers.** Alpamayo 1/1.5 are 10B and need at least 24 GB VRAM. Alpamayo 2 Super is 34B with a 72 GB peak. NVIDIA says production students are distilled from them, and the distillation script is not released.
2. **The Thor path is immature.** On Sep 7–8, 2026, NVIDIA staff said Alpamayo is not available for AGX Thor yet. On DRIVE Thor the reasoning text came back empty. The best community Jetson Thor result is 345 ms per step (FlashRT), and the official BF16 build takes 3.3 s.
3. **The RL environment is open, but no results are published.** AlpaGym (GRPO in AlpaSim, Alpamayo 1.5 only) ships with no reported gains. Production RL for 1.5 used 640 GPUs.
4. **Driving autonomously on public roads in Vietnam needs a company inside a sandbox.** The SHTP and QTSC sandboxes admit enterprises only, so an individual can't get a permit. The alternatives are a closed course or shadow mode.
5. **The licenses limit what you can ship.** Weights are under OpenMDW-1.1, but the model cards still say non-commercial and the sources conflict. The dataset license covers internal R&D only, forbids redistribution, and expires 12 months after download.

> *VN:* Năm sự thật quyết định chiến lược. Các checkpoint công khai là *teacher* — chúng sinh ra để chạy trên cloud và để distill, không phải để nhét vào xe. Bước distill (thứ NVIDIA giữ lại, chưa release script) và bước đưa xuống Thor (đang vỡ: CoC rỗng, 3.3 s/step ở BF16) chính là **hai khoảng trống kỹ thuật lớn nhất** của hệ sinh thái Alpamayo lúc này. Ai lấp được hai khoảng trống đó bằng số liệu đo được thì đang chứng minh đúng năng lực của một principal engineer physical AI. Sai lầm phổ biến: nghĩ "deploy lên xe thật" mới là ấn tượng. Thực tế ở NVIDIA, *closed-loop metric + latency budget trên Thor + safety argument* mới là thứ được đánh giá; một video xe tự lái trên đường công cộng không có permit là điểm trừ.

The gaps in facts 1–3 are the project. Closing them with measured numbers is the clearest proof of principal-level physical-AI skill available to one engineer with rented GPUs. This is also PhysLoop's autonomous-driving beachhead, made concrete. It competes for time with the KernVela re-cut, which you called the #1 project on Oct 5, so the two need an explicit priority call.

## The open course: what it gives you

The Think Autonomous *End-to-End Revolution* course teaches the 2023–2025 end-to-end canon. It does not cover today's frontier: VLAs, diffusion/flow trajectory heads, RL post-training, 3DGS closed loop, or edge serving. Use it to get fluent fast, not as the spine of the project. Three modules pay off directly; the rest is optional.

| Course item | Use for this project | Priority |
| --- | --- | --- |
| Tesla Patent Study 3/3: Trigger Classifiers | Template for the shadow-mode data engine: mining the moments where the student disagrees with the driver (Phase 5) | High |
| Module I: Simulators + "Which simulator is best suited to End-to-End?" | Mental model for closed loop vs open loop. AlpaSim's Transfuser plugin is the reference policy integration | High |
| Module II: TransFuser project (3 parts) + UniAD / planners | TransFuser is AlpaSim's baseline policy, so it is your sanity-check driver before Alpamayo. Train it once | High |
| Module III: "The End-To-End Lie" + Black Box project | Interpretability and auxiliary tasks, which map to checking whether the reasoning matches the action | Medium |
| Wayve's Marvellous Technologies | GAIA world-model context for a world model used as the evaluator | Medium |
| Tesla Patent Study 1–2 (virtual cameras, trajectory planning) | Background for multi-camera tokenization | Low |
| CARLA projects, quizzes | Skip unless you need CARLA (CarlaMayo exists if you do) | Low |

The course was written before Alpamayo's December 2025 release. What it lacks you get from primary sources (Section 4): the AR1 paper, alpamayo-recipes, AlpaGym, AlpaSim docs.

> *VN:* Khóa học này là "canon" E2E thế hệ trước (TransFuser 2022, UniAD 2023, Hydra-MDP 2024, GTRS 2025): module → gộp module bằng Transformer → sinh nhiều trajectory rồi chấm điểm. Frontier 2026 đã dịch sang **VLA có reasoning + action expert dạng flow-matching + RL trong closed loop dựng bằng 3DGS/world model**. Vì vậy cách dùng đúng: học nhanh ba phần đắt giá. (1) *Trigger classifiers* của Tesla là bản thiết kế data engine mà bạn sẽ tự xây ở shadow mode. (2) TransFuser là baseline chính thức trong AlpaSim — nếu TransFuser của bạn không ra điểm gần với số NVIDIA công bố thì pipeline eval của bạn đang sai, chưa nói gì đến Alpamayo. (3) Phần simulator giúp hiểu tại sao open-loop minADE không dự đoán được closed-loop: lỗi nhỏ tích lũy thành *covariate shift*, xe trôi ra khỏi phân phối dữ liệu huấn luyện, và chỉ closed loop mới đo được điều đó.

## Frontier landscape, Oct 2026

Every leading stack now trains in a world-model or 3DGS simulator and post-trains with RL. Big teachers are distilled into small onboard students. The labs split on two questions: whether to keep language at runtime, and which sensors to use.

| Player | Model / stack | Runtime language? | Sensors | Status Oct 2026 |
| --- | --- | --- | --- | --- |
| [NVIDIA](https://nvidianews.nvidia.com/news/european-mobility-leaders-build-safe-vehicles-and-robotaxi-fleets-on-nvidia-hyperion) | Alpamayo 1 → 1.5 → 2 Super (34B) → 2 Edge (2B, Thor). Alpamayo Stack = VLA + Halos classical safety stack | Yes, Chain-of-Causation text | Hyperion 10: 14 cameras, 9 radars, 1 lidar | Uber L4 in LA/SF H1 2027. Bolt/Lucid 25K+ cars. Volvo, Mercedes L3/L4 (GTC Berlin, Oct 21) |
| [Waymo](https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving) | Foundation Model: fast sensor-fusion path + slow Gemini-based driving VLM. Teacher → student; the same model also acts as simulator and critic | Slow path only | 6th gen: 13 cameras, 4 lidars, 6 radars, under $20k | \~400k paid rides/week, targeting 1M by end of 2026 |
| [Tesla](https://www.teslaoracle.com/2026/10/23/tesla-announces-robotaxi-launch-in-four-more-cities-releases-safety-data-via-new-website-no-kittens-were-hurt/) | One pixels→controls net with intermediate tokens and a neural world simulator | No (latent) | Cameras only | Robotaxi in 6 cities on FSD v15 (unsupervised); consumer cars on v14.3 |
| [XPeng](https://electrek.co/2026/06/05/xpeng-vla-interview-cvpr-language-poison-tesla-fsd/) | VLA 2.0: vision → action directly, language as input only | No (called "a bottleneck") | Cameras only | VW licenses it from 2027; \~$500M/yr training |
| [Wayve](https://wayve.ai/press/wayve-launches-gaia3/) | GAIA-3 (15B world model) used for evaluation | Not verified | Camera / radar / lidar agnostic | Nissan ProPILOT, Japan FY2027 |
| Academic | [DriveVLA-W0](https://arxiv.org/html/2510.12796v2) 86.1 EPDMS. [DriveZero](https://arxiv.org/html/2609.06055v1) 95.3 PDMS (above human). [Poutine](https://arxiv.org/abs/2506.11234) 7.99 RFS, 2025 Waymo E2E winner (3B VLM + GRPO) | Mixed | — | The RL-in-sim scaling results are the frontier |

Three trends drive the project:

1. **GRPO/PPO post-training is now standard** for driving VLAs (Poutine, AutoVLA, ReCogDrive, Alpamayo). The open question is reward design and reward hacking in closed loop.
2. **Diffusion/flow trajectory heads dominate.** The counter-example is DriveVLA-W0, where autoregressive decoding beat flow matching at 70M frames.
3. **Simulation is the bottleneck.** RAD and ReconDreamer-RL cut collisions by about 3x with PPO in 3DGS scenes. The AlpaSim 2026 challenge caps each policy at 16 GiB and 10 Hz, so a deployable student is a first-class target.

> *VN:* Điểm chung mấu chốt của mọi lab: **teacher lớn trên cloud → RL trong simulator học được (world model/3DGS) → distill xuống student trên xe**. Waymo nói thẳng "teacher → student", NVIDIA nói "Alpamayo là teacher", Tesla/XPeng thì giấu nó trong latent tokens. Điểm tranh cãi thật sự là *ngôn ngữ có nên nằm trên đường nóng (hot path) không*. NVIDIA giữ CoC text vì nó cho phép kiểm chứng an toàn và làm reward (reasoning–action consistency). XPeng bỏ vì mỗi token decode tốn \~10–25 ms trên chip xe. Đây chính là một câu hỏi nghiên cứu bạn có thể trả lời bằng số liệu: *student có CoC vs student không CoC, cùng budget 100 ms trên Thor, bên nào lái tốt hơn trong closed loop?* Không ai công bố thí nghiệm này một cách sạch sẽ.

## Alpamayo from first principles

Alpamayo is a VLM with a flow-matching action expert that attends to the VLM's KV cache, in the style of π0. Its 10B size is a cloud-teacher size. On Thor's 273 GB/s of memory bandwidth, one forward pass of a 10B model cannot fit in 100 ms.

**Tensor-level I/O for Alpamayo 1 / 1.5** ([model card](https://huggingface.co/nvidia/Alpamayo-R1-10B), [paper](https://arxiv.org/html/2511.00088v2)):

- **Cameras:** 4 (front-wide 120°, front-tele 30°, cross-left and cross-right 120°). Captured at 1080×1920 and downsampled to 320×576. Each camera contributes 4 frames: 0.4 s of history at 10 Hz.
- **Egomotion:** 16 past waypoints at 10 Hz, each xyz plus a 3×3 rotation. Timestamps are required.
- **Vision tokens:** about 160 per image, so 16 images give about 2,560 vision tokens per step (estimate).
- **Output 1, CoC text:** a variable-length Chain-of-Causation trace, typically 11–20 tokens ([latency study](https://arxiv.org/html/2605.08975v1)).
- **Output 2, trajectory:** 64 waypoints, 6.4 s at 10 Hz. The expert actually predicts acceleration and curvature, which are integrated through a unicycle model.
- **Size:** 8.2B backbone (Cosmos-Reason, a Qwen-VL family model) plus a 2.3B expert. The expert runs about 10 Euler denoising steps after a single prefill.

**Training pipeline** (AR1 paper; released recipes in [alpamayo-recipes](https://github.com/NVlabs/alpamayo-recipes)):

1. **VLM post-training.** Cosmos-Reason is trained on 3.7M VQA samples, then on 100K driving samples labeled with critical objects and next actions.
2. **CoC data.** Clips are 20 s, with a keyframe about 0.5 s before each behavior change. Each label records a closed-set decision (longitudinal × lateral), the critical components, and the reasoning trace. Labels come from humans plus auto-labelers (GPT-5, Qwen3-VL), with about 10% human-verified. Volume: AR1 700K traces, 1.5 about 3M, 2 Super about 3.7M.
3. **SFT.** The model learns to reason on CoC and to predict discrete trajectory tokens inside the VLM (π0.5-KI style). The flow-matching expert is trained to decode continuous trajectories.
4. **GRPO with a KL penalty to the SFT model.** There are three rewards: (a) an LRM critic grades reasoning 0–5; (b) a binary check that the reasoning matches the trajectory's meta-action; (c) trajectory quality, combining L2 to the expert, collisions and jerk. In the 0.5B ablation, RL moved the reasoning grade from 3.1 to 4.4, consistency from 0.62 to 0.83, and the close-encounter rate from 6.9% to 3.7%.
5. **Closed-loop RL (AlpaGym).** GRPO runs inside AlpaSim on NuRec 3DGS scenes with CAT-K reactive traffic. Example reward: progress +1, collision −10, off-road −5, −0.01 × distance to ground truth ([NVIDIA blog](https://developer.nvidia.com/blog/how-to-post-train-autonomous-vehicle-models-in-closed-loop-with-nvidia-alpamayo/)).
6. **Distillation into a Thor student.** NVIDIA does this internally and has released no script. The output is Alpamayo 2 Edge (2B), announced Oct 21, 2026; no weights or card were found.

**Latency budget.** Per control step the cost is prefill compute plus memory-bound decode plus memory-bound denoising:

```latex
t \approx \frac{2 N_{bb} T_{vis}}{\eta \cdot F_{peak}} + n_{tok} \cdot \frac{B_{bb}}{BW} + K \cdot \frac{B_{exp}}{BW}
```

Symbols: N\_bb = backbone parameters; T\_vis ≈ 2,560 vision tokens; η ≈ 0.4 = achieved fraction of peak; F\_peak = 517 dense FP8 TFLOPS (Jetson Thor); n\_tok ≈ 20 CoC tokens; B = weight bytes; BW = 273 GB/s; K = denoising steps.

| Config on Jetson Thor (estimate) | Prefill | CoC decode (20 tok) | Expert denoise | Total |
| --- | --- | --- | --- | --- |
| 10B teacher, FP8, K=10 | \~250 ms | \~600 ms | \~85 ms | \~0.9 s |
| 10B teacher, NVFP4 decode, FP8 expert, K=10 | \~250 ms | \~330 ms | \~85 ms | \~0.7 s |
| 2B student, NVFP4, 0.5B expert, K=4 | \~50 ms | \~70 ms | \~7 ms | \~135 ms |
| 2B student, no runtime CoC, K=2 | \~50 ms | 0 | \~4 ms | \~55 ms |

The 10B rows match the community FlashRT measurement (811–897 ms on Jetson Thor, [PR #229](https://github.com/flashrt-project/FlashRT/pull/229)), which is a sanity check on the model. The conclusion follows from bandwidth: 10 Hz needs a roughly 2B student, few-step flow sampling, and either short or no runtime reasoning.

> *VN:* Phân tích Feynman từng lớp. **(1) Vì sao là flow-matching expert?** Trajectory là biến liên tục và đa mode (rẽ trái *hoặc* chờ). Regression L2 sẽ lấy trung bình hai mode, ra một đường đâm vào dải phân cách. Flow-matching học trường vận tốc v(x\_t, t | KV cache) để đẩy nhiễu về đúng một mode, nên giữ được tính đa mode. **(2) Vì sao dự đoán acceleration + curvature thay vì xy?** Qua mô hình unicycle, mọi output đều khả thi về động học (không có bước nhảy ngang). Đổi lại, ở vận tốc ≈ 0 thì curvature = ω/v phân kỳ — đúng là issue #113 trên GitHub. **(3) Vì sao 10B không chạy 10 Hz trên Thor?** Decode mỗi token phải đọc toàn bộ trọng số từ DRAM: 11 GB FP8 ÷ 273 GB/s ≈ 40 ms/token, 20 token ≈ 0.8 s. Đây là giới hạn vật lý của memory bandwidth, kernel giỏi đến đâu cũng không vượt qua được. Muốn nhanh chỉ có 3 cách: giảm bytes (quantize, model nhỏ hơn), giảm số lần đọc (ít token CoC, ít bước denoise), hoặc tăng tính toán trên mỗi byte (speculative decoding, batching camera). Sai lầm phổ biến: tưởng NVFP4 làm mọi thứ nhanh 4×. Thực tế prefill bị compute-bound, và báo cáo trên forum cho thấy NVFP4 *fake-quant* trong PyTorch còn chậm hơn FP16 (455 s vs 424 s). Phải có kernel thật — đây là lợi thế CUDA/Triton của bạn.

## Hardware, software and legal reality

Buy a **Jetson AGX Thor dev kit ($5,499)**, not a DRIVE AGX Thor. Rent all training and simulation compute in the cloud. The car is a sensor platform that runs in shadow mode; it never acts on the model's output.

| Platform | Key spec | Price / access | Verdict |
| --- | --- | --- | --- |
| [Jetson AGX Thor T5000](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/) | 128 GB LPDDR5X at 273 GB/s; 517 dense FP8 TFLOPS; 40–130 W; up to 20 cameras through Holoscan Sensor Bridge; JetPack 7.2.1 with CUDA 13.2 and TensorRT 10.16 | $5,499 at [Newegg](https://www.newegg.com/nvidia-jetson-agx-thor-developer-kit-128-gb-256-bit-lpddr5x-273-gb-s/p/N82E16813190035), up from $3,499 in July 2026 | **Buy.** Same Blackwell Thor silicon; open JetPack; 2× the memory of DRIVE Thor |
| [DRIVE AGX Thor](https://developer.nvidia.com/drive/agx) | 1,000 INT8 TOPS; 64 GB; 16 GMSL2 + 2 GMSL3 inputs; 4 CAN; 350 W; DriveOS 7.x early access | Distributors only, no list price (\~$20–26k on marketplaces, unverified); 6–10 week lead time | Skip. It's the production target, but costs 4× as much and Alpamayo is broken on it today |
| Cloud RTX PRO 6000 (96 GB) | Fits Alpamayo 1.5 plus the AlpaSim renderer | vast.ai $1.00 cheapest / $1.60 median per GPU-hour ([live API](https://console.vast.ai/api/v0/bundles/), Oct 25) | Dev box, 2 GPUs |
| Cloud H100 SXM | Runs 2 Super at a 72 GB peak; SFT/RL recipes are validated on 8×H100 | vast.ai $1.93 / $2.40; Lambda $3.99 | RL and distillation |
| Cloud B200 / B300 | Native NVFP4 for testing quantization numerics | vast.ai B200 $7.50, B300 $5.90 (cheapest) | Short bursts only |

**Software stack.** The parts are all open, but they are young.

- **Simulation:** [AlpaSim](https://github.com/NVlabs/alpasim) (Apache-2.0) runs as gRPC microservices. Default renderer is NuRec 3DGS; physics and CAT-K reactive traffic are included. A policy plugs in as a driver container. It reports `collision_at_fault`, `offroad` and `dist_to_gt` metrics.
- **Closed-loop RL:** [AlpaGym](https://github.com/NVlabs/alpagym) (GRPO on Cosmos-RL) supports Alpamayo 1.5 only, with at least 2 GPUs of 40 GB or more. It has 13 commits so far.
- **Training recipes:** [alpamayo-recipes](https://github.com/NVlabs/alpamayo-recipes) covers SFT (DeepSpeed ZeRO-2, 8×H100), open-loop GRPO (at least 5×80 GB), and ModelOpt 0.43 quantization (FP8, NVFP4, W4A8).
- **Serving:** a NIM built on vLLM-Omni, one request at a time. [TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/130) support for 1.5 is unconfirmed. [FlashRT](https://github.com/flashrt-project/FlashRT/pull/229) reaches 345–897 ms on Jetson Thor.
- **Data:** the [PhysicalAI-AV dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) has about 1,700 h in 306K clips from 25 countries (133 TB total), plus about 1,500 NuRec scenes (about 1.5 GB each). Access is gated.

**Sensor rig for shadow mode.** The goal is to match the training camera geometry:

- **Cameras:** 4× GMSL2 IMX728/IMX623 at about $459 each. FOVs are 120° front-wide, 30° front-tele, and two 120° cross cameras.
- **Deserializer and sync:** a Holoscan Sensor Bridge deserializer (or a Connect Tech 8-input carrier). PTP keeps the cameras in sync.
- **Positioning:** a [ZED-X20P RTK GNSS](https://www.sparkfun.com/sparkfun-allband-gnss-rtk-breakout-zed-x20p-qwiic.html) board ($300) plus an IMU.
- **Vehicle signals:** speed and steering are read from CAN, read-only. A comma four ($999) can serve as that interface on a supported car.

**Legal.** Vietnam's 2024 Road Traffic Law (Art. 34) requires a permit for "smart vehicles". The HCMC SHTP and QTSC sandboxes are open to enterprises only, and in Hanoi the Phenikaa-X sandbox runs for 24 months ([SHTP](https://shtp.hochiminhcity.gov.vn/en/establish-a-sandbox-framework-to-support-trials-in-unmanned-aerial-vehicles-and-autonomous-vehicles.htm)). In California, anyone who installs AV technology counts as a manufacturer, which means $5M insurance and a DMV permit ([DMV](https://www.dmv.ca.gov/portal/file/adopted-regulatory-text-pdf)). That leaves an individual three options: shadow mode with no actuation, a closed course with a safety driver and hardware e-stop, or simulation.

> *VN:* Vì sao chọn Jetson Thor mà không phải DRIVE Thor? Cùng một con chip Blackwell, cùng 273 GB/s bandwidth — và bandwidth mới là thứ quyết định latency decode (Section 4). Jetson có 128 GB (nhét được cả teacher 10B lẫn student để so sánh A/B trên cùng máy), JetPack mở, giá 1/4. Số đo trên Jetson Thor chuyển sang DRIVE Thor gần như 1:1 về compute. Khác biệt thật nằm ở DriveOS/ASIL và GMSL native — thứ chỉ OEM cần. **Về pháp lý ở Việt Nam:** shadow mode (model chạy, dự đoán, ghi log; người lái hoàn toàn kiểm soát, không có đường actuation) về bản chất là một dashcam thông minh, không phải "xe tự lái". Dù vậy hãy coi đây là nhận định kỹ thuật, không phải tư vấn pháp lý (tôi không phải luật sư) — hãy xác nhận quy định về lắp camera/thiết bị lên xe trước. Nếu sau này muốn actuate thật, con đường đúng là hợp tác với một doanh nghiệp trong sandbox QTSC/SHTP, không phải tự làm trên đường công cộng.

## Candidate projects, ranked

**#1 is the full teacher→student→Thor loop (22/25).** Candidates #2 and #3 are each one phase of it, so it is built so you can stop at either and still ship.

Each criterion is scored 1–5; 5 is best:

- **Signal:** how closely it matches NVIDIA AV/Sim job requirements ([Research Scientist](https://jobs.nvidia.com/careers/job/893391627291), [ML & Sim Engineer](https://jobs.nvidia.com/careers/job/893392738069)).
- **Gap:** whether nobody has published it yet.
- **Edge:** how much it uses your CUDA, RL and inference skills.
- **Cost:** 5 = cheapest.

| # | Project | Signal | Solo feasibility | Gap | Edge | Cost | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **Closed-loop RL on Alpamayo 1.5 → distill to a ≤2B student → 10 Hz on Jetson Thor → HCMC shadow mode** | 5 | 4 | 5 | 5 | 3 | **22** |
| 2 | AlpaGym RL study alone: first published closed-loop gains plus a reward-hacking analysis | 4 | 5 | 4 | 4 | 4 | 21 |
| 3 | Real-time serving of the 10B teacher on Thor (kernels, NVFP4, fewer denoising steps) | 4 | 4 | 3 | 5 | 4 | 20 |
| 4 | Shadow mode plus a trigger-classifier data engine in HCMC traffic, no training | 3 | 4 | 4 | 2 | 4 | 17 |
| 5 | Re-implement TransFuser / DiffusionDrive in CARLA (the course projects) | 1 | 5 | 1 | 2 | 5 | 14 |
| 6 | Real actuation: openpilot fork or drive-by-wire on a closed course | 3 | 1 | 2 | 2 | 1 | 9 |
| 7 | Pretrain your own driving VLA | 2 | 1 | 2 | 3 | 1 | 9 |

The AlpaSim 2026 Closed-Loop Challenge is out: registration closed Oct 18 and PAI-AV entries are due Oct 31. Its 16 GiB / 10 Hz rules still serve as the student's acceptance spec.

> *VN:* Lý do #1 thắng: nó là *toàn bộ vòng lặp* mà job description của NVIDIA AV Sim mô tả ("large-scale RL frameworks for multimodal AV foundation models", "closed-loop E2E eval"). Đồng thời nó chạm đúng ba khoảng trống chưa ai công bố: RL gains trong AlpaGym, script distill, và student 10 Hz trên Thor. #6 và #7 trông "ngầu" nhưng là bẫy: #7 cần \~80K giờ dữ liệu và hàng trăm GPU, #6 bị chặn bởi pháp lý và an toàn, lại chứng minh kỹ năng tích hợp xe chứ không phải kỹ năng model. #5 là bài tập khóa học — tốt để khởi động, vô giá trị trên CV. Thiết kế #1 theo kiểu *có điểm dừng an toàn*: dừng sau Phase 2 vẫn có #2 để công bố, dừng sau Phase 4 vẫn có #2 + #3.

## #1 project: PhysLoop-AV, Teacher → Thor

**Goal:** a ≤2B Alpamayo-derived student that keeps at least 90% of the RL-tuned teacher's AlpaSim closed-loop score, fits in 16 GiB, and runs at p99 ≤ 100 ms on a Jetson AGX Thor. Validate it with hardware-in-the-loop and with 20–50 h of shadow-mode driving in Ho Chi Minh City. Plan: 12 weeks, about $11–15k all-in.

**Research question it answers:** under a fixed 100 ms Thor budget, does a student that reasons at runtime (CoC) drive better in closed loop than one that doesn't? This is the XPeng-vs-NVIDIA split, measured cleanly for the first time.

&#91;embedded content: PhysLoop-AV loop: 3 cloud stages, 3 edge stages\]

Read it clockwise. The teacher improves in closed loop and is distilled. The student proves itself on Thor, then in a real car. Its disagreements with the human driver flow back as new evaluation data.

| Phase | Weeks | Build | Gate (go / no-go) | Cloud cost |
| --- | --- | --- | --- | --- |
| P0 Reproduce | 1–2 | AlpaSim with the TransFuser and Alpamayo 1.5 drivers on 2× RTX PRO 6000; Prometheus dashboards | Match the model-card AlpaSim score ±0.1 on the public suite. The cards disagree (1.37 vs 0.81), so settle which is right first | \~$150 |
| P1 Closed-loop RL | 2–5 | AlpaGym GRPO on 1.5. Reward ablation: card reward vs + comfort/jerk vs + AlpaJudge reasoning–action consistency. A reward-hacking audit (freezing to avoid collisions, gaming progress) | At-fault collisions down ≥20% relative on held-out scenes, with no regression in progress or off-road | \~$500–1,100 (225–450 H100-h) |
| P2 Distill | 5–8 | Teacher = RL'd 1.5, plus 2 Super for CoC labels. Student = Cosmos-Reason2-2B + 0.3–0.5B flow expert. (a) offline KD on teacher rollouts, (b) on-policy DAgger in AlpaSim, (c) short GRPO on the student, (d) cut flow steps 10 → 2–4. Two arms: CoC and no-CoC | Student ≥90% of teacher AlpaSim score; ≤16 GiB | \~$800–1,500 |
| P3 Thor serving | 7–10 | NVFP4 weights, FP8 prefill, static KV cache with CUDA graphs, a fused flow-step kernel, camera-batched ViT. Upstream fix or PR for the empty-CoC Thor bug | p99 ≤ 100 ms with 4 cameras; 10 Hz sustained for 30 min; ADE drift ≤2% vs BF16 | Hardware only |
| P4 HIL | 9–11 | Thor serves the driver gRPC endpoint against cloud AlpaSim, which runs in lockstep so scores are unaffected by network delay. On-device time is logged per step. Plus real-time log replay from NVMe | Closed-loop score on Thor equals the cloud score within noise | \~$200 |
| P5 Shadow mode | 10–12 | Live 4-camera rig in a human-driven car. Student predictions vs the driver's path. A trigger classifier mines disagreements into an HCMC OOD benchmark (motorbike swarms, missing lane lines, wrong-way riders) | ≥20 h logged; top-k disagreement clips reviewed and labeled | — |
| P6 Ship | 12 | Open-source repo, tech report, demo video (Thor latency overlay + shadow trajectories) | Report reproducible from the repo | — |

**Hardware bill of materials**

| Item | Spec | Cost (USD) |
| --- | --- | --- |
| Jetson AGX Thor dev kit | T5000, 128 GB, JetPack 7.2.1 | 5,499 |
| 4× GMSL2 cameras | IMX728 or IMX623; 120° front-wide, 30° front-tele, 2× 120° cross | 1,836 |
| GMSL→Thor interface | Holoscan Sensor Bridge deserializer or Connect Tech 8-input carrier (quote) | \~1,000–3,000 (est.) |
| GNSS + IMU | ZED-X20P RTK ($300) + automotive-grade IMU | \~600–2,300 (est.) |
| Vehicle interface (read-only CAN) | comma four, optional, if your car is supported | 999 |
| Power, mounts, 2 TB NVMe | 12 V DC-DC or inverter, roof/windshield mounts | \~500 |
| Cloud compute, P0–P4 | vast.ai H100 at $1.93–2.40/h, RTX PRO 6000 at $1.00–1.60/h | \~2,000–3,000 |
| **Total** | Lean option: skip live cameras, log replay only, about $5.8k + cloud | **\~11,000–15,000** |

**Software stack.**

- **Cloud:** AlpaSim, AlpaGym + Cosmos-RL, alpamayo-recipes (SFT, RL, ModelOpt quantization), PyTorch, vLLM for teacher rollouts, NuRec scene suite (NGC key), W&B or Prometheus.
- **Edge:** JetPack 7.2.1 (CUDA 13.2, TensorRT 10.16), TensorRT Edge-LLM, custom CUDA/Triton kernels, Holoscan SDK + Sensor Bridge (PTP sync), ROS 2 optional for logging, and the AlpaSim gRPC driver proto.

**Metrics reported:**

- **Closed-loop (AlpaSim):** score, at-fault collisions, off-road, distance to ground truth.
- **Open-loop:** minADE6 at 6.4 s.
- **Reasoning:** LingoQA, plus reasoning–action consistency.
- **Thor serving:** latency p50 and p99 per stage, memory, energy per step.
- **Shadow mode:** disagreement rate.

> *VN:* Cách đọc bản spec này như một principal engineer. **(1) Mỗi phase có một gate đo được** — không qua gate thì không sang phase sau. P0 quan trọng nhất dù nhàm chán nhất: nếu bạn không tái tạo được số của NVIDIA thì mọi cải thiện sau đó đều vô nghĩa (bạn không biết mình đang đo cái gì). **(2) Vì sao DAgger ở P2?** KD offline chỉ dạy student trên phân phối trạng thái của *teacher*. Khi student tự lái, nó gây ra lỗi nhỏ và trôi vào các trạng thái teacher chưa bao giờ gặp — lỗi tích lũy O(T²ε). DAgger cho student lái và teacher gán nhãn ngay trên quỹ đạo đó, hạ lỗi về O(Tε). Đây chính là lý do closed-loop distillation > offline distillation. **(3) Vì sao HIL dùng lockstep?** Simulator đợi policy trả lời rồi mới bước tiếp, nên độ trễ mạng HCM↔cloud không làm sai điểm. Latency on-device được đo riêng và so với budget 100 ms. Muốn trung thực hơn nữa: thêm thí nghiệm *latency injection* — cho sim áp action trễ đúng bằng latency đo trên Thor, xem score rơi bao nhiêu. Đó là câu hỏi NVIDIA sẽ hỏi bạn. **(4) Shadow mode ở TP.HCM là vũ khí riêng của bạn:** giao thông xe máy dày đặc là OOD cực mạnh so với dữ liệu Mỹ/châu Âu (cần kiểm tra Việt Nam có nằm trong 25 quốc gia của dataset không). Một benchmark OOD Việt Nam có nhãn là thứ không lab Mỹ nào tự có được — và nó gắn với góc sovereign AI của bạn.

## Risks and kill criteria

The biggest risk is NVIDIA shipping the same result first, e.g. Alpamayo 2 Edge weights plus a distillation script. If that happens, pivot to beating Edge rather than racing it: run it in your loop as the baseline.

| Risk | Signal | Response / kill criterion |
| --- | --- | --- |
| NVIDIA releases Alpamayo 2 Edge weights or a distillation recipe | HF card or GitHub release | Use Edge as the P2 baseline. The contribution becomes RL'd teacher + DAgger + the CoC vs no-CoC study, measured against Edge |
| AlpaGym is too young (13 commits, 1.5 only) | P1 runs crash or score variance is above the effect size | Fall back to open-loop GRPO in alpamayo-recipes. Kill P1 if closed-loop variance stays above 0.1 after 3 seeds |
| P0 can't reproduce card numbers | Gap > 0.1 that persists | Open a GitHub issue with logs (that's useful signal in itself). Report relative gains only |
| Thor misses 100 ms | p99 > 150 ms after P3 | Drop runtime CoC (no-CoC arm), K=2, smaller expert. Kill the 10 Hz claim and report 5 Hz honestly if still over budget |
| Licenses | Weights OpenMDW-1.1 but cards say non-commercial; dataset is internal R&D only, no redistribution, 12-month expiry | Publish code, metrics and your own HCMC data. Never redistribute PhysicalAI-AV clips, NuRec scenes or derived weights without checking the license. Not legal advice |
| Shadow-mode legality or safety | Any doubt about mounting equipment | Hardware has no actuation path at all, and the human driver is always in control. Confirm local rules before driving. Skip P5 if unclear: P0–P4 stand on their own |
| Compute overrun | Cloud spend > $4k | Cap groups, cut scene count, use spot RTX PRO 6000s for rollouts and H100s only for the optimizer |
| Focus split with KernVela | Both slip | Make one call now. P3 (Thor kernels) reuses your KernVela skills, so pick this or that, not both half-done |

> *VN:* Kill criteria phải viết *trước* khi chạy, không phải sau — nếu không bạn sẽ tự thuyết phục mình rằng "chỉ cần thêm một tuần". Rủi ro thật sự không phải kỹ thuật mà là *phương sai*: closed-loop RL trong sim có variance rất cao giữa các seed. Nếu độ lệch chuẩn giữa 3 seed lớn hơn hiệu ứng bạn muốn chứng minh, kết quả đó là nhiễu, không phải phát hiện. Hãy luôn báo cáo mean ± CI trên ≥3 seed và trên held-out scenes. Về KernVela: hai dự án cùng dùng một kỹ năng lõi (kernel + RL environment), nhưng làm song song cả hai thì cả hai đều nửa vời. Quyết định là của bạn.

## Sources

Research agents opened these pages on Oct 25, 2026. Items they flagged as unverified are noted as such in the text: the DRIVE Thor price, the Alpamayo 2 Edge weights, the 1.5 AlpaSim score, and Tesla's architecture details.

**Alpamayo and NVIDIA tooling**

- [Alpamayo-R1 paper (arXiv 2511.00088)](https://arxiv.org/html/2511.00088v2)
- Model cards: [Alpamayo 1](https://huggingface.co/nvidia/Alpamayo-R1-10B), [Alpamayo 1.5](https://huggingface.co/nvidia/Alpamayo-1.5-10B), [Alpamayo 2 Super](https://huggingface.co/nvidia/Alpamayo2-Super), [AlpaJudge](https://huggingface.co/nvidia/alpajudge)
- Code: [alpamayo-recipes](https://github.com/NVlabs/alpamayo-recipes), [AlpaGym](https://github.com/NVlabs/alpagym), [AlpaSim](https://github.com/NVlabs/alpasim), [NVlabs/alpamayo issues](https://github.com/NVlabs/alpamayo/issues)
- [NVIDIA blog: closed-loop post-training with Alpamayo](https://developer.nvidia.com/blog/how-to-post-train-autonomous-vehicle-models-in-closed-loop-with-nvidia-alpamayo/)
- [Alpamayo llm-info page](https://www.nvidia.com/en-us/solutions/autonomous-vehicles/alpamayo/llm-info/)
- [PhysicalAI-AV dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- NVIDIA newsroom: [Alpamayo 2 Super](https://nvidianews.nvidia.com/news/nvidia-alpamayo-2-super-robotaxis), [GTC Berlin (Alpamayo 2 Edge, Alpamayo Stack)](https://nvidianews.nvidia.com/news/european-mobility-leaders-build-safe-vehicles-and-robotaxi-fleets-on-nvidia-hyperion), [Uber L4](https://investor.uber.com/news-events/news/press-release-details/2026/NVIDIA-to-Launch-L4-Software-Driven-Robotaxis-on-Uber-Across-28-Cities-by-2028/default.aspx)
- [AlpaSim 2026 Closed-Loop Challenge](https://opendrivelab.com/alpasim2026)

**Thor and serving**

- NVIDIA forums: [AGX Thor quantization thread](https://forums.developer.nvidia.com/t/how-to-perform-alpamayo1-5-fp8-quantization-for-nvidia-thor/382409), [DRIVE Thor empty-CoC thread](https://forums.developer.nvidia.com/t/drive-agx-thor-with-alpamayo/382501), [AlpaSim instability thread](https://forums.developer.nvidia.com/t/testing-alpamayo-1-5b-in-alpasim-model-instability/384744)
- [FlashRT PR #229 (Jetson Thor latency)](https://github.com/flashrt-project/FlashRT/pull/229)
- [Latency study on DGX Spark (arXiv 2605.08975)](https://arxiv.org/html/2605.08975v1)
- [Jetson Thor intro](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/)
- [DRIVE AGX](https://developer.nvidia.com/drive/agx)
- Pricing: [Jetson price increase (CNX)](https://cnx-software.com/2026/07/22/nvidia-increases-the-price-of-jetson-modules-and-devkits-by-up-to-101), [vast.ai live pricing](https://console.vast.ai/api/v0/bundles/), [Lambda](https://lambda.ai/pricing), [RunPod](https://www.runpod.io/pricing)

**Landscape**

- [Waymo Foundation Model](https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving)
- [XPeng VLA 2.0 interview](https://electrek.co/2026/06/05/xpeng-vla-interview-cvpr-language-poison-tesla-fsd/)
- [Wayve GAIA-3](https://wayve.ai/press/wayve-launches-gaia3/)
- [Tesla robotaxi expansion](https://www.teslaoracle.com/2026/10/23/tesla-announces-robotaxi-launch-in-four-more-cities-releases-safety-data-via-new-website-no-kittens-were-hurt/)
- [DriveVLA-W0](https://arxiv.org/html/2510.12796v2)
- [DriveZero](https://arxiv.org/html/2609.06055v1)
- [Poutine](https://arxiv.org/abs/2506.11234)
- [RAD](https://arxiv.org/html/2502.13144v2)
- [WOD-E2E / RFS](https://arxiv.org/html/2510.26125v3)
- NVIDIA jobs: [AV Research Scientist](https://jobs.nvidia.com/careers/job/893391627291), [AV ML & Sim Engineer](https://jobs.nvidia.com/careers/job/893392738069)

**Legal and sensors**

- [HCMC SHTP sandbox](https://shtp.hochiminhcity.gov.vn/en/establish-a-sandbox-framework-to-support-trials-in-unmanned-aerial-vehicles-and-autonomous-vehicles.htm)
- [QTSC controlled testing](https://qtsc.com.vn/en/news/ho-chi-minh-city-allows-controlled-testing-of-self-driving-vehicles-at-qtsc)
- [California DMV AV regulations](https://www.dmv.ca.gov/portal/file/adopted-regulatory-text-pdf)
- [Leopard IMX728 GMSL2](https://leopardimaging.com/product/automotive-cameras/cameras-by-interface/adi-gmsl2-cameras/li-imx728-gm2b/li-imx728-gm2b-120h-sf-aa/)
- [comma four](https://blog.comma.ai/comma-four/)
- [openpilot SAFETY.md](https://github.com/commaai/openpilot/blob/master/docs/SAFETY.md)
- Course: Think Autonomous, *End-to-End Revolution* (your open tab)
