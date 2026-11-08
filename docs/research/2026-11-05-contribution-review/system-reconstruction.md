# Dựng lại hệ thống: từ quan sát tới hành động, và vòng học

Agent-authored, 2026-11-05. Mỗi dòng ghi nguồn và loại bằng chứng. Phần suy luận/kernel dựa trên lane A, phần vòng học dựa trên audit 11-01 và bản sao pinned trong `evidence/rl-gap-review-20261101-v1/pinned-source/`, phần seed/thời gian do root đọc lại. **Không có gì ở đây đã chạy trên GPU**; dispatch kernel là "suy từ nguồn" cho tới khi có Nsight.

## 1. Bốn đối tượng phải phân biệt

| Đối tượng | Định nghĩa | Ai tạo ra | Đi đâu |
|---|---|---|---|
| Quỹ đạo được lấy mẫu | Một mẫu của expert flow-matching từ nhiễu ban đầu, trong khung ego | Sampler của expert | Ứng viên |
| Quỹ đạo được chọn | Ứng viên được dùng làm kế hoạch: (E) 1 mẫu, chỉ số [0, 0, 0]; (G) 1 mẫu, `IdentitySelector` (lane A) | Driver | Controller; với AlpaGym, cùng đường SDE được ghi để replay |
| Lệnh điều khiển | Đầu ra controller (MPC) bám kế hoạch | Controller AlpaSim | Động lực học xe |
| Đường đi thực thi | Tư thế ego sau động lực học, theo timestamp | Physics AlpaSim | Scorer, reward, quan sát kế tiếp |

`distance_to_gt` trong reward đo **đường đi thực thi** so với log (RMSE XY theo thời gian chồng lấn), không phải quỹ đạo dự đoán (`nvidia-stack.md`, `distance_to_gt.py`).

## 2. Đường quan sát → hành động: hai đường đi của cùng trọng số

(E) = driver Alpamayo 1.5 của AlpaSim `e2e_challenge` `e121e59` với alpamayo1.5 `36aeb4c` (công cụ đo S2, và là nơi checkpoint Gym được chuyển tới để đánh giá). (G) = rollout AlpaGym `972d160` với recipes PR #18 `1d99fc5`. Cột "Root" ghi hàng nào root đã đọc lại nguồn; các hàng còn lại dựa trên `lane-A.md`.

| Đại lượng | (E) đánh giá | (G) huấn luyện | Root |
|---|---|---|---|
| Camera | 4: cross_left, front_wide, cross_right, front_tele, sắp theo chỉ số | Giống | ✓ (G) |
| Frame render | 1080 cao, 1916/1920 rộng, 10 Hz | Cùng cấu hình renderer | — |
| Resize | Processor `smart_resize` → **320×576**, 180 token/ảnh, 2.880 token/lần gọi | `tvf.resize` → **320×512**, 160 token/ảnh, 2.560 token/lần gọi; tỉ lệ ngang/dọc 0,902 so với ảnh gốc | ✓ cấu hình và code resize; số token tái lập bằng `derive.py` |
| Frame mỗi camera | 4, gần nhất với t0 − {300, 200, 100, 0} ms | Ring 4 frame | — |
| Lịch sử ego | 16 tư thế, 0,1 s | 16 tư thế đã đệm | ✓ (G, cấu hình) |
| Prompt | "…chain-of-thought…, then output the future trajectory", kết thúc `<|cot_start|>` | "output the future trajectory.", kết thúc `<|traj_future_start|>` | ✓ (G) |
| Prefix | ≈3.090 token + CoC | ≈2.730 token | ước tính của lane A ±3% |
| Decode tự hồi quy | HF `generate`, T 0,6, top-p 0,98, tối đa 128 token | Không | ✓ (E: tham số; G: bỏ) |
| Backbone | Qwen3-VL-8B (Cosmos-Reason2-8B): ViT 0,576 B, LLM 8,22 B | Cùng trọng số | Tổng 11.078.526.194 tham số tái lập khớp safetensors (root chạy lại) |
| Expert | 36 lớp Qwen3, hidden 2048, 8 KV head, 2,279 B; **attention chung với KV của prefix VLM theo từng lớp**, không phải cross-attention; không causal | Cùng | — |
| Hành động | 64 × 2: gia tốc và độ cong chuẩn hoá mỗi 0,1 s, tích phân unicycle; chân trời 6,4 s; khung rig tại t0 (x tiến, y trái), mét | Cùng | — |
| Khử nhiễu | Euler **ODE**, 10 bước, `x0 = randn·T` từ RNG toàn cục | **SDE**, noise 0,2, 5 bước (4 ở smoke), generator theo hàng có seed | ✓ cả hai |
| Mẫu và chọn | 1 mẫu, chỉ số [0, 0, 0] | 1 mẫu, `IdentitySelector` | — |
| Tần số policy | 10 Hz, warm-up 1,7 s, `assert_zero_decision_delay` | 5 Hz, warm-up 1,6 s, 22 lần gọi policy | ✓ (G, preset) |
| Controller | MPC phi tuyến | MPC tuyến tính (mặc định của pin `46b401b`) | ✓ `base_config.yaml` hai pin |
| Batch | 1, vòng lặp tuần tự | Gom 1–4 hàng từ nhiều phiên tuỳ thời điểm | — |
| Dtype, attention | BF16, `sdpa` bị ép; attention của ViT tách theo từng ảnh (432 lần gọi SDPA mỗi lần gọi policy) | BF16, `sdpa` | — |
| CUDA graph / compile | Không bật (PR #36 là opt-in) | Không có trong dòng PR #18 | — |

**Dispatch kernel là suy từ nguồn.** Capture xác nhận: Nsight Systems `--trace=cuda,nvtx,osrt,cublas` với NVTX quanh ViT, prefill, từng bước `generate` và từng bước expert, sau warm-up; Nsight Compute trên năm kernel lớn nhất; log `attn_implementation` và `torch.backends.cuda.*_sdp_enabled()` khi nạp. Lưu ý vận hành: mọi report nsys phải được quét credential, kể cả SQLite và archive, trước khi lưu (commit `3741d92`).

### Roofline (lane A, suy ra, BF16, batch 1, RTX PRO 6000 Blackwell Workstation)

Đỉnh BF16 dense 503,8 TFLOPS, FP8 dense 1.007,6, băng thông 1.792 GB/s (whitepaper RTX PRO Blackwell v1.1, bảng 4); ridge 281 FLOP/B cho BF16. Điều này sửa câu "NVIDIA prints no 504" trong `04` §3.

| Giai đoạn | (G) | (E) | Chế độ |
|---|---|---|---|
| ViT, 16 ảnh | 1,01e13 FLOP, ≈20 ms ở đỉnh | 1,15e13, ≈23 ms | compute |
| Prefill LLM | 4,01e13, ≈80 ms | 4,57e13, ≈91 ms | compute |
| Decode CoC | — | ≈8,7 ms/token (đọc 15,6 GB trọng số mỗi token) | bộ nhớ |
| Expert mỗi bước | 3,4e11 FLOP, 4,96 GB, ≈2,8 ms | như (G) | bộ nhớ tới batch ≈6–8 |
| Cận một lần gọi | ≈117 ms | ≈281 ms ở 16 token CoC | — |

Đây là cận dưới theo đỉnh, chưa gồm overhead launch eager. PR #36 báo −15,5% latency mỗi lần gọi trong tích hợp AlpaGym nội bộ của NVIDIA, phù hợp với việc launch overhead của expert là đáng kể (một mẫu, sơ bộ).

**Tỉ trọng prefill VLM đóng băng trong trainer replay:** lane A ước 90,6% FLOP mỗi hàng, lane B ước ≈94%; hai con số dưới giả định khác nhau (số token expert, phần backward, độ dài prefix). Cả hai là suy ra, không phải đo.

## 3. Thời gian: tính toán có đi vào động lực học mô phỏng không?

| Câu hỏi | Trả lời | Nguồn |
|---|---|---|
| Thời gian suy luận thật có làm xe đi thêm trong mô phỏng không? | **Không.** Runtime `await` driver không deadline (lockstep); wall-clock dừng thời gian mô phỏng | Capstone report §3.5, đọc tại `e2e_challenge e121e59` |
| Độ trễ cấu hình | `planner_delay_us`, mặc định 0; `DelayBuffer` đọc một lần mỗi control tick, trả nguyên kế hoạch, không nội suy; độ cũ áp dụng = ⌈L/Δt⌉·Δt | Cùng nguồn; Δt = 100 ms ở challenge |
| Hệ quả cho tối ưu hiệu năng | Tăng tốc policy hoặc renderer chỉ mua **thông lượng wall-clock** (rollout/giờ, đô la/rollout), không đổi kết cục lái mô phỏng | Suy từ hai dòng trên |
| Tần số | Waypoint spacing không xác lập tần số lập kế hoạch; AlpaGym smoke dùng 5 Hz sau 1,6 s GT cưỡng bức, 22 bước hợp lệ | `nvidia-stack.md`; preset smoke |

Vì vậy mọi ứng viên "nhanh hơn" chỉ có giá trị ở ranh giới **hạ tầng mô phỏng/RL** trong nghiên cứu này. Giá trị ở ranh giới **xe** (độ tươi quan sát, deadline) cần thiết bị đích (S6/S7, đang hoãn).

## 4. Vòng học AlpaGym (pin `972d160`, recipes PR #18 `1d99fc5`, Cosmos-RL `d2a2c57`)

| Bước | Hành vi | Nguồn |
|---|---|---|
| 1. Nhóm | `n_generation` rollout của cùng một scene; smoke 2, `clrl_test_run` 6, mặc định 1 | Preset pinned |
| 2. Seed | Seed phiên cộng bộ đếm inference; generator theo hàng | `nvidia-stack.md` (policy.py, inference_model.py) |
| 3. Lấy mẫu | Expert continuous, VLM đóng băng (`cotrain_vlm=false`), không sinh CoT; SDE được ghi cho replay; `inference_step` 4 (smoke) | `nvidia-stack.md`; smoke dòng 18 |
| 4. Reward | Aggregate online của scorer AlpaSim `46b401b` + `distance_to_gt`; `progress_safety` cho baseline A | `conf_progress_safety.yaml` |
| 5. Thất bại | Retry với session UUID mới (và do đó seed mới) **trước** khi tính reward/advantage; nhóm là `n_target` episode thành công đầu tiên; hết lượt thì worker trả rỗng nhưng `rollout_generation` **ném `RuntimeError`**, nên không có nhóm thiếu im lặng. Điều này sửa câu "payload dropped" của audit 11-01 | `streaming_worker.py` dòng 233–319; `rollout_backend.py` dòng 234–241 (lane B) |
| 6. Advantage | `(r − mean)/(std + 1e-6)` theo nhóm, std quần thể | `grpo.py` dòng 38–47 |
| 7. Lọc độ cũ | Giữ số lượng mới nhất; cảnh báo chứ không loại mẫu cũ; `allowed_outdated_steps` 4 ở mặc định | `rollout_filter.py`; `conf_default.yaml` dòng 149 |
| 8. Dữ liệu huấn luyện | Một advantage mỗi rollout chép cho mọi hàng bước hợp lệ; padding nhận 0 | `trainer.py` dòng 267–299 |
| 9. Cập nhật | `grpo_optimization_iterations × ⌈rows / mini_batch⌉` optimizer step mỗi trainer step, bỏ qua minibatch gradient 0; `mini_batch: 1` | `trainer.py` dòng 301+; `conf_default.yaml` dòng 151–158 |
| 10. Ratio | `exp(clamp(mean log-density diff, ±5))`, clip 0,8–1,2; log-density trung bình trên mọi chiều không phải batch | `replay_objective.py`; `flow_matching.py` dòng 538–587 |
| 11. KL | `kl_beta: 0`; không dựng reference | `conf_default.yaml` dòng 160 |
| 12. Lưu checkpoint | `save_freq: 20`, `max_keep: 5` | `conf_default.yaml` dòng 144–147 |
| 13. Export | Chuyển checkpoint Gym sang định dạng inference; đổi tên config/tensor | `convert_alpagym_checkpoint_to_inference.py` |
| 14. Đánh giá | AlpaSim nhánh đánh giá (khác pin của Gym), gate PhysLoop | Draft S2 |

**Bổ sung của lane B (nguồn, chưa chạy):** nhóm là một payload của một scene, gồm `n_generation` episode AlpaSim độc lập; mọi anh em trong nhóm nhận cùng quan sát ở lần gọi policy đầu tiên (trừ nhiễu renderer); warm-up không gọi driver nhưng vẫn render mọi frame (≈26,7% frame). Độ trễ cho phép tới 4 trainer step; trọng số mới được ghi thẳng vào tham số đang dùng, không có fence; `weight_version` được đóng dấu theo payload, còn mỗi hàng ghi 0. Khi `train_batch_per_replica × replicas` không chia hết cho G, một nhóm bị chia qua hai trainer step; preset smoke chỉ huấn luyện trên một trong hai anh em. `perf` tắt mặc định và `train/iteration_time` luôn bằng 0,0.

Ba trục thời gian khác nhau: bước lái (5 Hz trong Gym), bước khử nhiễu bên trong expert (`inference_step`), và bước optimizer. Một advantage của cả tập rơi lên mọi bước lái; không có credit theo sự kiện.

## 5. Danh tính scene/phiên qua pipeline

| Điểm | Danh tính được giữ | Rủi ro |
|---|---|---|
| Lập lịch AlpaSim | `RolloutSpec`, `rollout_idx`, seed phiên = `random_seed + rollout_idx` (main) | Seed chồng lấn; seed 0 = ngẫu nhiên |
| Retry AlpaGym | UUID phiên mới mỗi lần thử | Scene khó có thể biến mất khỏi dữ liệu huấn luyện mà không để dấu nếu không có ledger riêng |
| Summary đánh giá | `run_uuid`, `clipgt_id`, `rollout_id`; `rollout_uid` là dense rank | Không bảo đảm road, stratum, replicate khớp giữa policy (`2026-10-31-s2-upstream.md`) |
| Gate PhysLoop | `scene_id`, `road_id`, `stratum`, `replicate_id` bắt buộc; trùng lặp bị từ chối | `road_id` chưa có nguồn đăng ký |
| Xoá offline | Quỹ đạo `img_is_black` bị xoá | Thất bại có thể biến mất khỏi tử và mẫu số |
