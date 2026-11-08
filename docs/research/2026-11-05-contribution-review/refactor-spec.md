# Đặc tả refactor và đóng góp

Agent-authored, 2026-11-05. Đặc tả, không phải code đã viết. Không file khoá nào bị sửa; không PR nào được mở. Mỗi mục ghi rõ ai phải quyết trước khi làm.

## 0. Ranh giới patch chung: một driver variant "trained-path" cho Alpamayo 1.5 trong AlpaSim

Ứng viên chính (đồng nhất đường policy, `candidates.md` §2) và dự phòng (CRN, §3) rơi vào **cùng một ranh giới patch**: một model variant của driver Alpamayo 1.5 trong AlpaSim. Mục 1 dưới đây là phần seed của variant đó.

### 0.1 Hành vi hiện tại và đề xuất

| Khía cạnh | Hiện tại (`e2e_challenge` `e121e59`, alpamayo1.5 `36aeb4c`) | Đề xuất (opt-in, mặc định giữ nguyên) |
|---|---|---|
| Prompt | Luôn sinh CoC (`..._with_vlm_rollout[_cfg_nav]`) | `prompt_mode ∈ {cot, traj_future}`; `traj_future` chỉ prefill rồi chạy expert, như AlpaGym |
| Bộ tích phân | Euler ODE 10 bước, `randn` toàn cục | `integrator ∈ {ode, sde}`, `steps`, `noise_level`, cấu hình được; mặc định giữ ODE 10 |
| Kích thước ảnh | Processor tự `smart_resize` → 320×576 | `pre_resize: null | [H, W]` tường minh, để tái tạo đúng 320×512 của Gym khi cần chẩn đoán |
| Seed | Không tiêu thụ | Generator theo phiên, seed suy bằng băm (§1) |
| Manifest | Không ghi các trường trên | Driver ghi sáu yếu tố và seed vào log mỗi phiên |

Phần code mới thật sự: một hàm lấy mẫu không-CoC cho lớp `Alpamayo1_5` (lớp phát hành chưa có; AlpaGym có `_sample_trajectories_from_data_without_vlm_rollout` trong dòng PR #18) và đường SDE. Không đổi trọng số, scorer, controller.

### 0.2 Thành phần chịu ảnh hưởng

Driver AlpaSim (`models/alpamayo1_5_model.py`, `models/alpamayo_base.py`, config driver), manifest S2 của PhysLoop, adapter. Phía AlpaGym chỉ có overlay cấu hình cục bộ (`input_size`), vì AlpaGym không nhận PR.

### 0.3 Thứ tự tích hợp

1. Huy chọn estimand đánh giá ở S2.1 (trained-path hay deployed-path); variant vẫn cần cho arm chẩn đoán dù chọn bên nào.
2. CPU: test cấu hình; test rằng `traj_future` không gọi `generate`; test hình học ảnh với JPEG tổng hợp; test seed.
3. GPU, sau S2 admission và trong trần đã duyệt: E25 (bên dưới) trên checkpoint gốc.
4. Khi có checkpoint RL đầu tiên: lặp E25.
5. Chỉ sau đó cân nhắc PR AlpaSim, sau khi kiểm lại release mới nhất (#224), PR đang mở và #208; chỉ đính kèm test CPU và số liệu được phép công bố (KC-L).

### 0.4 Kiểm tra

| ID | Input | Kết quả bắt buộc |
|---|---|---|
| E25 | Cùng tick đã ghi, cùng trọng số; giai thừa 2³ (ảnh × CoC × integrator), K mẫu mỗi ô | Báo `D_f` và `D_within` theo mét với khoảng bất định; ngưỡng do Huy đặt |
| E26 | Variant ở cấu hình mặc định | Bitwise giống driver stock (cùng seed nếu đã port seed; nếu không, cùng phân phối) — variant không được đổi đường stock |
| E27 | `prompt_mode: traj_future` + `integrator: sde` + `pre_resize: [320, 512]` trên một tick đã ghi, cùng seed hàng như Gym | Khớp đầu ra của `AlpamayoR1InferenceModel` của Gym trong dung sai số học đã đăng ký |

### 0.5 Fallback

Nếu không qualify được variant trong trần, giữ driver stock, ghi sáu yếu tố như trường đăng ký, và báo mọi kết luận S4.6 kèm caveat "đường đánh giá khác đường huấn luyện ở sáu yếu tố; chưa đo tác động".

## 1. Phần seed của variant: "common random numbers" (CRN) cho so sánh ghép cặp

Phía AlpaGym, cùng mục tiêu được phục vụ bằng một harness cục bộ (lane B, C2): wrapper quanh `build_policy_factory` ánh xạ `session_uuid → seed đã ghi`, một replay driver gửi lại (scene, UUID, seed) ở trọng số đông cứng, ở batch 1 và ở batch của run, và ledger ghi seed, UUID, weight version, hash mẫu được chọn mỗi bước. Test CPU dùng `AlpamayoPolicy` và `InferenceEngine` thật với `InferenceModel` giả.

### 1.1 Hành vi hiện tại (root đọc nguồn, 2026-11-05)

| Đường code | Hành vi | Nguồn |
|---|---|---|
| AlpaSim `e2e_challenge` @ `e121e59` (pin của draft S2) | Driver Alpamayo lưu `request.random_seed` nhưng không truyền vào inference; sampling không reseed → replicate **unseeded** | `docs/research/2026-10-31-s2-upstream.md`, mục Seed mode |
| AlpaSim `main` @ `affc2ea`, runtime | Seed phiên của rollout k = `spec.random_seed + k`; `random_seed = 0` nghĩa là "chọn ngẫu nhiên" | `src/runtime/alpasim_runtime/daemon/engine.py` dòng 210–212; `event_loop.py` dòng 574; PR #149 (merged 2026-08-17) |
| AlpaSim `main`, driver | `inference_seed = session.seed + session.inference_count` | `src/driver/src/alpasim_driver/main.py` dòng 713 |
| AlpaSim `main`, Alpamayo | Nếu `force_determinism`: `torch.manual_seed(inference_seed)` toàn cục mỗi lần `predict`, kèm `torch.use_deterministic_algorithms(True)` | `models/alpamayo_base.py` dòng 54–63, 612–615 |
| AlpaSim `main`, traffic và renderer | PR #149 ghi: trafficsim chỉ log seed; đường video-model renderer có sampling riêng; không hướng tới replay bit-exact | Mô tả PR #149 |
| AlpaGym `972d160` | Seed phiên cộng bộ đếm inference, generator theo từng hàng | `nvidia-stack.md` (policy.py, inference_model.py) |

### 1.2 Vấn đề cụ thể

1. **Luồng ngẫu nhiên chồng lấn [đại số trên nguồn].** Với phép cộng, replicate k ở lần gọi i dùng seed `S + k + i`; replicate k + 1 ở lần gọi i − 1 dùng **cùng** seed. Mỗi replicate thấy cùng một dãy nhiễu, chỉ lệch một tick. Nếu nhiều scene dùng chung `S`, các scene cùng chỉ số replicate dùng chung dãy nhiễu. Replicate vì thế không phải các lần rút độc lập; mức độ ảnh hưởng tới kết cục chưa đo **[giả thuyết]**, nhưng giả định độc lập mà ước lượng ICC và cận Chernoff theo road dựa vào không còn được bảo đảm bởi thiết kế.
2. **Seed 0 bị coi là "không seed" [nguồn].** `if spec.random_seed else 0` và `session_seed or None`. Đặt seed 0 lặng lẽ cho kết quả không tái lập được.
3. **RNG toàn cục [nguồn + giả thuyết].** `torch.manual_seed` toàn cục đặt lại mọi luồng RNG torch của tiến trình. Ở batch size 1, một phiên mỗi lần gọi, điều này đủ. Khi driver gộp nhiều phiên vào một batch, nhiễu của một phiên phụ thuộc vào thành phần batch; khi đó vừa mất tái lập, vừa mất ghép cặp giữa hai arm.
4. **Nhánh đánh giá hiện tại không tiêu thụ seed.** Trên `e121e59`, không có ghép cặp nhiễu giữa hai arm; mọi so sánh là cặp "noisy" theo nghĩa của S1.

### 1.3 Thay đổi đề xuất (phạm vi nhỏ nhất)

**(a) Phía AlpaSim, ứng viên PR upstream sau khi kiểm lại không có PR cạnh tranh:**
- Suy seed phiên và seed mỗi lần gọi bằng hàm băm không chồng lấn, ví dụ `numpy.random.SeedSequence([run_seed, scene_hash, replicate_idx])` cho phiên và `SeedSequence([session_seed, inference_count]).generate_state(1)` cho mỗi lần gọi, thay cho phép cộng.
- Phân biệt "chưa đặt" bằng `None` (hoặc trường `has_random_seed`) thay vì giá trị 0.
- Dùng `torch.Generator` riêng cho mỗi phiên và truyền vào sampler thay cho `manual_seed` toàn cục, để batching không làm đổi nhiễu của phiên.
- Giữ `force_determinism` như cờ riêng cho thuật toán tất định; CRN không đòi hỏi tất định bit-exact.

Đây là thay đổi về **cách sinh số ngẫu nhiên**, không đổi mô hình, scorer, controller hay estimand. Nó đổi chuỗi seed cụ thể so với hiện tại, nên một rollout đã ghi trước đây sẽ không tái lập được bằng seed cũ; cần cờ tương thích hoặc ghi rõ trong CHANGELOG.

**(b) Phía PhysLoop, kỹ thuật evaluator do agent sở hữu theo D-018, chỉ sau khi Huy chọn seed mode:**
- Manifest: `seed_mode ∈ {unseeded, crn_seeded}`; `master_seed`; hàm suy seed và phiên bản của nó; `pairing_key = (road_id, scene_id, replicate_id)`.
- Lịch seed được sinh **trước** khi chạy từ `master_seed` đã đăng ký; lưu hash của lịch; cấm chọn hoặc thay seed sau khi xem kết cục.
- Adapter kiểm: mỗi cặp A/B có cùng seed; mỗi (road, scene, replicate) có seed khác nhau; báo lỗi nếu trùng.
- Chẩn đoán coupling: tick phân kỳ đầu tiên giữa hai quỹ đạo cùng seed; khoảng cách quỹ đạo theo thời gian; discordance của endpoint nhị phân; ước lượng f = discordance(CRN) / discordance(độc lập) với khoảng bất định.
- Forecast power dùng `gate.verdict.paired_road_bounds` (import chỉ đọc), như `evidence/contribution-review-20261105-v1/crn_power.py`, với R và f đo được thay cho giá trị minh hoạ.

**(c) Port cho nhánh đánh giá.** Nếu Huy giữ `e2e_challenge` làm nhánh đánh giá: một "diagnostic port" chỉ thêm đường seed vào driver Alpamayo, pin riêng, qualify riêng, đúng như `S2_ENGINEERING_SPEC.md` dòng 115 đã dự liệu. Không trộn với thay đổi khác của `main`.

### 1.4 Giao diện bị ảnh hưởng

`RolloutSpec`/runtime config (seed), driver `PredictionInput.inference_seed`, sampler Alpamayo (generator), manifest S2 schema, `gate/alpasim_adapter.py` (kiểm cặp seed), một module forecast power mới **ngoài** `gate/verdict.py` và `gate/stats/`. Không chạm file khoá.

### 1.5 Thứ tự tích hợp

1. CPU: test đơn vị cho hàm suy seed (không chồng lấn trên lưới 10⁶ cặp; seed 0 hợp lệ), cho kiểm cặp trong adapter, và cho forecast (tái lập bảng `scientific-audit.md` §3).
2. Chờ Huy chọn seed mode và nhánh (M0, M1 trong `revised-plan-proposal.md`).
3. GPU trong pilot đã duyệt: E22 (cùng policy cùng seed), E23 (xáo seed, đối chứng âm).
4. S2 full: đo f cho cặp teacher–re-export trên toàn bộ schedule.
5. S4 engineering qualification: đo f cho teacher–snapshot RL sớm.
6. Chỉ sau 1–5: viết issue/PR AlpaSim với test và số đo được phép công bố (KC-L chặn số suy ra từ NuRec; test CPU và số tổng hợp thì công bố được).

### 1.6 Kiểm tra đúng và hiệu năng

- **Không lệch:** tỉ lệ biên của teacher dưới `crn_seeded` và `unseeded` không khác nhau ngoài nhiễu (kiểm bằng so sánh hai mẫu độc lập, không phải bằng gate NI).
- **Có ghép cặp:** E22 cho `d_r = 0` hoặc phân kỳ có nguyên nhân được ghi; E23 đưa discordance về mức độc lập.
- **Không chồng lấn:** test CPU trên hàm suy seed.
- **Chi phí:** thời gian mỗi tick có và không có `force_determinism` (thuật toán tất định có thể chậm hơn); CRN tự nó không cần cờ này.

### 1.7 Fallback

Nếu port không qualify, hoặc f ≈ 1 đối với checkpoint RL, quay lại `unseeded` như draft hiện tại, ghi forecast INSUFFICIENT dự kiến và để Huy quyết định tăng R, tăng margin hay dừng ở feasibility. Không có thay đổi nào cần rollback ở file khoá.

### 1.8 Ai quyết

- Huy: seed mode trong registration khoa học; nhánh đánh giá; việc thêm probe vào pilot; công bố.
- Agent (D-018): implementation evaluator, test, forecast, audit.
- Không cần unlock file khoá.

## 2. Bổ sung extraction và reward fixtures (S2.3, S4.3)

| Thay đổi | Ranh giới patch | Kiểm tra | Ai quyết |
|---|---|---|---|
| Fixture đa-actor cho `CollisionScorer` (E24) | Test mới dưới `tests/`, dùng helper công khai ở phiên bản đã pin, cạnh bản sửa của PR #234 | Known-answer: front + rear cùng timestep | Agent |
| Trường `render_status ∈ {clean, black_at_t, nonfinite_pose_at_t, unverified}` (lane C-1): đòi `img_is_black` phủ mọi tick, ghi tick đen đầu tiên, kiểm pose hữu hạn; bảng 2×2 (A hợp lệ, B hợp lệ) theo stratum | Extractor và `gate/alpasim_adapter.py` (không khoá) | Fixture: ảnh đen ở tick t rồi va chạm ở t + 3 → cách ly; thiếu `img_is_black` → `unverified` | Agent; Huy chọn loại trừ, imputation bất lợi hay outcome riêng |
| Danh tính scene theo nội dung (lane C-2): revision theo commit SHA, SHA-256 và phiên bản NRE mỗi USDZ trong manifest và ledger; kiểm hash cache trước mỗi đơn vị; chặn ghép cặp khi lệch | Catalog đông cứng (`scenes.scenes_csv`), `gate/s2_preflight.py`, extractor | Fixture: hai hash khác nhau cho cùng `scene_id` → từ chối ghép cặp | Agent; không cần upstream |
| Luật dữ liệu thiếu: imputation bất lợi (B thiếu → 0, A thiếu → 1) qua `verdict` không sửa; chỉ báo GO nếu phân tích bất lợi là GO (lane C §4) | Module phân tích ngoài file khoá | Tái lập `missing_sensitivity.json` | Huy đăng ký luật trước khi có kết cục |
| Báo `collision_rear` riêng cạnh `collision_any` và at-fault | Báo cáo S4.3 | Fixture xe sau tông khi ego phanh | Agent |
| Fixture warm-up và sau sự kiện đầu (E10 đã có) | Giữ | — | — |

## 3. Schema-v3 cho draft S2 (đề xuất)

Thêm `policy_arms` (liệt kê teacher và re-export với hash checkpoint), `seed_mode`/`master_seed`/`seed_derivation`, `road_identity_source`, `strata_road_counts`, `driver_path` (sáu yếu tố: `prompt_mode`, `input_hw`, `integrator`/`steps`/`noise_level`, `policy_hz`, `controller`, `cfg_nav`), `scene_assets` (commit revision, SHA-256, phiên bản NRE mỗi scene) và `missing_data_rule`. Phía S4.1 thêm các guard của lane B: `prefetch_rollout: false`; `allowed_outdated_steps` và `on_policy`; `train_batch_per_replica × policy_replicas ≡ 0 (mod n_generation)`; `force_determinism` cùng `max_batch_size`; số lần `retry_on_nonfinite`; SHA AlpaSim dùng cho huấn luyện. Validator mới từ chối khi số road mỗi stratum nhỏ hơn sàn của margin đã đăng ký, với thông báo nêu sàn đó. Đây là thay đổi ở `gate/s2_preflight.py` (không khoá), do agent viết sau khi Huy đồng ý schema.

## 4. Không refactor

- `gate/verdict.py`, `gate/stats/`: giữ nguyên; bound không phải nút cổ chai (A9).
- Trainer AlpaGym, estimator, reward: không đổi trong đặc tả này. Thay đổi estimator/anchor là thiết kế của Huy sau baseline.
- Không chuyển sang AlpaSim `main` chỉ vì mới hơn; mọi đổi nhánh đi qua M0.

## 5. Cập nhật Sổ tay PhysLoop cho v5

Artifact hiện tại (`D2FzQqKtMGDHeCQENWfwL7`, bản 2026-10-29) mô tả v4.1. Cần sửa, tách rõ nội dung đã thông qua và đề xuất:

| Phần | Trạng thái | Sửa thành |
|---|---|---|
| Header, pills | "chưa có code", KC-E, capstone #1 | S1 qualified (D-016), S2 chuẩn bị xong, host check đóng (D-020), v5 (D-021/D-022) |
| Phần 0, 1.1, 1.4, 1.5 | Thesis thầy → trò → Thor, câu hỏi phát hành | Câu hỏi G9 của v5; trò/Thor là hướng hoãn |
| 2.4 Đồ thị tám stage | Tám stage cùng hoạt động | S1 → S2 → S4; S3, S5–S8 hoãn |
| 3.F, 3.H, 3.J–3.N | Trò, sửa trò, batching, độ trễ, Thor, HIL, shadow | Chuyển sang phụ lục "hướng hoãn" |
| 3.E Thống kê ghép cặp | Đúng về gate | Thêm sàn số road, discordance và CRN (đề xuất, có nhãn) |
| 3.G RL cho thầy | v4.1 | Viết lại theo D-022: group size 2, tỉ số hình học, KL = 0 |
| Phần 4 Capstone, lịch 12 tuần | Đề xuất P-020…P-029 | Đánh dấu lịch sử; không lịch mới khi P-003 còn mở |
| 6.1 Bài tập | Một số theo v4.1 | Thêm bài tập sàn road và advantage nhóm hai; ô ★ để trống |

Việc cập nhật artifact là một nhiệm vụ riêng (đọc đủ 2.316 dòng trước khi publish lại).
