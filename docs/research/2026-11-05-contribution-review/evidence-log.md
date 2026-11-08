# Evidence log và bản đồ nguồn

Agent-authored, 2026-11-05. Mỗi hàng là một khẳng định mang tải cho quyết định. **Loại:** `exec-CPU` chạy trên máy này; `nguồn` đọc code tại SHA ghi rõ; `doc` tài liệu/README/issue nói vậy; `đại số` suy ra; `agent-A/B/C` do research agent đọc, root chưa đọc lại (xem `verification.md`); `giả thuyết` chưa kiểm. Ngày đọc là UTC.

## A. Trạng thái repo PhysLoop (đọc/chạy 2026-11-05)

| # | Khẳng định | Nguồn / phiên bản | Loại | Phạm vi | Bất định | Hệ quả |
|---|---|---|---|---|---|---|
| A1 | HEAD `3741d92` trên `s1-scaffold`, worktree sạch trước review | `git rev-parse HEAD` | exec-CPU | repo | — | Mọi tham chiếu code local dùng SHA này |
| A2 | `uv run --frozen pytest -q`: 756 passed trong 11,9 s | log trong `evidence/contribution-review-20261105-v1/` | exec-CPU | CPU, Python 3.12, lock hiện tại | Không chạy lại trong checkout độc lập | "723/754 test" trước đây nay là 756; S1 qualification không đổi |
| A3 | "441" chỉ là ràng buộc số lượng trong validator; `scene_list`, `road_list`, `road_independence_justification` đều `null` ở hai draft | `gate/s2_preflight.py` dòng 316; `experiments/S2_*_DRAFT.json` | nguồn | S2 registration | — | Population chưa có danh tính; số road độc lập chưa biết |
| A4 | Draft full: `replicates: 3`, `seed_mode: unseeded`, `force_determinism: false`, branch `e2e_challenge` @ `e121e59`, renderer `nre-ga@sha256:97f43e…`, teacher revision `7aba829…`, `total_cap_usd: 1100` | `experiments/S2_FULL_DRAFT.json` | nguồn | S2 | Draft, chưa đăng ký | CRN và pin nhánh cần quyết định mới |
| A5 | Draft không có trường liệt kê arm (teacher/re-export); phạm vi hẹp của D-021 chỉ nằm trong văn bản spec | `S2_FULL_DRAFT.json`; `S2_ENGINEERING_SPEC.md` dòng 47 | nguồn | S2 | — | Cần trường `policy_arms` trong schema-v3 (đề xuất) |
| A6 | Gate: GO khi `UB(mean d+) − LB(mean d−) < margin` mọi stratum, KL-Chernoff α/2 mỗi bên, đếm road là đơn vị độc lập; Holm regression `p = exp(−R·max(mean,0)²/2)` | `gate/verdict.py` dòng 25–57, 176–221 (khoá, chỉ đọc) | nguồn | S1/S2/S4 | — | Power phụ thuộc R và discordance, không phụ thuộc trực tiếp số replicate |
| A7 | Sàn all-zero `1 − (α/2)^{1/R}`, α = 0,05: 36/51/72/183 road cho margin 0,10/0,07/0,05/0,02 | `bound_compare.json`, `crn_power.json`; khớp `operational_road_bounds.csv` của S1 | exec-CPU + đại số | mỗi stratum | — | Margin nhỏ bất khả thi khi R nhỏ |
| A8 | P(GO \| tương đương) tăng mạnh khi discordance giảm, yếu khi tăng replicate (bảng trong `scientific-audit.md` §3) | `crn_power.py`, seed 20261105, 400 trial/ô | exec-CPU | mô hình nhị phân đồng nhất, minh hoạ | p, f, R là giả định | Căn cứ cơ chế cho ứng viên chính |
| A9 | Empirical Bernstein lỏng hơn bound khoá ở mọi ô kiểm | `bound_compare.py` | exec-CPU | cùng mô hình | Chưa kiểm betting CS | Loại đề xuất đổi bound |
| A10 | Oracle D-022 tái lập: ±0,998; ±1,000; −√5; rms 0,9999; tỉ số 4 vs 2; 116,7 nat | `oracles.py` | exec-CPU | công thức pinned | — | Toán estimator giữ nguyên như D-022 |
| A11 | Sổ tay PhysLoop (artifact `D2FzQqKtMGDHeCQENWfwL7`, phiên bản `1793265996-c90c`) mô tả plan v4.1 ngày 2026-10-29: "chưa có code", thesis thầy→trò→Thor, capstone P-020…P-029, lịch 12 tuần | đọc artifact | nguồn | tài liệu học | Chỉ đọc phần đầu và mục lục | Cần cập nhật theo v5; xem `refactor-spec.md` §5 |
| A12 | Commit `3741d92` sửa probe: nsys/ncu/dcgmi chạy qua `env -i` sau `sudo`; chưa chạy trên VM thật. Memory dự án ghi Vast key trong file nsys trên đĩa chưa được xác nhận revoke | `git show 3741d92`; memory `vast-nsys-captures-credentials` | nguồn | host probe | Trạng thái revoke chưa biết | Hạng mục mở trong gói S2; không in, không thử key |

## B. Upstream NVIDIA (root đọc trực tiếp, 2026-11-05)

| # | Khẳng định | Nguồn / phiên bản | Loại | Phạm vi | Bất định | Hệ quả |
|---|---|---|---|---|---|---|
| B1 | AlpaSim `main` vẫn ở `affc2eab…` (2026-08-18) | `gh api repos/NVlabs/alpasim/commits/main` | nguồn | — | — | Không có thay đổi main từ 11-01 |
| B2 | AlpaSim `e2e_challenge` **đã di chuyển** tới `297b4d00cfdd2c4f636b9d9bcee7a85c699174c0` (2026-11-03 15:02 UTC), khác pin `e121e59` của draft | `gh api …/commits/e2e_challenge` | nguồn | Nhánh đánh giá S2 | Nội dung thay đổi: xem lane B/C | Pin draft cần quyết định: giữ `e121e59` hay qualify HEAD mới; không tự thay |
| B3 | `CollisionScorer` giữ phân loại của actor cuối mỗi timestep; cùng mã ở `46b401b`, `e121e59`, `main` | raw `src/eval/src/eval/scorers/collision.py` tại ba ref | nguồn | collision_front/rear/lateral, at-fault | Tần suất đa-actor chưa biết | Thêm known-answer đa-actor; không mở PR |
| B4 | Lỗi B3 đã được báo: PR #234 (mở 2026-11-01, erkara) và issue #237 (mở 2026-11-04, coldstone7) | `gh pr list`, `gh issue view 237` | doc | AlpaSim | Chưa merge | Không cạnh tranh; pin phiên bản và ghi rõ |
| B5 | Offline: `collision_at_fault = front ∨ lateral`; bỏ bước trước `eval_relevant`; cắt sau `offroad_or_collision` đầu tiên; ở `46b401b` (và `main`) **xoá nguyên quỹ đạo** có `img_is_black`; nhánh `e2e_challenge` đã gỡ modifier xoá này (PR #109) | AlpaSim `46b401b` `aggregation/processing.py` dòng 38–66; lane C cho challenge | nguồn (root: `46b401b`); agent-C (challenge) | đánh giá | — | Reward online (`collision_any` gồm rear) ≠ at-fault; ảnh đen bị xoá (main/Gym) hoặc chấm như lỗi lái (challenge) |
| B6 | `progress_safety = progress − 10·collision_any − 5·offroad − 0,01·distance_to_gt` | AlpaGym `972d160` `conf/reward/progress_safety.yaml` (bản sao trong `evidence/rl-gap-review-20261101-v1/pinned-source/`) | nguồn | S4 baseline A | — | Phân tích động cơ trong `scientific-audit.md` §5 |
| B7 | Advantage `(r − mean)/(np.std(r) + eps)`; nhánh `unbiased` chỉ trừ mean | Cosmos-RL `d2a2c57` `grpo.py` dòng 38–47 (bản sao pinned) | nguồn | S4 | — | Như D-022 |
| B8 | Một advantage mỗi rollout được chép cho mọi hàng bước hợp lệ; padding nhận 0 | AlpaGym `972d160` `trainer.py` dòng 267–299 (bản sao pinned) | nguồn | S4 | — | Không có credit theo thời gian |
| B9 | AlpaGym AR1.5 policy: `input_size: [320, 512]`, 4 camera, 4 frame ngữ cảnh, 16 waypoint lịch sử, 64 waypoint tương lai, BF16; `_decode_to_chw_uint8` resize đúng về kích thước đó (không giữ tỉ lệ) | AlpaGym `972d160` `configs/policy/alpamayo_r1.yaml` dòng 7–17; `policies/alpamayo/buffers.py` dòng 236–253 | nguồn (root) | Đường huấn luyện S4 | Kích thước render gốc theo lane B (1080×1916) | Ảnh huấn luyện có tỉ lệ 1,6 thay vì 1,78 |
| B10 | AlpaSim `main`: `MIN_FRAME_HW = (320, 576)` "What the vision processor resizes 16:9 renders to"; driver từ chối ảnh nhỏ hơn vì ảnh 320×512 "passes through untouched" | AlpaSim `affc2ea` `models/alpamayo_base.py` dòng 369–370; `main.py` dòng 843–851 | nguồn (root) | Đường đánh giá | — | NVIDIA đã chặn lệch này ở AlpaSim, chưa ở preset AlpaGym |
| B11 | Driver đánh giá gọi `sample_trajectories_from_data_with_vlm_rollout(top_p=0.98, temperature=0.6)`, tức sinh CoC trước khi chạy expert | AlpaSim `e121e59` `models/alpamayo_base.py` dòng 233–245, 318–329 | nguồn (root) | Đường đánh giá S2/S4.6 | Lớp con A1.5 có thể override `_run_inference` (xem lane A) | Expert ở gate được điều kiện hoá trên KV gồm CoC đã sinh |
| B12 | AlpaGym mặc định `last_component="traj_future"`: prompt kết thúc ở `<|traj_future_start|>` và bỏ bước sinh tự hồi quy (≈2× latency) | AlpaGym `972d160` `inference_model.py` dòng 33–51, 118–126 | nguồn (root) | Đường huấn luyện S4 | Nhánh `cot` tồn tại nhưng không preset nào bật | Expert được RL huấn luyện **không** có CoC; lệch điều kiện huấn luyện/đánh giá |
| B13 | Recipes `main` đã sang `c53aa09`; #42, #43, #44, #45 merged 2026-11-03…05; PR #18 vẫn mở, `dirty` | lane B | agent-B | Tham chiếu trong CLAUDE.md | — | "PR #42" nay là code đã merge, chưa release |

## C. Thị trường GPU công khai (dated cost evidence)

Truy vấn công khai không xác thực, không dùng API key, không thuê: `https://console.vast.ai/api/v0/bundles/`, 2026-11-05 16:34 UTC, `verified`, `rentable`, on-demand, 500 GB. Raw response trong `evidence/contribution-review-20261105-v1/vast-public-offers/`.

| Cấu hình | Số offer | Rẻ nhất `dph_total` | Rẻ nhất có `vms_enabled` | Hệ quả |
|---|---:|---:|---:|---|
| 1× RTX PRO 6000 WS | 9 | $1,472/h | $1,606/h | S2 pilot trong trần $2,50/GPU-h đề xuất |
| 2× RTX PRO 6000 WS | 10 | $3,286/h | $4,939/h (1 offer) | S4 hai GPU kiểu VM khan hiếm; $650 ≈ 132 giờ wall ở giá VM |
| 2× RTX PRO 6000 Server | 2 | $3,438/h | không có | |
| 2× H100 SXM | 1 | $6,556/h | không có | |

Giá niêm yết không phải báo giá đặt chỗ; chưa gồm truyền dữ liệu, thuế, phí; khả dụng thay đổi theo phút.

## D. Từ lane research A/B/C

Log đầy đủ của từng lane nằm cuối `lane-A.md`, `lane-B.md`, `lane-C.md`. Bảng này chỉ giữ các khẳng định mang tải cho quyết định; cột "Root" cho biết root đã đọc lại hay chưa.

| # | Khẳng định | Nguồn / phiên bản | Loại | Root | Hệ quả |
|---|---|---|---|---|---|
| D1 | Đường đánh giá dùng Euler ODE 10 bước, `randn` toàn cục; đường Gym dùng SDE 0,2, 5 bước, generator theo hàng | alpamayo1.5 `36aeb4c` `diffusion/flow_matching.py` dòng 35, 171; Gym `alpamayo_r1.yaml` dòng 33–37 | nguồn | ✓ | Yếu tố 4 của finalist chính |
| D2 | Lớp `Alpamayo1_5` chỉ có sampler CoC (`..._with_vlm_rollout`, `..._cfg_nav`); driver A1.5 chọn một trong hai | `alpamayo1_5.py` dòng 244, 440; AlpaSim `e121e59` `alpamayo1_5_model.py` dòng 112–121 | nguồn | ✓ | Driver stock không chạy được cấu hình đã huấn luyện |
| D3 | Gym ném `NotImplementedError` khi replay CoT | Gym `bundle.py` dòng 74–82 | nguồn | ✓ | Không bật CoC phía huấn luyện được |
| D4 | Converter Gym → driver `alpamayo1_5` của AlpaSim để đánh giá closed-loop | `convert_alpagym_checkpoint_to_inference.py` dòng 4–9 | nguồn | ✓ | Lệch nằm trên quy trình NVIDIA ghi |
| D5 | Controller: MPC tuyến tính mặc định ở `46b401b`, phi tuyến ở `e121e59` | hai `base_config.yaml` dòng 11 | nguồn | ✓ (Gym override: quét một phần) | Yếu tố 6 |
| D6 | 441 = `nurec_curated_val` (177 scene catalog 26.01, 264 scene 26.04), train 1.761, không chồng lấn; chỉ có trên `e2e_challenge` | `data/scenes/sim_suites_curated.csv` @ `297b4d0`, SHA-256 `326cb0eb…4936` | nguồn + CPU | ✓ đếm và hash; phần phiên bản NRE: agent-C | Đăng ký như danh sách lịch sử, kèm phiên bản NRE |
| D7 | Không có road/drive/GPS/session ID trong file công khai; phương pháp nhóm road chưa công bố | PR #167; README `local_evaluation` | nguồn | agent-C | R không đếm được trước license |
| D8 | Nhánh challenge gỡ modifier xoá ảnh đen (PR #109); `ScorerGroup` nuốt exception của scorer | `processing.py` @ `297b4d0`; `scorers/base.py` | nguồn | agent-C | `render_status` bắt buộc |
| D9 | USDZ 26.04 bị thay toàn bộ ngày 2026-07-30; `hf_revision` là tên nhánh; cache không kiểm hash | thẻ NuRec; `sceneset.py` dòng 445–505 | doc + nguồn | agent-C | Danh tính tài sản theo hash |
| D10 | Đường stock `simulate` của `main` không đặt `RolloutSpec.random_seed` | `simulate/__main__.py` dòng 145–195 @ `affc2ea` | nguồn | agent-C | CRN cần request trực tiếp hoặc variant |
| D11 | Ở pin Gym, seed phiên là `random.randint` và `RolloutSpec` không có trường seed | AlpaSim `46b401b` `driver_service.py` dòng 75; `runtime.proto` | nguồn | agent-B | Nhóm không replay được nếu không có harness |
| D12 | Re-sim cặp với seed mới đảo thứ tự với xác suất 1/2; replay cùng seed đảo với `arccos(ρ)/π` | đại số | đại số | ✓ (đại số đơn giản) | S4.4 dùng replay cùng seed |
| D13 | Retry trước reward; hết retry thì `RuntimeError` | `streaming_worker.py` dòng 233–319; `rollout_backend.py` dòng 234–241 | nguồn | agent-B | Sửa audit 11-01 |
| D14 | Nhóm bị chia qua trainer step khi không chia hết cho G; smoke huấn luyện một trong hai anh em | `status.py`, `config_validation.py`, preset | nguồn + đại số | agent-B | Guard S4.1 |
| D15 | Trainer tính lại prefill VLM đóng băng cho mọi hàng, mọi minibatch, mọi iteration | `bundle.py` dòng 55–114; PR #18 `model.py` | nguồn | agent-A, agent-B (độc lập) | Cache chỉ có lợi khi μ ≥ 2 |
| D16 | Cận một lần gọi ≈117 ms (Gym) và ≈281 ms ở 16 token CoC (driver) trên RTX PRO 6000 WS | `derive.py`; whitepaper v1.1 bảng 4 | đại số | known-answer tham số ✓; roofline: agent-A | Estimand đường đi đổi chi phí S2 |
| D17 | PR #36: graph khoá theo hình dạng KV, tối đa 4, không evict; −15,5% latency mỗi lần gọi, một mẫu sơ bộ trong tích hợp nội bộ | alpamayo1.5 PR #36 | nguồn + doc | agent-A | Không phải ứng viên; prior art |
| D18 | NIM 1.0.0 phục vụ Alpamayo 1.5, luôn trả `reasoning_text`, không công bố latency; RTX PRO 6000 không trong danh sách validate | docs NIM 1.0.0 | doc | agent-A | Đường triển khai NVIDIA chạy CoC |
| D19 | Gate: UB ≈ 2√(2·m₊·ln(2/α)/R); imputation bất lợi tiêu khoảng 2× tỉ lệ thiếu của margin | `power_tables.py`, `missing_sensitivity.py` | CPU + đại số | công thức khớp bảng root | Luật dữ liệu thiếu phải đăng ký trước |
