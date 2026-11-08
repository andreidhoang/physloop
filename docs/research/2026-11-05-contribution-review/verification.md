# Verification record

Agent-authored, 2026-11-05 (UTC; một số đọc sau nửa đêm giờ Việt Nam 2026-11-06). Ghi lại ai đã kiểm cái gì, chỗ nào root bất đồng với lane hay với tài liệu trước, và những gì chưa kiểm.

## 1. Phương pháp

- Root đọc `DOC_AUTHORITY.md`, `docs/00_START_HERE.md`, `02` (toàn bộ), `DECISIONS.md` D-016…D-022, journal 10-31 và 11-01, ba file review 11-01 (`report.md`, `nvidia-stack.md`, `gap-review.md`, `protocol-design.md`), `S1_HANDOFF.md`, hai draft S2, `S2_RENTAL_APPROVAL.md`, `S2_IMPLEMENTATION_PLAN.md`, file khoá `gate/verdict.py` và đầu `gate/stats/__init__.py` (chỉ đọc), và mục lục Sổ tay.
- Ba agent nghiên cứu mới (không fork) nhận brief có danh sách "đã thiết lập, không làm lại" và không thấy xếp hạng của nhau. Lane C có thấy thư mục review của root xuất hiện nhưng ghi rằng không đọc nó.
- Root viết bảng power của mình **trước** khi lane trả kết quả, để không bị neo bởi đề xuất của lane.
- Không thuê máy, không chấp nhận license, không tải tài sản có cổng, không chạy GPU, không đăng gì, không commit, không sửa file khoá, hook hay settings. Chỉ có một truy vấn thị trường công khai không xác thực (Vast `bundles`, không dùng API key).

## 2. Root đã kiểm trực tiếp

| Khẳng định | Cách kiểm |
|---|---|
| 756 test pass ở `3741d92` | Chạy `uv run --frozen pytest -q` |
| Bảng power, sàn road, so sánh bound | Chạy `crn_power.py`, `bound_compare.py` dùng hàm khoá (import chỉ đọc) |
| Oracle D-022 | Chạy `oracles.py` |
| Seed AlpaSim `main`: `random_seed + rollout_idx`; `inference_seed = seed + inference_count`; seed 0 = ngẫu nhiên; `manual_seed` toàn cục khi `force_determinism` | Đọc `engine.py` 210–212, `event_loop.py` 574, `main.py` 713, `alpamayo_base.py` 54–63, 612–615 tại `affc2ea`; mô tả PR #149 |
| Chồng lấn luồng seed | `seed_overlap.py` (số học thuần) |
| Không có issue/PR AlpaSim về chồng lấn seed hay ghép cặp seed | `gh search issues`, `gh pr list` với `seed`, `determinism` |
| `CollisionScorer` giữ phân loại của actor cuối; đã có PR #234, issue #237 | Đọc `collision.py` tại `46b401b`, `e121e59`, `main`; `gh` |
| `collision_at_fault = front ∨ lateral`; xoá quỹ đạo ảnh đen ở `46b401b` | Đọc `processing.py` 38–66 tại `46b401b` |
| `progress_safety`, advantage, chép advantage cho mọi bước | Bản sao pinned ngày 11-01 |
| AlpaGym `input_size: [320, 512]`, resize không giữ tỉ lệ | Đọc `alpamayo_r1.yaml`, `buffers.py` tại `972d160` |
| AlpaSim `main` sàn 320×576 và chú thích về 320×512 | Đọc `alpamayo_base.py` 369–370, `main.py` 843–851 tại `affc2ea` |
| Driver đánh giá luôn sinh CoC (hai sampler, đều CoC) | Đọc `alpamayo_base.py`, `alpamayo1_5_model.py` tại `e121e59`; `alpamayo1_5.py` 244, 440 tại `36aeb4c` |
| Gym bỏ CoC; replay CoT ném `NotImplementedError` | Đọc `inference_model.py`, `bundle.py` tại `972d160` |
| SDE 0,2 / 5 bước (Gym) và ODE 10 bước (alpamayo1.5) | Đọc `alpamayo_r1.yaml` và `flow_matching.py` |
| Converter ghi rằng đầu ra dành cho driver `alpamayo1_5` của AlpaSim | Đọc docstring tại `972d160` |
| Controller: MPC tuyến tính mặc định ở `46b401b`, phi tuyến ở `e121e59`; không thấy override trong YAML Gym đã quét | Đọc hai `base_config.yaml`; quét YAML Gym (một phần cây) |
| 441 = `nurec_curated_val`, 1.761 = `nurec_curated_train` | Đếm lại CSV; SHA-256 khớp lane C |
| Tổng tham số 11.078.526.194; 2.560 so với 2.880 token ảnh | Chạy lại `derive.py` của lane A trên config công khai |
| `e2e_challenge` HEAD là `297b4d0` (2026-11-03) | `gh api` |
| PEGASUS và Glasserman–Yao tồn tại như đã trích | Tìm kiếm web |

## 3. Chỉ dựa trên lane (root chưa đọc lại)

- Lane A: FLOP, byte, roofline và cận latency; dispatch SDPA tách theo ảnh; chi tiết PR #36 (khoá theo hình dạng KV, tối đa 4 graph); NIM 1.0.0 không liệt kê RTX PRO 6000; số trong whitepaper RTX PRO Blackwell; tỉ trọng prefill 90,6%.
- Lane B: định nghĩa nhóm và luồng retry; lag và P2R không fence; nhóm bị chia qua trainer step; `perf` và `train/iteration_time = 0,0`; ước tính ≈94% prefill; recipes `c53aa09`; Cosmos-RL main.
- Lane C: thay đổi của `297b4d0` chỉ trong `local_evaluation/`; nhánh challenge gỡ modifier xoá ảnh đen (PR #109); retry khác nhau giữa nhánh; định nghĩa độ lệch; tài liệu Harmonizer; thẻ PAI-AV v26.11; USDZ 26.04 bị thay ngày 2026-07-30; bảng missingness và power B/C.

Sự đồng thuận giữa A và B về yếu tố ảnh là bằng chứng hội tụ, không phải xác nhận; root đã đọc lại nguồn của yếu tố đó.

## 4. Chỗ bất đồng và sửa chữa

| Chỗ | Trước | Sau | Ai phát hiện |
|---|---|---|---|
| Đường lỗi của AlpaGym | `nvidia-stack.md` (11-01): hết retry thì payload rỗng, "dropped" | Hết retry thì `rollout_generation` ném `RuntimeError`; không có nhóm thiếu im lặng ở pin này | Lane B; root ghi vào `system-reconstruction.md` |
| Xoá quỹ đạo ảnh đen | Bản nháp đầu của root ghi evaluator offline xoá chúng | Chỉ đúng với `main` và pin Gym; nhánh challenge đã gỡ modifier (PR #109) nên chấm như lỗi lái | Lane C; root đã sửa `scientific-audit.md` và B5 |
| Re-sim nhóm để kiểm G1 | Bản nháp đầu của root đề xuất re-sim với seed khác và seed giống | Re-sim seed mới vô nghĩa (đảo với xác suất 1/2); chỉ replay cùng seed | Lane B; root đã sửa |
| Đổi bound gate | Giả thuyết ban đầu của root: bound thích nghi phương sai sẽ chặt hơn | Empirical Bernstein lỏng hơn ở mọi ô | Root tự bác |
| Số yếu tố lệch đường đi | Báo cáo lane A: năm | Hand-back lane A và root: sáu (thêm controller) | Lane A; root đã đọc config |
| Thứ tự ứng viên | Bản nháp đầu của root: CRN là chính | Đồng nhất đường policy là chính, CRN dự phòng | Root, sau lane A/B; lý do ở `candidates.md` §2 |
| Tài liệu `04` | "NVIDIA prints no 504"; ngân sách CoC 64; Gym 320×576 | Whitepaper in 503,8; CoC 128; Gym 320×512 | Lane A (whitepaper, chưa root đọc), root (Gym) |

## 5. Chưa kiểm

- Mọi số đo runtime: f, R, tỉ lệ va chạm cơ sở, tác động của sáu yếu tố, độ tất định renderer/Harmonizer, tỉ trọng thời gian.
- Ánh xạ tag image NRE sang bản phát hành NuRec; NuRec image mà pin Gym dùng.
- Hằng số pixel của processor Alpamayo 1.5 (lane A suy từ helper; chưa đọc file processor có cổng).
- Hành vi tiến trình Cosmos khi `rollout_generation` ném lỗi.
- Liệu 320×512 là chủ ý cho Alpamayo 1.5 hay không.
- Liệu preset AlpaGym nào khác override controller (quét chưa hết cây).
- Không có rà soát độc lập thứ hai cho chính tài liệu này.
