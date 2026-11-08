# Báo cáo: đóng góp chính cho PhysLoop v5 và kế hoạch sửa đổi đề xuất

Agent-authored cho Đặng Huy, 2026-11-05. Phục vụ S2 (G1, G2, G3·R1, G7) và S4 (G9). Đây là nghiên cứu nguồn, đại số và tính toán CPU; **không có GPU, rental, license, download, commit hay công bố nào**. Không ô `____` nào được điền. Plan v5 vẫn có hiệu lực; mọi thay đổi ở đây là **đề xuất P-032** chờ Huy đăng ký.

## 1. Quyết định điều hành

**Giữ hướng v5.** Không có bằng chứng ủng hộ chuyển trọng tâm sang kernel hay triển khai: AlpaSim chạy lockstep, nên tăng tốc chỉ mua giờ GPU chứ không đổi kết cục lái mô phỏng, và thiết bị (S6/S7) vẫn bị cổng P-002. G9 vẫn là sản phẩm chính.

**Ứng viên chính: đồng nhất đường policy giữa huấn luyện và đánh giá.**
- *Loại:* correctness của quy trình nghiên cứu, cộng hạ tầng đánh giá có thể đưa lên AlpaSim.
- *Vấn đề (root đã đọc nguồn từng yếu tố):* NVIDIA ghi rằng checkpoint AlpaGym được chuyển sang driver `alpamayo1_5` của AlpaSim để đánh giá. Nhưng hai đường đó tính hai hàm khác nhau của cùng trọng số, ở sáu yếu tố: ảnh 320×512 (méo tỉ lệ) so với 320×576; **không CoC** so với **luôn sinh CoC** (tối đa 128 token, T 0,6); offset RoPE của token expert; SDE 5 bước có seed so với Euler ODE 10 bước không seed; 5 Hz so với 10 Hz; MPC tuyến tính so với phi tuyến. Lớp `Alpamayo1_5` phát hành không có sampler bỏ CoC; Gym không replay được CoT. Vì vậy S4.6 như đang viết đo một hàm khác với hàm RL tối ưu.
- *Thay đổi:* đăng ký tường minh sáu yếu tố và **estimand đường đánh giá** ngay ở S2.1; một driver variant "trained-path" opt-in; `input_size` 320×576 phía Gym nếu Huy chọn.
- *Độ tin cậy:* **cao** rằng lệch tồn tại (nguồn, hai lane độc lập tìm ra phần ảnh, root đọc lại cả sáu); **chưa biết** độ lớn tác động. Đó là một cơ chế ứng viên cho thất bại chuyển giao, chưa phải thất bại đã đo. Không gọi là bug của NVIDIA: 320×512 từng là mặc định của chính AlpaSim tại pin của Gym.

**Ứng viên dự phòng: ghép cặp nhiễu (common random numbers) và replay cùng seed.**
- *Loại:* hạ tầng nghiên cứu AV (bằng chứng đánh giá trên mỗi đô la, giữ nguyên giao thức).
- *Vấn đề:* nhánh đánh giá không tiêu thụ seed; AlpaSim `main` suy seed bằng phép cộng nên luồng nhiễu của các replicate chồng lấn (21/22 seed chung giữa replicate 0 và 1); seed 0 nghĩa là ngẫu nhiên; ở pin của Gym seed không đặt được, nên nhóm huấn luyện không replay được và G1 không đo được.
- *Cơ chế:* dùng chung nhiễu giữa hai arm làm tăng `Cov(A, B)` mà không đổi phân phối biên hay độc lập giữa road, nên gate khoá vẫn hợp lệ nguyên vẹn. Trong mô phỏng minh hoạ bằng chính bound của gate, giảm discordance còn 20% nâng P(GO | tương đương) ở margin 0,05 từ 50–59% lên 100%, trong khi gấp ba replicate chỉ nâng từ 50% lên 59%.
- *Độ tin cậy:* cao về cơ chế; lợi ích thật phụ thuộc f, chưa đo.
- Cả hai ứng viên nằm trong **cùng một ranh giới patch**: một driver variant Alpamayo 1.5 trong AlpaSim.

**Ẩn số quyết định.**
1. Độ lớn dịch chuyển đường đi `D_f` so với độ trải lấy mẫu (E25, ≈2–4,5 GPU-giờ sau khi S2 được admit; suy ra).
2. Số road độc lập R theo stratum. Không đếm được từ dữ liệu công khai; với mọi road chênh lệch bằng 0, vẫn cần 72 road mỗi stratum cho margin 0,05 và 183 cho 0,02.
3. Hệ số ghép cặp f cho re-export, snapshot RL sớm và checkpoint cuối.
4. Lựa chọn estimand của Huy: policy theo đường huấn luyện, hay theo đường triển khai (CoC, như NIM và challenge).

**Phép đo thông tin nhất tiếp theo** (nếu chỉ được làm một việc): sau khi Huy chấp nhận license, đếm road theo stratum trong 441 scene; trước khi tốn GPU, quyết định estimand đường đánh giá vì nó đổi cả chi phí S2 (cận ≈117 ms so với ≈281 ms mỗi lần gọi, suy ra).

## 2. Audit PhysLoop v5

**Đã xác minh hôm nay:** HEAD `3741d92`; 756 test pass (`uv run --frozen pytest -q`, 11,9 s). S1 qualified (D-016). S2 chuẩn bị miễn phí và host check Vast đã đóng ($0,60 từ credit cũ, D-020).

**Thang bằng chứng, giữ tách biệt:** host check (đã có, chỉ tổng hợp) ≠ smoke AlpaSim/AlpaGym (chưa) ≠ memory fit đầy đủ (chưa) ≠ pilot có ích thống kê (chưa) ≠ thí nghiệm khoa học (chưa) ≠ kết quả học (chưa).

**Không nhất quán tìm thấy.**

| # | Chỗ | Vấn đề |
|---|---|---|
| 1 | Sổ tay PhysLoop (artifact) | Vẫn mô tả v4.1 ngày 2026-10-29: "chưa có code", thesis thầy → trò → Thor, capstone, lịch 12 tuần |
| 2 | `04` §2, §3 | Gym ghi 320×576/2.880 token, thực tế 320×512/2.560; "NVIDIA prints no 504" trong khi whitepaper in 503,8 TFLOPS (lane A); ngân sách CoC là 128, không phải 64 |
| 3 | `CLAUDE.md`, `AGENTS.md`, `04` | Recipes #42, #44, #45 được mô tả là PR mở; đã merge 2026-11-03…05 (chưa release) |
| 4 | `nvidia-stack.md` | "payload dropped" khi hết retry; thực tế ném `RuntimeError` |
| 5 | Draft S2 | Pin `e121e59`; HEAD đã là `297b4d0` (chỉ đổi dữ liệu `local_evaluation/`) |
| 6 | Draft S2 | "441" chỉ là con số; `scene_list`, `road_list` null; không có trường arm, đường đi driver, hash tài sản |
| 7 | `protocol-design.md` §6 | "Order survives re-simulation" mơ hồ; re-sim seed mới vô nghĩa, cần replay cùng seed |
| 8 | Plan S4.6 | Giả định gate đo đúng hàm RL tối ưu; sáu yếu tố cho thấy không |

**Tài sản dùng lại được:** gate S1 đã qualify; preflight, adapter và verifier S2; route Vast KVM đã kiểm cùng watchdog; bản sao pinned-source ngày 11-01; script power/coupling hôm nay; danh sách 441 scene và phiên bản NRE từng scene (lane C).

**Chặn:** license PAI-AV/NuRec (và Cosmos cho Harmonizer); quyết định khoa học của Huy; rental cụ thể; xác nhận revoke Vast key xuất hiện trong file nsys trên đĩa; P-003, P-004.

## 3. Mục lục các sản phẩm bắt buộc

| # | Sản phẩm | File |
|---|---|---|
| 1 | Quyết định điều hành | §1 ở trên |
| 2 | Audit v5 | §2 ở trên; `evidence-log.md` mục A |
| 3 | Evidence log và bản đồ nguồn | `evidence-log.md`; `lane-A.md`, `lane-B.md`, `lane-C.md` có log riêng |
| 4 | Dựng lại hệ thống (quan sát → hành động, vòng học, thời gian, danh tính) | `system-reconstruction.md` |
| 5 | Audit khoa học (likelihood/KL, reward, gate, power, validity, missingness, contamination) | `scientific-audit.md` |
| 6 | So sánh ứng viên, prior art, ứng viên bị bác | `candidates.md` §1 |
| 7 | Phân tích finalist, phương trình, điều kiện thất bại, claim chính xác | `candidates.md` §2–§3 |
| 8 | Kế hoạch chuyển giao và kiểm chứng | `candidates.md` §2.6, §3.4, §4 |
| 9 | Kế hoạch sửa đổi hoàn chỉnh, bảng quyết định từng hạng mục, gói S2 | `revised-plan-proposal.md` |
| 10 | Đặc tả refactor và đóng góp | `refactor-spec.md` |
| 11 | Kịch bản tương lai và quyết định còn treo | `candidates.md` §5; `revised-plan-proposal.md` §6 |
| — | Ai kiểm gì, chỗ bất đồng | `verification.md` |
| — | Script, đầu ra, log test, giá công khai | `evidence/contribution-review-20261105-v1/` (`SHA256SUMS`) |

## 4. Ba điều Huy cần quyết trước khi tốn tiền

1. **Estimand đường đánh giá cho S2 và S4.6:** policy theo đường huấn luyện (cần driver variant; rẻ hơn mỗi lần gọi) hay theo đường triển khai (CoC; chấp nhận dịch chuyển như một phần câu hỏi). Đường còn lại có thể chạy như arm chẩn đoán.
2. **Population và road:** đăng ký danh sách 441 scene cùng nguồn danh tính road, và margin tương thích với số road theo stratum.
3. **Seed mode:** giữ unseeded như draft, hay CRN sau khi variant được qualify.

Những quyết định này không cần thêm tiền; chúng quyết định tiền S2 mua được gì.

## 5. Những gì báo cáo này không thiết lập

Không có số đo nào về tác động của sáu yếu tố, về f hay R, về tỉ lệ va chạm cơ sở, về độ tất định của renderer, hay về việc RL cải thiện lái. Mọi bảng power là minh hoạ với tham số giả định. Mọi số đo tương lai trên frame NuRec bị KC-L chặn công bố; code, test CPU và số liệu tổng hợp thì công bố được. Không có "first" nào được tuyên bố.
