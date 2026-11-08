# Kế hoạch sửa đổi đề xuất (v5.1-đề xuất, chưa được Huy đăng ký)

Agent-authored, 2026-11-05. **Đây là đề xuất P-032, không phải plan đã thông qua.** `docs/02_PHYSLOOP_PLAN.md` (v5, D-021, D-022) vẫn có hiệu lực cho tới khi Huy ghi một quyết định. Mọi ô `____` vẫn trống. Không có rental, license, download, unlock hay commit nào xảy ra.

Căn cứ: `candidates.md`, `scientific-audit.md`, `system-reconstruction.md`, `evidence-log.md`, `verification.md`.

## 1. Thay đổi so với v5

1. **Không đổi hướng, không đổi trần tiền, không thêm stage.** S1 → S2 → S4 → kết luận; G9 vẫn là sản phẩm chính.
2. **Quyết định đường policy được đánh giá phải đến trước S2 pilot.** Đường huấn luyện AlpaGym và đường đánh giá AlpaSim khác nhau ở sáu yếu tố (ảnh 320×512/320×576, có/không CoC, RoPE, SDE-5/ODE-10, 5/10 Hz, MPC tuyến tính/phi tuyến). S2 đo độ tin cậy, power và chi phí trên **chính** đường sẽ dùng ở S4.6, và S4.6 dùng lại teacher của S2, nên lựa chọn này thuộc S2.1, không phải S4.
3. **Hai ẩn số được đưa lên đầu S2:** số road độc lập theo stratum (R) và độ ghép cặp nhiễu giữa hai arm (f).
4. **Một driver variant "trained-path" có seed theo phiên** được đặc tả như công cụ đo, dùng cho arm chẩn đoán dù Huy chọn estimand nào.
5. **S4.4 dùng replay cùng seed,** không dùng re-sim seed mới (vô nghĩa về thông tin).
6. **S2.3 thêm `render_status`, danh tính scene theo hash và luật dữ liệu thiếu;** S4.1 thêm các guard cấu hình của lane B.
7. **S4.5 thêm một cơ chế ứng viên:** dịch chuyển đường policy giữa huấn luyện và đánh giá.

## 2. Quyết định cho từng hạng mục

| Hạng mục hiện có | Quyết định | Bằng chứng | Liên hệ với giá trị | Thay đổi | Kiểm chứng | Ai quyết |
|---|---|---|---|---|---|---|
| S1 gate core (`verdict.py`, `stats/`) | **Giữ** | 756 test pass; bound khoá chặt hơn empirical Bernstein (A9) | Dụng cụ đo hợp lệ | Không | Không | File vẫn khoá |
| S2.1 Quyền, phiên bản, đăng ký | **Sửa (trọng tâm)** | A3; B2, B9–B12; lane C §1 | R là trần power; đường policy quyết định cái gì được đo | Đăng ký: danh sách 441 scene của `nurec_curated_val` kèm phiên bản NRE từng scene; nguồn danh tính road; **estimand đường đánh giá** (trained-path hay deployed-path) và sáu yếu tố; pin `e121e59` hay `297b4d0` (khác nhau chỉ ở dữ liệu `local_evaluation/`) | Audit metadata sau license; nếu không đếm được road, dừng để sửa đổi khoa học như v5 đã nói | Huy |
| S2.2 Pilot 5 scene Harmonizer off/on | **Sửa nhỏ** | Chi phí phụ thuộc đường (≈117 vs ≈281 ms cận mỗi lần gọi, suy ra) | Forecast chi phí đúng đường | Pilot chạy trên đường đã chọn; thêm probe "cùng policy cùng seed hai lần" nếu variant sẵn sàng | Như v5 | Huy |
| S2.3 Extraction | **Bổ sung** | B3–B5; lane C §4 | Không đếm sai at-fault, không mất thất bại | `render_status`; hash USDZ; fixture đa-actor E24; luật dữ liệu thiếu | Known-answer CPU | Agent; Huy chọn luật phân tích |
| S2.4 Teacher + re-export, ≥3 replicate | **Sửa** | A8; lane C §6 | Control phải có khả năng ra GO khi đúng là giống nhau | Control re-export dùng CRN nếu đã qualify; replicate giữ cho ICC | E22, E23 | Huy (seed mode, floor) |
| S2.5 Validity, contamination | **Giữ + bổ sung** | Lane C §3; ít nhất 666/2.202 scene curated mang nhãn khác `alpasim` (suy ra) | Bin là hậu-điều trị | Bin chỉ là chẩn đoán; bảng 4×4 (bin_A × bin_B) theo stratum; audit contamination bắt buộc | Fixture | Huy (semantics) |
| S2.6 Power vận hành | **Sửa (trọng tâm)** | A7, A8; lane C bảng B | Biết trước S4 kết luận được gì | Forecast bằng bound khoá với R và f đo được, theo stratum | Tái lập bảng với số đo | Agent; Huy (margin) |
| S2.7 Chi phí, telemetry | **Bổ sung** | Lane A, B | Mọi tuyên bố tốc độ cần breakdown | Breakdown renderer / ViT / prefill / decode CoC / expert / physics / idle; NVTX | nsys sau warm-up | Agent |
| S2 [O] Slurm, fault injection, batching | **Hoãn** | Không ứng viên nào cần | — | — | — | — |
| S4.1 Đóng băng bài toán học | **Bổ sung** | Lane B §4 | Tránh nhóm bị chia qua update, phiên bản trộn | Ghi sáu yếu tố đường đi; guard `prefetch_rollout: false`, lag, chia hết cho G, `max_batch_size`, `retry_on_nonfinite`; `input_size` là quyết định của Huy **trước** baseline A | Kiểm cấu hình CPU | Huy (giá trị); agent (assert) |
| S4.2 Correctness | **Bổ sung** | Lane A §7 | Parity thật | E04 ở `max_batch_size: 1`; dung sai từ δ đo được; E25–E27 | CPU + GPU qualification | Agent; Huy (tolerance) |
| S4.3 Reward ↔ evaluation | **Sửa** | B5; `scientific-audit.md` §5 | Hiểu reward tăng nghĩa là gì | Báo `collision_rear` riêng; bốn fixture chênh lệch | Known-answer CPU | Agent; Huy |
| S4.4 Baseline ≥3 run | **Sửa nhỏ** | Lane B §2.3 | Tách nhiễu ngoại sinh khỏi chênh lệch do hành động | Replay **cùng seed** qua harness `session_uuid → seed`; báo tỉ lệ bitwise và tỉ lệ đảo thứ tự | Trong engineering qualification | Huy (cỡ mẫu); agent (harness) |
| S4.5 Một can thiệp | **Bổ sung một cơ chế** | `candidates.md` §2 | Không đổ lỗi cho RL điều do đường đi gây ra | Thêm "dịch chuyển đường policy huấn luyện/đánh giá" vào danh sách cơ chế; vẫn chỉ một can thiệp | E25 trên checkpoint RL đầu | Huy |
| S4.6 Đánh giá checkpoint | **Sửa** | A8; giá §C evidence-log | Đủ tiền đánh giá; đúng estimand | Dùng estimand đường đi của S2.1; CRN cùng lịch seed với teacher S2; số replicate từ f đo được | Forecast trước khi chạy | Huy |
| S4.7 Báo cáo | **Giữ** | — | — | Báo sáu yếu tố đường đi trong mọi claim | — | Huy |
| S3, S5–S8 | **Hoãn** | Không ứng viên nào cần | — | — | — | Quyết định riêng |
| Ngân sách $100/$1.200/$2.500 | **Giữ** | §C evidence-log | — | Không | — | Huy cho mọi hành động |

## 3. Milestone đề xuất

Không có lịch: P-003 vẫn mở.

**M0. Đăng ký population, đường đánh giá và pin (miễn phí, trước pilot).**
- Mục tiêu: danh sách 441 scene và nguồn danh tính road; estimand đường đánh giá; pin nhánh.
- Phụ thuộc: metadata có license do Huy chấp nhận (road không đếm được từ dữ liệu công khai).
- Bằng chứng: bảng road theo stratum; sáu trường đường đi; ghi chú pin.
- Chấp nhận: mỗi stratum có R ≥ sàn của margin Huy chọn (36/51/72/183 road cho 0,10/0,07/0,05/0,02).
- Dừng/chuyển hướng: R dưới sàn ở bất kỳ stratum nào → Huy đổi margin, gộp stratum hoặc đổi population **trước** khi chạy.

**M1. Driver variant và test CPU (miễn phí).**
- Mục tiêu: `prompt_mode`, integrator, `pre_resize`, generator theo phiên với seed băm; harness replay Gym.
- Chấp nhận: test CPU của `refactor-spec.md` §0–§1 qua; đường mặc định không đổi.
- Dừng: nếu cần sửa trọng số hoặc lớp mô hình ngoài một hàm lấy mẫu, dừng và đề xuất lại.

**M2. S2 pilot đã đăng ký (≤$100).** Như v5 trên đường đã chọn, cộng probe seed nếu Huy đồng ý. Chấp nhận và dừng như v5.

**M3. S2 full.** ICC/độ tin cậy, f₀ của cặp teacher–re-export, forecast power theo stratum. KC-2 như v5, cộng bảng "S4 có thể kết luận gì" với R và f đo được.

**M4. E25 open-loop trên checkpoint gốc (≈2–4,5 GPU-giờ, suy ra; cần admission và cổng tài sản như S2).**
- Mục tiêu: `D_f` của từng yếu tố so với `D_within`.
- Chuyển hướng: mọi `D_f ≤ D_within` → giữ cấu hình, ghi lại, bỏ arm chẩn đoán; ngược lại, Huy quyết định căn chỉnh hay báo kèm.

**M5. S4 engineering qualification (admission riêng).** E01–E27; f₁ giữa teacher và snapshot RL sớm; replay cùng seed. Nếu f₁ ≈ 1, CRN không giúp S4.6; forecast lại bằng replicate.

**M6–M8. S4 baseline ≥3 run, một can thiệp, đánh giá và báo cáo.** Như v5 S4.4–S4.7.

## 4. Kiểm thử bổ sung (tiếp nối E01–E21)

| ID | Input | Kết quả bắt buộc |
|---|---|---|
| E22 | Hai arm cùng lịch seed, policy giống hệt (re-export) | `d_r = 0` mọi road nếu tất định; nếu không, tick phân kỳ đầu và nguồn |
| E23 | Lịch seed bị xáo giữa hai arm | Discordance về mức unseeded |
| E24 | Hai actor chạm ego cùng timestep (front + rear) | Ghi đúng theo phiên bản scorer đã pin; khác biệt với bản sửa #234 được ghi |
| E25 | Giai thừa 2³ ảnh × CoC × integrator trên tick đã ghi | `D_f` và `D_within` theo mét |
| E26 | Variant ở cấu hình mặc định | Bitwise (hoặc cùng phân phối) với driver stock |
| E27 | Variant ở cấu hình Gym trên một tick đã ghi | Khớp `AlpamayoR1InferenceModel` của Gym trong dung sai đã đăng ký |

## 5. Gói S2 cập nhật (bị chặn tới khi có các phê duyệt)

**Tiền đề chính xác.**
1. Quyết định khoa học của Huy: population và nguồn road (M0), strata, margin, reliability floor, **estimand đường đánh giá**, seed mode, Harmonizer, terminal semantics, luật dữ liệu thiếu, dự đoán pilot. Tất cả `____`.
2. Quyền: PAI-AV và NuRec agreement do Huy chấp nhận; mốc first-download; KC-L/KC-L2 không đổi. Gói tham chiếu công khai 8 subject × 441 × 3 trong Git LFS có thể cho phương sai trong scene, nhưng việc tải nó có thuộc luật "không tải NuRec trước S2" hay KC-L không là quyết định của Huy.
3. Hành động: quote mới và lệnh tạo đúng một VM, theo `docs/S2_RENTAL_APPROVAL.md`.
4. **Credential:** memory dự án ghi Vast key có trong file nsys trên đĩa (đã gitignore) và chưa xác nhận revoke. Đề xuất Huy revoke/rotate trước rental kế tiếp. Agent không in, không thử key. Probe đã sửa (`3741d92`) chưa chạy trên VM thật; quét credential báo cáo đầu tiên, kể cả SQLite và archive.

**Cấu hình đề xuất (chưa duyệt).** Vast KVM 1× RTX PRO 6000 WS có `vms_enabled`, rẻ nhất công khai $1,606/h lúc 2026-11-05 16:34 UTC; pilot ≤8 GPU-h, ≤$2,50/GPU-h như đề xuất cũ; Nebius vẫn là dự phòng chưa kiểm. Watchdog control-plane bind ID và label như D-019. NIM 1.0.0 không liệt kê RTX PRO 6000 là GPU đã validate; điều này không ảnh hưởng driver AlpaSim nhưng nên ghi.

**Artifact bắt buộc.** Manifest đầy đủ (02 §3 S2) cộng sáu trường đường đi và hash USDZ; raw summary và mảng metric; ledger mọi attempt với `render_status`; breakdown thời gian; kết quả probe seed hoặc lý do không đo; cleanup evidence.

**Xử lý thất bại.** Không retry rental tự động; dừng ở trần; giữ mọi log; thất bại hạ tầng không gộp vào kết cục lái; trên nhánh challenge, retry lần đầu chỉ có trong log runtime, nên phải thu log.

**Tài nguyên.** Như v5. Ước tính lịch sử 463–1.150 ms mỗi bước rollout (`04`) cho 441 × 3 × 200 bước là 34–85 GPU-h mỗi sweep, khoảng $50–137 compute ở $1,47–1,61/h. Lane A gợi ý decode CoC chiếm phần lớn thời gian mỗi lần gọi trên đường đánh giá; một estimand không-CoC có thể rẻ hơn đáng kể. Cả hai là suy ra, không phải forecast đo được.

## 6. Hạng mục còn treo

| Mục | Trạng thái | Ai |
|---|---|---|
| P-003 giờ làm mỗi tuần | Mở | Huy |
| P-004 quyền công bố | Mở; KC-L/KC-L2 chặn | Huy |
| License PAI-AV/NuRec, NRE, Harmonizer (Harmonizer cần thêm repo Cosmos có cổng) | Chưa chấp nhận | Huy |
| Rental cụ thể cho pilot | Chưa duyệt | Huy |
| Vast credential trong file nsys | Chưa xác nhận revoke | Huy |
| Estimand đường đánh giá | Mới, chưa quyết | Huy |
| Ô cài đặt S4 (group size, lr, scene set, optimizer steps, retention, anchor) và `input_size` | Trống | Huy |
| Dự đoán S2/S4 | Trống | Huy |
| Pin `e2e_challenge` sau khi nhánh di chuyển | Chưa quyết | Huy |
| Cập nhật Sổ tay theo v5 | Đã làm 2026-11-07: artifact phiên bản 3, nội dung P-032 được đánh dấu đề xuất, ô ★ để trống | — |
| Sửa `04` (503,8 TFLOPS; CoC 128 token; Gym 320×512) và trạng thái "open" của recipes #42/#44/#45 trong CLAUDE.md/AGENTS.md | Đề xuất | Huy duyệt sửa tài liệu |
