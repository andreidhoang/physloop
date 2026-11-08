# So sánh ứng viên, phân tích finalist, kế hoạch chuyển giao và kiểm chứng

Agent-authored, 2026-11-05. Ba lane độc lập (A: phần cứng/kernel/suy luận; B: RL hành động và hệ mô phỏng; C: tính hợp lệ và chuyển giao) mỗi lane đề xuất tối đa hai ứng viên mà không thấy xếp hạng của nhau (lane C có thấy thư mục của root xuất hiện nhưng ghi rằng không đọc). Báo cáo gốc: `lane-A.md`, `lane-B.md`, `lane-C.md`. Root (agent điều phối) đọc lại các dòng nguồn mang tải (`verification.md`). **Sự đồng thuận giữa các agent không phải xác nhận thực nghiệm.** Không ứng viên nào đã chạy trên GPU.

## 1. Mọi ứng viên đã xét

| Ứng viên | Lane | Loại | Vấn đề có thật? | Quyết định | Lý do quyết định |
|---|---|---|---|---|---|
| **Đường policy huấn luyện ≠ đường đánh giá** (ảnh, CoC, RoPE, SDE/ODE, 5/10 Hz, controller) | A-1, B-C1 (phần ảnh) | Correctness + hạ tầng đánh giá | Có, root đọc lại mọi yếu tố | **Finalist chính** | Nếu không xử lý, kết luận G9 bị nhiễu bởi khác biệt đường đi bất kể power |
| **Common random numbers (CRN) và replay cùng seed** | root, B-C2, C §2/§6 | Hạ tầng đánh giá/RL | Có (seed không được tiêu thụ, seed cộng chồng lấn, re-sim seed mới vô nghĩa) | **Dự phòng** | Tăng power trên mỗi đô la và đo được G1; lợi ích phụ thuộc f chưa đo |
| Trường `render_status` trong extraction | C-1 | Công cụ đo S2.3/S2.5 | Có (challenge chấm ảnh đen như lỗi lái; main/Gym xoá) | **Nhận làm yêu cầu kỹ thuật S2** | Cần cho G2, rẻ, agent-owned; không phải đóng góp xếp hạng |
| Danh tính scene theo nội dung (SHA-256 mỗi USDZ, revision theo commit) | C-2 | Công cụ đo S2.1/S2.3 | Có (USDZ 26.04 bị thay 2026-07-30; revision là tên nhánh) | **Nhận làm yêu cầu kỹ thuật S2** | Không có nó, so sánh S2 ↔ S4 có thể lẫn Δ_asset |
| Cache prefix VLM trong trainer replay | A-2 | Hạ tầng RL | Có lãng phí (≈90,6% hoặc ≈94% FLOP mỗi hàng là prefill đóng băng, hai ước tính dưới giả định khác nhau), nhưng | **Hoãn (có điều kiện)** | Lợi ích bằng 0 khi `grpo_optimization_iterations = 1` (mọi preset); pipeline của NVIDIA bị giới hạn bởi rollout (PR #36) |
| Đổi bound của gate (empirical Bernstein) | root | Thống kê | Không | **Bác** | Bound khoá chặt hơn ở mọi ô kiểm (A9) |
| Bật CUDA graph expert / lượng tử hoá | A (xét) | Suy luận | — | **Bác cho nghiên cứu này** | Prior art (PR #36, recipe quant, NIM); lượng tử hoá đổi số học; tốc độ không đổi kết cục mô phỏng (lockstep) |
| Cache ViT giữa các tick | A (xét) | Suy luận | Đúng về toán | **Bác** | Trần 10–14% FLOP không-decode; đã có prior art (TensorRT-Edge-LLM) |
| Tái dùng KV LLM giữa các tick | A (xét) | Suy luận | — | **Bác** | Không chính xác (M-RoPE đổi vị trí); phá parity rollout/replay |
| Chia sẻ prefill đầu tiên trong nhóm | A (xét) | Hạ tầng RL | — | **Bác** | ≤4,5% và cần render tất định |
| Cache frame warm-up (`force_gt_frame_cache` của AlpaSim main) | B §3 | Hạ tầng RL | Có thể ≤26,7% công render | **Đo trước** | Cần đổi pin; tỉ trọng render chưa đo |
| PR sửa phân loại va chạm | root | Correctness | Có | **Không làm** | PR #234 và issue #237 đã do người khác mở |

## 2. Finalist chính: đồng nhất đường policy giữa huấn luyện (AlpaGym) và đánh giá (AlpaSim)

**Loại đóng góp:** correctness của quy trình nghiên cứu, cộng hạ tầng đánh giá có thể đưa upstream. Không phải tăng tốc suy luận, không phải kết quả lái.

### 2.1 Vấn đề, được root xác minh tại pin

NVIDIA ghi rằng checkpoint AlpaGym được chuyển đổi để driver `alpamayo1_5` của AlpaSim đánh giá closed-loop (`convert_alpagym_checkpoint_to_inference.py` dòng 4–9, `972d160`). Nhưng hai đường đó tính **hai hàm khác nhau** của cùng bộ trọng số:

| Yếu tố | Đường huấn luyện AlpaGym (G) | Đường đánh giá AlpaSim (E) | Nguồn root đã đọc |
|---|---|---|---|
| 1. Ảnh | `input_size: [320, 512]`, resize không giữ tỉ lệ | Processor đưa ảnh 16:9 về 320×576; AlpaSim `main` từ chối ảnh nhỏ hơn và ghi rằng 320×512 "passes through untouched" | `alpamayo_r1.yaml` dòng 14; `buffers.py` dòng 236–253; AlpaSim `affc2ea` `alpamayo_base.py` dòng 369–370, `main.py` dòng 843–851 |
| 2. Chain-of-Causation | `last_component="traj_future"`: prompt kết thúc ở `<|traj_future_start|>`, **không sinh CoC**; replay CoT ném `NotImplementedError` | `sample_trajectories_from_data_with_vlm_rollout[_cfg_nav]`, top-p 0,98, T 0,6: **luôn sinh CoC**; lớp phát hành chỉ có hai sampler đó | Gym `inference_model.py` dòng 33–51, 118–126; `bundle.py` dòng 74–82; alpamayo1.5 `36aeb4c` `alpamayo1_5.py` dòng 244, 440; AlpaSim `e121e59` `alpamayo1_5_model.py` dòng 112–121 |
| 3. Vị trí RoPE của 64 token expert | Sau prefix ≈2.730 token | Sau prefix ≈3.090 token + CoC | Suy từ 1–2 (lane A, chưa đo) |
| 4. Bộ tích phân | SDE, `noise_level` 0,2, 5 bước (4 ở smoke), generator theo hàng | Euler ODE, 10 bước mặc định, `randn` toàn cục | Gym `alpamayo_r1.yaml` dòng 33–37; alpamayo1.5 `flow_matching.py` dòng 35, 171 |
| 5. Tần số lập kế hoạch | 5 Hz (control 200 ms), warm-up 1,6 s | 10 Hz (100 ms), warm-up 1,7 s | Lane A; đã được `rl-foundations.md` nêu |
| 6. Controller | MPC tuyến tính (mặc định của AlpaSim `46b401b`: `controller: default # Defaults to linear MPC`); không thấy override trong các YAML của Gym đã quét | MPC phi tuyến (`e121e59` `base_config.yaml` dòng 11) | Root đọc hai `base_config.yaml`; lane A nêu trong bản hand-back |

Camera, thứ tự camera, nhãn frame, 16 tư thế lịch sử và 64 waypoint giống nhau. Root chạy lại known-answer của lane A: tổng tham số 11.078.526.194 khớp đúng số safetensors, và token ảnh 2.560 (G) so với 2.880 (E).

**Không rõ 320×512 có cố ý hay không.** Lane B ghi rằng 320×512 là mặc định của chính AlpaSim tại pin của Gym; AlpaSim `main` sau đó thêm sàn 320×576. Vì vậy đây là **lệch cấu hình giữa hai đường**, không được gọi là bug của NVIDIA.

### 2.2 Cơ chế nhân quả

RL đổi θ để tăng `E_{c∼p_train, x∼SDE₅(v_θ(·|c))}[R]`, còn S4.6 đo `E_{c∼p_eval, x∼ODE₁₀(v_θ(·|c))}[R]`. Cập nhật chỉ chuyển giao nếu tác động của Δθ lên trường vận tốc `v` khái quát được qua dịch chuyển điều kiện `c_train → c_eval`: −320 token ảnh, vị trí cột ảnh nhân 0,889 cho cùng một cảnh, KV của CoC (khoảng 16–128 token) chỉ có lúc đánh giá, offset RoPE dịch, kế hoạch được thực thi 200 ms thay vì 100 ms giữa hai lần lập, và bởi một controller khác loại. Độ lệch chuẩn mỗi bước SDE theo đơn vị hành động chuẩn hoá là [0,200; 0,179; 0,110; 0,073; 0,045], so với 0 của ODE (lane A, suy ra).

Hệ quả: một kết luận "RL không chuyển giao" hoặc "RL cải thiện" trong S4.6 bị **nhiễu** với dịch chuyển đường đi này. Đây là **một cơ chế ứng viên** cho thất bại chuyển giao, đặt cạnh danh sách của S4.5; nó chưa được chứng minh là có tác động lớn.

Theo luật thay đổi của D-022, phát hiện này thuộc loại (b): so sánh S4.6 như đang viết là bài toán đặt chưa đúng, vì gate đánh giá một hàm khác với hàm được tối ưu.

### 2.3 Baseline mạnh nhất và prior art

Không có. PR #36 (CUDA graph), recipe lượng tử hoá và NIM 1.0.0 đều không có chế độ "không CoC"; NIM luôn trả `reasoning_text`. AlpaSim `main` đã chặn đúng yếu tố 1 ở phía đánh giá, cho thấy NVIDIA nhận ra nguy cơ ảnh sai kích thước, nhưng preset AlpaGym (cập nhật lần cuối 2026-07-21) chưa đổi. Precedent "huấn luyện SDE, triển khai ODE" của Flow-GRPO chỉ liên quan yếu tố 4 và chưa được chứng minh cho mô hình này. Không tìm thấy issue/PR AlpaSim hay AlpaGym nào về các yếu tố 1–4 (tìm 2026-11-05).

### 2.4 Thay đổi hẹp

1. **Đăng ký (S2.1, quyết định của Huy):** ghi rõ sáu yếu tố trong manifest cho cả huấn luyện và đánh giá, và chọn **estimand đánh giá**: (a) *policy theo đường huấn luyện*, cần chế độ driver mới; hoặc (b) *policy theo đường triển khai* (CoC, như NIM và challenge), chấp nhận dịch chuyển như một phần câu hỏi. Đường còn lại được phép chạy như arm chẩn đoán. Tài liệu này **không** chọn thay Huy.
2. **Chế độ driver "trained-path" cho AlpaSim** (một model variant): `prompt_mode: traj_future` (prefill + expert, không `generate`), integrator/số bước/noise cấu hình được, pre-resize tường minh tuỳ chọn, và generator riêng cho mỗi phiên với seed suy bằng băm (dùng chung với ứng viên dự phòng). Cần gọi `_sample_trajectories_from_data_without_vlm_rollout` hoặc tương đương, mà lớp phát hành chưa có; đó là phần code mới.
3. **Phía AlpaGym:** `input_size: [320, 576]` là thay đổi chỉ-cấu-hình cho yếu tố 1, như overlay cục bộ của PhysLoop (AlpaGym không nhận PR). Chi phí: +12,5% token ảnh mỗi prefill. CoC phía huấn luyện không bật được (replay CoT chưa hỗ trợ).

### 2.5 Mục tiêu đo được và ranh giới

Ranh giới: suy luận policy → kết cục mô phỏng.

- **Open-loop:** với mỗi yếu tố f, độ phân kỳ chéo cấu hình `D_f = E‖τ_A − τ_B‖` (ADE trên 64 waypoint, mét) so với độ trải lấy mẫu trong cùng cấu hình `D_within`, trên cùng quan sát đã ghi.
- **Closed-loop:** chênh lệch điểm S2 ghép cặp giữa arm stock và arm trained-path, so với nhiễu chạy lại cùng policy.
- **Chi phí:** đường (G) có cận ≈117 ms mỗi lần gọi; đường (E) ≈281 ms ở 16 token CoC, vì decode CoC bị giới hạn bởi băng thông (≈8,7 ms/token) (lane A, suy ra trên RTX PRO 6000 WS, chưa đo). Lựa chọn estimand vì thế đổi cả forecast chi phí S2.

### 2.6 Thí nghiệm nhỏ nhất bảo toàn cơ chế

| Câu hỏi | Trả lời |
|---|---|
| Mục tiêu | Checkpoint Alpamayo 1.5 được chuyển từ AlpaGym sang driver AlpaSim, như quy trình NVIDIA ghi |
| Cơ chế | Dịch chuyển điều kiện của expert giữa hai đường đi |
| Tái lập | T0 (CPU, $0): test cấu hình và hình học, gồm chạy `_decode_to_chw_uint8` thật trên JPEG tổng hợp và số học `smart_resize`. T1 (sau khi S2 được admit, cùng cổng tài sản): open-loop giai thừa 2³ (ảnh × CoC × integrator) trên ≈500 tick phát triển đã ghi, K = 8 mẫu, ≈32 nghìn lần gọi, ước ≈2–4,5 GPU-giờ. T2: lặp T1 trên checkpoint RL đầu tiên. T3: arm closed-loop trained-path so với stock trên scene phát triển |
| Thuộc tính giữ nguyên | Frame, trọng số, scorer, camera, lịch sử |
| Hiệu ứng thiếu | Tích luỹ closed-loop và yếu tố tần số (cần T3); phản ứng của traffic |
| Điều kiện thất bại | Mọi `D_f ≤ D_within` (ngưỡng của Huy) ở checkpoint gốc **và** checkpoint RL đầu tiên → không có gì để sửa; giữ cấu hình và ghi lại |
| Kiểm chứng rộng hơn | T3 và S4.6 với estimand đã đăng ký; số đo trên frame NuRec bị KC-L chặn công bố |

### 2.7 Khi lợi ích biến mất hoặc đảo chiều

- Mọi `D_f` nằm trong độ trải lấy mẫu: không có gì cần sửa.
- Huy giữ gate ở cấu hình stock: arm trained-path chỉ là chẩn đoán, không thay verdict.
- Đánh giá không-CoC kém liên quan tới triển khai hơn vì NIM và challenge chạy CoC.
- Căn hình học huấn luyện về 576 tốn thêm khoảng 12% prefill và rời khỏi preset NVIDIA.

### 2.8 Claim chính xác

> Với workload **post-train expert Alpamayo 1.5 trong AlpaGym rồi đánh giá closed-loop checkpoint đã chuyển đổi trong AlpaSim**, implementation **AlpaGym `972d160` với recipes PR #18 `1d99fc5`, và AlpaSim `e2e_challenge` `e121e59` với alpamayo1.5 `36aeb4c`**, môi trường **log-replay traffic, renderer NRE đã pin**, phần cứng **không liên quan tới kết cục (lockstep); chỉ chi phí trên RTX PRO 6000 WS**, đường **render → resize → prefill VLM → (CoC) → expert flow-matching → kế hoạch → controller** có vấn đề **hai đường đi tính hai hàm khác nhau của cùng trọng số ở sáu yếu tố**. Thay đổi **đăng ký tường minh sáu yếu tố, một chế độ driver trained-path có seed theo phiên, và `input_size` 320×576 phía huấn luyện khi Huy chọn** được dự đoán làm **S4.6 đo đúng hàm đã được tối ưu, hoặc tách được dịch chuyển đường đi khỏi tác động của RL**, nhờ **loại bỏ (hoặc đo) dịch chuyển điều kiện `c_train → c_eval` của expert**, với giả định **trọng số đã chuyển đổi nạp đúng (E02) và quan sát đã ghi là đại diện**. Thí nghiệm **T1** kiểm cơ chế cục bộ; **T3** cần cho claim closed-loop. Kết quả **mọi `D_f` nằm trong độ trải lấy mẫu ở cả checkpoint gốc và checkpoint RL đầu tiên, và T3 tương đương trong margin của Huy** sẽ bác giả thuyết. Kết quả này **không** thiết lập rằng RL cải thiện hay làm xấu việc lái, không thiết lập an toàn, không chuyển sang Alpamayo 2/Super, NIM hay Thor, và không cho phép công bố số suy ra từ NuRec trước KC-L.

## 3. Dự phòng: ghép cặp nhiễu (CRN) cho so sánh checkpoint và replay cùng seed cho nhóm huấn luyện

**Loại đóng góp:** hạ tầng nghiên cứu AV (bằng chứng đánh giá trên mỗi đô la, giữ nguyên giao thức), kèm một sửa nhỏ về cách sinh seed của AlpaSim. Cùng ranh giới patch với finalist chính: driver variant ở §2.4 mang generator theo phiên và seed băm.

### 3.1 Vấn đề, cơ chế, mục tiêu

- **Người hưởng lợi:** ai so sánh hai checkpoint cùng kiến trúc trên một tập scene AlpaSim cố định bằng quy tắc ghép cặp; và chẩn đoán G1 của S4.4.
- **Hiện trạng, root đọc nguồn:** `e2e_challenge` không tiêu thụ seed; `main` suy seed bằng phép cộng (`random_seed + rollout_idx`, rồi `+ inference_count`), nên luồng nhiễu của replicate chồng lấn và lệch đúng một tick (`seed_overlap.out`: 21/22 seed chung giữa replicate 0 và 1), seed 0 nghĩa là "ngẫu nhiên", RNG toàn cục; đường `simulate` stock không đặt seed (lane C). Ở pin của Gym, seed phiên là `random.randint`, không đặt được, dù có được log (lane B).
- **Thay đổi hẹp:** lịch seed đăng ký trước, giống nhau cho hai arm, độc lập giữa (road, scene, replicate) nhờ suy băm; generator theo phiên; harness Gym `session_uuid → seed đã ghi` cho replay nhóm.
- **Cơ chế:** `Var(B − A) = Var(A) + Var(B) − 2·Cov(A, B)`; với cùng nhiễu `ε`, `‖Φ_θ(ε, o) − Φ_θ′(ε, o)‖ ≤ L_θ‖θ − θ′‖` nếu sampler Lipschitz theo tham số. Gate khoá tách `d+/d−`, nên discordance nhỏ hơn làm cận KL–Chernoff co theo khoảng `√(m·ln(2/α)/R)`. Với nhóm G = 2: nếu `R = a(ξ) + ε`, replay cùng seed đảo thứ tự với xác suất `arccos(ρ)/π`, `ρ = σ²_a/(σ²_a + σ²_e)`; re-sim với seed mới đảo với xác suất 1/2 bất kể tín hiệu (lane B).
- **Mục tiêu đo được:** f = discordance(CRN)/discordance(độc lập) cho re-export, snapshot RL sớm, checkpoint cuối; P(GO | tương đương) trên mỗi GPU-giờ; tỉ lệ tái lập bitwise và tỉ lệ đảo thứ tự khi replay cùng seed.

### 3.2 Bằng chứng ngoài PhysLoop

- [Glasserman và Yao, Management Science 38(6):884–908, 1992](https://pubsonline.informs.org/doi/10.1287/mnsc.38.6.884): CRN **bảo đảm** giảm phương sai khi đầu ra đơn điệu cùng chiều theo input ngẫu nhiên chung. Một rollout closed-loop không hiển nhiên thoả điều kiện đó, nên f phải đo.
- [PEGASUS (Ng và Jordan, UAI 2000)](https://arxiv.org/abs/1301.3878): cố định số ngẫu nhiên để biến bài toán thành chuyển trạng thái tất định khi so sánh policy.
- AlpaSim tự đi theo hướng tái lập (PR #149, `force_determinism`); issue #128 hỏi cách tái lập seed. Không có issue/PR nào về chồng lấn seed hay ghép cặp seed giữa arm (tìm 2026-11-05).
- Đóng góp **không** phải ý tưởng CRN mà là: lỗi chồng lấn seed cụ thể, port cho driver Alpamayo, và số đo coupling trên flow-matching closed-loop.

### 3.3 Khi lợi ích biến mất

- **R quá nhỏ:** sàn `1 − (α/2)^{1/R}` (α = 0,05) không đổi; với R = 50, margin 0,05 không bao giờ GO dù f = 0.
- **Policy phân kỳ:** checkpoint xa teacher rẽ khác nhau tại điểm rẽ nhánh dù cùng nhiễu; f → 1.
- **Nhiễu ngoài policy chiếm ưu thế:** renderer, Harmonizer (không có tuyên bố tất định), batch 1–4 của Gym, NaN retry.
- **Dùng seed sai:** chọn seed sau khi xem kết quả, tái dùng giữa pilot và full, dùng chung dãy nhiễu giữa các road.

### 3.4 Thí nghiệm nhỏ nhất

T0 (CPU): test suy seed, kiểm cặp, plumbing harness Gym với `InferenceModel` giả. T1 (pilot, nếu Huy thêm probe): cùng policy, cùng seed, hai lần trên 5 scene. T2 (S2 full): teacher vs re-export với CRN → f₀. T3 (S4 qualification): 4 episode replay hai lần ở batch 1, rồi 16 nhóm G = 2 replay ở batch của run (lane B); teacher vs snapshot RL sớm với CRN và với seed xáo → f₁.

### 3.5 Claim chính xác

> Với workload **so sánh ghép cặp hai checkpoint Alpamayo 1.5 cùng kiến trúc trên một tập scene AlpaSim cố định, và chẩn đoán nhóm GRPO trong AlpaGym**, implementation **AlpaSim `e2e_challenge` `e121e59` (không tiêu thụ seed), `main` `affc2ea` (seed cộng), Gym `972d160` với AlpaSim `46b401b` (seed không đặt được)**, môi trường **log-replay traffic, renderer đã pin**, phần cứng **chỉ ảnh hưởng chi phí**, đường **seed → sampler → quỹ đạo → scorer → gate theo road** có vấn đề **nhiễu giữa hai arm độc lập, luồng seed chồng lấn, nhóm không replay được**. Thay đổi **lịch seed chung cho hai arm, độc lập giữa (road, scene, replicate) bằng suy băm, generator theo phiên, harness replay** được dự đoán **thu hẹp cận trên vận hành và tăng P(GO | tương đương) trên mỗi GPU-giờ, và tách nhiễu ngoại sinh khỏi chênh lệch do hành động trong nhóm**, nhờ **tăng Cov(A, B) mà không đổi phân phối biên hay độc lập giữa road**, với giả định **nhiễu sampler là phần đáng kể của phương sai và policy chưa phân kỳ hoàn toàn**. **T2** kiểm cơ chế cục bộ; **T3** cần cho claim về checkpoint RL. Kết quả **f₁ không nhỏ hơn 1 có ý nghĩa, hoặc tỉ lệ biên dưới seeded lệch so với unseeded** sẽ bác giả thuyết. Kết quả **không** thiết lập checkpoint nào tốt hơn, không cho phép hạ margin, không chuyển sang traffic phản ứng hay kiến trúc khác.

## 4. Chuyển giao: điều thí nghiệm nhỏ thiết lập được và không thiết lập được

| Finalist | Thí nghiệm nhỏ thiết lập | Không thiết lập | Bằng chứng lớn hơn cần có |
|---|---|---|---|
| Đồng nhất đường policy | Độ lớn của từng yếu tố dịch chuyển trên quan sát cố định, ở checkpoint gốc và checkpoint RL đầu tiên | Quan sát tương lai mà policy đã đổi tạo ra; tác động closed-loop tích luỹ | Arm closed-loop T3; S4.6 với estimand đã đăng ký |
| CRN và replay | f trên các cặp đã đo; tỉ lệ đảo thứ tự | f cho checkpoint xa hơn; batch lớn hơn; traffic phản ứng | Đo lại f ở checkpoint được chọn trước S4.6 |

Không ứng viên nào đòi thiết bị, HIL hay shadow mode; S6–S8 giữ trạng thái hoãn.

## 5. Tương lai có thể làm ứng viên lỗi thời

| Kịch bản | Giả định | Tín hiệu quan sát được | Hệ quả |
|---|---|---|---|
| NVIDIA thêm chế độ không-CoC vào lớp Alpamayo 1.5 hoặc driver AlpaSim | Upstream nhận ra cùng vấn đề | Commit/PR mới ở alpamayo1.5 hoặc AlpaSim | Dùng bản upstream; không mở PR cạnh tranh; giữ phần đo |
| AlpaGym cập nhật preset 576 hoặc hỗ trợ replay CoT | Gym có commit mới | Commit Gym sau `972d160` | Yếu tố 1 hoặc 2 tự biến mất; giữ đo các yếu tố còn lại |
| Release AlpaSim sau challenge (#224) đổi driver/seed | Release đã hứa | Release mới sau v2026.5 | Kiểm lại pin; không tự đổi nhánh giữa nghiên cứu |
| AlpaSim seed bằng băm hoặc ghép cặp có sẵn | Người khác sửa | PR về seed | Dự phòng chỉ còn phần đo f |
| Chuyển sang Alpamayo 2/Super | Quyết định phạm vi mới | — | Cả hai ứng viên phải kiểm lại; không tự chuyển |
