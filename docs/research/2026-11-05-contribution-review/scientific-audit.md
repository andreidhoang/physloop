# Audit khoa học: likelihood/KL, reward, gate, power, validity

Agent-authored, 2026-11-05, phục vụ S2 (G1, G2, G3·R1, G7) và S4 (G9). Đây là phân tích nguồn, đại số và mô phỏng CPU minh hoạ; không phải kết quả chạy AlpaSim, không phải dự đoán khoa học của Huy, không điền ô `____` nào. Mọi con số được tính lại bằng script trong `evidence/contribution-review-20261105-v1/`.

Thuật ngữ nhãn bằng chứng: **[nguồn]** đọc code tại SHA ghi rõ; **[đại số]** suy ra từ công thức; **[CPU]** chạy trên máy này bằng hàm của repo; **[giả thuyết]** chưa đo.

## 1. Hệ phân cấp suy luận và estimand

| Tầng | Đơn vị độc lập? | Vai trò trong gate hiện tại | Hệ quả |
|---|---|---|---|
| Road | Có, nếu population được đăng ký đúng | Đơn vị độc lập duy nhất mà `gate/verdict.py` đếm (`len(d)` trong `bounded_mean_bounds`) | Độ rộng bound phụ thuộc số road R, không phụ thuộc số scene hay replicate |
| Scene trong road | Không | Trung bình đều trong road | Thêm scene trên cùng road chỉ làm mịn `d_r`, không tăng R |
| Simulator replicate | Không | Trung bình đều trong scene | Giảm biến động của `d_r` nhờ triệt tiêu, không tăng R |
| Training run | Có, cho estimand cấp thuật toán | Gate không mô hình hoá tầng này | Gate chỉ kết luận về **một checkpoint có tên** |
| Checkpoint được chọn trong run | Không | Chọn checkpoint là một lần so sánh bội | Phải đăng ký luật chọn; gate không tự hiệu chỉnh "best-of-many" |

**Estimand hiện có [nguồn]:** `VerdictResult.estimand = "equal road / equal scene within road / equal replicate within scene"`. Tức là: rút một road từ population đã đăng ký, lấy trung bình đều các scene của nó rồi các replicate. Đây là estimand về **checkpoint B so với checkpoint A**, không phải "thuật toán RL tốt hơn". Một kết luận cấp thuật toán cần biến ngẫu nhiên thứ hai (run huấn luyện độc lập), giao chéo với road; `protocol-design.md` §7 đã nêu, audit này giữ nguyên.

**"441 scenes" là gì trong repo [nguồn, repo]:** chỉ là một con số trong validator (`gate/s2_preflight.py` dòng 316: `len(scene_list) != (441 if full else 5)`). Cả hai draft có `scene_list: null`, `road_list: null`, `road_independence_justification: null`. Không có file danh sách scene nào được đăng ký. Vậy "441" hiện là một **ràng buộc số lượng**, chưa phải một population có danh tính. Kết luận về số road độc lập là **chưa biết**; phần 3 cho thấy đây là ẩn số quyết định.

## 2. INSUFFICIENT của S1 nghĩa là gì

S1 báo noisy-equivalent controls INSUFFICIENT gần như 100%. Bốn cách giải thích mà đề bài yêu cầu phân biệt:

| Giải thích | Bằng chứng | Kết luận |
|---|---|---|
| Lỗi implementation | 756 test pass hôm nay [CPU]; identical-policy controls GO 1000/1000; gross regression NO-GO; bound được kiểm độc lập trong S1 | Không có dấu hiệu |
| Bất định thích đáng | Khi hai arm có nhiễu độc lập, mỗi road mang phần dương và âm khác 0; bound KL tách `d+` và `d−` mỗi bên α/2 | Đúng: đây là hành vi mong muốn của một bound không tham số |
| Thiếu power | Bảng §3: với R road độc lập, ngay cả `d ≡ 0` cũng cần R ≥ 72 để chứng nhận margin 0,05 | Đúng, và tính được trước |
| Lệch giữa margin và thông tin | Margin nhỏ hơn sàn `1 − (α/2)^{1/R}` (α = 0,05) thì không thể GO với bất kỳ dữ liệu nào | Đúng cho margin nhỏ hoặc R nhỏ |

**Không được** gọi gate là "không đáng tin" vì nó từ chối tuyên bố tương đương. Đổi lại, cũng không được kỳ vọng control re-export của S2 tự động ra GO: dưới thiết kế **unseeded** hiện tại, teacher và bản re-export bit-identical chính là "noisy equivalent pair" của S1.

### Một phương án thay bound đã được kiểm và **bị bác**

Giả thuyết ban đầu của audit: một bound thích nghi theo phương sai trực tiếp trên `d` sẽ chặt hơn cách tách `d+/d−`. Kiểm bằng empirical Bernstein (Maurer và Pontil, 2009) trên cùng dữ liệu mô phỏng [CPU, `bound_compare.py`]:

| R | f (độ ghép cặp còn lại) | Median upper, gate khoá | Median upper, empirical Bernstein |
|---:|---:|---:|---:|
| 100 | 1,0 | 0,0914 | 0,1947 |
| 100 | 0,2 | 0,0565 | 0,1830 |
| 441 | 1,0 | 0,0482 | 0,0595 |
| 441 | 0,2 | 0,0235 | 0,0483 |

Bound khoá của gate **chặt hơn** ở mọi ô. Đề xuất "mở khoá `gate/stats` để đổi bound" vì thế không có căn cứ; nó bị loại khỏi danh sách ứng viên. (Một confidence sequence kiểu betting có thể chặt hơn; điều đó chưa được kiểm và không được tuyên bố.)

## 3. Power vận hành: số road là trần, ghép cặp nhiễu là đòn bẩy

Mô phỏng dùng **chính** `gate.verdict.paired_road_bounds` (import chỉ đọc). Mô hình minh hoạ: 441 scene chia đều cho R road độc lập; kết cục nhị phân; tỉ lệ biên p giống nhau ở hai arm (tương đương); `f` là phần discordance còn lại so với hai arm độc lập (`f = 1`: unseeded; `f < 1`: dùng chung số ngẫu nhiên). **p, f, R, margin đều minh hoạ, không phải giá trị đo hay margin của Huy.** 400 trial mỗi ô [CPU, `crn_power.py`, seed 20261105].

**Sàn tuyệt đối [đại số + CPU]:** với mọi `d_r = 0`, upper = `1 − (α/2)^{1/R}` (α = 0,05). Số road tối thiểu để chứng nhận: 36 (margin 0,10), 51 (0,07), 72 (0,05), 183 (0,02). Ngưỡng này áp dụng **cho từng stratum**, vì gate đòi mọi stratum đã đăng ký đều GO.

Xác suất GO khi hai checkpoint thật sự tương đương, p = 0,04:

| R road | Replicate | f = 1 (unseeded) | f = 0,5 | f = 0,2 | f = 0,05 |
|---:|---:|---|---|---|---|
| 100 | 1 | 0,3% @0,05 · 43% @0,10 | 1,3% · 95% | 22% · 100% | 82% · 100% |
| 100 | 3 | 0% · 82% | 0% · 100% | 6,8% · 100% | 93% · 100% |
| 200 | 3 | 0,5% · 100% | 37% · 100% | 99% · 100% | 100% · 100% |
| 441 | 1 | 1,3% @0,02 · 50% @0,05 | 5,2% · 92% | 33% · 100% | 94% · 100% |
| 441 | 3 | 0% @0,02 · 59% @0,05 | 0,5% · 99% | 20% · 100% | 98% · 100% |

Mỗi ô ghi P(GO) ở hai margin minh hoạ: hàng R = 100 và 200 là margin 0,05 · 0,10; hàng R = 441 là margin 0,02 · 0,05. Với R = 50, margin 0,05 **không bao giờ** GO, bất kể ghép cặp (sàn 0,0711).

Ba hệ quả:

1. **Số road độc lập là trần cứng.** Nếu 441 scene đến từ ít road, hoặc bị chia thành nhiều stratum, margin nhỏ là bất khả thi dù chi bao nhiêu replicate. Đây là câu hỏi đầu tiên S2 phải trả lời, trước cả chi phí.
2. **Replicate là đòn bẩy yếu.** Ở R = 441, p = 0,04, margin 0,05: từ 1 lên 3 replicate (gấp ba số rollout) chỉ nâng GO từ 50% lên 59%.
3. **Ghép cặp nhiễu là đòn bẩy mạnh và gần như miễn phí.** Cùng ô đó, giảm discordance còn 20% (f = 0,2) đưa GO lên 100% mà không thêm rollout nào. Với control re-export bit-identical, nếu pipeline tất định thì `f = 0`, mọi `d_r = 0`, và GO chỉ còn phụ thuộc R.

Mô hình này bỏ qua: dị biệt giữa road, endpoint liên tục, tương quan trong road, và việc coupling suy giảm khi policy phân kỳ. Nó chỉ chứng minh **cơ chế** và chiều tác động; độ lớn thật cần đo.

### Vì sao common random numbers không phá giả định của gate [đại số]

Gate chỉ giả định các `d_r` độc lập giữa road và bị chặn trong [−1, 1]. Nếu mỗi (scene, replicate) nhận một seed rút ngẫu nhiên, khai báo trước, và **cùng** seed được dùng cho cả hai arm:

- phân phối biên của từng arm không đổi, nên estimand không đổi;
- seed độc lập giữa các road, nên `d_r` vẫn độc lập;
- chỉ hiệp phương sai trong cặp tăng: `Var(B − A) = Var(A) + Var(B) − 2 Cov(A, B)`.

Do đó bound Chernoff vẫn hợp lệ nguyên vẹn; chỉ power thay đổi. Không cần sửa file khoá nào. Điều kiện: seed không được chọn sau khi xem kết quả, không được tái dùng seed giữa pilot và full, và đường code phải **thật sự tiêu thụ** seed. Trên driver Alpamayo của `e2e_challenge` tại `e121e59`, điều này **không đúng** (`docs/research/2026-10-31-s2-upstream.md`, mục Seed mode).

**Khi nào coupling mất tác dụng [giả thuyết có cơ chế]:** với flow-matching, hành động là `a = Φ_θ(ε, o)`. Cùng `ε`, sai khác giữa hai checkpoint bị chặn bởi `L_θ·‖θ − θ′‖` nếu `Φ` Lipschitz theo tham số; với `ε` độc lập, sai khác bằng cả độ trải của phân phối hành động. Trong closed loop, sai khác nhỏ có thể khuếch đại tại điểm rẽ nhánh (vượt trái hay phải, nhường hay đi) và qua renderer không tất định. Vì vậy f phải được **đo** cho từng loại cặp: control re-export, snapshot RL sớm, checkpoint RL cuối.

## 4. Toán của estimator hành động (đối chiếu lại D-022)

Tất cả giá trị dưới đây được tính lại [CPU, `oracles.py`] từ công thức pinned `(r − mean)/(np.std(r) + 1e-6)` (Cosmos-RL `d2a2c57`, `grpo.py` dòng 38–47; `unbiased` bỏ phép chia nhưng AlpaGym không lộ cờ này).

| Nhóm | Advantage | Ý nghĩa |
|---|---|---|
| G = 2, chênh 0,001 | ±0,998 | Chênh 0,1 m qua hạng `−0,01·D` có cùng trọng lượng với một va chạm |
| G = 2, (1, −9) do va chạm | ±1,000 | Hệ số −10 chỉ còn tác dụng qua **thứ tự** |
| G = 2, chênh 1e−5 | ±0,833 | Epsilon chỉ cắt khi chênh nhỏ hơn cỡ 1e−5 |
| G = 6, một va chạm | −2,236 = −√5 | Độc lập với mức phạt |
| G = 6, chỉ nhiễu | rms 0,9999 | Nhóm toàn nhiễu vẫn mang "năng lượng" advantage đầy đủ |

**Tỉ số likelihood [đại số]:** log-density được lấy trung bình trên mọi chiều không phải batch. Hai nhân tử có tỉ số 4 và 1 cho tỉ số chung 4 nhưng trung bình hình học 2. Tỉ số hình học 1,2 trên 640 phần tử tương ứng log-ratio chung 116,7 nat. Clip 0,8–1,2 vì vậy không phải trust region theo nghĩa phân phối chung.

**KL [nguồn + đại số]:** `kl_beta = 0` ở mọi preset; reference không được dựng; KL log ra bằng 0 theo cấu trúc. KL giữa hai Gaussian phương sai đơn vị lệch 0,1 là 0,005, nên kiểm thử KL phải có ca khác 0.

**Đánh giá lại hai giả thuyết D-022.**

- *G1, magnitude-blind/noise-ranked:* đúng về đại số ở G = 2. Hành vi dự đoán: gradient theo hướng giảm va chạm cần nhiều nhóm hơn so với advantage chỉ trừ trung bình (toy của gap review: 87–88 so với 29 nhóm ở tỉ lệ 0,08). Điều này **chưa** là thất bại lái xe đã đo. Thiếu một đo lường then chốt, và cách đo hay được nhắc tới là sai: re-simulate một cặp với **seed mới** tạo ra một cặp mới có thể hoán đổi, nên thứ tự trùng với thứ tự cũ với xác suất 1/2 bất kể tín hiệu (lane B, §2.3). Chỉ **replay cùng seed** mới cô lập được nhiễu ngoại sinh (renderer, Harmonizer, batch, số học GPU). Với mô hình `R = a(ξ) + ε`, tỉ lệ đảo thứ tự khi replay cùng seed là `arccos(ρ)/π`, `ρ = σ²_a/(σ²_a + σ²_e)`. Ở pin của Gym, seed phiên là `random.randint` và không đặt được, nên cần một harness ánh xạ `session_uuid → seed đã ghi`; xem `candidates.md`.
- *G2, unanchored drift:* không có anchor không phải là bug. Telemetry KL = 0 là **thiếu thông tin**, không phải bằng chứng drift nhỏ. Đo bằng khoảng cách offline tới checkpoint zero-update trên probe cố định. Thêm anchor là can thiệp mục tiêu của Huy.

Giữ nguyên thứ tự baseline trước: cả hai là cơ chế hợp lý, chưa là thất bại đo được.

## 5. Reward huấn luyện so với gate đánh giá

`progress_safety` tại AlpaGym `972d160` [nguồn, `conf_progress_safety.yaml`]: `R = P − 10·C_any − 5·O − 0,01·D`.

| Thành phần | Tín hiệu, đơn vị, thời gian | Trong evaluator offline (AlpaSim `46b401b`, `processing.py` dòng 38–66) |
|---|---|---|
| `P` progress | chiếu cuối lên GT đã chuẩn hoá; không đơn vị | `progress_clipped_rel` tính sau khi bỏ warm-up và cắt sau sự kiện đầu |
| `C_any` | MAX theo thời gian của front ∨ rear ∨ lateral, **gồm warm-up** trong aggregate online | `collision_at_fault = front ∨ lateral`, **loại rear**; bỏ bước trước `eval_relevant` |
| `O` offroad | MAX theo thời gian, heuristic hình học | Cắt các bước sau sự kiện offroad/collision đầu tiên |
| `D` | RMSE vị trí XY thực thi so với GT, mét | Không có tương ứng trực tiếp |
| Ảnh đen | Không xuất hiện trong reward | Phụ thuộc nhánh: ở `main` và pin của Gym `46b401b`, `RemoveTrajectoryWithEvent(img_is_black)` **xoá nguyên quỹ đạo**; ở nhánh đánh giá `e2e_challenge`, modifier này đã bị PR #109 gỡ (2026-06-16), nên ảnh đen dẫn tới va chạm/offroad/corridor bị **chấm như kết cục lái** (lane C) |

Bốn khác biệt có thể tạo động cơ lệch [giả thuyết có cơ chế, chưa đo]:

1. **Bị tông đuôi.** Traffic là log replay không phản ứng. Nếu policy phanh mạnh hơn người lái trong log, xe phía sau trong log vẫn chạy như cũ và tông đuôi. Reward phạt −10 (`C_any` gồm rear); gate at-fault không phạt. Động cơ: bám tốc độ của log, kể cả khi phanh phòng vệ là hợp lý. Đây là khác biệt giữa *reward* và *metric đánh giá*, không phải lỗi của một bên.
2. **Warm-up.** Theo audit 11-01 (`nvidia-stack.md`), aggregate online **có thể** gồm sự kiện trong 1,6 s GT cưỡng bức, còn offline bỏ chúng; PR #67 (merged 2026-04-10) chỉ bỏ timestep prerun, chưa có fixture chứng minh phạm vi. Đây là chênh lệch tiềm năng, chưa đo. Nếu cả nhóm cùng chịu, advantage centered loại nó; nếu không cùng (khác nhau do nhiễu renderer), nó thêm nhiễu xếp hạng.
3. **Sau sự kiện đầu.** Online lấy MAX nên các hàng sau va chạm vẫn mang advantage của cả tập; offline cắt sau sự kiện đầu. Một advantage cho mọi bước hợp lệ nghĩa là hàng sau va chạm cũng được "dạy".
4. **Ảnh đen, hai lỗi ngược chiều.** Nếu policy lái vào vùng tái dựng kém và sinh ảnh đen: trên `main`/pin Gym, evaluator xoá cả quỹ đạo khỏi tử số **và** mẫu số (complete-case deletion phụ thuộc policy, có thể làm checkpoint trông tốt hơn); trên nhánh đánh giá `e2e_challenge`, lỗi renderer bị tính là lỗi lái (có thể làm checkpoint trông tệ hơn). Cả hai đều cần một trường `render_status` riêng trong ledger và bảng 2×2 (A hợp lệ, B hợp lệ) theo stratum (ứng viên 1 của lane C, `candidates.md`).

**Lỗi phân loại va chạm đã được người khác báo:** `CollisionScorer` reset cờ trong vòng lặp actor nên chỉ giữ phân loại của actor cuối mỗi timestep [nguồn, cùng mã ở `46b401b`, `e121e59` và `main affc2ea`]. `collision_any` vẫn đúng; `front/rear/lateral` và do đó `collision_at_fault` có thể sai khi hai actor chạm ego cùng lúc. PR #234 (mở 2026-11-01, erkara) và issue #237 (mở 2026-11-04, coldstone7) đã có. PhysLoop **không** mở PR cạnh tranh; chỉ ghi phiên bản đã pin và thêm known-answer test nhiều actor vào audit extraction.

**Ba tầng cải thiện phải tách:** reward huấn luyện tăng; một metric mô phỏng (ví dụ at-fault) cải thiện; và kết luận dưới luật quyết định đã đăng ký (GO/NO-GO/INSUFFICIENT trên mọi stratum). Tầng sau không suy ra từ tầng trước. Gate offline **không độc lập** với reward theo nghĩa nhân quả: chúng chung scorer hình học, chung GT, chung renderer; một lỗi ở scorer va chạm sẽ lệch cả hai cùng chiều.

## 6. Validity, missingness, contamination

- **Bin độ lệch là hậu-điều trị.** Policy quyết định nó lệch bao xa khỏi log. So sánh chỉ trong bin `<1 m` chọn hai quần thể khác nhau cho hai arm. Dùng bin làm chẩn đoán; không làm stratum so sánh trừ khi Huy đăng ký estimand có điều kiện. Gate hiện chỉ **đếm** `≥4 m` (`exits`), không lọc; giữ nguyên.
- **Năm loại thất bại** phải tách trong ledger: hạ tầng (container, mạng, OOM), renderer/simulator (ảnh đen, lỗi NRE), trạng thái không hợp lệ do policy (rời vùng tái dựng), kết cục lái bất lợi thật (va chạm, offroad), artifact thiếu/hỏng. Hai loại đầu có thể độc lập với policy; loại ba và bốn thì không.
- **Phân tích độ nhạy khi thiếu dữ liệu:** với rollout thiếu ở arm B, báo hai cận: coi mọi rollout thiếu là thất bại (worst case) và là thành công (best case); verdict chỉ được báo nếu cả hai cận cùng kết luận, ngược lại báo INSUFFICIENT do thiếu dữ liệu. Không bao giờ điền 0 cho giá trị thiếu.
- **Contamination:** cần audit theo road, capture, scene, asset tái dựng, calibration, dùng trong huấn luyện và chọn checkpoint. Không thể tuyên bố đã loại trừ trùng lặp với dữ liệu pretraining của teacher khi provenance không công khai. Nhãn `alpamayo_aligned` trong parquet không tự nó là bằng chứng loại trừ (đã nêu trong S2.5).

## 7. Điều audit này không thiết lập

Không có gì ở đây đo f, R, tỉ lệ va chạm cơ sở, độ tất định của renderer, hay bất kỳ hiệu ứng học nào. Toàn bộ bảng là cơ chế và độ nhạy. Các giá trị thật thuộc S2 pilot và S2 full sau khi Huy đăng ký.
