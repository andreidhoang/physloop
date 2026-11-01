# Phân công nghiên cứu PhysLoop: Huy làm model, agent vận hành đánh giá

P-030 đã được Huy chấp thuận ngày 2026-10-31, ghi tại D-018, cùng các trần $100 pilot / $1.200 S2 / $2.500 cloud. Agent nhận ownership evaluator engineering; Huy giữ quyết định khoa học và model research. D-021 ngày 2026-11-01 đã đổi hướng chính thành S1 → S2 → S4 → kết luận nghiên cứu RL; S3/S5–S8 được hoãn khỏi nghiên cứu đầu tiên, còn file lock không thay đổi; hồ sơ thuê máy/license cụ thể vẫn cần duyệt trước chi tiền. Đây là dự án nghiên cứu độc lập, không phải tư cách đại diện NVIDIA.

## 1. Phân bổ tốt nhất cho mục tiêu của Huy

Sau khi S2 đủ điều kiện, đề xuất dành **70% thời gian Huy cho giả thuyết và can thiệp model**, **20% cho đọc kết quả, ablation và chẩn đoán**, **10% cho review các giả định đo lường quan trọng**. Đây là phân bổ tương đối, không tự đặt số giờ mỗi tuần thay P-003. Trong S2, phần lớn việc triển khai là của agent; Huy tập trung chốt câu hỏi, margin, strata, prediction và hiểu giới hạn của máy đo.

| Công việc | Huy giữ quyền quyết định | Agent gánh phần lặp lại |
|---|---|---|
| Research | Câu hỏi, cơ chế, giả thuyết có thể bị bác bỏ; biến can thiệp và ablation | Đọc nguồn chính thức, đối chiếu phiên bản, chuẩn bị bảng bằng chứng và review độc lập |
| Training/RL/distillation | Loss, estimator, reward, reference policy, dữ liệu được phép học, prediction | Cấu hình, checkpoint lineage, launch trong cap được duyệt, resume, logging, kiểm thử |
| Evaluation | Estimand, mức suy giảm chấp nhận, strata, validity, chẩn đoán | Ghép cặp, lịch chạy, audit thiếu/trùng, fixtures, báo cáo, plots, telemetry, tính toán theo core đã duyệt |
| Data | Tiêu chí contamination, quy tắc chọn dữ liệu, label/reward semantics | Joins, checksums, split audits, dedup, danh sách failure trên development set, phiên bản dataset |
| Release/spend | Bằng chứng đủ để chuyển bước; budget và license | Forecast, kiểm tra giới hạn, bản kết quả có thể review, lưu toàn bộ negative results |

Agent không được tự sửa margin sau khi thấy kết quả, suy ra quyền chi tiền từ một JSON đã điền, hay biến code do agent viết thành bằng chứng Huy đã tự thiết kế nó. Yêu cầu S1 trước đó chỉ mở quyền cho S1.

**Phân công đã được chấp thuận theo P-030:** giao cả phần triển khai evaluator core (reliability estimators, validity instrumentation, power/reporting và kiểm thử) cho agent, có review độc lập và known-answer controls; Huy giữ quyết định khoa học về đại lượng cần đo, margin, phạm vi kết luận và cách diễn giải. Như vậy Huy không phải tự viết mọi estimator để giữ quyền điều khiển nghiên cứu. D-018 đã ghi phạm vi ủy quyền và cập nhật đồng bộ `AGENTS.md`/`CLAUDE.md`. Nếu công việc cụ thể cần sửa file còn trong manifest bảo vệ, Huy vẫn tự unlock entry đó; không chuyển logic sang file khác để né guard. Các lựa chọn khoa học vẫn cần Huy chốt, nhưng việc triển khai evaluator ngoài các file đang khóa có thể tiến hành theo stage đã được mở. Loss/reward/estimator của RL và distillation vẫn là trọng tâm thiết kế của Huy.

## 2. Thứ tự nghiên cứu theo plan v5

1. **S2: kiểm chứng máy đo.** Giữ Alpamayo 1.5 làm baseline; kiểm tra protocol, completeness, validity, độ nhiễu, power thực tế và chi phí. Full FP8/TransFuser sweeps được hoãn; teacher/re-export/same-policy controls vẫn bắt buộc. Không tự đổi tập 441 scene hay quy tắc road/strata.
2. **S4: kiểm chứng đường update.** Pin đúng Gym/recipes/Cosmos/simulator; đối chiếu reward online với metric offline; kiểm tra identity, log-prob replay, frozen weights/buffers và zero-update export. Knob staleness không thay bằng chứng version thực tế. Thu raw metrics và toàn bộ retry/drop trước cleanup.
3. **S4: baseline rồi một cơ chế.** Sau admission và forecast, chạy một shipped arm với ít nhất ba training run độc lập nếu đủ ngân sách. Chọn một can thiệp từ development evidence; Huy viết cơ chế, prediction số và falsifier trước comparison. Ba run là mức chẩn đoán tối thiểu, không tự bảo đảm confidence interval tốt.
4. **S4: confirmation và kết luận.** Khóa checkpoint-selection rule, giữ confirmation roads độc lập. Muốn kết luận về training procedure cần fresh training runs sau khi chọn can thiệp hoặc thiết kế selection-aware được review; đổi sang road mới không xóa bias do chọn trên exploratory runs. Báo cáo gain, regression, negative result hoặc INSUFFICIENT.
5. **Dừng để quyết định nhánh sau.** Reasoning RL, Super, reactive traffic, teacher/student correction, Thor/HIL và scaling rộng là proposal riêng sau kết luận đầu tiên. Không tự chuyển sang open-loop GRPO khi Gym khó chạy: đó là câu hỏi và ngân sách khác.

[Review nguồn chính thức và paper](research/2026-11-01-rl-direction-review/report.md) phân biệt direct RL với privileged-teacher imitation và on-policy correction. Chọn direct expert RL vì câu hỏi học máy Huy muốn nghiên cứu, không vì đã chứng minh nó rẻ nhất. Agent không tự thiết kế reward/loss/estimator thay Huy hoặc điền prediction khoa học còn trống.

## 3. Bộ công cụ nên dùng

| Mục đích | Công cụ | Phần PhysLoop cần tự thêm |
|---|---|---|
| Closed-loop rollout và scoring | AlpaSim + NuRec, pin evaluator/renderer | Pairing, admission, validity provenance, road/stratum uncertainty |
| Teacher RL | AlpaGym + Cosmos-RL, smoke trước scaling | Research objective, controls, reward-hacking audit, gate trước/sau |
| Student training/correction | alpamayo-recipes phù hợp với checkpoint, trạng thái PR được ghi rõ | On-policy dataset lineage, ablations và cơ chế correction |
| Quantization | ModelOpt recipe theo phiên bản được pin | P-009 equivalence ladder và measured paired outcome |
| Profiling | AlpaSim Prometheus, host DCGM, Nsight Systems/NVTX, CUDA events | Goodput/cost ledger và liên kết mỗi số với device/protocol |
| Metadata/contamination | physical_ai_av và split metadata của revision đã đăng ký | Road/clip overlap audit, nguồn label, train/dev/eval separation |
| Difficulty và review hỗ trợ | Drive-IRT, AlpaJudge khi đúng stage và có quyền truy cập | Không dùng difficulty như cách chọn riêng cảnh thuận lợi; judge không tự thay verdict |

Không thêm một platform RL tổng quát hoặc một hệ thống tracking trả phí khi bộ NVIDIA đang dùng đã cung cấp đường chạy cần thiết. Trước hết dùng artifacts cục bộ có hash và log; thêm công cụ khi nó giải quyết một trở ngại đo được. Các công cụ này được chọn theo vai trò, không phải tất cả đều được tích hợp hoặc kiểm chứng runtime trong lần chuẩn bị S2 này.

## 4. Vòng cải thiện dữ liệu mà không làm hỏng đánh giá

Tách rõ **train**, **development** và **confirmation evaluation** theo nguồn clip/road trước khi mining. Đối chiếu cả 441 validation và 1,761 training scenes với source-split metadata; tên thư mục không bảo đảm không contamination. Kiểm tra overlap theo road, thời gian liền kề và source clip, không chỉ tên file. Benchmark công khai không tự trở thành một holdout chưa từng dùng.

Trên development pool, agent gom failure bằng quy tắc Huy duyệt: chất lượng render, trạng thái thiếu trong dữ liệu, hành vi của policy và hạ tầng là các nhóm riêng. Huy chọn cơ chế cần sửa. Agent sau đó tạo dataset version có source hash, teacher checkpoint, label method, state-visitation policy, filter và quyền sử dụng. Việc chọn thêm hard examples phải giữ một phần đại diện của phân phối gốc và trọng số sampling được ghi rõ; chỉ học hard cases có thể làm giảm chất lượng ở cảnh bình thường.

Nếu dùng rollout của student để hỏi teacher, teacher đang gán nhãn trên trạng thái mà **student thực sự đi tới**. Đó là lý do on-policy correction có thể khác supervised distillation trên log. Nhưng teacher không mặc nhiên đúng ở những trạng thái ngoài phân phối; cần quality/validity filter và ablation. Không sửa lỗi renderer bằng cách dạy policy bắt chước nhãn lỗi.

Không đưa nhãn hoặc failure đã mining từ confirmation set vào training rồi tiếp tục gọi set đó là holdout. Nếu đã xem để thiết kế model, ghi rõ nó là development evidence và cần confirmation khác. S1 kiểm soát lỗi cho phép so sánh đã đăng ký; nó không tự kiểm soát việc thử hàng trăm checkpoint rồi chọn kết quả đẹp nhất. Ngân sách lựa chọn model và lần confirmation cuối phải được ghi riêng trước khi xem kết quả.

Labels/weights có nguồn PAI-AV/NuRec vẫn đi qua KC-L2 trước publication. Việc giữ nội bộ hoặc chạy được tool không phải bằng chứng quyền phân phối. Agent chuẩn bị lineage; Huy quyết định và ghi các phê duyệt cần thiết.

## 5. Sáu điều Huy cần hiểu để điều khiển nghiên cứu

**Một: estimand là câu hỏi toán học.** “Trung bình trên tất cả scene” và “chọn đều một road rồi lấy trung bình trong road” có thể cho kết quả khác. Nếu road dài được lấy nhiều scene, scene mean ưu tiên road đó. Gate v0 hiện dùng equal-road. Hãy nói được chính xác kết quả của mình đại diện cho tập nào.

**Hai: observed gain khác evidence.** Điểm candidate tăng 1 pp là một quan sát; khoảng bất định có thể vẫn bao trùm suy giảm đáng kể. Non-inferiority GO nói candidate đạt margin đã chốt, không nhất thiết tốt hơn teacher. INSUFFICIENT có thể là kết luận đúng khi dữ liệu không phân biệt được hai khả năng.

**Ba: nhiều replicate không bằng nhiều road.** Replicate giúp thấy policy/render noise trên cùng tình huống. Một nghìn lần chạy lại một road không thay thế một nghìn road độc lập. Vì vậy tăng ngân sách có thể không giải quyết power nếu thiếu diversity và cluster support.

**Bốn: RL tối ưu reward, không tối ưu ý định của người viết reward.** Nếu có cách tăng reward bằng đứng yên, tránh vùng được chấm, hoặc lợi dụng simulator, optimizer có thể tìm ra. Reward tăng chỉ là training signal. Independent metrics, validity records và negative controls là cách thử xem reward có dẫn tới hành vi muốn có không. Huy cần chỉ ra một cách model có thể “lách” mỗi reward trước khi chạy.

**Năm: distillation đổi cả capacity và phân phối trạng thái.** Student có thể khớp teacher tốt trên log nhưng một sai số nhỏ đưa nó đến trạng thái teacher chưa từng dạy. On-policy correction thử xử lý distribution shift này. Nếu loss thấp hơn mà closed-loop xấu hơn, cần phân biệt nhãn, coverage, action head, reasoning budget và controller, thay vì tăng epoch vô điều kiện.

**Sáu: checkpoint tốt chưa phải hệ thống tốt.** Renderer, post-processing, precision và latency đều có thể đổi outcome khi weights giữ nguyên. Một thay đổi một lần và manifest đầy đủ giúp quy kết nguyên nhân. Tốc độ phải đo trên đúng device, sau warm-up, với synchronization thích hợp, và luôn đi cùng kiểm tra hành vi.

## 6. Một mẫu bàn giao cho mỗi model experiment

Huy cung cấp câu hỏi, một biến can thiệp, cơ chế, prediction số, margin/kill rule và ablation có thể bác bỏ giả thuyết. Agent trả lại frozen config/checkpoints/data lineage, forecast và cap, test/negative controls, toàn bộ outcomes với uncertainty và validity, các vi phạm protocol và bằng chứng tái lập. Huy kết luận giả thuyết nào còn đứng vững; agent đề xuất thí nghiệm tiếp theo nhưng không tự đổi tiêu chí sau kết quả.

Bài tự kiểm ngắn: “Nếu student tăng 2 pp trung bình nhưng một stratum hiếm có ít road và nhiều rollout ≥4 m, bằng chứng nào còn thiếu trước GO?” Một câu trả lời đủ cần nhắc estimand, per-stratum uncertainty, road support, validity và margin đã đăng ký. Không cần tự vận hành mọi dashboard, nhưng phải hiểu vì sao những điều đó có thể đảo kết luận.
