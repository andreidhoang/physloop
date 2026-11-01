# S2 hồ sơ duyệt thuê GPU và license

**Plan v5 scope (D-021):** S2 qualifies the teacher/re-export/same-policy controls for the S4 expert-RL study; full TransFuser/FP8 comparisons and later student/device stages are deferred. The five-scene pilot, full 441-scene protocol, ≥3 unseeded full replicates, nested caps and pending scientific/license/rental approvals remain. The closed D-019 rental does not authorize another rental.

**Cập nhật 2026-11-01.** D-018 đã chấp thuận P-030 evaluator engineering và các trần $100 pilot / $1.200 S2 / $2.500 cloud. D-019 tiếp tục ghi quyền thuê một Vast VM, setup, kiểm tra bằng phần mềm công khai/input tổng hợp và cleanup trong giới hạn riêng: ≤$10 all-in, ≤$2,50/h compute, deadline stop ngoài VM ở phút 60 và startup failure budget 20 phút. Quyền này đã được sử dụng cho instance `58628722`; không yêu cầu Huy duyệt lại host test đó. Pilot khoa học, nạp credit mới, gated-asset acceptance/download và chuyển dữ liệu được cấp phép vẫn chưa được duyệt.

**Mốc lịch sử:** bản 2026-10-31 ghi chưa thuê VM và chi phí $0; đây không còn là trạng thái hiện tại. Chi phí thực tế và xác nhận xóa tài nguyên được ghi riêng trong [báo cáo host test](research/2026-11-01-vast-host-validation.md), không suy ra từ quote. Phần mềm công khai đã dùng, gồm CUDA, vẫn chịu điều khoản sử dụng tương ứng; không có gated-asset acceptance/download trong host test.

Agent nhận trách nhiệm triển khai evaluator; Huy giữ quyết định khoa học, prediction, margin và model-research math. Hai contract đã cập nhật cùng nhau. Các entry đang khóa trong `.claude/starred-cores.txt` vẫn giữ nguyên; việc chuẩn bị hồ sơ này không cần unlock chúng.

## 1. Host đã kiểm tra và cấu hình dự phòng

**Vast là đường triển khai đã có kiểm tra thực tế trên một host.** Guest KVM Ubuntu 22.04 có root, RTX PRO 6000 Blackwell **Workstation Edition** 97.887 MiB VRAM và driver thực tế 580.95.05. CUDA integer oracle/repeat/negative control, kiểm tra khoảng 70,82 GiB VRAM, buildx và Compose DNS/network/shared mount/guest-host PID/GPU đều có bằng chứng thực thi. Nsight Compute đọc counter kernel thật; Nsight Systems ghi ba CUDA kernels. DCGM 4.7.1 có tensor-active đáp ứng idle → khoảng 0,576 → idle; exporter 4.8.4 với engine 4.6.1 riêng, embedded và chỉ listen loopback, cũng có scrape đáp ứng tải tổng hợp. [Bằng chứng và giới hạn](research/2026-11-01-vast-host-validation.md).

XID vẫn N/A trong DCGM và không xuất hiện trong scrape. NVML báo mask `0xf19c`, đăng ký event thành công, rồi wait timeout; chưa chứng minh đường phát hiện/delivery lỗi. Guest RAM thực tế 96,996 GiB, thấp hơn listing 123,438 GiB khoảng 21,42%; sizing phải dùng guest thực tế. Chưa chạy renderer/model/Harmonizer đồng thời, chưa đo memory fit hoặc throughput AlpaSim. Đây không phải quyết định chọn Vast lâu dài cho dữ liệu được cấp phép.

**Nebius bên dưới là phương án dự phòng chưa thuê hoặc kiểm thử.** Quote và giới hạn tám giờ chỉ là đề xuất cho scientific pilot, không phải giới hạn của host test D-019.

| Hạng mục | Đề xuất |
|---|---|
| Nhà cung cấp | **Nebius**, VM Regular on-demand, không commitment/spot |
| Khu vực chính | `us-central1`, Kansas City |
| Platform / preset | `gpu-rtx6000` / **`1gpu-24vcpu-218gb`** |
| GPU | **1 × NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB** |
| CPU / RAM | 24 vCPU / 218 GiB; phải xác minh đã nằm trong báo giá compute |
| Image | Family `ubuntu24.04-cuda13.0`; ghi image ID/driver thực tế khi tạo |
| Disk | **1 × 500 GiB managed Network SSD**, boot và work chung; không tạo bucket, snapshot hay filesystem bổ sung |
| Compute limit cần duyệt | **8 GPU-giờ cộng dồn**, bao gồm setup, tải image, warm-up và idle; đơn giá toàn bộ compute tối đa **$2,50/GPU-giờ** |
| Dừng | Chủ động stop qua control plane ở 7,5 giờ đã tính phí; xác nhận Stopped trước trần 8 giờ |
| Giữ dữ liệu | Export evidence hợp lệ vào nơi lưu riêng; xóa VM/disk chậm nhất bảy ngày sau tạo |
| Nếu hết chỗ | Có thể trình phương án Madrid `eu-south1` / `gpu-rtx6000-a`, cùng preset một GPU; không tự đổi region hoặc mua preset tám GPU |

Nguồn xác minh cấu hình: [Nebius platforms](https://docs.nebius.com/compute/virtual-machines/types) và [GPU images](https://docs.nebius.com/compute/storage/boot-disk-images). Đây là cấu hình được công bố, chưa phải một VM đã giữ chỗ. Một GPU 96 GB là ứng viên bootstrap; khả năng chạy đồng thời teacher BF16 + renderer + Harmonizer phải được kiểm tra trong cap. Nếu thiếu VRAM, stop và lập phương án khác, không tự nâng GPU count.

**Giá dự kiến:** 8 giờ × $1,80 = **$14,40 compute**; 500 GiB SSD × 168 giờ ở biểu giá hiện tại ≈ **$8,17**; tổng cơ sở ≈ **$22,57**, chưa gồm khoản phí/thuế thực tế chưa xác minh. Giá $1,80 áp dụng cho preset đầy đủ là suy luận cần được itemized quote xác nhận. [Nebius pricing](https://docs.nebius.com/compute/resources/pricing).

**Khoản thanh toán đầu tiên có thể phát sinh ngay:** thêm thẻ mới yêu cầu **$25**, ghi thành credit tài khoản. Nếu sau đó dịch vụ dùng $22,57 từ credit, không cộng thành $47,57 tiền chi; ledger tách tiền đã trả, usage accrued, credit còn lại và chi phí lưu trữ chưa tất toán. Không nạp thủ công thêm hoặc tăng cấu hình theo hồ sơ này. PAYG có cơ chế tự thu tiền ở ngưỡng số dư âm/đầu tháng; số dư và email budget không phải hard cap. [Card payments](https://docs.nebius.com/signup-billing/payments/card), [PAYG](https://docs.nebius.com/signup-billing/billing-models/payg).

Runpod Pods thông thường bị loại cho cấu hình này vì không cho chạy Docker Compose/Docker instance riêng. Giá GPU thấp hơn không đủ để xác nhận host tương thích. [Runpod limitations](https://docs.runpod.io/pods/overview#limitations).

**Snapshot công khai trước khi thuê:** search Vast lúc 13:29 ICT ngày 2026-11-01 tìm thấy năm offer Workstation 96 GB có VM; ví dụ Hàn Quốc $1,4013/h compute + $0,1389/h cho 500 GB disk. Đây là quote lịch sử, khác cấu hình disk/order của host test; không dùng nó làm hóa đơn hoặc bảo đảm offer còn tồn tại. [Phân tích ban đầu và ghi chú supersession](research/2026-11-01-vast-vm-fit.md). Full KVM VM cung cấp quyền guest cần cho Compose; không chứng minh Secure Cloud, quyền truy cập dữ liệu hay renderer compatibility trên mọi host.

## 2. Bước còn lại trước scientific pilot

Các bước Nebius dưới đây chỉ áp dụng nếu chọn phương án dự phòng. Nếu chọn một Vast offer tiếp theo, ghi order/quote, access plan và lifecycle cụ thể tương đương; không coi host đã kiểm tra là quyền thuê thêm hoặc tự động chuyển provider trong draft.

1. **Đăng nhập tài khoản Nebius hiện có để xem thông tin.** Kiểm tra ngày 2026-10-31 mở console tới trang đăng nhập, chưa vào được tài khoản; không gửi biểu mẫu đăng ký hay thông tin thanh toán. Trang được giữ lại tại [Nebius console](https://console.nebius.com). Nếu cần tạo tài khoản mới, Huy xem/đồng ý điều khoản nhà cung cấp; không đi qua bước thêm thẻ trước khi duyệt khoản $25.
2. **Lấy quote tài khoản:** kiểm tra quota/capacity; mở cấu hình ở bảng trên đến bước Review; lưu thời điểm, GPU count, region, itemized rate, disk, thuế/FX và billing minimum. Dừng trước Create VM. Nếu console bắt nạp $25 mới xem được quota, duyệt riêng khoản kích hoạt tài khoản $25; khoản này không bảo đảm có GPU và không cho phép tạo VM.
3. **Hoàn thành license/access review** bên dưới, chọn five-scene allowlist sau khi quyền truy cập và thời điểm S2 hợp lệ. Tải metadata trên máy không tính tiền GPU khi được phép, thay vì thuê VM rồi chờ access. Không tải cả corpus ~2,9 TB.
4. **Chốt pilot registration:** BF16, Harmonizer off/on, batch một, graph off, unseeded replicates, background traffic replay với ego do model điều khiển; prediction do Huy ghi trước measured run. Pilot chỉ đo compatibility/cost; không phải verdict model.
5. **Chuẩn bị cutoff trước Create VM:** có người vận hành được chỉ định và monitor/timer ngoài VM đã được chuẩn bị/kiểm thử để gọi control-plane stop; ngay sau Create mới bind VM ID thực tế và xác nhận monitor đã armed trước bootstrap. Theo dõi thời gian tính phí kể cả restart, dự phòng storage/thuế và total ≤$100. Watchdog theo exact instance ID + unique label đã được triển khai/kiểm thử và arm cho host test Vast; một rental mới vẫn cần deadline/binding và xác nhận riêng. Không thay cutoff bằng email alert.

Đối với scientific pilot tiếp theo, Huy duyệt lệnh tạo đúng VM sau quote và cutoff cụ thể. Không duyệt lại các trần tiền hoặc quyền host test D-019 đã cấp; phần đang chờ là scientific pilot và hành động/account/terms chưa được duyệt. Start gate khoa học vẫn là 2026-11-02 và xác minh challenge đã đóng; không suy ra gate đã qua chỉ từ ngày trên lịch.

## 3. License cần Huy xem và chấp nhận

| Asset | Trang chính thức / bước cần làm | Phạm vi pilot |
|---|---|---|
| PAI-AV | [Dataset page](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles); đọc và bấm nút chấp nhận agreement của dataset | Chỉ metadata/split cần thiết, không raw camera/lidar corpus |
| NuRec | [Dataset page](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec); agreement riêng, chấp nhận riêng | Năm scene đã chọn cùng metadata cần thiết |
| Alpamayo 1.5 10B | [Model và LICENSE](https://huggingface.co/nvidia/Alpamayo-1.5-10B); kiểm tra terms tại revision đã pin | Teacher BF16; hiện public page không có HF gate |
| NRE 26.04 | [NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nre/containers/nre-ga); đọc NVIDIA Software License Agreement và product terms được catalog dẫn tới | Renderer container; NGC individual account miễn phí, không mua enterprise subscription |
| Harmonizer | [Model page](https://huggingface.co/nvidia/Harmonizer) và NVIDIA Open Model License | Checkpoint non-temporal cho on arm; không training dataset |

Public dataset gates có điều khoản cho cá nhân đủ tuổi; không thấy yêu cầu email tổ chức tại các gate đã kiểm tra. Huy tự chấp nhận dưới đúng danh tính. Access grant, accepted terms, first download và quyền công bố là các thông tin khác nhau. [Bản audit license và thao tác chính xác](research/2026-10-31-s2-license-approval.md) có URL, nút bấm, hash legal text và các dependency cần xác minh.

Tạo token HF read phạm vi tối thiểu và NGC Catalog key có expiry; chỉ lưu secret trong nơi quản lý credential riêng, không gửi token vào chat/Git/manifest. PAI-AV/NuRec có mốc 12 tháng từ initial delivery/download; metadata cũng thuộc phạm vi dataset, nên ghi mốc đầu tiên cho từng dataset ngay khi nhận. Chưa có mốc nào được khởi động trong công việc này.

**Cloud access cần được xét theo cách dùng thật:** private VM/storage do Huy kiểm soát, quyền hỗ trợ của provider, bản sao, backup, retention và ai nhận dữ liệu/kết quả. Điều khoản hạn chế hosting/disclosure không tự chứng minh mọi private VM được phép hoặc bị cấm. Nếu không xác định được cách dùng phù hợp, làm rõ trước transfer/rental thay vì coi budget approval là license permission. Không gửi licensed frames, metadata hay benchmark outputs vào dịch vụ chat/model bên ngoài chỉ vì P-030 đã giao việc cho agent.

Không tự xin/tải toàn bộ Cosmos-Reason2: chưa xác minh nó là dependency bắt buộc của teacher loader đã chọn. Harmonizer liên quan Cosmos-Predict2; chỉ thêm dependency gated khi xác minh NRE image thực sự cần fetch riêng, rồi trình terms tương ứng. Public processor/config dependencies cũng được pin và ghi nguồn. Public 26.10 mới hơn không tự thay renderer 26.04 trong protocol.

Dataset confidentiality/publication, derived labels/weights và renderer benchmark disclosure có điều khoản riêng. Hồ sơ pilot này chỉ phục vụ nghiên cứu nội bộ; KC-L/KC-L2 và renderer-publication review chưa qua. Chưa gửi bất kỳ yêu cầu nào cho NVIDIA.

## 4. Sau khi Huy duyệt scientific pilot và hành động cụ thể

1. Nếu cần, Huy thực hiện account/license acceptance và khoản credit $25 đã duyệt; ghi receipt, không ghi secret.
2. Xác minh các gate trước run, final quote và cutoff. Tạo đúng một VM, bind monitor, bắt đầu ledger ngay. Nếu không arm/verify được cutoff, stop trước khi làm tiếp.
3. Kiểm tra CUDA/GPU identity, Docker/Compose, DCGM, renderer, VRAM và immutable pins. Bootstrap cũng tiêu GPU-hours; nếu thất bại, giữ logs và stop thay vì tự mở rộng cap.
4. Chạy đúng pilot đã đăng ký khi preflight và bằng chứng đáp ứng yêu cầu. Chưa mở full sweep.
5. Với phương án Nebius tám giờ nếu được duyệt, dừng sớm ở 7,5 billed hours hoặc sớm hơn khi dự báo chạm trần. Dùng `nebius compute instance stop --id <VM_ID>` hoặc Compute → Virtual machines → Stop, rồi xác minh **Stopped**. Guest Linux shutdown có thể khiến Nebius restart và tiếp tục tính phí. Với Vast, dùng control-plane action và deadline của order được duyệt; giới hạn D-019 vẫn là 60 phút cho host test. [Nebius stop procedure](https://docs.nebius.com/compute/virtual-machines/stop-start).
6. Export/checksum evidence vào nơi riêng được phép; xóa VM cùng managed disk trước retention deadline. Kiểm tra riêng standalone disks/snapshots/buckets nếu có. Approval lệnh thuê bao gồm cleanup đúng VM/managed disk sau khi export/checksum đã xác minh; không xóa bản evidence duy nhất hoặc tài nguyên ngoài phạm vi đã duyệt. [Delete procedure](https://docs.nebius.com/compute/virtual-machines/delete).

## 5. Những mục duyệt còn lại

- **Terms/access:** Huy xác nhận danh tính chấp nhận, asset allowlist, private-host/access/retention plan và tự chấp nhận các agreement thực sự cần.
- **Funding nếu cần:** khoản credit ban đầu đúng $25 cho Nebius, với điều kiện thanh toán/PAYG đã xem; không tự suy thành quyền thuê máy.
- **Order scientific pilot sau khi có quote:** ghi đúng provider/VM, operational limits và disk lifecycle, dưới pilot cap đã được duyệt; quote/time cutoff phải được đính kèm. Nebius là phương án dự phòng, Vast là route có host evidence nhưng chưa có licensed-provider adoption. Thay đổi region/SKU/GPU count cần được ghi rõ trong approval.
- **Scientific pilot:** prediction và registration vẫn là điều kiện trước measured run. Full-sweep release là quyết định sau pilot, không nằm trong hồ sơ order đầu tiên.

Host test Vast đã có quyền D-019 và bằng chứng thực thi; xem báo cáo riêng cho chi phí/cleanup. **Scientific pilot vẫn chờ license receipts, private-host/access plan, metadata/scene allowlist, protocol/prediction, quote mới và cutoff phù hợp.** Nebius login/quote/capacity và runtime chưa được xác minh. Không tự chuyển các draft attestation sang true chỉ từ kết quả kiểm tra phần mềm công khai.
