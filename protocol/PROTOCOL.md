# Journal Protocol V1.3 (LOCKED) — Pseudo-Cold Leakage and Checkpoint Selection

Ngày khóa: 28/10/2026. Thay thế V1.2 (27/10/2026, chưa từng được hash). Mọi khác biệt so với V1.2 được liệt kê ở mục 11 và trong DEVIATIONS.md.

Tóm tắt: 4 domain Amazon × 4 split mới × 5 training seed = 80 cặp (domain, split, seed); 15 arm; 1.200 trajectory. Confirmatory endpoint là regret trên cold-DEV-700 với hai contrast theo thứ tự P1 → P2, đơn vị là 16 split. Cold-TEST-300 chỉ mở một lần sau khi khóa kết quả DEV. Toàn bộ code được viết mới, độc lập với ADC.

## 0. Bàn giao và cách làm việc

- Bài hội nghị: "When Pseudo-Cold Validation Leaks: Checkpoint-Selection Fidelity in Strict Item Cold-Start Recommendation", ADC 2026 (LNCS), đã được accept. Bài journal là bản mở rộng nộp IEEE Access.
- Reviewer yêu cầu: control clean-replacement, control tắt ID, leakage phủ một phần, nhiều split hơn, so sánh fold giả lập với cold thật, làm rõ cách evaluation.
- Exploratory trên split ADC (data_seed = 137): hiệu ứng Video Games bền với mọi selection rule; hiệu ứng CSJ ở q = 0,3346 (4,58×) biến mất khi làm mượt vì phụ thuộc đỉnh PV ở epoch 4. Đây là nguồn gốc của RQ3.
- Code chạy trên máy người dùng: Windows, Python 3.11.9, numpy 1.26.4, pandas 2.3.3, torch 2.2.2+cu121, torchvision 0.17.2, NVIDIA RTX A5000. Phiên Claude viết code; người dùng chạy và gửi log.
- Trao đổi bằng tiếng Việt; không đưa ra claim không có số liệu kiểm chứng; kết quả bất lợi vẫn được báo cáo.
- Mọi thay đổi sau khi hash phải được người dùng duyệt và ghi vào DEVIATIONS.md.
- Ràng buộc nộp bài: trích bài ADC, nêu rõ phần mới, similarity với bài ADC dưới 35%.

## 1. Nguồn bằng chứng và vai trò

| Nguồn | Vai trò |
|---|---|
| Bài ADC 2026 (VG + CSJ, data_seed = 137) | Prior study, trích dẫn; không tính vào confirmatory |
| Phân tích smoothing/horizon trên split 137 | Exploratory; nguồn sinh RQ3 |
| Cold-DEV-700 của 16 split mới | Confirmatory cho P1, P2 (regret) |
| Cold-TEST-300 của 16 split mới | Replication theo tập item độc lập; mở một lần sau khi khóa DEV |

Split 137 chỉ được tái tạo trong bộ nhớ để chứng minh dữ liệu và thuật toán chia trùng với ADC (G1b, đã đạt); không train trên split 137.

## 2. Câu hỏi nghiên cứu

| RQ | Câu hỏi | Loại | Trả lời reviewer |
|---|---|---|---|
| RQ1 | Contamination (q = 0,3346) có làm tăng within-horizon regret trên các split/domain mới không? | P1 confirmatory | R2-1 |
| RQ2 | Phần tăng regret có vượt tác động của việc thay positive training không (P2); vai trò của kích hoạt item-ID (ước lượng) | P2 confirmatory + ước lượng | R2-4, R3-1, R3-2 |
| RQ3 | Kết luận có bền với trailing MA5 và horizon không; đỉnh nhọn sớm có lặp lại không | Ước lượng (secondary) | Mở rộng mới |
| RQ4 | Leakage phủ một phần item PV gây tổn hại bao nhiêu | Ước lượng | R2-3 |
| RQ5 | Tín hiệu phía development có báo trước regret không; fold PV có đại diện cho cold thật không | Mô tả | R3-3 |

## 3. Dataset và split

### 3.1 Domain (đã build và đóng băng)

| Tag | Category Amazon Reviews 2023 | Nguồn core | Users × items | Interactions | core_sha256 |
|---|---|---|---|---|---|
| VG | Video_Games | Core ADC (Colab) | 2.999 × 5.000 | 88.267 | 4eddd629f2c5f5d2911de80de64264781225fe200410c1387bb6df068b7f7198 |
| CSJ | Clothing_Shoes_and_Jewelry | Core ADC | 3.000 × 5.000 | 73.836 | 0b962c5db51584f5fb75353e4d3b32bb1a0406034472315024a18c515885d20e |
| SO | Sports_and_Outdoors | build_core.py (V1.3) | 3.000 × 5.000 | 57.541 | 25ea74552d1ba085bb714b3645c016fb5da6321512473a3caaf7bd27b99764ff |
| TG | Toys_and_Games | build_core.py (V1.3) | 3.000 × 5.000 | 64.802 | 0d63f1b91606c799ae21149771035a3054910a45485439c94249633f1b5c4535 |

Domain index cố định: VG = 1, CSJ = 2, SO = 3, TG = 4. Thứ tự chạy: VG → CSJ → SO → TG.

Quy tắc build (giống ADC, hiện thực lại): official 5-core rating_only; 2 × 5.000 item có nhiều interaction nhất; 3.000 user có nhiều interaction nhất trong nhóm đó (≥ 3); item ứng viên có ≥ 2 interaction với nhóm user; mỗi item một ảnh (variant MAIN trước; hi_res > large > thumb; RGB JPEG chất lượng 95); chọn lại user (≥ 3, tối đa 3.000), 5.000 item đầu (≥ 2), lọc user lần cuối. Features: ResNet-50 `ResNet50_Weights.DEFAULT` (= IMAGENET1K_V2), transform chuẩn của weights, bỏ fc, chuẩn hóa L2, 2.048 chiều.

Tiêu chí feasibility (F): đúng 5.000 item; 2.900 ≤ users ≤ 3.000; mọi item có ảnh; degree user tối thiểu ≥ 3; degree item tối thiểu ≥ 2. Cả 4 domain đạt.

Nhất quán features: trích lại 64 item mỗi domain trên máy người dùng cho cosine = 1,00000 (min, median, mean) ở cả 4 domain.

### 3.2 Seed và chia split

- data_seed = 237, 337, 437, 537 (mỗi seed là một split); split_seed = 7 cho mọi split; data_seed = 137 không dùng để train.
- Cold pool: `default_rng(data_seed).choice(5000, 1000, replace=False)`.
- Mỗi user, interaction warm sắp theo timestamp, hòa theo `parent_asin`: ≥ 3 warm → cuối là warm-test, kế cuối là warm-val, còn lại train; đúng 2 → một train, một warm-test; ≤ 1 → train.
- Ba fold phân tầng theo degree train (cùng RNG: `jitter = rng.random(#warm)`, `lexsort((jitter, degree))`, chia vòng tròn); fold 0 là PV.
- Cold DEV/TEST: `default_rng(7).permutation(cold đã sắp)`, 300 item đầu là TEST, 700 còn lại là DEV.
- Các split cùng domain dùng chung core; cold pool hai split trùng trung bình 196,5 item (183–210). Suy luận có điều kiện trên bốn core.

### 3.3 Vai trò holdout trong mỗi split

| Phần | Kích thước (khoảng qua 16 split) | Vai trò |
|---|---|---|
| Strict train | 26.280–43.312 cặp | Positive training |
| Pseudo-validation (PV), fold 0 | catalog 1.334 item; 13.238–21.792 cặp | Nguồn duy nhất để chọn checkpoint; là thứ bị leak |
| Cold-DEV | 700 item; 7.720–12.913 cặp | Endpoint confirmatory, tính mỗi epoch |
| Cold-TEST | 300 item | Niêm phong trong `sealed/test.npz`; không load khi training |
| Warm-val | 2.998–3.000 cặp | Chỉ dùng cho RQ5 |
| Warm-test | ~3.000 cặp | Không dùng (chỉ ghi hash) |

Artifact DEV (`dev.npz`) chỉ chứa: train_pairs, pv_pairs, pv_items, warm_items, negative_pool, warm_val_pairs, item_train_degree, cold_dev_items, cold_dev_pairs. Danh tính 1.000 item cold suy ra được từ phần bù của warm_items; cái được niêm phong là interaction của TEST.

Chance floor (G8, đã đạt): NDCG@10 kỳ vọng của xếp hạng ngẫu nhiên trên cold-DEV: VG 0,0100–0,0103; CSJ 0,0094–0,0096; SO 0,0087–0,0089; TG 0,0091–0,0092; Recall@10 = 10/700. Popularity trùng với random vì mọi item cold có degree train = 0.

## 4. Mô hình, arm và ngân sách

Backbone A (VBPR-style ID + visual hybrid; không gọi là "canonical VBPR"): user embedding 64 chiều; item = ID embedding × trained-ID mask + MLP(2048 → 128 → ReLU → 64) của visual; score = tích vô hướng. Init: user và ID theo N(0; 0,05), Linear theo mặc định PyTorch. BPR loss, Adam lr 1e-3, weight decay 1e-6, B = 2.048, 120 epoch. Mask ban đầu = item xuất hiện trong strict train.

Backbone B (visual-only): cùng code với A, mask = 0 cho mọi item suốt quá trình; score = user · MLP(visual).

| Arm | Backbone | Dose q (k slot/batch) | Nội dung k slot | ID của item PV | Tier |
|---|---|---|---|---|---|
| A0 CLEAN | A | 0 | — (master plan) | Tắt | 1 |
| A1 LEAKY | A | 0,10 (205) | Positive PV, rút có hoàn lại từ toàn bộ PV | Bật khi item vào batch | 1 |
| A2 LEAKY | A | 0,20 (410) | như A1 | như A1 | 1 |
| A3 LEAKY | A | 0,3346 (685) | như A1 | như A1 | 1 |
| A4 CR | A | 0,20 (410) | Positive strict train, RNG riêng | Tắt | 1 |
| A5 CR | A | 0,3346 (685) | như A4 | Tắt | 1 |
| A6 IDoff | A | 0,20 (410) | Positive PV như LEAKY | Luôn tắt | 1 |
| A7 IDoff | A | 0,3346 (685) | Positive PV như LEAKY | Luôn tắt | 1 |
| B0 CLEAN | B | 0 | — | Không có ID | 1 |
| B2 LEAKY | B | 0,20 (410) | Positive PV | Không có ID | 1 |
| B3 LEAKY | B | 0,3346 (685) | Positive PV | Không có ID | 1 |
| A8 PC10 | A | 0,20 (410) | Positive PV chỉ từ tập con 10% item PV | Bật khi vào batch | 2 |
| A9 PC30 | A | 0,20 (410) | tập con 30% (chứa tập 10%) | như A8 | 2 |
| A10 PC60 | A | 0,20 (410) | tập con 60% (chứa tập 30%) | như A8 | 2 |
| B1 LEAKY | B | 0,10 (205) | Positive PV | Không có ID | 2 |

### 4.1 Seed hiệu dụng và các luồng RNG (khóa)

- Seed hiệu dụng: `eff = 1.000.000 · domain_index + 1.000 · data_seed + r`, với r ∈ {42, 52, 62, 72, 82}. Ví dụ VG, split 237, r = 42 → eff = 1.237.042. Mọi luồng ngẫu nhiên của training suy ra từ eff, nên các split và các domain không dùng chung nhiễu theo seed.
- Salt: PLAN_SALT = 101, NEG_SALT = 202, CR_SALT = 303, PC_SALT = 404.
- Init: `torch.manual_seed(eff)` rồi dựng model; cùng một init state cho mọi arm, cả A và B, trong cùng (domain, split, r).
- Master plan, epoch e: `rng = default_rng([eff, PLAN_SALT, e])`; `order = rng.permutation(n_train)`; số batch = ceil(n_train / 2048); batch b lấy 2.048 phần tử liên tiếp của order, batch cuối được bù từ đầu order. Với mỗi batch, theo thứ tự: `slots = rng.choice(2048, 685, replace=False)`, `pv_fill = rng.integers(0, n_pv, 685)`. Mọi arm (kể cả CLEAN và B) dựng master plan với k_max = 685, nên master plan giống hệt nhau giữa các arm.
- Arm có liều k dùng tiền tố `slots[:k]`; LEAKY và IDoff điền `pv_fill[:k]`. Vị trí slot lồng nhau 205 ⊂ 410 ⊂ 685.
- CR, epoch e: `rng_cr = default_rng([eff, CR_SALT, e])`; mỗi batch rút 685 chỉ số có hoàn lại vào strict train; dùng tiền tố k.
- PC: tập con chọn một lần theo split, không phụ thuộc r: `default_rng([domain_index, data_seed, PC_SALT]).permutation(item PV có ≥ 1 cặp)`; 10% đầu ⊂ 30% ⊂ 60% (làm tròn round). Epoch e: `rng_pc = default_rng([eff, PC_SALT, e])`; mỗi batch rút 410 cặp đều (có hoàn lại) trong các cặp PV có item thuộc tập con.
- Negative khớp theo slot: epoch e, `rng_neg = default_rng([eff, NEG_SALT, e])` sinh ma trận ứng viên `cand[b, j, m]` (m = 1…16) rút đều từ negative pool, không phụ thuộc user. Negative của slot (b, j) là ứng viên đầu tiên không thuộc tập item strict-train của user ở slot đó. Nếu cả 16 ứng viên đều đã thấy: rút tiếp từ `default_rng([eff, NEG_SALT, e, b, j])` cho đến khi gặp item chưa thấy. Hệ quả: hai arm dùng cùng negative ở mọi slot có cùng user.
- Ghi mask ID: `mask[unique(item ids), 0] = 1` (G6).
- Tính tất định: `torch.backends.cudnn.deterministic = True`, `benchmark = False`, `torch.use_deterministic_algorithms(True)`, biến môi trường `CUBLAS_WORKSPACE_CONFIG=:4096:8` đặt trước khi khởi tạo CUDA.

### 4.2 Ngân sách

- 15 arm × 4 domain × 4 split × 5 seed = 1.200 trajectory; Tier 1 = 880, Tier 2 = 320.
- Thời gian mỗi trajectory đo ở pilot. Nếu vượt ngân sách, cắt theo thứ tự cố định: PC của TG → PC của SO → B1 → PC của CSJ. Chỉ cắt vì tài nguyên hoặc lỗi kỹ thuật đã định nghĩa, không dựa trên regret đã quan sát. Không cắt gì trong Tier 1.

## 5. Selection rule, horizon, logging và checkpoint

Argmax ở H = 120 là quy tắc quyết định cho P1/P2; trailing MA5 là rule thứ hai được khóa; các rule và horizon khác là sensitivity, tính lại từ log.

| Rule | Định nghĩa | Vai trò |
|---|---|---|
| R1 argmax | e* = argmax PV_e trên e = 1…120; hòa chọn epoch sớm nhất | Primary (P1, P2) |
| R2 trailing MA5 | s_e = trung bình PV từ epoch max(1, e−4) đến e; e* = argmax s_e; hòa chọn sớm nhất | Secondary, khóa trước |
| Sensitivity | Full-window MA5 và argmax chỉ từ epoch 5; trailing MA3, MA9; centered MA5; early stopping patience 10 | Mô tả |

- Mọi chỉ số chọn checkpoint dùng giá trị đã làm tròn 8 chữ số thập phân, đúng như ghi trong CSV; s_e tính trong float64 từ các giá trị đã làm tròn. Trainer và script phân tích dùng chung một hàm chọn.
- Horizon: mọi arm train đủ 120 epoch. Cắt ở H ∈ {40, 80} chỉ mô tả.
- Log mỗi epoch (CSV, 8 chữ số, flush và fsync): PV NDCG@10 và Recall@10; cold-DEV NDCG@10 và Recall@10; warm-val NDCG@10; PV-noID NDCG@10 (ID term tạm tắt rồi khôi phục; memorization gap = PV − PV-noID); số item PV đang ID-active; BPR loss trung bình; s_e của MA5; epoch đang được chọn theo R1 và R2.
- Log mỗi trajectory: số update, positive exposure, PV exposure, coverage cặp và item, SHA-256 của master plan và init state.
- Ghi file: trajectory ghi vào `.partial`, chỉ đổi tên khi đủ 120 epoch; chạy lại tự xóa `.partial` và ghi sự kiện vào RUN_EVENTS.log; trajectory hoàn tất không bao giờ bị ghi đè.

Checkpoint (không truy cập TEST khi training):

| Phạm vi | State được lưu | Dung lượng ước tính |
|---|---|---|
| Mọi trajectory (1.200) | Hai state đang tốt nhất theo R1 và R2 | ~7,6 GB |
| A0, A3, A5, A7 (320 trajectory) | Thêm lưới cố định epoch 1–20 và 30, 40, …, 120 (30 state) | ~30 GB |

State bỏ ma trận visual nhưng giữ trained_id_mask; ghi SHA-256 của ma trận visual và của từng state vào manifest.

Evaluation (ghi nguyên văn vào bài): xếp hạng trên toàn bộ catalog của holdout (PV ~1.334 item, cold-DEV 700, cold-TEST 300); không sampled negative; không lọc item đã thấy; chỉ tính user có ít nhất một positive trong holdout; NDCG@10 trung bình đều theo user; top-k bằng torch.topk. Warm-val xếp hạng trên warm items và loại item user đã có trong strict train ∪ PV.

## 6. Kế hoạch thống kê

Có đúng hai kiểm định confirmatory, P1 rồi P2, theo thứ tự cố định, trên 16 đơn vị split.

Endpoint: within-horizon regret tuyệt đối trên cold-DEV-700, rule R1, H = 120:

R = max_{e ∈ 1…120} C_e − C_{e*}, e* = argmax_e PV_e

| Contrast | So sánh | Câu hỏi |
|---|---|---|
| P1 (primary) | A3 LEAKY(0,3346) − A0 CLEAN | Contamination có làm tăng regret trên dữ liệu mới không? |
| P2 (key mechanistic) | A3 LEAKY(0,3346) − A5 CR(0,3346) | Phần tăng có vượt tác động của việc thay positive training không? |

Thủ tục (chung cho P1 và P2):
1. Đơn vị: với domain j và split s, d_{j,s} = trung bình qua 5 seed của (R^A − R^B).
2. Thống kê: T = trung bình của 16 giá trị d_{j,s} (mỗi domain 4 split, trọng số domain bằng nhau).
3. Kiểm định: sign-flip hai phía trên 16 giá trị, liệt kê đủ 2^16 = 65.536 mẫu dấu; p = tỷ lệ mẫu có |T*| ≥ |T|.
4. Xác nhận khi đủ ba điều kiện: p < 0,05; T > 0; ít nhất 3/4 domain có trung bình d_j > 0.
5. Fixed-sequence: P2 chỉ được kiểm định confirmatory khi P1 đã được xác nhận; mỗi bước α = 0,05. Nếu P1 không đạt, P2 chỉ báo cáo mô tả.
6. Luôn báo cáo: T; CI 95% bằng đảo ngược kiểm định sign-flip trên 16 split; trung bình từng domain cùng 4 giá trị split và CI t (df = 3) trên 4 giá trị split; số split và số cặp (split, seed) cùng chiều. Không dùng kết quả gộp dương để che một domain ngược chiều.

Phạm vi suy luận: có điều kiện trên bốn core và bốn split đã quan sát của mỗi domain; không suy rộng ra quần thể domain. Không kiểm định riêng từng domain.

### 6.1 Kiểm tra type-I (G7, trước run đầu tiên)

Mô hình sinh: d_{j,s} = u_j + w_j + v_{j,s} + ē_{j,s}, với ē là trung bình của 5 sai số theo seed (độ lệch chuẩn σ_e mỗi seed). Mỗi ô 10.000 lần lặp, seed mô phỏng cố định, mô phỏng toàn bộ quy tắc ở bước 4.

| Kịch bản | Tham số |
|---|---|
| Lưới chung | σ_e ∈ {0,001; 0,003; 0,005}; σ_v/σ_e ∈ {0; 0,5; 1; 2} |
| (a) chuẩn | u = 0, w = 0, v và e chuẩn |
| (b) lệch phải, đuôi nặng | e lognormal đã trừ trung bình, σ_ln ∈ {1,0; 1,5} |
| (c) u_j trái dấu, tổng bằng 0 | u = (+δ, −δ, 0, 0) và (+δ, +δ, −δ, −δ), δ ∈ {0,002; 0,005} |
| (d) tương quan trong domain | w_j ~ N(0, ρ·τ²), v ~ N(0, (1−ρ)·τ²), τ² = σ_v² + σ_e²/5; ρ = 0,2 là tiêu chí; ρ = 0,5 chỉ báo cáo sensitivity |

Tiêu chí: tỷ lệ "xác nhận" ≤ 0,0544 (= 0,05 + 2 SE Monte Carlo) ở mọi ô tiêu chí. Nếu thất bại: thay bước 3 bằng t-test một mẫu trên 16 giá trị split (df = 15) và mô phỏng lại. Nếu vẫn thất bại: P1/P2 chỉ báo cáo ước lượng, không có p-value, và ghi deviation. Kết quả mô phỏng được lưu cùng hash của script.

Tham khảo (mô phỏng sơ bộ ngày 28/10, không thay thế G7): (a) 0,023–0,026; (b) 0,001–0,023; (c) 0,005–0,011; (d) ρ = 0,2: 0,050; ρ = 0,5: 0,102.

### 6.2 Secondary (ước lượng, đơn vị split, không confirmatory)

| Phân tích | Contrast |
|---|---|
| Dose | A1, A2 − A0 |
| MA5 interaction (RQ3) | (A3 − A0) dưới R1 trừ (A3 − A0) dưới R2; tương tự cho A2 |
| Đỉnh nhọn sớm (RQ3) | Tỷ lệ trajectory có argmax PV ở epoch ≤ 10 và PV tại đó > max PV trên các epoch > 20, theo arm |
| ID (RQ2) | A3 − A7, A7 − A0; ở q = 0,20: A2 − A6, A6 − A0 |
| CR ở q = 0,20 (RQ2) | A2 − A4, A4 − A0 |
| Backbone B (RQ2) | B2, B3 − B0; B1 nếu có |
| Coverage (RQ4) | A8, A9, A10, A2 − A0 theo coverage 10/30/60/100%; ghi rõ số lần lặp mỗi cặp tăng khi coverage giảm ở liều cố định |
| Horizon | Regret ở H = 40, 80, 120 |
| Dị biệt giữa domain | Normalized regret R / max C_e; ρ Spearman(PV, cold-DEV) |
| RQ5 | Spearman giữa regret và PV inflation, memorization gap tại e*, lệch epoch chọn bởi PV so với warm-val; so sánh fold PV với cold-DEV-700 (KS cho degree toàn dữ liệu, total variation cho category, MMD có permutation cho visual); so sánh với cả pool 1.000 item chỉ làm cùng lúc mở TEST |

Khóa diễn giải:
- Các contrast cơ chế được báo cáo trực tiếp (R_LEAKY − R_CR, R_LEAKY − R_IDoff) kèm CI cấp split. Không tuyên bố tương đương.
- A_ID = 1 − Δ_IDoff / Δ_LEAKY chỉ tính ở domain có CI t (df = 3) của Δ_LEAKY nằm hoàn toàn trên 0; ngược lại ghi "không xác định". A_ID > 0,5 được mô tả là "substantial attenuation under ID disabling", không quy kết nhân quả.
- Failure chỉ xuất hiện dưới argmax, hoặc không tái lập ở một domain, được viết đúng như vậy.

## 7. Cold-TEST-300

TEST được mở đúng một lần, sau khi script phân tích DEV đã chạy và kết quả DEV đã được khóa bằng hash; chỉ inference trên các state đã lưu.

1. Trong lúc training, không code nào mở thư mục `sealed/`; loader DEV từ chối đường dẫn chứa `sealed`.
2. Khóa DEV: chạy script phân tích, ghi SHA-256 của bảng kết quả vào DEV_RESULTS_LOCK.json.
3. Dry-run: nạp lại mọi state đã lưu, chấm PV và cold-DEV, đối chiếu log (sai lệch ≤ 1e-6), kiểm hash.
4. Mở TEST một lần cho cả 16 split; ghi COLD_TEST_OPENED.json; evaluator từ chối chạy lần hai.

| Chỉ số | Arm | Định nghĩa |
|---|---|---|
| Selected TEST NDCG@10 | Cả 15 arm | NDCG@10 trên TEST tại e* của R1 và của R2 |
| Grid-restricted TEST regret | A0, A3, A5, A7 | max trên G ∪ {e*} của TEST NDCG − TEST NDCG tại e*, G = {1, …, 20} ∪ {30, 40, …, 120} |
| ρ trên lưới | A0, A3, A5, A7 | Spearman(PV, TEST NDCG) trên 30 epoch của G |

Chỉ số thứ hai là cận dưới của regret đầy đủ trên TEST, không gọi là full-horizon regret. P1 và P2 được tính lại trên grid-restricted TEST regret bằng cùng thủ tục ở mục 6, ghi nhãn "replication trên tập item độc lập"; kết quả này không thay đổi kết luận confirmatory của DEV.

## 8. Validity gate

Mỗi gate là một assert cứng trong code. Gate fail thì dừng, sửa, ghi deviation, và không nhìn contrast nào.

| Gate | Thời điểm | Kiểm tra | Trạng thái |
|---|---|---|---|
| G0 Build | Sau khi dựng từng split | Core giống nhau giữa 4 split cùng domain; DEV và TEST rời nhau theo item; overlap train/PV = 0; negative pool sạch; hợp các phần của split = toàn bộ core; artifact DEV riêng | Đạt 16/16 (28/10/2026) |
| G1b ADC data | Trước pilot | Split 137 tái tạo trong bộ nhớ khớp ADC: CSJ 10/10 fingerprint + 6 số đếm; VG 6 số đếm + tập cold-DEV + hash features | Đạt (28/10/2026) |
| G8 Chance floor | Sau G0 | Random và popularity (random tie-break) trên cold-DEV của từng split; Monte Carlo khớp giá trị chính xác | Đạt 16/16 (28/10/2026) |
| G1 Determinism | Pilot | Cùng một trajectory chạy hai lần cho log trùng khớp ở 8 chữ số | |
| G2 Plan invariance | Mọi run | Hash master plan và init state giống nhau giữa mọi arm cùng (domain, split, r); CR và PC không đổi master plan hay vị trí slot | |
| G3 Exposure | Mọi run | Đúng k slot mỗi batch; số update và số positive slot bằng nhau giữa các arm; PC chỉ leak từ tập con; CR không chứa cặp PV nào | |
| G4 Cleanliness | Mọi run | CLEAN, CR: overlap train/PV = 0 cặp và 0 item; negative pool không chứa item PV hay cold | |
| G5 ID mechanics | Mọi run | A0, A4, A5, A6, A7: mask ID của mọi item PV = 0 ở mọi epoch; backbone B: mask = 0 cho mọi item | |
| G6 Mask write | Code review | Ghi mask qua unique và chỉ số 2 chiều | |
| G7 Analysis lock | Trước run đầu tiên | Script phân tích đúng trên trajectory giả lập có hiệu ứng biết trước; mô phỏng type-I đạt mục 6.1; hash script và kết quả | |
| G9 Checkpoint | Mọi run | State nạp lại khớp log (≤ 1e-6); đủ 30 state lưới cho A0/A3/A5/A7; SHA-256 trong manifest | |

Pilot: VG, data_seed 237, r = 42, đủ 15 arm, cộng một lần chạy lặp A3 cho G1. Chỉ đọc output của gate và thời gian chạy, không tính contrast. Nếu mọi gate qua, dữ liệu pilot được giữ làm một phần của bộ dữ liệu chính.

## 9. Hướng dẫn triển khai

Toàn bộ code được viết mới trong `D:\CheckPoint\journal_v1\code\`; không import, copy hay chạy code hoặc file nào của ADC. ADC chỉ là tài liệu tham khảo: đặc tả thuật toán và các con số đã công bố để đối chiếu (G1b).

Cấu trúc:
- `D:\CheckPoint\CANTOR_online\<Category>\`: dữ liệu nguồn, chỉ đọc.
- `journal_v1\code\`: code; `journal_v1\data\<Category>\core\` và `split_<data_seed>\` (kèm `sealed\`); `runs\`, `states\`, `analysis\`, `logs\`.
- `journal_v1\PROTOCOL.md`, `PROTOCOL_HASH.txt`, `DEVIATIONS.md`.

Code giai đoạn data (đã chạy): constants.py, common.py, core_check.py, splitlib.py, make_splits.py, data_dev.py, chance_floor.py, resnet_features.py, build_core.py, feature_consistency.py.

Thứ tự thực hiện:
1. Đã xong: dựng 4 domain × 4 split; G0, G1b, G8 đạt; nhất quán features.
2. Viết hằng số thí nghiệm, module model/sampler/metrics/trainer, script phân tích và script mô phỏng type-I; qua G6 và G7; hash code và kết quả mô phỏng.
3. Phân tích fold PV so với cold-DEV-700 (RQ5, không cần train).
4. Pilot (mục 8); qua G1–G5, G9; cập nhật ngân sách.
5. Tier 1 theo thứ tự VG → CSJ → SO → TG.
6. Tier 2 theo cùng thứ tự; cắt giảm theo mục 4.2 nếu cần.
7. Chạy script phân tích DEV đã khóa một lần; ghi DEV_RESULTS_LOCK.json.
8. Mở cold-TEST theo mục 7.
9. Mọi phân tích thêm sau bước 7 được dán nhãn exploratory.

Deviation: mọi thay đổi sau khi hash (sửa lỗi, cắt giảm compute, đổi thủ tục thống kê do mô phỏng, lỗi kỹ thuật) ghi vào DEVIATIONS.md gồm ngày, lý do, dữ liệu nào đã được nhìn trước khi đổi, arm bị ảnh hưởng. Bài báo công bố đủ danh sách này.

## 10. Khóa phiên bản

- File này được khóa bằng SHA-256 ghi trong PROTOCOL_HASH.txt. Kiểm tra trên Windows: `certutil -hashfile PROTOCOL.md SHA256`.
- Không mở file này bằng trình soạn thảo rồi lưu lại; thay đổi dù một ký tự cũng làm đổi hash. Mọi thay đổi sau khóa đi qua DEVIATIONS.md, hoặc một phiên bản protocol mới có hash riêng.
- Tại thời điểm khóa đã được nhìn: kết quả và code ADC (prior study); số đếm, overlap, chance floor và nhất quán features của 16 split mới. Chưa có model nào được train trên các split mới.

## 11. Khác biệt so với V1.2

| # | V1.2 | V1.3 | Lý do |
|---|---|---|---|
| 1 | Copy module ADC, sửa trong `run_unified_trajectory` | Viết mới toàn bộ, ADC chỉ để tham khảo | Quyết định của người dùng |
| 2 | 3 domain (domain 3 = Home_and_Kitchen, fallback Toys_and_Games) | 4 domain: VG, CSJ, SO, TG; không fallback | Quyết định của người dùng, trước khi tải dữ liệu SO/TG; Home_and_Kitchen không được tải |
| 3 | 12 split, 4.096 mẫu dấu, ≥ 2/3 domain | 16 split, 65.536 mẫu dấu, ≥ 3/4 domain | Hệ quả của mục 2; giữ nghĩa "đa số domain" |
| 4 | 900 trajectory; cắt PC domain 3 → B1 → PC CSJ | 1.200 trajectory; cắt PC TG → PC SO → B1 → PC CSJ | Hệ quả của mục 2 |
| 5 | Training seed 42…82 dùng chung mọi split | Seed hiệu dụng theo (domain, split, r) | Seed chung tạo tương quan giữa các split, vi phạm giả định độc lập của sign-flip |
| 6 | Sampler negative của ADC (rút lại theo user) | Negative khớp theo slot | Contrast giữa arm chỉ khác ở slot bị thay |
| 7 | Kịch bản (d) ICC 0,5 là tiêu chí | ρ = 0,2 là tiêu chí; ρ = 0,5 chỉ sensitivity | Với ICC 0,5 không kiểm định cấp split nào giữ được type-I khi chỉ có vài domain; phạm vi suy luận vốn có điều kiện trên các core |
| 8 | CI cấp domain bằng sign-flip | CI t (df = 3) trên 4 giá trị split | Sign-flip trên 4 split không cho được CI 95% |
| 9 | Tiêu chí build "3.000 users" | 2.900–3.000 users, đúng 5.000 item, mọi item có ảnh | VG có 2.999 user |
| 10 | RQ5 so fold PV với cold pool | So với cold-DEV-700; pool 1.000 khi mở TEST | Không đọc artifact TEST trước khi mở |
| 11 | G1b: chạy lại code ADC | G1b: tái tạo split 137 so với fingerprint ADC (đã đạt) | Hệ quả của mục 1 |
| 12 | Không quy định | Tính tất định bằng deterministic algorithms; chọn checkpoint trên giá trị làm tròn 8 chữ số | Bảo đảm G1 và khớp giữa state đã lưu với phân tích |
