# TEST ANALYSIS PLAN — Journal V1.3, cold-TEST-300 (Bước B v4, freeze trước mọi lần đọc sealed)

Ngày soạn: 30/10/2026. Bản v2 sửa theo STEP_B_REVIEW_VERDICT_20261030 (B1–B5); bản v3 sửa theo STEP_B_V2_REVIEW_VERDICT_20261030 (F1–F4, câu chữ về tập ứng viên); bản v4 sửa theo STEP_B_V3_REVIEW_VERDICT_20261030 (F5: định danh checkpoint).
Tài liệu này được freeze cùng `analysis_test.py` và `test_schema.py` qua manifest độc lập `TEST_PLAN_FREEZE.json` (§8).

**Nhãn bắt buộc:** *precommitted sealed-TEST replication informed by DEV*.
- Kế hoạch được viết **sau** khi đã thấy kết quả DEV.
- Đây là kiểm chứng ngoài mẫu, cam kết trước khi xem TEST; không phải giả thuyết độc lập với DEV.
- Kết quả TEST **không thay đổi** kết luận confirmatory đã khóa của DEV (Protocol §7).

## 0. Mốc và ràng buộc

- **DEV_RESULTS_LOCK.json** `0775d9fb819ce0f0bf85333a3fc801d1cd0265c344ae47457384066cd980ab3b`.
  - P1 trên DEV: T = 0,003766, p = 0,00140.
  - P2 trên DEV: T = 0,003364, p = 0,00989.
  - Cả hai đã xác nhận.
- **PROTOCOL.md** `c5c2913b…`; **DEVIATIONS.md** `e6ad3dd4…`.
- **POST_LOCK_ADDENDUM.md** = PL1 (bản `d26b7e91…`, đã dùng cho RQ5) + PL2 (freeze kế hoạch TEST). SHA của bản cuối **không ghi trong văn bản này** mà được ghim trong `TEST_PLAN_FREEZE.json`, để không có phụ thuộc vòng (F2).
- **Protocol §7**, nguyên văn:
  > Grid-restricted TEST regret | A0, A3, A5, A7 | max trên G ∪ {e*} của TEST NDCG − TEST NDCG tại e*, G = {1, …, 20} ∪ {30, 40, …, 120}
  >
  > Chỉ số thứ hai là cận dưới của regret đầy đủ trên TEST, không gọi là full-horizon regret. P1 và P2 được tính lại trên grid-restricted TEST regret bằng cùng thủ tục ở mục 6, ghi nhãn "replication trên tập item độc lập"; kết quả này không thay đổi kết luận confirmatory của DEV.
- **Lộ trình:**
  - Bước B (văn bản này): không đọc dữ liệu nào.
  - Bước C (dry-run DEV): phải được duyệt và đạt.
  - Bước D (mở TEST): cần reviewer duyệt riêng; source của bộ chấm TEST cũng phải được freeze trước khi chạy thật (§8).

## 1. Dữ liệu và giao diện (khóa trong `test_schema.py`)

| File | Ghi bởi | Nội dung |
|---|---|---|
| `analysis\dev_final\{per_trajectory.csv, units.csv, results.json}` | DEV (đã khóa) | e\* của R1 và R2 ở H = 120, `selected_cold`, regret DEV đầy đủ. Cả 3 file được kiểm SHA với DEV lock; lock được kiểm SHA nguyên file |
| `journal_v1\TEST_PLAN_FREEZE.json` | B (freeze) | SHA của 3 file freeze, DEV lock, addendum; `created_utc` (§8) |
| `analysis\test_dryrun\dryrun_scores.csv`, `TEST_DRYRUN_REPORT.json` | C | 12.000 dòng logic. Mỗi dòng gồm khóa (domain, data_seed, arm, r, state, epoch, state_sha256, weights_sha256), **6 metric** PV/DEV/warm-val/PV-noID chấm lại, và `max_abs_diff_vs_log` = sai lệch lớn nhất của 6 metric so với dòng log CSV cùng epoch. Phải hữu hạn, nằm trong [0, 1e-6]. Áp dụng cho **mọi** trạng thái logic có log, không chỉ e\* |
| `analysis\test_open\receipts\01_PREPARED.json` | D | Bản ghi đầu chuỗi, bất biến |
| `analysis\test_open\COLD_TEST_OPENED.json` | D | Marker O_EXCL, là bản ghi OPENED, tạo trước lần đọc sealed đầu tiên |
| `analysis\test_open\receipts\02r_RESUMED_<k>.json` | D | Chỉ có khi reviewer cấp `RESUME_APPROVAL_<k>.json` |
| `analysis\test_open\receipts\03_SCORES_COMPLETE.json` | D | SHA của `test_scores.csv`, số user TEST mỗi split, `n_rows` = 12.000 |
| `analysis\test_open\test_scores.csv` | D | 12.000 dòng logic: khóa (gồm `state_sha256`, `weights_sha256`), `test_ndcg10`, `test_recall10`, `test_n_users` |
| `analysis\test_open\receipts\04_ANALYSIS_COMPLETE.json` | `analysis_test.py --final` (hoặc `--finalize-receipt` sau crash, có phê duyệt) | SHA của 3 output, SHA code phân tích, SHA freeze, liên kết tới 03 |
| `analysis\test_open\receipts\05_LOCKED.json` | `analysis_test.py --lock` | SHA của 3 output, SHA của 04, liên kết tới 04 |
| `analysis\test_final\{test_per_trajectory.csv, test_units.csv, test_results.json}` | `analysis_test.py --final --freeze-sha …` | Chạy một lần; từ chối nếu thư mục không rỗng |

**Định danh checkpoint (F5).**
- `state_sha256` = SHA-256 của **byte file** checkpoint. Phải bằng `sha256` ghi cho file đó trong manifest trajectory; bước C kiểm điều này.
- `weights_sha256` = `test_schema.weights_fingerprint(state_dict)`: fingerprint chuẩn hóa của trọng số, độc lập với cách tuần tự hóa. Cách tính: các entry sắp theo tên; mỗi entry băm `tên|dtype|shape` rồi băm byte C-order little-endian.
- `R1.pt`, `R2.pt` và `eNNN.pt` cùng epoch là các file **khác byte**, vì object đã lưu chứa trường `kind` riêng. Vì vậy định danh "cùng checkpoint" là **`weights_sha256` bằng nhau**, kèm cùng điểm trên mọi metric.
- Bước C và D tính `weights_sha256` từ chính state đã nạp để chấm. Cùng một dòng logic phải có cùng `state_sha256` và `weights_sha256` ở dry-run và ở TEST.

**Trạng thái logic:**
- R1 và R2 của mọi trajectory: 1.200 × 2 = 2.400 dòng.
- 30 trạng thái lưới `e001…e020, e030…e120` của A0, A3, A5, A7: 320 × 30 = 9.600 dòng.
- Tổng **12.000** dòng. Mỗi dòng logic được chấm riêng, kể cả khi hai file trùng epoch.

**Quy ước chấm TEST** (bước D phải hiện thực đúng như vậy; giống hệt cách chấm cold-DEV ở Protocol §5):
- xếp hạng trên **full cold-TEST holdout catalog (300 items)** của split. Đây đúng là quy ước của `metrics.py` đã dùng cho DEV: PV xếp trên catalog PV (~1.334 item), cold-DEV trên 700 item của nó. Không xếp trên 5.000 item nguồn, và không lấy mẫu thêm trong 300 item;
- không dùng sampled negative, không lọc item đã thấy;
- chỉ tính user có ít nhất một positive trong TEST;
- NDCG@10 trung bình đều theo user; top-k bằng `torch.topk`;
- chỉ inference trên state đã lưu.

`test_n_users` phải > 0 và **bất biến** trong mọi trạng thái logic của cùng một (domain, data_seed). Bước D đối chiếu nó với manifest TEST đã niêm phong rồi ghi vào `03_SCORES_COMPLETE.test_users_by_split`.

## 2. Endpoint chính và thủ tục (confirmatory-replication)

**Endpoint**, cho mỗi trajectory của A0, A3, A5, A7, với e\* = R1 ở H = 120 lấy từ DEV đã khóa:

GR = max_{e ∈ G ∪ {e\*}} TEST(e) − TEST(e\*), với G = {1..20} ∪ {30, 40, …, 120}.

- Tập ứng viên chứa e\*, nên **GR ≥ 0** (gate dừng nếu âm).
- GR của **một arm** là cận dưới của regret TEST đầy đủ của arm đó.
- **Hiệu hai GR** (P1, P2) **không nhất thiết** là cận dưới của hiệu hai regret đầy đủ. Bản thảo phải phân biệt hai khái niệm này.
- Khi hòa, oracle trên lưới lấy epoch sớm nhất. Quy tắc này chỉ ảnh hưởng `test_grid_oracle_epoch`.

**Thủ tục:** giống hệt Protocol §6, dùng `stats_confirm.py` đã khóa G7 và test đã khóa (`signflip`; kiểm qua `analysis_dev.locked_test()` và đối chiếu với `results.json` của DEV).
1. Đơn vị d_{j,s} = trung bình qua 5 seed của (GR^A − GR^B); 16 đơn vị theo thứ tự đã khóa.
2. T = trung bình 16 đơn vị. Sign-flip hai phía, liệt kê đủ 2^16 mẫu.
3. **Replicated** khi đủ ba điều kiện: p < 0,05; T > 0; ít nhất 3/4 domain có trung bình d_j > 0.
4. **Fixed sequence:** P1 = A3 − A0 trước. P2 = A3 − A5 chỉ được đánh giá nếu P1 replicated; nếu không, P2 chỉ mô tả (`replicated = null`).
5. Luôn báo cáo:
   - T; CI 95% đảo ngược sign-flip;
   - từng domain: trung bình, 4 giá trị split, CI t (df = 3);
   - số split dương; số cặp (split, seed) dương trên 80.

**Cách diễn đạt kết quả** (chốt trước):

| Kết quả | Cách viết |
|---|---|
| P1 replicated, P2 replicated | "Hiệu ứng lặp lại trên tập item TEST độc lập theo chỉ số grid-restricted" |
| P1 replicated, P2 không | "Tác hại so với CLEAN lặp lại; phần vượt đối chứng CR không lặp lại ở α = 0,05 trên TEST" |
| P1 không replicated, T > 0 | "Cùng chiều nhưng không đạt tiêu chí lặp lại; DEV vẫn là kết luận confirmatory duy nhất" |
| T ≤ 0 | "Không lặp lại / ngược chiều trên TEST"; báo cáo nổi bật, không giấu |

## 3. Secondary (ước lượng, không confirmatory)

| Mục | Định nghĩa |
|---|---|
| S1 | P1 và P2 trên GR với e\* = R2 (trailing MA5) |
| S2 | Selected TEST NDCG@10 và Recall@10 tại e\* của R1 và R2, cho 15 arm: trung bình theo arm và theo domain |
| S3 | ρ Spearman(PV, TEST NDCG) trên 30 epoch lưới, cho A0/A3/A5/A7. PV lấy từ dry-run (đã khớp log ≤ 1e-6) |
| S4 | Mô tả theo arm: GR (R1, R2), DEV grid-restricted regret, DEV full regret |

## 4. Exploratory trên TEST — 21 contrast, một họ Holm, cam kết trước khi xem TEST

- Mỗi mục tính trên 16 đơn vị bằng cùng thủ tục; gắn nhãn *exploratory (precommitted before TEST)*.
- **Dấu kỳ vọng** = dấu của ước lượng điểm DEV cho contrast tương ứng. Dấu này chỉ để đối chiếu hướng: kiểm định vẫn **hai phía**; không phải kiểm định một phía; không biến các mục này thành confirmatory.
- Báo cáo `direction_as_expected`, p chưa hiệu chỉnh, và **p Holm trong đúng 21 contrast này**.
- Có thể đưa toàn bộ 21 contrast vào phần supplementary để tránh diễn giải chọn lọc. Không thêm mục mới sau khi xem TEST.

| Mã | Contrast trên TEST | Dấu kỳ vọng | Giá trị tương ứng trên DEV |
|---|---|---|---|
| E1a | NDCG được chọn: A0 − A3 (R1) | + | +0,00577 |
| E1b | Trần trên lưới max_{G ∪ {e\*}}: A0 − A3 (R1) | + | +0,00201 (trần đầy đủ) |
| E1c | NDCG được chọn: A5 − A3 (R1) | + | +0,00544 |
| E1d | Trần trên lưới: A5 − A3 (R1) | + | +0,00208 |
| E2a | (GR_R1 − GR_R2) của A3 − A0 | + | +0,00219 |
| E2b | (GR_R1 − GR_R2) của A3 − A5 | + | +0,00206 |
| E3a / E3b | GR_R1 / GR_R2: A3 − A7 | + / + | +0,00264 / +0,00028 |
| E3c / E3d | GR_R1 / GR_R2: A7 − A0 | + / + | +0,00113 / +0,00130 |
| E4a–d | NDCG được chọn, R1: A0 − A8, A0 − A9, A0 − A10, A0 − A2 | + + + + | +0,00357, +0,00216, +0,00248, +0,00172 |
| E4e–h | NDCG được chọn, R2: A0 − A8, A0 − A9, A0 − A10, A0 − A2 | + + + + | +0,00080, +0,00100, +0,00174, +0,00133 |
| E4i / E4j | NDCG được chọn: A2 − A8 dưới R1 / dưới R2 | + / − | +0,00184 / −0,00053 |
| E4k | Tương tác coverage × rule: (A2 − A8)_R1 − (A2 − A8)_R2 | + | +0,00238 |

**Cách đọc:**
- E1: đồng nhất thức số học, không phải phân rã nhân quả. NDCG được chọn bị mất = phần trần lưới bị mất + phần GR tăng thêm.
- E2, E3: phần riêng của ID ở liều cao có chỉ hiện ra dưới argmax hay không.
- E4: đảo chiều coverage × rule, đo trên NDCG được chọn. A8–A10 không có state lưới nên không tính GR cho các arm này.

## 5. Hiệu chuẩn lưới trên DEV (E5) — báo cáo riêng, không phải TEST

| Mã | Contrast | So sánh với |
|---|---|---|
| E5a | P1 trên **DEV** grid-restricted regret (A3 − A0, R1), từ điểm DEV chấm lại ở bước C | P1 trên DEV regret đầy đủ (+0,00377): báo cáo T của cả hai và tỷ số lưới/đầy đủ |
| E5b | Như E5a cho P2 (A3 − A5) | +0,00336 |

- Nhãn: *DEV-only grid calibration, exploratory after the DEV lock; not TEST*.
- Chỉ dùng DEV. **Không nằm trong họ Holm của TEST** và không được gọi là replication.
- Mục đích: lượng hóa việc lưới thưa làm hiệu ứng co lại bao nhiêu, để đọc mức GR trên TEST.

**Phần RQ5 làm cùng lúc mở TEST** (PL1, DEVIATIONS A5): script riêng ở bước D, cùng quy ước PL1, kết quả là mô tả.
- Degree trên toàn core, PV so với cold-DEV.
- PV so với pool 1.000 item và PV so với TEST-300.

## 6. Gate và quy tắc xử lý (đều đã hiện thực và kiểm thử)

**`run_analysis` dừng** nếu:
- header sai; dòng thiếu / thừa / trùng (phải đúng 12.000 khóa);
- metric không hữu hạn hoặc ngoài [0, 1];
- `max_abs_diff_vs_log` không hữu hạn, âm hoặc > 1e-6;
- `test_n_users` ≤ 0, hoặc thay đổi trong cùng một split;
- state lưới có epoch khác tên của nó;
- `state_sha256`, `weights_sha256` hoặc epoch khác nhau giữa dry-run và TEST; `weights_sha256` không phải 64 ký tự hex;
- epoch của state R1/R2 khác e\* đã khóa;
- NDCG DEV chấm lại lệch `selected_cold` đã khóa > 1e-6;
- hai dòng logic cùng trọng số (R1 hoặc R2 so với state lưới cùng epoch; R1 so với R2 khi cùng epoch) có `weights_sha256` khác nhau (F5: không phải cùng checkpoint), hoặc lệch > 1e-9 ở **bất kỳ** metric nào: 2 metric TEST (NDCG, Recall) và 6 metric dry-run (F3);
- GR âm;
- `per_trajectory` thiếu R1 hoặc R2;
- thư mục output không rỗng.

**`--final --freeze-sha <sha>` còn yêu cầu:**
- manifest freeze hợp lệ, kể cả SHA thực tế của `POST_LOCK_ADDENDUM.md` khớp manifest (§8, F1);
- chuỗi receipt bất biến PREPARED → OPENED → (RESUMED…) → SCORES_COMPLETE, chưa có ANALYSIS_COMPLETE hay LOCKED (§8);
- DEV lock = `0775d9fb…`, và cả 3 file DEV khớp lock;
- manifest và PREPARED ghim đúng DEV lock;
- `dryrun_scores.csv` và `TEST_DRYRUN_REPORT.json` khớp PREPARED;
- `test_scores.csv` khớp SCORES_COMPLETE; `n_rows` = 12.000;
- số user mỗi split khớp `test_users_by_split`;
- các file G7 không đổi; test đã khóa trùng test đã dùng cho DEV.

`--final` chỉ đọc trong `journal_v1\analysis\`, `TEST_ANALYSIS_PLAN.md`, `TEST_PLAN_FREEZE.json`, `POST_LOCK_AUDIT_LOG.md` và code. Audit hook trong selftest xác nhận nó không mở gì trong `data\` hay `sealed`.

**Không có xử lý ngoại lệ linh hoạt:**
- Không loại trajectory, không đổi rule, lưới hay đơn vị.
- Lỗi nào cũng dẫn tới dừng hẳn và chuyển reviewer.

## 7. Báo cáo bắt buộc

- Báo cáo tất cả mục ở §2 đến §5, kể cả kết quả âm; theo từng domain và gộp.
- Không chọn rule, horizon hay arm theo TEST; không dùng TEST để chọn mô hình.
- Mỗi bảng dẫn `opening_id`, SHA của manifest freeze và DEV lock.

## 8. Freeze, receipt và crash (B1, B5)

**Manifest freeze độc lập (B1).** `journal_v1\TEST_PLAN_FREEZE.json` chứa:
- `manifest_type`, `protocol_version`, `created_utc`;
- SHA-256 của `analysis_test.py`, `test_schema.py`, `TEST_ANALYSIS_PLAN.md`;
- SHA của DEV lock và của POST_LOCK_ADDENDUM.md.

SHA của chính manifest được giữ ở **ba nơi độc lập**, không suy ra từ nhau lúc chạy:
1. mục L8 trong `POST_LOCK_AUDIT_LOG.md` (append-only), ghi cùng phê duyệt của reviewer **trước bước D**;
2. tham số tay `--freeze-sha` do người chạy nhập từ giá trị đã duyệt;
3. trường `freeze_manifest_sha256` của `01_PREPARED.json`.

`--final` kiểm cả ba, kiểm `created_utc` của manifest **sớm hơn** PREPARED, và kiểm 3 file hiện tại khớp manifest. Bộ chấm TEST của bước D (`open_test.py` và các module nó dùng) cũng phải được freeze trước khi chạy thật. SHA của chúng ghi trong `01_PREPARED.scorer_code_sha256`, và được duyệt ở gate bước D.

**Chuỗi receipt bất biến (B5).**
- Mỗi bản ghi là một file riêng, tạo bằng O_EXCL và **không bao giờ ghi đè**.
- Trường chung: `opening_id`, `state`, `utc`, `prev_sha256`. Trong đó `prev_sha256` là SHA của **file** bản ghi trước; PREPARED có `prev_sha256` = null.
- Trường bắt buộc theo từng loại bản ghi:
  - **PREPARED:** `freeze_manifest_sha256`, `dev_results_lock_sha256`, `dryrun_scores_sha256`, `dryrun_report_sha256`, `expected_sealed_sha256` (16 file sealed), `scorer_code_sha256`.
  - **OPENED** (= marker `COLD_TEST_OPENED.json`): tạo bằng O_EXCL **trước** lần đọc sealed đầu tiên; nếu đã tồn tại thì bước D dừng vĩnh viễn.
  - **SCORES_COMPLETE:** `test_scores_sha256`, `test_users_by_split`, `n_rows`.
- `--final` kiểm: đúng thứ tự trạng thái; liên kết `prev_sha256`; cùng `opening_id`; thời gian tăng dần.

**Crash (B5).**
- Nếu crash: dừng; giữ marker, log và kết quả từng phần kèm SHA; không xóa marker; không mở lại như một lượt mới; không chọn lại checkpoint.
- Crash trong lúc chấm TEST (bước D): chỉ phục hồi khi reviewer cấp `RESUME_APPROVAL_<k>.json` (có `opening_id`). Khi đó bước D ghi `02r_RESUMED_<k>.json`, chứa `resume_approval_file`, `resume_approval_sha256` và `reason`, được liên kết vào chuỗi.
- Phục hồi dùng đúng code, kế hoạch, state và quy tắc chọn đã freeze.
- Báo cáo công khai rằng có lượt phục hồi có kiểm soát; không mô tả như thể dữ liệu chỉ được đọc vật lý một lần.
- `--final` chỉ chạy khi chuỗi kết thúc ở SCORES_COMPLETE hợp lệ. Mỗi bản ghi RESUMED phải trỏ tới file phê duyệt còn nguyên SHA và đúng `opening_id`.

**Ai ghi receipt cuối, và crash giữa output với receipt (F4).**
- `analysis_test.py --final`:
  - kiểm chuỗi tới SCORES_COMPLETE;
  - ghi 3 output vào `analysis\test_final\` (thư mục phải rỗng);
  - tính SHA output;
  - ghi `04_ANALYSIS_COMPLETE.json` gồm `prev_sha256` = SHA của 03, `output_sha256`, `analysis_code_sha256`, `freeze_manifest_sha256`, `recovered_with_approval` = null.
- `analysis_test.py --lock`:
  - kiểm lại manifest và chuỗi tới ANALYSIS_COMPLETE;
  - kiểm DEV lock; SHA code phân tích không đổi; output trùng `04.output_sha256` (đúng 3 file);
  - ghi `05_LOCKED.json` (liên kết tới 04) và in `TEST RESULTS LOCKED`.
- Mọi bản ghi tạo qua `test_schema.create_record`:
  - ghi file tạm bằng O_EXCL và fsync;
  - hard-link sang tên cuối (thất bại nếu tên đã tồn tại, không bao giờ ghi đè);
  - xóa file tạm.
  - Nhờ vậy bản ghi hoặc xuất hiện đầy đủ, hoặc không xuất hiện. Nếu còn sót file `*.tmp` trong `receipts\` thì mọi bước dừng.
- Output **tất định**: `test_results.json` không chứa timestamp (thời điểm nằm trong receipt), nên tính lại cho ra cùng từng byte.
- Nếu crash **sau khi ghi output nhưng trước khi có 04**:
  - `--final` từ chối, vì output không rỗng; `--lock` từ chối, vì thiếu 04.
  - Dừng và giữ nguyên mọi thứ. Reviewer cấp `analysis\test_open\<file>.json` với `opening_id` và `"stage": "ANALYSIS_RECEIPT"`.
  - `analysis_test.py --finalize-receipt --approval <file> --freeze-sha <sha>`: tính lại vào `analysis\test_final_recheck\`, yêu cầu 3 file **trùng từng byte** với output hiện có, rồi mới ghi 04 với `recovered_with_approval` = {file, SHA}. Nếu khác thì dừng.
- Crash khi đang ghi output (thiếu file) hoặc khi ghi bản ghi: dừng, chuyển reviewer; không có đường phục hồi tự động.

**Thứ tự freeze không vòng (F2).**
1. Chốt nội dung `analysis_test.py` và `test_schema.py` (SHA ở bảng dưới).
2. Chốt văn bản này. Nó ghi SHA code, không ghi SHA addendum hay manifest.
3. Thêm PL2 vào cuối `POST_LOCK_ADDENDUM.md` (giữ nguyên từng byte phần PL1). PL2 ghi SHA code và kế hoạch, **không** ghi SHA manifest.
4. Tạo `TEST_PLAN_FREEZE.json`, ghim SHA 3 file và SHA addendum bản cuối.
5. Reviewer duyệt SHA manifest.
6. Chỉ khi đó mới thêm L8 vào `POST_LOCK_AUDIT_LOG.md`, ghi SHA manifest đã duyệt.

Sau bước 5, manifest và mọi file nó ghim là bất biến.

**SHA freeze của bản v4:**

| File | SHA-256 |
|---|---|
| `analysis_test.py` | `22b68f7f09c01c8dcc8a0a0bf37fffa82998a549f95915bd97733e78cc30bc0c` |
| `test_schema.py` | `07dd7e61d3db5280ae2c0e0e49cbd450bd7f097fcc9f3806b3c17593a0565da2` |
| `TEST_ANALYSIS_PLAN.md`, `POST_LOCK_ADDENDUM.md` | ghim trong `TEST_PLAN_FREEZE.json` |

Mọi thay đổi sau freeze cần: mục PLn mới, reviewer duyệt, manifest mới, mục audit log mới — tất cả **trước khi mở TEST**. Sau khi mở TEST, không được sửa gì.
