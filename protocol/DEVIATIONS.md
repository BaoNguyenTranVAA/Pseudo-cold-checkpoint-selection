# DEVIATIONS — Journal Protocol

Nhật ký chỉ ghi thêm, không xóa hay sửa mục cũ. Mỗi mục gồm: ngày, nội dung, lý do, dữ liệu đã được nhìn trước khi đổi, arm bị ảnh hưởng.

## Phần A. Sửa đổi trước khi khóa (V1.2 → V1.3), đã được người dùng duyệt

Các mục dưới đây xảy ra trước khi protocol được hash lần đầu, nên không phải deviation sau khóa. Chúng được ghi lại để bài báo công bố đầy đủ.

| # | Ngày | Nội dung | Lý do | Đã nhìn trước khi đổi | Arm ảnh hưởng |
|---|---|---|---|---|---|
| A1 | 28/10/2026 | Viết lại toàn bộ code; không dùng code hay file ADC; ADC chỉ để tham khảo | Quyết định của người dùng | Code và kết quả ADC (prior study) | Tất cả |
| A2 | 28/10/2026 | Seed hiệu dụng theo split (sau đó mở rộng theo domain, xem A9); kịch bản (d) tiêu chí ρ = 0,2, ρ = 0,5 chỉ sensitivity | Mô phỏng sơ bộ: với ICC 0,5 cả sign-flip và t-test có type-I ≈ 0,10 | Mô phỏng tổng hợp; không có dữ liệu split mới | Thủ tục thống kê |
| A3 | 28/10/2026 | CI cấp domain dùng t (df = 3) trên 4 giá trị split | Sign-flip trên 4 split có p nhỏ nhất 0,125, không cho CI 95% | Không | Báo cáo ước lượng, A_ID |
| A4 | 28/10/2026 | Tiêu chí build F: đúng 5.000 item, 2.900–3.000 user, mọi item có ảnh | VG có 2.999 user | Số đếm core VG/CSJ | Không |
| A5 | 28/10/2026 | RQ5 so fold PV với cold-DEV-700; pool 1.000 chỉ khi mở TEST | Không đọc artifact TEST trước khi mở | Không | RQ5 |
| A6 | 28/10/2026 | Negative sampler khớp theo slot; tái tạo thứ tự RNG để đối chiếu split 137 | Contrast sạch hơn; chứng minh dữ liệu trùng ADC | Không | Tất cả |
| A7 | 28/10/2026 | Bỏ Home_and_Kitchen trước khi tải; dùng Sports_and_Outdoors và Toys_and_Games, không fallback | Quyết định của người dùng | Không có dữ liệu Home_and_Kitchen nào được tải | Tất cả |
| A8 | 28/10/2026 | 4 domain: 16 đơn vị split, ≥ 3/4 domain, 1.200 trajectory, thứ tự cắt PC TG → PC SO → B1 → PC CSJ | Hệ quả của A7, người dùng xác nhận 4 domain | Số đếm, overlap, chance floor và nhất quán features của 16 split; chưa có output model | Thủ tục thống kê, ngân sách |
| A9 | 28/10/2026 | Seed hiệu dụng = 1.000.000·domain_index + 1.000·data_seed + r | Các domain cùng số user (CSJ, SO, TG) sẽ có init giống hệt nhau nếu seed chỉ theo split | Không | Tất cả |
| A10 | 28/10/2026 | Bật deterministic algorithms; chọn checkpoint trên giá trị làm tròn 8 chữ số | Bảo đảm G1 và khớp state đã lưu với phân tích | Không | Tất cả |

## Phần B. Ghi nhận kỹ thuật giai đoạn data (không đổi thiết kế)

| # | Ngày | Ghi nhận | Ảnh hưởng |
|---|---|---|---|
| B1 | 28/10/2026 | core_check với SO/TG: torch.load chế độ weights_only thất bại vì cache features có thêm trường thông tin pipeline; script chuyển sang load đầy đủ file do chính build_core.py tạo | Không; mọi kiểm tra thứ tự item, fingerprint, chuẩn L2 và hash đều đạt |
| B2 | 28/10/2026 | image_manifest.json của VG chứa đường dẫn Colab | Không; chỉ dùng để đếm độ phủ item |

## Phần C. Deviation sau khi khóa V1.3

Chưa có.

### Ghi thêm ngày 28/10/2026 (sau khóa V1.3, trước mọi huấn luyện)

Sáu mục C1–C6 dưới đây được đề xuất sau phản biện độc lập và được người phản biện duyệt ngày 28/10/2026, kèm các chỉnh sửa kỹ thuật của họ. PROTOCOL.md giữ nguyên SHA-256 c5c2913b…7483. Tại thời điểm ghi: chưa có model nào được train trên 16 split mới; dữ liệu đã nhìn gồm số đếm, overlap, chance floor, nhất quán features và mô phỏng tổng hợp. Không thay đổi số domain, số split, số seed, arm hay ngân sách 1.200 trajectory.

#### C1. Liều danh nghĩa và liều hiệu dụng (làm rõ)

- Số slot giữ nguyên: 205 / 410 / 685 trên batch 2.048.
- Tên arm dùng nhãn danh nghĩa q = 0,10 / 0,20 / 0,3346. Nhãn 0,3346 là tỷ lệ tự nhiên của VG trong ADC: 21.441 / (42.636 + 21.441).
- Liều hiệu dụng: 205/2048 = 0,10009765625; 410/2048 = 0,2001953125; 685/2048 = 0,33447265625, tức 10,01% / 20,02% / 33,45%.
- Bài báo luôn ghi kèm liều hiệu dụng bên cạnh nhãn danh nghĩa.
- Arm ảnh hưởng: không (chỉ nhãn).

#### C2. Null, nguồn ngẫu nhiên và phạm vi suy luận (bổ sung câu bị sót)

Câu khóa:

> Phân tích confirmatory có điều kiện trên bốn domain core cố định, xét tính biến thiên do cơ chế lấy split và training seed đã khóa. Sign-flip trên 16 split effects được diễn giải dưới giả định phân phối chung bất biến với các phép đổi dấu độc lập, với điều kiện đủ là các split effects độc lập và đối xứng quanh 0. Thủ tục này không phải randomization inference từ việc gán ngẫu nhiên CLEAN/LEAKY, và không mặc nhiên kiểm định giả thuyết yếu rằng trung bình hiệu ứng của bốn domain bằng 0. Các kịch bản hiệu ứng domain trái dấu được báo cáo như phân tích độ nhạy đối với giả thuyết yếu, không phải bằng chứng tổng quát về kiểm soát Type-I dưới giả thuyết ấy.

Làm rõ thêm:
- Câu "có điều kiện trên bốn core và bốn split đã quan sát" ở mục 6 của V1.3 được hiểu như sau: bốn core cố định. Nguồn biến thiên mà sign-flip dựa vào là cơ chế lấy mẫu split (data_seed) và training seed theo protocol. Không điều kiện hóa trên 16 split cụ thể.
- Tính độc lập của cơ chế lấy mẫu phải được xác nhận bằng bằng chứng, không chỉ suy ra từ việc dùng seed khác nhau:
  - mỗi split rút cold pool từ RNG riêng theo data_seed;
  - mọi luồng ngẫu nhiên của training suy ra từ seed hiệu dụng riêng;
  - trainer ghi seed hiệu dụng và hash init state;
  - G2 kiểm thêm rằng hash init state khác nhau giữa các bộ (domain, split, r) khác nhau.
- CI 95% bằng đảo ngược sign-flip được báo cáo kèm giả định của nó: hiệu ứng dịch chuyển chung cho các split. Khi hiệu ứng khác nhau giữa domain, khoảng này không được diễn giải là CI hợp lệ cho mọi dạng trung bình dị biệt.

Arm ảnh hưởng: không (thủ tục thống kê và cách viết).

#### C3. Diễn giải P2 (làm rõ)

- Câu kết luận của P2 được khóa ở dạng: "tác động của contamination vượt quá control thay slot bằng positive strict-train rút đều có hoàn lại theo protocol".
- A3 và A5 còn khác nhau ở phân bố user và item của positive được chèn, ở mức lặp lại cặp train, và ở việc kích hoạt ID. Vì vậy không khẳng định phân rã nhân quả hoàn chỉnh.
- Arm ảnh hưởng: không (diễn giải).

#### C4. Mở rộng G7 (thắt chặt)

1. Mô phỏng cùng lúc ba arm A0, A3, A5 ở mức (domain, split, r) để giữ tương quan giữa P1 (A3 − A0) và P2 (A3 − A5), vì hai contrast dùng chung A3.
2. Báo cáo riêng ba đại lượng: tỷ lệ bác bỏ P1; tỷ lệ bác bỏ P2 khi null của P2 đúng; tỷ lệ xác nhận cả chuỗi P1 → P2.
3. Kịch bản bổ sung (f): P1 có hiệu ứng thật, P2 null đúng (A3 = A5 > A0).
4. Kịch bản bổ sung (e): phương sai theo seed khác nhau giữa các domain. Tỷ lệ σ_e giữa domain lớn nhất và nhỏ nhất ∈ {1; 2; 3}.
5. Kiểm tra p-value theo tiêu chuẩn super-uniform: P(p ≤ α) ≤ α + 2 SE Monte Carlo với α ∈ {0,01; 0,025; 0,05; 0,10}. Báo cáo kèm histogram p-value; không đòi hỏi đều tuyệt đối vì p-value của sign-flip rời rạc.
6. Ô tiêu chí PASS/FAIL gồm các kịch bản thuộc null đối xứng mạnh: (a) chuẩn, (b) lệch phải đuôi nặng, (e) phương sai không đồng nhất. Riêng P2 có thêm ô (f).
   - Tiêu chí: tỷ lệ xác nhận theo quy tắc ba điều kiện ≤ 0,0544, và super-uniform ở các mức α nêu trên.
   - Kịch bản (c), u_j trái dấu tổng bằng 0, chỉ báo cáo độ nhạy đối với giả thuyết yếu, không phải ô tiêu chí.
7. Chuỗi dự phòng giữ như mục 6.1 của V1.3: sign-flip, rồi t-test (df = 15), rồi chỉ báo cáo ước lượng.

Arm ảnh hưởng: không (G7).

#### C5. Vai trò kịch bản (d): chọn phương án (i)

- Kịch bản (d), thành phần ngẫu nhiên chung ở cấp domain, chỉ là sensitivity; không dùng ρ = 0,2 hay bất kỳ ρ nào làm cổng PASS/FAIL.
- Báo cáo đầy đủ đường cong ρ ∈ {0,1; 0,2; 0,3; 0,5}.
- Lý do: phạm vi suy luận có điều kiện trên bốn core cố định. Phần chung của domain khi đó là hiệu ứng cố định u_j, không phải thành phần ngẫu nhiên w_j. Điều chỉnh này được thực hiện trước khi có kết quả huấn luyện, để tiêu chí khớp phạm vi suy luận, không phải để giúp thí nghiệm PASS.
- Báo cáo cuối phải phân biệt ba điều:
  - G7 PASS theo các kịch bản thuộc null đã xác định trước;
  - các dạng phụ thuộc cấp domain mà thủ tục không kiểm soát được Type-I (theo đường cong ρ);
  - giới hạn suy rộng ra domain chưa quan sát.
- G7 PASS không được dùng để khẳng định kiểm định hợp lệ dưới mọi dạng phụ thuộc.
- Arm ảnh hưởng: không (G7).

#### C6. Gate G4b: kiểm toán truy cập dữ liệu (thắt chặt, ba lớp)

1. **Data loader.** Tiến trình training chỉ đọc dữ liệu qua data_dev.load_dev và data_dev.load_features. Input dữ liệu hợp lệ gồm: `split_<seed>\dev.npz`, `split_<seed>\manifest_dev.json`, `core\features.npy`, `core\CORE_MANIFEST.json`. Không đọc CSV gốc, `core\core.npz` hay bất kỳ thứ gì trong `sealed\`.
2. **Runtime audit.** `sys.addaudithook` ghi nhận các sự kiện mở file. Đường dẫn được chuẩn hóa kiểu Windows (abspath, realpath, normcase).
   - Từ chối và dừng tiến trình khi mở nguồn dữ liệu bị cấm: mọi thứ dưới `CANTOR_online\`, mọi file `core.npz`, mọi đường dẫn có thành phần `sealed`.
   - Giám sát toàn bộ cây `journal_v1\data\`; truy cập ngoài danh sách input hợp lệ cũng bị từ chối.
   - Code, thư viện, checkpoint và output (`runs\`, `states\`, `logs\`) không thuộc danh sách dữ liệu input và không bị chặn.
   - Danh sách file dữ liệu đã truy cập được lưu vào manifest của từng trajectory.
   - Audit hook là công cụ ghi nhận và kiểm toán, không phải sandbox bảo mật.
3. **Kiểm thử độc lập.** Negative tests cố tình yêu cầu đọc `core.npz`, `sealed\test.npz` và CSV trong `CANTOR_online\`, để chứng minh gate chặn được. Sau mỗi lần chạy, đối chiếu log truy cập với danh sách hợp lệ.
4. G4b PASS phải có bằng chứng thực tế: log truy cập file và kết quả negative tests, không chỉ đoạn code đăng ký hook.

Arm ảnh hưởng: không (gate kiểm toán, áp dụng cho mọi trajectory).

#### C7. Phân loại lại các kịch bản G7 theo giả định đối xứng (làm rõ C4, ghi ngày 28/10/2026)

Người phản biện chỉ ra rằng kịch bản (b), sai số lognormal đã trừ trung bình, có trung bình 0 nhưng không đối xứng. Vì vậy (b) không thuộc null đối xứng mạnh mà tính chính xác của sign-flip dựa vào. Mục 6 của C4 đã xếp sai (b) vào ô tiêu chí. C7 thay thế phân loại ô của C4 như sau; các điểm còn lại của C4 giữ nguyên.

| Kịch bản | Mô tả | Vai trò |
|---|---|---|
| (a) | Sai số chuẩn, u = 0, w = 0 | Ô tiêu chí (null đối xứng mạnh) |
| (b′) | Sai số đuôi nặng nhưng đối xứng: Student-t với df ∈ {3; 5}, chuẩn hóa về phương sai σ_e² | Ô tiêu chí (null đối xứng mạnh) |
| (e) | Phương sai theo seed khác nhau giữa domain, tỷ lệ lớn nhất/nhỏ nhất ∈ {1; 2; 3}, sai số chuẩn | Ô tiêu chí (null đối xứng mạnh) |
| (f) | P1 có hiệu ứng thật, P2 null đúng (A3 = A5 > A0); A3 và A5 sinh theo cấu trúc hoán đổi được | Ô tiêu chí cho P2 và cho chuỗi P1 → P2 |
| (b) | Sai số lognormal lệch phải đã trừ trung bình, σ_ln ∈ {1,0; 1,5} | Độ nhạy khi giả định đối xứng không còn đúng; không gọi là null đối xứng mạnh |
| (c) | u_j trái dấu, tổng bằng 0 | Độ nhạy đối với giả thuyết yếu |
| (d) | Thành phần ngẫu nhiên chung cấp domain, ρ ∈ {0,1; 0,2; 0,3; 0,5} | Độ nhạy (theo C5) |

- Tiêu chí PASS/FAIL giữ như C4: tỷ lệ xác nhận theo quy tắc ba điều kiện ≤ 0,0544 và super-uniform ở α ∈ {0,01; 0,025; 0,05; 0,10}, áp dụng cho mọi ô tiêu chí, trên toàn lưới σ_e ∈ {0,001; 0,003; 0,005} và σ_v/σ_e ∈ {0; 0,5; 1; 2}.
- Kết quả của (b), (c), (d) được báo cáo đầy đủ; nếu (b) vượt ngưỡng, bài báo ghi rõ thủ tục nhạy với bất đối xứng, không thay đổi kết luận PASS/FAIL của G7.
- Đây là chỉnh sửa cách xác định điều kiện kiểm thử; không thay đổi dữ liệu, mô hình, arm, endpoint hay số lượt huấn luyện. Chưa có model nào được train tại thời điểm ghi.

Arm ảnh hưởng: không (G7).

#### C8. Seed chia split theo domain: sửa phụ thuộc giữa các domain (thay đổi cách chia split, ghi ngày 28/10/2026)

Người phản biện yêu cầu đánh giá phụ thuộc giữa các domain do seed chia cold pool dùng chung. Kiểm tra xác nhận đây là một phụ thuộc thật và gần như tuyệt đối.

- **Nguyên nhân.** Mục 3.2 của V1.3 rút cold pool bằng `default_rng(data_seed).choice(5000, 1000)`. Bốn domain đều có đúng 5.000 item, nên cùng một data_seed chọn ra cùng một tập chỉ số item cold ở cả 4 domain. Chỉ số item lại là thứ hạng độ phổ biến, vì mapping của mọi domain được sắp theo số interaction giảm dần. Phép chia DEV/TEST `default_rng(7).permutation(...)` cũng đưa cùng các vị trí thứ hạng vào TEST, ở mọi split và mọi domain.
- **Bằng chứng** (mô phỏng 400 seed trên core VG, SO, TG, không train):
  - với quy tắc cũ, cold-DEV của hai domain bất kỳ trùng 700/700 chỉ số; tương quan của log-degree trung bình của cold-DEV giữa các domain là 0,993–0,996;
  - với quy tắc mới, độ trùng trung bình là 98,4, đúng mức kỳ vọng 700²/5000 = 98, và tương quan nằm trong khoảng −0,006 đến 0,013.
  - Như vậy, với quy tắc cũ, các split effect cùng data_seed ở 4 domain không độc lập, vi phạm điều kiện đủ của sign-flip đã khóa ở C2.
- **Quy tắc mới** (`SPLIT_RNG_RULE = C8_domain_seeded`):
  - cold pool và jitter phân tầng fold: `default_rng([domain_index, data_seed])`;
  - phép chia DEV/TEST: `default_rng([domain_index, data_seed, 7])`;
  - domain_index: VG = 1, CSJ = 2, SO = 3, TG = 4. Phần còn lại của thuật toán chia không đổi.
- **ADC.** Quy tắc cũ chỉ còn dùng để tạo lại split 137 trong bộ nhớ cho G1b; kết quả G1b không đổi.
- **Hệ quả dữ liệu.**
  - Core của 4 domain không bị ảnh hưởng.
  - 16 split đã dựng ngày 28/10/2026 theo quy tắc cũ bị thay thế: xóa `split_*`, `SPLITS_REPORT.json`, `CHANCE_FLOOR.json` của mỗi domain, rồi dựng lại.
  - G0 và G8 phải chạy lại trên các split mới. Code từ chối nạp split dựng theo quy tắc cũ.
- **Gate bổ sung G0c** (`cross_domain_check.py`): với mọi data_seed và mọi cặp domain, số chỉ số cold-DEV trùng nhau phải nằm trong [60; 140] (kỳ vọng 98, độ lệch chuẩn khoảng 8,5). Kết quả ghi vào `CROSS_DOMAIN_REPORT.json`.
- **Bằng chứng độc lập trong domain.** Các split trong cùng domain có cold pool trùng trung bình khoảng 197/1000, đúng mức rút độc lập. Từ nay phép chia DEV/TEST cũng khác nhau giữa các split.
- **Đã nhìn trước khi đổi:** số đếm, overlap, chance floor và nhất quán features của 16 split cũ; mô phỏng tương quan nêu trên. Chưa có model nào được train.
- **Không thay đổi:** số domain, data_seed (237, 337, 437, 537), tỷ lệ cold 20%, tỷ lệ TEST 30%, quy tắc chia theo thời gian, fold phân tầng, arm, endpoint, ngân sách. Seed training hiệu dụng và tập con PC vốn đã chứa domain_index.

Arm ảnh hưởng: tất cả (dữ liệu split được dựng lại trước mọi huấn luyện).

#### C9. Dung sai Monte Carlo của G7 theo họ kiểm tra, ngưỡng nhị thức chính xác (duyệt ngày 28/10/2026)

- **Vấn đề.** C4 mục 5 đặt ngưỡng super-uniform α + 2 SE cho từng phép kiểm tra. G7 có K = 576 phép kiểm tra super-uniform dưới null đúng: 84 ô tiêu chí (a 12, b′ 24, e 24, f 24) tạo ra 144 cặp (ô, contrast) thuộc null, nhân với 4 mức α. Với một kiểm định chính xác, mỗi phép kiểm tra có khoảng 2,3% khả năng vượt ngưỡng do nhiễu Monte Carlo. Kỳ vọng 13 lần thất bại giả, và xác suất có ít nhất một lần là 0,999998. Bản thử 400 lần lặp cho 18 thất bại giả với sign-flip, vốn chính xác về mặt toán học.
- **Quyết định của người phản biện.** Duyệt nguyên tắc kiểm soát sai số Monte Carlo theo họ, dùng ngưỡng nhị thức chính xác với Bonferroni thay cho xấp xỉ chuẩn:
  - mức mỗi phép kiểm tra = 0,05 / 576 = 0,0000868;
  - FAIL ở mức α khi số lần p ≤ α (trong N = 10.000 lần lặp) ≥ c_α, với c_α là số nguyên nhỏ nhất thỏa P(Binomial(N, α) ≥ c_α) ≤ 0,05/576.

| α | Số lần tối đa còn PASS | Tỷ lệ tương ứng | C4 (α + 2 SE), chỉ báo cáo |
|---|---:|---:|---:|
| 0,010 | 139 | 0,0139 | 0,011990 |
| 0,025 | 311 | 0,0311 | 0,028122 |
| 0,050 | 584 | 0,0584 | 0,054359 |
| 0,100 | 1.114 | 0,1114 | 0,106000 |

- **Không thay đổi:**
  - ngưỡng p < 0,05 của kiểm định confirmatory P1/P2;
  - ngưỡng tỷ lệ xác nhận ≤ 0,0544 của C4, áp dụng riêng cho P1, cho P2 standalone và cho chuỗi P1 → P2;
  - danh sách ô tiêu chí theo C7 và chuỗi dự phòng sign-flip → t-test → chỉ ước lượng.
- **Thực hiện.**
  - `constants.G7_SU_RULE = "C9_exact_binomial"`.
  - `g7_typeI.py` tính và ghi cả kết quả theo C9 (ràng buộc) và theo C4 nguyên văn (chỉ báo cáo).
  - Summary ghi đủ K, N, seed mô phỏng, bảng ngưỡng, xác suất thất bại giả mỗi phép kiểm tra và hash code.
  - `g7_lock.py` từ chối khóa nếu lần chạy dùng quy tắc khác hoặc không đủ 10.000 lần lặp.
- **Sửa kèm theo** (người phản biện yêu cầu, mục 3). CI gộp của P1/P2 và các ước lượng secondary dùng đúng phương pháp của test đã khóa:
  - sign-flip: đảo ngược kiểm định dưới giả định dịch chuyển chung;
  - t-test: khoảng t với df = 15;
  - estimate_only: không có p-value và không có CI gộp.
  - CI cấp domain (t, df = 3) luôn ghi rõ phương pháp.
- **Kiểm thử lại.** Sau khi sửa module dùng chung, self-test tăng từ 47 lên 53 kiểm tra; 6 kiểm tra mới bao phủ CI theo từng phương pháp và chế độ estimate_only. Phải chạy lại `g7_selftest.py` trước `g7_typeI.py`.
- **Đã nhìn trước khi đổi:** chỉ bản thử 400 lần lặp dùng để kiểm tra code (không phải kết quả G7) và phép tính giải tích ở trên. Chưa có model nào được train; G7 bước 2 chưa chạy với 10.000 lần lặp.

Arm ảnh hưởng: không (G7 và cách báo cáo CI).

#### B3. Ghi nhận dữ liệu: 16 split dựng lại theo C8 (28/10/2026)

Thay cho bảng số đếm ở mục 3.3 của PROTOCOL.md, vốn tính theo quy tắc chia split cũ. Các gate đã đạt: G0 16/16, G1b (VG 6/6 số đếm + tập cold-DEV; CSJ 10/10 fingerprint), G8 16/16, G0c (cold-DEV trùng giữa các domain trung bình 95,5, khoảng 74–112, kỳ vọng 98).

| Domain | Cặp train | Cặp PV | Cặp cold-DEV | Batch/epoch | NDCG@10 ngẫu nhiên |
|---|---|---|---|---|---|
| VG | 43.390–43.694 | 21.825–22.011 | 11.495–12.239 | 22 | 0,0099–0,0101 |
| CSJ | 34.762–35.281 | 17.563–18.201 | 9.889–10.793 | 17–18 | 0,0093–0,0096 |
| SO | 26.572–26.793 | 13.423–13.509 | 7.804–8.080 | 13–14 | 0,0087–0,0088 |
| TG | 30.406–30.807 | 15.278–15.525 | 8.794–9.448 | 15–16 | 0,0091–0,0093 |

Trong cùng domain, cold pool của hai split trùng trung bình 196–209 item (kỳ vọng 200).

#### B4. Ghi nhận kỹ thuật: lỗ ghi qua os.open trong guard G4b (29/10/2026, người phản biện duyệt)

- **Lỗi.** Bản đầu của `data_guard.py` (SHA-256 58f4cedab93a6f74…) chỉ đọc tham số thứ hai của sự kiện audit `open`. Với `os.open`, tham số này là None; cờ truy cập nằm ở tham số thứ ba. Vì vậy một lệnh `os.open` với `O_WRONLY`, `O_RDWR`, `O_APPEND` hoặc `O_TRUNC` vào file DEV allowlist không bị chặn. Bộ G4b cũ vẫn báo PASS (18/18 trên VG 237 và TG 537), vì không có phép thử nào ghi qua `os.open`.
- **Phát hiện.** Người phản biện phát hiện khi đọc source, và tái hiện bằng một file DEV giả trong thư mục tạm.
- **Sửa** (`data_guard.py` SHA-256 f19f1a01a4748f07…, `g4b_check.py` d298cf224f9d681a…):
  - guard đọc cả mode và cờ, từ chối mọi cờ ghi, gồm `O_TRUNC` và `O_EXCL`;
  - chặn ghi qua alias (hard link) theo danh tính file;
  - chặn xóa, đổi tên, truncate và các lệnh tương tự trong cây dữ liệu;
  - bổ sung 16 kiểm thử.
- **Cách kiểm thử.** Các phép thử có thể phá hủy file chỉ chạy trên dữ liệu giả (sandbox) hoặc trên bản sao byte của 4 file allowlist (thư mục decoy). Trên dữ liệu thật chỉ có phép mở rồi đóng, không ghi byte nào.
- **Bằng chứng.**
  - Bộ kiểm thử mới chạy với guard cũ: FAIL 22/34, bytes của bản sao bị thay đổi.
  - Với guard mới, trên máy người dùng (Windows, Python 3.11.9, torch 2.2.2+cu121): VG 237 34/34 PASS và TG 537 34/34 PASS (report G4B_REPORT_VG_237_2026-10-29T002616Z, G4B_REPORT_TG_537_2026-10-29T002653Z).
  - SHA-256 của 7 file dữ liệu thật (4 file allowlist, sealed/test.npz, sealed/manifest_test.json, core.npz) khớp trước và sau mỗi lần chạy, theo các report trên.
- **Thời điểm.** Lỗi được phát hiện và sửa trước mọi huấn luyện; chưa có model nào được train trên 16 split.
- **Không thay đổi:** thiết kế V1.3, arm, endpoint, tiêu chí thống kê. G7 vẫn khóa (g7_lock_verify PASS cả 9 file, 29/10/2026).
- **Cùng đợt** (sửa code, không đổi thiết kế): `model.py` tách bộ đếm thành `Activation(n_slots, n_unique, n_new)` và `n_active_unique`; `g6_check.py` so khớp bảng 15 arm với bảng protocol chép tay. G6: 107/107 PASS (report G6_REPORT_2026-10-29T002709Z).

Arm ảnh hưởng: không (gate kiểm toán dữ liệu).
