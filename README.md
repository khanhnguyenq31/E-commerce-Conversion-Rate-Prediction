Tập các file trong thư mục `data/processed/` chứa dữ liệu đã được tiền xử lý phục vụ cho phân tích hành vi và phân tích tỷ lệ chuyển đổi (Conversion Rate - CR). File `subdataset_for_conversion.csv` là một mẫu (sample) cấp người dùng được tạo để phân tích nhanh và phát triển mô hình.

Các file quan trọng (mô tả nhanh)
--------------------------------
- `full_cleaned.csv`
  - Dữ liệu đầy đủ đã làm sạch (fulldata). Đây là nguồn dữ liệu gốc đã được tiền xử lý từ các file raw.

- `subdataset_for_conversion.csv`
  - Subdataset (sample) cấp người dùng: chứa tất cả các event của những user được chọn trong mẫu.
  - Kích thước mẫu hiện tại: 20.000 users (tổng event ghi lại trong file: ~120k hàng).
  - Mẫu được tạo theo chiến lược user-level sampling và có oversample cho những user đã chuyển đổi (converters) để đảm bảo đủ positive cases.

- `subdataset_for_conversion_meta.json`
  - Metadata về sampling: chứa các trường quan trọng:
    - `population_users`: tổng số người dùng trong `full_cleaned.csv`.
    - `population_converters`: số người dùng có ít nhất 1 `transaction` trong population.
    - `sampled_users_total`, `sampled_conv`, `sampled_nonconv`.
    - `sampling_frac_conv` / `sampling_frac_nonconv` (tỷ lệ chọn mẫu theo lớp).
    - `weights`: inverse sampling fractions (dùng để bù trọng số khi cần estimate population từ sample).

- `subdataset_for_conversion_stratum.json`
  - Mapping `visitorid -> 'conv'|'nonconv'`.
  - Dùng để áp trọng số từng user khi tính weighted estimates (ví dụ: weighted CR).

- `report/figures/cr_summary.csv`
  - Tóm tắt CR: unweighted CR trên sample, CI (95%), population CR (từ meta), và weighted CR (nếu mapping tồn tại).

Scripts liên quan
-----------------
- `scripts/create_subdataset.py` hoặc `scripts/create_subdataset_for_conversion.py`
  - Script tạo subdataset (user-level sampling, 2-pass chunked). Có tham số: `N_USERS`, `MIN_CONVERTERS`, `CHUNK`.

- `scripts/compute_cr.py`
  - Tính và in: unweighted CR (sample) + Wilson CI, population CR (từ meta), weighted CR (nếu tồn tại `subdataset_for_conversion_stratum.json`).
  - Ghi `report/figures/cr_summary.csv`.

Khuyến nghị cho người nhận (analyst)
------------------------------------
1. Lưu ý rằng sample này đã oversample converters (do đó sample CR ≠ population CR). 
Để ước lượng CR cho population, dùng `population_converters / population_users` từ meta hoặc tính weighted CR bằng `meta['weights']` và `subdataset_for_conversion_stratum.json`.

2. Nếu muốn tính weighted CR tự động, hãy tạo file stratum mapping (nếu chưa có):
   - mapping được tạo từ `subdataset_for_conversion.csv` bằng script nhỏ (visitor có transaction -> 'conv', ngược lại -> 'nonconv').

3. Để nhanh chóng xem các chỉ số cơ bản:
   - Chạy `python .\scripts\compute_cr.py` để nhận báo cáo CR đầy đủ và lưu `report/figures/cr_summary.csv`.

4. Nếu cần phát triển mô hình hoặc EDA sâu hơn, đề xuất sử dụng `subdataset_for_conversion.csv` (nhanh) để thử nghiệm và `full_cleaned.csv` để chạy tính toán population-level hoặc training trên toàn bộ dữ liệu khi cần.

Thông tin bổ sung
- Khoảng thời gian dữ liệu trong mẫu: ~2015-05-03 → 2015-09-18 (kiểm tra lại nếu cần lọc theo ngày).

Ghi chú vận hành ( thêm )
--------------------------------
- File `subdataset_for_conversion_stratum.json` đã được tạo và nằm trong `data/processed/` — nếu không có, bạn có thể tạo lại bằng đoạn mã phía dưới.
- `scripts/compute_cr.py` sẽ tự động tính weighted CR nếu tìm thấy `subdataset_for_conversion_stratum.json` cùng với `subdataset_for_conversion_meta.json`; kết quả tóm tắt lưu tại `report/figures/cr_summary.csv`.

Lệnh tạo stratum map (nếu cần tái tạo)
python .\scripts\mapping_user_stratum.py

- Tính báo cáo CR (unweighted, population, weighted nếu mapping có):
  python .\scripts\compute_cr.py


------
Khi file data/processed/full_cleaned.csv thay đổi, cần chạy lại toàn bộ pipeline để cập nhật subdataset, meta, và kết quả phân tích tỷ lệ chuyển đổi (Conversion Rate).

1. Kiểm tra file đầu vào

Mục đích: xác nhận dữ liệu đầy đủ, cột hợp lệ, timestamp đúng format.

In vài dòng đầu, kiểm tra null, định dạng thời gian.

2. Tạo subdataset mới

Mục đích: chọn mẫu user (theo oversampling converter), trích toàn bộ events của họ.

python .\scripts\create_subdataset.py

Sinh ra:

data/processed/subdataset_for_conversion.csv

data/processed/subdataset_for_conversion_meta.json

data/processed/subdataset_for_conversion_stratum.json (nếu có)

3. (Tuỳ) Tạo lại stratum map

Nếu script chưa tạo:

python .\scripts\mapping_user_stratum.py

Output: subdataset_for_conversion_stratum.json

4. Kiểm tra tính nhất quán của mẫu

Mục đích: đảm bảo meta, sample, stratum khớp nhau.

python .\scripts\check_sampling_consistency.py

In ra tổng converter, non-converter, cảnh báo nếu mismatch.

5. Tính lại Conversion Rate (CR)

Mục đích: xuất kết quả CR có và không trọng số.

python .\scripts\compute_cr.py

Tạo report/figures/cr_summary.csv
(chứa unweighted CR, weighted CR, CI, population CR)