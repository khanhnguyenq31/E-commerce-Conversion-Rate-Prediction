# Dự án Dự đoán Tỷ lệ Chuyển đổi E-commerce (Conversion Rate Prediction)

Dự án này tập trung vào việc phân tích dữ liệu hành vi người dùng trên nền tảng thương mại điện tử để dự đoán khả năng chuyển đổi (mua hàng) và đề xuất các chiến lược kinh doanh dựa trên luật kết hợp.

## Cấu trúc Dự án

```text
├── app/                    # Ứng dụng Dashboard (Streamlit)
├── data/                   # Dữ liệu (Raw & Processed - Ignored by Git)
├── models/                 # Các mô hình đã huấn luyện (.joblib)
├── notebooks/              # Quy trình phân tích dữ liệu (Jupyter Notebooks)
│   ├── 01_preprocessing.ipynb
│   ├── 02_baseline_classification.ipynb
│   ├── 03_advanced_classification.ipynb
│   └── 04_association_rules.ipynb
├── report/                 # Báo cáo và hình ảnh kết quả
│   └── figures/            # Các biểu đồ trực quan hóa (Confusion Matrix, ROC,...)
├── src/                    # Các script Python xử lý pipeline
│   ├── compute_cr.py       # Tính toán tỷ lệ chuyển đổi
│   ├── create_subdataset.py # Lấy mẫu dữ liệu phân tầng
│   └── eda_minimal.py      # Script EDA và trực quan hóa tự động
└── README.md               # Hướng dẫn sử dụng
```

## Quy trình Thực thi (3 Giai đoạn)

### Giai đoạn 1: Huấn luyện và Phân tích (Notebooks)

Thực thi tuần tự các Notebook trong thư mục `notebooks/`:

1.  **`01_preprocessing.ipynb`**: Làm sạch dữ liệu, xây dựng tập dữ liệu đã xử lý và chuẩn bị các đặc trưng đầu vào.
    - _Output chính_: `data/processed/events_cleaned.csv`.
2.  **`02_baseline_classification.ipynb`**: Trích xuất đặc trưng (Feature Engineering) trên lịch sử sự kiện và huấn luyện mô hình Decision Tree làm mốc cơ sở (Baseline).
    - _Output chính_: `models/decision_tree_baseline_metrics.json`.
3.  **`03_advanced_classification.ipynb`**: Áp dụng thuật toán Random Forest, xử lý mất cân bằng dữ liệu (SMOTE), và tối ưu XGBoost kết hợp GridSearchCV. So sánh hiệu suất giữa các mô hình.
    - _Output chính_: Các file `.joblib`, `.json`, bài đánh giá trong `data/processed/` và biểu đồ tĩnh trong `report/figures/`.
4.  **`04_association_rules.ipynb`**: Áp dụng thuật toán Apriori để tìm các luật kết hợp (Association Rules) giữa các sản phẩm được tương tác cùng nhau.
    - _Output chính_: `data/processed/association_rules.csv`.

### Giai đoạn 2: Trích xuất Mẫu Phân Tầng (Stratified Sampling)

Sau khi có `events_cleaned.csv` từ bước 1, sử dụng các kịch bản Python trong `src/` để trích xuất 20,000 người dùng theo phương pháp Lấy mẫu Phân tầng nhằm mục đích thống kê Conversion Rate (CR) và Metadata.

Mở terminal tại thư mục gốc của dự án và chạy tuần tự:

```bash
# B1: Trực quan hóa EDA (Thống kê và biểu đồ phân phối)
python src/eda_minimal.py --cleaned_events data/processed/events_cleaned.csv

# B2: Lấy mẫu 20,000 users và lưu trữ metadata
python src/create_subdataset.py

# B2: Tạo mapping phân tầng (converter / non-converter)
python src/mapping_user_stratum.py

# B3: Kiểm tra tính nhất quán quá trình lấy mẫu
python src/check_sampling_consistency.py

# B4: Tính toán tỷ lệ CR và khoảng tin cậy Wilson CI 95%
python src/compute_cr.py
```

_Các Output sinh ra bao gồm:_

- `data/processed/subdataset_for_conversion.csv`
- `data/processed/subdataset_for_conversion_meta.json`
- `data/processed/subdataset_for_conversion_stratum.json`
- `report/figures/cr_summary.csv`

### Giai đoạn 3: Trực quan hóa Giao diện (Streamlit Dashboard)

Khởi chạy ứng dụng Web cục bộ trên Streamlit để khám phá dữ liệu (EDA), đánh giá chi tiết, so sánh sức mạnh phân loại của thuật toán và tra cứu Luật kết hợp giỏ hàng.

```bash
streamlit run app/streamlit_app.py
```

## 3. Lưu ý Vận hành

- Nếu dữ liệu trong `data/raw/` thay đổi, toàn bộ pipeline từ Giai đoạn 1 đến Giai đoạn 2 cần được chạy lại định kì.
- Ứng dụng Streamlit ở Giai đoạn 3 được lập trình theo cơ chế phòng thủ (giảm thiểu lỗi Exception). Tuy nhiên, nếu bị thiếu file đầu ra do chưa chạy hoàn tất Giai đoạn 1 và 2, giao diện sẽ xuất hiện tệp cảnh báo yêu cầu thực hiện cung cấp dữ liệu theo đúng pipeline.
