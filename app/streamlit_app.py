import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve

st.set_page_config(layout="wide", page_title="Conversion Rate Data Mining Dashboard")
st.title("Phân tích Hành vi Chuyển đổi Khách hàng (Data Mining)")

# --- Sidebar / Navigation ---
st.sidebar.header('Điều hướng')
page = st.sidebar.radio('Chọn Trang', [
    '1. Khám phá Dữ liệu (EDA)', 
    '2. Đánh giá Mô hình Đơn lẻ', 
    '3. So sánh các Mô hình', 
    '4. Luật kết hợp Apriori'
])

# --- Chức năng an toàn cấu trúc ---
@st.cache_data(ttl=300)
def load_csv_safe(path, **kwargs):
    if os.path.exists(path):
        try:
            return pd.read_csv(path, **kwargs)
        except Exception:
            return None
    return None

def load_json_safe(path):
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None

# --- Page 1: EDA ---
if page == '1. Khám phá Dữ liệu (EDA)':
    st.header("Khám phá dữ liệu (Exploratory Data Analysis)")
    st.write("Tổng quan về bộ dữ liệu của dự án Data Mining sau khi làm sạch.")
    
    events_path = os.path.join(os.path.dirname(__file__), '../data/processed/events_cleaned.csv')
    if not os.path.exists(events_path):
        st.warning(f"Chưa có tệp dữ liệu. Vui lòng chạy **Notebook 01** trước.")
    else:
        df = load_csv_safe(events_path)
        if df is not None:
            st.success("Tải dữ liệu thành công!")
            st.write(f"**Số lượng dòng:** {df.shape[0]:,} | **Số lượng cột:** {df.shape[1]}")
            st.dataframe(df.head(100))
            
            st.subheader("Cột thuộc tính (Columns)")
            st.write(", ".join(df.columns))
            
            if 'event' in df.columns:
                st.subheader("Phân bố loại sự kiện (Event Types)")
                event_counts = df['event'].value_counts().reset_index()
                event_counts.columns = ['event', 'count']
                fig = px.bar(event_counts, x='event', y='count', title="Số lượng từng sự kiện")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Gặp lỗi khi đọc file dữ liệu.")

# --- Page 2: Modeling ---
elif page == '2. Đánh giá Mô hình Đơn lẻ':
    st.header("Đánh giá mô hình (Model Evaluation)")
    
    base_path = os.path.dirname(__file__)
    model_options = {
        "Decision Tree (Baseline)": {
            "metrics": os.path.join(base_path, "../models/decision_tree_baseline_metrics.json"),
            "preds": os.path.join(base_path, "../outputs/predictions_output.csv"),
            "fi": os.path.join(base_path, "../outputs/feature_importances.csv")
        },
        "Random Forest (Original)": {
            "metrics": os.path.join(base_path, "../models/random_forest_original_metrics.json"),
            "preds": os.path.join(base_path, "../outputs/predictions_output.csv"),
            "fi": os.path.join(base_path, "../outputs/feature_importances.csv")
        },
        "Random Forest (SMOTE)": {
            "metrics": os.path.join(base_path, "../models/random_forest_smote_metrics.json"),
            "preds": os.path.join(base_path, "../outputs/predictions_output_resampled.csv"),
            "fi": os.path.join(base_path, "../outputs/feature_importances_resampled.csv")
        },
        "XGBoost (GridSearchCV)": {
            "metrics": os.path.join(base_path, "../models/xgboost_gridsearch_metrics.json"),
            "preds": os.path.join(base_path, "../outputs/predictions_output_xgboost.csv"),
            "fi": os.path.join(base_path, "../outputs/feature_importances_xgboost.csv")
        }
    }
    
    selected_model = st.sidebar.selectbox("Chọn mô hình để xem:", list(model_options.keys()))
    paths = model_options[selected_model]
    
    # Check Metrics
    metrics = load_json_safe(paths['metrics'])
    if metrics:
        st.subheader("1. Chỉ số đánh giá tổng hợp (Metrics)")
        cols = st.columns(5)
        # Safely handling output logic
        cols[0].metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        cols[1].metric("Precision", f"{metrics.get('precision', 0):.4f}")
        cols[2].metric("Recall", f"{metrics.get('recall', 0):.4f}")
        cols[3].metric("F1-Score", f"{metrics.get('f1', 0):.4f}")
        cols[4].metric("ROC AUC", f"{metrics.get('roc_auc', 0):.4f}")
    else:
        st.warning(f"Chưa có tệp cấu hình `{paths['metrics']}` cho mô hình này. Vui lòng chạy các **Notebook 02, 03**.")

    # Check Predictions
    if paths['preds']:
        preds = load_csv_safe(paths['preds'])
        if preds is not None and 'label (converted)' in preds.columns and 'predict (converted)' in preds.columns:
            st.subheader("2. Confusion Matrix & ROC Curve")
            
            y_true = preds['label (converted)']
            y_pred = preds['predict (converted)']
            
            c1, c2 = st.columns(2)
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            cm_fig = go.Figure(data=go.Heatmap(
                z=cm, x=["Pred:0","Pred:1"], y=["Actual:0","Actual:1"], 
                colorscale="Blues", text=cm, texttemplate="%{text}"
            ))
            cm_fig.update_layout(title="Confusion Matrix")
            c1.plotly_chart(cm_fig, use_container_width=True)
            
            # ROC Curve
            if 'predict_probability (converted)' in preds.columns:
                y_proba = preds['predict_probability (converted)']
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                roc_fig = go.Figure()
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
                roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), showlegend=False))
                roc_fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
                c2.plotly_chart(roc_fig, use_container_width=True)
            else:
                c2.info("Không có dữ liệu xác suất (probability) để vẽ ROC Curve.")
    else:
        st.info("⏳ Cần chạy **Notebook 02** để tạo dữ liệu này.")

    # Check Feature Importances
    if paths['fi']:
        fi_df = load_csv_safe(paths['fi'])
        if fi_df is not None and 'feature' in fi_df.columns and 'importance' in fi_df.columns:
            st.subheader("3. Feature Importances")
            fi_plot = fi_df.sort_values('importance', ascending=False).head(20)
            fi_fig = px.bar(fi_plot, x='importance', y='feature', orientation='h', title='Mức độ quan trọng của đặc trưng (Top 20)')
            st.plotly_chart(fi_fig, use_container_width=True)

# --- Page 3: Comparison ---
elif page == '3. So sánh các Mô hình':
    st.header("So sánh Kết quả các Mô hình")
    st.write("So sánh độ hiệu quả tổng thể giữa các mô hình đã huấn luyện (Lấy từ kết quả Notebook 03 ở thư mục `report/figures/`).")
    
    metrics_to_plot = ['precision', 'recall', 'f1-score', 'roc-auc']
    found_any = False
    
    base_path = os.path.dirname(__file__)
    c1, c2 = st.columns(2)
    col_idx = 0
    for metric in metrics_to_plot:
        img_path = os.path.join(base_path, f"../report/figures/comparison_plot_{metric}.png")
        if os.path.exists(img_path):
            if col_idx % 2 == 0:
                c1.image(img_path, caption=f"Chỉ số {metric.upper()}", use_container_width=True)
            else:
                c2.image(img_path, caption=f"Chỉ số {metric.upper()}", use_container_width=True)
            col_idx += 1
            found_any = True
            
    if not found_any:
        st.warning("Không tìm thấy các tệp biểu đồ so sánh `comparison_plot_*.png`. Vui lòng chạy cell vẽ biểu đồ trong **Notebook 03**.")

# --- Page 4: Association Rules ---
elif page == '4. Luật kết hợp Apriori':
    st.header("Luật kết hợp giỏ hàng (Association Rules)")
    st.write("Dựa trên thuật toán Apriori, phân tích các sản phẩm/trang đích thường được xem hoặc mua cùng nhau.")
    
    base_path = os.path.dirname(__file__)
    rules_path = os.path.join(base_path, '../outputs/association_rules/all_rules.csv')
    if not os.path.exists(rules_path):
        st.warning(f"Chưa tìm thấy tập luật. Vui lòng chạy **Notebook 04** trước.")
    else:
        rules_df = load_csv_safe(rules_path)
        if rules_df is not None:
            st.success(f"Tải thành công {rules_df.shape[0]:,} điều kiện/luật.")
            
            # Filters
            st.subheader("Bộ lọc (Filter)")
            c1, c2, c3 = st.columns(3)
            min_support = c1.slider("Minimum Support", 0.0, 1.0, 0.01, 0.01)
            min_confidence = c2.slider("Minimum Confidence", 0.0, 1.0, 0.1, 0.05)
            min_lift = c3.slider("Minimum Lift", 0.0, 10.0, 1.0, 0.5)
            
            search_item = st.text_input("Tìm kiếm theo chuỗi (Ví dụ mã Item hoặc Category trong Antecedents / Consequents):", "")
            
            # Apply Filter
            filtered = rules_df[
                (rules_df['support'] >= min_support) &
                (rules_df['confidence'] >= min_confidence) &
                (rules_df['lift'] >= min_lift)
            ]
            
            if search_item:
                filtered = filtered[
                    filtered['antecedents'].astype(str).str.contains(search_item, case=False, na=False) |
                    filtered['consequents'].astype(str).str.contains(search_item, case=False, na=False)
                ]
                
            st.write(f"**Số lượng luật thỏa mãn:** {filtered.shape[0]}")
            st.dataframe(filtered.head(500))
        else:
            st.error("Gặp lỗi file association_rules.csv không tải được.")

