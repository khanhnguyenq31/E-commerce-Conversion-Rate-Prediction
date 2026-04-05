import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import subprocess
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix


st.set_page_config(layout="wide", page_title="Conversion Model Dashboard")

st.title("Conversion Model Results")

# --- Sidebar / configuration ---
st.sidebar.header('Configuration')
outputs_dir = st.sidebar.text_input('Outputs folder (OLAP CSVs / PNGs)', value='scripts/outputs')
events_path = st.sidebar.text_input('Events CSV (used if OLAP CSVs missing)', value='data/processed/events_cleaned.csv')

st.sidebar.markdown('---')
page = st.sidebar.selectbox('Page', ['Model', 'OLAP', 'Compare'])

# Show model selector only on Model page
preds_path = None
fi_path = None
if page == 'Model':
    model_choice = st.sidebar.selectbox("Chọn model để hiển thị", ["Resampled (SMOTE)", "XGBoost"])
    # paths for different model outputs (assumed to be in outputs_dir)
    resampled_preds_path = os.path.join(outputs_dir, 'predictions_output_resampled.csv')
    xgb_preds_path = os.path.join(outputs_dir, 'predictions_output_xgboost.csv')
    resampled_fi_path = os.path.join(outputs_dir, 'feature_importances_resampled.csv')
    xgb_fi_path = os.path.join(outputs_dir, 'feature_importances_xgboost.csv')

    if model_choice == "Resampled (SMOTE)":
        preds_path = resampled_preds_path
        fi_path = resampled_fi_path
    elif model_choice == "XGBoost":
        preds_path = xgb_preds_path
        fi_path = xgb_fi_path

    st.sidebar.write('Predictions file:')
    st.sidebar.write(preds_path)

# Regenerate OLAP CSVs via olap.py (runs with --skip-plots to only persist CSVs)
if page == 'OLAP':
    if st.sidebar.button('Regenerate OLAP CSVs'):
        cmd = [sys.executable, 'scripts/olap.py', '--data-path', events_path, '--out-dir', outputs_dir, '--skip-plots']
        st.sidebar.info('Running OLAP script... this may take a moment.')
        with st.spinner('Running olap.py...'):
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
                st.sidebar.success('OLAP regeneration completed')
                st.sidebar.code(proc.stdout[-4000:])
            except subprocess.CalledProcessError as e:
                st.sidebar.error('OLAP script failed')
                st.sidebar.code(e.stderr[-4000:])


@st.cache_data(ttl=300)
def load_csv_safe(path, index_col=None):
    try:
        return pd.read_csv(path, index_col=index_col)
    except Exception:
        return None


def render_model_page():
    # --- Load predictions ---
    preds = load_csv_safe(preds_path)
    if preds is None:
        st.error(f"Không tìm thấy hoặc không thể đọc predictions file: {preds_path}")
        return

    # Normalize expected column names if necessary
    required_cols = ['label (converted)', 'predict (converted)', 'predict_probability (converted)']
    for c in required_cols:
        if c not in preds.columns:
            st.error(f"File {preds_path} thiếu cột '{c}'.")
            return

    y_true = preds['label (converted)']
    y_pred = preds['predict (converted)']
    y_proba = preds['predict_probability (converted)']

    # Compute metrics and show KPIs
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_proba)
    except Exception:
        roc_auc = float('nan')

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Accuracy", f"{accuracy:.3f}")
    k2.metric("Precision", f"{precision:.3f}")
    k3.metric("Recall", f"{recall:.3f}")
    k4.metric("F1", f"{f1:.3f}")
    k5.metric("ROC AUC", f"{roc_auc:.3f}" if not np.isnan(roc_auc) else 'n/a')

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_fig = go.Figure(data=go.Heatmap(z=cm, x=["Pred:0","Pred:1"], y=["Actual:0","Actual:1"], colorscale="Blues", text=cm, texttemplate="%{text}"))
    cm_fig.update_layout(title="Confusion Matrix", xaxis_title="", yaxis_title="")
    st.plotly_chart(cm_fig, width='stretch')

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})"))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), showlegend=False))
    roc_fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(roc_fig, width='stretch')

    # --- Comparison: attempt to load other model predictions for side-by-side view ---
    # We'll try to load both resampled and xgboost preds (if different from current selection)
    other_models = {}
    try:
        resampled_path = os.path.join(outputs_dir, 'predictions_output_resampled.csv')
        xgb_path = os.path.join(outputs_dir, 'predictions_output_xgboost.csv')
        if preds_path and os.path.abspath(preds_path) != os.path.abspath(resampled_path) and os.path.exists(resampled_path):
            other_models['Resampled (SMOTE)'] = load_csv_safe(resampled_path)
        if preds_path and os.path.abspath(preds_path) != os.path.abspath(xgb_path) and os.path.exists(xgb_path):
            other_models['XGBoost'] = load_csv_safe(xgb_path)
    except Exception:
        other_models = other_models

    # Helper to compute metrics from a predictions dataframe
    def _compute_metrics_from_df(df):
        if df is None:
            return None
        required_cols_local = ['label (converted)', 'predict (converted)', 'predict_probability (converted)']
        for c in required_cols_local:
            if c not in df.columns:
                return None
        y_t = df['label (converted)']
        y_p = df['predict (converted)']
        y_pr = df['predict_probability (converted)']
        try:
            roc = roc_auc_score(y_t, y_pr)
        except Exception:
            roc = float('nan')
        return dict(
            accuracy=accuracy_score(y_t, y_p),
            precision=precision_score(y_t, y_p, zero_division=0),
            recall=recall_score(y_t, y_p, zero_division=0),
            f1=f1_score(y_t, y_p, zero_division=0),
            roc_auc=roc
        )

    # Compute metrics for other models if available
    comparisons = {}
    for name, df in other_models.items():
        m = _compute_metrics_from_df(df)
        if m is not None:
            comparisons[name] = m

    # Present side-by-side metrics (selected model on left, others on right)
    if comparisons:
        # Build comparison table
        rows = []
        # include current selection as baseline
        rows.append((model_choice, dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1, roc_auc=roc_auc)))
        for n, m in comparisons.items():
            rows.append((n, m))

        # Create a DataFrame for display
        comp_df = pd.DataFrame({name: vals for name, vals in rows})
        # format numeric columns
        comp_disp = comp_df.copy()
        for c in comp_disp.columns:
            comp_disp[c] = comp_disp[c].apply(lambda v: f"{v:.3f}" if (isinstance(v, float) or isinstance(v, np.floating)) and not np.isnan(v) else ("n/a" if (isinstance(v, float) and np.isnan(v)) else str(v)))

        st.subheader('Model Comparison (Selected vs available models)')
        st.dataframe(comp_disp.T)

        # Simple automated commentary
        # Compare first other model to baseline
        try:
            baseline = rows[0][1]
            for other_name, other_vals in rows[1:]:
                comments = []
                # recall vs precision tradeoff
                if other_vals['recall'] > baseline['recall'] and other_vals['precision'] < baseline['precision']:
                    comments.append(f"{other_name} increases recall but reduces precision compared to {model_choice}.")
                if other_vals['precision'] > baseline['precision'] and other_vals['recall'] < baseline['recall']:
                    comments.append(f"{other_name} increases precision but reduces recall compared to {model_choice}.")
                # overall ROC AUC
                if other_vals['roc_auc'] > baseline['roc_auc']:
                    comments.append(f"{other_name} achieves higher ROC AUC ({other_vals['roc_auc']:.3f}) than {model_choice} ({baseline['roc_auc']:.3f}).")
                elif other_vals['roc_auc'] < baseline['roc_auc']:
                    comments.append(f"{other_name} has lower ROC AUC ({other_vals['roc_auc']:.3f}) than {model_choice} ({baseline['roc_auc']:.3f}).")
                # F1 comparison
                if other_vals['f1'] > baseline['f1']:
                    comments.append(f"{other_name} shows higher F1 ({other_vals['f1']:.3f}) than {model_choice} ({baseline['f1']:.3f}).")

                if not comments:
                    comments = [f"No large differences detected between {other_name} and {model_choice}."]
                st.markdown(f"**Insights vs {other_name}:** {' '.join(comments)}")
        except Exception:
            pass

    # Optional: show feature importances and sample predictions for the selected model
    show_details = st.checkbox('Show feature importances & sample predictions for selected model', value=True)
    if show_details:
        # Feature importances for the selected model
        fi = load_csv_safe(fi_path)
        if fi is not None and 'feature' in fi.columns and 'importance' in fi.columns:
            fi_plot = fi.sort_values('importance', ascending=False).head(30)
            fi_fig = px.bar(fi_plot, x='importance', y='feature', orientation='h', title='Top Feature Importances (selected model)', height=600)
            st.plotly_chart(fi_fig, width='stretch')
        else:
            st.info('Feature importance file for the selected model not found; skipping.')

        # Predictions preview + download for the selected model
        st.subheader('Sample predictions (first 200 rows) — selected model')
        try:
            st.dataframe(preds.head(200))
            csv = preds.to_csv(index=True).encode('utf-8')
            safe_name = model_choice.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            st.download_button('Download selected model predictions CSV', data=csv, file_name=f'predictions_{safe_name}.csv', mime='text/csv')
        except Exception:
            st.info('Unable to preview/download predictions for the selected model.')

    
def render_compare_page():
    st.header('Compare: OLAP vs RF & XGBoost')
    st.write('Interactive comparison: time dynamics (hourly) and segmentation precision (Top P selection).')

    # File paths (from outputs_dir)
    rf_pred_path = os.path.join(outputs_dir, 'predictions_output_resampled.csv')
    rf_imp_path = os.path.join(outputs_dir, 'feature_importances_resampled.csv')
    xgb_pred_path = os.path.join(outputs_dir, 'predictions_output_xgboost.csv')
    xgb_imp_path = os.path.join(outputs_dir, 'feature_importances_xgboost.csv')
    olap_hourly_csv = os.path.join(outputs_dir, 'olap_drilldown_hourly.csv')

    rf_pred = load_csv_safe(rf_pred_path)
    rf_imp = load_csv_safe(rf_imp_path)
    xgb_pred = load_csv_safe(xgb_pred_path)
    xgb_imp = load_csv_safe(xgb_imp_path)
    olap_hourly = load_csv_safe(olap_hourly_csv, index_col=0)

    # If OLAP hourly not available, try to compute from events CSV
    if olap_hourly is None:
        ev_path = find_events_path(events_path, outputs_dir)
        if ev_path:
            try:
                ev = pd.read_csv(ev_path)
                ev['timestamp'] = pd.to_datetime(ev['timestamp'], errors='coerce')
                ev = ev.dropna(subset=['timestamp'])
                ev['hour'] = ev['timestamp'].dt.hour
                ev['conversion'] = ((ev['event'] == 'transaction') & (ev.get('transactionid', -1) >= 0)).astype(int)
                hourly = ev.groupby('hour').agg(conversions=('conversion','sum'), total_views=('conversion','count'))
                hourly['conversion_rate'] = (hourly['conversions'] / hourly['total_views']).fillna(0)
                olap_hourly = hourly
            except Exception:
                olap_hourly = None

    def _extract_hour_importance(imp_df):
        # returns Series indexed 0..23 with importance (0 if missing)
        if imp_df is None:
            return pd.Series(0, index=range(24))
        if 'feature' not in imp_df.columns or 'importance' not in imp_df.columns:
            return pd.Series(0, index=range(24))
        mask = imp_df['feature'].astype(str).str.contains('most_frequent_hour', na=False)
        if not mask.any():
            return pd.Series(0, index=range(24))
        df = imp_df[mask].copy()
        def _parse_hour(x):
            try:
                return int(float(str(x).rsplit('_', 1)[-1]))
            except Exception:
                return None
        df['hour'] = df['feature'].apply(_parse_hour)
        df = df.dropna(subset=['hour'])
        if df.empty:
            return pd.Series(0, index=range(24))
        series = df.set_index('hour')['importance']
        series.index = series.index.astype(int)
        return series.reindex(range(24), fill_value=0)

    rf_hour_imp = _extract_hour_importance(rf_imp)
    xgb_hour_imp = _extract_hour_importance(xgb_imp)

    # Time Dynamics plot
    st.subheader('Time Dynamics: OLAP hourly conversion vs model hour-importance')
    if olap_hourly is None:
        st.warning('No OLAP hourly data available and could not compute from events; skipping time dynamics chart.')
    else:
        try:
            # Ensure olap_hourly has conversion_rate and hour index
            olap_plot = olap_hourly.copy()
            if 'conversion_rate' in olap_plot.columns:
                conv = olap_plot['conversion_rate']
            elif 'conversion_rate' not in olap_plot.columns and 'total_views' in olap_plot.columns and 'conversions' in olap_plot.columns:
                conv = (olap_plot['conversions'] / olap_plot['total_views']).fillna(0)
            else:
                conv = olap_plot.iloc[:, 0]

            conv_idx = list(range(0, 24))
            conv_vals = [conv.loc[h] if h in conv.index else 0 for h in conv_idx]

            fig_td = make_subplots(specs=[[{"secondary_y": True}]])
            fig_td.add_trace(go.Bar(x=conv_idx, y=[v * 100 for v in conv_vals], name='OLAP conversion (%)', marker_color='#3498db', opacity=0.5), secondary_y=False)
            fig_td.update_yaxes(title_text='Conversion (%)', secondary_y=False, tickformat=',.0%')

            # Add RF and XGB importances as lines (if available)
            if rf_hour_imp is not None and rf_hour_imp.sum() > 0:
                fig_td.add_trace(go.Scatter(x=list(rf_hour_imp.index), y=rf_hour_imp.values, mode='lines+markers', name='RF feature importance', line=dict(dash='dash', color='#e74c3c')), secondary_y=True)
            if xgb_hour_imp is not None and xgb_hour_imp.sum() > 0:
                fig_td.add_trace(go.Scatter(x=list(xgb_hour_imp.index), y=xgb_hour_imp.values, mode='lines+markers', name='XGB feature importance', line=dict(color='#8e44ad')), secondary_y=True)

            fig_td.update_xaxes(title_text='Hour of day (0-23)')
            fig_td.update_yaxes(title_text='Feature importance (relative)', secondary_y=True)
            fig_td.update_layout(height=480)
            st.plotly_chart(fig_td, width='stretch')
        except Exception as e:
            st.error(f'Error rendering time dynamics: {e}')

    # Segmentation Precision
    st.subheader('Segmentation Precision: Top P selection')
    top_pct = st.slider('Top percent to evaluate (Top P%)', min_value=5, max_value=50, value=20, step=5)
    results = {}
    # determine base preds to inspect for OLAP rule columns
    base_pred = rf_pred if rf_pred is not None else (xgb_pred if xgb_pred is not None else None)
    if base_pred is None:
        st.warning('No prediction files available to compute segmentation.')
    else:
        top_n = max(1, int(len(base_pred) * (top_pct / 100.0)))
        # OLAP rule-based: choose candidate column
        candidate_cols = [c for c in base_pred.columns if 'session_duration' in c or 'total_events' in c or 'unique_items_viewed' in c]
        if candidate_cols:
            rule_col = candidate_cols[0]
            olap_sel = base_pred.nlargest(top_n, rule_col)
            results['OLAP (rule)'] = olap_sel['label (converted)'].mean()
        else:
            results['OLAP (rule)'] = 0.0

        if rf_pred is not None:
            try:
                rf_sel = rf_pred.nlargest(top_n, 'predict_probability (converted)')
                results['Random Forest'] = rf_sel['label (converted)'].mean()
            except Exception:
                results['Random Forest'] = None
        if xgb_pred is not None:
            try:
                xgb_sel = xgb_pred.nlargest(top_n, 'predict_probability (converted)')
                results['XGBoost'] = xgb_sel['label (converted)'].mean()
            except Exception:
                results['XGBoost'] = None

        # Build bar chart
        if results:
            names = [k for k,v in results.items() if v is not None]
            vals = [results[k] for k in names]
            fig_seg = px.bar(x=names, y=vals, text=[f"{v:.1%}" for v in vals], title=f'Top {top_pct}% Segmentation Precision')
            fig_seg.update_yaxes(range=[0, max(vals) * 1.2 if vals else 1])
            st.plotly_chart(fig_seg, width='stretch')

    # Diagnostics: show which files were picked up
    st.markdown('**Compare sources found:**')
    for nm, path in [('rf_pred', rf_pred_path), ('rf_imp', rf_imp_path), ('xgb_pred', xgb_pred_path), ('xgb_imp', xgb_imp_path), ('olap_hourly', olap_hourly_csv)]:
        exists = os.path.exists(path)
        st.write(f"- {nm}: {'FOUND' if exists else 'MISSING'} ({path})")

    # Feature importances: prefer RF then XGBoost
    if rf_imp is not None:
        fi = rf_imp.copy()
        fi_title = 'Random Forest Feature Importances'
    elif xgb_imp is not None:
        fi = xgb_imp.copy()
        fi_title = 'XGBoost Feature Importances'
    else:
        fi = None

    if fi is not None and 'importance' in fi.columns and 'feature' in fi.columns:
        fi = fi.sort_values('importance', ascending=False).head(30)
        fi_fig = px.bar(fi, x='importance', y='feature', orientation='h', title=fi_title, height=600)
        st.plotly_chart(fi_fig, width='stretch')
    else:
        st.info('No feature importances found for RF or XGBoost; skipping.')

    # Predictions preview + download (use base_pred as the compare base)
    st.subheader('Sample predictions (first 200 rows) - base for compare')
    if base_pred is None:
        st.info('No prediction CSVs available to preview or download.')
    else:
        st.dataframe(base_pred.head(200))
        csv = base_pred.to_csv(index=True).encode('utf-8')
        if rf_pred is not None:
            safe_name = 'predictions_resampled'
        elif xgb_pred is not None:
            safe_name = 'predictions_xgboost'
        else:
            safe_name = 'predictions_compare'
        st.download_button('Download predictions CSV', data=csv, file_name=f'{safe_name}.csv', mime='text/csv')


def find_events_path(events_path, outputs_dir):
    candidates = [
        events_path,
        os.path.join(outputs_dir, 'events_cleaned.csv'),
        os.path.join('data', 'processed', 'events_cleaned.csv'),
        os.path.join('data', 'events_cleaned.csv'),
        'events_cleaned.csv'
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def plot_olap_from_csvs(outputs_dir, events_path):
    st.header('OLAP Analysis')
    olap_monthly_csv = os.path.join(outputs_dir, 'olap_rollup_monthly.csv')
    olap_cat_csv = os.path.join(outputs_dir, 'olap_rollup_categories.csv')

    mon_df = load_csv_safe(olap_monthly_csv, index_col=0)
    cat_df = load_csv_safe(olap_cat_csv, index_col=0)

    # Try to read precomputed drilldown and pivot CSVs (so we don't need raw events)
    drill_hourly_csv = os.path.join(outputs_dir, 'olap_drilldown_hourly.csv')
    drill_dow_csv = os.path.join(outputs_dir, 'olap_drilldown_dow.csv')
    pivot1_csv = os.path.join(outputs_dir, 'olap_pivot_1_conversion_rate.csv')
    pivot2_csv = os.path.join(outputs_dir, 'olap_pivot_2_conversion_rate.csv')

    drill_hourly_df = load_csv_safe(drill_hourly_csv, index_col=0)
    drill_dow_df = load_csv_safe(drill_dow_csv, index_col=0)
    pivot1_df = load_csv_safe(pivot1_csv, index_col=0)
    pivot2_df = load_csv_safe(pivot2_csv, index_col=0)

    events_df = None
    # If rollup/category missing OR drilldown/pivots missing, try to load raw events
    need_events = (mon_df is None or cat_df is None) or (drill_hourly_df is None and drill_dow_df is None and pivot1_df is None and pivot2_df is None)
    if need_events:
        resolved = find_events_path(events_path, outputs_dir)
        if resolved and os.path.exists(resolved):
            try:
                events_df = pd.read_csv(resolved)
            except Exception:
                events_df = None

    # If events_df available, compute missing pieces (only for what we don't already have)
    if events_df is not None:
        events_df['timestamp'] = pd.to_datetime(events_df['timestamp'], errors='coerce')
        events_df = events_df.dropna(subset=['timestamp'])
        events_df['month'] = events_df['timestamp'].dt.to_period('M')
        events_df['hour'] = events_df['timestamp'].dt.hour
        events_df['day_of_week'] = events_df['timestamp'].dt.day_name()
        events_df['conversion'] = ((events_df['event'] == 'transaction') & (events_df.get('transactionid', -1) >= 0)).astype(int)

        if mon_df is None:
            monthly = events_df.groupby('month').agg(conversions=('conversion','sum'), total_views=('conversion','count'))
            monthly['conversion_rate'] = (monthly['conversions'] / monthly['total_views']).round(4)
            mon_df = monthly
        if cat_df is None and 'categoryid' in events_df.columns:
            valid = events_df[events_df['categoryid'].notna()].copy()
            valid['category_name'] = 'Category_' + valid['categoryid'].fillna(-1).astype(int).astype(str)
            category = valid.groupby('category_name').agg(conversions=('conversion','sum'), total_views=('conversion','count'))
            category['conversion_rate'] = (category['conversions'] / category['total_views']).round(4)
            cat_df = category.sort_values('conversion_rate', ascending=False)

    # Diagnostics: show which sources are used
    sources = []
    if mon_df is not None:
        sources.append(f"monthly CSV: {os.path.basename(olap_monthly_csv)}")
    if cat_df is not None:
        sources.append(f"category CSV: {os.path.basename(olap_cat_csv)}")
    if drill_hourly_df is not None:
        sources.append(f"drilldown hourly CSV: {os.path.basename(drill_hourly_csv)}")
    if drill_dow_df is not None:
        sources.append(f"drilldown dow CSV: {os.path.basename(drill_dow_csv)}")
    if pivot1_df is not None:
        sources.append(f"pivot1 CSV: {os.path.basename(pivot1_csv)}")
    if pivot2_df is not None:
        sources.append(f"pivot2 CSV: {os.path.basename(pivot2_csv)}")
    if events_df is not None:
        sources.append(f"events source: {os.path.basename(resolved) if 'resolved' in locals() and resolved else events_path}")

    # if sources:
    #     st.info('Using data sources: ' + ', '.join(sources))

    # Monthly visualization (use same chart types as olap.py: lines for both series)
    if mon_df is not None:
        mon_plot = mon_df.copy()
        try:
            mon_plot.index = mon_plot.index.astype(str)
        except Exception:
            pass
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Monthly Conversion Rate', 'Monthly View Volume'))
        # Conversion rate as line
        fig.add_trace(go.Scatter(x=mon_plot.index, y=mon_plot['conversion_rate'], mode='lines+markers', name='Conversion Rate'), row=1, col=1)
        fig.update_yaxes(tickformat=',.0%', row=1, col=1)
        # View volume as line to match olap.py
        fig.add_trace(go.Scatter(x=mon_plot.index, y=mon_plot['total_views'], mode='lines+markers', name='View Volume', line=dict(color='#A23B72')), row=1, col=2)
        fig.update_layout(height=480)
        st.plotly_chart(fig, width='stretch')
    else:
        st.warning('No monthly rollup CSV or events data available')

    # Top categories
    if cat_df is not None:
        top_n = st.slider('Top N categories', min_value=5, max_value=50, value=10)
        top_cat_by_rate = cat_df.sort_values('conversion_rate', ascending=False).head(top_n).reset_index()
        fig_cat = px.bar(top_cat_by_rate, x='conversion_rate', y=top_cat_by_rate.columns[0], orientation='h', title=f'Top {top_n} Categories by Conversion Rate')
        fig_cat.update_xaxes(tickformat=',.0%')
        st.plotly_chart(fig_cat, width='stretch')

        # For view-volume, compute top N by total_views across full cat_df (matches olap visualization)
        top_cat_by_vol = cat_df.nlargest(top_n, 'total_views').reset_index()
        fig_cat_vol = px.bar(top_cat_by_vol, x=top_cat_by_vol.columns[0], y='total_views', title=f'Top {top_n} Categories by View Volume')
        st.plotly_chart(fig_cat_vol, width='stretch')
    else:
        st.info('No category rollup CSV available')

    # Drilldowns: prefer precomputed CSVs, otherwise fall back to events_df
    if drill_hourly_df is not None:
        hourly = drill_hourly_df.copy()
        # ensure columns
        if 'conversion_rate' not in hourly.columns and 'conversion_rate' in hourly:
            pass
        hourly = hourly.reset_index()
        fig2 = make_subplots(specs=[[{'secondary_y': True}]])
        fig2.add_trace(go.Scatter(x=hourly['hour'], y=hourly['conversion_rate'], mode='lines+markers', name='Conversion Rate'), secondary_y=False)
        fig2.add_trace(go.Bar(x=hourly['hour'], y=hourly['total_views'], name='View Count', opacity=0.6), secondary_y=True)
        fig2.update_yaxes(title_text='Conversion Rate', tickformat=',.0%', secondary_y=False)
        fig2.update_yaxes(title_text='View Count', secondary_y=True)
        fig2.update_layout(title_text='Hourly Conversion Rate & View Volume')
        st.plotly_chart(fig2, width='stretch')
    elif events_df is not None:
        hourly = events_df.groupby('hour').agg(conversions=('conversion','sum'), total_views=('conversion','count'))
        hourly['conversion_rate'] = (hourly['conversions'] / hourly['total_views']).round(4)
        hourly = hourly.reset_index()
        fig2 = make_subplots(specs=[[{'secondary_y': True}]])
        fig2.add_trace(go.Scatter(x=hourly['hour'], y=hourly['conversion_rate'], mode='lines+markers', name='Conversion Rate'), secondary_y=False)
        fig2.add_trace(go.Bar(x=hourly['hour'], y=hourly['total_views'], name='View Count', opacity=0.6), secondary_y=True)
        fig2.update_yaxes(title_text='Conversion Rate', tickformat=',.0%', secondary_y=False)
        fig2.update_yaxes(title_text='View Count', secondary_y=True)
        fig2.update_layout(title_text='Hourly Conversion Rate & View Volume')
        st.plotly_chart(fig2, width='stretch')

    # Day-of-week
    if drill_dow_df is not None:
        dow = drill_dow_df.copy()
        dow = dow.reset_index()
        # Ensure consistent weekday ordering Monday -> Sunday
        day_col = dow.columns[0]
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        try:
            if day_col in dow.columns:
                dow[day_col] = dow[day_col].astype(str)
                dow[day_col] = pd.Categorical(dow[day_col], categories=day_order, ordered=True)
                dow = dow.sort_values(day_col)
        except Exception:
            pass
        fig3 = px.bar(dow, x=day_col, y='conversion_rate', title='Conversion Rate by Day of Week', category_orders={day_col: day_order})
        fig3.update_yaxes(tickformat=',.0%')
        st.plotly_chart(fig3, width='stretch')
    elif events_df is not None:
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        dow = events_df.groupby('day_of_week').agg(conversions=('conversion','sum'), total_views=('conversion','count'))
        dow['conversion_rate'] = (dow['conversions'] / dow['total_views']).round(4)
        dow = dow.reindex([d for d in day_order if d in dow.index]).reset_index()
        fig3 = px.bar(dow, x='day_of_week', y='conversion_rate', title='Conversion Rate by Day of Week')
        fig3.update_yaxes(tickformat=',.0%')
        st.plotly_chart(fig3, width='stretch')

    # Pivot heatmaps: prefer precomputed pivot CSVs
    if pivot1_df is not None:
        try:
            p1 = pivot1_df.copy()
            p1_filled = p1.fillna(0)
            z = p1_filled.values
            x = list(p1_filled.columns)
            y = list(p1_filled.index)

            def _fmt_cell(v):
                if pd.isna(v):
                    return ''
                try:
                    f = float(v)
                except Exception:
                    return str(v)
                if 0 <= f <= 1:
                    return f"{f:.1%}"
                return f"{int(f):,}"

            heat = go.Heatmap(z=z, x=x, y=y, colorscale='RdYlGn', colorbar=dict(title='Conversion Rate'))
            fig_p1 = go.Figure(data=[heat])
            annotations = []
            for i_row, row in enumerate(z):
                for j_col, val in enumerate(row):
                    text = _fmt_cell(val)
                    annotations.append(dict(x=x[j_col], y=y[i_row], text=text, showarrow=False, font=dict(color='black' if (not pd.isna(val) and val < 0.5) else 'white')))
            fig_p1.update_layout(title='Conversion Rate by Availability / Time Segment', annotations=annotations, autosize=True)
            st.plotly_chart(fig_p1, width='stretch')
        except Exception:
            pass
    elif events_df is not None:
        try:
            events_df['time_segment'] = events_df['hour'].apply(lambda h: 'Morning' if 6<=h<12 else ('Afternoon' if 12<=h<18 else ('Evening' if 18<=h<24 else 'Night')))
            avail_col = 'availability_status' if 'availability_status' in events_df.columns else 'available'
            pivot1 = pd.pivot_table(events_df, values='conversion', index=avail_col, columns='time_segment', aggfunc='mean')
            fig_p1 = px.imshow(pivot1.fillna(0).values, x=pivot1.columns, y=pivot1.index, color_continuous_scale='RdYlGn', labels=dict(x='Time Segment', y=avail_col, color='Conversion Rate'), aspect='auto')
            st.plotly_chart(fig_p1, width='stretch')
        except Exception:
            pass

    if pivot2_df is not None:
        try:
            p2 = pivot2_df.copy()
            p2_filled = p2.fillna(0)
            z2 = p2_filled.values
            x2 = list(p2_filled.columns)
            y2 = list(p2_filled.index)

            def _fmt_cell2(v):
                if pd.isna(v):
                    return ''
                try:
                    f = float(v)
                except Exception:
                    return str(v)
                if 0 <= f <= 1:
                    return f"{f:.1%}"
                return f"{int(f):,}"

            heat2 = go.Heatmap(z=z2, x=x2, y=y2, colorscale='RdYlGn', colorbar=dict(title='Conversion Rate'))
            fig_p2 = go.Figure(data=[heat2])
            annotations2 = []
            for i_row, row in enumerate(z2):
                for j_col, val in enumerate(row):
                    text = _fmt_cell2(val)
                    annotations2.append(dict(x=x2[j_col], y=y2[i_row], text=text, showarrow=False, font=dict(color='black' if (not pd.isna(val) and val < 0.5) else 'white')))
            fig_p2.update_layout(title='Conversion Rate Heatmap (Day of Week × Time Segment)', annotations=annotations2, autosize=True)
            st.plotly_chart(fig_p2, width='stretch')
        except Exception:
            pass
    elif events_df is not None:
        try:
            pivot2 = pd.pivot_table(events_df, values='conversion', index='day_of_week', columns='time_segment', aggfunc='mean')
            day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            pivot2 = pivot2.reindex([d for d in day_order if d in pivot2.index])
            fig_p2 = px.imshow(pivot2.fillna(0).values, x=pivot2.columns, y=pivot2.index, color_continuous_scale='RdYlGn', labels=dict(x='Time Segment', y='Day of Week', color='Conversion Rate'), aspect='auto')
            st.plotly_chart(fig_p2, width='stretch')
        except Exception:
            pass

    # CSV previews
    st.subheader('OLAP CSV previews')
    if cat_df is not None:
        st.write('Top categories (rollup)')
        st.dataframe(cat_df.head(200))
    if mon_df is not None:
        st.write('Monthly rollup')
        st.dataframe(mon_df.head(200))


if page == 'Model':
    render_model_page()
elif page == 'OLAP':
    resolved_events = find_events_path(events_path, outputs_dir)
    events_arg = resolved_events if resolved_events is not None else events_path
    plot_olap_from_csvs(outputs_dir, events_arg)
elif page == 'Compare':
    render_compare_page()
