from pathlib import Path
import json
import pandas as pd
import argparse
import matplotlib.pyplot as plt

PLOTTING = True

try:
    import squarify
    HAS_SQUARIFY = True
except Exception:
    HAS_SQUARIFY = False

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _normalize_id_series(s: pd.Series) -> pd.Series:
    """Normalize id-like series for safer comparisons.

    - cast to str, strip whitespace
    - remove trailing ".0" (CSV float->int artifacts)
    - convert common 'nan'/'None'/empty to actual missing (pd.NA)
    Returns the normalized Series (dtype: object) with missing values as pd.NA.
    """
    if s is None:
        return s
    try:
        s2 = s.astype(str).str.strip()
    except Exception:
        # fallback: coerce via apply
        s2 = s.map(lambda x: '' if pd.isna(x) else str(x).strip())
    # remove trailing .0 that appears when numbers were written as floats
    s2 = s2.str.replace(r"\.0$", "", regex=True)
    # normalize obvious missing string tokens to actual NA
    s2 = s2.replace({'nan': pd.NA, 'None': pd.NA, '<NA>': pd.NA, '': pd.NA})
    return s2

def analyze_category_tree(raw_cat_path: Path, clean_cat_path: Path, outdir: Path, items_df=None):
    """Compute valid/total metric and produce treemap (or fallback)."""
    raw = pd.read_csv(raw_cat_path, low_memory=False) if raw_cat_path.exists() else None
    clean = pd.read_csv(clean_cat_path, low_memory=False) if clean_cat_path.exists() else None
    ensure_dir(outdir)

    total_raw = int(len(raw)) if raw is not None else 0
    valid_clean = 0
    if clean is not None and 'categoryid' in clean.columns:
        valid_clean = int(clean['categoryid'].notna().sum())

    metric = {'total_raw_categories': total_raw, 'valid_categories_in_clean': valid_clean}
    metric['valid_over_total'] = (valid_clean / total_raw) if total_raw > 0 else None

    # write metric
    with open(outdir / 'category_metric.json', 'w', encoding='utf-8') as f:
        json.dump(metric, f, indent=2)

    # Treemap: prefer sizes by item counts per category if items_df provided, else use cleaned category counts
    try:
        if PLOTTING:
            if items_df is not None and 'categoryid' in items_df.columns:
                sizes = items_df['categoryid'].value_counts().head(50)
                labels = [f"{i} ({v})" for i, v in zip(sizes.index.astype(str), sizes.values)]
                vals = sizes.values.tolist()
            elif clean is not None and 'categoryid' in clean.columns:
                sizes = clean['categoryid'].value_counts().head(50)
                labels = [f"{i} ({v})" for i, v in zip(sizes.index.astype(str), sizes.values)]
                vals = sizes.values.tolist()
            else:
                vals = []
                labels = []

            if vals:
                plt.figure(figsize=(12,8))
                if HAS_SQUARIFY:
                    squarify.plot(sizes=vals, label=labels, alpha=0.8)
                    plt.axis('off')
                    plt.title('Treemap: top categories by item count / category count')
                    plt.tight_layout()
                    plt.savefig(outdir / 'category_treemap.png')
                    plt.close()
                else:
                    # fallback: bar chart top categories
                    plt.barh(range(len(vals)), vals[::-1])
                    plt.yticks(range(len(vals)), labels[::-1])
                    plt.tight_layout()
                    plt.title('Top categories (fallback)')
                    plt.savefig(outdir / 'category_topbar_fallback.png')
                    plt.close()
    except Exception:
        pass

    return metric

def table_summary_metrics(fp: Path):
    """Return basic table-level metrics: rows, cols, percent_missing (overall)."""
    if not fp.exists():
        return None
    try:
        df = pd.read_csv(fp, low_memory=False)
    except Exception:
        # fallback: try reading with only header to count cols and lines
        try:
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                header = f.readline()
                cols = len(header.strip().split(',')) if header else 0
                rows = sum(1 for _ in f)
            return {'rows': rows, 'cols': cols, 'percent_missing': None}
        except Exception:
            return None

    rows, cols = df.shape
    # percent missing: fraction of all cells that are NA
    try:
        total_cells = rows * cols
        missing = int(df.isna().sum().sum())
        percent_missing = missing / total_cells * 100.0 if total_cells > 0 else None
    except Exception:
        percent_missing = None

    return {'rows': int(rows), 'cols': int(cols), 'percent_missing': percent_missing}

def analyze_item_properties(raw1: Path, raw2: Path | None, cleaned_items: Path, outdir: Path):
    """Histogram of properties-per-item (raw vs cleaned) and duplicate rates."""
    parts = []
    if raw1.exists():
        parts.append(pd.read_csv(raw1, low_memory=False))
    if raw2 and raw2.exists():
        parts.append(pd.read_csv(raw2, low_memory=False))
    raw_df = pd.concat(parts, ignore_index=True) if parts else None
    cleaned_df = pd.read_csv(cleaned_items, low_memory=False) if cleaned_items.exists() else None
    ensure_dir(outdir)

    results = {}
    # duplicate rate
    if raw_df is not None:
        dup_rate_raw = float(raw_df.duplicated().mean()) * 100.0
    else:
        dup_rate_raw = None
    if cleaned_df is not None:
        dup_rate_clean = float(cleaned_df.duplicated().mean()) * 100.0
    else:
        dup_rate_clean = None
    results['duplicate_rate_percent'] = {'raw': dup_rate_raw, 'clean': dup_rate_clean}

    # properties per item histogram
    try:
        if raw_df is not None and 'itemid' in raw_df.columns:
            raw_counts = raw_df.groupby('itemid').size()
            raw_counts.to_csv(outdir / 'raw_properties_per_item_counts.csv', header=['count'])
        else:
            raw_counts = None

        # for cleaned items: if it's tall (property/value) try to compute counts, otherwise count non-null property columns
        if cleaned_df is not None:
            if 'property' in cleaned_df.columns and 'itemid' in cleaned_df.columns:
                clean_counts = cleaned_df.groupby('itemid').size()
                clean_counts.to_csv(outdir / 'clean_properties_per_item_counts.csv', header=['count'])
            else:
                # count number of non-null columns per row as proxy
                nonnull_counts = cleaned_df.notna().sum(axis=1)
                nonnull_counts.to_csv(outdir / 'clean_properties_per_item_counts_proxy.csv', index=False, header=['nonnull_count'])
                clean_counts = nonnull_counts
        else:
            clean_counts = None

        # histogram plot
        if PLOTTING:
            plt.figure(figsize=(8,4))
            if raw_counts is not None:
                plt.hist(raw_counts.values, bins=50, alpha=0.6, label='raw')
            if clean_counts is not None:
                plt.hist(clean_counts.values if hasattr(clean_counts, 'values') else clean_counts, bins=50, alpha=0.6, label='clean')
            plt.legend()
            plt.title('Properties per item: raw vs clean')
            plt.tight_layout()
            plt.savefig(outdir / 'properties_per_item_hist.png')
            plt.close()
    except Exception:
        pass

    # write results
    with open(outdir / 'item_properties_summary.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    return results

def plot_size_comparison(metrics_map: dict, outdir: Path):
    """Plot overall size comparison (rows) between raw and clean for the main tables provided in metrics_map.

    metrics_map: mapping name -> {'raw': {...}, 'clean': {...}} where inner dict comes from table_summary_metrics
    """
    ensure_dir(outdir)
    # build bars for rows
    names = []
    raw_vals = []
    clean_vals = []
    for name, d in metrics_map.items():
        names.append(name)
        raw_rows = d.get('raw', {}).get('rows') if d.get('raw') else 0
        clean_rows = d.get('clean', {}).get('rows') if d.get('clean') else 0
        raw_vals.append(raw_rows or 0)
        clean_vals.append(clean_rows or 0)

    try:
        if PLOTTING and names:
            import numpy as _np
            x = range(len(names))
            width = 0.35
            plt.figure(figsize=(10, 4))
            plt.bar([i - width/2 for i in x], raw_vals, width=width, label='raw', alpha=0.7)
            plt.bar([i + width/2 for i in x], clean_vals, width=width, label='clean', alpha=0.7)
            plt.xticks(x, names, rotation=45, ha='right')
            plt.ylabel('rows')
            plt.title('Overall size comparison: Raw vs Clean')
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / 'size_chart.png')
            plt.close()
    except Exception:
        # silently ignore plotting errors
        pass

def generate_interpretation(metrics_map: dict, events_rates: dict | None, outdir: Path):
    """Generate a short textual interpretation based on metrics_map and event join rates."""
    ensure_dir(outdir)
    lines = []
    # summarize row/col changes
    for name, d in metrics_map.items():
        raw = d.get('raw')
        clean = d.get('clean')
        if raw is None and clean is None:
            lines.append(f"{name}: no data available.")
            continue
        r_rows = raw.get('rows') if raw else None
        c_rows = clean.get('rows') if clean else None
        r_cols = raw.get('cols') if raw else None
        c_cols = clean.get('cols') if clean else None
        r_miss = raw.get('percent_missing') if raw else None
        c_miss = clean.get('percent_missing') if clean else None

        # row change
        if r_rows is not None and c_rows is not None:
            try:
                pct = (c_rows - r_rows) / r_rows * 100.0 if r_rows > 0 else None
            except Exception:
                pct = None
            if pct is None:
                lines.append(f"{name}: rows raw={r_rows}, clean={c_rows}.")
            else:
                lines.append(f"{name}: rows changed from {r_rows} -> {c_rows} ({pct:+.1f}% change).")
        else:
            lines.append(f"{name}: rows raw={r_rows}, clean={c_rows}.")

        # missingness
        if r_miss is not None and c_miss is not None:
            try:
                miss_delta = c_miss - r_miss
            except Exception:
                miss_delta = None
            if miss_delta is not None:
                lines.append(f"  Missingness: raw={r_miss:.2f}%, clean={c_miss:.2f}% (delta {miss_delta:+.2f}%).")
        elif c_miss is not None:
            lines.append(f"  Missingness in clean: {c_miss:.2f}%.")

    # event join rates summary
    if events_rates:
        raw_rates = events_rates.get('raw')
        clean_rates = events_rates.get('clean')
        if raw_rates:
            lines.append(f"Events (raw): total={raw_rates.get('total_rows')}, meta_match={raw_rates.get('meta_match_pct'):.1f}% , cat_match={raw_rates.get('cat_match_pct'):.1f}%")
        if clean_rates:
            lines.append(f"Events (clean): total={clean_rates.get('total_rows')}, meta_match={clean_rates.get('meta_match_pct'):.1f}% , cat_match={clean_rates.get('cat_match_pct'):.1f}%")

    # small heuristic advice
    # if the rows increased significantly for events or items, warn about possible multiplicative join
    try:
        ev_raw_rows = metrics_map.get('events', {}).get('raw', {}).get('rows')
        ev_clean_rows = metrics_map.get('events', {}).get('clean', {}).get('rows')
        if ev_raw_rows and ev_clean_rows and ev_clean_rows > ev_raw_rows * 1.1:
            lines.append('Warning: event row count increased by >10% after cleaning — check for multiplicative joins or duplicate generation.')
    except Exception:
        pass

    # write interpretation
    interp_fp = outdir / 'interpretation.txt'
    try:
        with open(interp_fp, 'w', encoding='utf-8') as f:
            for l in lines:
                f.write(l + '\n')
    except Exception:
        pass

    # also return as string
    return '\n'.join(lines)

def analyze_events(raw_events: Path, cleaned_events: Path, meta_fp: Path, cat_fp: Path, outdir: Path, chunksize: int = 200_000):
    """Event type bar chart (raw vs clean) and join success rates (%) against metadata & category tree."""
    ensure_dir(outdir)
    # event type distributions
    ev_raw = None
    ev_clean = None
    try:
        if raw_events.exists():
            ev_raw = pd.read_csv(raw_events, usecols=['event'], low_memory=False)['event'].value_counts().to_dict()
    except Exception:
        ev_raw = None
    try:
        if cleaned_events.exists():
            ev_clean = pd.read_csv(cleaned_events, usecols=['event'], low_memory=False)['event'].value_counts().to_dict()
    except Exception:
        ev_clean = None

    # plot bar chart
    try:
        if PLOTTING and (ev_raw or ev_clean):
            keys = sorted(set((ev_raw or {}).keys()) | set((ev_clean or {}).keys()))
            raw_vals = [ev_raw.get(k, 0) for k in keys]
            clean_vals = [ev_clean.get(k, 0) for k in keys]
            x = range(len(keys))
            width = 0.35
            plt.figure(figsize=(10,5))
            plt.bar([i - width/2 for i in x], raw_vals, width=width, label='raw', alpha=0.7)
            plt.bar([i + width/2 for i in x], clean_vals, width=width, label='clean', alpha=0.7)
            plt.xticks(x, keys, rotation=45, ha='right')
            plt.legend()
            plt.title('Event type distribution: raw vs clean')
            plt.tight_layout()
            plt.savefig(outdir / 'event_type_bar_raw_vs_clean.png')
            plt.close()
    except Exception:
        pass

    # join success rates (chunked)
    meta = pd.read_csv(meta_fp, low_memory=False) if meta_fp.exists() else None
    cat = pd.read_csv(cat_fp, low_memory=False) if cat_fp.exists() else None
    # normalize id-like columns for safe comparisons
    if meta is not None and 'itemid' in meta.columns:
        meta['itemid'] = _normalize_id_series(meta['itemid'])
        if 'categoryid' in meta.columns:
            meta['categoryid'] = _normalize_id_series(meta['categoryid'])
    if cat is not None and 'categoryid' in cat.columns:
        cat['categoryid'] = _normalize_id_series(cat['categoryid'])

    def compute_join_rates(events_path: Path):
        total = 0
        matched_meta = 0
        matched_cat = 0
        if not events_path.exists():
            return None
        for chunk in pd.read_csv(events_path, chunksize=chunksize, low_memory=False):
            total += len(chunk)
            if 'itemid' in chunk.columns and meta is not None:
                chunk['itemid'] = _normalize_id_series(chunk['itemid'])
                # merge using available meta columns; if meta lacks categoryid, still count itemid matches via _merge
                # capture event-level category (if present) so we can check it separately
                if 'categoryid' in chunk.columns:
                    # create a stable column for event's own category (normalized)
                    chunk['__evt_cat__'] = _normalize_id_series(chunk['categoryid'])
                else:
                    chunk['__evt_cat__'] = pd.NA

                # prepare right side (meta). rename the meta category to avoid pandas suffixing
                if 'categoryid' in meta.columns:
                    right = meta[['itemid', 'categoryid']].drop_duplicates(subset=['itemid']).rename(columns={'categoryid': '__meta_cat__'})
                else:
                    right = meta[['itemid']].drop_duplicates(subset=['itemid'])

                merged = chunk.merge(right, on='itemid', how='left', indicator=True)

                # matched_meta: rows where itemid found in metadata (both)
                matched_meta += int((merged['_merge'] == 'both').sum())

                # for category match: check either event's category or meta's category against category lookup
                if cat is not None:
                    # use dropna normalized string set
                    cat_set = set(cat['categoryid'].dropna().astype(str))
                    # event category matches (we created __evt_cat__ on chunk)
                    evt_matches = merged['__evt_cat__'].notna() & merged['__evt_cat__'].astype(str).isin(cat_set)
                    meta_matches = False
                    if '__meta_cat__' in merged.columns:
                        meta_matches = merged['__meta_cat__'].notna() & merged['__meta_cat__'].astype(str).isin(cat_set)
                    # count rows where either event or meta provides a valid category
                    matched_cat += int((evt_matches | meta_matches).sum())
            else:
                # no itemid or no meta -> 0 matches
                pass
        if total == 0:
            return {'total_rows': 0, 'meta_match_pct': None, 'cat_match_pct': None}
        return {'total_rows': int(total), 'meta_match_pct': matched_meta / total * 100.0 if total>0 else None, 'cat_match_pct': matched_cat / total * 100.0 if total>0 else None}

    rates_raw = compute_join_rates(raw_events) if raw_events.exists() else None
    rates_clean = compute_join_rates(cleaned_events) if cleaned_events.exists() else None

    summary = {'event_counts_raw': ev_raw, 'event_counts_clean': ev_clean, 'join_rates_raw': rates_raw, 'join_rates_clean': rates_clean}
    with open(outdir / 'events_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    return summary

def main():
    p = argparse.ArgumentParser(description='Minimal EDA focused on categories, items, events')
    p.add_argument('--raw_events', default='data/raw/events.csv')
    p.add_argument('--cleaned_events', default='data/processed/full_cleaned.csv')
    p.add_argument('--raw_item1', default='data/raw/item_properties_part1.csv')
    p.add_argument('--raw_item2', default='data/raw/item_properties_part2.csv')
    p.add_argument('--cleaned_items', default='data/processed/item_info_cleaned.csv')
    # Use cleaned item info by default for join checks (prefer one-row-per-item cleaned output)
    p.add_argument('--meta', default='data/processed/item_info_cleaned.csv')
    p.add_argument('--raw_cat', default='data/raw/category_tree.csv')
    p.add_argument('--cleaned_cat', default='data/processed/category_tree_cleaned.csv')
    p.add_argument('--outdir', default='report/figures_minimal')
    p.add_argument('--chunksize', type=int, default=200000)
    args = p.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # load cleaned items if needed for treemap sizing
    cleaned_items_df = pd.read_csv(args.cleaned_items, low_memory=False) if Path(args.cleaned_items).exists() else None

    cat_metric = analyze_category_tree(Path(args.raw_cat), Path(args.cleaned_cat), outdir, items_df=cleaned_items_df)
    item_summary = analyze_item_properties(Path(args.raw_item1), Path(args.raw_item2) if args.raw_item2 else None, Path(args.cleaned_items), outdir)
    events_summary = analyze_events(Path(args.raw_events), Path(args.cleaned_events), Path(args.meta), Path(args.cleaned_cat), outdir, chunksize=args.chunksize)

    # build and write table-level summary metrics
    metrics_map = {}
    metrics_map['categories'] = {'raw': table_summary_metrics(Path(args.raw_cat)), 'clean': table_summary_metrics(Path(args.cleaned_cat))}
    metrics_map['items'] = {'raw': table_summary_metrics(Path(args.raw_item1)), 'clean': table_summary_metrics(Path(args.cleaned_items))}
    metrics_map['events'] = {'raw': table_summary_metrics(Path(args.raw_events)), 'clean': table_summary_metrics(Path(args.cleaned_events))}

    with open(outdir / 'summary_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_map, f, indent=2)

    # overall size chart (raw vs clean)
    plot_size_comparison(metrics_map, outdir)

    # generate a short textual interpretation and include it in the overall summary
    interp = generate_interpretation(metrics_map, {'raw': events_summary.get('join_rates_raw') if events_summary else None, 'clean': events_summary.get('join_rates_clean') if events_summary else None}, outdir)

    overall = {'category': cat_metric, 'items': item_summary, 'events': events_summary, 'metrics': metrics_map, 'interpretation': interp}
    with open(outdir / 'eda_minimal_summary.json', 'w', encoding='utf-8') as f:
        json.dump(overall, f, indent=2)

    print('Minimal EDA complete. Outputs in', outdir)

if __name__ == '__main__':
    main()