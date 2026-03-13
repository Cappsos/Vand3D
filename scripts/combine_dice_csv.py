import pandas as pd
import numpy as np
from pathlib import Path

def load_and_clean_csv(path):
    """Load a CSV, drop corrupted lines if needed."""
    try:
        return pd.read_csv(path)
    except Exception:
        # Attempt a simple clean: drop lines with wrong columns
        lines = Path(path).read_text().splitlines()
        header = lines[0].split(',')
        clean = [l for l in lines if len(l.split(',')) == len(header)]
        return pd.read_csv(pd.compat.StringIO("\n".join(clean)))


def combine_metrics(base_dir: Path, ks, method_filter=None):
    """
    Scan for each k in ks under base_dir/3D/{k_folder}/
    and combine the per-patient and lesion-bin CSVs into
    two comprehensive DataFrames with 'k_shot' column.
    """
    global_dfs = []
    lesion_dfs = []

    for k in ks:
        folder = f"{k}_patient" if k != 'full' else 'full_shot'
        root = base_dir / '3D' / folder
        metrics_file = root / 'all_methods_metrics.csv'
        lesion_file = root / 'lesion_size_analysis.csv'

        # Load per-patient metrics
        if metrics_file.exists():
            df_m = load_and_clean_csv(metrics_file)
            # filter out summary rows
            df_m = df_m[df_m['patient'] != 'GLOBAL'].copy()
            df_m['k_shot'] = k
            global_dfs.append(df_m)

        # Load lesion-level metrics
        if lesion_file.exists():
            df_l = load_and_clean_csv(lesion_file)
            df_l = df_l[df_l['patient'] != 'SUMMARY'].copy()
            df_l['k_shot'] = k
            lesion_dfs.append(df_l)

    global_all = pd.concat(global_dfs, ignore_index=True) if global_dfs else pd.DataFrame()
    lesion_all = pd.concat(lesion_dfs, ignore_index=True) if lesion_dfs else pd.DataFrame()
    return global_all, lesion_all


def summarize_global(global_df):
    """Compute aggregate stats per k_shot and method."""
    if global_df.empty:
        return pd.DataFrame()
    # Ensure numeric types
    num_cols = ['dice3d','hausdorff95','roc_auc','avg_precision','f1_max','lesion_iou','lesion_FP_count']
    for c in num_cols:
        global_df[c] = pd.to_numeric(global_df[c], errors='coerce')

    agg = global_df.groupby(['k_shot','method'])[num_cols].agg(
        ['mean','std','median','min','max']
    )
    # flatten columns
    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
    return agg.reset_index()


def summarize_lesions(lesion_df):
    """Compute per-k and per-size-bin summary of lesions."""
    if lesion_df.empty:
        return pd.DataFrame()
    # numeric conversions
    lesion_df['lesion_count'] = lesion_df['lesion_count'].astype(int)
    lesion_df['detected_count'] = lesion_df['detected_count'].astype(int)
    lesion_df['detection_rate'] = pd.to_numeric(lesion_df['detection_rate'], errors='coerce')
    lesion_df['avg_dice'] = pd.to_numeric(lesion_df['avg_dice'], errors='coerce')

    agg = lesion_df.groupby(['k_shot','size_bin']).agg(
        total_lesions=('lesion_count','sum'),
        total_detected=('detected_count','sum'),
        detection_rate_mean=('detection_rate','mean'),
        detection_rate_std=('detection_rate','std'),
        avg_dice_mean=('avg_dice','mean'),
        avg_dice_std=('avg_dice','std')
    )
    return agg.reset_index()


def save_results(global_summary, lesion_summary, out_dir: Path):
    """Persist summary DataFrames to CSV."""
    out_dir.mkdir(exist_ok=True, parents=True)
    if not global_summary.empty:
        global_summary.to_csv(out_dir / 'global_summary_by_k.csv', index=False)
    if not lesion_summary.empty:
        lesion_summary.to_csv(out_dir / 'lesion_summary_by_k.csv', index=False)


def main():
    BASE = Path('/home/nicoc/Vand/Vand3D/test_results_dice3d')
    ks = [1,5,10,15,20,30,40,50,'full']
    global_df, lesion_df = combine_metrics(BASE, ks)
    gsum = summarize_global(global_df)
    lsum = summarize_lesions(lesion_df)
    save_results(gsum, lsum, BASE / 'comprehensive_analysis')
    print('Saved summaries for global metrics and lesion analyses.')

if __name__ == '__main__':
    main()
