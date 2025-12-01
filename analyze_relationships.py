# Description: Main Data Processing Script
# Loads SAHM and Gold data, processes it, merges it, computes correlation, and saves the final dataset.
# Outputs: SAHM_vs_Gold_Monthly.csv

import pandas as pd
from pathlib import Path
import math

# Parameters(Things I have set)
THRESHOLD = 0.5  # strong correlation
MIN_WINDOW = 2   # 2 months
MAX_WINDOW = 12  # 12 months
TOP_N = 15       # top N windows to print across all window sizes

def load_and_prepare():
    script_dir = Path(__file__).resolve().parent
    sahm = pd.read_csv(script_dir / 'SAHMREALTIME.csv')
    sahm['observation_date'] = pd.to_datetime(sahm['observation_date'])
    sahm = sahm.set_index('observation_date')

    gold = pd.read_csv(script_dir / 'XAU_USD Historical Data.csv', quotechar='"')
    
    # normalize column names
    if 'Date' not in gold.columns:
        raise ValueError('Gold CSV missing Date column')
    
    # keep Date and Price/Value
    if 'Price' in gold.columns:
        gold = gold[['Date', 'Price']]
    elif 'Value' in gold.columns:
        gold = gold[['Date', 'Value']]
        gold = gold.rename(columns={'Value': 'Price'})
    else:
        cols = [c for c in gold.columns if c.lower() != 'date']
        gold = gold[['Date', cols[0]]]
        gold = gold.rename(columns={cols[0]: 'Price'})

    gold['Date'] = pd.to_datetime(gold['Date'], infer_datetime_format=True, errors='coerce')
    gold = gold.set_index('Date')
    gold = gold.sort_index(ascending=True)

    # clean price
    if gold['Price'].dtype == object:
        gold['Price'] = gold['Price'].str.replace(',', '', regex=False)
    gold['Price'] = pd.to_numeric(gold['Price'], errors='coerce')
    gold = gold.dropna(subset=['Price'])

    # resample to monthly averages (month start)
    gold_monthly = gold['Price'].resample('MS').mean().to_frame()
    gold_monthly.columns = ['Gold_Price_Monthly_Avg']

    merged = sahm.join(gold_monthly, how='inner')
    merged = merged.dropna()
    merged.index = pd.to_datetime(merged.index)

    return merged


# Function that finds strong intervals of correlation(monthly rolling correlation)
def find_strong_intervals(merged, threshold=THRESHOLD, min_w=MIN_WINDOW, max_w=MAX_WINDOW):
    results = []  # list of (window_months, end_index, start_date, end_date, corr_value)

    for w in range(min_w, max_w + 1):
        # result indexed like merged, value corresponds to window ending at that index
        corr_series = merged['SAHMREALTIME'].rolling(window=w).corr(merged['Gold_Price_Monthly_Avg'])
        cond = corr_series.abs() >= threshold
        cond = cond.fillna(False)

        # compress consecutive True runs to intervals
        runs = []
        run_start = None
        run_end = None
        for i, flag in enumerate(cond):
            if flag and run_start is None:
                run_start = i
                run_end = i
            elif flag and run_start is not None:
                run_end = i
            elif not flag and run_start is not None:
                runs.append((run_start, run_end))
                run_start = None
                run_end = None
        if run_start is not None:
            runs.append((run_start, run_end))

        idx = merged.index
        for (s_idx, e_idx) in runs: # correlation ending at position i, windows covers [i-w+1 .. i]
            start_date = idx[max(0, s_idx - w + 1)]
            end_date = idx[e_idx]
            avg_corr = corr_series.iloc[s_idx:e_idx + 1].mean() # average correlation over the run
            max_corr = corr_series.iloc[s_idx:e_idx + 1].max()
            min_corr = corr_series.iloc[s_idx:e_idx + 1].min()
            results.append({
                'window_months': w,
                'start_date': pd.to_datetime(start_date).date(),
                'end_date': pd.to_datetime(end_date).date(),
                'length_months': ( (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) ) + 1,
                'avg_corr': float(avg_corr),
                'max_corr': float(max_corr) if not math.isnan(max_corr) else None,
                'min_corr': float(min_corr) if not math.isnan(min_corr) else None,
            })

    # Compute top windows by absolute corr (single-window maxima)
    single_windows = []
    for w in range(min_w, max_w + 1):
        corr_series = merged['SAHMREALTIME'].rolling(window=w).corr(merged['Gold_Price_Monthly_Avg'])
        for i, val in enumerate(corr_series):
            if pd.isna(val):
                continue
            single_windows.append({
                'window_months': w,
                'end_date': merged.index[i].date(),
                'start_date': merged.index[max(0, i - w + 1)].date(),
                'corr': float(val)
            })

    single_windows_sorted = sorted(single_windows, key=lambda x: abs(x['corr']), reverse=True)

    return results, single_windows_sorted[:TOP_N]


if __name__ == '__main__':
    merged = load_and_prepare()
    intervals, top_windows = find_strong_intervals(merged)

    print('\nSummary of intervals with |corr| >= {:.2f} (window sizes {}-{} months):\n'.format(THRESHOLD, MIN_WINDOW, MAX_WINDOW))
    if not intervals:
        print('No intervals found exceeding the threshold.')
    else: # sort by avg_corr magnitude
        intervals_sorted = sorted(intervals, key=lambda x: abs(x['avg_corr']), reverse=True)
        for it in intervals_sorted:
            print(f"{it['window_months']:2d}m  {it['start_date']} -> {it['end_date']}  length={it['length_months']}m  avg_corr={it['avg_corr']:.3f}  max={it['max_corr']:.3f}  min={it['min_corr']:.3f}")

    print('\nTop single windows by absolute correlation (end_date, window_months, corr):\n')
    for w in top_windows:
        print(f"{w['end_date']}  {w['window_months']:2d}m  corr={w['corr']:.3f}  ({w['start_date']} -> {w['end_date']})")

    # Save results to CSV for reference
    out_dir = Path(__file__).resolve().parent
    pd.DataFrame(intervals).to_csv(out_dir / 'strong_intervals_summary.csv', index=False)
    pd.DataFrame(top_windows).to_csv(out_dir / 'top_windows_summary.csv', index=False)

    print('\nWrote strong_intervals_summary.csv and top_windows_summary.csv')
