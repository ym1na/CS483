# Description: Plot Relationships between SAHM and Gold Prices
# Generates plots for selected intervals showing SAHM and Gold price trends.
# Plots: 
#   4-panel shw=owing high correlation periods
#   a full timeline plot showing SAHM vs Gold prices across the entire dataset

# Inputs: SAHMREALTIME.csv, XAU_USD Historical Data.csv
# Outputs: sahm_gold_relationships.png, sahm_gold_timeline.png
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def load_and_prepare():
    script_dir = Path(__file__).resolve().parent
    sahm = pd.read_csv(script_dir / 'SAHMREALTIME.csv')
    sahm['observation_date'] = pd.to_datetime(sahm['observation_date'])
    sahm = sahm.set_index('observation_date')

    gold = pd.read_csv(script_dir / 'XAU_USD Historical Data.csv', quotechar='"')
    gold['Date'] = pd.to_datetime(gold['Date'])
    if 'Price' in gold.columns:
        gold = gold[['Date', 'Price']]
    elif 'Value' in gold.columns:
        gold = gold[['Date', 'Value']]
        gold = gold.rename(columns={'Value': 'Price'})
    else:
        cols = [c for c in gold.columns if c.lower() != 'date']
        gold = gold[['Date', cols[0]]]
        gold = gold.rename(columns={cols[0]: 'Price'})

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
    return merged

def plot_interval(df, start_date, end_date, window_months, corr_value, ax=None, title=None):
    """Plot SAHM and gold for a specific interval with correlation value in title"""
    interval = df.loc[start_date:end_date]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot SAHM on left y-axis
    color1 = '#1f77b4'  # blue
    ax.plot(interval.index, interval['SAHMREALTIME'], 
            color=color1, label='SAHM', linewidth=2)
    ax.set_ylabel('SAHM Recession Indicator', color=color1)
    ax.tick_params(axis='y', labelcolor=color1)
    
    # Plot Gold on right y-axis
    ax2 = ax.twinx()
    color2 = '#ff7f0e'  # orange
    ax2.plot(interval.index, interval['Gold_Price_Monthly_Avg'],
             color=color2, label='Gold Price (USD)', linewidth=2)
    ax2.set_ylabel('Gold Price (USD)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Title with correlation
    if title is None:
        title = f"{start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}\n{window_months}-month window, correlation = {corr_value:.3f}"
    ax.set_title(title)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    return ax

def main():
    # Load data
    merged_df = load_and_prepare()
    
    # Define intervals to plot (selected from our analysis)
    intervals = [
        # Strong positive correlations (different time periods)
        {
            'start': '2009-04-01',
            'end': '2009-07-01',
            'window': 3,
            'corr': 0.986,
            'title': '2009 Post-Crisis Recovery\n3-month window, correlation = 0.986'
        },
        {
            'start': '2013-09-01',
            'end': '2014-08-01',
            'window': 5,
            'corr': 0.892,
            'title': '2013-2014 Period\n5-month window, correlation = 0.892'
        },
        {
            'start': '2018-09-01',
            'end': '2019-03-01',
            'window': 4,
            'corr': 0.957,
            'title': '2018-2019 Period\n4-month window, correlation = 0.957'
        },
        # Strong negative correlation
        {
            'start': '2010-02-01',
            'end': '2010-06-01',
            'window': 3,
            'corr': -0.975,
            'title': '2010 Negative Relationship\n3-month window, correlation = -0.975'
        }
    ]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.ravel()
    
    # Plot each interval
    for i, interval in enumerate(intervals):
        plot_interval(
            merged_df,
            pd.Timestamp(interval['start']),
            pd.Timestamp(interval['end']),
            interval['window'],
            interval['corr'],
            ax=axes[i],
            title=interval['title']
        )
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot
    plot_path = Path(__file__).resolve().parent / 'sahm_gold_relationships.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to: {plot_path}")
    
    # Also create a full timeline plot
    plt.figure(figsize=(20, 8))
    ax = plt.gca()
    
    # Plot full timeline
    color1 = '#1f77b4'  # blue
    ax.plot(merged_df.index, merged_df['SAHMREALTIME'], 
            color=color1, label='SAHM', linewidth=2)
    ax.set_ylabel('SAHM Recession Indicator', color=color1)
    ax.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax.twinx()
    color2 = '#ff7f0e'  # orange
    ax2.plot(merged_df.index, merged_df['Gold_Price_Monthly_Avg'],
             color=color2, label='Gold Price (USD)', linewidth=2)
    ax2.set_ylabel('Gold Price (USD)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('Full Timeline: SAHM Recession Indicator vs Gold Price')
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the full timeline plot
    timeline_path = Path(__file__).resolve().parent / 'sahm_gold_timeline.png'
    plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
    print(f"Saved timeline plot to: {timeline_path}")

if __name__ == '__main__':
    main()