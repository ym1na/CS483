import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
DATA_PATH = os.path.join('data', 'master_dataset.csv')
OUTPUT_IMG = 'statistical_diagnosis.png'

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    # Ensure we have the target variable
    if 'Gold_Price' not in df.columns:
        raise KeyError("Gold_Price column missing")
    return df

def check_stationarity(series, name):
    """
    Runs ADF and KPSS tests to check for Random Walk behavior.
    (Replicates logic from Random_Walk_Analysis.ipynb)
    """
    print(f"\n--- Analyzing: {name} ---")
    
    # 1. ADF Test (Null Hypothesis: Non-Stationary / Unit Root)
    adf_result = adfuller(series.dropna())
    adf_p = adf_result[1]
    print(f"ADF p-value:  {adf_p:.4f}", end=" ")
    if adf_p < 0.05:
        print("(Stationary - REJECT Null)")
    else:
        print("(Non-Stationary - FAIL to Reject)")

    # 2. KPSS Test (Null Hypothesis: Stationary)
    kpss_result = kpss(series.dropna(), regression='c', nlags="auto")
    kpss_p = kpss_result[1]
    print(f"KPSS p-value: {kpss_p:.4f}", end=" ")
    if kpss_p > 0.05:
        print("(Stationary - FAIL to Reject)")
    else:
        print("(Non-Stationary - REJECT Null)")

def main():
    print("--- 02_2: Statistical Diagnosis (Colleague's Findings) ---")
    
    # 1. Load Data
    df = load_data()
    price = df['Gold_Price']
    
    # Calculate Log Returns (The Transformation)
    log_returns = np.log(price / price.shift(1)).dropna()

    # 2. Test Raw Prices
    print("\n[Finding 1]: Raw Prices follow a Random Walk")
    check_stationarity(price, "Raw Gold Price")
    
    # 3. Test Returns
    print("\n[Finding 2]: Log Returns are Stationary (Predictable)")
    check_stationarity(log_returns, "Log Returns")

    # 4. Generate Diagnostic Plots (ACF/PACF)
    # This visualizes the "Memory" of the market
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top Left: Raw Price
    axes[0, 0].plot(price, color='blue')
    axes[0, 0].set_title('Raw Gold Prices (Non-Stationary)')
    axes[0, 0].grid(True)
    
    # Top Right: Log Returns
    axes[0, 1].plot(log_returns, color='green')
    axes[0, 1].set_title('Log Returns (Stationary)')
    axes[0, 1].grid(True)
    
    # Bottom Left: ACF of Price (Shows long-term memory = Random Walk)
    plot_acf(price, ax=axes[1, 0], lags=40, title='ACF: Raw Price (High Correlation)')
    
    # Bottom Right: ACF of Returns (Shows almost zero memory = Efficiency)
    plot_acf(log_returns, ax=axes[1, 1], lags=40, title='ACF: Returns (Low Correlation)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    print(f"\nSaved diagnosis plot to {OUTPUT_IMG}")
    print("-" * 40)
    print("CONCLUSION: Raw prices cannot be predicted via regression.")
    print("CONCLUSION: We must model Returns to beat the baseline.")
    print("-" * 40)

if __name__ == "__main__":
    main()