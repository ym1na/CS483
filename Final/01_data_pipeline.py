import pandas as pd
import pandas_datareader.data as web
import datetime
import os

# --- CONFIGURATION ---
DATA_DIR = 'data'
GOLD_FILE = 'XAU_USD Historical Data.csv'
OUTPUT_FILE = os.path.join(DATA_DIR, 'master_dataset.csv')

# FUSION: Combining your indicators with dnguy44's list
FRED_SERIES = {
    'SAHMREALTIME': 'SAHMREALTIME', # Recession Indicator
    'T10Y2Y': 'T10Y2Y',             # Yield Curve Spread
    'VIXCLS': 'VIXCLS',             # Volatility
    'DEXUSEU': 'DEXUSEU',           # Exchange Rate
    'DTWEXBGS': 'DTWEXBGS',         # Dollar Index
    
    # NEW from dnguy44:
    'FEDFUNDS': 'FEDFUNDS',         # Fed Funds Rate (Crucial for Gold)
    'GS10': 'GS10',                 # 10-Year Treasury Yield
    'GS2': 'GS2',                   # 2-Year Treasury Yield
    
    # Macro Growth Factors
    'CPIAUCSL': 'CPIAUCSL',         # CPI (Inflation)
    'GDP': 'GDP',                   # GDP
    'PCEPI': 'PCEPI',               # PCE Inflation
    'SP500': 'SP500',               # Market
    'DCOILWTICO': 'Crude_Oil'       # FIXED: Naming consistency
}

def load_local_gold_data():
    path = os.path.join(DATA_DIR, GOLD_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}")
    
    print(f"Loading local Gold data from {path}...")
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    if df['Price'].dtype == object:
        df['Price'] = df['Price'].astype(str).str.replace(',', '')
    df['Price'] = df['Price'].astype(float)
    
    df = df.rename(columns={'Price': 'Gold_Price'})
    df = df.set_index('Date').sort_index()
    return df[['Gold_Price']]

def fetch_macro_data():
    start_date = datetime.datetime(2000, 1, 1)
    end_date = datetime.datetime.now()
    print("Fetching FUSION Macro data (FRED)...")
    try:
        df = web.DataReader(list(FRED_SERIES.keys()), 'fred', start_date, end_date)
        df = df.rename(columns=FRED_SERIES)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def main():
    # 1. Load Gold
    gold_df = load_local_gold_data()

    # 2. Load Macro
    macro_df = fetch_macro_data()
    
    # 3. Merge
    print("Merging datasets...")
    merged_df = gold_df.join(macro_df, how='left')
    
    # 4. Fill Gaps
    merged_df.ffill(inplace=True)
    merged_df.dropna(inplace=True)
    
    merged_df.to_csv(OUTPUT_FILE)
    print(f"Success! Fusion dataset saved to {OUTPUT_FILE}")
    print(f"Columns: {list(merged_df.columns)}")

if __name__ == "__main__":
    main()