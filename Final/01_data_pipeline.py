import pandas as pd
import pandas_datareader.data as web
import datetime
import os
import time

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
    'FEDFUNDS': 'FEDFUNDS',         # Fed Funds Rate
    'GS10': 'GS10',                 # 10-Year Treasury Yield
    'GS2': 'GS2',                   # 2-Year Treasury Yield
    'CPIAUCSL': 'CPIAUCSL',         # CPI (Inflation)
    'GDP': 'GDP',                   # GDP
    'PCEPI': 'PCEPI',               # PCE Inflation
    'SP500': 'SP500',               # Market
    'DCOILWTICO': 'Crude_Oil'       # Oil
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
    # Fetch extra history to ensure we have coverage for the start of 2000
    start_date = datetime.datetime(1999, 1, 1) 
    end_date = datetime.datetime.now()
    
    # --- RETRY LOGIC ---
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Fetching FUSION Macro data (FRED)... Attempt {attempt + 1}/{max_retries}")
            df = web.DataReader(list(FRED_SERIES.keys()), 'fred', start_date, end_date)
            df = df.rename(columns=FRED_SERIES)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            if attempt < max_retries - 1:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("CRITICAL ERROR: Failed to fetch Macro Data after 3 attempts.")
                return None # Return None to signal failure

def main():
    # 1. Load Gold
    gold_df = load_local_gold_data()
    print(f"Gold Data Range: {gold_df.index.min().date()} to {gold_df.index.max().date()}")

    # 2. Load Macro
    macro_df = fetch_macro_data()
    
    if macro_df is None or macro_df.empty:
        print("\n!!! STOPPING !!!")
        print("Cannot create Master Dataset because Macro Data is missing.")
        print("Please check your internet connection and try again.")
        exit(1) # Stop the script with an error code
    
    # 3. Merge (Left Join keeps ALL Gold dates)
    print("Merging datasets...")
    merged_df = gold_df.join(macro_df, how='left')
    
    # 4. INTELLIGENT FILLING
    merged_df.ffill(inplace=True)
    merged_df.bfill(inplace=True)
    
    merged_df.dropna(subset=['Gold_Price'], inplace=True)
    
    merged_df.to_csv(OUTPUT_FILE)
    print(f"Success! Master dataset saved to {OUTPUT_FILE}")
    print(f"Final Row Count: {len(merged_df)}")
    print(f"Columns: {list(merged_df.columns)}")

if __name__ == "__main__":
    main()