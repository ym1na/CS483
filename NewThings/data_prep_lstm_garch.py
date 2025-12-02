# Description: Comprehensive Data Preparation for LSTM-GARCH Hybrid Model
# Combines SAHM indicator, daily gold prices, and macro market data
# Outputs: Scaled training/testing data, feature matrices, and metadata
# Inputs: SAHMREALTIME.csv, XAU_USD Historical Data.csv, macro data

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
LOOKBACK_WINDOW = 30  # Use 30 days of history for LSTM
FORECAST_HORIZON = 5  # Predict 5 days ahead
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test

# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_gold_prices():
    """Load and prepare daily gold prices from XAU_USD Historical Data.csv"""
    script_dir = Path(__file__).resolve().parent.parent / 'ym1na' / 'data'
    data_dir = script_dir
    
    gold_path = data_dir / 'XAU_USD Historical Data.csv'
    gold = pd.read_csv(gold_path, quotechar='"')
    
    # Normalize column names
    if 'Price' in gold.columns and 'Date' in gold.columns:
        gold = gold[['Date', 'Price']]
    elif 'Value' in gold.columns and 'Date' in gold.columns:
        gold = gold[['Date', 'Value']]
        gold = gold.rename(columns={'Value': 'Price'})
    else:
        cols = [c for c in gold.columns if c.lower() != 'date']
        gold = gold[['Date', cols[0]]]
        gold = gold.rename(columns={cols[0]: 'Price'})
    
    # Convert date and handle price cleaning
    gold['Date'] = pd.to_datetime(gold['Date'], infer_datetime_format=True, errors='coerce')
    if gold['Price'].dtype == object:
        gold['Price'] = gold['Price'].str.replace(',', '', regex=False)
    gold['Price'] = pd.to_numeric(gold['Price'], errors='coerce')
    
    # Sort and remove duplicates/NaN
    gold = gold.dropna(subset=['Date', 'Price'])
    gold = gold.sort_values('Date')
    gold = gold.drop_duplicates(subset=['Date'], keep='first')
    gold = gold.set_index('Date')
    
    print(f"[OK] Loaded gold prices: {len(gold)} daily records from {gold.index.min()} to {gold.index.max()}")
    return gold[['Price']]


def load_sahm_indicator():
    """Load SAHM recession indicator data"""
    script_dir = Path(__file__).resolve().parent.parent / 'ym1na' / 'data'
    data_dir = script_dir
    
    sahm_path = data_dir / 'SAHMREALTIME.csv'
    sahm = pd.read_csv(sahm_path)
    sahm['observation_date'] = pd.to_datetime(sahm['observation_date'])
    sahm = sahm.set_index('observation_date')
    sahm = sahm.sort_index()
    
    print(f"[OK] Loaded SAHM data: {len(sahm)} monthly records from {sahm.index.min()} to {sahm.index.max()}")
    return sahm[['SAHMREALTIME']]


# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

def compute_technical_indicators(gold_df):
    """Compute technical indicators for gold prices"""
    df = gold_df.copy()
    
    # Log returns
    df['log_return'] = np.log(df['Price'] / df['Price'].shift(1))
    
    # Simple Moving Averages
    df['SMA_5'] = df['Price'].rolling(window=5).mean()
    df['SMA_10'] = df['Price'].rolling(window=10).mean()
    df['SMA_20'] = df['Price'].rolling(window=20).mean()
    
    # Exponential Moving Average
    df['EMA_12'] = df['Price'].ewm(span=12, adjust=False).mean()
    
    # MACD (12-26-9)
    exp1 = df['Price'].ewm(span=12, adjust=False).mean()
    exp2 = df['Price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_SMA'] = df['Price'].rolling(window=20).mean()
    df['BB_STD'] = df['Price'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_SMA'] + (df['BB_STD'] * 2)
    df['BB_Lower'] = df['BB_SMA'] - (df['BB_STD'] * 2)
    df['BB_Position'] = (df['Price'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Momentum
    df['Momentum_5'] = df['Price'].diff(5)
    df['Momentum_10'] = df['Price'].diff(10)
    
    # Volatility (Rolling standard deviation of returns)
    df['Volatility_10'] = df['log_return'].rolling(window=10).std()
    df['Volatility_20'] = df['log_return'].rolling(window=20).std()
    
    # Rate of Change
    df['ROC_5'] = (df['Price'] - df['Price'].shift(5)) / df['Price'].shift(5) * 100
    df['ROC_10'] = (df['Price'] - df['Price'].shift(10)) / df['Price'].shift(10) * 100
    
    # Relative Strength Index (RSI)
    delta = df['Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    print(f"[OK] Computed {len(df.columns)-1} technical indicators")
    return df


def align_with_sahm(gold_df, sahm_df):
    """Align SAHM (monthly) with gold (daily) data using forward fill"""
    # Resample SAHM to daily (forward fill)
    sahm_daily = sahm_df.asfreq('D').fillna(method='ffill')
    
    # Align indices
    aligned = gold_df.join(sahm_daily, how='left')
    aligned = aligned.fillna(method='ffill')  # Handle any remaining NaN
    
    print(f"[OK] Aligned gold and SAHM data: {len(aligned)} daily records")
    return aligned


# ============================================================================
# 3. DATA PREPROCESSING & NORMALIZATION
# ============================================================================

def create_sequences(data, lookback=LOOKBACK_WINDOW, forecast_horizon=FORECAST_HORIZON):
    """
    Create sequences for LSTM training
    data: (n_samples, n_features)
    returns: (X, y) where X has shape (n_sequences, lookback, n_features)
             and y has shape (n_sequences, forecast_horizon)
    """
    X, y = [], []
    
    for i in range(len(data) - lookback - forecast_horizon + 1):
        # Input: past 'lookback' days
        X.append(data[i:(i + lookback)])
        # Target: next 'forecast_horizon' days (price values)
        y.append(data[(i + lookback):(i + lookback + forecast_horizon), 0])  # Only price
    
    return np.array(X), np.array(y)


def prepare_data(test_size=0.2, lookback=LOOKBACK_WINDOW, forecast_horizon=FORECAST_HORIZON):
    """
    Main data preparation pipeline
    Returns:
        - X_train, y_train, X_test, y_test: LSTM sequences
        - scaler: fitted MinMaxScaler for inverse transformation
        - gold_prices_original: original unscaled prices for comparison
        - metadata: dict with useful info
    """
    print("\n" + "="*70)
    print("STARTING DATA PREPARATION PIPELINE")
    print("="*70)
    
    # Step 1: Load data
    gold_df = load_gold_prices()
    sahm_df = load_sahm_indicator()
    
    # Step 2: Feature engineering
    gold_df = compute_technical_indicators(gold_df)
    
    # Step 3: Align SAHM with gold
    aligned_df = align_with_sahm(gold_df, sahm_df)
    
    # Step 4: Remove rows with NaN (due to indicator calculation windows)
    aligned_df = aligned_df.dropna()
    print(f"[OK] After removing NaN rows: {len(aligned_df)} records")
    
    # Save original gold prices before scaling
    gold_prices_original = aligned_df[['Price']].copy()
    
    # Step 5: Normalize all features to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(aligned_df)
    print(f"[OK] Scaled {len(aligned_df.columns)} features to [0, 1]")
    
    # Step 6: Create sequences
    X, y = create_sequences(scaled_data, lookback=lookback, forecast_horizon=forecast_horizon)
    print(f"[OK] Created sequences: X.shape={X.shape}, y.shape={y.shape}")
    
    # Step 7: Train-test split
    split_idx = int(len(X) * (1 - test_size))
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    print(f"[OK] Train-test split ({(1-test_size)*100:.0f}/{test_size*100:.0f}%)")
    print(f"  - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  - X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Metadata
    # Calculate the actual index in aligned_df for the split
    # Sequences start at row 'lookback', so split_idx in sequences corresponds to 
    # aligned_df index of (lookback + split_idx)
    split_date_idx = lookback + split_idx
    test_date_idx = lookback + split_idx + lookback
    
    metadata = {
        'lookback': lookback,
        'forecast_horizon': forecast_horizon,
        'n_features': aligned_df.shape[1],
        'feature_names': list(aligned_df.columns),
        'scaler_min': scaler.data_min_,
        'scaler_max': scaler.data_max_,
        'train_split_date': aligned_df.index[min(split_date_idx, len(aligned_df)-1)],
        'test_start_date': aligned_df.index[min(test_date_idx, len(aligned_df)-1)],
        'test_end_date': aligned_df.index[-1],
    }
    
    print("\n" + "="*70)
    print(f"DATA PREPARATION COMPLETE")
    print(f"Training set: {metadata['train_split_date']} to {metadata['test_start_date']}")
    print(f"Test set: {metadata['test_start_date']} to {metadata['test_end_date']}")
    print("="*70 + "\n")
    
    return X_train, y_train, X_test, y_test, scaler, gold_prices_original, aligned_df, metadata


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, scaler, orig_prices, aligned_data, meta = prepare_data()
    
    print("\nData preparation successful!")
    print(f"Features: {meta['feature_names']}")
    print(f"Scaler range: [{scaler.data_min_.min():.2f}, {scaler.data_max_.max():.2f}]")
