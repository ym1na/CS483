import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os

# --- CONFIGURATION ---
DATA_PATH = os.path.join('data', 'model_ready_dataset.csv')
SEQ_LENGTH = 60
TEST_SPLIT = 0.2

def load_and_engineer_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")
    
    # 1. Load Data
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print(f"Loaded Data Columns: {list(df.columns)}")

    # Check for Fusion columns
    if 'FEDFUNDS' not in df.columns:
        raise KeyError("Missing 'FEDFUNDS'. Please run the updated 01_data_pipeline.py")
    
    # --- STATIONARITY TRANSFORMATION ---
    
    # Target: Log Returns of Gold
    df['Log_Ret'] = np.log(df['Gold_Price'] / df['Gold_Price'].shift(1)) * 100
    
    # Macro Returns (Pct Change)
    # Uses 'Crude_Oil' consistently now
    df['Oil_Ret'] = df['Crude_Oil'].pct_change()
    df['SP500_Ret'] = df['SP500'].pct_change()
    df['DEX_Ret'] = df['DEXUSEU'].pct_change()
    
    # Rate Changes (Differencing)
    df['Rate_Change'] = df['FEDFUNDS'].diff()
    df['Yield_Spread_Change'] = df['T10Y2Y'].diff()
    df['Inflation_Change'] = df['CPIAUCSL'].pct_change()
    
    # Technicals
    df['SMA_14'] = df['Gold_Price'].rolling(window=14).mean()
    
    # Volatility Feature
    returns_clean = df['Log_Ret'].dropna()
    model = arch_model(returns_clean, vol='Garch', p=1, q=1, dist='Normal')
    res = model.fit(disp='off')
    df.loc[returns_clean.index, 'GARCH_Vol'] = res.conditional_volatility

    # Clean
    df.dropna(inplace=True)
    
    # Select Stationary Features
    features = ['Log_Ret', 'Oil_Ret', 'SP500_Ret', 'DEX_Ret', 'Rate_Change', 
                'Yield_Spread_Change', 'Inflation_Change', 'GARCH_Vol', 
                'Disaster_Flag', 'Epidemic_Start_Flag', 'avg_tone']
    
    print(f"Engineered Features used for training: {features}")
    return df, features

def create_sequences(data, target_idx, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, target_idx])
    return np.array(X), np.array(y)

def build_bidirectional_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        Dense(32, activation='swish'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def main():
    print("--- Training FUSION Alpha Model ---")
    
    # 1. Load & Engineer
    df, feature_cols = load_and_engineer_data()
    
    # 2. Scale
    data_subset = df[feature_cols].values
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_subset)
    
    target_idx = 0 # Log_Ret is the first column
    
    # 3. Sequence
    X, y = create_sequences(scaled_data, target_idx, SEQ_LENGTH)
    
    # 4. Split
    split = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 5. Train
    model = build_bidirectional_model((X_train.shape[1], X_train.shape[2]))
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # 6. Predict & Reconstruct
    pred_scaled = model.predict(X_test)
    
    dummy = np.zeros((len(pred_scaled), len(feature_cols)))
    dummy[:, target_idx] = pred_scaled.flatten()
    pred_returns = scaler.inverse_transform(dummy)[:, target_idx]
    
    # Reconstruct Prices
    test_indices = df.index[split + SEQ_LENGTH:]
    prev_prices = df['Gold_Price'].iloc[split + SEQ_LENGTH - 1 : -1].values
    predicted_prices = prev_prices * np.exp(pred_returns / 100)
    actual_prices = df['Gold_Price'].iloc[split + SEQ_LENGTH:].values
    
    # 7. Evaluate
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mae = mean_absolute_error(actual_prices, predicted_prices)
    
    print("\n" + "="*40)
    print("FUSION MODEL RESULTS")
    print("="*40)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    
    plt.figure(figsize=(14, 7))
    plt.plot(test_indices, actual_prices, label='Actual Price', color='blue', alpha=0.6)
    plt.plot(test_indices, predicted_prices, label='Fusion Forecast', color='purple', linestyle='--', alpha=0.9)
    plt.title(f'Fusion Strategy Forecast (RMSE: {rmse:.2f})')
    plt.legend()
    plt.grid(True)
    plt.savefig('fusion_forecast.png')
    print("Saved fusion_forecast.png")

if __name__ == "__main__":
    main()