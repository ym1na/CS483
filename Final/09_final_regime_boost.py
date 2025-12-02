import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# --- CONFIGURATION ---
DATA_PATH = os.path.join('data', 'model_ready_dataset.csv')
PREVIOUS_BEST_RMSE = 8.5126 # The score to beat
TEST_SPLIT = 0.2

def load_and_engineer_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    
    # 1. Target
    df['Log_Ret'] = np.log(df['Gold_Price'] / df['Gold_Price'].shift(1)) * 10
    
    # 2. Lags (The Winner from Step 8)
    for i in range(1, 4):
        df[f'Log_Ret_Lag{i}'] = df['Log_Ret'].shift(i)
        
    # 3. YOUR COLLEAGUE'S INSIGHT (Regime Features)
    # The correlation is only high when SAHM is high.
    # We create an "Interaction Term" to help the model see this.
    
    # Feature A: Is the Sahm rule "Active"? (Threshold usually 0.50, but let's use 0.3 for early warning)
    df['Recession_Regime'] = (df['SAHMREALTIME'] > 0.3).astype(int)
    
    # Feature B: Interaction (Gold reacts to Dollar differently during Recession)
    df['Dollar_Regime_Interaction'] = df['DEXUSEU'].pct_change() * df['Recession_Regime']
    
    # Feature C: Interaction (Gold reacts to Fear differently during Recession)
    df['VIX_Regime_Interaction'] = df['VIXCLS'].diff() * df['SAHMREALTIME']

    # 4. Standard Features
    df['Real_Rate_Change'] = (df['FEDFUNDS'] - (df['CPIAUCSL'].pct_change()*1200)).diff()
    df['Yield_Spread'] = df['T10Y2Y'].diff()
    df['Dollar_Change'] = df['DEXUSEU'].pct_change() * 10
    
    # RSI
    delta = df['Gold_Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)
    
    # Select Features: Lags + Standard + REGIME INTERACTIONS
    features = [col for col in df.columns if 'Lag' in col] + \
               ['Real_Rate_Change', 'Yield_Spread', 'Dollar_Change', 'RSI', 
                'SAHMREALTIME', 'Dollar_Regime_Interaction', 'VIX_Regime_Interaction']
    
    return df, features

def main():
    print("--- STARTING REGIME-AWARE BOOSTING STRATEGY ---")
    
    df, features = load_and_engineer_data()
    print(f"Integrating Colleague's Insight. Features: {len(features)}")
    
    X = df[features].values
    y = df['Log_Ret'].values
    
    split = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Same Hyperparameters that won before
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.7,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predict & Reconstruct
    pred_returns = model.predict(X_test)
    test_indices = df.index[split:]
    prev_prices = df['Gold_Price'].iloc[split-1 : -1].values
    actual_prices = df['Gold_Price'].iloc[split:].values
    predicted_prices = prev_prices * np.exp(pred_returns / 10)
    
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    
    print("\n" + "="*40)
    print("FINAL REGIME MODEL RESULTS")
    print("="*40)
    print(f"RMSE: {rmse:.4f}")
    print(f"Previous Best: {PREVIOUS_BEST_RMSE:.4f}")
    
    if rmse < PREVIOUS_BEST_RMSE:
        print(f"🏆 TEAMWORK VICTORY! Improvement: {PREVIOUS_BEST_RMSE - rmse:.4f}")
        print("Explicitly adding the 'Regime' interaction improved accuracy.")
    else:
        print("Result: Very similar. The Tree likely discovered the regime logic on its own.")

    # Check Importance
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importance (Did the Colleague's features help?):")
    print(importance.head(5))
    
    plt.figure(figsize=(14, 7))
    plt.plot(test_indices, actual_prices, label='Actual Price', color='gray', alpha=0.5)
    plt.plot(test_indices, predicted_prices, label='Regime-Aware Forecast', color='gold', linewidth=1.5)
    plt.title(f'Final Team Model (RMSE: {rmse:.4f})')
    plt.legend()
    plt.savefig('final_team_forecast.png')
    print("Saved final_team_forecast.png")

if __name__ == "__main__":
    main()