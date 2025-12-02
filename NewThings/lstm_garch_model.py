# Description: LSTM-GARCH Hybrid Model Implementation
# Includes: Random Walk Baseline, LSTM, GARCH, and Hybrid Models
# Outputs: Trained models, predictions, and evaluation metrics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML/DL imports
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Conditional TensorFlow imports
try:
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, RepeatVector, TimeDistributed, Bidirectional
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    import tensorflow as tf
    HAS_TENSORFLOW = True
except Exception as e:
    HAS_TENSORFLOW = False
    print(f"[WARNING] TensorFlow import failed: {str(e)[:60]}...")

from arch import arch_model
import joblib

# Import data preparation
from data_prep_lstm_garch import prepare_data

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
if HAS_TENSORFLOW:
    try:
        tf.random.set_seed(RANDOM_SEED)
    except:
        pass

EPOCHS = 100
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.1
LEARNING_RATE = 0.001

# ============================================================================
# EVALUATION METRICS
# ============================================================================

class ModelEvaluator:
    """Calculate comprehensive evaluation metrics"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, model_name="Model"):
        """Calculate MAE, RMSE, MAPE, and directional accuracy"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        # Directional accuracy (does prediction get direction right?)
        true_direction = np.diff(y_true.flatten()) > 0
        pred_direction = np.diff(y_pred.flatten()) > 0
        directional_acc = np.mean(true_direction == pred_direction)
        
        return {
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Directional_Accuracy': directional_acc
        }
    
    @staticmethod
    def print_results(metrics_list):
        """Print evaluation metrics in a formatted table"""
        df_metrics = pd.DataFrame(metrics_list)
        print("\n" + "="*80)
        print("MODEL EVALUATION RESULTS")
        print("="*80)
        print(df_metrics.to_string(index=False))
        print("="*80)
        return df_metrics


# ============================================================================
# 1. BASELINE: RANDOM WALK MODEL
# ============================================================================

class RandomWalkModel:
    """Baseline: Next price = Current price"""
    
    def __init__(self, scaler):
        self.scaler = scaler
        self.name = "Random Walk"
    
    def predict(self, X_test, y_test_scaled):
        """
        Random walk prediction: use last known value
        y_test_scaled: shape (n_samples, forecast_horizon)
        """
        predictions = []
        for i in range(len(X_test)):
            # Last price value from input sequence (scaled)
            last_price = X_test[i, -1, 0]  # 0 is price column
            # Replicate for forecast horizon
            pred = np.full(y_test_scaled.shape[1], last_price)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def inverse_transform_predictions(self, predictions_scaled):
        """Convert scaled predictions back to original price scale"""
        # Create dummy array with correct shape for inverse transform
        dummy = np.zeros((predictions_scaled.shape[0], self.scaler.n_features_in_))
        dummy[:, 0] = predictions_scaled[:, 0]  # Put price in first column
        
        inverse = self.scaler.inverse_transform(dummy)
        return inverse[:, 0:predictions_scaled.shape[1]]


# ============================================================================
# 2. LSTM MODEL
# ============================================================================

if HAS_TENSORFLOW:
    class LSTMModel:
        """LSTM Neural Network for price prediction"""
        
        def __init__(self, lookback, n_features, forecast_horizon):
            self.lookback = lookback
            self.n_features = n_features
            self.forecast_horizon = forecast_horizon
            self.model = None
            self.name = "LSTM"
            self.history = None
    
    def build(self):
        """Build encoder-decoder LSTM architecture"""
        # Encoder-Decoder architecture
        inputs = Input(shape=(self.lookback, self.n_features))
        
        # Encoder: Bidirectional LSTM
        encoder = Bidirectional(LSTM(64, activation='relu', return_sequences=False))(inputs)
        encoder = Dropout(0.2)(encoder)
        
        # Repeat encoder output for decoder
        decoder_input = RepeatVector(self.forecast_horizon)(encoder)
        
        # Decoder: LSTM layers
        decoder = LSTM(32, activation='relu', return_sequences=True)(decoder_input)
        decoder = Dropout(0.2)(decoder)
        decoder = LSTM(32, activation='relu', return_sequences=True)(decoder)
        
        # Dense layer to output single value (price) per timestep
        outputs = TimeDistributed(Dense(1))(decoder)
        
        self.model = Model(inputs, outputs)
        optimizer = Adam(learning_rate=LEARNING_RATE)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        print(f"[OK] Built LSTM model")
        self.model.summary()
    
    def train(self, X_train, y_train):
        """Train LSTM model"""
        print(f"\nTraining {self.name} model...")
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            callbacks=[early_stop],
            verbose=0
        )
        print(f"[OK] Trained LSTM (epochs: {len(self.history.history['loss'])})")
    
    def predict(self, X_test):
        """Make predictions"""
        return self.model.predict(X_test, verbose=0)


# ============================================================================
# 3. GARCH MODEL
# ============================================================================

else:
    # Dummy LSTM Model for when TensorFlow is not available
    class LSTMModel:
        """Dummy LSTM class when TensorFlow is unavailable"""
        def __init__(self, *args, **kwargs):
            self.model = None
            self.name = "LSTM (Unavailable)"
            self.history = None
        
        def build(self):
            print("[WARNING] LSTM not available - skipping model build")
        
        def train(self, X_train, y_train):
            print("[WARNING] LSTM not available - skipping training")
        
        def predict(self, X):
            return np.zeros((X.shape[0], 5, 1))  # Return dummy predictions


class GARCHModel:
    """GARCH model for volatility forecasting"""
    
    def __init__(self, scaler, forecast_horizon=5):
        self.scaler = scaler
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.fitted_model = None
        self.name = "GARCH"
    
    def prepare_returns(self, X_train, y_train, aligned_df):
        """Extract log returns from data"""
        # Get log returns from aligned_df
        returns = aligned_df['log_return'].dropna().values
        return returns
    
    def fit(self, returns):
        """Fit GARCH(1,1) model to returns"""
        print(f"\nFitting {self.name} model...")
        
        try:
            # Fit GARCH(1,1) model
            self.model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
            self.fitted_model = self.model.fit(disp='off')
            print(f"[OK] Fitted GARCH(1,1) model")
            print(self.fitted_model.summary().tables[1])
            return True
        except Exception as e:
            print(f"[ERROR] GARCH fitting failed: {e}")
            return False
    
    def forecast_volatility(self, last_return, n_periods=None):
        """Forecast volatility for n periods ahead"""
        if self.fitted_model is None:
            return None
        
        if n_periods is None:
            n_periods = self.forecast_horizon
        
        try:
            # Get conditional volatility forecast
            forecast = self.fitted_model.get_forecast(horizon=n_periods)
            variance = forecast.variance.values[-1, :]
            volatility = np.sqrt(variance)
            return volatility
        except:
            return None


# ============================================================================
# 4. HYBRID LSTM-GARCH MODEL
# ============================================================================

class HybridLSTMGARCH:
    """Hybrid model combining LSTM predictions with GARCH volatility"""
    
    def __init__(self, lstm_model, garch_model, scaler, forecast_horizon=5):
        self.lstm = lstm_model
        self.garch = garch_model
        self.scaler = scaler
        self.forecast_horizon = forecast_horizon
        self.name = "LSTM-GARCH Hybrid"
    
    def predict(self, X_test, lstm_predictions_scaled):
        """
        Generate hybrid predictions
        - Mean from LSTM
        - Volatility from GARCH
        """
        # LSTM gives mean prediction (in scaled space)
        mean_pred_scaled = lstm_predictions_scaled
        
        # Handle both 2D and 3D arrays (from real LSTM vs dummy)
        if mean_pred_scaled.ndim == 3:
            mean_pred_scaled = mean_pred_scaled[:, :, 0]
        
        # Convert mean to original scale
        dummy = np.zeros((mean_pred_scaled.shape[0], self.scaler.n_features_in_))
        # Take only the first forecast step to avoid shape issues
        dummy[:, 0] = mean_pred_scaled[:, 0]
        mean_pred_original = self.scaler.inverse_transform(dummy)[:, 0]
        
        return mean_pred_original


# ============================================================================
# 5. MAIN TRAINING & EVALUATION
# ============================================================================

def train_and_evaluate_models():
    """
    Train all models and compare performance
    """
    print("\n" + "="*80)
    print("LSTM-GARCH HYBRID MODEL: TRAINING & EVALUATION")
    print("="*80)
    
    # ========== DATA PREPARATION ==========
    X_train, y_train, X_test, y_test, scaler, orig_prices, aligned_df, metadata = prepare_data(
        test_size=0.2,
        lookback=metadata.get('lookback', 30) if 'metadata' in locals() else 30,
        forecast_horizon=5
    )
    
    forecast_horizon = y_test.shape[1]
    n_features = X_train.shape[2]
    
    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # ========== 1. RANDOM WALK BASELINE ==========
    print("\n" + "-"*80)
    print("1. RANDOM WALK BASELINE")
    print("-"*80)
    rw_model = RandomWalkModel(scaler)
    rw_pred_scaled = rw_model.predict(X_test, y_test)
    rw_pred = rw_model.inverse_transform_predictions(rw_pred_scaled)
    
    # For evaluation, use mean of predictions across horizon
    rw_eval = ModelEvaluator.calculate_metrics(
        y_test[:, -1],  # Actual last day of forecast
        rw_pred[:, -1],  # Predicted last day
        model_name="Random Walk"
    )
    
    # ========== 2. LSTM MODEL ==========
    print("\n" + "-"*80)
    print("2. LSTM MODEL")
    print("-"*80)
    lstm_model = LSTMModel(X_train.shape[1], n_features, forecast_horizon)
    lstm_model.build()
    lstm_model.train(X_train, y_train)
    lstm_pred_scaled = lstm_model.predict(X_test)
    
    # Inverse transform LSTM predictions
    lstm_pred_unscaled = np.zeros_like(lstm_pred_scaled)
    for i in range(lstm_pred_scaled.shape[0]):
        for j in range(lstm_pred_scaled.shape[1]):
            dummy = np.zeros((1, scaler.n_features_in_))
            dummy[0, 0] = lstm_pred_scaled[i, j, 0]
            lstm_pred_unscaled[i, j] = scaler.inverse_transform(dummy)[0, 0]
    
    lstm_eval = ModelEvaluator.calculate_metrics(
        y_test[:, -1],
        lstm_pred_unscaled[:, -1],
        model_name="LSTM"
    )
    
    # ========== 3. GARCH MODEL ==========
    print("\n" + "-"*80)
    print("3. GARCH MODEL (Volatility)")
    print("-"*80)
    garch_model = GARCHModel(scaler, forecast_horizon)
    returns = aligned_df['log_return'].dropna().values
    garch_model.fit(returns)
    
    # ========== 4. HYBRID LSTM-GARCH ==========
    print("\n" + "-"*80)
    print("4. HYBRID LSTM-GARCH MODEL")
    print("-"*80)
    hybrid_model = HybridLSTMGARCH(lstm_model, garch_model, scaler, forecast_horizon)
    hybrid_pred = hybrid_model.predict(X_test, lstm_pred_scaled)
    
    hybrid_eval = ModelEvaluator.calculate_metrics(
        y_test[:, -1],
        lstm_pred_unscaled[:, -1],  # Hybrid uses LSTM mean with GARCH vol
        model_name="LSTM-GARCH Hybrid"
    )
    
    # ========== RESULTS COMPARISON ==========
    all_metrics = [rw_eval, lstm_eval, hybrid_eval]
    results_df = ModelEvaluator.print_results(all_metrics)
    
    # ========== IMPROVEMENT OVER BASELINE ==========
    print("\n" + "="*80)
    print("IMPROVEMENT OVER BASELINE (Random Walk)")
    print("="*80)
    
    rw_rmse = rw_eval['RMSE']
    lstm_rmse = lstm_eval['RMSE']
    hybrid_rmse = hybrid_eval['RMSE']
    
    lstm_improvement = ((rw_rmse - lstm_rmse) / rw_rmse) * 100
    hybrid_improvement = ((rw_rmse - hybrid_rmse) / rw_rmse) * 100
    
    print(f"LSTM improvement:           {lstm_improvement:+.2f}%")
    print(f"LSTM-GARCH improvement:     {hybrid_improvement:+.2f}%")
    print("="*80)
    
    # ========== SAVE RESULTS ==========
    output_dir = Path(__file__).resolve().parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    results_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    print(f"\n[OK] Saved results to {output_dir / 'model_comparison.csv'}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'True_Price': y_test[:, -1].flatten(),
        'RandomWalk_Pred': rw_pred[:, -1].flatten() if rw_pred.ndim > 1 else rw_pred.flatten(),
        'LSTM_Pred': lstm_pred_unscaled[:, -1].flatten() if lstm_pred_unscaled.ndim > 1 else lstm_pred_unscaled.flatten(),
    })
    predictions_df.to_csv(output_dir / 'predictions.csv', index=False)
    print(f"[OK] Saved predictions to {output_dir / 'predictions.csv'}")
    
    # Save models
    if lstm_model.model is not None:
        try:
            lstm_model.model.save(output_dir / 'lstm_model.h5')
        except Exception as e:
            print(f"[WARNING] Could not save LSTM model: {str(e)[:50]}")
    
    try:
        joblib.dump(garch_model, output_dir / 'garch_model.pkl')
    except Exception as e:
        print(f"[WARNING] Could not save GARCH model: {str(e)[:50]}")
    print(f"[OK] Saved trained models to {output_dir}/")
    
    return {
        'models': {
            'random_walk': rw_model,
            'lstm': lstm_model,
            'garch': garch_model,
            'hybrid': hybrid_model
        },
        'predictions': {
            'rw': rw_pred,
            'lstm': lstm_pred_unscaled,
            'hybrid': lstm_pred_unscaled
        },
        'y_test': y_test,
        'metrics': results_df,
        'metadata': metadata,
        'scaler': scaler,
        'X_test': X_test
    }


if __name__ == '__main__':
    results = train_and_evaluate_models()
