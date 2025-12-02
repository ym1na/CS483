# Description: Visualization & Analysis of Model Predictions
# Creates comprehensive comparison plots between models
# Generates detailed analysis reports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from lstm_garch_model import train_and_evaluate_models
from data_prep_lstm_garch import prepare_data

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_model_predictions(results, save_path=None):
    """Plot actual vs predicted prices for all models"""
    try:
        y_test = results['y_test']
        predictions = results['predictions']
        
        # Handle empty predictions (when LSTM not available)
        if not predictions or len(predictions.get('lstm', [])) == 0:
            print("[WARNING] No predictions available - skipping plot_model_predictions")
            return None
        
        # Take last day of forecast horizon for clarity
        actual = y_test[:, -1]
        rw_pred = predictions['rw']
        lstm_pred = predictions['lstm']
        hybrid_pred = predictions['hybrid']
        
        # Flatten if needed
        if rw_pred.ndim > 1:
            rw_pred = rw_pred[:, -1] if rw_pred.shape[1] > 1 else rw_pred.flatten()
        if lstm_pred.ndim > 1:
            lstm_pred = lstm_pred[:, -1] if lstm_pred.shape[1] > 1 else lstm_pred.flatten()
        if hybrid_pred.ndim > 1:
            hybrid_pred = hybrid_pred[:, -1] if hybrid_pred.shape[1] > 1 else hybrid_pred.flatten()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # Time index for x-axis
        time_idx = np.arange(len(actual))
        
        # Plot 1: All predictions overlaid
        ax = axes[0, 0]
        ax.plot(time_idx, actual, 'o-', linewidth=2, markersize=4, label='Actual', color='black')
        ax.plot(time_idx, rw_pred, 's-', linewidth=1.5, markersize=3, label='Random Walk', alpha=0.7)
        ax.plot(time_idx, lstm_pred, '^-', linewidth=1.5, markersize=3, label='LSTM', alpha=0.7)
        ax.plot(time_idx, hybrid_pred, 'd-', linewidth=1.5, markersize=3, label='LSTM-GARCH', alpha=0.7)
        ax.set_title('All Models: Actual vs Predicted Gold Prices', fontweight='bold', fontsize=12)
        ax.set_xlabel('Test Sample Index')
        ax.set_ylabel('Gold Price (USD/oz)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Prediction errors
        ax = axes[0, 1]
        rw_error = np.abs(actual - rw_pred)
        lstm_error = np.abs(actual - lstm_pred)
        hybrid_error = np.abs(actual - hybrid_pred)
        
        ax.plot(time_idx, rw_error, 's-', linewidth=1.5, markersize=3, label='Random Walk', alpha=0.7)
        ax.plot(time_idx, lstm_error, '^-', linewidth=1.5, markersize=3, label='LSTM', alpha=0.7)
        ax.plot(time_idx, hybrid_error, 'd-', linewidth=1.5, markersize=3, label='LSTM-GARCH', alpha=0.7)
        ax.set_title('Absolute Prediction Errors', fontweight='bold', fontsize=12)
        ax.set_xlabel('Test Sample Index')
        ax.set_ylabel('Absolute Error (USD/oz)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Error distribution (box plot)
        ax = axes[1, 0]
        error_data = [rw_error, lstm_error, hybrid_error]
        bp = ax.boxplot(error_data, labels=['Random Walk', 'LSTM', 'LSTM-GARCH'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightcoral', 'lightblue', 'lightgreen']):
            patch.set_facecolor(color)
        ax.set_title('Error Distribution Comparison', fontweight='bold', fontsize=12)
        ax.set_ylabel('Absolute Error (USD/oz)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Residuals (error vs actual)
        ax = axes[1, 1]
        ax.scatter(actual, lstm_error, alpha=0.6, s=50, label='LSTM', color='blue')
        ax.scatter(actual, hybrid_error, alpha=0.6, s=50, label='LSTM-GARCH', color='green')
        ax.set_title('Residuals vs Actual Price', fontweight='bold', fontsize=12)
        ax.set_xlabel('Actual Gold Price (USD/oz)')
        ax.set_ylabel('Absolute Error (USD/oz)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Saved prediction plot to {save_path}")
        
        return fig
    
    except Exception as e:
        print(f"[WARNING] Could not generate prediction plot: {str(e)[:60]}")
        return None


def plot_metrics_comparison(results, save_path=None):
    """Plot model metrics comparison"""
    metrics_df = results['metrics']
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    metrics = ['MAE', 'RMSE', 'MAPE', 'Directional_Accuracy']
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = metrics_df[metric].values
        models = metrics_df['Model'].values
        
        if metric == 'Directional_Accuracy':
            # For accuracy, higher is better
            bars = ax.bar(models, values, color=colors)
            ax.set_ylim([0, 1])
        else:
            bars = ax.bar(models, values, color=colors)
        
        ax.set_title(f'{metric}', fontweight='bold', fontsize=12)
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved metrics plot to {save_path}")
    
    return fig


def plot_lstm_training_history(lstm_model, save_path=None):
    """Plot LSTM training history (loss curves)"""
    if lstm_model.history is None:
        print("[WARNING] No training history available")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax = axes[0]
    ax.plot(lstm_model.history.history['loss'], label='Training Loss', linewidth=2)
    ax.plot(lstm_model.history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax.set_title('LSTM Model: Loss During Training', fontweight='bold', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MAE
    ax = axes[1]
    ax.plot(lstm_model.history.history['mae'], label='Training MAE', linewidth=2)
    ax.plot(lstm_model.history.history['val_mae'], label='Validation MAE', linewidth=2)
    ax.set_title('LSTM Model: MAE During Training', fontweight='bold', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE (USD/oz)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved training history plot to {save_path}")
    
    return fig


def generate_analysis_report(results, save_path=None):
    """Generate detailed text analysis report"""
    metrics_df = results['metrics']
    metadata = results['metadata']
    
    report = []
    report.append("=" * 80)
    report.append("LSTM-GARCH HYBRID MODEL: ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Dataset info
    report.append("DATASET INFORMATION")
    report.append("-" * 80)
    report.append(f"Lookback Window:      {metadata['lookback']} days")
    report.append(f"Forecast Horizon:     {metadata['forecast_horizon']} days")
    report.append(f"Number of Features:   {metadata['n_features']}")
    report.append(f"Features Used:        {', '.join(metadata['feature_names'][:5])}...")
    report.append(f"Training Period:      {metadata['train_split_date']} to {metadata['test_start_date']}")
    report.append(f"Test Period:          {metadata['test_start_date']} to {metadata['test_end_date']}")
    report.append("")
    
    # Model performance
    report.append("MODEL PERFORMANCE METRICS")
    report.append("-" * 80)
    for idx, row in metrics_df.iterrows():
        report.append(f"\n{row['Model']}:")
        report.append(f"  MAE:                   ${row['MAE']:.4f}")
        report.append(f"  RMSE:                  ${row['RMSE']:.4f}")
        report.append(f"  MAPE:                  {row['MAPE']:.4f}%")
        report.append(f"  Directional Accuracy:  {row['Directional_Accuracy']:.4f}")
    
    report.append("\n" + "-" * 80)
    
    # Improvements
    report.append("\nIMPROVEMENT ANALYSIS")
    report.append("-" * 80)
    
    rw_rmse = metrics_df[metrics_df['Model'] == 'Random Walk']['RMSE'].values[0]
    lstm_rmse = metrics_df[metrics_df['Model'] == 'LSTM']['RMSE'].values[0]
    hybrid_rmse = metrics_df[metrics_df['Model'] == 'LSTM-GARCH Hybrid']['RMSE'].values[0]
    
    lstm_imp = ((rw_rmse - lstm_rmse) / rw_rmse) * 100
    hybrid_imp = ((rw_rmse - hybrid_rmse) / rw_rmse) * 100
    
    report.append(f"LSTM vs Random Walk:      {lstm_imp:+.2f}% RMSE improvement")
    report.append(f"LSTM-GARCH vs Random Walk: {hybrid_imp:+.2f}% RMSE improvement")
    
    if hybrid_imp > 0:
        report.append(f"\n[OK] LSTM-GARCH model OUTPERFORMS the Random Walk baseline by {hybrid_imp:.2f}%")
    else:
        report.append(f"\n[ERROR] LSTM-GARCH model underperforms the Random Walk baseline")
    
    report.append("\n" + "=" * 80)
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"[OK] Saved analysis report to {save_path}")
    
    print("\n" + report_text)
    
    return report_text


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run full pipeline: train models, generate visualizations, and report"""
    
    print("\n" + "="*80)
    print("STARTING FULL ANALYSIS PIPELINE")
    print("="*80)
    
    # Train and evaluate models
    results = train_and_evaluate_models()
    
    # Create output directory
    output_dir = Path(__file__).resolve().parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Generate plots
    plot_model_predictions(results, save_path=output_dir / 'model_predictions.png')
    plot_metrics_comparison(results, save_path=output_dir / 'metrics_comparison.png')
    plot_lstm_training_history(results['models']['lstm'], save_path=output_dir / 'lstm_training_history.png')
    
    # Generate report
    print("\n" + "="*80)
    print("GENERATING ANALYSIS REPORT")
    print("="*80)
    generate_analysis_report(results, save_path=output_dir / 'analysis_report.txt')
    
    print("\n" + "="*80)
    print("[OK] ANALYSIS COMPLETE")
    print("="*80)
    print(f"All results saved to: {output_dir}/")
    
    return results


if __name__ == '__main__':
    results = main()
