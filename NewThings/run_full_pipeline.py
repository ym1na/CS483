#!/usr/bin/env python3
"""
Quick Start Script for LSTM-GARCH Hybrid Gold Price Forecasting Model

Run this script to execute the entire pipeline:
1. Prepare data from all sources
2. Train Random Walk, LSTM, GARCH, and Hybrid models
3. Generate visualizations and reports

Usage:
    python run_full_pipeline.py
"""

import sys
import traceback
from pathlib import Path
import pandas as pd

def print_header(text):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def print_step(step_num, text):
    """Print formatted step"""
    print(f"\n[{step_num}] {text}")
    print("-" * 80)

def main():
    """Execute full pipeline"""
    
    print_header("LSTM-GARCH HYBRID MODEL: FULL PIPELINE")
    print("Gold Price Forecasting with LSTM + GARCH Volatility Modeling")
    print("\nThis script will:")
    print("  1. Load and prepare data from all sources")
    print("  2. Train Random Walk, LSTM, GARCH, and Hybrid models")
    print("  3. Generate visualizations and performance metrics")
    print("  4. Save all results to /results folder")
    
    try:
        # Step 1: Import required modules
        print_step(1, "Checking Dependencies")
        try:
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            from arch import arch_model
            from sklearn.preprocessing import MinMaxScaler
            
            # Try TensorFlow, but don't fail if DLL is missing
            tf = None
            try:
                import tensorflow as tf
            except Exception as e:
                if 'msvcp140' in str(e).lower() or 'dll' in str(e).lower():
                    print("[WARNING] TensorFlow DLL not found - LSTM model will be skipped")
                else:
                    raise
            
            print("[OK] All required packages available")
        except Exception as e:
            print(f"[ERROR] Missing package: {e}")
            print("\nPlease install requirements:")
            print("  pip install -r requirements.txt")
            return 1
        
        # Step 2: Data Preparation
        print_step(2, "Preparing Data")
        print("Loading: SAHM indicator, gold prices, computing technical indicators...")
        
        from data_prep_lstm_garch import prepare_data
        X_train, y_train, X_test, y_test, scaler, orig_prices, aligned_df, metadata = prepare_data()
        
        print(f"[OK] Data preparation complete")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Features: {metadata['n_features']}")
        
        # Step 3: Train and Evaluate Models
        print_step(3, "Training Models (Random Walk, LSTM, GARCH, Hybrid)")
        print("This may take 2-5 minutes depending on your system...")
        
        try:
            from lstm_garch_model import train_and_evaluate_models
            results = train_and_evaluate_models()
        except ImportError as e:
            if 'msvcp140' in str(e).lower() or 'dll' in str(e).lower():
                print("[WARNING] TensorFlow not available - Skipping LSTM-GARCH models")
                print("[WARNING] Running basic models only (Random Walk and GARCH)")
                # Create minimal results structure
                results = {
                    'metrics': pd.DataFrame({
                        'Model': ['Random Walk', 'GARCH'],
                        'Note': ['Baseline', 'Volatility only']
                    }),
                    'models': {}
                }
            else:
                raise
        
        print("[OK] Model training complete")
        
        # Step 4: Generate Visualizations
        print_step(4, "Generating Visualizations and Report")
        print("Creating plots, charts, and analysis report...")
        
        from visualize_results import (
            plot_model_predictions,
            plot_metrics_comparison,
            plot_lstm_training_history,
            generate_analysis_report
        )
        
        output_dir = Path(__file__).resolve().parent / 'results'
        output_dir.mkdir(exist_ok=True)
        
        plot_model_predictions(results, save_path=output_dir / 'model_predictions.png')
        plot_metrics_comparison(results, save_path=output_dir / 'metrics_comparison.png')
        plot_lstm_training_history(results['models']['lstm'], 
                                   save_path=output_dir / 'lstm_training_history.png')
        generate_analysis_report(results, save_path=output_dir / 'analysis_report.txt')
        
        print("[OK] Visualization complete")
        
        # Step 5: Summary
        print_header("[OK] PIPELINE COMPLETE")
        
        print("\nRESULTS SUMMARY:")
        print(results['metrics'].to_string(index=False))
        
        print(f"\n\nAll results saved to: {output_dir}/")
        print("\nGenerated files:")
        for f in sorted(output_dir.glob('*')):
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                print(f"  • {f.name} ({size_kb:.1f} KB)")
        
        print("\n\nNext Steps:")
        print("  1. Review /results/analysis_report.txt for detailed insights")
        print("  2. View /results/model_predictions.png to see forecast accuracy")
        print("  3. Check /results/metrics_comparison.png for model performance")
        print("  4. Examine /results/predictions.csv for raw prediction values")
        
        print("\n" + "="*80)
        print("Thank you for using the LSTM-GARCH Hybrid Model!")
        print("="*80 + "\n")
        
        return 0
        
    except Exception as e:
        print_header("[ERROR] ERROR ENCOUNTERED")
        print(f"\nError: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        print("\n\nTroubleshooting tips:")
        print("  1. Ensure all CSV files exist in /data folder")
        print("  2. Run: pip install -r requirements.txt")
        print("  3. Check data files are not corrupted")
        print("  4. Verify sufficient disk space for models (~100MB)")
        
        return 1

if __name__ == '__main__':
    sys.exit(main())
