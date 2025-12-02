#!/usr/bin/env python3
"""
Verification Script - Test that all imports and paths work correctly
Run this BEFORE running the full pipeline to catch any issues
"""

import sys
from pathlib import Path

def test_directory_structure():
    """Verify folder structure is correct"""
    print("=" * 80)
    print("VERIFICATION: Checking Directory Structure")
    print("=" * 80)
    
    base_dir = Path(__file__).resolve().parent.parent
    print(f"\nBase directory: {base_dir}")
    
    required_paths = [
        base_dir / "ym1na" / "data" / "SAHMREALTIME.csv",
        base_dir / "ym1na" / "data" / "XAU_USD Historical Data.csv",
        base_dir / "NewThings" / "data_prep_lstm_garch.py",
        base_dir / "NewThings" / "lstm_garch_model.py",
        base_dir / "NewThings" / "visualize_results.py",
        base_dir / "requirements.txt",
    ]
    
    all_exist = True
    for path in required_paths:
        exists = path.exists()
        status = "[OK]" if exists else "[ERROR]"
        print(f"{status} {path.relative_to(base_dir)}")
        if not exists:
            all_exist = False
    
    return all_exist

def test_imports():
    """Test that all imports work"""
    print("\n" + "=" * 80)
    print("VERIFICATION: Testing Imports")
    print("=" * 80)
    
    try:
        print("\n[1] Importing standard libraries...")
        import numpy as np
        import pandas as pd
        print("[OK] NumPy and Pandas imported successfully")
        
        print("\n[2] Importing ML libraries...")
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        print("[OK] Scikit-learn imported successfully")
        
        print("\n[3] Importing deep learning libraries...")
        import tensorflow as tf
        from tensorflow.keras.models import Model
        print("[OK] TensorFlow/Keras imported successfully")
        
        print("\n[4] Importing statsmodels...")
        from arch import arch_model
        print("[OK] ARCH/GARCH imported successfully")
        
        print("\n[5] Importing local modules...")
        from data_prep_lstm_garch import prepare_data
        print("[OK] data_prep_lstm_garch imported successfully")
        
        from lstm_garch_model import train_and_evaluate_models
        print("[OK] lstm_garch_model imported successfully")
        
        from visualize_results import plot_model_predictions
        print("[OK] visualize_results imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"\n[ERROR] Import error: {e}")
        print("\nTo fix, run:")
        print("  pip install -r requirements.txt")
        return False

def test_data_loading():
    """Test that data files can be loaded"""
    print("\n" + "=" * 80)
    print("VERIFICATION: Testing Data Loading")
    print("=" * 80)
    
    try:
        from data_prep_lstm_garch import load_gold_prices, load_sahm_indicator
        
        print("\n[1] Loading SAHM data...")
        sahm = load_sahm_indicator()
        print(f"[OK] Loaded SAHM data: {len(sahm)} records")
        
        print("\n[2] Loading gold prices...")
        gold = load_gold_prices()
        print(f"[OK] Loaded gold data: {len(gold)} records")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Data loading error: {e}")
        return False

def main():
    """Run all verification tests"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "LSTM-GARCH PROJECT VERIFICATION" + " " * 27 + "║")
    print("╚" + "=" * 78 + "╝")
    
    results = []
    
    # Test 1: Directory structure
    try:
        results.append(("Directory Structure", test_directory_structure()))
    except Exception as e:
        print(f"[ERROR] Error in directory check: {e}")
        results.append(("Directory Structure", False))
    
    # Test 2: Imports
    try:
        results.append(("Module Imports", test_imports()))
    except Exception as e:
        print(f"[ERROR] Error in import test: {e}")
        results.append(("Module Imports", False))
    
    # Test 3: Data loading
    try:
        results.append(("Data Loading", test_data_loading()))
    except Exception as e:
        print(f"[ERROR] Error in data loading test: {e}")
        results.append(("Data Loading", False))
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "[OK] PASS" if passed else "[ERROR] FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("[OK] ALL CHECKS PASSED - Ready to run full pipeline!")
        print("\nNext step: python run_full_pipeline.py")
        return 0
    else:
        print("[ERROR] SOME CHECKS FAILED - Please fix issues above")
        print("\nCommon fixes:")
        print("  1. Check CSV files are in ym1na/data/ folder")
        print("  2. Run: pip install -r requirements.txt")
        print("  3. Ensure you're in the NewThings/ directory")
        return 1

if __name__ == "__main__":
    sys.exit(main())
