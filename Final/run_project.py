import os
import time

def run_script(script_name):
    print(f"\n{'='*50}")
    print(f">>> RUNNING: {script_name}")
    print(f"{'='*50}")
    result = os.system(f"python {script_name}")
    if result != 0:
        print(f"ERROR: {script_name} failed. Stopping execution.")
        exit(1)
    time.sleep(1) # Pause briefly for readability

def main():
    print("--- STARTING FINAL PROJECT PIPELINE ---")
    
    # 1. Data Collection & Cleaning (Fetches 2000-2019 Data)
    run_script("01_data_pipeline.py")
    
    # 2. Establish the Baseline (RMSE 9.34)
    run_script("02_baseline_random_walk.py")
    
    # 3. Statistical Diagnosis (The "Why" - Colleague's Findings)
    run_script("02_2_statistical_diagnosis.py")
    
    # 4. Feature Engineering (Adds Disasters, Sentiment, Regimes)
    run_script("03_feature_engineering.py")
    
    # 5. The Winning Model (Regime-Aware Gradient Boosting - RMSE 8.45)
    # Note: We skip the intermediate experiments (04-08) and run the Champion.
    run_script("09_final_regime_boost.py")
    
    print("\n" + "="*50)
    print("PROJECT COMPLETE.")
    print("All results and charts have been saved to this folder.")
    print("="*50)

if __name__ == "__main__":
    main()