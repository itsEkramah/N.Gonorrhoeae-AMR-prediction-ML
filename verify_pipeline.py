import sys
import os
sys.path.append(os.getcwd())

from src.data_processing import load_data, preprocess_data, get_train_test_split
from src.models import get_models, evaluate_models_cv
import pandas as pd

def test_pipeline():
    print("Testing pipeline...")
    try:
        # Load Data
        print("Loading data...")
        # Use Ciprofloxacin as it has the largest file, good stress test, or maybe Azithromycin for speed?
        # Let's use Azithromycin (azm)
        raw_data, code = load_data('Azithromycin', 'DATA')
        print(f"Data loaded. Shape: {raw_data.shape}")
        if raw_data.empty:
            print("Raw data is empty! Checking merge...")
            # Debugging load_data logic manually here or assume load_data needs fix
            # Let's just print what we can
            pass
        
        # Preprocess
        print("Preprocessing...")
        target_col = f'{code}_sr'
        print(f"Target column: {target_col}")
        if target_col not in raw_data.columns:
             print(f"Target column {target_col} not found in data columns: {raw_data.columns}")

        X, y = preprocess_data(raw_data, target_col)
        print(f"Preprocessed. X shape: {X.shape}, y shape: {y.shape}")
        
        # Split
        print("Splitting...")
        X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=0.2)
        
        # Models
        print("Initializing models...")
        models = get_models()
        
        # Evaluate CV (use subset for speed if needed, but 3k samples is fast)
        print("Evaluating CV...")
        # Reduce max_iter for LR to speed up if needed, or just run
        results = evaluate_models_cv(models, X_train, y_train)
        print("CV Results:")
        print(results)
        
        print("Pipeline verification successful!")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
