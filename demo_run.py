import pandas as pd
import numpy as np
import sys
import os

# Ensure we can import the package from current directory
sys.path.append(os.getcwd())

from core_recommender import ModelExecutor
from sklearn.datasets import make_classification, make_regression

def run_classification_demo():
    print("\n" + "="*50)
    print("🧪 TEST 1: CLASSIFICATION PIPELINE")
    print("="*50)
    
    # 1. Generate Synthetic Data
    print("Generating synthetic classification data...")
    X, y = make_classification(
        n_samples=200, 
        n_features=10, 
        n_informative=5, 
        n_redundant=2, 
        n_classes=2, 
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(10)])
    df['target_class'] = y
    
    # Add some categorical noise to test pipeline robustness
    df['cat_feature'] = np.random.choice(['A', 'B', 'C'], size=200)
    
    # 2. Initialize Executor
    print("Initializing ModelExecutor...")
    executor = ModelExecutor(task_type='classification', n_jobs=2) # n_jobs=2 to test parallelism
    
    # 3. Run
    print("Running execution engine...")
    try:
        results = executor.run(df, target_column='target_class')
        
        # 4. Report
        print("\n✅ Classification Run Completed!")
        print(f"Task Type: {results['task_type']}")
        print(f"Best Model: {results['best_model']['name']}")
        print(f"Best Metrics: {results['best_model']['metrics']}")
        
        print("\nLeaderboard:")
        for res in results['leaderboard']:
            status = res['status']
            metrics = res['metrics'] if status == 'success' else f"ERROR: {res['error']}"
            print(f" - {res['model']}: {status.upper()} | {metrics}")
            
    except Exception as e:
        print(f"\n❌ Classification Run Failed: {e}")
        import traceback
        traceback.print_exc()

def run_regression_demo():
    print("\n" + "="*50)
    print("🧪 TEST 2: REGRESSION PIPELINE")
    print("="*50)
    
    # 1. Generate Synthetic Data
    print("Generating synthetic regression data...")
    X, y = make_regression(
        n_samples=200, 
        n_features=10, 
        n_informative=5, 
        noise=0.1, 
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(10)])
    df['target_value'] = y
    
    # 2. Initialize Executor
    print("Initializing ModelExecutor...")
    executor = ModelExecutor(task_type='regression', n_jobs=2)
    
    # 3. Run
    print("Running execution engine...")
    try:
        results = executor.run(df, target_column='target_value')
        
        # 4. Report
        print("\n✅ Regression Run Completed!")
        print(f"Task Type: {results['task_type']}")
        print(f"Best Model: {results['best_model']['name']}")
        print(f"Best Metrics: {results['best_model']['metrics']}")
        
        print("\nLeaderboard:")
        for res in results['leaderboard']:
            status = res['status']
            metrics = res['metrics'] if status == 'success' else f"ERROR: {res['error']}"
            print(f" - {res['model']}: {status.upper()} | {metrics}")
            
    except Exception as e:
        print(f"\n❌ Regression Run Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 STARTING END-TO-END VERIFICATION")
    run_classification_demo()
    run_regression_demo()
    print("\n" + "="*50)
    print("🏁 VERIFICATION FINISHED")
