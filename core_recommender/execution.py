import pandas as pd
import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

# --- MODEL IMPORTS ---
from core_recommender.modeling.baseModel import BaseModel
from core_recommender.modeling.knn import KNNModel
from core_recommender.modeling.linearRegression import LinearRegressionModel
from core_recommender.modeling.logisticRegression import LogisticRegressionModel
# Future imports:
from core_recommender.modeling.randomForest import RandomForestModel
from core_recommender.modeling.decisionTrees import DecisionTreeModel
from core_recommender.modeling.svms import SVMModel
from core_recommender.modeling.naiveBayes import NaiveBayesModel

# =========================================================================
# Execution Engine
# =========================================================================

class ModelExecutor:
    """
    The orchestrator that manages the end-to-end model selection pipeline.
    
    Responsibilities:
    1. Detect Task Type (Regression vs Classification)
    2. Split Data
    3. Instantiate Models
    4. Train Models Concurrently
    5. Rank and Select Best Model
    6. Generate Diagnostics
    """
    
    def __init__(self, task_type: str = 'auto', n_jobs: int = -1, random_state: int = 42):
        """
        Args:
            task_type: 'classification', 'regression', or 'auto' (inferred from target).
            n_jobs: Number of parallel jobs for training models (-1 for all cores).
            random_state: Seed for reproducibility.
        """
        self.task_type = task_type
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.results = []
        self.best_model_name = None
        self.best_model_metrics = None
        self.best_model_instance = None

    def _infer_task_type(self, y: pd.Series) -> str:
        """Infers classification or regression based on target variable properties."""
        if self.task_type != 'auto':
            return self.task_type
            
        # Heuristic: 
        # If float and many unique values -> Regression
        # If object/string/bool/category -> Classification
        # If int and few unique values (<20) -> Classification
        # If int and many unique values -> Regression (Assumed, but ambiguous)
        
        if pd.api.types.is_float_dtype(y):
            return 'regression'
        elif pd.api.types.is_object_dtype(y) or pd.api.types.is_bool_dtype(y) or pd.api.types.is_categorical_dtype(y):
            return 'classification'
        elif pd.api.types.is_integer_dtype(y):
            if y.nunique() < 20:
                print(f"Target has {y.nunique()} unique integers. Inferring CLASSIFICATION.")
                return 'classification'
            else:
                print(f"Target has {y.nunique()} unique integers. Inferring REGRESSION.")
                return 'regression'
        
        # Default fallback
        return 'regression'

    def _get_candidate_models(self, task_type: str) -> List[BaseModel]:
        """Instantiates the list of models appropriate for the task."""
        models = []
        
        # 1. KNN (Universal)
        models.append(KNNModel(is_classification=(task_type == 'classification')))
        
        if task_type == 'classification':
            # 2. Logistic Regression
            models.append(LogisticRegressionModel())
            
            # Future Classification Models:
            models.append(RandomForestModel(is_classification=True))
            models.append(DecisionTreeModel(is_classification=True))
            models.append(SVMModel(is_classification=True))
            models.append(NaiveBayesModel())
            
        elif task_type == 'regression':
            # 2. Linear Regression
            models.append(LinearRegressionModel())
            
            # Future Regression Models:
            models.append(RandomForestModel(is_classification=False))
            models.append(DecisionTreeModel(is_classification=False))
            models.append(SVMModel(is_classification=False))
            
        return models

    def _train_single_model(self, model: BaseModel, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
        """
        Worker function to train and evaluate a single model.
        Returns a dictionary with results.
        """
        try:
            print(f"[{model.name}] Training started...")
            
            # 1. Preprocess
            # Note: We pass raw data to preprocess(), which returns transformed numpy arrays
            # BUT, in our architecture, data splitting happens on the raw DF usually?
            # Wait, our BaseModel.preprocess takes DataFrame.
            # So X_train should be DataFrame here.
            
            X_train_proc, y_train_proc, preprocessor = model.preprocess(X_train, y_train)
            
            # 2. Fit (includes GridSearch)
            model.fit(X_train_proc, y_train_proc)
            
            # 3. Evaluate
            # Need to transform X_test using the fitted preprocessor
            # y_test needs to be transformed if label encoding happened (for classification)
            
            # Transform Test Data
            X_test_proc = preprocessor.transform(X_test)
            X_test_proc = np.asarray(X_test_proc)
            
            # Transform y_test if model has label encoder (e.g., LogisticRegression)
            # Transform y_test if model has label encoder (e.g., LogisticRegression)
            if hasattr(model, 'label_encoder') and model.label_encoder is not None:
                y_test_proc = model.label_encoder.transform(y_test)
            elif isinstance(y_test, (pd.Series, pd.DataFrame)):
                 y_test_proc = y_test.values
            else:
                 y_test_proc = y_test

            # Ensure y_test_proc is numpy array
            y_test_proc = np.asarray(y_test_proc)

            metrics = model.calculate_metrics(X_test_proc, y_test_proc)
            
            print(f"[{model.name}] Completed successfully.")
            return {
                'model_name': model.name,
                'status': 'success',
                'metrics': metrics,
                'model_instance': model,
                'preprocessor': preprocessor,
                'test_data_proc': (X_test_proc, y_test_proc) # Store for final diagnostics
            }
            
        except Exception as e:
            print(f"[{model.name}] FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'model_name': model.name,
                'status': 'failed',
                'error': str(e)
            }

    def run(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Main execution point.
        """
        start_time = time.time()
        
        # 1. Basic Validation
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
            
        # 2. Separate Features and Target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # 3. Infer Task
        inferred_task = self._infer_task_type(y)
        print(f"Task inferred as: {inferred_task.upper()}")
        
        # 4. Train/Test Split
        # Stratify if classification
        stratify = y if inferred_task == 'classification' else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=self.random_state,
            stratify=stratify
        )
        
        # 5. Initialize Models
        models = self._get_candidate_models(inferred_task)
        print(f"Selected {len(models)} models for training: {[m.name for m in models]}")
        
        # 6. Concurrent Training
        # We use joblib.Parallel to run _train_single_model
        # Note: Models are complex objects. Joblib pickling works usually, but debugging can be hard.
        # Ensure n_jobs is managed.
        
        results_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._train_single_model)(model, X_train, y_train, X_test, y_test) 
            for model in models
        )
        
        self.results = results_list
        
        # 7. Model Selection (Ranking)
        valid_results = [r for r in results_list if r['status'] == 'success']
        if not valid_results:
            raise RuntimeError("All models failed validation.")
            
        if inferred_task == 'classification':
            # Rank by F1 Score (Weighted) - Higher is better
            primary_metric = 'F1 Score'
            best_run = max(valid_results, key=lambda x: x['metrics'].get('F1 Score', 0)) # Look for F1 Score or F1 Score (Weighted)
             # Adjust key match slightly if naming varies
            if 'F1 Score' not in best_run['metrics']:
                 # Try matching fuzzy or specific keys
                 # Check first result to see keys
                 sample_keys = valid_results[0]['metrics'].keys()
                 # Find key containing 'F1'
                 f1_key = next((k for k in sample_keys if 'F1' in k), 'Accuracy')
                 primary_metric = f1_key
                 best_run = max(valid_results, key=lambda x: x['metrics'].get(f1_key, 0))
        else:
            # Rank by RMSE - Lower is better
            primary_metric = 'RMSE'
            best_run = min(valid_results, key=lambda x: x['metrics'].get('RMSE', float('inf')))

        self.best_model_name = best_run['model_name']
        self.best_model_metrics = best_run['metrics']
        self.best_model_instance = best_run['model_instance']
        
        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"   {primary_metric}: {self.best_model_metrics.get(primary_metric)}")
        
        # 8. Generate Diagnostics for Best Model
        # We need to call get_diagnostic_data on the best model instance
        # We pass the PROCESSED test data stored in the result
        X_test_proc_best, y_test_proc_best = best_run['test_data_proc']
        
        diagnostics = self.best_model_instance.get_diagnostic_data(X_test_proc_best, y_test_proc_best)
        
        # 9. Return Final Summary
        summary = {
            'task_type': inferred_task,
            'total_time': time.time() - start_time,
            'leaderboard': [
                {
                    'model': r['model_name'], 
                    'metrics': r['metrics'],
                    'status': r['status'],
                    'error': r.get('error')
                } 
                for r in self.results
            ],
            'best_model': {
                'name': self.best_model_name,
                'metrics': self.best_model_metrics,
                'diagnostics': diagnostics
            }
        }
        
        return summary
