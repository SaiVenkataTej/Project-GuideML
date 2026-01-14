import pandas as pd
import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
import traceback

# Import centralized logger
from core_recommender.logger import get_logger

logger = get_logger(__name__)

# --- MODEL IMPORTS ---
from core_recommender.modeling.baseModel import BaseModel
from core_recommender.modeling.knn import KNNModel
from core_recommender.modeling.linearRegression import LinearRegressionModel
from core_recommender.modeling.logisticRegression import LogisticRegressionModel
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
    
    Rationale:
    ----------
    - **Centralized Logic**: Encapsulates the complexity of model selection, data splitting, and evaluation in one place.
    - **Zero Leakage**: Strict separation of training and testing phases; feature engineering pipelines are fitted ONLY on training splits.
    - **Concurrency**: Manages parallel execution of multiple models to speed up the search process.

    Responsibilities:
    1. Detect Task Type (Regression vs Classification)
    2. Split Data (Train/Test)
    3. Instantiate Models (Based on Task Type)
    4. Train Models Concurrently (Parallel Processing)
    5. Rank and Select Best Model (Based on Metrics)
    6. Generate Diagnostics (Confusion Matrix, Plots, etc.)
    """
    
    def __init__(self, task_type: str = 'auto', n_jobs: int = -1, random_state: int = 42):
        """
        Initializes the ModelExecutor.

        Args:
            task_type: The type of machine learning task. Options:
                       - 'classification': Force classification mode.
                       - 'regression': Force regression mode.
                       - 'auto': Infer from the target variable (default).
            n_jobs: Number of parallel jobs for training models. 
                    Defaults to -1 (Use all available cores).
            random_state: Seed for reproducibility across splits and models. Defaults to 42.
        """
        self.task_type = task_type
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.results = []
        self.best_model_name = None
        self.best_model_metrics = None
        self.best_model_instance = None

    def _infer_task_type(self, y: pd.Series) -> str:
        """
        Infers whether the problem is 'classification' or 'regression' based on the target variable.

        Rationale:
        ----------
        - **Ambiguity Handling**: Heuristics differentiate between low-cardinality integers (likely classification) and high-cardinality integers (likely regression).
        - **Robustness**: Falls back to safe defaults if detection fails.

        Args:
            y: The target variable series.

        Returns:
            str: 'classification' or 'regression'.
        """
        if self.task_type != 'auto':
            return self.task_type
            
        # Heuristic: 
        # If float and many unique values -> Regression
        # If object/string/bool/category -> Classification
        # If int and few unique values (<20) -> Classification
        # If int and many unique values -> Regression (Assumed, but ambiguous)
        
        if pd.api.types.is_object_dtype(y) or pd.api.types.is_bool_dtype(y) or pd.api.types.is_categorical_dtype(y):
            # If target is string/object, it MUST be classification (unless we want to try NLP regression, out of scope)
            # Check if it looks like numbers stored as strings?
            try:
                pd.to_numeric(y, errors='raise')
                # If it converts, it might be regression, proceed to integer check
            except Exception:
                return 'classification'
            
        if pd.api.types.is_float_dtype(y):
            return 'regression'
        elif pd.api.types.is_integer_dtype(y) or (pd.api.types.is_object_dtype(y) and y.str.isnumeric().all()):
            # Treat numeric strings as integers for this check
            n_unique = y.nunique()
            if n_unique < 20:
                logger.info(f"Target has {n_unique} unique values. Inferring CLASSIFICATION.")
                return 'classification'
            else:
                logger.info(f"Target has {n_unique} unique values. Inferring REGRESSION.")
                return 'regression'
        
        # Default fallback
        return 'regression'

    def _detect_and_drop_leakage(self, df: pd.DataFrame, target_column: str, threshold: float = 0.95) -> pd.DataFrame:
        """
        Systemically identifies and removes "leaky" features that have near-perfect
        correlation or mathematical identity with the target.
        """
        if df.empty or target_column not in df.columns:
            return df
            
        logger.info("🔍 Automated Leakage Detection Check...")
        
        # 1. Correlation-based Leakage Detection (Numeric)
        numeric_df = df.select_dtypes(include=[np.number])
        if target_column not in numeric_df.columns:
             # Target is not numeric, skip corr check or handle differently
             return df
             
        corr_matrix = numeric_df.corr().abs()
        target_corr = corr_matrix[target_column].sort_values(ascending=False)
        
        leaky_features = []
        
        # We skip the target column itself (index 0 usually, corr=1.0)
        for feature, correlation in target_corr.items():
            if feature == target_column:
                continue
            if correlation > threshold:
                logger.warning(f"⚠️  Suspiciously high correlation: '{feature}' has {correlation:.4f} correlation with target")
                leaky_features.append(feature)

        # 2. Basic Mathematical Identity Check (e.g. Target = A * B)
        # We check pairs of numeric features
        candidates = [c for c in numeric_df.columns if c != target_column]
        for i, col1 in enumerate(candidates):
            for col2 in candidates[i+1:]:
                # Check for multiplication: Target ≈ col1 * col2
                # (Allowing for small epsilon due to floating point)
                prod = numeric_df[col1] * numeric_df[col2]
                if np.allclose(prod, numeric_df[target_column], rtol=1e-5, atol=1e-8):
                    logger.warning(f"⚠️  Mathematical Identity detected: {target_column} == {col1} * {col2}")
                    # Usually, the "rate" is the leakier part. We drop both to be safe or the more redundant one.
                    # Here we flag the one with higher target correlation if not already flagged
                    if col1 not in leaky_features: leaky_features.append(col1)
                    if col2 not in leaky_features: leaky_features.append(col2)

        if leaky_features:
            leaky_features = list(set(leaky_features)) # Unique
            logger.warning(f"🚩 Dropping {len(leaky_features)} leaky features: {leaky_features}")
            df = df.drop(columns=leaky_features)
        else:
            logger.info("✅ No obvious leakage detected based on numeric correlations")
            
        return df

    def _handle_outliers(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Removes rows containing extreme outliers in the target variable using the 
        Interquartile Range (IQR) method.
        """
        if df.empty or target_column not in df.columns:
            return df
            
        # Only apply to numeric target columns
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            return df
            
        logger.info(f"📉 Checking for outliers in '{target_column}'...")
        
        Q1 = df[target_column].quantile(0.25)
        Q3 = df[target_column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        initial_count = len(df)
        df_filtered = df[(df[target_column] >= lower_bound) & (df[target_column] <= upper_bound)]
        removed_count = initial_count - len(df_filtered)
        
        if removed_count > 0:
            logger.warning(f"🚩 Filtered {removed_count} outlier rows using IQR ({lower_bound:.2f} to {upper_bound:.2f})")
        else:
            logger.debug("✅ No extreme outliers detected in the target variable")
            
        return df_filtered


    def _get_candidate_models(self, task_type: str, include_models: Optional[List[str]] = None) -> List[BaseModel]:
        """
        Instantiates the list of models appropriate for the task.

        Args:
            task_type: 'classification' or 'regression'.
            include_models: Optional list of model names (matching BaseModel.name or substrings).

        Returns:
            List[BaseModel]: A list of instantiated model objects ready for training.
        """
        all_candidates = []
        
        # 1. KNN (Universal)
        all_candidates.append(KNNModel(is_classification=(task_type == 'classification')))
        
        if task_type == 'classification':
            all_candidates.append(LogisticRegressionModel())
            all_candidates.append(RandomForestModel(is_classification=True))
            all_candidates.append(DecisionTreeModel(is_classification=True))
            all_candidates.append(SVMModel(is_classification=True))
            all_candidates.append(NaiveBayesModel())
            
        elif task_type == 'regression':
            all_candidates.append(LinearRegressionModel())
            all_candidates.append(RandomForestModel(is_classification=False))
            all_candidates.append(DecisionTreeModel(is_classification=False))
            all_candidates.append(SVMModel(is_classification=False))

        if not include_models:
            logger.info(f"No specific models selected - training all {len(all_candidates)} available models")
            return all_candidates

        # Filter candidates based on user selection
        logger.debug(f"Filtering models. User selected: {include_models}")
        logger.debug(f"Available models: {[m.name for m in all_candidates]}")
        
        selected_models = []
        for model in all_candidates:
            # Check if any part of the include name matches the model name (e.g., 'KNN' in 'KNN (Regression)')
            matched_by = None
            for inc in include_models:
                if inc.lower() in model.name.lower():
                    matched_by = inc
                    break
            
            if matched_by:
                selected_models.append(model)
                logger.debug(f"✅ Matched: '{model.name}' (selected via '{matched_by}')")
        
        # fallback: if filter resulted in empty, return all (robustness)
        if not selected_models:
            logger.warning(f"⚠️ No models matched user selection {include_models}. Defaulting to ALL models as fallback.")
            logger.warning(f"   Available model names were: {[m.name for m in all_candidates]}")
            return all_candidates
        
        logger.info(f"Training {len(selected_models)} selected models: {[m.name for m in selected_models]}")
        return selected_models

    def _train_single_model(self, model: BaseModel, X_train, y_train, X_test, y_test, progress_callback=None) -> Dict[str, Any]:
        """
        Worker function to train and evaluate a single model.
        
        Rationale:
        ----------
        - **Isolation**: Runs in its own scope, handling exceptions so one failed model doesn't crash the whole pipeline.
        - **Standardized Encoding**: Uses pre-encoded targets to ensure consistency across models.
        - **Metrics**: Calculates model-agnostic metrics on the hold-out test set.

        Args:
            model: The model instance to train.
            X_train: Training features.
            y_train: Training targets.
            X_test: Test features.
            y_test: Test targets.
            progress_callback: Optional callable for sub-step progress (local percent, message).

        Returns:
            Dict[str, Any]: A dictionary containing successful results or failure info.
        """
        try:
            # 1. Preprocess (Get template and target encoding)
            _, y_train_encoded, preprocessor = model.preprocess(X_train, y_train)
            
            # 2. Fit (Passing RAW X_train to allow internal pipeline tuning)
            if progress_callback:
                progress_callback(40, f"Training {model.name}...")
            
            model.fit(X_train, y_train_encoded)
            
            # 3. Evaluate
            if progress_callback:
                progress_callback(80, f"Evaluating {model.name}...")
            
            # Since we centralized encoding in run(), y_train/y_test are ALREADY numeric.
            # We skip local encoding to prevent mismatches and leakage from split samples.
            y_test_encoded = np.asarray(y_test)

            # Pass RAW X_test to evaluation - the model's pipeline handles the pre -> model flow.
            metrics = model.calculate_metrics(X_test, y_test_encoded)

            # For back-compat and internal summary, we still want to see the shape or some proc data.
            X_test_proc = preprocessor.transform(X_test)
            X_test_proc = np.asarray(X_test_proc)

            if progress_callback:
                progress_callback(100, f"Finished {model.name}")
            
            return {
                'model_name': model.name,
                'status': 'success',
                'metrics': metrics,
                'diagnostics': model.get_diagnostic_data(X_test, y_test_encoded),
                'model_instance': model,
                'preprocessor': preprocessor,
                'test_data_proc': (X_test_proc, y_test_encoded) 
            }
        except Exception as e:
            # Print full trace for debugging - this helps identify EXACTLY why a model failed
            traceback.print_exc()
            return {
                'model_name': model.name,
                'status': 'failed',
                'error': str(e)
            }

    def _get_feature_names(self, column_transformer: ColumnTransformer) -> List[str]:
        """
        Helper method to extract human-readable feature names after transformations.
        
        Args:
            column_transformer: The fitted ColumnTransformer object.
            
        Returns:
            List[str]: List of transformed feature names.
        """
        feature_names = []
        
        # Loop through transformers in the ColumnTransformer
        for name, transformer, columns in column_transformer.transformers_:
            if name == 'remainder' and transformer == 'drop':
                continue
            
            # If transformer is a Pipeline, get the last step's feature names if it supports them
            # or use the original columns if it's just a scaler/imputer
            if hasattr(transformer, 'get_feature_names_out'):
                try:
                    names = transformer.get_feature_names_out(columns)
                    feature_names.extend(names)
                except Exception:
                    feature_names.extend(columns)
            else:
                feature_names.extend(columns)
                
        return feature_names

    def run(self, df: pd.DataFrame, target_column: str, progress_callback=None, include_models: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Main execution point for the AutoML pipeline.

        Args:
            df: The input pandas DataFrame containing features and the target column.
            target_column: The name of the column to predict.
            progress_callback: Optional callable for progress updates. 
                              Expected signature: (percent: int, message: str)
            include_models: Optional list of specific models to train.

        Returns:
            Dict[str, Any]: A summary dictionary containing:
                            - 'task_type': Inferred or specified task type.
                            - 'total_time': Total execution time in seconds.
                            - 'leaderboard': List of results for all models.
                            - 'best_model': Dictionary with details of the best performing model.
        
        Raises:
            ValueError: If target_column is not in df.
            RuntimeError: If all models fail to train.
        """
        start_time = time.time()
        
        logger.info("="*70)
        logger.info("🚀 STARTING AutoML Pipeline - ModelExecutor.run()")
        logger.info(f"📊 Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        logger.info(f"🎯 Target: '{target_column}'")
        logger.info(f"⚙️  Task Mode: {self.task_type} | Cores: {self.n_jobs} | Random Seed: {self.random_state}")
        if include_models:
            logger.info(f"🎯 User Model Filter: {include_models}")
        logger.info("="*70)
        
        # 1. Basic Validation
        logger.debug("[Step 1/10] Validating input data...")
        if target_column not in df.columns:
            logger.error(f"❌ Target column '{target_column}' not found. Available: {list(df.columns)}")
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
        
        logger.info("✅ Input validation passed")
            
        # 2. Separate Features and Target
        logger.debug("[Step 2/10] Separating features (X) and target (y)...")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        logger.info(f"📈 Features: {X.shape[1]} columns | Target: {y.nunique()} unique values")

        # --- DATA CLEANING & TYPE ENFORCEMENT ---
        logger.debug("[Step 3/10] Cleaning and enforcing data types...")
        # 1. Attempt to convert all object columns to numeric where possible
        # 2. If valid strings exist, ensure they are kept as object/category for OneHot
        converted_cols = []
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    # Try converting to numeric
                    X[col] = pd.to_numeric(X[col])
                    converted_cols.append(col)
                except (ValueError, TypeError):
                    # If failed, it contains strings.
                    # Ensure it's explicitly 'category' or 'object' for the pipeline to hit 'cat' step
                    X[col] = X[col].astype(str)
        
        if converted_cols:
            logger.info(f"🔄 Auto-converted {len(converted_cols)} object columns to numeric: {converted_cols}")
        logger.info("✅ Data type enforcement complete")
        
        # Also clean Target if it's supposed to be specific type? 
        # For now, let inference handle y. 
        # But ensure X has no mixed types that confuse sklearn.
        # ----------------------------------------
        
        # 3. Infer Task
        logger.debug("[Step 4/10] Inferring task type (classification vs regression)...")
        inferred_task = self._infer_task_type(y)
        logger.info(f"📊 Task inferred as: {inferred_task.upper()}")
        
        # 4. Outlier Management (Optional implementation detail)
        logger.debug("[Step 5/10] Handling outliers in target variable...")
        rows_before = len(df)
        # We handle outliers before training to prevent skewed metrics
        df = self._handle_outliers(df, target_column)
        rows_after = len(df)
        if rows_before != rows_after:
            logger.info(f"🧙 Removed {rows_before - rows_after} outlier rows ({((rows_before - rows_after) / rows_before * 100):.1f}%)")
        else:
            logger.info("✅ No outliers detected")

        # 5. Automated Leakage Detection (Systemic fix)
        logger.debug("[Step 6/10] Running automated leakage detection...")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        df_processed = pd.concat([X, y], axis=1) # Rejoin temporarily for check
        features_before = X.shape[1]
        df_clean = self._detect_and_drop_leakage(df_processed, target_column)
        
        # Update X and y after cleaning
        y = df_clean[target_column]
        X = df_clean.drop(columns=[target_column])
        features_after = X.shape[1]
        if features_before != features_after:
            logger.warning(f"⚠️ Dropped {features_before - features_after} potentially leaky features")
        
        # 4. Target Encoding (Centralized for Zero Leakage & Consistency)
        self.label_encoder = None
        if inferred_task == 'classification':
            self.label_encoder = LabelEncoder()
            y = pd.Series(self.label_encoder.fit_transform(y), index=y.index)
            logger.debug(f"Target labels encoded: {self.label_encoder.classes_}")

        # 5. Train/Test Split
        logger.debug("[Step 8/10] Splitting data into train/test sets...")
        # Stratify if classification
        stratify = y if inferred_task == 'classification' else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=self.random_state,
            stratify=stratify
        )
        logger.info(f"📊 Train: {len(X_train)} samples | Test: {len(X_test)} samples (80/20 split)")
        if stratify is not None:
            logger.debug("Stratified split applied for balanced class distribution")
        
        # 5. Initialize Models
        logger.debug("[Step 9/10] Instantiating candidate models...")
        models = self._get_candidate_models(inferred_task, include_models=include_models)
        logger.info(f"🤖 Initialized {len(models)} models: {[m.name for m in models]}")
        
        logger.info("="*70)
        logger.info("[Step 10/10] 🏋️ TRAINING PHASE - Starting parallel model training...")
        logger.info("="*70)
        
        if progress_callback:
            progress_callback(10, f"Starting training on {len(models)} models...")

        # 6. Training (Hybrid Sequential/Parallel for Progress visibility)
        # To show progress, we iterate through models. 
        # But we still want each model to use multi-core (managed by model.n_jobs).
        # If n_jobs is -1, the Parallel wrapper below would otherwise hide sub-progress.
        
        results_list = []
        n_models = len(models)
        for i, model in enumerate(models):
            # Define a local callback that maps the model's 0-100% to the global range
            def sub_step_callback(local_percent, message):
                if progress_callback:
                    # Global percent = Base + (Offset * (local_percent / 100))
                    # Training range is 10-90 (total span = 80)
                    base = 10 + (i / n_models) * 80
                    span = 80 / n_models
                    global_percent = int(base + (span * (local_percent / 100)))
                    progress_callback(global_percent, message)

            result = self._train_single_model(model, X_train, y_train, X_test, y_test, progress_callback=sub_step_callback)
            results_list.append(result)

        if progress_callback:
            progress_callback(90, "Finalizing model selection...")

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
        
        # 8. Generate Diagnostics for Best Model
        # We need to call get_diagnostic_data on the best model instance
        # IMPORTANT: Use RAW X_test, and y_test (which is already encoded from the split)
        diagnostics = self.best_model_instance.get_diagnostic_data(X_test, np.asarray(y_test))
        
        # 9. Return Final Summary
        # 10. Extract Explainability Data
        feature_names = self._get_feature_names(best_run['preprocessor'])
        # If feature selector was used, we need to filter these names
        if hasattr(self.best_model_instance, 'feature_selector') and self.best_model_instance.feature_selector is not None:
             indices = self.best_model_instance.feature_selector.get_support(indices=True)
             feature_names = [feature_names[i] for i in indices]

        importance_data = self.best_model_instance.get_feature_importance()
        # Merge names if importance exists
        if 'importances' in importance_data:
             importance_data['feature_names'] = feature_names

        summary = {
            'task_type': inferred_task,
            'total_time': time.time() - start_time,
            'leaderboard': [
                {
                    'model': r['model_name'], 
                    'metrics': r.get('metrics', {}),
                    'status': r['status'],
                    'error': r.get('error')
                } 
                for r in self.results
            ],
            'best_model': {
                'name': self.best_model_name,
                'metrics': self.best_model_metrics,
                'diagnostics': diagnostics,
                'explainability': {
                    'importance': importance_data,
                    'parameters': self.best_model_instance.get_parameter_descriptions() if hasattr(self.best_model_instance, 'get_parameter_descriptions') else {}
                }
            }
        }
        
        total_time = time.time() - start_time
        logger.info("="*70)
        logger.info(f"✅ PIPELINE COMPLETE - Total time: {total_time:.2f}s")
        logger.info(f"🏆 Best Model: {self.best_model_name}")
        logger.info(f"📊 Performance: {list(self.best_model_metrics.values())[0]:.4f} ({list(self.best_model_metrics.keys())[0]})")
        logger.info(f"📝 Trained {len(results_list)} models successfully")
        logger.info("="*70)
        
        if progress_callback:
            progress_callback(100, f"Completed! Best model: {self.best_model_name}")

        return summary
