import pandas as pd
import numpy as np
import time
import traceback
import os
import inspect
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

# --- PROJECT IMPORTS ---
from core_recommender.logger import get_logger
from core_recommender.modeling.base_model import BaseModel
from core_recommender.exceptions import (
    DataValidationError,
    InsufficientDataError,
    ModelTrainingError,
    PipelineError,
)

# Registry-based model discovery (OCP / DIP)
# Importing the concrete model modules causes their @register_model decorators
# to fire, populating the registry.  The executor never needs to be touched
# when a new model is added.
from core_recommender.modeling.registry import get_registered_models
import core_recommender.modeling.knn                # noqa: F401 — triggers registration
import core_recommender.modeling.linear_regression   # noqa: F401
import core_recommender.modeling.logistic_regression # noqa: F401
import core_recommender.modeling.random_forest       # noqa: F401
import core_recommender.modeling.decision_trees      # noqa: F401
import core_recommender.modeling.svms               # noqa: F401
import core_recommender.modeling.naive_bayes         # noqa: F401

from core_recommender.profiling import DataProfiler
from .visualization import plot_shap_summary

logger = get_logger(__name__)

# =========================================================================
# Execution Engine
# =========================================================================

class ModelExecutor:
    """The orchestrator that manages the end-to-end model selection pipeline.
    
    This class encapsulates the complexity of:
    1.  Task Inference (Classification vs. Regression).
    2.  Data Profiling and Cleaning (Leakage detection, Outlier removal).
    3.  Model Instantiation and Selection.
    4.  Parallel Training and Evaluation.
    5.  Result Aggregation and Ranking.

    Attributes:
        task_type (str): 'classification', 'regression', or 'auto'.
        n_jobs (int): Number of parallel jobs (-1 for all cores).
        random_state (int): Seed for reproducibility.
        results (List[Dict[str, Any]]): Storage for training results.
        best_model_name (Optional[str]): Name of the best performing model.
        best_model_metrics (Optional[Dict[str, float]]): Metrics of the best model.
        best_model_instance (Optional[BaseModel]): The actual fitted best model object.
        pipeline_log (List[Dict[str, str]]): structured log of pipeline steps.
    """
    
    def __init__(self, task_type: str = 'auto', n_jobs: int = -1, random_state: int = 42) -> None:
        """Initializes the ModelExecutor.

        Args:
            task_type (str, optional): 'classification', 'regression', or 'auto'. Defaults to 'auto'.
            n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
            random_state (int, optional): Random seed. Defaults to 42.
        """
        self.task_type: str = task_type
        self.n_jobs: int = n_jobs
        self.random_state: int = random_state
        self.results: List[Dict[str, Any]] = []
        self.best_model_name: Optional[str] = None
        self.best_model_metrics: Optional[Dict[str, float]] = None
        self.best_model_instance: Optional[BaseModel] = None
        self.pipeline_log: List[Dict[str, str]] = []
        self.label_encoder: Optional[LabelEncoder] = None

    def _log_step(self, step: str, details: str, icon: str = "fas fa-info-circle") -> None:
        """Helper to append an entry to the pipeline log."""
        self.pipeline_log.append({
            "step": step,
            "details": details,
            "icon": icon,
            "timestamp": time.strftime("%H:%M:%S")
        })

    def _infer_task_type(self, y: pd.Series) -> str:
        """Infers whether the problem is 'classification' or 'regression'.

        Heuristic:
        - Float dtype -> Regression
        - Object/Categorical/Bool -> Classification
        - Integer with < 20 unique values -> Classification
        - Integer with >= 20 unique values -> Regression

        Args:
            y (pd.Series): The target variable.

        Returns:
            str: 'classification' or 'regression'.
        """
        if self.task_type != 'auto':
            return self.task_type
            
        if pd.api.types.is_object_dtype(y) or pd.api.types.is_bool_dtype(y) or isinstance(y.dtype, pd.CategoricalDtype):
            # Try converting to numeric to see if they are numbers stored as strings
            try:
                pd.to_numeric(y, errors='raise')
                # If conversion works, fall through to numeric checks (unless it's boolean which is definitely classif)
                if pd.api.types.is_bool_dtype(y):
                     return 'classification'
            except (ValueError, TypeError):
                return 'classification'
            
        if pd.api.types.is_float_dtype(y):
            return 'regression'
        
        # Check integers (or numeric strings converted to standard types)
        # We rely on nunique for ambiguity
        n_unique = y.nunique()
        if n_unique < 20:
            logger.info(f"Target has {n_unique} unique values. Inferring CLASSIFICATION.")
            return 'classification'
        else:
            logger.info(f"Target has {n_unique} unique values. Inferring REGRESSION.")
            return 'regression'

    def _detect_and_drop_leakage(self, df: pd.DataFrame, target_column: str, threshold: float = 0.95) -> pd.DataFrame:
        """Identifies and removes features with suspiciously high correlation to the target.

        Args:
            df (pd.DataFrame): Input dataframe.
            target_column (str): Name of target column.
            threshold (float, optional): Correlation threshold. Defaults to 0.95.

        Returns:
            pd.DataFrame: Cleaned dataframe with leaky features removed.
        """
        if df.empty or target_column not in df.columns:
            return df
            
        logger.info("🔍 Automated Leakage Detection Check...")
        
        # 1. Correlation-based Leakage (Numeric only)
        numeric_df = df.select_dtypes(include=[np.number])
        if target_column not in numeric_df.columns:
             return df
             
        corr_matrix = numeric_df.corr().abs()
        target_corr = corr_matrix[target_column].sort_values(ascending=False)
        
        leaky_features: List[str] = []
        
        for feature, correlation in target_corr.items():
            if feature == target_column:
                continue
            if correlation > threshold:
                logger.warning(f"⚠️ Suspiciously high correlation: '{feature}' has {correlation:.4f} correlation with target")
                leaky_features.append(str(feature))

        # 2. Mathematical Identity Check (A * B = Target)
        candidates = [c for c in numeric_df.columns if c != target_column]
        for i, col1 in enumerate(candidates):
            for col2 in candidates[i+1:]:
                prod = numeric_df[col1] * numeric_df[col2]
                if np.allclose(prod, numeric_df[target_column], rtol=1e-5, atol=1e-8):
                    logger.warning(f"⚠️ Identity detected: {target_column} == {col1} * {col2}")
                    if col1 not in leaky_features: leaky_features.append(col1)
                    if col2 not in leaky_features: leaky_features.append(col2)

        # 3. Categorical/Object Leakage Check
        # Check if any non-numeric column has suspiciously low entropy or mirrors the target
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        for col in cat_cols:
            if col == target_column: continue
            # If a column name contains "target", "label", "result" it's a high risk
            if any(word in col.lower() for word in ['target', 'label', 'result', 'outcome', 'output']):
                logger.warning(f"⚠️ High-risk column name detected: '{col}' looks like a target proxy.")
                if col not in leaky_features: leaky_features.append(col)
                continue
                
            # If unique values are identical to target (1:1 mapping)
            if df[col].nunique() == df[target_column].nunique():
                # Cross-tab check for perfect correlation
                contingency = pd.crosstab(df[col], df[target_column])
                if (contingency.values > 0).sum() == df[col].nunique():
                    logger.warning(f"⚠️ Categorical Leakage: '{col}' has a perfect 1:1 mapping with the target.")
                    if col not in leaky_features: leaky_features.append(col)

        if leaky_features:
            leaky_features = list(set(leaky_features))
            logger.warning(f"🚩 Dropping {len(leaky_features)} leaky features: {leaky_features}")
            self._log_step("Leakage Check", f"Removed {len(leaky_features)} leaky features: {', '.join(leaky_features)}", "fas fa-shield-alt")
            return df.drop(columns=leaky_features)
        else:
            logger.info("✅ No obvious leakage detected.")
            self._log_step("Values Check", "No data leakage detected. Features look healthy.", "fas fa-shield-alt")
            return df

    def _handle_outliers(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Removes rows with extreme outliers in the target variable using IQR.

        Args:
            df (pd.DataFrame): Input dataframe.
            target_column (str): Target column name.

        Returns:
            pd.DataFrame: Dataframe with outliers removed.
        """
        if df.empty or target_column not in df.columns:
            return df
            
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
            msg = f"Removed {removed_count} outlier rows using IQR rule."
            logger.warning("🚩 " + msg)
            self._log_step("Outlier Detection", msg, "fas fa-filter")
        else:
            logger.debug("✅ No extreme outliers detected in the target variable")
            self._log_step("Outlier Detection", "No extreme outliers found in target variable.", "fas fa-check")
            
        return df_filtered

    def _get_candidate_models(self, task_type: str, include_models: Optional[List[str]] = None) -> List[BaseModel]:
        """Instantiates candidate models from the global registry.

        Satisfies OCP / DIP: the executor never needs modification when new
        models are added.  Each registered class must accept an optional
        ``is_classification`` boolean constructor parameter.

        Args:
            task_type (str): 'classification' or 'regression'.
            include_models (Optional[List[str]]): Whitelist of model name
                substrings.  ``None`` means all registered models.

        Returns:
            List[BaseModel]: Ready-to-train model instances.
        """
        is_clf = (task_type == 'classification')
        registered = get_registered_models(task_type)

        all_candidates: List[BaseModel] = []
        for cls in registered:
            try:
                # Models that support both tasks accept is_classification kwarg;
                # task-specific models (e.g. LogisticRegression) do not.
                sig = inspect.signature(cls.__init__)
                if 'is_classification' in sig.parameters:
                    all_candidates.append(cls(is_classification=is_clf))
                else:
                    all_candidates.append(cls())
            except Exception as exc:  # pragma: no cover
                logger.warning(f"⚠️ Could not instantiate {cls.__name__}: {exc}")

        if not include_models:
            logger.info(
                f"No specific models selected — training all "
                f"{len(all_candidates)} registered models."
            )
            return all_candidates

        # Whitelist filter
        selected_models: List[BaseModel] = []
        for model in all_candidates:
            for inc in include_models:
                possible_matches = [p.strip().lower() for p in inc.split(',')]
                if any(p in model.name.lower() for p in possible_matches if p):
                    selected_models.append(model)
                    break

        if not selected_models:
            logger.warning(
                f"⚠️ No models matched selection {include_models}. "
                f"Defaulting to ALL."
            )
            return all_candidates

        logger.info(
            f"Training {len(selected_models)} selected models: "
            f"{[m.name for m in selected_models]}"
        )
        return selected_models

    def _train_single_model(self, model: BaseModel, X_train: pd.DataFrame, y_train: Union[pd.Series, np.ndarray], 
                            X_test: pd.DataFrame, y_test: Union[pd.Series, np.ndarray], progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Trains and evaluates a single model.
        
        Args:
            model (BaseModel): The model instance.
            X_train (pd.DataFrame): Training features.
            y_train (Union[pd.Series, np.ndarray]): Training targets.
            X_test (pd.DataFrame): Test features.
            y_test (Union[pd.Series, np.ndarray]): Test targets.
            progress_callback (Optional[Callable]): Callback for progress updates.

        Returns:
            Dict[str, Any]: Result dictionary with status, metrics, and diagnostics.
        """
        try:
            # 1. Preprocess
            # Note: preprocess returns (X_transformed, y_transformed, preprocessor)
            _, y_train_encoded, preprocessor = model.preprocess(X_train, y_train)
            
            # 2. Fit
            if progress_callback:
                progress_callback(40, f"Training {model.name}...")
            
            # We pass RAW X_train because the model's internal fit() typically handles 
            # its own pipeline (which includes the preprocessor we just built).
            # The BaseModels in this project are designed to take a DataFrame in fit()
            # and wrap it in a Pipeline with the preprocessor.
            # y_train_encoded is passed because targets usually don't go through the ColumnTransformer in the pipeline.
            model.fit(X_train, y_train_encoded)
            
            # 3. Evaluate
            if progress_callback:
                progress_callback(80, f"Evaluating {model.name}...")
            
            # Calculate metrics using RAW X_test (model handles transform via pipeline)
            # y_test should be already encoded/numeric suitable for metric calc
            metrics = model.calculate_metrics(X_test, y_test)

            # For internal storage, we might want the processed test data
            X_test_proc = preprocessor.transform(X_test)
            X_test_proc = np.asarray(X_test_proc)

            if progress_callback:
                progress_callback(100, f"Finished {model.name}")
            
            return {
                'model_name': model.name,
                'status': 'success',
                'metrics': metrics,
                'diagnostics': model.get_diagnostic_data(X_test, y_test),
                'model_instance': model,
                'preprocessor': preprocessor,
                'test_data_proc': (X_test_proc, y_test) 
            }
        except Exception as e:
            logger.error(
                f"❌ [{type(e).__name__}] {model.name} failed: {e}",
                exc_info=True
            )
            return {
                'model_name': model.name,
                'status': 'failed',
                'error': f"[{type(e).__name__}] {e}"
            }

    def _get_feature_names(self, column_transformer: ColumnTransformer) -> List[str]:
        """Extracts human-readable feature names from the ColumnTransformer.

        Args:
            column_transformer (ColumnTransformer): Fitted transformer.

        Returns:
            List[str]: List of feature names.
        """
        feature_names = []
        
        for name, transformer, columns in column_transformer.transformers_:
            if name == 'remainder' and transformer == 'drop':
                continue
            
            if hasattr(transformer, 'get_feature_names_out'):
                try:
                    names = transformer.get_feature_names_out(columns)
                    feature_names.extend(names)
                except (AttributeError, ValueError, TypeError) as e:
                    logger.debug(f"Failed to get feature names from {name}: {e}")
                    feature_names.extend(columns)
            else:
                feature_names.extend(columns)
                
        return feature_names

    def run(self, df: pd.DataFrame, target_column: str, progress_callback: Optional[Callable] = None, 
            include_models: Optional[List[str]] = None) -> Dict[str, Any]:
        """Executes the full AutoML pipeline.

        Args:
            df (pd.DataFrame): Input dataframe.
            target_column (str): Name of the target variable.
            progress_callback (Optional[Callable]): Function to report progress (percent, message).
            include_models (Optional[List[str]]): List of models to include.

        Returns:
            Dict[str, Any]: Summary of results including task type, leaderboard, and best model details.

        Raises:
            ValueError: If target_column is not found.
            RuntimeError: If all models fail.
        """
        start_time = time.time()
        
        logger.info("="*70)
        logger.info("🚀 STARTING AutoML Pipeline - ModelExecutor.run()")
        logger.info(f"📊 Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        logger.info(f"🎯 Target: '{target_column}'")
        
        # 1. Validation
        if target_column not in df.columns:
            logger.error(f"❌ Target column '{target_column}' not found.")
            raise DataValidationError(
                f"Target column '{target_column}' not found in the dataset.",
                column=target_column,
                available_columns=list(df.columns)
            )
        
        self._log_step("Initialization", f"Loaded dataset with {df.shape[0]} rows.", "fas fa-database")
            
        # 2. Features/Target Split
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # --- DATA CLEANING ---
        # Convert object columns to numeric where possible
        converted_cols = []
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col])
                    converted_cols.append(col)
                except (ValueError, TypeError):
                    X[col] = X[col].astype(str) # Ensure explicit string type
        
        if converted_cols:
            logger.info(f"🔄 Auto-converted {len(converted_cols)} object columns to numeric.")

        # 3. Infer Task
        inferred_task = self._infer_task_type(y)
        logger.info(f"📊 Task inferred as: {inferred_task.upper()}")
        
        # 4. Outlier Management (Regression only)
        if inferred_task == 'regression':
             # Note: We do this on the original 'df' but we already split X/y. 
             # Re-merge to filter rows safely
             temp_df = pd.concat([X, y], axis=1)
             temp_df = self._handle_outliers(temp_df, target_column)
             X = temp_df.drop(columns=[target_column])
             y = temp_df[target_column]
        
        # --- PROFILING & LEAKAGE ---
        profiler = DataProfiler()
        # Analyze expects a dataframe and series
        analysis_result = profiler.analyze(X, y)
        suggestions = analysis_result.get('suggestions', {})
        
        for msg in suggestions.get('messages', []):
            self._log_step("Intelligent Profiling", msg, "fas fa-brain")
            


        # Leakage
        temp_df = pd.concat([X, y], axis=1)
        # We need to preserve feature count to check diff
        features_before = X.shape[1]
        temp_df_clean = self._detect_and_drop_leakage(temp_df, target_column)
        
        X = temp_df_clean.drop(columns=[target_column])
        y = temp_df_clean[target_column]
        
        features_after = X.shape[1]
        if features_before != features_after:
            logger.warning(f"⚠️ Dropped {features_before - features_after} leaky features")
        
        # 5. Target Encoding
        self.label_encoder = None
        # We need to handle y for training.
        # If classification, we encode it to integers.
        # If regression, we ensure it's numeric/float.
        
        if inferred_task == 'classification':
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            y = pd.Series(y_encoded, index=y.index)
            self._log_step("Target Encoding", f"Encoded classes: {list(self.label_encoder.classes_)}", "fas fa-list")
        else:
            # For regression, ensure float
            y = pd.to_numeric(y, errors='coerce').fillna(0) # Simple fill check, ideally handled by outlier logic pre-step

        # 6. Train/Test Split
        stratify = y if inferred_task == 'classification' else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=self.random_state,
            stratify=stratify
        )
        logger.info(f"📊 Train: {len(X_train)} | Test: {len(X_test)}")
        self._log_step("Data Splitting", f"80/20 data split. Train: {len(X_train)}, Test: {len(X_test)}.", "fas fa-cut")

        # 7. Model Selection
        models = self._get_candidate_models(inferred_task, include_models=include_models)
        


        if progress_callback:
            progress_callback(10, f"Starting training on {len(models)} models...")

        # 8. Training Loop
        results_list = []
        n_models = len(models)
        for i, model in enumerate(models):
            # Callback adapter
            def sub_step_callback(local_p, msg):
                if progress_callback:
                    # Map local 0-100 to global slice
                    # Global slice for training is approx 10% to 90%
                    base = 10 + (i / n_models) * 80
                    span = 80 / n_models
                    global_p = int(base + (span * (local_p / 100)))
                    progress_callback(global_p, msg)

            result = self._train_single_model(
                model, X_train, y_train, X_test, y_test, 
                progress_callback=sub_step_callback
            )
            results_list.append(result)
            
            if result['status'] == 'success':
                self._log_step("Model Assembly", f"Successfully built {model.name}.", "fas fa-microchip")
            else:
                self._log_step("Model Failure", f"Failed to build {model.name}: {result.get('error', 'Unknown Error')}", "fas fa-exclamation-triangle")

        # Sort results so the best model is at the top of the leaderboard.
        def get_sort_key(r):
            is_success = 1 if r.get('status') == 'success' else 0
            if inferred_task == 'classification':
                metric_val = r.get('metrics', {}).get('F1 Score', -1.0)
                return (is_success, metric_val)
            else:
                metric_val = r.get('metrics', {}).get('RMSE', float('inf'))
                return (is_success, -metric_val)
        
        results_list = sorted(results_list, key=get_sort_key, reverse=True)
        self.results = results_list
        valid_results = [r for r in results_list if r['status'] == 'success']
        
        if not valid_results:
            raise RuntimeError("All models failed to train.")

        # 9. Ranking
        if inferred_task == 'classification':
            best_run = max(valid_results, key=lambda x: x['metrics'].get('F1 Score', 0))
        else:
            best_run = min(valid_results, key=lambda x: x['metrics'].get('RMSE', float('inf')))

        self.best_model_name = best_run['model_name']
        self.best_model_metrics = best_run['metrics']
        self.best_model_instance = best_run['model_instance']
        
        # 10. Pipeline Transparency
        try:
            transformers = best_run['preprocessor'].transformers_
            technique_details = []
            for name, trans, cols in transformers:
                if name == 'remainder': continue
                if hasattr(trans, 'steps'): # Pipeline
                    for step_name, step_obj in trans.steps:
                        technique_details.append(step_obj.__class__.__name__)
                else: # Transformer
                    technique_details.append(trans.__class__.__name__)
            
            technique_details = list(set(technique_details))
            self._log_step("Feature Preprocessing", f"Techniques: {', '.join(technique_details)}", "fas fa-cogs")
        except (AttributeError, ValueError, TypeError) as e:
            logger.debug(f"Failed to extract technique details: {e}")
            self._log_step("Feature Preprocessing", "Standard scaling/imputation.", "fas fa-cogs")

        # 11. Explainability & Summary
        diagnostics = self.best_model_instance.get_diagnostic_data(X_test, np.asarray(y_test))
        self._log_step('Diagnostics Generation', 'Computed performance metrics and visualizations.', 'fas fa-chart-bar')
        feature_names = self._get_feature_names(best_run['preprocessor'])
        
        # Filter feature names if selector was used
        if hasattr(self.best_model_instance, 'feature_selector'):
            fs = self.best_model_instance.feature_selector
            if hasattr(fs, 'get_support'):
                indices = fs.get_support(indices=True)
                feature_names = [feature_names[i] for i in indices if i < len(feature_names)]

        # Single call — executor never needs to know which model-specific
        # features exist.  Each model decides what it exposes via its override.
        tailored_diagnostics = self.best_model_instance.get_tailored_diagnostics()
        
        # 12. Modular SHAP (Explainability Layer)
        # This section generates global feature impact plots using SHAP values.
        try:
            # Deterministic path for the SHAP artifact to be consumed by the Flask frontend.
            shap_path = os.path.join('interface', 'static', 'images', 'shap_summary.png')
            
            # Robustness: Ensure the directory structure exists (especially important for fresh clones).
            if not os.path.exists(os.path.dirname(shap_path)):
                os.makedirs(os.path.dirname(shap_path), exist_ok=True)
                
            # Convert test data back to a labeled DataFrame for better SHAP plot annotations.
            # Using the preprocessed test data so the shapes and dimensions match the model expectations.
            X_test_proc, _ = best_run['test_data_proc']
            if X_test_proc.shape[1] == len(feature_names):
                X_test_df = pd.DataFrame(X_test_proc, columns=feature_names)
            else:
                cols = feature_names[:X_test_proc.shape[1]] if X_test_proc.shape[1] < len(feature_names) else [f"feature_{i}" for i in range(X_test_proc.shape[1])]
                X_test_df = pd.DataFrame(X_test_proc, columns=cols)
            
            # Generate the visualization. 
            # Note: We pass the underlying 'best_estimator' if it's wrapped in a Pipeline.
            plot_shap_summary(self.best_model_instance.best_estimator, X_test_df, self.best_model_name, shap_path)
            logger.info("✅ SHAP Summary Plot generated successfully.")
        except Exception as e:
            # SHAP is a value-added feature; we log the failure but do not halt the entire pipeline.
            logger.warning(f"⚠️ SHAP generation skipped: {str(e)}", exc_info=True)

        # 13. Construct Comprehensive Final Result Summary
        summary = {
            'task_type': inferred_task,
            'total_time': time.time() - start_time,
            'classes': list(self.label_encoder.classes_) if self.label_encoder else None,
            'leaderboard': [
                {
                    'model': r['model_name'], 
                    'name': r['model_name'], 
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
                'classes': list(self.label_encoder.classes_) if self.label_encoder else None,
                'feature_names': feature_names,
                'tailored_diagnostics': tailored_diagnostics,
                'explainability': {
                    'parameters': self.best_model_instance.get_parameter_descriptions(),
                    'has_shap': True  # Flag to trigger app.py plotting
                },
                'pipeline_log': self.pipeline_log
            }
        }
        
        logger.info(f"🏆 Best Model: {self.best_model_name}")
        if progress_callback:
            progress_callback(100, f"Completed! Best model: {self.best_model_name}")

        return summary
