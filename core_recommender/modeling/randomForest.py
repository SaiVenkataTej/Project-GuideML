import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Union
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

# --- PROJECT IMPORTS ---
from core_recommender.modeling.baseModel import BaseModel
from core_recommender.preprocessing import (
    get_imputer,
    get_one_hot_encoder,
    get_ordinal_encoder,
    get_select_from_model
)
from core_recommender.evaluation import (
    calculate_f1_score,
    calculate_roc_auc_score,
    calculate_rmse,
    get_oob_score,
    get_tree_depth,
    get_leaf_count
)

# --- DEFAULT CONFIGURATION ---
CONFIG = {
    'task_type': 'auto', # 'classification', 'regression', 'auto'
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'max_features': ['sqrt', 'log2', None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'selection_threshold': 'median',
    'cv_folds': 5,
    'random_state': 42,
    'n_jobs': -1,
    'class_weight': 'balanced' # 'balanced', 'balanced_subsample'
}

# =========================================================================
# RandomForestModel Class
# =========================================================================

class RandomForestModel(BaseModel):
    """
    A concrete implementation of Random Forest for both Classification and Regression.
    
    This ensemble method constructs a multitude of decision trees at training time. 
    It supports automatic task type inference, cost-sensitive learning for imbalanced data, 
    and incorporates specialized metrics like OOB (Out-of-Bag) error.
    """
    def __init__(self, is_classification: bool = True, config: Dict[str, Any] = CONFIG):
        """
        Initializes the Random Forest model.

        Args:
            is_classification: True for classification tasks, False for regression.
            config: Dictionary containing hyperparameters (e.g., 'n_estimators', 'max_depth').
                    Defaults to the global CONFIG dictionary.
        """
        
        task_name = "Classification" if is_classification else "Regression"
        name = f"Random Forest ({task_name})"
        super().__init__(name=name, config=config)
        
        self.task_type = config.get('task_type', 'auto')
        self.is_classification = is_classification

    def _infer_task_type(self, y: pd.Series):
        """Infers classification or regression if set to auto."""
        if self.task_type != 'auto':
            self.is_classification = (self.task_type == 'classification')
            return

        if pd.api.types.is_float_dtype(y):
            self.is_classification = False
        elif pd.api.types.is_object_dtype(y) or pd.api.types.is_bool_dtype(y) or pd.api.types.is_categorical_dtype(y):
            self.is_classification = True
        elif pd.api.types.is_integer_dtype(y):
             # Heuristic: < 20 unique values = Classification
             self.is_classification = (y.nunique() < 20)
        else:
             self.is_classification = False

    def preprocess(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
        """
        Constructs and applies the feature pipeline optimized for Random Forest.
        
        Pipeline Steps:
        1. Numerical: Median imputation. (Scaling is generally not required for Trees).
        2. Categorical: Most frequent imputation. One-Hot encoding (or Ordinal).
        3. Feature Selection: Model-based selection using a lightweight Random Forest.
        
        Args:
            X: Input features DataFrame.
            y: Target Series.
            
        Returns:
            Tuple containing:
            - Selected feature array (np.ndarray)
            - Transformed target array (np.ndarray)
            - The fitted ColumnTransformer object (Note: Does not include the selection step)
            
        Raises:
            ValueError: If target variable 'y' contains NaNs.
        """
        if y.isna().any():
             raise ValueError("Target variable 'y' contains missing values.")

        self._infer_task_type(y)
        
        # 1. Pipeline Construction
        # ------------------------
        
        # Numerical Steps: Impute -> Pass (Trees handle scaling well, generally no scaling needed)
        # However, SimpleImputer is required.
        num_steps = [('imputer', get_imputer(strategy='median'))]
        
        # Note: SelectFromModel logic is typically applied AFTER encoding.
        # We will add it to the final pipeline or manually here if needed per column type.
        # But commonly, trees handle features as is. 
        # Requirement: "SelectFromModel (Tree - based selection)"
        
        numerical_pipeline = Pipeline(steps=num_steps)

        # Categorical Steps: Impute -> Ordinal or OneHot
        # Trees work well with Ordinal encoding for high cardinality, OneHot for low.
        # Requirement: "OrdinalEncoder | OneHotEncoder"
        # We'll stick to OneHot as general safe default, or Ordinal if requested.
        # Let's use OneHot for now to be safe with Scikit implementation standards.
        cat_steps = [
            ('imputer', get_imputer(strategy='most_frequent')),
            ('onehot', get_one_hot_encoder(handle_unknown='ignore', sparse_output=False))
        ]
        cat_pipeline = Pipeline(steps=cat_steps)

        # 2. Composition
        # --------------
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, X.select_dtypes(include=np.number).columns.tolist()),
                ('cat', cat_pipeline, X.select_dtypes(include=['object', 'category']).columns.tolist())
            ],
            remainder='drop',
            n_jobs=self.config.get('n_jobs', -1)
        )
        
        # 3. Fit-Transform
        # ----------------
        X_transformed = preprocessor.fit_transform(X, y)
        X_transformed = np.asarray(X_transformed)

        # 4. Feature Selection (SelectFromModel)
        # --------------------------------------
        # We apply this ON TOP of the transformed features. 
        # We need a lightweight estimator for selection.
        if self.is_classification:
            sel_est = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        else:
            sel_est = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            
        self.feature_selector = get_select_from_model(
            estimator=sel_est,
            threshold=self.config.get('selection_threshold', 'median')
        )
        
        # Fit selector
        X_selected = self.feature_selector.fit_transform(X_transformed, y)
        X_selected = np.asarray(X_selected)

        # 5. Target Encoding
        # ------------------
        if self.is_classification:
            le = LabelEncoder()
            y_transformed = le.fit_transform(y)
            self.label_encoder = le
        else:
            y_transformed = np.asarray(y)

        # Return X_selected as the training data
        return X_selected, y_transformed, preprocessor

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the Random Forest model using Optuna (Tier 3) tuning strategy.
        
        Args:
            X_train: Training features array.
            y_train: Training target array.
        """
        from core_recommender.tuning import run_optuna_optimization

        # 1. Base Strategy
        if self.is_classification:
            base_cls = RandomForestClassifier
            scoring = 'f1_weighted'
            cv = StratifiedKFold(n_splits=self.config.get('cv_folds', 5), shuffle=True, random_state=self.config.get('random_state', 42))
        else:
            base_cls = RandomForestRegressor
            scoring = 'neg_root_mean_squared_error'
            cv = KFold(n_splits=self.config.get('cv_folds', 5), shuffle=True, random_state=self.config.get('random_state', 42))

        # 2. Run Optuna
        print(f"[{self.name}] Starting Training with OPTUNA strategy...")
        
        self.best_estimator = run_optuna_optimization(
            estimator_class=base_cls,
            param_space_func=self._get_optuna_space,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=scoring,
            n_trials=self.config.get('n_trials', 20),
            n_jobs=self.config.get('n_jobs', -1),
            random_state=self.config.get('random_state', 42)
        )
        # Capture study
        if hasattr(self.best_estimator, 'study_'):
            self.study = self.best_estimator.study_

        self.model = self.best_estimator
        print(f"[{self.name}] Best parameters: {self.study.best_params if hasattr(self, 'study') else 'N/A'}")

    def _get_optuna_space(self, trial):
        """Defines the search space for Random Forest."""
        # Config defaults
        n_est_range = self.config.get('n_estimators', [100, 300])
        n_est = trial.suggest_int('n_estimators', min(n_est_range), max(n_est_range))
        
        depth_conf = self.config.get('max_depth', [10, 30])
        # Filter None for ranges (if config has None, replace with generic high int like 50)
        depth_vals = [d for d in depth_conf if d is not None]
        if not depth_vals: depth_vals = [10, 50]
        max_depth = trial.suggest_int('max_depth', min(depth_vals), max(depth_vals))
        
        # Determine max_features space 
        # (Optuna suggests categorical from list, but Scikit RF accepts specific strings or ints)
        max_feat_options = [f for f in self.config.get('max_features', ['sqrt', 'log2']) if f is not None]
        max_features = trial.suggest_categorical('max_features', max_feat_options if max_feat_options else ['sqrt'])
        
        # Min samples (critical for pruning trees)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)

        base_params = {
            'n_estimators': n_est,
            'max_depth': max_depth,
            'max_features': max_features,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': self.config.get('random_state', 42),
            'oob_score': True # Keep this enabled
        }
        
        if self.is_classification:
            base_params['class_weight'] = self.config.get('class_weight', 'balanced')
            
        return base_params

    def calculate_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Calculates task-specific performance metrics.

        Metrics Include:
        - OOB Score (Out-of-Bag estimate)
        - F1 Score, ROC AUC (Classification)
        - RMSE (Regression)

        Args:
            X_test: Test features array.
            y_test: Test target array.
            
        Returns:
            Dict[str, float]: Dictionary of calculated metrics.
        """
        # Note: X_test must be transformed AND selected (via self.feature_selector) before passed here
        # BUT preprocess() returns preprocessor, not selector for external use.
        # This is an architectural challenge. 
        # Implication: The preprocessor returned by preprocess() handles transformation.
        # The selector is internal. 
        # FIX: We must apply selector in calculate_metrics? 
        # Ideally, `preprocess` should return a full pipeline object that includes selection.
        # But our interface returns X_proc, y_proc, preprocessor_obj.
        # Standard fix: We assume X_test passed here is ALREADY processed by the caller 
        # using the preprocessor AND we need to apply selection here manually if not included in preprocessor.
        # Preprocessor (ColumnTransformer) checks feature columns. 
        # SelectFromModel changes shape.
        
        # Apply Feature Selection to X_test if it exists
        if hasattr(self, 'feature_selector'):
            # Check if X_test matches selected shape or original shape?
            # User (execution.py) calls preprocessor.transform(X_test_raw) -> X_test_proc
            # Then passes X_test_proc here.
            # So X_test_proc has shape BEFORE selection.
            # We must transform it.
            try:
                X_test = self.feature_selector.transform(X_test)
            except Exception:
                # Fallback if already transformed or shape mismatch
                pass

        y_pred = self.best_estimator.predict(X_test)
        
        metrics = {}
        
        # Common
        metrics['OOB Score'] = get_oob_score(self.best_estimator)
        
        if self.is_classification:
            # Classification Metrics
            metrics['F1 Score'] = calculate_f1_score(y_test, y_pred, average='weighted')
            # ROC AUC needs proba
            if hasattr(self.best_estimator, "predict_proba"):
                y_proba = self.best_estimator.predict_proba(X_test)
                # Handle binary vs multi
                metrics['ROC AUC'] = calculate_roc_auc_score(y_test, y_proba)
        else:
            # Regression Metrics
            metrics['RMSE'] = calculate_rmse(y_test, y_pred)
            
        return metrics

    def get_diagnostic_data(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Retrieves diagnostic data for visualization.
        
        Includes:
        - Predictions and Truth values
        - Feature Importance (MDI)
        - Single Estimator (Tree) for structure visualization
        - OOB Score
        
        Args:
            X_test: Test features array.
            y_test: Test target array.
            
        Returns:
            Dict[str, Any]: Data dictionary for the visualization module.
            
        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if not hasattr(self, 'best_estimator'):
             raise RuntimeError("Model must be fitted before diagnostics.")
             
        # Apply Selection (same logic as metrics)
        if hasattr(self, 'feature_selector'):
             try:
                X_test = self.feature_selector.transform(X_test)
             except Exception:
                pass

        y_pred = self.best_estimator.predict(X_test)
        
        # Feature Importance (MDI)
        mdi_importance = self.best_estimator.feature_importances_
        
        # Permutation Importance (Computationally expensive, so maybe optional or small n_repeats)
        # Requirement: "Feature Importance Plot (MDI or Permutation Importance)"
        # We'll stick to MDI for speed in default, but permit calc if requested.
        
        # Tree Structure (Extract one estimator)
        single_estimator = self.best_estimator.estimators_[0]

        return {
            'y_pred': y_pred,
            'y_test': y_test,
            'model_name': self.name,
            'feature_importances_mdi': mdi_importance,
            'single_estimator': single_estimator, # For Viz
            'oob_score': get_oob_score(self.best_estimator)
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Retrieves the Mean Decrease in Impurity (MDI) feature importance.
        
        Returns:
            Dict[str, float]: Dictionary containing importance array (indices are keys implicitly via direct return logic).
        """
        if hasattr(self.best_estimator, 'feature_importances_'):
             # Return array, logic to map to names would require knowing feature names here
             return {'importances': self.best_estimator.feature_importances_}
        return {}
