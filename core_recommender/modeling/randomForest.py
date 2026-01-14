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

# Import centralized logger
from core_recommender.logger import get_logger
logger = get_logger(__name__)

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
    
    Overview:
    ---------
    Random Forest is an ensemble learning method that constructs a multitude of decision trees at 
    training time. For classification tasks, the output is the class selected by most trees (mode). 
    For regression tasks, the mean or average prediction of the individual trees is returned.
    
    Key Features:
    -------------
    - **Auto-Task Inference**: Automatically detects Classification vs Regression based on target data types.
    - **Feature Selection**: Integrated `SelectFromModel` step using a lightweight forest to discard 
      irrelevant features before the main training loop.
    - **Robustness**: Handles high-dimensional data and multicollinearity well.
    - **OOB Error**: uses Out-of-Bag samples to estimate generalization error without a separate validation set.

    Configuration (`CONFIG`):
    -------------------------
    - `n_estimators`: List[int] - Number of trees in the forest.
    - `max_depth`: List[int] - Maximum depth of the tree.
    - `max_features`: List[str|int] - Number of features to consider when looking for the best split.
    - `min_samples_leaf`: List[int] - Minimum samples required to be at a leaf node.
    - `selection_threshold`: str|float - Threshold for feature selection (e.g., 'median', 'mean').
    """
    def __init__(self, is_classification: bool = True, config: Dict[str, Any] = CONFIG):
        """
        Initializes the Random Forest model.

        Args:
            is_classification (bool): 
                - True for classification tasks.
                - False for regression.
                - Note: This flag may be overridden by `_infer_task_type` if `task_type` is 'auto'.
            config (Dict[str, Any]): 
                Dictionary containing hyperparameters (e.g., 'n_estimators', 'max_depth').
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
        Constructs and applies a feature pipeline optimized for Random Forest, including an embedded 
        feature selection step.
        
        Rationale:
        ----------
        - **Scaling**: Tree-based models are invariant to monotonic transformations, so scaling is 
          generally not required (unlike SVM/PCA).
        - **Imputation**: Missing values are imputed (Median for numerical, Most Frequent for categorical).
        - **Encoding**: One-Hot Encoding is used for categoricals to make them compatible with Scikit-Learn's 
          Random Forest implementation (which technically requires numeric input).
        - **Feature Selection**: A preliminary Random Forest is trained to determine feature importance. 
          Features below the `selection_threshold` (e.g., 'median') are discarded to reduce noise 
          and improve training speed.
        
        Args:
            X (pd.DataFrame): Input features DataFrame.
            y (pd.Series): Target Series.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
                - **X_selected**: Numpy array of features *after* selection.
                - **y_transformed**: Transformed target array.
                - **preprocessor**: The fitted `ColumnTransformer` (Note: This handles transformation, 
                  but the *Selector* is a separate object stored in `self.feature_selector`).
            
        Raises:
            ValueError: If target variable 'y' contains NaNs.
        """
        logger.debug(f"[{self.name}] Entering preprocess()...")
        logger.debug(f"[{self.name}] Input shape: X={X.shape}, y={y.shape}")
        
        if y.isna().any():
            logger.error(f"[{self.name}] Target variable contains {y.isna().sum()} NaN values")
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
        
        self.preprocessor = preprocessor
        
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
        logger.debug(f"[{self.name}] Applying SelectFromModel for feature selection...")
        X_selected = self.feature_selector.fit_transform(X_transformed, y)
        X_selected = np.asarray(X_selected)
        n_features_before = X_transformed.shape[1]
        n_features_after = X_selected.shape[1]
        logger.info(f"[{self.name}] Feature selection: {n_features_before} → {n_features_after} features (threshold={self.config.get('selection_threshold', 'median')})")

        # 5. Target Encoding (Skip if already numeric/pre-encoded by Executor)
        if self.is_classification:
            if not np.issubdtype(y.dtype, np.number):
                le = LabelEncoder()
                y_transformed = le.fit_transform(y)
                self.label_encoder = le
            else:
                y_transformed = y.values
                self.label_encoder = None
        else:
            y_transformed = np.asarray(y)

        # Return X_selected as the training data
        logger.debug(f"[{self.name}] Preprocessing complete. Output shape: X_selected={X_selected.shape}, y_transformed={y_transformed.shape}")
        return X_selected, y_transformed, preprocessor

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """
        Trains the Random Forest model using a Unified Pipeline to prevent data leakage.
        """
        logger.info(f"[{self.name}] Starting training...")
        logger.debug(f"[{self.name}] Training data shape: X={X_train.shape}, y={y_train.shape}")
        
        from core_recommender.tuning import run_optuna_optimization
        from sklearn.base import clone

        # 1. Create a clone of the preprocessor template
        # We use a fresh clone to ensure no state from previous fits (if any) persists
        preprocessor_template = clone(self.preprocessor)

        # 2. Base Strategy
        if self.is_classification:
            base_cls = RandomForestClassifier
            scoring = 'f1_weighted'
            cv = StratifiedKFold(n_splits=self.config.get('cv_folds', 5), shuffle=True, random_state=self.config.get('random_state', 42))
            logger.debug(f"[{self.name}] Task: Classification. Scoring: {scoring}, CV: StratifiedKFold (n_splits={self.config.get('cv_folds', 5)})")
        else:
            base_cls = RandomForestRegressor
            scoring = 'neg_root_mean_squared_error'
            cv = KFold(n_splits=self.config.get('cv_folds', 5), shuffle=True, random_state=self.config.get('random_state', 42))
            logger.debug(f"[{self.name}] Task: Regression. Scoring: {scoring}, CV: KFold (n_splits={self.config.get('cv_folds', 5)})")

        # 3. Define Pipeline-based Tuning Strategy
        # To prevent leakage, we wrap the preprocessor and model in a Pipeline
        # and pass THIS to cross_val_score or the tuner.
        
        def pipeline_objective(trial):
            params = self._get_optuna_space(trial)
            model_inst = base_cls(**params)
            
            # Create full pipeline for this trial
            # Note: We include SelectFromModel if configured
            pipeline_steps = [('pre', preprocessor_template)]
            
            # Add selector if threshold is provided
            if self.config.get('selection_threshold'):
                 if self.is_classification:
                     sel_est = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                 else:
                     sel_est = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                 pipeline_steps.append(('selector', get_select_from_model(estimator=sel_est, threshold=self.config.get('selection_threshold'))))
                 
            pipeline_steps.append(('model', model_inst))
            pipe = Pipeline(steps=pipeline_steps)
            
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=self.config.get('n_jobs', -1))
            return scores.mean()

        # 4. Run Optuna Optimization manually or via tuning.py 
        # (tuning.py's run_optuna_optimization is built for raw estimates, we'll adapt or use it)
        # For simplicity and perfect leakage control, we'll do the study here or update tuning.py
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logger.debug(f"[{self.name}] Starting Optuna optimization (n_trials={self.config.get('n_trials', 15)})...")
        study = optuna.create_study(direction='maximize')
        study.optimize(pipeline_objective, n_trials=self.config.get('n_trials', 15))
        
        # 5. Build Best Model (Full Pipeline)
        best_params = study.best_params
        best_model_inst = base_cls(**best_params)
        
        final_steps = [('pre', preprocessor_template)]
        if self.config.get('selection_threshold'):
             if self.is_classification:
                 sel_est = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
             else:
                 sel_est = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
             final_steps.append(('selector', get_select_from_model(estimator=sel_est, threshold=self.config.get('selection_threshold'))))
        
        final_steps.append(('model', best_model_inst))
        
        self.best_estimator = Pipeline(steps=final_steps)
        self.best_estimator.fit(X_train, y_train)
        
        self.study = study
        self.model = self.best_estimator
        
        best_score = study.best_value
        best_params = study.best_params
        logger.info(f"✅ [{self.name}] Training complete")
        logger.info(f"[{self.name}] Best CV Score: {best_score:.4f}")
        logger.debug(f"[{self.name}] Best params: {best_params}")

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

    def calculate_metrics(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """
        Calculates task-specific performance metrics using the full Pipeline.
        """
        # We use the full pipeline (best_estimator) which handles pre -> selector -> model
        y_pred = self.best_estimator.predict(X_test)
        
        metrics = {}
        
        # Access the model step from pipeline for OOB score
        final_model = self.best_estimator.named_steps['model']
        metrics['OOB Score'] = get_oob_score(final_model)
        
        if self.is_classification:
            metrics['F1 Score'] = calculate_f1_score(y_test, y_pred, average='weighted')
            if hasattr(final_model, "predict_proba"):
                # Proba needs preprocessing/selectors too
                y_proba = self.best_estimator.predict_proba(X_test)
                metrics['ROC AUC'] = calculate_roc_auc_score(y_test, y_proba)
        else:
            metrics['RMSE'] = calculate_rmse(y_test, y_pred)
            
        return metrics

    def get_diagnostic_data(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Retrieves diagnostic data for visualization using the Pipeline.
        """
        if not hasattr(self, 'best_estimator'):
             raise RuntimeError("Model must be fitted before diagnostics.")
             
        y_pred = self.best_estimator.predict(X_test)
        
        # Access steps for specialized data
        final_model = self.best_estimator.named_steps['model']
        mdi_importance = final_model.feature_importances_
        single_estimator = final_model.estimators_[0]

        return {
            'y_pred': y_pred,
            'y_test': y_test,
            'model_name': self.name,
            'feature_importances_mdi': mdi_importance,
            'single_estimator': single_estimator,
            'oob_score': get_oob_score(final_model)
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Retrieves the Mean Decrease in Impurity (MDI) feature importance.
        """
        final_model = self.best_estimator.named_steps['model']
        if hasattr(final_model, 'feature_importances_'):
             return {'importances': final_model.feature_importances_.tolist()}
        return {}

    def get_parameter_descriptions(self) -> Dict[str, Dict[str, str]]:
        """
        Returns descriptions of the most important tuned parameters.
        """
        final_model = self.best_estimator.named_steps['model']
        params = final_model.get_params()
        descriptions = {
            'n_estimators': {
                'value': str(params.get('n_estimators')),
                'desc': 'The number of decision trees in the forest. More trees usually improve performance but increase computation time.'
            },
            'max_depth': {
                'value': str(params.get('max_depth')),
                'desc': 'The maximum depth of each tree. Limiting depth prevents the model from over-fitting to specific details.'
            },
            'min_samples_split': {
                'value': str(params.get('min_samples_split')),
                'desc': 'The minimum number of samples required to split an internal node. Higher values make the model more conservative.'
            }
        }
        return descriptions
