import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Union

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.base import clone
import optuna

# --- PROJECT IMPORTS ---
from core_recommender.modeling.baseModel import BaseModel
from core_recommender.preprocessing import (
    get_imputer,
    get_one_hot_encoder,
    get_select_from_model
)
from core_recommender.evaluation import (
    calculate_f1_score,
    calculate_roc_auc_score,
    calculate_rmse,
    get_oob_score
)

# Import centralized logger
from core_recommender.logger import get_logger
logger = get_logger(__name__)

# --- DEFAULT CONFIGURATION ---
CONFIG = {
    'task_type': 'auto', # 'classification', 'regression', 'auto'
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'max_features': ['sqrt', 'log2', None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'selection_threshold': 'median',
    'cv_folds': 3,
    'random_state': 42,
    'n_jobs': -1,
    'class_weight': 'balanced', # 'balanced', 'balanced_subsample'
    'n_trials': 10
}

# =========================================================================
# RandomForestModel Class
# =========================================================================

class RandomForestModel(BaseModel):
    """A concrete implementation of Random Forest for both Classification and Regression.
    
    Attributes:
        task_type (str): 'classification', 'regression', or 'auto'.
        is_classification (bool): Flag indicating task type.
        label_encoder (LabelEncoder): Encoder for target variable (Classification only).
        best_estimator (Pipeline): The fitted pipeline after tuning.
        study (optuna.Study): The Optuna study object.
    """
    def __init__(self, is_classification: bool = True, config: Dict[str, Any] = CONFIG) -> None:
        """Initializes the Random Forest model.

        Args:
            is_classification (bool): True for classification. Defaults to True.
            config (Dict[str, Any], optional): Hyperparameters. Defaults to GLOBAL config.
        """
        task_name = "Classification" if is_classification else "Regression"
        name = f"Random Forest ({task_name})"
        super().__init__(name=name, config=config)
        
        self.task_type = config.get('task_type', 'auto')
        self.is_classification = is_classification
        self.label_encoder: Optional[LabelEncoder] = None
        self.best_estimator: Optional[Pipeline] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_selector: Any = None

    def _infer_task_type(self, y: pd.Series) -> None:
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
        """Constructs and applies a feature pipeline optimized for Random Forest.
        
        Available steps:
        1. Numerical: Imputation (Median).
        2. Categorical: Imputation (Mode) -> OneHot Encoding.
        3. Feature Selection: Embedded Random Forest selection.
        
        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, ColumnTransformer]: X_selected, y_transformed, preprocessor.
            
        Raises:
            ValueError: If target variable 'y' contains NaNs.
        """
        logger.debug(f"[{self.name}] Entering preprocess()...")
        
        if pd.isna(y).any():
            logger.error(f"[{self.name}] Target variable contains {pd.isna(y).sum()} NaN values")
            raise ValueError("Target variable 'y' contains missing values.")

        self._infer_task_type(y)
        
        # 1. Pipeline Construction
        num_steps = [('imputer', get_imputer(strategy='median'))]
        numerical_pipeline = Pipeline(steps=num_steps)

        cat_steps = [
            ('imputer', get_imputer(strategy='most_frequent')),
            ('onehot', get_one_hot_encoder(handle_unknown='ignore', sparse_output=False))
        ]
        cat_pipeline = Pipeline(steps=cat_steps)

        # 2. Composition
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
        X_transformed = preprocessor.fit_transform(X, y)
        X_transformed = np.asarray(X_transformed)

        # 4. Feature Selection (SelectFromModel)
        if self.is_classification:
            sel_est = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        else:
            sel_est = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            
        self.feature_selector = get_select_from_model(
            estimator=sel_est,
            threshold=self.config.get('selection_threshold', 'median')
        )
        
        logger.debug(f"[{self.name}] Applying SelectFromModel for feature selection...")
        X_selected = self.feature_selector.fit_transform(X_transformed, y)
        X_selected = np.asarray(X_selected)
        n_features_before = X_transformed.shape[1]
        n_features_after = X_selected.shape[1]
        logger.info(f"[{self.name}] Feature selection: {n_features_before} -> {n_features_after} features")

        # 5. Target Encoding
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
            self.label_encoder = None

        return X_selected, y_transformed, preprocessor

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> None:
        """Trains the Random Forest model using Pipeline-based Optuna optimization.
        
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (np.ndarray): Training targets.
        """
        logger.info(f"[{self.name}] Starting training...")
        
        if self.preprocessor is None:
             raise RuntimeError("Preprocessor not initialized.")

        # 1. Pipeline Prep
        preprocessor_template = clone(self.preprocessor)

        if self.is_classification:
            base_cls = RandomForestClassifier
            scoring = 'f1_weighted'
            cv = StratifiedKFold(n_splits=self.config.get('cv_folds', 5), shuffle=True, random_state=self.config.get('random_state', 42))
        else:
            base_cls = RandomForestRegressor
            scoring = 'neg_root_mean_squared_error'
            cv = KFold(n_splits=self.config.get('cv_folds', 5), shuffle=True, random_state=self.config.get('random_state', 42))

        # 2. Pipeline Objective
        def pipeline_objective(trial):
            params = self._get_optuna_space(trial)
            model_inst = base_cls(**params)
            
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
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)  # Sequential CV to avoid nested parallelism
            return scores.mean()

        # 3. Optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(pipeline_objective, n_trials=self.config.get('n_trials', 10))
        
        # 4. Build Best Model
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
        
        logger.info(f"✅ [{self.name}] Training complete. Best Score: {study.best_value:.4f}")

    def _get_optuna_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Defines the search space for Optuna."""
        n_est_range = self.config.get('n_estimators', [100, 300])
        n_est = trial.suggest_int('n_estimators', min(n_est_range), max(n_est_range))
        
        depth_conf = self.config.get('max_depth', [10, 30])
        depth_vals = [d for d in depth_conf if d is not None]
        if not depth_vals: depth_vals = [10, 50]
        max_depth = trial.suggest_int('max_depth', min(depth_vals), max(depth_vals))
        
        max_feat_options = [f for f in self.config.get('max_features', ['sqrt', 'log2']) if f is not None]
        max_features = trial.suggest_categorical('max_features', max_feat_options if max_feat_options else ['sqrt'])
        
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
        """Calculates performance metrics.
        
        Args:
            X_test (pd.DataFrame): Test features.
            y_test (np.ndarray): Test targets.

        Returns:
            Dict[str, float]: Performance metrics (F1/ROC for Classif, RMSE for Reg) + OOB Score.
        """
        if self.best_estimator is None:
             raise RuntimeError("Model must be fitted before calculating metrics.")
             
        y_pred = self.best_estimator.predict(X_test)
        
        metrics = {}
        
        final_model = self.best_estimator.named_steps['model']
        metrics['OOB Score'] = get_oob_score(final_model)
        
        if self.is_classification:
            metrics['F1 Score'] = calculate_f1_score(y_test, y_pred, average='weighted')
            if hasattr(final_model, "predict_proba"):
                y_proba = self.best_estimator.predict_proba(X_test)
                metrics['ROC AUC'] = calculate_roc_auc_score(y_test, y_proba)
        else:
            metrics['RMSE'] = calculate_rmse(y_test, y_pred)
            
        return metrics

    def get_diagnostic_data(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """Retrieves diagnostic data for visualization."""
        if self.best_estimator is None:
             raise RuntimeError("Model must be fitted before diagnostics.")
             
        y_pred = self.best_estimator.predict(X_test)
        
        final_model = self.best_estimator.named_steps['model']
        mdi_importance = final_model.feature_importances_
        single_estimator = final_model.estimators_[0]

        return {
            'y_pred': y_pred,
            'y_true': y_test,
            'y_proba': self.best_estimator.predict_proba(X_test) if self.is_classification else None,
            'model_name': self.name,
            'feature_importances_mdi': mdi_importance,
            'single_estimator': single_estimator,
            'oob_score': get_oob_score(final_model)
        }

    def get_feature_importance(self) -> Dict[str, Any]:
        """Retrieves the Mean Decrease in Impurity (MDI)."""
        if self.best_estimator is None:
            return {}
            
        final_model = self.best_estimator.named_steps['model']
        if hasattr(final_model, 'feature_importances_'):
             return {'importances': final_model.feature_importances_.tolist()}
        return {}
    
    def get_parameter_descriptions(self) -> Dict[str, Dict[str, str]]:
        """
        Extracts and describes the final tuned hyperparameters of the Random Forest.
        
        This method retrieves the 'best_params' from the completed Optuna study,
        mapping the internal technical values to a structured dictionary for the 
        dashboard's "Model DNA" section.
        
        Returns:
            Dict[str, Dict[str, str]]: A dictionary mapping parameter names to their 
            values and technical descriptions.
        """
        if hasattr(self, 'study') and self.study:
            best_params = self.study.best_params
            return {
                'n_estimators': {
                    'value': str(best_params.get('n_estimators')), 
                    'desc': 'The number of individual decision trees in the ensemble forest.'
                },
                'max_depth': {
                    'value': str(best_params.get('max_depth')), 
                    'desc': 'Maximum vertical depth allowed for each tree, controlling model complexity and overfitting.'
                },
                'min_samples_split': {
                    'value': str(best_params.get('min_samples_split')), 
                    'desc': 'The minimum number of data samples required to trigger a split in an internal node.'
                }
            }
        return {}
