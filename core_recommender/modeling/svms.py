import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Union

from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
import optuna

# --- PROJECT IMPORTS ---
from core_recommender.modeling.base_model import BaseModel
from core_recommender.modeling.registry import register_model
from core_recommender.preprocessing import (
    get_imputer,
    get_one_hot_encoder,
    get_standard_scaler,
    get_minmax_scaler,
    get_pca_reducer
)
from core_recommender.evaluation import (
    calculate_accuracy,
    calculate_f1_score,
    calculate_rmse,
    calculate_r2_score
)

# Import centralized logger
from core_recommender.logger import get_logger
logger = get_logger(__name__)

# --- DEFAULT CONFIGURATION ---
CONFIG = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'scaler': 'standard',     
    'pca_components': 0.95,   
    'cv_folds': 3,
    'n_iter': 10,             
    'random_state': 42,
    'n_jobs': -1
}

# =========================================================================
# SVMModel Class
# =========================================================================

@register_model(task='both')
class SVMModel(BaseModel):
    """A concrete implementation of Support Vector Machines (SVM) for Classification and Regression.
    
    Attributes:
        is_classification (bool): Flag indicating task type.
        label_encoder (LabelEncoder): Encoder for target variable (Classification only).
        best_estimator (Pipeline): The fitted pipeline after tuning.
        study (optuna.Study): The Optuna study object.
    """
    def __init__(self, is_classification: bool = True, config: Dict[str, Any] = CONFIG) -> None:
        """Initializes the SVM model.

        Args:
            is_classification (bool, optional): True for classification. Defaults to True.
            config (Dict[str, Any], optional): Hyperparameters. Defaults to GLOBAL config.
        """
        task_name = "Classification" if is_classification else "Regression"
        name = f"SVM ({task_name})"
        super().__init__(name=name, config=config)
        
        self.is_classification = is_classification
        self.label_encoder: Optional[LabelEncoder] = None
        self.best_estimator: Optional[Pipeline] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        
        # Initialize Model Instance & Param Grid
        if self.is_classification:
            self.model_instance = SVC(class_weight='balanced', probability=True, random_state=config.get('random_state', 42))
        else:
            self.model_instance = SVR()

        self.param_distributions = {
            'C': config.get('C', [0.1, 1, 10, 100]),
            'kernel': config.get('kernel', ['linear', 'rbf']),
            'gamma': config.get('gamma', ['scale', 'auto']),
        }

    def preprocess(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
        """Constructs and applies a robust feature pipeline for SVMs.
        
        Pipeline Steps:
        1. Numerical: Median imputation -> Scaling (Standard/MinMax) -> PCA (Optional).
        2. Categorical: Most frequent imputation -> One-Hot Encoding.
        
        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, ColumnTransformer]: X_transformed, y_transformed, preprocessor.
            
        Raises:
            ValueError: If target variable 'y' contains NaNs.
        """
        logger.debug(f"[{self.name}] Entering preprocess()...")
        
        # 1. Pipeline Construction
        # Numerical Steps
        num_steps: List[Tuple[str, Any]] = [('imputer', get_imputer(strategy='median'))]
        
        if self.config.get('scaler') == 'minmax':
            num_steps.append(('scaler', get_minmax_scaler()))
        else:
            num_steps.append(('scaler', get_standard_scaler())) 

        # PCA (Dimensionality Reduction)
        if self.config.get('pca_components') is not None:
            num_steps.append(('pca', get_pca_reducer(n_components=self.config.get('pca_components', 0.95))))

        numerical_pipeline = Pipeline(steps=num_steps)

        # Categorical Steps
        cat_steps: List[Tuple[str, Any]] = [
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

        X_transformed = preprocessor.fit_transform(X, y)
        X_transformed = np.asarray(X_transformed)

        # 4. Target Encoding
        if self.is_classification:
            if not np.issubdtype(y.dtype, np.number):
                le = LabelEncoder()
                y_transformed = le.fit_transform(y)
                self.label_encoder = le
            else:
                y_transformed = y.values
                self.label_encoder = None
        else:
            y_transformed = y.values
            self.label_encoder = None
            
        return X_transformed, y_transformed, preprocessor

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> None:
        """Trains the SVM model using Pipeline-based Optuna optimization.
        
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (np.ndarray): Training targets.
        """
        logger.info(f"[{self.name}] Starting training...")
        
        if self.preprocessor is None:
             raise RuntimeError("Preprocessor not initialized.")

        # 1. Base Strategy
        if self.is_classification:
            base_cls = SVC
            scoring = 'f1_weighted' 
            cv = StratifiedKFold(n_splits=self.config.get('cv_folds', 5), shuffle=True, random_state=self.config.get('random_state', 42))
        else:
            base_cls = SVR
            scoring = 'neg_root_mean_squared_error'
            cv = KFold(n_splits=self.config.get('cv_folds', 5), shuffle=True, random_state=self.config.get('random_state', 42))

        # 2. Pipeline-based Tuning
        preprocessor_template = clone(self.preprocessor)
        
        def pipeline_objective(trial):
            params = self._get_optuna_space(trial)
            model_inst = base_cls(**params)
            
            pipe = Pipeline(steps=[
                ('pre', preprocessor_template),
                ('model', model_inst)
            ])
            
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)  # Sequential CV to avoid nested parallelism
            return scores.mean()

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(pipeline_objective, n_trials=self.config.get('n_iter', 10))
        
        # 3. Build Best Model
        best_params = study.best_params
        # Ensure we keep the fixed parameters that aren't tuned
        if self.is_classification:
            best_params['probability'] = True
            best_params['class_weight'] = 'balanced'
            best_params['random_state'] = self.config.get('random_state', 42)
            
        best_model_inst = base_cls(**best_params)
        
        self.best_estimator = Pipeline(steps=[
            ('pre', preprocessor_template),
            ('model', best_model_inst)
        ])
        self.best_estimator.fit(X_train, y_train)
        
        self.study = study
        self.model = self.best_estimator
        
        logger.info(f"✅ [{self.name}] Training complete. Best Score: {study.best_value:.4f}")

    def _get_optuna_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Defines the search space for SVM."""
        # Kernels - limit to rbf/linear for speed unless poly requested
        k_options = self.config.get('kernel', ['linear', 'rbf'])
        # Filter generic lists to available ones
        k_options = [k for k in k_options if k in ['linear', 'rbf', 'poly', 'sigmoid']]
        kernel = trial.suggest_categorical('kernel', k_options)
        
        # C (Regularization) - Log scale
        c_range = self.config.get('C', [0.1, 100])
        # If list of specific values is passed instead of range, utilize categorical
        if isinstance(c_range, list) and len(c_range) > 2 and not isinstance(c_range[0], (int, float)):
             # fallback or specific list
             C = trial.suggest_categorical('C', c_range)
        else:
             # assume range or generic list of floats
             c_min, c_max = min(c_range), max(c_range)
             C = trial.suggest_float('C', c_min, c_max, log=True)
        
        # Gamma
        gamma_options = [g for g in self.config.get('gamma', ['scale', 'auto']) if isinstance(g, str)]
        if not gamma_options: gamma_options = ['scale']
        gamma = trial.suggest_categorical('gamma', gamma_options)

        params = {
            'C': C,
            'kernel': kernel,
            'gamma': gamma
        }
        
        if self.is_classification:
            params['class_weight'] = 'balanced'
            params['probability'] = True 
            params['random_state'] = self.config.get('random_state', 42)
            
        return params

    def calculate_metrics(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """Calculates performance metrics.
        
        Args:
            X_test (pd.DataFrame): Test features.
            y_test (np.ndarray): Test targets.

        Returns:
            Dict[str, float]: F1/Accuracy for Classif, RMSE/R2 for Reg.
        """
        if self.best_estimator is None:
             raise RuntimeError("Model must be fitted before calculating metrics.")
             
        y_pred = self.best_estimator.predict(X_test)
        
        metrics = {}
        if self.is_classification:
            metrics['Accuracy'] = calculate_accuracy(y_test, y_pred) 
            metrics['F1 Score'] = calculate_f1_score(y_test, y_pred, average='weighted')
        else:
            metrics['RMSE'] = calculate_rmse(y_test, y_pred)
            metrics['R2 Score'] = calculate_r2_score(y_test, y_pred)
        
        return metrics

    def get_diagnostic_data(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """Retrieves diagnostic data for visualization."""
        if self.best_estimator is None:
             raise RuntimeError("Model must be fitted before diagnostics.")
        
        y_pred = self.best_estimator.predict(X_test)
        final_model = self.best_estimator.named_steps['model']
        
        data = {
            'y_pred': y_pred,
            'y_true': y_test,
            'model_name': self.name,
            'support_vectors': final_model.support_vectors_,
            'n_support': final_model.n_support_ if hasattr(final_model, 'n_support_') else None,
            'is_classification': self.is_classification
        }

        if self.is_classification and hasattr(final_model, 'predict_proba'):
            data['y_proba'] = self.best_estimator.predict_proba(X_test)

        return data
    
    def get_tailored_diagnostics(self) -> Dict[str, Any]:
        """SVM-specific diagnostics.

        Returns:
            Dict[str, Any]:
                * ``support_vectors``  — The training examples that lie on or
                  within the margin (the most influential data points).
                * ``n_support``        — Number of support vectors per class
                  (classification only).
                * ``coefficients``     — Linear SVM decision weights (only
                  available when ``kernel == 'linear'``).
        """
        if self.best_estimator is None:
            return {}

        final_model = self.best_estimator.named_steps['model']
        diagnostics: Dict[str, Any] = {
            'support_vectors': (
                final_model.support_vectors_.tolist()
                if hasattr(final_model, 'support_vectors_')
                else None
            ),
            'n_support': (
                final_model.n_support_.tolist()
                if hasattr(final_model, 'n_support_')
                else None
            ),
        }

        # Linear kernel → expose coefficients as a bonus
        if getattr(final_model, 'kernel', '') == 'linear' and hasattr(final_model, 'coef_'):
            coefs = final_model.coef_
            if coefs.ndim > 1:
                coefs = coefs[0]
            diagnostics['coefficients'] = coefs.tolist()

        return diagnostics

    def get_parameter_descriptions(self) -> Dict[str, Dict[str, str]]:
        """
        Extracts and describes the final tuned hyperparameters of the SVM model.
        
        This method retrieves the 'best_params' from the completed Optuna study,
        providing a human-readable bridge between technical parameters (Model DNA)
        and the dashboard UI.
        
        Returns:
            Dict[str, Dict[str, str]]: A dictionary mapping parameter names to their 
            values and technical descriptions.
        """
        if hasattr(self, 'study') and self.study:
            best_params = self.study.best_params
            return {
                'C': {
                    'value': f"{best_params.get('C', 0):.4f}", 
                    'desc': 'Regularization strength (Inverse). Higher values prioritize training accuracy over margin width.'
                },
                'kernel': {
                    'value': str(best_params.get('kernel')), 
                    'desc': 'The mathematical function used to map input data into a high-dimensional feature space.'
                },
                'gamma': {
                    'value': str(best_params.get('gamma')), 
                    'desc': 'Kernel coefficient. Defines how far the influence of a single training example reaches.'
                }
            }
        return {}
