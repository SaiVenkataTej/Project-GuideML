import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Union

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone

# --- PROJECT IMPORTS ---
from core_recommender.modeling.base_model import BaseModel
from core_recommender.modeling.registry import register_model
from core_recommender.preprocessing import (
    get_imputer,
    get_one_hot_encoder,
    get_standard_scaler,
    get_minmax_scaler,
    get_yeo_johnson_transformer,
    get_select_k_best
)
from core_recommender.evaluation import (
    calculate_accuracy,
    calculate_f1_score,
    calculate_log_loss,
    calculate_precision
)

# Import centralized logger
from core_recommender.logger import get_logger
from core_recommender.exceptions import DataValidationError, ConfigurationError
logger = get_logger(__name__)

# --- DEFAULT CONFIGURATION ---
CONFIG = {
    'model_type': 'gaussian', # 'gaussian', 'multinomial'
    'scaler': 'standard',     # 'standard', 'minmax'
    'feature_selection': 'k_best',
    'k_best': 10,
    'k_best_score_func': 'f_classif', # 'f_classif', 'chi2', 'mutual_info_classif'
    'cv_folds': 3,
    'random_state': 42,
    'n_jobs': -1
}

# =========================================================================
# NaiveBayesModel Class
# =========================================================================

@register_model(task='classification')
class NaiveBayesModel(BaseModel):
    """A concrete implementation of Naive Bayes (Gaussian, Multinomial) for Classification.
    
    Attributes:
        model_type (str): 'gaussian' or 'multinomial'.
        best_estimator (Pipeline): The fitted pipeline after tuning.
        label_encoder (LabelEncoder): Encoder for target variable.
    """
    def __init__(self, config: Dict[str, Any] = CONFIG) -> None:
        """Initializes the Naive Bayes model.

        Args:
            config (Dict[str, Any], optional): Hyperparameters. Defaults to GLOBAL config.
        """
        name = f"Naive Bayes ({config.get('model_type', 'gaussian').capitalize()})"
        super().__init__(name=name, config=config)
        
        self.model_type = config.get('model_type', 'gaussian')
        self.best_estimator: Optional[Pipeline] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        
        # Initialize Model Instance based on type
        if self.model_type == 'gaussian':
            # var_smoothing will be grid searched (handles numerical stability)
            self.model_instance = GaussianNB()
            self.param_grid = {
                'var_smoothing': np.logspace(0, -9, num=10) 
            }
        elif self.model_type == 'multinomial':
            # alpha (smoothing parameter) will be grid searched
            self.model_instance = MultinomialNB()
            self.param_grid = {
                'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
            }
        else:
            raise ConfigurationError(
                f"Unsupported model_type: '{self.model_type}'. "
                f"Expected 'gaussian' or 'multinomial'."
            )

    def preprocess(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
        """Constructs and applies the feature pipeline optimized for Naive Bayes.
        
        Pipeline Steps:
        1. Numerical (Gaussian): Mean imputation -> Yeo-Johnson transformation -> Scaling.
        2. Numerical (Multinomial): Median imputation -> MinMax Scaling.
        3. Feature Selection: Optional K-Best.
        4. Categorical: Most Frequent Imputation -> One-Hot Encoding.
        
        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, ColumnTransformer]: X_transformed, y_transformed, preprocessor.
            
        Raises:
            ValueError: If target variable 'y' contains NaNs.
        """
        logger.debug(f"[{self.name}] Entering preprocess()...")
        
        # Check for NaNs in target
        if y.isna().any():
            logger.error(f"[{self.name}] Target variable contains {y.isna().sum()} NaN values")
            raise DataValidationError(
                "Target variable contains NaN values. Handle missing targets before training.",
                column='y'
            )

        # 1. Pipeline Construction
        # Numerical Steps
        num_steps: List[Tuple[str, Any]] = []
        if self.model_type == 'gaussian':
            num_steps.append(('imputer', get_imputer(strategy='mean'))) 
            num_steps.append(('yeo_johnson', get_yeo_johnson_transformer()))
            
            if self.config.get('scaler') == 'minmax':
                num_steps.append(('scaler', get_minmax_scaler()))
            else:
                num_steps.append(('scaler', get_standard_scaler()))
                
        elif self.model_type == 'multinomial':
            num_steps.append(('imputer', get_imputer(strategy='median')))
            num_steps.append(('scaler', get_minmax_scaler())) 

        # Feature Selection
        if self.config.get('feature_selection') == 'k_best':
            num_steps.append(('select_k_best', get_select_k_best(
                k=self.config.get('k_best', 10), 
                score_func=self.config.get('k_best_score_func', 'f_classif')
            )))

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
        if not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y_transformed = le.fit_transform(y)
            self.label_encoder = le
        else:
            y_transformed = y.values
            self.label_encoder = None

        return X_transformed, y_transformed, preprocessor

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> None:
        """Trains the Naive Bayes model using a Unified Pipeline.
        
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (np.ndarray): Training targets.
        """
        logger.info(f"[{self.name}] Starting training...")
        
        if self.preprocessor is None:
             raise RuntimeError("Preprocessor not initialized.")

        # 1. Pipeline Construction
        preprocessor_template = clone(self.preprocessor)
        pipe = Pipeline(steps=[
            ('pre', preprocessor_template),
            ('model', self.model_instance)
        ])

        # 2. Adjust Param Grid
        pipeline_params = {f'model__{k}': v for k, v in self.param_grid.items()}

        # 3. CV Strategy
        cv = StratifiedKFold(
            n_splits=self.config.get('cv_folds', 5),
            shuffle=True,
            random_state=self.config.get('random_state', 42)
        )

        # 4. GridSearch on Pipeline
        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=pipeline_params,
            scoring='f1_weighted',
            cv=cv,
            n_jobs=self.config.get('n_jobs', -1),
            verbose=0
        )

        grid_search.fit(X_train, y_train)
        
        self.best_estimator = grid_search.best_estimator_
        self.model = grid_search
        
        logger.info(f"✅ [{self.name}] Training complete. Best CV Score: {grid_search.best_score_:.4f}")

    def calculate_metrics(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """Calculates performance metrics.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (np.ndarray): Test targets.

        Returns:
            Dict[str, float]: Accuracy, F1, Precision, Log Loss.
        """
        if self.best_estimator is None:
            raise RuntimeError("Model must be fitted before calculating metrics.")

        y_pred = self.best_estimator.predict(X_test)
        y_proba = self.best_estimator.predict_proba(X_test)
        
        metrics = {}
        metrics['Accuracy'] = calculate_accuracy(y_test, y_pred)
        metrics['F1 Score'] = calculate_f1_score(y_test, y_pred, average='weighted')
        metrics['Precision'] = calculate_precision(y_test, y_pred, average='weighted')
        metrics['Log Loss'] = calculate_log_loss(y_test, y_proba)
        
        return metrics

    def get_diagnostic_data(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """Retrieves diagnostic data for visualization."""
        if self.best_estimator is None:
             raise RuntimeError("Model must be fitted before diagnostics.")

        y_pred = self.best_estimator.predict(X_test)
        y_proba = self.best_estimator.predict_proba(X_test)

        final_model = self.best_estimator.named_steps['model']

        return {
            'y_pred': y_pred,
            'y_proba': y_proba,
            'y_true': y_test,
            'model_name': self.name,
            'feature_log_prob': getattr(final_model, 'feature_log_prob_', None)
        }
    
    def get_tailored_diagnostics(self) -> Dict[str, Any]:
        """Naive-Bayes-specific diagnostics.

        Returns:
            Dict[str, Any]:
                * ``feature_log_prob``    — Log-probability of each feature
                  per class (Multinomial) or ``None`` for Gaussian variants.
                  Shape: (n_classes, n_features).  Useful for understanding
                  which features are most discriminative per class.
                * ``class_log_prior``     — Log prior probability of each
                  class.  Reflects class imbalance captured by the model.
                * ``model_type``          — ``'gaussian'`` or ``'multinomial'``.
        """
        if self.best_estimator is None:
            return {}

        final_model = self.best_estimator.named_steps['model']
        return {
            'model_type': self.model_type,
            'feature_log_prob': (
                final_model.feature_log_prob_.tolist()
                if hasattr(final_model, 'feature_log_prob_')
                else None
            ),
            'class_log_prior': (
                final_model.class_log_prior_.tolist()
                if hasattr(final_model, 'class_log_prior_')
                else None
            ),
        }

    def get_parameter_descriptions(self) -> Dict[str, Dict[str, str]]:
        """Returns descriptions of the most important tuned parameters."""
        if hasattr(self.model, 'best_params_'):
            best_params = self.model.best_params_
            if self.model_type == 'gaussian':
                return {
                    'var_smoothing': {
                        'value': f"{best_params.get('model__var_smoothing'):.2e}",
                        'desc': 'Portion of largest variance added to variances for stability.'
                    }
                }
            else:
                return {
                    'alpha': {
                        'value': str(best_params.get('model__alpha')),
                        'desc': 'Additive smoothing parameter.'
                    }
                }
        return {}
