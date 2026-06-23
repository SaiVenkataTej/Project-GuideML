import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Union

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone

# --- PROJECT IMPORTS ---
from core_recommender.modeling.baseModel import BaseModel
from core_recommender.modeling.registry import register_model
from core_recommender.preprocessing import (
    get_imputer, 
    get_one_hot_encoder, 
    get_standard_scaler, 
    get_rfe_selector, 
    get_select_from_model
)
from core_recommender.evaluation import (
    calculate_accuracy,
    calculate_f1_score,
    calculate_roc_auc_score,
    calculate_log_loss,
    calculate_precision_recall_score, 
    measure_prediction_latency
)

# Import centralized logger
from core_recommender.logger import get_logger
from core_recommender.exceptions import DataValidationError
logger = get_logger(__name__)

# --- DEFAULT CONFIGURATION ---
CONFIG = {
    'scaler': 'standard',
    'feature_selection': 'rfe',  # 'rfe', 'model_based', 'none'
    'n_features_to_select': 10,  # For RFE
    'cv_folds': 3,
    'random_state': 42,
    'n_jobs': -1
}

# =========================================================================
# LogisticRegressionModel Class
# =========================================================================

@register_model(task='classification')
class LogisticRegressionModel(BaseModel):
    """A concrete implementation of Logistic Regression for Classification tasks.
    
    Attributes:
        model_instance (LogisticRegression): The underlying Scikit-learn estimator.
        param_grid (Dict[str, List[Any]]): Hyperparameter grid for tuning.
        preprocessor (ColumnTransformer): The feature engineering pipeline.
        best_estimator (Pipeline): The fitted pipeline after tuning.
        label_encoder (LabelEncoder): Encoder for target variable.
    """
    def __init__(self, config: Dict[str, Any] = CONFIG) -> None:
        """Initializes the Logistic Regression model.

        Args:
            config (Dict[str, Any], optional): Dictionary containing hyperparameters. 
                Defaults to the global CONFIG dictionary.
        """
        super().__init__(
            name="Logistic Regression",
            config=config
        )
        
        # Initialize the base estimator with class_weight='balanced'
        self.model_instance = LogisticRegression(
            class_weight='balanced', 
            max_iter=1000, 
            random_state=config.get('random_state', 42)
        )
        
        # Params for GridSearch
        self.param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0], # Inverse of regularization strength
            'penalty': ['l1', 'l2', 'elasticnet'], 
            'solver': ['saga'], # 'saga' supports all penalties including elasticnet
            'l1_ratio': [0.5] # Only used if penalty='elasticnet'
        }
        self.preprocessor: Optional[ColumnTransformer] = None
        self.best_estimator: Optional[Pipeline] = None
        self.label_encoder: Optional[LabelEncoder] = None

    def preprocess(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
        """Constructs and applies the feature pipeline optimized for Logistic Regression.
        
        Pipeline Steps:
        1. Numerical: Median imputation -> Standard Scaling -> Feature Selection.
        2. Categorical: Most frequent imputation -> One-Hot Encoding.
        
        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, ColumnTransformer]: Transformed X, y, and fitted preprocessor.
            
        Raises:
            ValueError: If target variable 'y' contains NaNs.
        """
        logger.debug(f"[{self.name}] Entering preprocess()...")
        logger.debug(f"[{self.name}] Input shape: X={X.shape}, y={y.shape}")
        
        if y.isna().any():
            logger.error(f"[{self.name}] Target variable contains {y.isna().sum()} NaN values")
            raise DataValidationError(
                "Target variable contains NaN values. Handle missing targets before training.",
                column='y'
            )

        # 1. Numerical Pipeline
        num_steps: List[Tuple[str, Any]] = [
            ('imputer', get_imputer(strategy='median')),
            ('scaler', get_standard_scaler())
        ]
        
        # Feature Selection Logic
        fs_strategy = self.config.get('feature_selection')
        selection_estimator = LogisticRegression(max_iter=500, random_state=self.config.get('random_state'))
        
        if fs_strategy == 'rfe':
            num_steps.append(('rfe', get_rfe_selector(
                estimator=selection_estimator, 
                n_features_to_select=self.config.get('n_features_to_select', 10)
            )))
            logger.debug(f"[{self.name}] Using RFE feature selection")
        elif fs_strategy == 'model_based':
            num_steps.append(('select_from_model', get_select_from_model(
                estimator=selection_estimator,
                threshold='median'
            )))
            logger.debug(f"[{self.name}] Using model-based feature selection")

        numerical_pipeline = Pipeline(steps=num_steps)

        # 2. Categorical Pipeline
        cat_pipeline = Pipeline(steps=[
            ('imputer', get_imputer(strategy='most_frequent')), 
            ('onehot', get_one_hot_encoder(handle_unknown='ignore', sparse_output=False))
        ])

        # 3. ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, X.select_dtypes(include=np.number).columns.tolist()),
                ('cat', cat_pipeline, X.select_dtypes(include=['object', 'category']).columns.tolist())
            ],
            remainder='passthrough',
            n_jobs=self.config.get('n_jobs', -1)
        )

        self.preprocessor = preprocessor

        # 4. Fit and Transform X
        X_transformed = preprocessor.fit_transform(X, y)
        X_transformed = np.asarray(X_transformed)
        
        # 5. Encode Target y
        if not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y_transformed = le.fit_transform(y)
            self.label_encoder = le
            logger.debug(f"[{self.name}] Target variable encoded using LabelEncoder.")
        else:
            y_transformed = y.values
            self.label_encoder = None
            logger.debug(f"[{self.name}] Target variable is already numeric.")

        return X_transformed, y_transformed, preprocessor

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> None:
        """Trains the Logistic Regression model using a Unified Pipeline.
        
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (np.ndarray): Training targets.
        """
        logger.info(f"[{self.name}] Starting training...")
        logger.debug(f"[{self.name}] Training data shape: X={X_train.shape}, y={y_train.shape}")
        
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
        cv_strategy = StratifiedKFold(
            n_splits=self.config.get('cv_folds', 5), 
            shuffle=True, 
            random_state=self.config.get('random_state', 42)
        )

        # 4. GridSearch
        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=pipeline_params,
            scoring='f1_weighted',
            cv=cv_strategy,
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
            Dict[str, float]: Accuracy, F1, ROC AUC, Log Loss, Precision, Recall, Latency.
        """
        if self.best_estimator is None:
             raise RuntimeError("Model must be fitted before calculating metrics.")

        y_pred = self.best_estimator.predict(X_test)
        y_proba = self.best_estimator.predict_proba(X_test)
        
        metrics = {}
        metrics['Accuracy'] = calculate_accuracy(y_test, y_pred)
        metrics['F1 Score'] = calculate_f1_score(y_test, y_pred, average='weighted')
        metrics['ROC AUC'] = calculate_roc_auc_score(y_test, y_proba)
        metrics['Log Loss'] = calculate_log_loss(y_test, y_proba)
        
        prec, rec = calculate_precision_recall_score(y_test, y_pred, average='weighted')
        metrics['Precision'] = prec
        metrics['Recall'] = rec
        
        metrics['Prediction Latency (s)'] = measure_prediction_latency(self.best_estimator, X_test)

        return metrics

    def get_diagnostic_data(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """Retrieves diagnostic data for visualization.
        
        Args:
            X_test (pd.DataFrame): Test features.
            y_test (np.ndarray): Test targets.

        Returns:
            Dict[str, Any]: Predictions, probabilities, coefficients, etc.
        """
        if self.best_estimator is None:
             raise RuntimeError("Model must be fitted before diagnostics.")
             
        y_pred = self.best_estimator.predict(X_test)
        y_proba = self.best_estimator.predict_proba(X_test)
        
        final_model = self.best_estimator.named_steps['model']
        coefs = final_model.coef_
        if coefs.ndim > 1:
            coefs = coefs[0]

        return {
            'y_pred': y_pred,
            'y_proba': y_proba,
            'y_true': y_test,
            'coefficients': coefs,
            'model_name': self.name,
            'is_odds_ratio': True
        }

    def get_tailored_diagnostics(self) -> Dict[str, Any]:
        """Logistic-Regression-specific diagnostics.

        Returns:
            Dict[str, Any]:
                * ``coefficients`` — The model's decision boundary weights per
                  feature (flattened to 1-D for binary / OvR scenarios).
                  Useful for interpreting feature influence direction.
        """
        if self.best_estimator is None:
            return {}

        final_model = self.best_estimator.named_steps['model']
        if not hasattr(final_model, 'coef_'):
            return {}

        coefs = final_model.coef_
        if coefs.ndim > 1:
            coefs = coefs[0]
        return {'coefficients': coefs.tolist()}

    def get_parameter_descriptions(self) -> Dict[str, Dict[str, str]]:
        """
        Extracts and describes the final tuned hyperparameters of the Logistic Regression.
        
        This method retrieves the 'best_params_' from the GridSearchCV results,
        providing transparency into the model's regularization and penalty strategy
        within the dashboard UI.
        
        Returns:
            Dict[str, Dict[str, str]]: A dictionary mapping parameter names to their 
            values and technical descriptions.
        """
        if self.model and hasattr(self.model, 'best_params_'):
            best_params = self.model.best_params_
            return {
                'C': {
                    'value': f"{best_params.get('model__C', 1.0):.4f}", 
                    'desc': 'Inverse of regularization strength. Smaller values specify stronger regularization, penalizing complexity.'
                },
                'penalty': {
                    'value': str(best_params.get('model__penalty')), 
                    'desc': 'The regularization method used to prevent overfitting (e.g., L1 for sparsity, L2 for shrinkage).'
                }
            }
        return {}
