import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

# --- SKLEARN IMPORTS ---
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder

# --- PROJECT IMPORTS ---
from core_recommender.modeling.baseModel import BaseModel
from core_recommender.dataHandling import apply_label_encoder_target
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

# --- DEFAULT CONFIGURATION ---
CONFIG = {
    'scaler': 'standard',
    'feature_selection': 'rfe',  # 'rfe', 'model_based', 'none'
    'n_features_to_select': 10,  # For RFE
    'cv_folds': 5,
    'random_state': 42,
    'n_jobs': -1
}

# =========================================================================
# LogisticRegressionModel Class
# =========================================================================

class LogisticRegressionModel(BaseModel):
    """
    A concrete implementation of Logistic Regression for Classification tasks.
    
    Rationale:
    ----------
    - **Probabilistic Output**: Directly models the probability of class membership using the sigmoid function.
    - **Robustness**: Regularization (L1/L2) handles high-dimensional data effectively.
    - **Class Imbalance**: The `class_weight='balanced'` parameter automatically adjusts weights inversely proportional to class frequencies, crucial for rare event detection.

    This model supports various regularization penalties (L1, L2, ElasticNet) to handle high-dimensional 
    data and prevent overfitting.
    """
    def __init__(self, config: Dict[str, Any] = CONFIG):
        """
        Initializes the Logistic Regression model with configurable hyperparameters.

        Args:
            config: Dictionary containing hyperparameters (e.g., 'C', 'penalty', 'solver').
                    Defaults to the global CONFIG dictionary.
        """
        
        super().__init__(
            name="Logistic Regression",
            config=config
        )
        
        # Initialize the base estimator with class_weight='balanced'
        # to automatically handle imbalanced datasets (common in recommendation).
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

    def preprocess(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
        """
        Constructs and applies the feature pipeline optimized for Logistic Regression.
        
        Rationale:
        ----------
        - **Scaling**: Mandatory. Gradient descent converges much faster on scaled data, and regularization assumes uniform scale.
        - **RFE (Recursive Feature Elimination)**: Iteratively removes weakest features to build a smaller, more robust model.

        Pipeline Steps:
        1. Numerical: Median imputation. Standard Scaling. Feature Selection (RFE or Model-Based).
        2. Categorical: Most frequent imputation. One-Hot encoding.
        
        Args:
            X: Input features DataFrame.
            y: Target Series.
            
        Returns:
            Tuple containing:
            - Transformed feature array (np.ndarray)
            - Transformed target array (np.ndarray)
            - The fitted ColumnTransformer object
            
        Raises:
            ValueError: If target variable 'y' contains NaNs.
        """
        
        logger.debug(f"[{self.name}] Entering preprocess()...")
        logger.debug(f"[{self.name}] Input shape: X={X.shape}, y={y.shape}")
        
        # Check for NaNs in target y
        if y.isna().any():
            logger.error(f"[{self.name}] Target variable contains {y.isna().sum()} NaN values")
            raise ValueError("Target variable 'y' contains missing values (NaNs).")

        # 1. Numerical Pipeline
        num_steps = [
            ('imputer', get_imputer(strategy='median')),
            ('scaler', get_standard_scaler())
        ]
        
        # Feature Selection Logic
        fs_strategy = self.config.get('feature_selection')
        
        # Note: RFE and SelectFromModel need an estimator passed to them.
        # We generally use a simple lightweight estimator for selection (e.g., simple LogReg or Tree)
        selection_estimator = LogisticRegression(max_iter=500, random_state=self.config.get('random_state'))
        
        if fs_strategy == 'rfe':
            num_steps.append(('rfe', get_rfe_selector(
                estimator=selection_estimator, 
                n_features_to_select=self.config.get('n_features_to_select', 10)
            )))
            logger.debug(f"[{self.name}] Using RFE feature selection (n_features={self.config.get('n_features_to_select', 10)})")
        elif fs_strategy == 'model_based':
            num_steps.append(('select_from_model', get_select_from_model(
                estimator=selection_estimator,
                threshold='median'
            )))
            logger.debug(f"[{self.name}] Using model-based feature selection (threshold=median)")

        numerical_pipeline = Pipeline(steps=num_steps)

        # 2. Categorical Pipeline
        cat_pipeline = Pipeline(steps=[
            ('imputer', get_imputer(strategy='most_frequent')), # Mode strategy
            # handle_unknown='ignore'
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
        
        # 5. Encode Target y (Skip if already numeric/pre-encoded by Executor)
        if not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y_transformed = le.fit_transform(y)
            self.label_encoder = le
            logger.debug(f"[{self.name}] Target variable encoded using LabelEncoder.")
        else:
            y_transformed = y.values
            self.label_encoder = None
            logger.debug(f"[{self.name}] Target variable is already numeric, skipping encoding.")

        logger.debug(f"[{self.name}] Preprocessing complete. Transformed X shape: {X_transformed.shape}")
        return X_transformed, y_transformed, preprocessor

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """
        Trains the Logistic Regression model using a Unified Pipeline to prevent data leakage.
        """
        logger.info(f"[{self.name}] Starting training...")
        logger.debug(f"[{self.name}] Training data shape: X={X_train.shape}, y={y_train.shape}")
        
        from sklearn.model_selection import StratifiedKFold, GridSearchCV
        from sklearn.base import clone

        # 1. Pipeline Construction
        preprocessor_template = clone(self.preprocessor)
        pipe = Pipeline(steps=[
            ('pre', preprocessor_template),
            ('model', self.model_instance)
        ])
        logger.debug(f"[{self.name}] Pipeline constructed with preprocessor and model instance.")

        # 2. Adjust Param Grid
        pipeline_params = {f'model__{k}': v for k, v in self.param_grid.items()}
        logger.debug(f"[{self.name}] Parameter grid for GridSearchCV: {pipeline_params}")

        # 3. CV Strategy
        cv_strategy = StratifiedKFold(
            n_splits=self.config.get('cv_folds', 5), 
            shuffle=True, 
            random_state=self.config.get('random_state', 42)
        )
        logger.debug(f"[{self.name}] Using StratifiedKFold with {self.config.get('cv_folds', 5)} folds.")

        # 4. GridSearch on Pipeline
        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=pipeline_params,
            scoring='f1_weighted',
            cv=cv_strategy,
            n_jobs=self.config.get('n_jobs', -1),
            verbose=0
        )
        logger.debug(f"[{self.name}] Starting GridSearchCV...")

        grid_search.fit(X_train, y_train)
        
        self.best_estimator = grid_search.best_estimator_
        self.model = grid_search
        
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_
        logger.info(f"✅ [{self.name}] Training complete")
        logger.info(f"[{self.name}] Best CV Score: {best_score:.4f} (F1-weighted)")
        logger.debug(f"[{self.name}] Best params: {best_params}")

    def calculate_metrics(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """
        Calculates performance metrics using the full Pipeline.
        """
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
        
        # Latency
        metrics['Prediction Latency (s)'] = measure_prediction_latency(self.best_estimator, X_test)

        return metrics

    def get_diagnostic_data(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Retrieves diagnostic data for visualization using the Pipeline.
        """
        if not hasattr(self, 'best_estimator'):
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
            'y_test': y_test,
            'coefficients': coefs,
            'model_name': self.name,
            'is_odds_ratio': True
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Retrieves feature importance based on model coefficients.
        """
        final_model = self.best_estimator.named_steps['model']
        if hasattr(final_model, 'coef_'):
            coefs = final_model.coef_
            if coefs.ndim > 1: coefs = coefs[0]
            return {'importances': coefs.tolist()}
        return {}
    def get_parameter_descriptions(self) -> Dict[str, Dict[str, str]]:
        """
        Returns descriptions of the most important tuned parameters.
        """
        # Access best params from GridSearch results
        best_params = self.model.best_params_
        descriptions = {
            'C': {
                'value': f"{best_params.get('model__C'):.4f}",
                'desc': 'Inverse regularization strength. Smaller values specify stronger regularization, helping to prevent overfitting.'
            },
            'penalty': {
                'value': str(best_params.get('model__penalty')),
                'desc': 'The type of regularization applied (L1, L2, or ElasticNet) to penalize complex models.'
            }
        }
        return descriptions
