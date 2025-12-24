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
    
    This model predicts the probability of an outcome using the logistic sigmoid function.
    It supports various regularization penalties (L1, L2, ElasticNet) to handle high-dimensional 
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
        
        # Check for NaNs in target y
        if y.isna().any():
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
        elif fs_strategy == 'model_based':
            num_steps.append(('select_from_model', get_select_from_model(
                estimator=selection_estimator,
                threshold='median'
            )))

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

        # 4. Fit and Transform X
        X_transformed = preprocessor.fit_transform(X, y)
        X_transformed = np.asarray(X_transformed)
        
        # 5. Encode Target y (LabelEncoder)
        # Ensure y is suitable for classification (0, 1, ...)
        # We reuse the helper from dataHandling or do it locally.
        # Since we need to return a numpy array for y_transformed:
        le = LabelEncoder()
        y_transformed = le.fit_transform(y)
        
        # Store label encoder for later inverse transform if needed
        self.label_encoder = le

        return X_transformed, y_transformed, preprocessor

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the Logistic Regression model using GridSearchCV (Tier 1) via factory.
        
        Args:
            X_train: Training features array.
            y_train: Training target array.
        """
        from core_recommender.tuning import get_grid_search_tuner

        cv_strategy = StratifiedKFold(
            n_splits=self.config.get('cv_folds', 5), 
            shuffle=True, 
            random_state=self.config.get('random_state', 42)
        )

        self.model = get_grid_search_tuner(
            estimator=self.model_instance,
            param_grid=self.param_grid,
            scoring='f1_weighted',
            cv=cv_strategy,
            n_jobs=self.config.get('n_jobs', -1),
            verbose=1
        )

        print(f"[{self.name}] Starting GridSearchCV (Tier 1)...")
        self.model.fit(X_train, y_train)
        
        self.best_estimator = self.model.best_estimator_
        print(f"[{self.name}] Best parameters: {self.model.best_params_}")

    def calculate_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Calculates standard classification performance metrics.

        Metrics Include:
        - Accuracy
        - F1 Score (weighted)
        - ROC AUC
        - Log Loss
        - Precision (weighted)
        - Recall (weighted)
        - Prediction Latency (seconds)

        Args:
            X_test: Test features array.
            y_test: Test target array.
            
        Returns:
            Dict[str, float]: Dictionary of calculated metrics.
        """
        y_pred = self.best_estimator.predict(X_test)
        y_proba = self.best_estimator.predict_proba(X_test)
        
        metrics = {}
        metrics['Accuracy'] = calculate_accuracy(y_test, y_pred)
        metrics['F1 Score'] = calculate_f1_score(y_test, y_pred, average='weighted')
        metrics['ROC AUC'] = calculate_roc_auc_score(y_test, y_proba)
        metrics['Log Loss'] = calculate_log_loss(y_test, y_proba)
        
        # Precision and Recall (simultaneous)
        prec, rec = calculate_precision_recall_score(y_test, y_pred, average='weighted')
        metrics['Precision'] = prec
        metrics['Recall'] = rec
        
        metrics['Prediction Latency (s)'] = measure_prediction_latency(self.best_estimator, X_test)

        return metrics

    def get_diagnostic_data(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Retrieves diagnostic data for visualization.
        
        Includes:
        - Predictions and Probabilities (for ROC/PR curves)
        - Coefficients (Weights) of independent variables (for Odds Ratio analysis)
        
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
             
        y_pred = self.best_estimator.predict(X_test)
        y_proba = self.best_estimator.predict_proba(X_test)
        
        # Get coefficients
        # LogisticRegression coef_ is shape (1, n_features) for binary, (n_classes, n_features) for multi.
        # We assume binary or take the first class components for simplicity in basic bar chart 
        # or pass raw and let visualization handle.
        coefs = self.best_estimator.coef_
        if coefs.ndim > 1:
            coefs = coefs[0] # Take first class vs rest for binary, or just first row

        return {
            'y_pred': y_pred,
            'y_proba': y_proba,
            'y_test': y_test,
            'coefficients': coefs,
            'model_name': self.name,
            'is_odds_ratio': True # Hint for visualization to exponentiate
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Retrieves feature importance based on the magnitude of the model coefficients.
        
        Returns:
            Dict[str, float]: Dictionary mapping feature indices to coefficient values.
        """
        return {} 
