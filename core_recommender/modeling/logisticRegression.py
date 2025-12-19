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
    """
    def __init__(self, config: Dict[str, Any] = CONFIG):
        
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
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2', 'elasticnet'], 
            'solver': ['saga'], 
            'l1_ratio': [0.5] 
        }

    def preprocess(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
        """
        Constructs and applies the feature pipeline optimized for Logistic Regression.
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
        Trains using Stratified K-Fold CV and GridSearchCV.
        """
        cv_strategy = StratifiedKFold(
            n_splits=self.config.get('cv_folds', 5), 
            shuffle=True, 
            random_state=self.config.get('random_state', 42)
        )

        self.model = GridSearchCV(
            estimator=self.model_instance,
            param_grid=self.param_grid,
            scoring='f1_weighted', # Optimize for F1 Score (good for imbalance)
            cv=cv_strategy,
            n_jobs=self.config.get('n_jobs', -1),
            verbose=1
        )

        print(f"Starting GridSearchCV for {self.name}...")
        self.model.fit(X_train, y_train)
        
        self.best_estimator = self.model.best_estimator_
        print(f"Best parameters found: {self.model.best_params_}")

    def calculate_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Calculates F1, ROC-AUC, Accuracy, Log Loss, Precision, Recall.
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
        Returns data for visualizations including probabilities for ROC/PR curves
        and coefficients for Odds Ratios.
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
        Returns coefficients as importance.
        """
        return {} 
