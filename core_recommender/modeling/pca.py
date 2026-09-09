"""
pca.py — Experimental PCA wrapper (NOT registered in the model registry)

STATUS: Experimental / Unused in production pipeline.

This module implements a PCA-based dimensionality reduction wrapper that follows
the BaseModel interface. It is NOT decorated with @register_model and is therefore
NOT included in ModelExecutor's training runs.

If you wish to enable it, add the @register_model(task='both') decorator above
the class definition AND import this module in execution.py alongside the other
model imports.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# --- PROJECT IMPORTS ---
from core_recommender.modeling.base_model import BaseModel
from core_recommender.preprocessing import (
    get_imputer,
    get_standard_scaler
)
# Note: calculate_rmse import is generally not needed for strict PCA unless reconstructing. 
# We'll keep it if we implement reconstruction error checks explicitly.

# Import centralized logger
from core_recommender.logger import get_logger
logger = get_logger(__name__)

# --- DEFAULT CONFIGURATION ---
CONFIG = {
    'n_components': 0.95, # float (variance) or int (count)
    'whiten': False,
    'random_state': 42
}

# =========================================================================
# PCAModel Class (Unsupervised)
# =========================================================================

class PCAModel(BaseModel):
    """A concrete implementation of Principal Component Analysis (PCA) for unsupervised dimensionality reduction.
    
    Attributes:
        model_instance (PCA): The underlying Scikit-learn PCA object.
        preprocessor (ColumnTransformer): The feature engineering pipeline (Scaling is mandatory).
    """
    def __init__(self, config: Dict[str, Any] = CONFIG) -> None:
        """Initializes the PCA model wrapper.

        Args:
            config (Dict[str, Any], optional): Hyperparameters. Defaults to GLOBAL config.
        """
        name = "Principal Component Analysis (PCA)"
        super().__init__(name=name, config=config)
        
        self.model_instance = PCA(
            n_components=config.get('n_components', 0.95),
            whiten=config.get('whiten', False),
            random_state=config.get('random_state', 42)
        )
        self.preprocessor: Optional[ColumnTransformer] = None
        self.label_encoder = None

    def preprocess(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
        """Constructs and applies a feature pipeline for PCA.
        
        Pipeline Steps:
        1. Numerical: Mean Imputation -> Standard Scaling (Critical for PCA).
        2. Categorical: Dropped (PCA is for continuous variables).
        
        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series, Optional): Target variable (Ignored).
            
        Returns:
            Tuple[np.ndarray, np.ndarray, ColumnTransformer]: X_transformed, y_dummy, preprocessor.
        """
        logger.debug(f"[{self.name}] Entering preprocess()...")
        
        # 1. Pipeline Construction
        num_steps = [
            ('imputer', get_imputer(strategy='mean')),
            ('scaler', get_standard_scaler())
        ]
        numerical_pipeline = Pipeline(steps=num_steps)

        # 2. Composition (Drop non-numeric)
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, X.select_dtypes(include=np.number).columns.tolist())
            ],
            remainder='drop',
            n_jobs=-1
        )
        self.preprocessor = preprocessor

        # 3. Fit-Transform
        X_transformed = preprocessor.fit_transform(X)
        X_transformed = np.asarray(X_transformed)

        # 4. Target (Dummy)
        y_transformed = y.values if y is not None else np.zeros(len(X))
        
        return X_transformed, y_transformed, preprocessor

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> None:
        """Fits the PCA transformer on the training data.
        
        Args:
            X_train (np.ndarray): Training features array (standardized).
            y_train (np.ndarray, optional): Training target array (Ignored).
        """
        logger.debug(f"[{self.name}] Fitting PCA...")
        self.model_instance.fit(X_train)
        self.model = self.model_instance 
        
        n_comps = self.model_instance.n_components_
        var_ratio = np.sum(self.model_instance.explained_variance_ratio_)
        logger.info(f"[{self.name}] Fitted with {n_comps} components explaining {var_ratio:.2%} variance")

    def calculate_metrics(self, X_test: np.ndarray, y_test: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculates PCA-specific metrics by reconstructing the test data.
        
        Args:
            X_test (np.ndarray): Test features array (standardized).
            y_test (np.ndarray, optional): Ignored.

        Returns:
            Dict[str, float]: Reconstruction RMSE, Explained Variance, n_components.
        """
        if self.model_instance is None:
             raise RuntimeError("Model must be fitted before calculating metrics.")

        X_pca = self.model_instance.transform(X_test)
        X_inverse = self.model_instance.inverse_transform(X_pca)
        
        mse = np.mean(np.square(X_test - X_inverse))
        rmse = np.sqrt(mse)
        
        metrics = {
            'Reconstruction RMSE': float(rmse),
            'Explained Variance': float(np.sum(self.model_instance.explained_variance_ratio_)),
            'n_components': int(self.model_instance.n_components_)
        }
        return metrics

    def get_diagnostic_data(self, X_test: np.ndarray, y_test: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Retrieves diagnostic data for visualization (e.g., Scree Plot)."""
        if self.model_instance is None:
             raise RuntimeError("Model must be fitted before diagnostics.")
             
        return {
            'explained_variance_ratio': self.model_instance.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(self.model_instance.explained_variance_ratio_),
            'singular_values': self.model_instance.singular_values_,
            'model_name': self.name
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """Returns the Explained Variance Ratio per component index."""
        if self.model_instance is None:
             return {}
        return dict(enumerate(self.model_instance.explained_variance_ratio_))
