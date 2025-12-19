import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# --- PROJECT IMPORTS ---
from core_recommender.modeling.baseModel import BaseModel
from core_recommender.preprocessing import (
    get_imputer,
    get_standard_scaler
)
from core_recommender.evaluation import calculate_rmse

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
    """
    A concrete implementation of Principal Component Analysis (PCA) for unsupervised dimensionality reduction.
    Adapts the supervised BaseModel interface by ignoring target labels during fit.
    """
    def __init__(self, config: Dict[str, Any] = CONFIG):
        
        name = "Principal Component Analysis (PCA)"
        super().__init__(name=name, config=config)
        
        # Initialize PCA
        self.model_instance = PCA(
            n_components=config.get('n_components', 0.95),
            whiten=config.get('whiten', False),
            random_state=config.get('random_state', 42)
        )

    def preprocess(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
        """
        Constructs a numeric-only pipeline. Categorical features are typically dropped or must be pre-encoded.
        For PCA, we usually focus on numerical features.
        """
        # 1. Pipeline Construction
        # ------------------------
        
        # Numerical Steps only
        num_steps = []
        num_steps.append(('imputer', get_imputer(strategy='mean'))) # Mean for PCA
        num_steps.append(('scaler', get_standard_scaler())) # Scaling is critical for PCA

        numerical_pipeline = Pipeline(steps=num_steps)

        # 2. Composition
        # --------------
        # We only process numerical columns. Categorical columns are dropped.
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, X.select_dtypes(include=np.number).columns.tolist())
            ],
            remainder='drop',
            n_jobs=-1
        )

        # 3. Fit-Transform
        # ----------------
        X_transformed = preprocessor.fit_transform(X)
        X_transformed = np.asarray(X_transformed)

        # 4. Target
        # ---------
        # PCA is unsupervised, but interface requires y. Pass through or dummy.
        y_transformed = y.values if y is not None else np.zeros(len(X))
        self.label_encoder = None

        return X_transformed, y_transformed, preprocessor

    def fit(self, X_train: np.ndarray, y_train: np.ndarray = None):
        """
        Fits PCA on X_train. y_train is ignored.
        """
        print(f"[{self.name}] Fitting PCA...")
        self.model_instance.fit(X_train)
        self.model = self.model_instance # consistent with export
        
        n_comps = self.model_instance.n_components_
        var_ratio = np.sum(self.model_instance.explained_variance_ratio_)
        print(f"[{self.name}] Fitted with {n_comps} components explaining {var_ratio:.2%} variance.")

    def calculate_metrics(self, X_test: np.ndarray, y_test: np.ndarray = None) -> Dict[str, float]:
        """
        Calculates Reconstruction Error (RMSE) and Total Explained Variance.
        """
        # Transform and Inverse Transform to calculate reconstruction error
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

    def get_diagnostic_data(self, X_test: np.ndarray, y_test: np.ndarray = None) -> Dict[str, Any]:
        """
        Returns Scree Plot data in 'feature_importance' slot or custom.
        """
        return {
            'explained_variance_ratio': self.model_instance.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(self.model_instance.explained_variance_ratio_),
            'singular_values': self.model_instance.singular_values_,
            'model_name': self.name
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Returns the components (loadings) magnitude? 
        Or just explained variance per component.
        """
        # Just returning explained variance ratio per component index
        return dict(enumerate(self.model_instance.explained_variance_ratio_))
