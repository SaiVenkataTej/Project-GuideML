import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Import centralized logger
from core_recommender.logger import get_logger

logger = get_logger(__name__)
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
    
    Overview:
    ---------
    PCA is a statistical procedure that uses an orthogonal transformation to convert a set of observations 
    of possibly correlated variables into a set of values of linearly uncorrelated variables called 
    principal components.
    
    This implementation adapts PCA into the project's standard `BaseModel` interface, enabling:
    1. **Integration**: Seamless usage within the main execution engine alongside supervised models.
    2. **Consistency**: Unified preprocessing (Imputation + Scaling) which is critical for valid PCA.
    3. **Evaluation**: Reconstruction error metrics to quantify information loss.

    Configuration (`CONFIG`):
    -------------------------
    - `n_components`: float|int - 
        - If float < 1.0 (e.g., 0.95), it represents the threshold of explained variance to retain.
        - If int >= 1, it represents the exact number of components to keep.
    - `whiten`: bool - When True, multiply the components mainly by the square root of n_samples and 
      divide by the singular values to ensure uncorrelated outputs with unit component-wise variances.
      Useful for subsequent steps like K-Means.
    """
    def __init__(self, config: Dict[str, Any] = CONFIG):
        """
        Initializes the PCA model wrapper with the specified configuration.

        Args:
            config (Dict[str, Any]): 
                Configuration dictionary containing:
                - 'n_components': Target variance or component count.
                - 'whiten': Boolean for whitening option.
                - 'random_state': Seed for reproducibility.
                Defaults to the global CONFIG dictionary.
        """
        
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
        Constructs and applies a feature pipeline tailored for PCA.
        
        Rationale:
        ----------
        - **Scaling**: PCA seeks to maximize variance. Without Standardization (mean=0, std=1), 
          variables with larger scales (e.g., Salary) would dominate variables with smaller scales 
          (e.g., Age), distorting the principal components.
        - **Imputation**: Mean imputation is used for numerical fields.
        - **Categorical Data**: PCA is mathematically defined for continuous numerical data. 
          Categorical features are **dropped** in this pipeline to focus on numerical dimensionality reduction. 
          Use Multi-Factor Analysis (MFA) or related techniques for mixed data types if needed.
        
        Args:
            X (pd.DataFrame): Input dataframe containing numerical columns for PCA.
            y (pd.Series, Optional): Target variable. Ignored by PCA but required by the interface.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
                - **X_transformed**: Standardized and imputed numerical data (ready for PCA fit).
                - **y_transformed**: Passthrough target data (or dummy array if None).
                - **preprocessor**: The fitted `ColumnTransformer` (only processing 'num' columns).
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
        Fits the PCA transformer on the training data.
        
        Args:
            X_train (np.ndarray): Training features array (standardized).
            y_train (np.ndarray, optional): Training target array (Ignored).
        """
        logger.debug(f"[{self.name}] Fitting PCA...")
        self.model_instance.fit(X_train)
        self.model = self.model_instance # Assign to self.model for export compatibility
        
        n_comps = self.model_instance.n_components_
        var_ratio = np.sum(self.model_instance.explained_variance_ratio_)
        logger.info(f"[{self.name}] Fitted with {n_comps} components explaining {var_ratio:.2%} variance")

    def calculate_metrics(self, X_test: np.ndarray, y_test: np.ndarray = None) -> Dict[str, float]:
        """
        Calculates PCA-specific metrics by reconstructing the test data.
        
        Metrics Explained:
        ------------------
        - **Reconstruction RMSE**: Routinely used to measure how much information is lost. 
          Calculated as the RMSE between the original data and the data reconstructed from the 
          reduced components. Lower is better.
        - **Explained Variance**: The cumulative variance explained by the selected components. 
          Higher (closer to 1.0) is generally better, but must be balanced with reduction.
        - **n_components**: The actual number of components used.

        Args:
            X_test (np.ndarray): Test features array (standardized).
            y_test (np.ndarray, optional): Test target array (Ignored).
            
        Returns:
            Dict[str, float]: Dictionary of calculated metrics.
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
        Retrieves diagnostic data for visualization (e.g., Scree Plot).
        
        Args:
            X_test (np.ndarray): Test features array.
            y_test (np.ndarray, optional): Test target array.
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'explained_variance_ratio': Array of variance explained by each component.
                - 'cumulative_variance': Array of cumulative variance explained.
                - 'singular_values': Singular values corresponding to each component.
                - 'model_name': Name of the model.
            
        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        return {
            'explained_variance_ratio': self.model_instance.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(self.model_instance.explained_variance_ratio_),
            'singular_values': self.model_instance.singular_values_,
            'model_name': self.name
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Returns the Explained Variance Ratio per component index as a proxy for "importance".
        
        Note:
        -----
        In PCA, "Feature Importance" typically refers to the loading of each original feature on 
        the principal components, which is a matrix (components x features). 
        Here, we return the importance *of the components themselves* (explained variance).
        
        Returns:
            Dict[str, float]: Dictionary mapping component index (as str) to variance explained.
        """
        return dict(enumerate(self.model_instance.explained_variance_ratio_))
