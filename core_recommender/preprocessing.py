"""
data_preprocessing.py: Modular and Configurable Scikit-learn Transformers

This module provides distinct, pure functions for configuring various scikit-learn
preprocessing and feature engineering transformers. These functions are designed
to be used procedurally, enabling end-users to easily assemble custom
ColumnTransformers and Pipelines.
"""

import numpy as np
from typing import Optional, Union, Callable, Literal, Any
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,

    FunctionTransformer,
    PowerTransformer
)
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_regression,
    f_classif,
    SelectFromModel,
    RFE,
    chi2,
    mutual_info_classif
)
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


# =============================================================================
# 1. Imputation
# =============================================================================

def get_imputer(strategy: Literal['mean', 'median', 'most_frequent', 'constant'] = 'median',
                fill_value: Optional[Any] = None) -> SimpleImputer:
    """
    Returns a configured SimpleImputer object.

    Args:
        strategy: The imputation strategy. Defaults to 'median'.
        fill_value: Value to use when strategy='constant'. Defaults to None.
    """
    return SimpleImputer(strategy=strategy, fill_value=fill_value)


# =============================================================================
# 2. Encoding
# =============================================================================


def get_one_hot_encoder(handle_unknown: Literal['error', 'ignore'] = 'ignore',
                        sparse_output: bool = False) -> OneHotEncoder:
    """
    Returns a configured OneHotEncoder object.

    Args:
        handle_unknown: Whether to 'ignore' or 'error' on unknown categories.
                        Defaults to 'ignore'.
        sparse_output: Whether to return a sparse matrix. Defaults to False (dense output).
    """
    return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=sparse_output)

def get_ordinal_encoder(handle_unknown: Literal['error', 'use_encoded_value'] = 'use_encoded_value',
                        unknown_value: int = -1) -> OrdinalEncoder:
    """
    Returns a configured OrdinalEncoder object.
    
    Args:
        handle_unknown: 'error' or 'use_encoded_value'. Defaults to 'use_encoded_value'.
        unknown_value: Value to use for unknown categories if handle_unknown is 'use_encoded_value'.
    """
    return OrdinalEncoder(handle_unknown=handle_unknown, unknown_value=unknown_value)



# =============================================================================
# 3. Scaling & Normalization
# =============================================================================

def get_standard_scaler() -> StandardScaler:
    """
    Returns a configured StandardScaler (Z-score normalization).
    """
    return StandardScaler()

def get_minmax_scaler() -> MinMaxScaler:
    """
    Returns a configured MinMaxScaler (scaling to [0, 1]).
    """
    return MinMaxScaler()

def get_robust_scaler() -> RobustScaler:
    """
    Returns a configured RobustScaler (scaling using quartiles, robust to outliers).
    """
    return RobustScaler()


# =============================================================================
# 4. Transformation (Distribution/Form)
# =============================================================================

def get_log_transformer() -> FunctionTransformer:
    """
    Returns a FunctionTransformer applying np.log1p (log(1+x)), suitable for
    right-skewed, non-negative data.
    """
    return FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)

def get_box_cox_transformer() -> PowerTransformer:
    """
    Returns a PowerTransformer configured for Box-Cox transformation.
    (Requires input data to be strictly positive).
    """
    return PowerTransformer(method='box-cox')

def get_yeo_johnson_transformer() -> PowerTransformer:
    """
    Returns a PowerTransformer configured for Yeo-Johnson transformation.
    (Supports zero and negative data).
    """
    return PowerTransformer(method='yeo-johnson')


# =============================================================================
# 5. Feature Selection & Dimensionality Reduction
# =============================================================================

def get_variance_threshold(threshold: float = 0.0) -> VarianceThreshold:
    """
    Returns a VarianceThreshold object.

    Args:
        threshold: Features with variance lower than this threshold will be removed.
                   Defaults to 0.0 (removes zero-variance features).
    """
    return VarianceThreshold(threshold=threshold)

def get_select_k_best(k: int = 10, 
                      score_func: Union[str, Callable] = 'f_regression') -> SelectKBest:
    """
    Returns a configured SelectKBest object.

    Args:
        k: Number of top features to select. Defaults to 10.
        score_func: Scoring function (e.g., 'f_regression', 'f_classif', or the callable itself).
                    Defaults to 'f_regression'.
    """
    if isinstance(score_func, str):
        if score_func == 'f_regression':
            func = f_regression
        elif score_func == 'f_classif':
            func = f_classif
        elif score_func == 'chi2':
            func = chi2
        elif score_func == 'mutual_info_classif':
            func = mutual_info_classif
        else:
            raise ValueError(f"Unknown score_func string: {score_func}. Use 'f_regression', 'f_classif', 'chi2', or 'mutual_info_classif'.")
    else:
        func = score_func
        
    return SelectKBest(score_func=func, k=k)

def get_rfe_selector(estimator: BaseEstimator, 
                     n_features_to_select: Union[int, float] = 10,
                     step: Union[int, float] = 1) -> RFE:
    """
    Returns a configured Recursive Feature Elimination (RFE) object.

    Args:
        estimator: The base estimator used to determine feature importance.
        n_features_to_select: The number of features to select, or the fraction
                              of features (float) to select. Defaults to 10.
        step: The number of features to remove at each iteration. Defaults to 1.
    """
    return RFE(estimator=estimator, 
               n_features_to_select=n_features_to_select, 
               step=step)

def get_select_from_model(estimator: BaseEstimator, 
                          threshold: Union[str, float] = 'median') -> SelectFromModel:
    """
    Returns a configured SelectFromModel object.

    Args:
        estimator: The fitted estimator (e.g., LinearSVC, RandomForest) with
                   coef_ or feature_importances_ attributes.
        threshold: The threshold for feature importance (e.g., 'median', 'mean', or a float value).
    """
    # Note: SelectFromModel expects the estimator to be an *instance*, 
    # but the fitting happens when SelectFromModel is used in a pipeline step.
    return SelectFromModel(estimator=estimator, threshold=threshold)

def get_pca_reducer(n_components: Union[int, float, None] = 0.95) -> PCA:
    """
    Returns a configured Principal Component Analysis (PCA) object.

    Args:
        n_components: Number of components to keep. If float between 0.0 and 1.0, 
                      it specifies the variance ratio to be preserved. Defaults to 0.95.
    """
    return PCA(n_components=n_components)
def get_nca_reducer(n_components: Optional[int] = None, 
                    random_state: Optional[int] = None) -> NCA:
    """
    Returns a configured Neighborhood Components Analysis (NCA) object.

    Args:
        n_components: Number of components to keep. If None, all components are kept.
        random_state: Seed for reproducibility. Defaults to None.
    """
    return NCA(n_components=n_components, random_state=random_state)