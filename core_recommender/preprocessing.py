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
    chi2,
    mutual_info_classif,
    RFE,
    SelectFromModel
)
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer


# =============================================================================
# 1. Imputation
# =============================================================================

def get_imputer(strategy: Literal['mean', 'median', 'most_frequent', 'constant'] = 'median',
                fill_value: Optional[Any] = None) -> SimpleImputer:
    """Returns a configured SimpleImputer object for handling missing values.

    Args:
        strategy (str): The imputation strategy. 
                        Options: 'median', 'mean', 'most_frequent', 'constant'.
                        Defaults to 'median'.
        fill_value (Any, optional): Constant value to use when strategy='constant'. Defaults to None.
        
    Returns:
        SimpleImputer: A scikit-learn SimpleImputer instance ready for fitting.
    """
    return SimpleImputer(strategy=strategy, fill_value=fill_value)


# =============================================================================
# 2. Encoding
# =============================================================================


def get_one_hot_encoder(handle_unknown: Literal['error', 'ignore'] = 'ignore',
                        sparse_output: bool = False) -> OneHotEncoder:
    """Returns a configured OneHotEncoder object for nominal categorical variables.

    Args:
        handle_unknown (str): Strategy to handle new categories found in test data.
                              Options: 'ignore', 'error'. Defaults to 'ignore'.
        sparse_output (bool): Whether to return a sparse matrix. Defaults to False.

    Returns:
        OneHotEncoder: A scikit-learn OneHotEncoder instance.
    """
    return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=sparse_output)

def get_ordinal_encoder(handle_unknown: Literal['error', 'use_encoded_value'] = 'use_encoded_value',
                        unknown_value: int = -1) -> OrdinalEncoder:
    """Returns a configured OrdinalEncoder object for ordinal categorical variables.
    
    Args:
        handle_unknown (str): Strategy to handle new categories.
                              Options: 'use_encoded_value', 'error'. Defaults to 'use_encoded_value'.
        unknown_value (int): The integer value to use for unknown categories. Defaults to -1.

    Returns:
        OrdinalEncoder: A scikit-learn OrdinalEncoder instance.
    """
    return OrdinalEncoder(handle_unknown=handle_unknown, unknown_value=unknown_value)



# =============================================================================
# 3. Scaling & Normalization
# =============================================================================

def get_standard_scaler() -> StandardScaler:
    """Returns a configured StandardScaler for Z-score normalization.
    
    Centers data to Mean = 0 and Variance = 1.
    
    Returns:
        StandardScaler: A scikit-learn StandardScaler instance.
    """
    return StandardScaler()

def get_minmax_scaler() -> MinMaxScaler:
    """Returns a configured MinMaxScaler.
    
    Scales features to a given range, typically [0, 1].
    
    Returns:
        MinMaxScaler: A scikit-learn MinMaxScaler instance.
    """
    return MinMaxScaler()

def get_robust_scaler() -> RobustScaler:
    """Returns a configured RobustScaler.
    
    Scales features using statistics that are robust to outliers (quartiles).
    
    Returns:
        RobustScaler: A scikit-learn RobustScaler instance.
    """
    return RobustScaler()


# =============================================================================
# 4. Transformation (Distribution/Form)
# =============================================================================

def get_log_transformer() -> FunctionTransformer:
    """Returns a FunctionTransformer applying np.log1p (log(1+x)).
    
    Used to reduce skewness and compress long tails in distribution.

    Returns:
        FunctionTransformer: A scikit-learn FunctionTransformer instance configured for log transformation.
    """
    return FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)

def get_box_cox_transformer() -> PowerTransformer:
    """Returns a PowerTransformer configured for Box-Cox transformation.
    
    Requires input data to be strictly positive.

    Returns:
        PowerTransformer: A scikit-learn PowerTransformer instance (method='box-cox').
    """
    return PowerTransformer(method='box-cox')

def get_yeo_johnson_transformer() -> PowerTransformer:
    """Returns a PowerTransformer configured for Yeo-Johnson transformation.
    
    Supports zero and negative data.

    Returns:
        PowerTransformer: A scikit-learn PowerTransformer instance (method='yeo-johnson').
    """
    return PowerTransformer(method='yeo-johnson')


# =============================================================================
# 5. Feature Selection & Dimensionality Reduction
# =============================================================================

def get_variance_threshold(threshold: float = 0.0) -> VarianceThreshold:
    """Returns a VarianceThreshold object to remove low-variance features.

    Args:
        threshold (float): Features with variance lower than this will be removed. Defaults to 0.0.

    Returns:
        VarianceThreshold: A scikit-learn VarianceThreshold instance.
    """
    return VarianceThreshold(threshold=threshold)

def get_select_k_best(k: int = 10, 
                      score_func: Union[str, Callable] = 'f_regression') -> SelectKBest:
    """Returns a configured SelectKBest object to select the top 'k' features.

    Args:
        k (int): Number of top features to select. Defaults to 10.
        score_func (Union[str, Callable]): Measures dependency between features and target.
                    Options: 'f_regression', 'f_classif', 'chi2', 'mutual_info_classif'.
                    Defaults to 'f_regression'.

    Returns:
        SelectKBest: A scikit-learn SelectKBest instance.

    Raises:
        ValueError: If an unknown `score_func` string is provided.
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
            raise ValueError(f"Unknown score_func string: {score_func}.")
    else:
        func = score_func
        
    return SelectKBest(score_func=func, k=k)

def get_rfe_selector(estimator: BaseEstimator, 
                     n_features_to_select: Union[int, float] = 10,
                     step: Union[int, float] = 1) -> RFE:
    """Returns a configured Recursive Feature Elimination (RFE) object.

    Args:
        estimator (BaseEstimator): The base estimator (model) used to assign weights.
        n_features_to_select (Union[int, float]): Target number or percentage of features. Defaults to 10.
        step (Union[int, float]): Number of features to remove at each iteration. Defaults to 1.

    Returns:
        RFE: A scikit-learn RFE instance.
    """
    return RFE(estimator=estimator, 
               n_features_to_select=n_features_to_select, 
               step=step)

def get_select_from_model(estimator: BaseEstimator, 
                          threshold: Union[str, float] = 'median') -> SelectFromModel:
    """Returns a configured SelectFromModel object.

    Args:
        estimator (BaseEstimator): The base estimator used to compute feature importance.
        threshold (Union[str, float]): The threshold value to use for feature selection.
                                       Defaults to 'median'.
    
    Returns:
        SelectFromModel: A scikit-learn SelectFromModel instance.
    """
    return SelectFromModel(estimator=estimator, threshold=threshold)

def get_pca_reducer(n_components: Union[int, float, None] = 0.95) -> PCA:
    """Returns a configured Principal Component Analysis (PCA) object.

    Args:
        n_components (Union[int, float, None]): Desired dimensionality.
                      If float < 1.0, represents variance to explain.
                      Defaults to 0.95.

    Returns:
        PCA: A scikit-learn PCA instance.
    """
    return PCA(n_components=n_components)

def get_nca_reducer(n_components: Optional[int] = None, 
                    random_state: Optional[int] = None) -> NCA:
    """Returns a configured Neighborhood Components Analysis (NCA) object.

    Args:
        n_components (Optional[int]): Number of components to keep. Defaults to None.
        random_state (Optional[int]): Seed for reproducibility. Defaults to None.

    Returns:
        NCA: A scikit-learn NeighborhoodComponentsAnalysis instance.
    """
    return NCA(n_components=n_components, random_state=random_state)