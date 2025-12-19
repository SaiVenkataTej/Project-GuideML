import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from typing import Literal, Dict, Any, Union, List
from collections.abc import Collection

# =========================================================================
# 🧺 data_handling_functions.py: Modular Data Preparation (F1, F3)
# 
# Contains reusable functions strictly for Encoding.
# Splitting, Imputation, and Scaling are handled externally.
# =========================================================================

# --- 1. Feature Encoding Functions (Categorical Features) ---

def apply_one_hot_encoder(
    data: pd.DataFrame,
    drop_first: bool = False,
    handle_unknown_param: Literal['error', 'ignore'] = 'ignore'
) -> pd.DataFrame:
    """
    Applies One-Hot Encoding to nominal categorical features and returns the transformed DataFrame.
    
    Args:
        data: DataFrame containing categorical features.
        drop_first: If True, drops the first category (to avoid multicollinearity).
        handle_unknown_param: How to handle new categories during transformation ('ignore' is safer).
        
    Returns:
        The transformed DataFrame (pd.DataFrame).
    """
    encoder = OneHotEncoder(
        drop='first' if drop_first else None,
        handle_unknown=handle_unknown_param,
        sparse_output=False
    )
    
    transformed_array = encoder.fit_transform(data)
    
    # Get the feature names for the new columns
    feature_names = encoder.get_feature_names_out(data.columns)
    
    return pd.DataFrame(
        transformed_array,
        columns=feature_names,
        index=data.index
    )

def apply_ordinal_encoder(
    data: pd.DataFrame,
    categories: Union[Literal['auto'], List[List[str]], List[str]] = 'auto'
) -> pd.DataFrame:
    """
    Applies Ordinal Encoding (integer mapping) to rank-based features and returns the transformed DataFrame.

    Args:
        data: DataFrame containing categorical features.
        categories: Specifies the order of categories. Can be:
            - 'auto': Categories are inferred and ordered alphabetically.
            - List[List[str]]: The standard sklearn format (e.g., [['S', 'M', 'L'], ['Low', 'Med', 'High']]).
            - List[str]: A single flat list of categories if ONLY ONE COLUMN is in the input data.

    Returns:
        The transformed DataFrame (pd.DataFrame).
    """
    
    # Standardize 'categories' for the sklearn API, handling flat list input
    categories_param = categories

    if categories != 'auto':
        # 1. Ensure input is iterable if not 'auto'
        if not isinstance(categories, (list, tuple, np.ndarray, Collection)):
            raise TypeError("The 'categories' argument must be iterable (list, tuple, etc.) or 'auto'.")

        # 2. Handle the case where the user passes a single FLAT list of categories
        # BUT the input data has multiple columns (which is common).
        # The correct format is always a list of lists, where the outer list has length data.shape[1].
        
        # Check if the input is a list of strings (flat list)
        is_flat_list_of_strings = categories and all(isinstance(c, str) for c in categories)

        if is_flat_list_of_strings:
            num_cols = data.shape[1]
            if num_cols == 1:
                # If only one column, wrap it: ['S', 'M', 'L'] -> [['S', 'M', 'L']]
                categories_param = [list(categories)]
            else:
                # If multiple columns, we cannot safely assume the user meant the same order for all columns.
                # Since the original code implies standardization, we'll repeat the list for all columns
                # as the most likely user intent when passing a single flat list to multi-column data.
                # If this is wrong, the user must provide the standard List[List[str]] format.
                categories_param = [list(categories)] * num_cols
        
        elif categories and not all(isinstance(c, list) for c in categories):
            # If it's a non-flat, non-string list (e.g., list of ints), convert to list to avoid runtime errors
            categories_param = list(categories)

    # 3. Create the encoder with the standardized parameter
    encoder = OrdinalEncoder(
        categories=categories_param
    )
    
    transformed_array = encoder.fit_transform(data)
    
    return pd.DataFrame(
        transformed_array,
        columns=data.columns,
        index=data.index
    )
# --- 2. Target Encoding Function ---

def apply_label_encoder_target(
    target_series: pd.Series
) -> pd.Series:
    """
    Applies Label Encoding to a target variable (y) for classification (0, 1, 2...) and returns the transformed Series.
    
    Args:
        target_series: The target variable (y) as a Series.
        
    Returns:
        The encoded Series (pd.Series).
    """
    encoder = LabelEncoder()
    
    return pd.Series(
        encoder.fit_transform(target_series),
        index=target_series.index,
        name=target_series.name
    )

# =========================================================================
# END OF data_handling_functions.py
# =========================================================================