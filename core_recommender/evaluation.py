import numpy as np
import numpy.typing as npt # <-- ADDED FOR TYPING
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    accuracy_score, f1_score, roc_auc_score, 
    precision_score, log_loss, precision_recall_fscore_support
)
from typing import Union, List, Optional, Tuple, Literal, Any
import time
from collections.abc import Callable

# Define common data types for typing clarity
# Use ArrayLike for inputs that can be lists/numpy arrays
ArrayLike = npt.ArrayLike 
Array = np.ndarray # Use Array for variables *known* to be numpy arrays

# =========================================================================
# 📊 evaluation_metrics.py: Modular Evaluation Functions
# =========================================================================

# --- 1. Regression Metrics ---

# Changed Array to ArrayLike in all function signatures
def calculate_rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculates Root Mean Squared Error (RMSE)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculates Mean Absolute Error (MAE)."""
    return mean_absolute_error(y_true, y_pred)

def calculate_r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculates R^2 Score (Coefficient of Determination)."""
    return r2_score(y_true, y_pred)

def calculate_adjusted_r2(y_true: ArrayLike, y_pred: ArrayLike, n_samples: int, n_features: int) -> float:
    """Calculates Adjusted R^2 Score."""
    r2 = calculate_r2_score(y_true, y_pred)
    
    n = n_samples
    p = n_features
    
    # Adjusted R^2 = 1 - [(1 - R^2) * (n - 1) / (n - p - 1)]
    if (n - p - 1) <= 0:
        return np.nan 
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# --- 2. Classification Metrics ---

def calculate_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculates Classification Accuracy."""
    return float(accuracy_score(y_true, y_pred))

def calculate_f1_score(y_true: ArrayLike, y_pred: ArrayLike, average: Literal['micro', 'macro', 'samples', 'weighted', 'binary'] = 'weighted') -> float:
    """Calculates F1-Score (balance of Precision and Recall)."""
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))

def calculate_roc_auc_score(
    y_true: ArrayLike, 
    y_proba: ArrayLike, 
    multi_class_strategy: Literal['raise', 'ovo', 'ovr'] = 'ovr', 
    average: Optional[Literal['weighted']] = 'weighted'
) -> float:
    """
    Calculates Area Under the Receiver Operating Characteristic Curve (ROC-AUC).
    Requires probability scores (y_proba). Enhanced for robust multi-class support.
    """
    # Ensure inputs are NumPy arrays for safety
    y_proba_arr = np.asarray(y_proba) # Renamed to y_proba_arr
    y_true_arr = np.asarray(y_true) # Renamed to y_true_arr
    
    # Initialize variables
    y_score = y_proba_arr
    is_multiclass = False
    
    if y_proba_arr.ndim == 2:
        if y_proba_arr.shape[1] > 2:
            # Multi-class (N, M) case. y_score remains (N, M).
            is_multiclass = True
        else:
            # Binary (N, 2): Extract the positive class probability (index 1)
            y_score = y_proba_arr[:, 1]
            
    elif y_proba_arr.ndim == 1:
        # Binary (N,). Single score array. y_score is already y_proba_arr
        pass 
    else:
        raise ValueError("Invalid y_proba format. Expected shape (N,), (N, 2), or (N, M) for M > 2 classes.")
    
    # Conditionally build kwargs with only valid arguments
    kwargs: dict[str, Any] = {}
    if average is not None:
        kwargs['average'] = average
    
    if is_multiclass:
        if multi_class_strategy == 'raise':
            raise ValueError(
                "Multi-class input detected. Specify multi_class_strategy as 'ovo' or 'ovr'."
            )
        kwargs['multi_class'] = multi_class_strategy
        
    return float(roc_auc_score(y_true_arr, y_score, **kwargs)) # Pass y_true_arr and y_score

def calculate_precision(y_true: ArrayLike, y_pred: ArrayLike, average: Literal['micro', 'macro', 'samples', 'weighted', 'binary'] = 'weighted') -> float:
    """Calculates Classification Precision."""
    return float(precision_score(y_true, y_pred, average=average, zero_division=0))

def calculate_log_loss(y_true: ArrayLike, y_proba: ArrayLike) -> float:
    """Calculates Log Loss (Cross-Entropy Loss)."""
    return float(log_loss(y_true, y_proba))

def calculate_precision_recall_score(y_true: ArrayLike, y_pred: ArrayLike, average: Literal['micro', 'macro', 'samples', 'weighted', 'binary'] = 'weighted') -> Tuple[float, float]:
    """Calculates Precision and Recall simultaneously."""
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    # Ensure result is a float
    return float(precision), float(recall)

# --- 3. Model-Specific Diagnostics (Input remains compatible with Any or object) ---

def get_oob_score(model_instance) -> Optional[float]:
    """Retrieves the Out-of-Bag (OOB) score from a fitted tree-based model (Random Forest)."""
    if hasattr(model_instance, 'oob_score_') and model_instance.oob_score_ is not None:
        return model_instance.oob_score_
    return None

def get_tree_depth(model_instance) -> Optional[int]:
    """Retrieves the maximum depth of a decision tree or average depth for ensembles."""
    if hasattr(model_instance, 'tree_'):
        return model_instance.tree_.max_depth
    elif hasattr(model_instance, 'estimators_'):
        # Calculate mean depth for ensemble models
        depths = [e.tree_.max_depth for e in model_instance.estimators_ if hasattr(e, 'tree_')]
        return int(round(np.mean(depths))) if depths else None
    return None

def get_leaf_count(model_instance) -> Optional[int]:
    """Retrieves the number of leaf nodes for a decision tree or average leaf count for ensembles."""
    if hasattr(model_instance, 'tree_'):
        return model_instance.tree_.n_leaves
    elif hasattr(model_instance, 'estimators_'):
        # Calculate mean leaf count for ensemble models
        leaf_counts = [e.tree_.n_leaves for e in model_instance.estimators_ if hasattr(e, 'tree_')]
        return int(round(np.mean(leaf_counts))) if leaf_counts else None
    return None

def measure_prediction_latency(model_instance, X_test: np.ndarray, n_runs: int = 100) -> float:
    """Measures the average time taken for a model (like KNN) to predict on the test set."""
    if not hasattr(model_instance, 'predict'):
        return np.nan
    
    # Warm-up run
    try:
        model_instance.predict(X_test[:1])
    except Exception:
        # Catch any errors during warm-up and ignore (e.g., if X_test is empty)
        pass 
    
    start_time = time.time()
    for _ in range(n_runs):
        model_instance.predict(X_test)
    end_time = time.time()
    
    # Return average latency per prediction run in seconds
    return (end_time - start_time) / n_runs