import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    accuracy_score, f1_score, roc_auc_score, 
    precision_score, log_loss, precision_recall_fscore_support
)
from typing import Union, List, Optional, Tuple, Literal, Any
import time

# Use npt.ArrayLike for inputs that can be lists/numpy arrays
ArrayLike = npt.ArrayLike 

# =========================================================================
# 📊 evaluation.py: Modular Evaluation Functions
# =========================================================================

# --- 1. Regression Metrics ---

def calculate_rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculates Root Mean Squared Error (RMSE).

    RMSE is the standard metric for regression, penalizing large errors
    more heavily than small ones.

    Args:
        y_true (ArrayLike): Ground truth (correct) target values.
        y_pred (ArrayLike): Estimated target values.

    Returns:
        float: The root mean squared error. Lower is better.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def calculate_mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculates Mean Absolute Error (MAE).

    Args:
        y_true (ArrayLike): Ground truth (correct) target values.
        y_pred (ArrayLike): Estimated target values.

    Returns:
        float: The mean absolute error. Lower is better.
    """
    return float(mean_absolute_error(y_true, y_pred))

def calculate_r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculates R^2 Score (Coefficient of Determination).

    Args:
        y_true (ArrayLike): Ground truth (correct) target values.
        y_pred (ArrayLike): Estimated target values.

    Returns:
        float: The R^2 score. Best possible score is 1.0.
    """
    return float(r2_score(y_true, y_pred))

def calculate_adjusted_r2(y_true: ArrayLike, y_pred: ArrayLike, n_samples: int, n_features: int) -> float:
    """Calculates Adjusted R^2 Score.
    
    The adjusted R-squared increases only if the new term improves the model 
    more than would be expected by chance.

    Args:
        y_true (ArrayLike): Ground truth (correct) target values.
        y_pred (ArrayLike): Estimated target values.
        n_samples (int): Number of samples in the dataset.
        n_features (int): Number of features used in the model.

    Returns:
        float: The adjusted R^2 score. Returns NaN if n_samples <= n_features + 1.
    """
    r2 = calculate_r2_score(y_true, y_pred)
    
    if (n_samples - n_features - 1) <= 0:
        return np.nan 
        
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

# --- 2. Classification Metrics ---

def calculate_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculates Classification Accuracy.

    Args:
        y_true (ArrayLike): Ground truth (correct) labels.
        y_pred (ArrayLike): Predicted labels.

    Returns:
        float: The accuracy score (fraction of correct predictions).
    """
    return float(accuracy_score(y_true, y_pred))

def calculate_f1_score(y_true: ArrayLike, y_pred: ArrayLike, average: Literal['micro', 'macro', 'samples', 'weighted', 'binary'] = 'weighted') -> float:
    """Calculates F1-Score (harmonic mean of Precision and Recall).

    Useful when classes are imbalanced.

    Args:
        y_true (ArrayLike): Ground truth (correct) labels.
        y_pred (ArrayLike): Predicted labels.
        average (str, optional): Strategy for multiclass aggregation. Defaults to 'weighted'.

    Returns:
        float: The F1 score. 1 is best, 0 is worst.
    """
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))

def calculate_roc_auc_score(
    y_true: ArrayLike, 
    y_proba: ArrayLike, 
    multi_class_strategy: Literal['raise', 'ovo', 'ovr'] = 'ovr', 
    average: Optional[Literal['weighted', 'macro', 'micro']] = 'weighted'
) -> float:
    """Calculates Area Under the Receiver Operating Characteristic Curve (ROC-AUC).
    
    Measures the quality of the model's ranking ability, regardless of the decision threshold.

    Args:
        y_true (ArrayLike): Ground truth (correct) labels.
        y_proba (ArrayLike): Predicted probabilities.
        multi_class_strategy (str, optional): Strategy for handling multi-class ('ovr', 'ovo'). Defaults to 'ovr'.
        average (str, optional): Averaging strategy. Defaults to 'weighted'.

    Returns:
        float: The ROC-AUC score. 1 is perfect, 0.5 is random variance.
    """
    y_proba_arr = np.asarray(y_proba)
    y_true_arr = np.asarray(y_true)
    
    # Check dimensions
    if y_proba_arr.ndim == 2:
        if y_proba_arr.shape[1] == 2:
            # Binary classification: use probability of the positive class
            y_score = y_proba_arr[:, 1]
            multi_class = 'raise' # Not multiclass
        else:
             # Multiclass
             y_score = y_proba_arr
             multi_class = multi_class_strategy
    else:
        # Assumed binary 1D array of probabilities
        y_score = y_proba_arr
        multi_class = 'raise'

    kwargs: Dict[str, Any] = {}
    if average is not None:
        kwargs['average'] = average
    
    if multi_class != 'raise':
        kwargs['multi_class'] = multi_class

    # Scikit-learn's roc_auc_score handles binary cases automatically 
    # if y_score is 1D or (n_samples, 1).
    # For multiclass, it needs multi_class and average params.
    
    try:
        return float(roc_auc_score(y_true_arr, y_score, **kwargs))
    except ValueError:
        # Fallback for edge cases (e.g., only one class present in y_true)
        return 0.5

def calculate_precision(y_true: ArrayLike, y_pred: ArrayLike, average: Literal['micro', 'macro', 'samples', 'weighted', 'binary'] = 'weighted') -> float:
    """Calculates Classification Precision.

    Args:
        y_true (ArrayLike): Ground truth (correct) labels.
        y_pred (ArrayLike): Predicted labels.
        average (str, optional): Aggregation strategy. Defaults to 'weighted'.

    Returns:
        float: The precision score.
    """
    return float(precision_score(y_true, y_pred, average=average, zero_division=0))

def calculate_log_loss(y_true: ArrayLike, y_proba: ArrayLike) -> float:
    """Calculates Log Loss (Cross-Entropy Loss).

    Args:
        y_true (ArrayLike): Ground truth (correct) labels.
        y_proba (ArrayLike): Predicted probabilities.

    Returns:
        float: The log loss. LOWER is better.
    """
    try:
        return float(log_loss(y_true, y_proba))
    except ValueError:
        # Can happen if y_true contains labels not in y_proba columns
        return 10.0 # High penalty

def calculate_precision_recall_score(y_true: ArrayLike, y_pred: ArrayLike, average: Literal['micro', 'macro', 'samples', 'weighted', 'binary'] = 'weighted') -> Tuple[float, float]:
    """Calculates Precision and Recall simultaneously.

    Args:
        y_true (ArrayLike): Ground truth (correct) labels.
        y_pred (ArrayLike): Predicted labels.
        average (str, optional): Aggregation strategy. Defaults to 'weighted'.

    Returns:
        Tuple[float, float]: (precision, recall).
    """
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return float(precision), float(recall)

# --- 3. Model-Specific Diagnostics ---

def get_oob_score(model_instance: Any) -> Optional[float]:
    """Retrieves the Out-of-Bag (OOB) score from a fitted tree-based model.

    Args:
        model_instance (Any): The trained model instance.

    Returns:
        Optional[float]: The OOB score if available, else None.
    """
    if hasattr(model_instance, 'oob_score_') and model_instance.oob_score_ is not None:
        return float(model_instance.oob_score_)
    return None

def get_tree_depth(model_instance: Any) -> Optional[int]:
    """Retrieves the maximum depth of a decision tree or average depth for ensembles.

    Args:
        model_instance (Any): The trained model instance.

    Returns:
        Optional[int]: The depth (or mean depth), else None.
    """
    if hasattr(model_instance, 'tree_'):
        return int(model_instance.tree_.max_depth)
    elif hasattr(model_instance, 'estimators_'):
        depths = [e.tree_.max_depth for e in model_instance.estimators_ if hasattr(e, 'tree_')]
        if depths:
            return int(round(np.mean(depths)))
    return None

def get_leaf_count(model_instance: Any) -> Optional[int]:
    """Retrieves the number of leaf nodes for a decision tree or average leaf count for ensembles.

    Args:
        model_instance (Any): The trained model instance.

    Returns:
        Optional[int]: The leaf count (or mean leaf count), else None.
    """
    if hasattr(model_instance, 'tree_'):
        return int(model_instance.tree_.n_leaves)
    elif hasattr(model_instance, 'estimators_'):
        leaf_counts = [e.tree_.n_leaves for e in model_instance.estimators_ if hasattr(e, 'tree_')]
        if leaf_counts:
            return int(round(np.mean(leaf_counts)))
    return None

def measure_prediction_latency(model_instance: Any, X_test: np.ndarray, n_runs: int = 100) -> float:
    """Measures the average time taken for a model to predict on the test set.

    Args:
        model_instance (Any): The trained model instance.
        X_test (np.ndarray): Test data features.
        n_runs (int, optional): Number of runs to average. Defaults to 100.

    Returns:
        float: Average latency per prediction run in seconds.
    """
    if not hasattr(model_instance, 'predict'):
        return np.nan
    
    # Warm-up
    try:
        model_instance.predict(X_test[:1])
    except Exception:
        pass 
    
    start_time = time.time()
    for _ in range(n_runs):
        model_instance.predict(X_test)
    end_time = time.time()
    
    return float((end_time - start_time) / n_runs)