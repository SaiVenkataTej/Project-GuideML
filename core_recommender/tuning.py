import optuna
import numpy as np
from typing import Dict, Any, Callable, List, Optional, Union
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.base import BaseEstimator

# Import centralized logger
from core_recommender.logger import get_logger

logger = get_logger(__name__)

# =========================================================================
# 🎛️ tuning.py: Modular Hyperparameter Optimization Functions
# =========================================================================

def get_grid_search_tuner(
    estimator: BaseEstimator,
    param_grid: Dict[str, List[Any]],
    cv: Any,
    scoring: str,
    n_jobs: int = -1,
    verbose: int = 1
) -> GridSearchCV:
    """Returns a configured GridSearchCV object for exhaustive hyperparameter search.

    Args:
        estimator (BaseEstimator): The scikit-learn model instance to tune.
        param_grid (Dict[str, List[Any]]): Dictionary with parameter names as keys and lists 
            of parameter settings to try as values.
        cv (Any): Cross-validation splitting strategy (e.g., KFold object or int).
        scoring (str): A single string (e.g., 'accuracy') or a callable to evaluate predictions.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1.
        verbose (int, optional): Controls verbosity. Defaults to 1.

    Returns:
        GridSearchCV: The configured grid search object ready for fitting.
    """
    return GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose
    )

def get_random_search_tuner(
    estimator: BaseEstimator,
    param_distributions: Dict[str, Any],
    cv: Any,
    scoring: str,
    n_iter: int = 10,
    n_jobs: int = -1,
    verbose: int = 1,
    random_state: int = 42
) -> RandomizedSearchCV:
    """Returns a configured RandomizedSearchCV object for random hyperparameter sampling.

    Args:
        estimator (BaseEstimator): The scikit-learn model instance.
        param_distributions (Dict[str, Any]): Dictionary with parameter names as keys and 
            distributions or lists as values.
        cv (Any): Cross-validation splitting strategy.
        scoring (str): Scoring metric string.
        n_iter (int, optional): Number of parameter settings that are sampled. Defaults to 10.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1.
        verbose (int, optional): Controls verbosity. Defaults to 1.
        random_state (int, optional): Seed for random number generator. Defaults to 42.

    Returns:
        RandomizedSearchCV: The configured random search object.
    """
    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state
    )

def get_halving_grid_search_tuner(
    estimator: BaseEstimator,
    param_grid: Dict[str, List[Any]],
    cv: Any,
    scoring: str,
    factor: int = 3,
    n_jobs: int = -1,
    verbose: int = 1,
    random_state: int = 42
) -> Any:
    """Returns a configured HalvingGridSearchCV object using successive halving.

    Args:
        estimator (BaseEstimator): The scikit-learn model instance.
        param_grid (Dict[str, List[Any]]): Dictionary with parameter names as keys and lists of settings.
        cv (Any): Cross-validation splitting strategy.
        scoring (str): Scoring metric string.
        factor (int, optional): The 'halving' parameter. Defaults to 3.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1.
        verbose (int, optional): Controls verbosity. Defaults to 1.
        random_state (int, optional): Seed for random number generator. Defaults to 42.

    Returns:
        HalvingGridSearchCV: The configured halving grid search object.

    Raises:
        ImportError: If scikit-learn version is too old or experimental import fails.
    """
    try:
        from sklearn.experimental import enable_halving_search_cv  # noqa
        from sklearn.model_selection import HalvingGridSearchCV
    except ImportError:
        raise ImportError("HalvingGridSearchCV requires scikit-learn >= 0.24 and explicit import of experimental features.")

    return HalvingGridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        factor=factor,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state
    )

def run_optuna_optimization(
    estimator_class: Any, 
    param_space_func: Callable[[optuna.Trial], Dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    cv: Any,
    scoring: str,
    n_trials: int = 20,
    timeout: Optional[int] = None,
    n_jobs: int = -1,
    random_state: int = 42
) -> BaseEstimator:
    """Executes an Optuna hyperparameter optimization study and returns the best fitted model.
    
    Uses Bayesian Optimization (TPE) to pinpoint promising areas of the hyperparameter space.

    Args:
        estimator_class (Any): The class of the model to instantiate. Not an instance.
        param_space_func (Callable[[optuna.Trial], Dict[str, Any]]): Function that takes a trial 
            and returns hyperparameters.
        X (np.ndarray): Training features.
        y (np.ndarray): Training target.
        cv (Any): Cross-validation splitting strategy.
        scoring (str): Scoring metric string.
        n_trials (int, optional): Number of trials for optimization. Defaults to 20.
        timeout (Optional[int], optional): Stop study after given seconds. Defaults to None.
        n_jobs (int, optional): Number of parallel jobs for the model. Defaults to -1.
        random_state (int, optional): Seed for the sampler. Defaults to 42.

    Returns:
        BaseEstimator: The best model found, fitted on the full dataset.
                       Includes a `.study_` attribute containing the Optuna study.
    """
    
    def objective(trial: optuna.Trial) -> float:
        # 1. Suggest params using the user-provided function
        params = param_space_func(trial)
        
        # 2. Instantiate model
        # We try to pass n_jobs to the model constructor if it accepts it.
        try:
            model = estimator_class(**params, n_jobs=n_jobs)
        except TypeError:
            # Fallback if model doesn't accept n_jobs
            model = estimator_class(**params)

        # 3. Cross-validate
        # Enforce n_jobs=1 for cross_val_score to avoid nested parallelism
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        
        return float(scores.mean())

    # Optuna setup
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    
    logger.info(f"   [Optuna] Running {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    
    logger.info(f"   [Optuna] Completed {n_trials} trials")
    logger.info(f"   [Optuna] Best Score: {study.best_value:.4f}")
    logger.debug(f"   [Optuna] Best Params: {study.best_params}")
    
    # Re-train best model
    best_params = study.best_params
    
    try:
        best_model = estimator_class(**best_params, n_jobs=n_jobs)
    except TypeError:
        best_model = estimator_class(**best_params)
         
    best_model.fit(X, y)
    
    # Attach study for future diagnostics
    best_model.study_ = study
    
    return best_model
