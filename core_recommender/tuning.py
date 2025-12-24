
import optuna
import numpy as np
from typing import Dict, Any, Callable, List, Optional
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.base import BaseEstimator

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
    """
    Returns a configured GridSearchCV object (Standard Exhaustive Search).
    
    Matches the factory pattern of preprocessing.py.

    Args:
        estimator: The scikit-learn model instance to tune.
        param_grid: Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
        cv: Cross-validation splitting strategy (e.g., KFold object).
        scoring: A single string (e.g., 'accuracy') or a callable to evaluate the predictions on the test set.
        n_jobs: Number of jobs to run in parallel. Defaults to -1 (all processors).
        verbose: Controls the verbosity: the higher, the more messages.

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
) -> Any:
    """
    Returns a configured RandomizedSearchCV object (Random Sampling Search).
    
    Args:
        estimator: The scikit-learn model instance.
        param_distributions: Dictionary with parameters names (str) as keys and distributions or lists as values.
        cv: Cross-validation splitting strategy.
        scoring: Scoring metric string.
        n_iter: Number of parameter settings that are sampled.
        n_jobs: Number of jobs to run in parallel.
        verbose: Controls verbosity.
        random_state: Seed for random number generator.

    Returns:
        RandomizedSearchCV: The configured random search object.
    """
    from sklearn.model_selection import RandomizedSearchCV
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
    """
    Returns a configured HalvingGridSearchCV object (Successive Halving).
    
    Args:
        estimator: The scikit-learn model instance.
        param_grid: Dictionary with parameters names (str) as keys and lists of parameter settings.
        cv: Cross-validation splitting strategy.
        scoring: Scoring metric string.
        factor: The 'halving' parameter, determining the proportion of candidates selected for each subsequent iteration.
        n_jobs: Number of jobs to run in parallel.
        verbose: Controls verbosity.
        random_state: Seed for random number generator.

    Returns:
        HalvingGridSearchCV: The configured halving grid search object.
    """
    # HalvingGridSearchCV is experimental in some versions, import locally to handle potential variability
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
    """
    Executes an Optuna hyperparameter optimization study (Bayesian Optimization) and returns the best fitted model.
    
    Matches the functional execution pattern of dataHandling.py.

    Args:
        estimator_class: The class of the model to instantiate (e.g., KNeighborsClassifier). Not an instance.
        param_space_func: A function that takes an optuna.Trial and returns a dictionary of hyperparameters.
        X: Training features.
        y: Training target.
        cv: Cross-validation splitting strategy.
        scoring: Scoring metric string.
        n_trials: Number of trials (iterations) for optimization. Defaults to 20.
        timeout: Stop study after the given number of seconds. Defaults to None.
        n_jobs: Number of parallel jobs for the *model* (if applicable). 
        random_state: Seed for the sampler.

    Returns:
        BaseEstimator: The best model found, already fitted on the full dataset.
                       The returned model has an attached attribute `.study_` containing the Optuna study.
    """
    
    def objective(trial):
        # 1. Suggest params using the user-provided function
        params = param_space_func(trial)
        
        # 2. Instantiate model
        # We try to pass n_jobs to the model constructor if it accepts it.
        # This allows the model to use parallelism during its own fit/predict if CV is serial.
        try:
            model = estimator_class(**params, n_jobs=n_jobs)
        except TypeError:
            # Fallback if model doesn't accept n_jobs (e.g., some simple regressors)
            model = estimator_class(**params)

        # 3. Cross-validate
        # We enforce n_jobs=1 for cross_val_score to avoid nested parallelism oversubscription
        # (Model handles threads internally via n_jobs, or Optuna runs sequential trials)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        
        return scores.mean()

    # Optuna setup
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    
    print(f"   [Optuna] Running {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    print(f"   [Optuna] Best Score: {study.best_value:.4f}")
    print(f"   [Optuna] Best Params: {study.best_params}")
    
    # Re-train best model
    best_params = study.best_params
    
    try:
        best_model = estimator_class(**best_params, n_jobs=n_jobs)
    except TypeError:
        best_model = estimator_class(**best_params)
         
    best_model.fit(X, y)
    
    # Attach study for future diagnostics (e.g. elbow plots, parallel ccoard)
    best_model.study_ = study
    
    return best_model
