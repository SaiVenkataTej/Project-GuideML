# Project Change Log

This document tracks detailed information about changes made to the codebase, including the reasoning ("Why") and the technical implementation ("How").

## 2025-12-20

### Fix: naming inconsistency in `core_recommender/modeling/svms.py`

**Why:**
During the verification of the project execution using `demo_run.py`, the application crashed with an `ImportError`.
The error message was: `ImportError: cannot import name 'calculate_r2' from 'core_recommender.evaluation'`.
This indicated that the `SVMModel` class was attempting to import a function `calculate_r2` that did not exist in the `evaluation.py` module. Upon inspection of `evaluation.py`, the correct function name was found to be `calculate_r2_score`.

**How:**
1.  **Modified `core_recommender/modeling/svms.py`**:
    *   **Import Statement**: Changed `from core_recommender.evaluation import ..., calculate_r2` to `from core_recommender.evaluation import ..., calculate_r2_score`.
    *   **Method Usage**: Inside the `calculate_metrics` method, updated the call `metrics['R2 Score'] = calculate_r2(y_test, y_pred)` to `metrics['R2 Score'] = calculate_r2_score(y_test, y_pred)`.

**Impact:**
This change resolves the `ImportError` and allows the `SVMModel` (and subsequently the `ModelExecutor`) to load and execute correctly. It ensures consistency between the evaluation module and the modeling modules.

### Fix: constructor signature mismatch in `core_recommender/modeling/randomForest.py`

**Why:**
The verification script `demo_run.py` failed with a `TypeError` when initializing the `RandomForestModel`.
The error was: `TypeError: RandomForestModel.__init__() got an unexpected keyword argument 'is_classification'`.
This occurred because the `ModelExecutor` (in `execution.py`) instantiates models with `is_classification=...` argument to explicitly set their mode, but the `RandomForestModel` constructor did not accept this argument, relying instead on internal inference or config.

**How:**
1.  **Modified `core_recommender/modeling/randomForest.py`**:
    *   **Constructor Update**: Updated `__init__` method signature to accept `is_classification: bool = True`.
    *   **Logic Update**: Added `self.is_classification = is_classification` to properly store the passed argument, ensuring it aligns with the orchestrator's expectations.

**Impact:**
This ensures `RandomForestModel` aligns with the standard `BaseModel` contract used by `ModelExecutor`, allowing it to be instantiated correctly during the pipeline execution.

### Fix: invalid parameter handling in GridSearchCV across multiple models

**Why:**
Running `demo_run.py` revealed `ValueError: Invalid parameter ... for estimator` errors during the execution of `GridSearchCV` (or `RandomizedSearchCV`) for multiple models (`KNN`, `Logistic Regression`, `Decision Tree`, `SVM`, `Linear Regression`, `Naive Bayes`).
The error occurred because the parameter grids defined in these classes used the prefix `estimator__` (e.g., `estimator__n_neighbors`). This prefix is conventionally used when `GridSearchCV` wraps a `Pipeline` and needs to route parameters to a named step (usually called `'estimator'`).
However, in our architecture, `GridSearchCV` wraps the model instance directly (e.g., `KNeighborsClassifier`), not a `Pipeline` containing it. Therefore, `GridSearchCV` expects parameters to match the estimator's arguments directly (e.g., `n_neighbors`).

**How:**
1.  **Modified `param_grid` (or `param_distributions`) in the following files**:
    *   `core_recommender/modeling/knn.py`
    *   `core_recommender/modeling/logisticRegression.py`
    *   `core_recommender/modeling/decisionTrees.py`
    *   `core_recommender/modeling/svms.py`
    *   `core_recommender/modeling/linearRegression.py`
    *   `core_recommender/modeling/naiveBayes.py`
2.  **Specific Changes**: Removed the `estimator__` prefix from all keys in the parameter dictionaries.
    *   *Example (KNN)*: Changed `'estimator__n_neighbors'` to `'n_neighbors'`.
    *   *Example (Linear Regression)*: Changed `'estimator__alpha'` to `'alpha'`.

**Impact:**
This change aligns the parameter grid structure with the estimator being validated, allowing `GridSearchCV` to correctly set parameters and execute the cross-validation process without error.

### Fix: `AttributeError` in `ModelExecutor` during regression evaluation

**Why:**
The regression pipeline execution failed for several models (`KNN`, `Decision Tree`, `SVM`) with the error `AttributeError: 'NoneType' object has no attribute 'transform'`.
This occurred in `core_recommender/execution.py` when the code attempted to transform the test target (`y_test`) using `model.label_encoder`.
The code checked `if hasattr(model, 'label_encoder'):`, which evaluates to `True` because these models explicitly set `self.label_encoder = None` during preprocessing (for regression tasks). Since the value was `None`, calling `.transform()` on it caused the crash.

**How:**
1.  **Modified `core_recommender/execution.py`**:
    *   Updated the condition to: `if hasattr(model, 'label_encoder') and model.label_encoder is not None:`.

**Impact:**
This prevents the executor from attempting to use a `None` label encoder, correctly falling back to using the raw `y_test` values for regression models, ensuring the pipeline completes successfully.
