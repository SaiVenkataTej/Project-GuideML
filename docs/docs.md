# GuideML — Source Documentation

**Version:** 1.0 (V1) | **Type:** Local AutoML Application | **Stack:** Python, Flask, scikit-learn, Optuna

This document is the canonical technical reference for the GuideML source code. Every section is derived directly from reading the actual files in the repository. Nothing here is assumed or invented.

---

## Table of Contents

1. [Project Purpose](#1-project-purpose)
2. [Repository Layout](#2-repository-layout)
3. [Core Package — `core_recommender/`](#3-core-package--core_recommender)
   - [logger.py](#31-loggerpy)
   - [exceptions.py](#32-exceptionspy)
   - [profiling.py](#33-profilingpy)
   - [preprocessing.py](#34-preprocessingpy)
   - [data_handling.py](#35-data_handlingpy)
   - [tuning.py](#36-tuningpy)
   - [evaluation.py](#37-evaluationpy)
   - [knowledge_base.py](#38-knowledge_basepy)
   - [visualization.py](#39-visualizationpy)
   - [execution.py — The Pipeline Orchestrator](#310-executionpy--the-pipeline-orchestrator)
4. [Modeling Sub-package — `core_recommender/modeling/`](#4-modeling-sub-package--core_recommendermodeling)
   - [registry.py](#41-registrypy)
   - [base_model.py](#42-base_modelpy)
   - [Concrete Models](#43-concrete-models)
5. [Interface Layer — `interface/`](#5-interface-layer--interface)
6. [Logging System — `logs/`](#6-logging-system--logs)
7. [Actual Pipeline Execution Trace](#7-actual-pipeline-execution-trace)
8. [Design Principles Applied](#8-design-principles-applied)
9. [Honest Limitations](#9-honest-limitations)

---

## 1. Project Purpose

GuideML automates the **initial model selection phase** of the machine learning lifecycle for tabular CSV data. A user uploads a dataset, selects which models to test, and the application runs a full pipeline: cleaning → profiling → splitting → training → hyperparameter tuning → evaluation → SHAP explainability → a ranked leaderboard. The best model is downloadable as a `.pkl` artifact.

**What it is not:** It does not support deep learning, cloud deployment, multi-user sessions, or advanced feature engineering. These are explicitly excluded from V1 scope.

---

## 2. Repository Layout

```
Folder1/
├── core_recommender/       # All ML logic — completely decoupled from UI
│   ├── modeling/           # Individual model implementations + registry
│   ├── execution.py        # Main pipeline orchestrator (ModelExecutor)
│   ├── profiling.py        # Heuristic dataset profiler (DataProfiler)
│   ├── preprocessing.py    # Pure functions returning sklearn transformers
│   ├── data_handling.py    # Encoding utility functions
│   ├── tuning.py           # Hyperparameter optimization functions
│   ├── evaluation.py       # Metric calculation functions
│   ├── knowledge_base.py   # Static model descriptions for UI rendering
│   ├── visualization.py    # Matplotlib/Seaborn plot generation
│   ├── exceptions.py       # Custom exception hierarchy
│   └── logger.py           # Centralized logging configuration
├── interface/
│   ├── app.py              # Flask web server — 6 routes
│   ├── templates/          # Jinja2 HTML templates
│   └── static/             # CSS, JS, images, uploaded datasets, saved models
├── docs/                   # All documentation
├── logs/
│   ├── guideml.log         # All INFO and above logs from every module
│   └── error.log           # WARNING and above only (errors, warnings)
├── data/                   # Input datasets
├── tests/                  # pytest unit tests
├── scripts/                # Utility / demo scripts
└── requirements.txt
```

---

## 3. Core Package — `core_recommender/`

### 3.1 `logger.py`

**Single responsibility:** Configure and distribute loggers across every module.

Every module calls `get_logger(__name__)`, which returns a `logging.Logger` bound to two file handlers and one console handler. Handlers are only added once — re-calls to `get_logger` with the same name return the existing logger (guarded by `if logger.handlers: return logger`).

| Handler | Level | Destination |
|---|---|---|
| Console (`StreamHandler`) | `INFO` | `stdout` (UTF-8 forced for Windows emoji compatibility) |
| File handler | `DEBUG` | `logs/guideml.log` (append mode) |
| Error file handler | `WARNING` | `logs/error.log` (append mode) |

**Key log format:** `%(asctime)s | %(levelname)-8s | %(name)s | %(message)s`

**`ProgressLogHandler`** is a custom `logging.Handler` that forwards log records containing a `progress_percent` attribute to a callback function. This is the hook designed for future Flask/UI real-time progress integration. `log_progress()` is a helper that creates a log record with this attribute attached.

`set_log_level(level)` globally adjusts verbosity for the entire `core_recommender` logger tree.

---

### 3.2 `exceptions.py`

A typed exception hierarchy rooted at `GuideMLError(Exception)`. The rationale is documented in the module: previously all modules raised generic built-ins (`ValueError`, `RuntimeError`), making it impossible to distinguish data problems from model problems in `except` clauses.

```
GuideMLError
├── DataValidationError         # target column missing, NaNs in y, wrong types
│   └── InsufficientDataError   # empty DataFrame, fewer rows than CV folds
├── ModelTrainingError          # estimator fit() failed, Optuna failed
│   ├── ModelNotFittedError     # export()/predict() called before fit()
│   └── ModelExportError        # joblib.dump() failed (OSError wrapper)
├── ConfigurationError          # unknown task type, unknown score_func string
│   └── RegistryError           # invalid @register_model task string
├── PipelineError               # no models remain after filtering, unhandled preprocessing
└── VisualisationError          # image save path invalid, array shape mismatch
```

All exceptions accept `**context` keyword arguments stored on `self.context`. `__str__` appends them to the message: `"Target not found [column='rating']"`.

---

### 3.3 `profiling.py`

**Class:** `DataProfiler`

Called once per pipeline run by `ModelExecutor.run()` before training begins. Takes `X: pd.DataFrame` and `y: pd.Series`.

**What `analyze()` computes:**

| Metric | How it's computed | What it means |
|---|---|---|
| `n_samples` | `X.shape[0]` | Row count |
| `n_features` | `X.shape[1]` | Feature count |
| `ratio` | `n_samples / max(n_features, 1)` | Sample-to-feature ratio (overfitting risk) |
| `sparsity` | `(zero-count / total elements)` on numeric columns only | Proportion of zeros |
| `is_gaussian` | Shapiro-Wilk test on up to 3 sampled features (max 500 rows each), `p > 0.05` for all | Whether features appear normally distributed |
| `linearity_score` | Mean of `abs(Pearson correlation)` between all numeric features and `y` | Average linear relationship strength |
| `max_linear_score` | Max of same correlations | Strongest single feature's linear relationship |

**`_generate_rules()`** fires five heuristic rules that append plain-English `messages` to `suggestions`:

1. `n_samples > 20000` → warns that SVM and KNN may be slow.
2. `n_features > n_samples` → recommends regularization (p > n problem).
3. `is_gaussian == True` → informs features appear normally distributed.
4. `max_linear_score > 0.7` → reports strong linear signal.
5. `max_linear_score < 0.2` AND `n_samples > 100` AND `y` is numeric → reports **"Weak linear signal detected."** (this is what fired in the actual run log).
6. Classification only: if `min_class_proportion < 10%` → class imbalance warning.

**Actual run output (from `logs/guideml.log`):**
```
Profile Summary: {'n_samples': 2800, 'n_features': 15, 'ratio': 186.67,
 'sparsity': 0.02698, 'is_gaussian': False,
 'linearity_score': 0.02148, 'max_linear_score': 0.04737}
Suggestions: ['Weak linear signal detected.']
```
The `max_linear_score` of `0.047` is well below the `0.2` threshold, so Rule 5 fired. This means linear models like ElasticNet are unlikely to perform well on this data.

---

### 3.4 `preprocessing.py`

Pure factory functions — each returns a **configured but unfitted** scikit-learn transformer. No state is held in this module.

**1. Imputation**
- `get_imputer(strategy='median', fill_value=None)` → `SimpleImputer`
  - Strategies: `'mean'`, `'median'`, `'most_frequent'`, `'constant'`

**2. Encoding**
- `get_one_hot_encoder(handle_unknown='ignore', sparse_output=False)` → `OneHotEncoder`
  - `handle_unknown='ignore'` silently zeros out unseen categories at test time (production-safe).
- `get_ordinal_encoder(handle_unknown='use_encoded_value', unknown_value=-1)` → `OrdinalEncoder`

**3. Scaling**
- `get_standard_scaler()` → `StandardScaler` (Z-score: mean=0, std=1)
- `get_minmax_scaler()` → `MinMaxScaler` (range [0,1])
- `get_robust_scaler()` → `RobustScaler` (IQR-based, outlier-resistant)

**4. Distribution Transformation**
- `get_log_transformer()` → `FunctionTransformer(np.log1p, inverse=np.expm1)` — reduces right-skew
- `get_box_cox_transformer()` → `PowerTransformer(method='box-cox')` — strictly positive data only
- `get_yeo_johnson_transformer()` → `PowerTransformer(method='yeo-johnson')` — supports zeros and negatives

**5. Feature Selection & Dimensionality Reduction**
- `get_variance_threshold(threshold=0.0)` → removes zero-variance features
- `get_select_k_best(k=10, score_func='f_regression')` → `SelectKBest`
  - Supported strings: `'f_regression'`, `'f_classif'`, `'chi2'`, `'mutual_info_classif'`
  - Raises `ConfigurationError` on unknown strings
- `get_rfe_selector(estimator, n_features_to_select=10, step=1)` → `RFE`
- `get_select_from_model(estimator, threshold='median')` → `SelectFromModel`
- `get_pca_reducer(n_components=0.95)` → `PCA` (float < 1 = variance to retain)
- `get_nca_reducer(n_components=None, random_state=None)` → `NeighborhoodComponentsAnalysis`

---

### 3.5 `data_handling.py`

Three standalone functions for encoding. These are separate from `preprocessing.py` because they operate on `pd.DataFrame` and return `pd.DataFrame` (preserving index and column names), whereas `preprocessing.py` returns unfitted sklearn objects.

- **`apply_one_hot_encoder(data, drop_first=False, handle_unknown_param='ignore')`**
  Applies OHE to a full DataFrame. Uses `get_feature_names_out()` to recover named columns. Returns a new DataFrame with preserved index.

- **`apply_ordinal_encoder(data, categories='auto')`**
  Handles three forms of the `categories` argument:
  - `'auto'` — sklearn infers from data.
  - `List[List[str]]` — standard per-column specification.
  - `List[str]` (flat list) — convenience form. If `data` has one column, wraps in outer list. If multiple columns, repeats the list for all columns (documented assumption).

- **`apply_label_encoder_target(target_series)`**
  Encodes a classification target `y` to integers 0, 1, 2... Preserves series name and index. Used for standalone encoding; `ModelExecutor` uses `sklearn.LabelEncoder` directly for encoding during the pipeline run.

---

### 3.6 `tuning.py`

Four hyperparameter optimization strategies. Each model in `modeling/` chooses one internally.

**`get_grid_search_tuner(estimator, param_grid, cv, scoring, n_jobs=-1)`**
Returns `GridSearchCV`. Exhaustive — tries every combination. Used by simpler models (Linear Regression, Logistic Regression).

**`get_random_search_tuner(estimator, param_distributions, cv, scoring, n_iter=10, random_state=42)`**
Returns `RandomizedSearchCV`. Samples `n_iter` combinations. Used by Decision Trees.

**`get_halving_grid_search_tuner(estimator, param_grid, cv, scoring, factor=3, random_state=42)`**
Returns `HalvingGridSearchCV`. Uses successive halving — starts with all candidates on a small budget, eliminates the worst fraction (`1/factor`) each round. Requires `from sklearn.experimental import enable_halving_search_cv`. Used by KNN.

**`run_optuna_optimization(estimator_class, param_space_func, X, y, cv, scoring, n_trials=20, random_state=42)`**
The most powerful tuner. Uses Optuna's **Tree-structured Parzen Estimator (TPE)** sampler — a form of Bayesian Optimization.

How it works:
1. `param_space_func(trial)` is a user-supplied callable that uses `trial.suggest_*()` methods to define the search space.
2. Each trial instantiates and cross-validates one parameter set.
3. TPE builds a probabilistic model of which hyperparameters lead to better scores and samples the next trial accordingly. This is smarter than random search.
4. After `n_trials`, the best parameters are used to refit a final model on all training data.
5. The `study_` attribute is attached to the returned model for later diagnostics.

Used by **Random Forest** and **SVM**.

> **Honesty note:** `optuna.logging.set_verbosity(optuna.logging.WARNING)` suppresses Optuna's own verbose output. Optuna trials appear silent in the console during runs.

---

### 3.7 `evaluation.py`

Standalone metric functions. All accept `ArrayLike` (lists, numpy arrays, pandas Series). All return `float`.

**Regression metrics:**
- `calculate_rmse(y_true, y_pred)` — `sqrt(MSE)`. Primary ranking metric for regression. **Lower is better.**
- `calculate_mae(y_true, y_pred)` — Mean Absolute Error. Less sensitive to outliers than RMSE.
- `calculate_r2_score(y_true, y_pred)` — Coefficient of determination. 1.0 = perfect.
- `calculate_adjusted_r2(y_true, y_pred, n_samples, n_features)` — Penalizes for added features. Returns `np.nan` if `n_samples <= n_features + 1`.

**Classification metrics:**
- `calculate_accuracy(y_true, y_pred)` — Fraction correct.
- `calculate_f1_score(y_true, y_pred, average='weighted')` — Harmonic mean of precision and recall. Primary ranking metric for classification. Handles class imbalance via `average`.
- `calculate_roc_auc_score(y_true, y_proba, multi_class_strategy='ovr', average='weighted')` — Handles binary (uses positive-class column `y_proba[:, 1]`) and multiclass (uses full probability matrix). Falls back to `0.5` on `ValueError`.
- `calculate_precision(y_true, y_pred, average='weighted')` — Precision score.
- `calculate_log_loss(y_true, y_proba)` — Cross-entropy. Falls back to `10.0` on error (high penalty).
- `calculate_precision_recall_score(y_true, y_pred)` → `Tuple[float, float]`

**Model-specific diagnostics:**
- `get_oob_score(model_instance)` — Returns `model.oob_score_` if the attribute exists (Random Forest with `oob_score=True`).
- `get_tree_depth(model_instance)` — `tree_.max_depth` for single trees, mean across `estimators_` for ensembles.
- `get_leaf_count(model_instance)` — `tree_.n_leaves` for single trees, mean for ensembles.
- `measure_prediction_latency(model_instance, X_test, n_runs=100)` — Runs `predict()` 100 times after a warm-up call. Returns average seconds per call.

---

### 3.8 `knowledge_base.py`

A single dictionary `MODEL_KNOWLEDGE` mapping model name keys to dicts with `title`, `story`, `how_it_works`, `pros`, `cons`, `best_for`. Consumed by the Flask app (`app.py` imports it) to populate the dashboard UI with plain-English model explanations. No logic — pure static data.

Keys: `"KNN"`, `"Random Forest"`, `"Decision Tree"`, `"SVM"`, `"Naive Bayes"`, `"Logistic Regression"`, `"Linear Regression"`.

---

### 3.9 `visualization.py`

Plotting functions using Matplotlib and Seaborn with a **non-interactive (`Agg`) backend** — critical for server-side rendering in Flask where there is no display.

Key functions (based on `app.py` imports):
- `plot_correlation_heatmap(df, target_column, save_path)` — Pearson correlation heatmap.
- `plot_feature_histograms(df, columns, save_path)` — Up to 6 numeric features.
- `plot_roc_curve(y_true, y_proba, model_name, save_path)` — Binary classification only.
- `plot_confusion_matrix(y_true, y_pred, classes, save_path)` — Labelled confusion matrix.
- `plot_feature_importance(model, feature_names, save_path)` — For tree-based models.
- `plot_coefficient_bar_chart(coefficients, feature_names, save_path)` — For linear models.
- `plot_predicted_vs_actual(y_true, y_pred, save_path)` — Regression diagnostic.
- `plot_residual_plot(y_true, y_pred, save_path)` — Residuals vs. fitted values.
- `plot_shap_summary(model, X_test_df, model_name, save_path)` — Global SHAP feature importance.

---

### 3.10 `execution.py` — The Pipeline Orchestrator

**Class:** `ModelExecutor`

This is the central class. `run()` is the single public entry point. It coordinates all other modules. 670 lines.

**Constructor:** `ModelExecutor(task_type='auto', n_jobs=-1, random_state=42)`

**Attributes after `run()` completes:**
- `self.results` — sorted list of all model result dicts
- `self.best_model_name` — name of the winner
- `self.best_model_metrics` — metrics dict of the winner
- `self.best_model_instance` — the actual fitted `BaseModel` object
- `self.pipeline_log` — list of `{step, details, icon, timestamp}` dicts shown in the UI's pipeline trace panel

**`_infer_task_type(y)`** — Heuristic:
```
float dtype                        → REGRESSION
object / bool / Categorical dtype  → CLASSIFICATION (with numeric conversion attempt)
integer, nunique < 20              → CLASSIFICATION
integer, nunique >= 20             → REGRESSION
```
In the actual run: `'potential_rating'` is integer with **34 unique values** → `REGRESSION`.

**`_handle_outliers(df, target_column)`** — IQR method on the target variable only (not features):
```
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
Lower bound = Q1 - 1.5 * IQR
Upper bound = Q3 + 1.5 * IQR
```
Rows outside bounds are dropped. Regression only. Logs a `WARNING` if any rows are removed.

**`_detect_and_drop_leakage(df, target_column, threshold=0.95)`** — Three checks:
1. **Correlation leakage:** Any numeric feature with `abs(Pearson corr) > 0.95` to the target is dropped.
2. **Identity leakage:** Checks if any two features multiply together to equal the target (`np.allclose`).
3. **Categorical leakage:** Flags columns whose names contain `'target'`, `'label'`, `'result'`, `'outcome'`, `'output'`, or that have a perfect 1:1 value mapping with the target.

In the actual run: `"✅ No obvious leakage detected."`

**`_get_candidate_models(task_type, include_models=None)`** — Pulls classes from `get_registered_models(task_type)`. Uses `inspect.signature()` to check if a model class accepts `is_classification` — dual-task models (KNN, SVM, Random Forest) accept it; single-task models (Logistic Regression, Linear Regression) do not. Applies a name-substring whitelist filter when `include_models` is provided.

**`_train_single_model(model, X_train, y_train, X_test, y_test, progress_callback)`** — Per-model training:
1. `model.preprocess(X_train, y_train)` — builds a `ColumnTransformer` (preprocessor) fitted on training data.
2. `model.fit(X_train, y_train_encoded)` — trains the sklearn estimator (internally wraps in a Pipeline with the preprocessor).
3. `model.calculate_metrics(X_test, y_test)` — evaluates on raw `X_test` (the fitted pipeline handles transformation internally).
4. `preprocessor.transform(X_test)` — produces processed arrays for SHAP.
5. Returns a dict with `status`, `metrics`, `diagnostics`, `model_instance`, `preprocessor`, `test_data_proc`.
6. On any exception: logs `ERROR` with `exc_info=True` and returns `{'status': 'failed', 'error': ...}` — the pipeline continues with remaining models.

**`run(df, target_column, progress_callback=None, include_models=None)`** — Full 13-step pipeline:

| Step | What happens |
|---|---|
| 1 | Validates `target_column` exists. Raises `DataValidationError` if not. |
| 2 | Splits `df` into `X` and `y`. Auto-converts object columns to numeric where possible. |
| 3 | `_infer_task_type(y)` |
| 4 | `_handle_outliers()` on the target — regression only |
| 5 | `DataProfiler().analyze(X, y)` — generates profile and suggestion messages |
| 6 | `_detect_and_drop_leakage()` |
| 7 | Target encoding: `LabelEncoder` for classification, `pd.to_numeric` for regression |
| 8 | `train_test_split(test_size=0.2, random_state=42, stratify=y if classification)` |
| 9 | `_get_candidate_models()` — filtered by `include_models` |
| 10 | Sequential training loop over all candidate models via `_train_single_model()` |
| 11 | Ranking: best = lowest RMSE (regression) or highest F1 (classification) |
| 12 | SHAP summary plot generation for the best model (gracefully skipped on failure) |
| 13 | Constructs and returns the final summary dict |

**Return value of `run()`:**
```python
{
  'task_type': str,
  'total_time': float,
  'classes': list | None,          # LabelEncoder classes for classification
  'leaderboard': [{'model', 'name', 'metrics', 'status', 'error'}, ...],
  'best_model': {
    'name': str,
    'metrics': dict,
    'diagnostics': dict,           # y_true, y_pred, y_proba, residuals etc.
    'classes': list | None,
    'feature_names': list,
    'tailored_diagnostics': dict,  # model-specific (importances, SVs, elbow etc.)
    'explainability': {
      'parameters': dict,          # from get_parameter_descriptions()
      'has_shap': True
    },
    'pipeline_log': list           # ordered log of pipeline steps for UI
  }
}
```

---

## 4. Modeling Sub-package — `core_recommender/modeling/`

### 4.1 `registry.py`

A module-level dictionary `_REGISTRY` with two keys: `'classification'` and `'regression'`, each holding a list of `BaseModel` subclasses.

**`@register_model(task)`** — a class decorator that appends the decorated class to the appropriate list(s). `task='both'` appends to both. Raises `RegistryError` for invalid task strings. The decorator is **non-destructive** — it returns the class unchanged.

**`get_registered_models(task)`** — returns a copy of the list for the given task.

**`clear_registry()`** — empties both lists. Documented as test-only.

Adding a new model to the system requires **zero changes** to `registry.py` or `execution.py`. Only the new model file needs to exist and be imported. The imports at the top of `execution.py` trigger the decorators:
```python
import core_recommender.modeling.knn                # triggers @register_model
import core_recommender.modeling.linear_regression
import core_recommender.modeling.logistic_regression
import core_recommender.modeling.random_forest
import core_recommender.modeling.decision_trees
import core_recommender.modeling.svms
import core_recommender.modeling.naive_bayes
```

### 4.2 `base_model.py`

**Abstract Base Class:** `BaseModel(ABC)`

Enforces a strict contract on all models. Documents SOLID principles in its docstring.

**Abstract methods (must be implemented):**
- `preprocess(X, y)` → `(X_transformed, y_transformed, fitted_preprocessor)`
- `fit(X_train, y_train)` → `None`
- `calculate_metrics(X_test, y_test)` → `Dict[str, float]`
- `get_diagnostic_data(X_test, y_test)` → `Dict[str, Any]`

**Optional override (ISP):**
- `get_tailored_diagnostics()` → `Dict[str, Any]` — returns `{}` by default. Concrete models override to expose algorithm-specific data.

**Concrete method:**
- `export(filepath)` — serializes `self.model` using `joblib.dump()`. Raises `ModelNotFittedError` if `self.model is None`. Raises `ModelExportError` (wrapping `OSError`) on write failure.

**Key attributes:**
- `self.name` — human-readable string
- `self.config` — hyperparameter dict
- `self.model` — the unfitted sklearn estimator
- `self.best_estimator` — the fitted estimator after tuning (set by each concrete model's `fit()`)
- `self.metrics` — computed metric dict

### 4.3 Concrete Models

All files in `modeling/` follow the same pattern: subclass `BaseModel`, decorate with `@register_model`, implement the four abstract methods, choose a tuning strategy from `tuning.py`, and optionally override `get_tailored_diagnostics()`.

| File | Class | Task | Tuner |
|---|---|---|---|
| `knn.py` | `KNNModel` | both | `HalvingGridSearchCV` |
| `linear_regression.py` | `LinearRegressionModel` | regression | `GridSearchCV` (ElasticNet) |
| `logistic_regression.py` | `LogisticRegressionModel` | classification | `GridSearchCV` |
| `random_forest.py` | `RandomForestModel` | both | Optuna (TPE) |
| `decision_trees.py` | `DecisionTreeModel` | both | `RandomizedSearchCV` |
| `svms.py` | `SVMModel` | both | Optuna (TPE) |
| `naive_bayes.py` | `NaiveBayesModel` | classification | `GridSearchCV` |
| `pca.py` | — | — | Dimensionality reduction wrapper |

In the actual run, the selected models were:
```
['KNN (Regression)', 'SVM (Regression)', 'Linear Regression (ElasticNet)', 'Random Forest (Regression)']
```

---

## 5. Interface Layer — `interface/`

**`app.py`** — Flask application. 348 lines. Single global `LAST_RESULTS` dict stores the latest pipeline output.

**Windows-specific fix at line 6–8:**
```python
JOBLIB_TEMP = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tmp', 'joblib'))
os.makedirs(JOBLIB_TEMP, exist_ok=True)
os.environ['JOBLIB_TEMP_FOLDER'] = JOBLIB_TEMP
```
This sets joblib's temp directory to a stable local path to prevent `FileNotFoundError` during multiprocessing on Windows.

**Routes:**

| Route | Method | What it does |
|---|---|---|
| `/` | GET | Renders `index.html` (upload form) |
| `/process` | POST | Validates file + target column, saves CSV, generates heatmap and histogram plots, runs `ModelExecutor.run()` synchronously, generates diagnostic plots, stores results in `LAST_RESULTS`, redirects to `/dashboard` |
| `/dashboard` | GET | Reads `LAST_RESULTS`, renders `dashboard.html` with leaderboard, best model details, plots |
| `/download_model` | GET | Serves the best model `.pkl` artifact |

> **Known architectural limitation:** The pipeline runs **synchronously on the request thread**. Flask blocks until all models finish training. For large datasets or many models, this will cause the browser to hang. There is no async/background task system in V1.

**`/process` detailed flow:**
1. Validates: file exists, filename not empty, `.csv` extension, target column specified.
2. Parses `models` from the form (handles comma-separated values from multi-select).
3. Saves the CSV to `static/uploads/dataset.csv`.
4. Cleans old `.png` files from `static/images/`.
5. Generates `heatmap.png` and `histograms.png` (first 6 numeric features, excluding target).
6. Instantiates `ModelExecutor(random_state=42)` and calls `run()`.
7. Based on `task_type`, generates: ROC curve (binary classification only), confusion matrix, feature importance or coefficient bar chart, predicted vs. actual or residual plot.
8. The SHAP plot is generated inside `execution.run()` itself (not in `app.py`), saved to `static/images/shap_summary.png`.
9. Handles `DataValidationError` and `GuideMLError` with `flash()` messages and redirect.

---

## 6. Logging System — `logs/`

### `logs/guideml.log`
All `INFO`, `WARNING`, `ERROR`, and `DEBUG` messages from every module. Appended to on every run. The actual last run recorded:

```
2026-06-30 20:32:34 | INFO | core_recommender.execution | STARTING AutoML Pipeline - ModelExecutor.run()
2026-06-30 20:32:34 | INFO | core_recommender.execution | Dataset: 2800 rows × 16 columns
2026-06-30 20:32:34 | INFO | core_recommender.execution | Target: 'potential_rating'
2026-06-30 20:32:34 | INFO | core_recommender.execution | Target has 34 unique values. Inferring REGRESSION.
2026-06-30 20:32:34 | INFO | core_recommender.execution | Task inferred as: REGRESSION
2026-06-30 20:32:34 | INFO | core_recommender.execution | Checking for outliers in 'potential_rating'...
2026-06-30 20:32:34 | INFO | core_recommender.profiling  | Intelligent Profiler: Analyzing dataset characteristics...
2026-06-30 20:32:34 | INFO | core_recommender.profiling  | Profile Summary: {n_samples:2800, n_features:15, ratio:186.67, ...}
2026-06-30 20:32:34 | INFO | core_recommender.profiling  | Suggestions: ['Weak linear signal detected.']
2026-06-30 20:32:34 | INFO | core_recommender.execution  | Automated Leakage Detection Check...
2026-06-30 20:32:34 | INFO | core_recommender.execution  | No obvious leakage detected.
2026-06-30 20:32:34 | INFO | core_recommender.execution  | Train: 2240 | Test: 560
2026-06-30 20:32:34 | INFO | core_recommender.execution  | Training 4 selected models: [KNN, SVM, ElasticNet, Random Forest]
2026-06-30 20:32:38 | INFO | core_recommender.modeling.knn | [KNN (Regression)] Starting training...
```

The log stops at KNN training starting (line 16 of the file) — the run was still in progress or not yet completed at the time of this documentation.

### `logs/error.log`
**Empty.** This confirms: no warnings, no errors, no exceptions were raised in the last run. The pipeline ran cleanly through all steps up to KNN training.

---

## 7. Actual Pipeline Execution Trace

This is what happened in the last run, step by step, backed by the actual log:

```
Input:  2800 rows × 16 columns, target = 'potential_rating'

Step 1: Validation           — target column found ✓
Step 2: Feature/Target split — X: 15 features, y: 'potential_rating'
Step 3: Task inference       — integer, 34 unique values → REGRESSION
Step 4: Outlier check        — IQR method on 'potential_rating' (result not logged = no removals)
Step 5: Profiling            — ratio=186.67, sparsity=2.7%, not Gaussian,
                               max_linear_score=0.047 → "Weak linear signal detected."
Step 6: Leakage check        — correlation, identity, and categorical checks → clean
Step 7: Target encoding      — regression: pd.to_numeric() on y
Step 8: Train/test split     — 80/20, no stratify (regression), seed=42
                               Train: 2240 | Test: 560
Step 9: Model selection      — 4 models selected by UI: KNN, SVM, ElasticNet, Random Forest
Step 10: Training            — KNN started at 20:32:38 (4 seconds after pipeline start)
```

---

## 8. Design Principles Applied

| Principle | Where applied | How |
|---|---|---|
| **SRP** | `logger.py`, `exceptions.py`, `evaluation.py`, `preprocessing.py` | Each module has one job |
| **OCP** | `registry.py` + `base_model.py` | New models added without touching `ModelExecutor` |
| **LSP** | All concrete models | All implement `preprocess/fit/calculate_metrics/get_diagnostic_data` contracts |
| **ISP** | `get_tailored_diagnostics()` in `base_model.py` | Not abstract — models only implement what they expose |
| **DIP** | `ModelExecutor` depends on `BaseModel`, not concrete classes | Registry returns `BaseModel` subtypes; executor uses `BaseModel` interface |

---

## 9. Honest Limitations

These are real limitations observed in the actual source code — not invented:

- **Synchronous Flask routing:** `ModelExecutor.run()` blocks the HTTP request thread. The browser hangs until all models finish. There is no `async`, no `threading`, no Celery worker queue.
- **Sequential training loop:** Despite `n_jobs=-1` on the executor constructor, the outer training loop in `run()` is a standard Python `for` loop. Parallelism is delegated inside individual models (e.g., `GridSearchCV(n_jobs=-1)`), not across models.
- **Single-session, no persistence:** Results are stored in a module-level global `LAST_RESULTS`. If the Flask server restarts or a second user connects, the previous results are lost or overwritten.
- **No advanced feature engineering:** `preprocessing.py` has the tools (PCA, RFE, variance threshold), but these are available as factory functions only — they are not wired into an automated feature selection step within `run()`.
- **Error.log is empty now, but not always:** The error file is always appended to. On a future run with bad data, warnings and tracebacks will accumulate there.
- **`app.py` has debug print statements:** Line 72 has `print(f"DEBUG: Raw 'models' from form: {selected_models}")` — a development artifact not yet cleaned up.
