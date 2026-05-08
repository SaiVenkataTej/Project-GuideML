# 📚 API Reference

**"The encyclopedia of our codebase."**

This document provides detailed technical specifications for every Python file in the backend. Use this when you need to know exactly what a function takes as arguments or what a class method returns.

---

## 1. Core Package (`core_recommender/`)

### `execution.py`
**Class: `ModelExecutor`**
*   **Description**: The orchestrator that manages the end-to-end model selection pipeline.
*   **`__init__(self, task_type='auto', n_jobs=-1)`**:
    *   Initializes concurrency settings.
    *   `task_type`: 'classification', 'regression', or 'auto' (inferred).
*   **`run(self, df, target_column)` -> Dict**:
    *   **Main Entry Point**.
    *   Takes a DataFrame and target name.
    *   Returns a dictionary with `leaderboard`, `best_model`, and `diagnostics`.

### `preprocessing.py`
**Module: Data Cleaning & Transformation**
*   **`get_imputer(strategy='median')`**: Returns a configured `SimpleImputer`.
*   **`get_one_hot_encoder(handle_unknown='ignore')`**: Returns a robust OHE for categorical data.
*   **`get_standard_scaler()`**: Returns a Z-score normalizer.

### `dataHandling.py`
**Module: Encoding Utilities**
*   **`apply_one_hot_encoder(data, drop_first=False)`**: Applies OHE to a DataFrame and returns a *new* DataFrame with proper column names.
*   **`apply_ordinal_encoder(data, categories='auto')`**: Encodes ordinal features (Low < Med < High).

### `evaluation.py`
**Module: Metric Calculation**
*   **`calculate_rmse(y_true, y_pred)`** → `float`
*   **`calculate_mae(y_true, y_pred)`** → `float`
*   **`calculate_r2_score(y_true, y_pred)`** → `float`
*   **`calculate_adjusted_r2(y_true, y_pred, n_samples, n_features)`** → `float`
*   **`calculate_accuracy(y_true, y_pred)`** → `float`
*   **`calculate_f1_score(y_true, y_pred, average='weighted')`** → `float`
*   **`calculate_roc_auc_score(y_true, y_proba)`** → `float`
*   **`calculate_precision(y_true, y_pred)`** → `float`
*   **`calculate_log_loss(y_true, y_proba)`** → `float`
*   **`measure_prediction_latency(model, X_test, n_runs=100)`** → `float`

### `knowledge_base.py`
**Module: Static Knowledge**
*   **`MODEL_KNOWLEDGE` (Dict)**: Contains descriptions, pros/cons, and ideal use cases for every supported model. Used to populate the UI.

### `logger.py`
**Module: System Logging**
*   **`get_logger(name)`**: Returns a configured Python `logging.Logger` instance that writes to `logs/guideml.log`.

### `profiling.py`
**Class: `DataProfiler`**
*   **`analyze(self, X, y)`**:
    *   Scans the dataset for issues (high cardinality, nulls, skew).
    *   Returns a "Suggestions" list to guide the user.

### `tuning.py`
**Module: Hyperparameter Optimization**
*   **`get_grid_search_tuner(estimator, param_grid, cv, scoring)`**: Returns a configured `GridSearchCV` object. Used for simpler models (Linear/Logistic Regression).
*   **`get_random_search_tuner(estimator, param_distributions, cv, scoring, n_iter=10)`**: Returns a configured `RandomizedSearchCV` object. Used for Decision Trees.
*   **`get_halving_grid_search_tuner(estimator, param_grid, cv, scoring, factor=3)`**: Returns a `HalvingGridSearchCV` object using successive halving. Used for KNN.
*   **`run_optuna_optimization(estimator_class, param_space_func, X, y, cv, scoring, n_trials=20)`**: Executes a full Optuna Bayesian optimization study and returns the best fitted model. Used for Random Forest and SVM.

### `visualization.py`
**Module: Plotting Utilities**
*   **`plot_correlation_heatmap(df)`**: Saves a `.png` of the correlation matrix.
*   **`plot_confusion_matrix(y_true, y_pred)`**: Saves a confusion matrix for classification tasks.
*   **`plot_roc_curve(y_true, y_proba)`**: Saves the ROC curve. Handles 2D probability arrays.
*   **`plot_shap_summary(model, X, model_name, save_path)`**: Generates global feature influence plots using SHAP values.

---

## 2. Models (`core_recommender/modeling/`)

### `baseModel.py`
**Class: `BaseModel` (Abstract Base Class)**
*   **All models inherit from this.**
*   **`fit(X, y)`**: Matches sklearn API.
*   **`predict(X)`**: Returns predictions.
*   **`preprocess(X, y)`**: Internal pipeline setup.
*   **`get_parameter_descriptions()`**: Returns a dictionary of tuned hyperparameters and their technical descriptions (Model DNA).

### Specific Implementations
Each of these files contains a class inheriting from `BaseModel`, implementing the specific algorithm logic:
*   **`knn.py`**: `KNNModel` (K-Nearest Neighbors).
*   **`linearRegression.py`**: `LinearRegressionModel`.
*   **`logisticRegression.py`**: `LogisticRegressionModel`.
*   **`decisionTrees.py`**: `DecisionTreeModel`.
*   **`randomForest.py`**: `RandomForestModel`.
*   **`svms.py`**: `SVMModel` (Support Vector Machine).
*   **`naiveBayes.py`**: `NaiveBayesModel` (GaussianNB).
*   **`PCA.py`**: Wrapper for dimensionality reduction steps.

---

## 3. Interface (`interface/`)

### `app.py`
**Module: Flask Web Server**
*   **`@app.route('/')`** — `GET`: Renders the homepage (`index.html`).
*   **`@app.route('/process', methods=['POST'])`**: Accepts file upload and target column, runs the full `ModelExecutor.run()` pipeline **synchronously** on the request thread, then redirects to `/dashboard`.
*   **`@app.route('/dashboard')`** — `GET`: Reads from the global `LAST_RESULTS` dict and renders `dashboard.html`.
*   **`@app.route('/download_model')`** — `GET`: Serves `best_model.pkl` as a file download.
