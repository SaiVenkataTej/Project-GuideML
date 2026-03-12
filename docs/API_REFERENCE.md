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
*   **`calculate_classification_metrics(y_true, y_pred)`**:
    *   Returns: Dictionary {Accuracy, Precision, Recall, F1}.
*   **`calculate_regression_metrics(y_true, y_pred)`**:
    *   Returns: Dictionary {RMSE, MAE, R2}.

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
*   **`tune_hyperparameters(model, X_train, y_train)`**:
    *   Uses `Optuna` (Bayesian Optimization) to find the best settings for the given model.
    *   Returns the *tuned* model instance.

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
*   **`@app.route('/')`**: Renders the homepage (`index.html`).
*   **`@app.route('/process', methods=['POST'])`**: Accepts file uploads and starts the background job.
*   **`background_training(job_id)`**: The threaded worker function that calls `ModelExecutor`.

