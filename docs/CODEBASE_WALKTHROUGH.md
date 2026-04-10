# 🏗️ The Senior Developer's Walkthrough: Understanding the Codebase

**"Welcome to the team. Let's look under the hood."**

When you join a new company, no one expects you to memorize every line of code on Day 1. Instead, you need to understand the **Architecture** (the blueprint) and the **Data Flow** (how information moves).

This guide is your map to the `GuideML` codebase.

---

## 1. The High-Level Architecture
Our system follows a straightforward, synchronous Web App model.

1.  **The Client (Frontend)**: The browser/HTML. It utilizes standard JS and Bootstrap to render the interface and masks loading times with UI overlays.
2.  **The Server (Backend)**: The Flask app (`app.py`). It receives POST requests and routes them directly to the recommender engine.
3.  **The Core Engine (`core_recommender/`)**: The heavy lifter. This is where the ML modeling and optimization algorithms reside.

**Key Design Reality:** We use **Synchronous Processing**.
*   *How it works:* The user uploads a CSV, and the Flask application entirely blocks the thread to execute the ML pipeline from start to finish. Once the pipeline completes and global variables are populated, it triggers an HTTP redirect back to the client. This means long analyses will stall the browser request until complete.

---

## 2. Directory Structure: "Where does stuff live?"

*   **`interface/`**: The face of the operation.
    *   `app.py`: The Flask web server. *Note: Data routing currently relies on a `global` variable (`LAST_RESULTS`), meaning concurrent users will overwrite each other.*
    *   `templates/` & `static/`: HTML, CSS, JS. Hardcoded dataset uploads land in `static/uploads/`.
*   **`core_recommender/`**: The brain.
    *   `execution.py`: The **Manager**. It orchestrates the whole flow synchronously (Load -> Clean -> Train -> Rank) acting as a single large God-object.
    *   `modeling/`: The **Workers**. Each file (`knn.py`, `randomForest.py`) inherits from `BaseModel`.
    *   `preprocessing.py`: The **Transformers**. Sklearn wrappers for scaling, imputing, and encoding.
    *   `dataHandling.py`: Additional data utilities.
    *   `tuning.py`: The **Optimizer**. Uses Optuna to find the best settings via cross-validation.
    *   `evaluation.py`: Calculates accuracy, RMSE, and F1 scores.
    *   `visualization.py`: Generates the `.png` charts (ROC, Heatmaps).
    *   `knowledge_base.py`: Holds descriptions and UI explanations for model types.
    *   `logger.py`: Writes `.log` files to disk. *(Note: The UI progress handler functionality exists but is currently disconnected from the Flask app).*
*   **`logs/`**: Raw text logs tracking execution and errors.
*   **`tests/`**: Unit testing folder (currently empty).

---

## 3. Tracing the Data Flow: "Follow the CSV"

Let's trace exactly what happens when a user uploads `data.csv`:

### Step 1: The Upload (Interface)
*   **User** clicks "Initialize Engine" on `index.html`.
*   **JS** pushes a loading spinner and sends a traditional `POST` to `/process` in `app.py`.
*   **`app.py`** blindly overwrites whatever is currently in `uploads/dataset.csv`.
*   **`app.py`** hands the data directly to `ModelExecutor.run()`, freezing the web request.

### Step 2: The Handover & Prep (Execution)
*   **Executor** reads the CSV.
*   **Executor** heavily cleans data inline: stripping outliers, inferring task types, detecting target leakage based on absolute correlation, and encoding targets with `LabelEncoder`.

### Step 3: The Race & Optimization (Training & Tuning)
*   The Executor creates instances of models (KNN, Random Forest, etc.) from `modeling/`.
*   It utilizes a **sequential Python `for` loop** to train each model one-by-one.
*   Inside the sequence, it wraps the model with preprocessing steps and calls `tuning.py` to run Optuna Bayesian parameter optimization.

### Step 4: The Decision (Evaluation)
*   Each model makes predictions on a test set.
*   The Executor calculates metrics and ranks models by F1 Score (Classification) or RMSE (Regression).
*   It saves the "Best Model" estimator as `best_model.pkl`.

### Step 5: The Results (Visualization & Frontend)
*   The Executor calls `visualization.py` to save `roc_curve.png`, `shap_summary.png`, etc. as local files.
*   The Executor bundles all numerical results and pushes them onto the global `LAST_RESULTS` dictionary.
*   `app.py` releases the frozen POST request, issuing a redirect (`302`) to `/dashboard`.
*   `/dashboard` reads the global dict, maps metrics to descriptions in `knowledge_base.py`, and renders the HTML page.

---

## 4. The Support System: "Behind the Scenes"

### 1. The Logger (`logger.py`)
Our `logger.py` handles writing errors and execution tracks to `logs/guideml.log`. While UI progress handler logic was drafted for this module, the web application currently ignores it across the synchronous Flask pipeline. 

### 2. The UI Pipeline Log (`execution.py`)
To mimic progress logs, the `execution.py` class manually appends text dicts like `{"step": "Splitting", "details": "..."}` to its internal list. This exact list simply gets printed on screen at the very end when the dashboard loads.

---

## 5. Why We Wrote It This Way

### "Why separate `core_recommender` from `interface`?"
**Modularity.**
Make a change to the ML logic? You don't break the website.
Want to swap the website for a mobile app? You don't need to rewrite the ML logic.

### "Why use Classes for Models?"
**Standardization.**
Every model (`KNN`, `SVM`, etc.) inherits from `BaseModel`. They all *must* have a `fit()` and `predict()` method. This means the Executor doesn't need to know *which* model it's running. It just treats them all the same (Polymorphism).

---

**Next Steps:**
Now that you have the map, check out `API_REFERENCE.md` for the technical details of every function.
