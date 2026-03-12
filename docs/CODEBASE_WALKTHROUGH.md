# 🏗️ The Senior Developer's Walkthrough: Understanding the Codebase

**"Welcome to the team. Let's look under the hood."**

When you join a new company, no one expects you to memorize every line of code on Day 1. Instead, you need to understand the **Architecture** (the blueprint) and the **Data Flow** (how information moves).

This guide is your map to the `GuideML` codebase.

---

## 1. The High-Level Architecture
Our system follows a classic **Client-Server** model, but with a twist for heavy computation.

1.  **The Client (Frontend)**: The browser/HTML. It's dumb. It just displays what it's told.
2.  **The Server (Backend)**: The Flask app (`app.py`). It's the traffic cop. It handles requests but delegates heavy work.
3.  **The Core Engine (`core_recommender/`)**: The heavy lifter. This is where the ML magic happens. It's completely separate from the web app.

**Key Design Decision:** We use **Asynchronous Processing**.
*   *Problem:* ML training takes time. If we run it in the main web request, the browser will timeout.
*   *Solution:* We spawn a **Background Thread** to run the ML pipeline. The web server returns "Message Received!" immediately, and the client asks for updates every few seconds.

---

## 2. Directory Structure: "Where does stuff live?"

*   **`interface/`**: The face of the operation.
    *   `app.py`: The Flask web server.
    *   `templates/` & `static/`: HTML, CSS, JS.
*   **`core_recommender/`**: The brain.
    *   `execution.py`: The **Manager**. It orchestrates the whole flow (Load -> Clean -> Train -> Rank).
    *   `modeling/`: The **Workers**. Each file (`knn.py`, `randomForest.py`) defines one specific algorithm.
    *   `preprocessing.py`: The **Janitor**. Cleans data before the workers see it.
    *   `dataHandling.py`: The **Translator**. Handles feature encoding (One-Hot, Ordinal) to turn text into numbers.
    *   `tuning.py`: The **Optimizer**. Uses Grid Search or Optuna to find the perfect settings for each model.
    *   `evaluation.py`: The **Judge**. Calculates accuracy, RMSE, and F1 scores to see who really won.
    *   `visualization.py`: The **Artist**. Turns cold numbers into beautiful charts and heatmaps.
    *   `knowledge_base.py`: The **Storyteller**. Holds the simple definitions and "stories" for every model.
    *   `logger.py`: The **Record Keeper**. Tracks every step and sends progress updates back to the UI.
*   **`data/`**: Storage for uploaded CSVs.
*   **`logs/`**: The black box recorder. If something crashes, look here.

---

## 3. Tracing the Data Flow: "Follow the CSV"

Let's trace exactly what happens when a user uploads `data.csv`:

### Step 1: The Upload (Interface)
*   **User** clicks "Upload" on `index.html`.
*   **JS** captures the file and sends it to `/process` in `app.py`.
*   **`app.py`** saves the file to `uploads/` and generates a unique `job_id` (e.g., `abc-123`).
*   **`app.py`** starts a background thread calling `background_training()`.
*   **Response:** "Job Started! ID: abc-123".

### Step 2: The Handover & Prep (Execution & dataHandling)
Now the background thread takes over.
*   It calls `ModelExecutor.run()` in `core_recommender/execution.py`.
*   **Executor** reads the CSV.
*   **Executor** calls `preprocessing.py` to fix missing values.
*   **Executor** uses `dataHandling.py` to encode categorical columns so the models can read them.

### Step 3: The Race & Optimization (Training & Tuning)
*   The Executor creates instances of models (KNN, Random Forest, etc.) from `modeling/`.
*   It calls `tuning.py` to run hyperparameter optimization (like Optuna). We don't just train; we hunt for the *best version* of each model.
*   It uses `joblib` to train them **in parallel** (multiprocessing).

### Step 4: The Decision (Evaluation)
*   Each model makes predictions on a test set.
*   The Executor calls `evaluation.py` to calculate precision, recall, and error rates.
*   The Executor compares these scores and picks the definitive winner.
*   It saves the "Best Model" as a `.pkl` file.
*   The Executor extracts the **"Model DNA"** (tuned parameters) from the winning estimator.

### Step 5: The Results (Visualization & Frontend)
*   The Executor calls `visualization.py` to generate ROC curves, confusion matrices, and **SHAP Global Influence** plots.
*   The JS on the frontend has been asking "Are you done?" every 2 seconds.
+   `app.py` finally says "Yes!", providing links to the generated visualizations and model metrics.
*   The browser redirects to `/dashboard`, which reads the results, the "stories" from `knowledge_base.py`, and renders the charts.

---

## 4. The Support System: "Behind the Scenes"

We have two silent partners working throughout this entire process:

### 1. The Logger (`logger.py`)
This isn't just for errors. Our logger has a special `ProgressLogHandler`. When the ML script is 40% done with training, it tells the Logger, which then tells the Flask app, which then tells the User's browser. It's how we keep the progress bar moving.

### 2. The Knowledge Base (`knowledge_base.py`)
Users aren't always data scientists. When the dashboard says "Random Forest," the frontend pulls a "Story" from `knowledge_base.py` to explain it (e.g., *"Imagine a council of wise experts..."*). It bridges the gap between math and understanding.

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
