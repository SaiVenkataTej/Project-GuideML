# ⚙️ Technical Workflows & Edge Cases

**"The devil is in the details. Here is how we handle them."**

This document is for the architects and senior developers who need to know exactly *what happens when things go wrong*. We've built robust pipelines to handle the chaos of real-world data.

---

## 1. Data Ingestion Pipeline
**Goal:** Safely load user data without crashing the server.

### The Workflow:
1.  **Upload Request**: User POSTs a file to `/process`.
2.  **Validation Layer 1 (File Type)**:
    *   Check extension: Must be `.csv`.
    *   **Edge Case:** User uploads a non-CSV file. *Action: Flash error and redirect back to homepage.*
3.  **Validation Layer 2 (Content)**:
    *   Pandas `read_csv()` attempts to parse.
    *   **Edge Case:** CSV is malformed or empty. *Action: Caught by the outer `try/except` in `app.py`, flashes an error message to the user.*
4.  **Target Column Check**:
    *   Does the requested "Target" column exist in the DataFrame?
    *   **Edge Case:** Typo in target name. *Action: Return 400 JSON error with message.*

---

## 2. Automated Preprocessing Workflow
**Goal:** Turn messy raw data into clean model-ready matrices.

### The Pipeline (`core_recommender/preprocessing.py`):
1.  **Leakage Detection**:
    *   Before anything, we check correlation.
    *   **Edge Case:** A feature has 99% correlation with the target or zero variance after encoding.
    *   *Action:* Drop the feature to prevent cheating or singular matrices. Expanded to include categorical leakage via variance checks.
2.  **Imputation (Filling Holes)**:
    *   *Numeric Strategy:* Median (Robust to outliers).
    *   *Categorical Strategy:* Most Frequent (Mode).
    *   **Edge Case:** A column is 100% empty. *Action: Drop the column entirely.*
3.  **Encoding**:
    *   *Low Cardinality (<10 categories):* One-Hot Encoding.
    *   *High Cardinality (>10 categories):* Frequency Encoding (to avoid massive dimensionality).
    *   **Edge Case:** New category appears in Test set. *Action: Handle Unknown='ignore' (OHE).*

---

## 3. Training & Selection Strategy
**Goal:** Find the best model efficiently.

### The Sequential Training Loop (`core_recommender/execution.py`):
1.  **Task Inference**:
    *   Is the target a number (Regression) or a category (Classification)?
    *   **Edge Case:** Target is numeric (0, 1) but represents classes. *Action: Heuristic check (if unique values < 20 -> Classification).*
2.  **Sequential Execution**:
    *   Models are trained one-by-one using a standard Python `for` loop.
    *   **Edge Case:** One model crashes (e.g., SVM on non-scaled data).
    *   *Action:* Each model is wrapped in a `try/except` block in `_train_single_model()`. Log the error, record `status: 'failed'` for that model, but **let the others finish.**
3.  **Scoring**:
    *   *Regression:* RMSE (Lower is better).
    *   *Classification:* F1-Score (Higher is better, handles imbalance).

---

## 4. Error Handling & Recovery Matrix
| Scenario | Detection | System Action | User Feedback |
| :--- | :--- | :--- | :--- |
| **Server Overload / Large Dataset** | No guard currently implemented | Flask thread blocks until training completes; gateway may time out | Browser hangs; no user feedback until redirect |
| **NaN Values in Target** | `pd.to_numeric` + `fillna(0)` in `execution.py` | Coerced to numeric; NaNs filled with 0 | Warning in logs only |
| **All Models Fail** | Results list empty after training loop | `RuntimeError` raised, caught by outer `try/except` in `app.py` | Flash error and redirect to homepage |
| **Windows IO Lock** | `joblib` Temp Folder error | Redirected to `interface/tmp/joblib` via env variable set at startup | Handled internally and silently |
| **SHAP Computation Failure** | `try/except` in `execution.py` around SHAP block | Logs a warning and skips plot generation | Dashboard loads without the SHAP card |
| **Non-CSV File Upload** | `.endswith('.csv')` check in `app.py` | Flash error and redirect to homepage | "Only CSV files are allowed" |

---

## 5. Developer Workflow: Adding a New Model

Want to add "XGBoost"?
1.  Create `core_recommender/modeling/xgboost.py`.
2.  Inherit from `BaseModel`.
3.  Implement all abstract methods: `preprocess()`, `fit()`, `calculate_metrics()`, `get_diagnostic_data()`, and `get_feature_importance()`.
4.  Add to `execution.py` imports and register in `_get_candidate_models()`.
5.  **Done.** The Executor picks it up automatically.
