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
    *   Check size: Must be < 50MB (Configurable).
    *   **Edge Case:** User uploads an image named `data.csv`. *Action: Reject immediately.*
3.  **Validation Layer 2 (Content)**:
    *   Pandas `read_csv()` attempts to parse.
    *   **Edge Case:** CSV is malformed or empty. *Action: Catch `pd.errors.EmptyDataError`, return friendly error.*
4.  **Target Column Check**:
    *   Does the requested "Target" column exist?
    *   **Edge Case:** Typo in target name. *Action: Return 400 Bad Request with available columns.*

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

### The Parallel Execution (`core_recommender/execution.py`):
1.  **Task Inference**:
    *   Is the target a number (Regression) or a category (Classification)?
    *   **Edge Case:** Target is numeric (0, 1) but represents classes. *Action: Heuristic check (if unique values < 20 -> Classification).*
2.  **Concurrency**:
    *   We use `joblib.Parallel` with `n_jobs=-1` (Use all cores).
    *   **Edge Case:** One model crashes (e.g., SVM on non-scaled data).
    *   *Action:* Wrap each training loop in a `try/except` block. Log the error, fail that specific model, but **let the others finish.**
3.  **Scoring**:
    *   *Regression:* RMSE (Lower is better).
    *   *Classification:* F1-Score (Higher is better, handles imbalance).

---

## 4. Error Handling & Recovery matrix
| Scenario | Detection | System Action | User Feedback |
| :--- | :--- | :--- | :--- |
| **Server Overload** | 503 Service Unavailable | Queue the request | "Server busy, please wait..." |
| **Nan Values in Target** | Pre-check | Drop rows with NaN target | Warning in logs |
| **All Models Fail** | Results list empty | Raise `RuntimeError` | "Training failed. Check data quality." |
| **Windows IO Lock** | `joblib` Temp Error | Redirect to `tmp/joblib` | Handled internally (retry logic) |
| **SHAP Computation** | Library crash/timeout | Log warning & skip plot | "Skipped due to data complexity" |
| **Browser Closed** | Socket disconnect | Thread continues (orphan) | Job completes in background (results saved) |

---

## 5. Developer Workflow: Adding a New Model

Want to add "XGBoost"?
1.  Create `core_recommender/modeling/xgboost.py`.
2.  Inherit from `BaseModel`.
3.  Implement `fit()`, `predict()`, and `get_parameter_descriptions()`.
4.  Add to `execution.py` imports and register in `ModelExecutor`.
5.  **Done.** The Executor picks it up automatically.
