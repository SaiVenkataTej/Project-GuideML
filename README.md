# GuideML (Building)
We are in the phase of working and building with AI applications. We use the AI & ML models unknowingly for our daily purpose; it’s become an integral part of us.

To bridge the gap between complex ML knowledge and practical use, we are implementing a Model Recommender application that simplifies the initial model selection process for users with tabular data. This application serves as a local platform designed to execute an automated, standardized Machine Learning pipeline. 

The goal is to provide users with easy access to evaluate basic ML models based on their datasets, allowing them to get their best models from us without needing specialized coding knowledge. This project emphasizes architectural rigor (modular, OOP design) and intelligent optimization (Bayesian hyperparameter search) to deliver validated results and a clear diagnostic report.

## Purpose
This project is a single-session, local application that automates the initial phases of the Machine Learning lifecycle. It is engineered with a focus on **software architecture** and **diagnostic depth**, serving as a practical tool for quickly identifying the best-performing traditional ML model for a given tabular dataset.

The core strength of the application lies in its **Modular, Object-Oriented design** and its use of **Bayesian Hyperparameter Optimization (Optuna)** to efficiently tune models — replacing the brute-force approach of GridSearchCV with intelligent TPE-sampled search that finds optimal parameters in fewer evaluations.

> **Current Limitation:** Model training is **sequential** (one model at a time) and Flask routing is **synchronous**. This means long training runs will block the browser session until completion. Parallel execution (`n_jobs`) is planned for a future version.

## Technical Highlights (What You'll Find Inside)

This project demonstrates proficiency in standard software and ML engineering concepts:

* **Architecture (OOP):** Implements a core package (`core_recommender/`) decoupled from the UI. All models inherit from a **`BaseModel` abstract class** that enforces a standard contract via four abstract methods: `preprocess()`, `fit()`, `calculate_metrics()`, and `get_diagnostic_data()`.
* **Optimization:** Includes logic for **Bayesian Hyperparameter Optimization** (Optuna, TPE sampler) with cross-validation, applied automatically to the best-ranked model.
* **Intelligent Profiling:** Heuristic profiling generates warnings for dataset dimensionality, sparsity, and linearity, plus automated leakage detection (correlation, mathematical identity, and categorical 1:1 mapping checks).
* **Diagnostics:** Generates diagnostic visualizations (SHAP summary, ROC curve, Confusion Matrix, Residual plots, Feature Importance) using non-interactive Matplotlib backends.

## Directory Structure
* `core_recommender/`: Main package source code.
* `interface/`: Web application interface.
* `scripts/`: Utility scripts and demos.
* `docs/`: Project documentation.
    * `docs.md`: Consolidated canonical technical documentation (User Guide, ML Theory, Frontend Guide, Codebase Walkthrough, Workflows, Team Roles, and Update Log).
    * `Techniques.md`: Full breakdown of techniques used per model.
    * `API_REFERENCE.md`: API specifications for developers.
* `logs/`: System logs.
* `data/`: Datasets.
* `tests/`: Unit tests (Implemented: checks model pipelines, profiling, visualization, and execution).

## 📚 Documentation Index

We have created detailed guides for every type of user:

*   **[Canonical Technical Documentation](docs/docs.md)**: 🚀 Start here! Includes User Guide, ML Theory, Frontend architecture, Codebase walkthrough, Workflows, Team roles, and Update log.
*   **[Techniques Guide](docs/Techniques.md)**: 🔬 Full breakdown of every preprocessing, tuning, and evaluation technique used per model.
*   **[API Reference](docs/API_REFERENCE.md)**: 📚 Technical specifications for every function and route.

---

## 🎯 Project Scope: What It Actually Does

The application provides an automated, sequential pipeline:

1. **Ingestion & Preprocessing:** Reads a CSV file (blocking request), infers regression or classification task type, handles basic imputation/encoding, and performs a standard train/test split.
2. **Sequential Model Training:** Trains a suite of traditional ML models iteratively using a standard loop and cross-validation score ranking.
3. **Model Selection & Tuning:** Ranks all models and applies **Bayesian Hyperparameter Optimization** (Optuna) automatically.
4. **Output & Export:** Displays a final ranked leaderboard and a visualization suite upon request completion. Facilitates the download of the best model artifact.

---

## Project Boundaries (Exclusions)

To maintain a manageable scope and ensure timely completion, the following features are **explicitly excluded** from Version 1.0 (V1):

* **Deep Learning Models (S1):** No integration of TensorFlow, PyTorch, or Neural Networks.
* **Cloud Deployment/Live Hosting (S7):** The application is strictly a **local application** designed to run on a user's machine.
* **Multi-User Management (S4):** No user accounts, registration, or authentication.
* **Advanced Feature Engineering (S2):** Preprocessing is limited to standard scaling and one-hot encoding. Outlier removal applies only a basic IQR filter on the target variable (regression only); no feature-level outlier handling or custom feature creation.

---

## Setup and Execution

### Prerequisites

* Python 3.10+
* The following libraries (as defined in `requirements.txt`):
  * `scikit-learn`
  * `pandas`
  * `numpy`
  * `joblib >= 1.3.0`
  * `flask`
  * `optuna`
  * `shap`
  * `matplotlib`
  * `seaborn`
  * `scipy`

### Installation (Using Conda/venv)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SaiVenkataTej/Project-GuideML.git
    cd Project-GuideML
    ```
2.  **Create and activate the environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate 
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### How to Run the Application

1.  Ensure your environment is active.
2.  Run the main interface file 
    ```bash
    python interface/app.py 
    ```
3.  Open your web browser and navigate to the local host address displayed (e.g., `http://127.0.0.1:5000`).

---

## 📸 Final Deliverables

* **Source Code** (This repository)
* **System Architecture Diagram** (In the main documentation)
* **Best Model Artifact**
