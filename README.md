# GuideML (Building)
We are in the phase of working and building with AI applications. We use the AI & ML models unknowingly for our daily purpose; it’s become an integral part of us.

To bridge the gap between complex ML knowledge and practical use, we are implementing a Model Recommender application that simplifies the initial model selection process for users with tabular data. This application serves as a local, high-performance platform designed to execute an automated, standardized Machine Learning pipeline. 

The goal is to provide users with easy access to evaluate basic ML models based on their datasets, allowing them to get their best models from us without needing specialized coding knowledge. This project emphasizes architectural rigor (modular, OOP design) and performance optimization (concurrency) to deliver fast, validated results and a clear diagnostic report.

## Purpose
This project is a high-performance, single-session local application that automates the initial phases of the Machine Learning lifecycle. It is engineered with a focus on speed, architecture, and advanced diagnostics, serving as a powerful tool for quickly identifying the best-performing traditional ML model for a given dataset.

The core strength of the application lies in its **Modular, Object-Oriented design** and its use of **multiprocessing/multithreading** to run training pipelines concurrently, drastically reducing model selection time.

## Technical Highlights (What You'll Find Inside)

This project demonstrates proficiency in standard software and ML engineering concepts:

* **Architecture (OOP):** Implements a core package (`core_recommender/`) decoupled from the UI. All models inherit from a **`BaseModel` abstract class** with customizable `preprocess()` and `evaluate()` methods.
* **Optimization:** Includes logic for **Bayesian Hyperparameter Optimization** (Optuna) with cross-validation.
* **Intelligent Profiling:** Basic heuristic profiling to generate warnings for dataset dimensionality, sparsity, and linearity.
* **Diagnostics:** Generates crucial diagnostic visualizations (SHAP, ROC, Residuals) using non-interactive Matplotlib backends.

*Note: The current architecture utilizes sequential processing for model training and relies on synchronous Flask routing, meaning long training times will block the active browser session.*

## Directory Structure
* `core_recommender/`: Main package source code.
* `interface/`: Web application interface.
* `scripts/`: Utility scripts and demos.
* `docs/`: Project documentation.
    * `USER_GUIDE.md`: For new users.
    * `ML_THEORY.md`: For understanding the ML concepts.
    * `FRONTEND_GUIDE.md`: For understanding the Web UI.
    * `CODEBASE_WALKTHROUGH.md`: For developers.
    * `WORKFLOWS.md`: For architects.
    * `API_REFERENCE.md`: For API details.
    * `TEAM_ROLES.md`: Team structure overview.
* `logs/`: System logs.
* `data/`: Datasets.
* `tests/`: Unit tests (Currently Empty - high priority for future implementation).

## 📚 Documentation Index

We have created detailed guides for every type of user:

*   **[User Guide](docs/USER_GUIDE.md)**: 🚀 Start here! How to install and run the app.
*   **[ML Theory](docs/ML_THEORY.md)**: 🧠 Explanation of the machine learning models used.
*   **[Frontend Guide](docs/FRONTEND_GUIDE.md)**: 🎨 How the HTML, CSS, and JS work together.
*   **[Codebase Walkthrough](docs/CODEBASE_WALKTHROUGH.md)**: 🏗️ A map of the actual project architecture and data flow.
*   **[Workflows](docs/WORKFLOWS.md)**: ⚙️ Deep dive into logic, pipelines, and edge cases.
*   **[API Reference](docs/API_REFERENCE.md)**: 📚 Technical specifications.

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
* **Advanced Feature Engineering (S2):** Preprocessing is limited to basic scaling and encoding; no custom feature creation or advanced outlier removal.

---

## Setup and Execution

### Prerequisites

* Python 3.10+
* The following libraries (as defined in `requirements.txt`): 
* `scikit-learn` 
* `pandas` 
* `numpy`
* `joblib`
* `Flask/Bolt`

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
