# GuideML (Building)
We are in the phase of working and building with AI applications. We use the AI & ML models unknowingly for our daily purpose; it’s become an integral part of us.

To bridge the gap between complex ML knowledge and practical use, we are implementing a Model Recommender application that simplifies the initial model selection process for users with tabular data. This application serves as a local, high-performance platform designed to execute an automated, standardized Machine Learning pipeline. 

The goal is to provide users with easy access to evaluate basic ML models based on their datasets, allowing them to get their best models from us without needing specialized coding knowledge. This project emphasizes architectural rigor (modular, OOP design) and performance optimization (concurrency) to deliver fast, validated results and a clear diagnostic report.

## Purpose
This project is a high-performance, single-session local application that automates the initial phases of the Machine Learning lifecycle. It is engineered with a focus on speed, architecture, and advanced diagnostics, serving as a powerful tool for quickly identifying the best-performing traditional ML model for a given dataset.

The core strength of the application lies in its **Modular, Object-Oriented design** and its use of **multiprocessing/multithreading** to run training pipelines concurrently, drastically reducing model selection time.

## Technical Highlights (What You'll Find Inside)

This project demonstrates proficiency in advanced software and ML engineering concepts:

* **Concurrency & Performance:** Uses **Joblib/Multiprocessing** to execute training of 10 models simultaneously and **Multithreading** to optimize data I/O.
* **Architecture (OOP):** Implements a core package (`core_recommender/`) that is decoupled from the UI. All models inherit from a **`BaseModel` abstract class** with customizable `preprocess()` and `evaluate()` methods.
* **Optimization:** Includes logic for **Limited Hyperparameter Optimization** (Grid Search) applied only to the top 3 best-performing models.
* **Diagnostics:** Generates crucial diagnostic visualizations for analysis.

---

## 🎯 Project Scope: What It Does (Product Functions)

The application provides a complete, automated pipeline:

1. **Ingestion & Preprocessing:** Reads a CSV file, handles basic imputation/encoding, and performs a stratified train/test split.
2. **Concurrent Training:** Trains a fixed suite of traditional ML models concurrently using **multiprocessing** and **k-fold cross-validation**.
3. **Model Selection & Tuning:** Ranks all models (F8) and applies **Limited Hyperparameter Optimization** to the top 3 models.
4. **Output & Export:** Displays a final ranked leaderboard and a comprehensive visualization suite. Facilitates the download of the best model artifact.

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
    git clone [Your Repository URL]
    cd model_recommender_app
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
