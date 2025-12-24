import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, KFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

# --- PROJECT IMPORTS ---
from core_recommender.modeling.baseModel import BaseModel
from core_recommender.preprocessing import (
    get_imputer,
    get_one_hot_encoder,
    get_standard_scaler,
    get_minmax_scaler,
    get_pca_reducer
)
from core_recommender.evaluation import (
    calculate_accuracy,
    calculate_f1_score,
    calculate_rmse,
    calculate_r2_score
)

# --- DEFAULT CONFIGURATION ---
CONFIG = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'scaler': 'standard',     # 'standard', 'minmax'
    'pca_components': 0.95,   # float for variance, int for components, None to disable
    'cv_folds': 5,
    'n_iter': 10,             # RandomizedSearch iterations
    'random_state': 42,
    'n_jobs': -1
}

# =========================================================================
# SVMModel Class
# =========================================================================

class SVMModel(BaseModel):
    """
    A concrete implementation of Support Vector Machines (SVM) for both Classification (SVC) and Regression (SVR).
    
    This model finds the optimal hyperplane that maximizes the margin between classes (SVC) 
    or fits the error within a threshold (SVR). It supports various kernels (Linear, RBF, Poly)
    to handle non-linear relationships and integrates Dimensionality Reduction (PCA) automatically.
    """
    def __init__(self, is_classification: bool = True, config: Dict[str, Any] = CONFIG):
        """
        Initializes the SVM model with task-specific configurations.

        Args:
            is_classification: True for classification tasks, False for regression.
            config: Dictionary containing hyperparameters (e.g., 'C', 'kernel', 'gamma', 'pca_components').
                    Defaults to the global CONFIG dictionary.
        """
        
        task_name = "Classification" if is_classification else "Regression"
        name = f"SVM ({task_name})"
        super().__init__(name=name, config=config)
        
        self.is_classification = is_classification
        
        # Initialize Model Instance & Param Grid
        if self.is_classification:
            # SVC with probability=True for diagnostics (optional, but good for ROC/LogLoss)
            # Class Weights 'balanced' (from requirements) to handle class imbalance.
            self.model_instance = SVC(class_weight='balanced', probability=True, random_state=config.get('random_state', 42))
        else:
            self.model_instance = SVR()

        self.param_distributions = {
            'C': config.get('C', [0.1, 1, 10, 100]),
            'kernel': config.get('kernel', ['linear', 'rbf']),
            'gamma': config.get('gamma', ['scale', 'auto']),
        }

    def preprocess(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
        """
        Constructs and applies the feature pipeline optimized for SVM.
        
        Pipeline Steps:
        1. Numerical: Median imputation. Standard Scaling (Critical for SVM). PCA (Dimensionality Reduction).
        2. Categorical: Most frequent imputation. One-Hot encoding.
        
        Args:
            X: Input features DataFrame.
            y: Target Series.
            
        Returns:
            Tuple containing:
            - Transformed feature array (np.ndarray)
            - Transformed target array (np.ndarray)
            - The fitted ColumnTransformer object
        """
        # 1. Pipeline Construction
        # ------------------------
        
        # Numerical Steps
        num_steps = []
        num_steps.append(('imputer', get_imputer(strategy='median'))) # Median Imputation (Req)
        
        if self.config.get('scaler') == 'minmax':
            num_steps.append(('scaler', get_minmax_scaler()))
        else:
            num_steps.append(('scaler', get_standard_scaler())) # Standard Scaler (Req)

        # PCA (Dimensionality Reduction - Req)
        if self.config.get('pca_components') is not None:
            num_steps.append(('pca', get_pca_reducer(n_components=self.config.get('pca_components', 0.95))))

        numerical_pipeline = Pipeline(steps=num_steps)

        # Categorical Steps
        cat_steps = []
        cat_steps.append(('imputer', get_imputer(strategy='most_frequent'))) 
        # OneHot Encoding (Req)
        cat_steps.append(('onehot', get_one_hot_encoder(handle_unknown='ignore', sparse_output=False)))
        
        # Note: PCA acts on dense arrays usually. If OneHot creates many features, passing them to PCA 
        # (in a global step) might be better. 
        # However, `ColumnTransformer` runs parallel. 
        # For simplicity and robustness, we will process num and cat separately. 
        # If PCA is desired on *everything*, we would need a second Pipeline wrapping the ColumnTransformer.
        # Given the requirements usually imply PCA on continuous features or all features:
        # We will apply PCA strictly to the numeric pipeline here as per common practice unless 'all' is specified.
        # But wait, OneHot creates sparse/many features. PCA is often used there too.
        # Let's stick to Numerical PCA for now to avoid blowing up complexity unless implied otherwise.
        
        cat_pipeline = Pipeline(steps=cat_steps)

        # 2. Composition
        # --------------
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, X.select_dtypes(include=np.number).columns.tolist()),
                ('cat', cat_pipeline, X.select_dtypes(include=['object', 'category']).columns.tolist())
            ],
            remainder='drop',
            n_jobs=self.config.get('n_jobs', -1)
        )

        # 3. Fit-Transform
        # ----------------
        X_transformed = preprocessor.fit_transform(X, y)
        X_transformed = np.asarray(X_transformed)

        # 4. Target Processing
        # --------------------
        if self.is_classification:
            le = LabelEncoder()
            y_transformed = le.fit_transform(y)
            self.label_encoder = le
        else:
            y_transformed = y.values # No encoding for regression target
            self.label_encoder = None

        return X_transformed, y_transformed, preprocessor

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the SVM model using Optuna (Tier 3) tuning strategy.
        
        Args:
            X_train: Training features array.
            y_train: Training target array.
        """
        from core_recommender.tuning import run_optuna_optimization

        # 1. Base Strategy
        if self.is_classification:
            base_cls = SVC
            # using 'accuracy' or 'f1_weighted' depending on preference. 
            # Config defaulted RandomizedSearch to f1_weighted.
            scoring = 'f1_weighted' 
            cv = StratifiedKFold(n_splits=self.config.get('cv_folds', 5), shuffle=True, random_state=self.config.get('random_state', 42))
        else:
            base_cls = SVR
            scoring = 'neg_root_mean_squared_error'
            cv = KFold(n_splits=self.config.get('cv_folds', 5), shuffle=True, random_state=self.config.get('random_state', 42))

        # 2. Run Optuna
        print(f"[{self.name}] Starting Training with OPTUNA strategy...")
        
        self.best_estimator = run_optuna_optimization(
            estimator_class=base_cls,
            param_space_func=self._get_optuna_space,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=scoring,
            n_trials=self.config.get('n_iter', 20), # Use n_iter config as n_trials for Optuna
            n_jobs=self.config.get('n_jobs', -1),
            random_state=self.config.get('random_state', 42)
        )
        
        # Capture study
        if hasattr(self.best_estimator, 'study_'):
            self.study = self.best_estimator.study_

        self.model = self.best_estimator
        print(f"[{self.name}] Best parameters: {self.study.best_params if hasattr(self, 'study') else 'N/A'}")

    def _get_optuna_space(self, trial):
        """Defines the search space for SVM."""
        # Kernels
        k_options = self.config.get('kernel', ['linear', 'rbf'])
        kernel = trial.suggest_categorical('kernel', k_options)
        
        # C (Regularization) - Log scale is crucial for C
        c_range = self.config.get('C', [0.1, 100])
        # If user passed a list [0.1, 1, 10, 100], finding min/max for range log search
        c_min, c_max = min(c_range), max(c_range)
        C = trial.suggest_float('C', c_min, c_max, log=True)
        
        # Gamma (Kernel coeff)
        # Svc accepts 'scale', 'auto' OR float. Optuna needs categorical or float.
        # We can mix them by suggesting a string from categorical, and if it's not scale/auto, treat as float?
        # Actually, standard practice: usually 'scale' or 'auto' are good enough. 
        # If we really want to tune gamma float value, we need a separate float range.
        # For simplicity/robustness similar to previous RandomizedSearch config which had ['scale', 'auto']
        gamma_options = [g for g in self.config.get('gamma', ['scale', 'auto']) if isinstance(g, str)]
        if not gamma_options: gamma_options = ['scale'] # Fallback
        gamma = trial.suggest_categorical('gamma', gamma_options)

        params = {
            'C': C,
            'kernel': kernel,
            'gamma': gamma
        }
        
        if self.is_classification:
            params['class_weight'] = 'balanced'
            params['probability'] = True # For diagnostics (ROC)
            params['random_state'] = self.config.get('random_state', 42)
            
        return params

    def calculate_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Calculates task-specific performance metrics.

        Metrics Include:
        - Accuracy, F1 Score (Classification)
        - RMSE, R2 Score (Regression)
        """
        y_pred = self.best_estimator.predict(X_test)
        
        metrics = {}
        if self.is_classification:
            metrics['Accuracy'] = calculate_accuracy(y_test, y_pred) # Req
            metrics['F1 Score'] = calculate_f1_score(y_test, y_pred, average='weighted') # Req
        else:
            metrics['RMSE'] = calculate_rmse(y_test, y_pred)
            metrics['R2 Score'] = calculate_r2_score(y_test, y_pred)
        
        return metrics

    def get_diagnostic_data(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Retrieves diagnostic data for visualization.
        """
        if not hasattr(self, 'best_estimator'):
             raise RuntimeError("Model must be fitted before diagnostics.")
        
        y_pred = self.best_estimator.predict(X_test)
        
        data = {
            'y_pred': y_pred,
            'y_test': y_test,
            'model_name': self.name,
            'support_vectors': self.best_estimator.support_vectors_,
            'n_support': self.best_estimator.n_support_ if hasattr(self.best_estimator, 'n_support_') else None,
            'is_classification': self.is_classification
        }

        if self.is_classification and hasattr(self.best_estimator, 'predict_proba'):
            data['y_proba'] = self.best_estimator.predict_proba(X_test)

        return data
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Retrieves feature importance (Coefficients) for Linear kernel SVMs only.
        """
        # Check kernel on the fitted estimator instance directly
        if getattr(self.best_estimator, 'kernel', '') == 'linear':
            if hasattr(self.best_estimator, 'coef_'):
                 # Returning raw coefs not super useful without feature names & multi-class handling
                 # But keeping generic as requested
                 return {'coef_': self.best_estimator.coef_}
        return {}
