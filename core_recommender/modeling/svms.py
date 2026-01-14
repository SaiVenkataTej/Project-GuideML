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

# Import centralized logger
from core_recommender.logger import get_logger
logger = get_logger(__name__)

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
    A concrete implementation of Support Vector Machines (SVM) for both Classification (SVC) 
    and Regression (SVR) within the core_recommender framework.
    
    Overview:
    ---------
    Support Vector Machines works by finding the optimal hyperplane that separates data points 
    of different classes (SVC) or fits the data within a specified margin of error (SVR).
    This implementation encapsulates the complexity of:
    1. Pipeline Construction: Integrating Scaling, PCA, and OneHot Encoding automatically.
    2. Hyperparameter Tuning: Automating the search for optimal 'C', 'kernel', and 'gamma' using Optuna.
    3. Diagnostics: Providing support vector counts, probabilities, and decision function analysis.

    Key Features:
    -------------
    - **Dual Mode**: Automatically switches between SVC and SVR based on `is_classification` flag.
    - **Dimensionality Reduction**: Integrated PCA step to handle high-dimensional data, improving 
      SVM performance and training time (controlled via `pca_components`).
    - **Robust Scaling**: Enforces Standardization or MinMax scaling, which is a strict requirement 
      for SVM convergence.
    - **Balanced Weights**: Handles class imbalance automatically in classification mode.

    Configuration (`CONFIG`):
    -------------------------
    - `kernel`: List[str] - Kernels to try (e.g., 'linear', 'rbf'). Linear is faster; RBF captures non-linearity.
    - `C`: List[float] - Regularization parameter range. Low C = simple decision surface (high bias), 
      High C = complex surface (high variance).
    - `gamma`: List[Union[str, float]] - Kernel coefficient. 'scale' is recommended.
    - `pca_components`: float|int|None - Variance ratio to keep (if < 1.0) or count (if int). None to disable.
    """
    def __init__(self, is_classification: bool = True, config: Dict[str, Any] = CONFIG):
        """
        Initializes the SVM model with task-specific configurations and sets up the internal 
        scikit-learn estimator.

        Args:
            is_classification (bool): 
                - If `True`, initializes a `SVC` (Support Vector Classifier).
                - If `False`, initializes a `SVR` (Support Vector Regressor).
            config (Dict[str, Any]): 
                Configuration dictionary containing hyperparameters and pipeline settings.
                Keys should match those in the global `CONFIG` dictionary. 
                Defaults are used for missing keys.
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
        Constructs and applies a robust feature preprocessing pipeline optimized for Support Vector Machines.
        
        Rationale:
        ----------
        SVMs are distance-based algorithms heavily influenced by feature scales. 
        - **Scaling**: A StandardScaler (or MinMax) is mandatory to ensure all features contribute 
          equally to the margin calculation.
        - **Imputation**: Missing values are imputed (Median for numerical, Mode for categorical) 
          as SVMs cannot handle NaNs.
        - **Encoding**: Categorical variables are One-Hot Encoded.
        - **Dimensions**: Optional PCA reduction is applied to numerical data to mitigate the 
          "Curse of Dimensionality" and speed up convergence.

        Args:
            X (pd.DataFrame): Raw input features.
            y (pd.Series): Raw target variable.

        Returns:
            Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
                - **X_transformed**: Numpy array of processed features ready for training.
                - **y_transformed**: Numpy array of processed target (Label Encoded if classification).
                - **preprocessor**: The fitted `ColumnTransformer` object, essential for ensuring 
                  test data undergoes the exact same transformation.
        """
        logger.debug(f"[{self.name}] Entering preprocess()...")
        logger.debug(f"[{self.name}] Input shape: X={X.shape}, y={y.shape}")
        
        # 1. Pipeline Construction
        # ------------------------
        
        # Numerical Steps
        num_steps = []
        num_steps.append(('imputer', get_imputer(strategy='median'))) # Median Imputation (Req)
        
        if self.config.get('scaler') == 'minmax':
            num_steps.append(('scaler', get_minmax_scaler()))
            logger.debug(f"[{self.name}] Using MinMaxScaler for feature scaling")
        else:
            num_steps.append(('scaler', get_standard_scaler())) # Standard Scaler (Req)
            logger.debug(f"[{self.name}] Using StandardScaler for feature scaling")

        # PCA (Dimensionality Reduction - Req)
        if self.config.get('pca_components') is not None:
            num_steps.append(('pca', get_pca_reducer(n_components=self.config.get('pca_components', 0.95))))
            logger.debug(f"[{self.name}] Applying PCA (n_components={self.config.get('pca_components', 0.95)})")

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
        logger.debug(f"[{self.name}] ColumnTransformer configured with numerical and categorical pipelines.")

        self.preprocessor = preprocessor

        X_transformed = preprocessor.fit_transform(X, y)
        X_transformed = np.asarray(X_transformed)
        logger.debug(f"[{self.name}] X transformed shape: {X_transformed.shape}")

        if self.is_classification:
            if not np.issubdtype(y.dtype, np.number):
                le = LabelEncoder()
                y_transformed = le.fit_transform(y)
                self.label_encoder = le
                logger.debug(f"[{self.name}] Target variable LabelEncoded for classification.")
            else:
                y_transformed = y.values
                self.label_encoder = None
                logger.debug(f"[{self.name}] Target variable is already numerical for classification.")
        else:
            y_transformed = y.values
            self.label_encoder = None
            logger.debug(f"[{self.name}] Target variable for regression.")

        logger.debug(f"[{self.name}] Preprocessing complete. Output shapes: X={X_transformed.shape}, y={y_transformed.shape}")
        return X_transformed, y_transformed, preprocessor

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """
        Trains the SVM model using a Unified Pipeline to prevent data leakage.
        """
        logger.info(f"[{self.name}] Starting training...")
        logger.debug(f"[{self.name}] Training data shape: X={X_train.shape}, y={y_train.shape}")
        
        from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
        from sklearn.base import clone
        import optuna

        # 1. Base Strategy
        if self.is_classification:
            base_cls = SVC
            scoring = 'f1_weighted' 
            cv = StratifiedKFold(n_splits=self.config.get('cv_folds', 5), shuffle=True, random_state=self.config.get('random_state', 42))
            logger.debug(f"[{self.name}] Classification task: SVC, scoring='{scoring}', using StratifiedKFold.")
        else:
            base_cls = SVR
            scoring = 'neg_root_mean_squared_error'
            cv = KFold(n_splits=self.config.get('cv_folds', 5), shuffle=True, random_state=self.config.get('random_state', 42))
            logger.debug(f"[{self.name}] Regression task: SVR, scoring='{scoring}', using KFold.")

        # 2. Pipeline-based Tuning
        preprocessor_template = clone(self.preprocessor)
        
        def pipeline_objective(trial):
            params = self._get_optuna_space(trial)
            model_inst = base_cls(**params)
            
            pipe = Pipeline(steps=[
                ('pre', preprocessor_template),
                ('model', model_inst)
            ])
            
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=self.config.get('n_jobs', -1))
            return scores.mean()

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logger.debug(f"[{self.name}] Starting Optuna optimization (n_trials={self.config.get('n_iter', 20)})...")
        study = optuna.create_study(direction='maximize')
        study.optimize(pipeline_objective, n_trials=self.config.get('n_iter', 20))
        
        # 3. Build Best Model
        best_params = study.best_params
        best_model_inst = base_cls(**best_params)
        
        self.best_estimator = Pipeline(steps=[
            ('pre', preprocessor_template),
            ('model', best_model_inst)
        ])
        self.best_estimator.fit(X_train, y_train)
        
        self.study = study
        self.model = self.best_estimator
        
        best_score = study.best_value
        best_params = study.best_params
        logger.info(f"✅ [{self.name}] Training complete")
        logger.info(f"[{self.name}] Best CV Score: {best_score:.4f}")
        logger.debug(f"[{self.name}] Best params: {best_params}")

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

    def calculate_metrics(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """
        Calculates performance metrics using the full Pipeline.
        """
        y_pred = self.best_estimator.predict(X_test)
        
        metrics = {}
        if self.is_classification:
            metrics['Accuracy'] = calculate_accuracy(y_test, y_pred) 
            metrics['F1 Score'] = calculate_f1_score(y_test, y_pred, average='weighted')
        else:
            metrics['RMSE'] = calculate_rmse(y_test, y_pred)
            metrics['R2 Score'] = calculate_r2_score(y_test, y_pred)
        
        return metrics

    def get_diagnostic_data(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Retrieves diagnostic data for visualization using the Pipeline.
        """
        if not hasattr(self, 'best_estimator'):
             raise RuntimeError("Model must be fitted before diagnostics.")
        
        y_pred = self.best_estimator.predict(X_test)
        final_model = self.best_estimator.named_steps['model']
        
        data = {
            'y_pred': y_pred,
            'y_test': y_test,
            'model_name': self.name,
            'support_vectors': final_model.support_vectors_,
            'n_support': final_model.n_support_ if hasattr(final_model, 'n_support_') else None,
            'is_classification': self.is_classification
        }

        if self.is_classification and hasattr(final_model, 'predict_proba'):
            data['y_proba'] = self.best_estimator.predict_proba(X_test)

        return data
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Retrieves feature importance (Coefficients) for Linear kernel SVMs only.
        """
        final_model = self.best_estimator.named_steps['model']
        if getattr(final_model, 'kernel', '') == 'linear':
            if hasattr(final_model, 'coef_'):
                 return {'coef_': final_model.coef_}
        return {}

    def get_parameter_descriptions(self) -> Dict[str, Dict[str, str]]:
        """
        Returns descriptions of the most important tuned parameters.
        """
        final_model = self.best_estimator.named_steps['model']
        params = final_model.get_params()
        descriptions = {
            'C': {
                'value': f"{params.get('C'):.4f}",
                'desc': 'The regularization parameter. It controls the trade-off between maximizing the margin and minimizing classification errors.'
            },
            'kernel': {
                'value': str(params.get('kernel')),
                'desc': 'The type of kernel used (e.g., RBF or Linear). Kernels allow SVM to find non-linear boundaries in higher-dimensional space.'
            },
            'gamma': {
                'value': str(params.get('gamma')),
                'desc': 'Defines how far the influence of a single training example reaches. Low values mean "far" and high values mean "close".'
            }
        }
        return descriptions
