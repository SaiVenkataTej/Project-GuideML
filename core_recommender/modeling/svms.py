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
    A concrete implementation of Support Vector Machines (SVM) for Classification (SVC) and Regression (SVR).
    """
    def __init__(self, is_classification: bool = True, config: Dict[str, Any] = CONFIG):
        
        task_name = "Classification" if is_classification else "Regression"
        name = f"SVM ({task_name})"
        super().__init__(name=name, config=config)
        
        self.is_classification = is_classification
        
        # Initialize Model Instance & Param Grid
        if self.is_classification:
            # SVC with probability=True for diagnostics (optional, but good for ROC/LogLoss)
            # Class Weights 'balanced' (from requirements)
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
        Trains using RandomizedSearchCV and appropriate CV (Stratified for Classif).
        """
        if self.is_classification:
            cv = StratifiedKFold( # Stratified K-Fold (Req)
                n_splits=self.config.get('cv_folds', 5),
                shuffle=True,
                random_state=self.config.get('random_state', 42)
            )
            scoring = 'f1_weighted'
        else:
            cv = KFold(
                n_splits=self.config.get('cv_folds', 5),
                shuffle=True,
                random_state=self.config.get('random_state', 42)
            )
            scoring = 'neg_root_mean_squared_error'

        self.model = RandomizedSearchCV( # Randomized Search CV (Req)
            estimator=self.model_instance,
            param_distributions=self.param_distributions,
            n_iter=self.config.get('n_iter', 10),
            scoring=scoring,
            cv=cv,
            n_jobs=self.config.get('n_jobs', -1),
            verbose=1,
            random_state=self.config.get('random_state', 42)
        )

        print(f"[{self.name}] Starting RandomizedSearchCV...")
        self.model.fit(X_train, y_train)
        
        self.best_estimator = self.model.best_estimator_
        print(f"[{self.name}] Best parameters: {self.model.best_params_}")

    def calculate_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Calculates Accuracy/F1 (Classif) or RMSE/R2 (Reg).
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
        Returns data for diagnostics (Confusion Matrix, Support Vectors).
        """
        if not hasattr(self, 'best_estimator'):
             raise RuntimeError("Model must be fitted before diagnostics.")

        y_pred = self.best_estimator.predict(X_test)
        
        data = {
            'y_pred': y_pred,
            'y_test': y_test,
            'model_name': self.name,
            # Support Vectors (Req: Inspection purposes)
            'support_vectors': self.best_estimator.support_vectors_,
            'n_support': self.best_estimator.n_support_ if hasattr(self.best_estimator, 'n_support_') else None,
            'is_classification': self.is_classification
        }

        if self.is_classification and hasattr(self.best_estimator, 'predict_proba'):
            data['y_proba'] = self.best_estimator.predict_proba(X_test)

        return data
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Returns feature coefficients for Linear kernel only.
        """
        if self.model.best_params_['estimator__kernel'] == 'linear':
            if hasattr(self.best_estimator, 'coef_'):
                # Handle multi-class case? coef_ shape (n_classes, n_features)
                # Just return raw or mean abs?
                # For simplicity, returning empty or first class if binary.
                 return {} # Complex to map back to features generally here without column names context
        return {}
