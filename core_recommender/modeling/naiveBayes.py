import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder

# --- PROJECT IMPORTS ---
from core_recommender.modeling.baseModel import BaseModel
from core_recommender.preprocessing import (
    get_imputer,
    get_one_hot_encoder,
    get_ordinal_encoder,
    get_standard_scaler,
    get_minmax_scaler,
    get_yeo_johnson_transformer,
    get_select_k_best
)
from core_recommender.evaluation import (
    calculate_accuracy,
    calculate_f1_score,
    calculate_log_loss,
    calculate_precision
)

# --- DEFAULT CONFIGURATION ---
CONFIG = {
    'model_type': 'gaussian', # 'gaussian', 'multinomial'
    'scaler': 'standard',     # 'standard', 'minmax'
    'feature_selection': 'k_best',
    'k_best': 10,
    'k_best_score_func': 'f_classif', # 'f_classif', 'chi2', 'mutual_info_classif'
    'cv_folds': 5,
    'random_state': 42,
    'n_jobs': -1
}

# =========================================================================
# NaiveBayesModel Class
# =========================================================================

class NaiveBayesModel(BaseModel):
    """
    A concrete implementation of Naive Bayes (Gaussian, Multinomial) for Classification.
    """
    def __init__(self, config: Dict[str, Any] = CONFIG):
        
        name = f"Naive Bayes ({config.get('model_type', 'gaussian').capitalize()})"
        super().__init__(name=name, config=config)
        
        self.model_type = config.get('model_type', 'gaussian')
        
        # Initialize Model Instance
        if self.model_type == 'gaussian':
            # var_smoothing will be grid searched
            self.model_instance = GaussianNB()
            self.param_grid = {
                'var_smoothing': np.logspace(0, -9, num=10)
            }
        elif self.model_type == 'multinomial':
            # alpha will be grid searched
            self.model_instance = MultinomialNB()
            self.param_grid = {
                'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
            }
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def preprocess(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
        """
        Constructs and applies the feature pipeline optimized for Naive Bayes.
        """
        # Check for NaNs object target
        if y.isna().any():
             raise ValueError("Target variable 'y' contains missing values.")

        # 1. Pipeline Construction
        # ------------------------
        
        # Numerical Steps
        num_steps = []
        if self.model_type == 'gaussian':
            num_steps.append(('imputer', get_imputer(strategy='mean'))) # Mean for Gaussian
            num_steps.append(('yeo_johnson', get_yeo_johnson_transformer())) # Power Transform
            
            if self.config.get('scaler') == 'minmax':
                num_steps.append(('scaler', get_minmax_scaler()))
            else:
                num_steps.append(('scaler', get_standard_scaler()))
                
        elif self.model_type == 'multinomial':
            # Multinomial expects non-negative counts. 
            # Impute (median/mean) -> MinMax (to ensure positive) or keep raw if counts
            num_steps.append(('imputer', get_imputer(strategy='median')))
            num_steps.append(('scaler', get_minmax_scaler())) # Ensure non-negative

        # Feature Selection (Applies to numericals mostly here, but can also be applied after one-hot)
        # Note: Scikit-learn Feature selection is usually applied *after* column transformer.
        # But we build pipelines per column type. Let's add selection to numerical pipeline for now, 
        # or globally. In this architecture, we return X_transformed.
        # We'll add selection to the final numeric pipeline.
        
        if self.config.get('feature_selection') == 'k_best':
            num_steps.append(('select_k_best', get_select_k_best(
                k=self.config.get('k_best', 10), 
                score_func=self.config.get('k_best_score_func', 'f_classif')
            )))

        numerical_pipeline = Pipeline(steps=num_steps)

        # Categorical Steps
        cat_steps = []
        cat_steps.append(('imputer', get_imputer(strategy='most_frequent'))) # Mode for categorical
        
        # Encoding
        if self.model_type == 'multinomial':
            # OneHot for Multinomial
            cat_steps.append(('onehot', get_one_hot_encoder(handle_unknown='ignore', sparse_output=False)))
        else:
            # For Gaussian, we usually OneHot too. 
            # If user wanted Ordinal for 'CategoricalNB', that's a different path.
            # Assuming GaussianNB handles OneHot encoded features reasonably well as continuous 0/1.
            cat_steps.append(('onehot', get_one_hot_encoder(handle_unknown='ignore', sparse_output=False)))

        cat_pipeline = Pipeline(steps=cat_steps)

        # 2. Composition
        # --------------
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, X.select_dtypes(include=np.number).columns.tolist()),
                ('cat', cat_pipeline, X.select_dtypes(include=['object', 'category']).columns.tolist())
            ],
            remainder='drop', # Drop unknown types
            n_jobs=self.config.get('n_jobs', -1)
        )

        # 3. Fit-Transform
        # ----------------
        X_transformed = preprocessor.fit_transform(X, y) # SelectKBest (f_classif) needs y
        X_transformed = np.asarray(X_transformed)

        # 4. Target Encoding
        # ------------------
        le = LabelEncoder()
        y_transformed = le.fit_transform(y)
        self.label_encoder = le

        return X_transformed, y_transformed, preprocessor

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains using Stratified K-Fold CV and GridSearchCV.
        """
        cv = StratifiedKFold(
            n_splits=self.config.get('cv_folds', 5),
            shuffle=True,
            random_state=self.config.get('random_state', 42)
        )

        self.model = GridSearchCV(
            estimator=self.model_instance,
            param_grid=self.param_grid,
            scoring='f1_weighted',
            cv=cv,
            n_jobs=self.config.get('n_jobs', -1),
            verbose=1
        )

        print(f"Starting GridSearchCV for {self.name}...")
        self.model.fit(X_train, y_train)
        
        self.best_estimator = self.model.best_estimator_
        print(f"Best parameters found: {self.model.best_params_}")

    def calculate_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Calculates Accuracy, F1, Log Loss, Precision.
        """
        y_pred = self.best_estimator.predict(X_test)
        y_proba = self.best_estimator.predict_proba(X_test)
        
        metrics = {}
        metrics['Accuracy'] = calculate_accuracy(y_test, y_pred)
        metrics['F1 Score'] = calculate_f1_score(y_test, y_pred, average='weighted')
        metrics['Precision'] = calculate_precision(y_test, y_pred, average='weighted')
        metrics['Log Loss'] = calculate_log_loss(y_test, y_proba)
        
        return metrics

    def get_diagnostic_data(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Returns data for Confusion Matrix, Histograms, etc.
        """
        if not hasattr(self, 'best_estimator'):
             raise RuntimeError("Model must be fitted before diagnostics.")

        y_pred = self.best_estimator.predict(X_test)
        y_proba = self.best_estimator.predict_proba(X_test)

        return {
            'y_pred': y_pred,
            'y_proba': y_proba,
            'y_test': y_test,
            'model_name': self.name,
            'feature_log_prob': getattr(self.best_estimator, 'feature_log_prob_', None)
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        NB doesn't have standard feature importance like Trees/Linear.
        We can potentially return permutation importance if implemented generically,
        but Base interface implies simple attribute extraction.
        Returning empty for now.
        """
        return {}
