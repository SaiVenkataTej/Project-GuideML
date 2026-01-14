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

# Import centralized logger
from core_recommender.logger import get_logger
logger = get_logger(__name__)

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
    
    Rationale:
    ----------
    - **Independence Assumption**: Assumes features are independent given the class. While rarely true in practice, it often works surprisingly well.
    - **Efficiency**: Extremely fast training and prediction (linear time complexity).
    - **Small Data**: Performs well even with less training data compared to complex models like Neural Networks.

    This model assumes independence between predictors. It is highly efficient and scalable, 
    often serving as a strong baseline for text classification and other high-dimensional tasks.
    Gaussian NB is used for continuous features, while Multinomial NB is suited for counts.
    """
    def __init__(self, config: Dict[str, Any] = CONFIG):
        """
        Initializes the Naive Bayes model.

        Args:
            config: Dictionary containing hyperparameters (e.g., 'model_type', 'var_smoothing', 'alpha').
                    Defaults to the global CONFIG dictionary.
        """
        
        name = f"Naive Bayes ({config.get('model_type', 'gaussian').capitalize()})"
        super().__init__(name=name, config=config)
        
        self.model_type = config.get('model_type', 'gaussian')
        
        # Initialize Model Instance based on type
        if self.model_type == 'gaussian':
            # var_smoothing will be grid searched (handles numerical stability)
            self.model_instance = GaussianNB()
            self.param_grid = {
                'var_smoothing': np.logspace(0, -9, num=10) 
            }
        elif self.model_type == 'multinomial':
            # alpha (smoothing parameter) will be grid searched
            self.model_instance = MultinomialNB()
            self.param_grid = {
                'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
            }
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def preprocess(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
        """
        Constructs and applies the feature pipeline optimized for Naive Bayes.
        
        Rationale:
        ----------
        - **Gaussian NB**: Assumes features follow a normal distribution. Yeo-Johnson transformation helps force this normality.
        - **Multinomial NB**: Designed for counts (e.g., text freq). Requires non-negative data (MinMax scaling ensures this).
        
        Pipeline Steps:
        1. Numerical (Gaussian): Mean imputation. Yeo-Johnson transformation. Scaling.
        2. Numerical (Multinomial): Median imputation. MinMax Scaling (ensure non-negative).
        3. Categorical: Most frequent imputation. One-Hot encoding.
        
        Args:
            X: Input features DataFrame.
            y: Target Series.
            
        Returns:
            Tuple containing:
            - Transformed feature array (np.ndarray)
            - Transformed target array (np.ndarray)
            - The fitted ColumnTransformer object
            
        Raises:
            ValueError: If target variable 'y' contains NaNs.
        """
        logger.debug(f"[{self.name}] Entering preprocess()...")
        logger.debug(f"[{self.name}] Input shape: X={X.shape}, y={y.shape}")
        logger.debug(f"[{self.name}] Model type: {self.model_type}")
        
        # Check for NaNs in target
        if y.isna().any():
            logger.error(f"[{self.name}] Target variable contains {y.isna().sum()} NaN values")
            raise ValueError("Target variable 'y' contains missing values.")

        # 1. Pipeline Construction
        # ------------------------
        
        # Numerical Steps
        num_steps = []
        if self.model_type == 'gaussian':
            num_steps.append(('imputer', get_imputer(strategy='mean'))) # Mean for Gaussian
            num_steps.append(('yeo_johnson', get_yeo_johnson_transformer())) # Power Transform
            logger.debug(f"[{self.name}] Using Gaussian pipeline: Mean imputation + Yeo-Johnson transformation")
            
            if self.config.get('scaler') == 'minmax':
                num_steps.append(('scaler', get_minmax_scaler()))
            else:
                num_steps.append(('scaler', get_standard_scaler()))
                
        elif self.model_type == 'multinomial':
            # Multinomial expects non-negative counts. 
            # Impute (median/mean) -> MinMax (to ensure positive) or keep raw if counts
            num_steps.append(('imputer', get_imputer(strategy='median')))
            num_steps.append(('scaler', get_minmax_scaler())) # Ensure non-negative
            logger.debug(f"[{self.name}] Using Multinomial pipeline: Median imputation + MinMaxScaler (non-negative)")

        # Feature Selection (Applies to numericals mostly here, but can also be applied after one-hot)
        # Note: Scikit-learn Feature selection is usually applied *after* column transformer.
        # But we build pipelines per column type. Let's add selection to numerical pipeline for now, 
        # or globally. In this architecture, we return X_transformed.
        
        if self.config.get('feature_selection') == 'k_best':
            num_steps.append(('select_k_best', get_select_k_best(
                k=self.config.get('k_best', 10), 
                score_func=self.config.get('k_best_score_func', 'f_classif')
            )))
            logger.debug(f"[{self.name}] Numerical pipeline includes feature selection (k_best={self.config.get('k_best', 10)})")

        numerical_pipeline = Pipeline(steps=num_steps)
        logger.debug(f"[{self.name}] Numerical pipeline created with steps: {[s[0] for s in num_steps]}")

        # Categorical Steps
        cat_steps = []
        cat_steps.append(('imputer', get_imputer(strategy='most_frequent'))) # Mode for categorical
        
        # Encoding
        if self.model_type == 'multinomial':
            # OneHot for Multinomial
            cat_steps.append(('onehot', get_one_hot_encoder(handle_unknown='ignore', sparse_output=False)))
            logger.debug(f"[{self.name}] Categorical pipeline: Most frequent imputation + One-Hot encoding for Multinomial")
        else:
            # For Gaussian, we usually OneHot too. 
            # If user wanted Ordinal for 'CategoricalNB', that's a different path.
            # Assuming GaussianNB handles OneHot encoded features reasonably well as continuous 0/1.
            cat_steps.append(('onehot', get_one_hot_encoder(handle_unknown='ignore', sparse_output=False)))
            logger.debug(f"[{self.name}] Categorical pipeline: Most frequent imputation + One-Hot encoding for Gaussian")

        cat_pipeline = Pipeline(steps=cat_steps)
        logger.debug(f"[{self.name}] Categorical pipeline created with steps: {[s[0] for s in cat_steps]}")

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
        logger.debug(f"[{self.name}] ColumnTransformer created. Numerical features: {len(X.select_dtypes(include=np.number).columns)}, Categorical features: {len(X.select_dtypes(include=['object', 'category']).columns)}")

        self.preprocessor = preprocessor

        logger.debug(f"[{self.name}] Fitting and transforming X...")
        X_transformed = preprocessor.fit_transform(X, y)
        X_transformed = np.asarray(X_transformed)
        logger.debug(f"[{self.name}] X transformed shape: {X_transformed.shape}")

        # 4. Target Encoding (Skip if already numeric/pre-encoded by Executor)
        if not np.issubdtype(y.dtype, np.number):
            logger.debug(f"[{self.name}] Target variable is not numeric, applying LabelEncoder.")
            le = LabelEncoder()
            y_transformed = le.fit_transform(y)
            self.label_encoder = le
            logger.debug(f"[{self.name}] Target classes: {le.classes_}")
        else:
            logger.debug(f"[{self.name}] Target variable is already numeric, skipping LabelEncoder.")
            y_transformed = y.values
            self.label_encoder = None
        logger.debug(f"[{self.name}] Preprocessing complete. Exiting preprocess().")

        return X_transformed, y_transformed, preprocessor

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """
        Trains the Naive Bayes model using a Unified Pipeline to prevent data leakage.
        """
        logger.info(f"[{self.name}] Starting training...")
        logger.debug(f"[{self.name}] Training data shape: X={X_train.shape}, y={y_train.shape}")
        
        from sklearn.model_selection import StratifiedKFold, GridSearchCV
        from sklearn.base import clone

        # 1. Pipeline Construction
        preprocessor_template = clone(self.preprocessor)
        pipe = Pipeline(steps=[
            ('pre', preprocessor_template),
            ('model', self.model_instance)
        ])
        logger.debug(f"[{self.name}] Training pipeline created: preprocessor + {self.model_type} Naive Bayes.")

        # 2. Adjust Param Grid
        pipeline_params = {f'model__{k}': v for k, v in self.param_grid.items()}
        logger.debug(f"[{self.name}] Parameter grid for GridSearchCV: {pipeline_params}")

        # 3. CV Strategy
        cv = StratifiedKFold(
            n_splits=self.config.get('cv_folds', 5),
            shuffle=True,
            random_state=self.config.get('random_state', 42)
        )
        logger.debug(f"[{self.name}] Cross-validation strategy: StratifiedKFold with {self.config.get('cv_folds', 5)} splits.")

        # 4. GridSearch on Pipeline
        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=pipeline_params,
            scoring='f1_weighted',
            cv=cv,
            n_jobs=self.config.get('n_jobs', -1),
            verbose=0
        )
        logger.debug(f"[{self.name}] Initializing GridSearchCV with scoring 'f1_weighted' and n_jobs={self.config.get('n_jobs', -1)}.")

        grid_search.fit(X_train, y_train)
        
        self.best_estimator = grid_search.best_estimator_
        self.model = grid_search
        
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_
        logger.info(f"✅ [{self.name}] Training complete")
        logger.info(f"[{self.name}] Best CV Score: {best_score:.4f} (F1-weighted)")
        logger.debug(f"[{self.name}] Best params: {best_params}")

    def calculate_metrics(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """
        Calculates performance metrics using the full Pipeline.
        """
        logger.debug(f"[{self.name}] Calculating metrics on test data (shape: X={X_test.shape}, y={y_test.shape}).")
        if not hasattr(self, 'best_estimator'):
            raise RuntimeError("Model must be fitted before calculating metrics.")

        y_pred = self.best_estimator.predict(X_test)
        y_proba = self.best_estimator.predict_proba(X_test)
        
        metrics = {}
        metrics['Accuracy'] = calculate_accuracy(y_test, y_pred)
        metrics['F1 Score'] = calculate_f1_score(y_test, y_pred, average='weighted')
        metrics['Precision'] = calculate_precision(y_test, y_pred, average='weighted')
        metrics['Log Loss'] = calculate_log_loss(y_test, y_proba)
        
        return metrics

    def get_diagnostic_data(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Retrieves diagnostic data for visualization.
        
        Includes:
        - Predictions and Probabilities (for ROC curves)
        - Feature Log Probabilities (for internal model inspection if supported)
        
        Args:
            X_test: Test features array.
            y_test: Test target array.
            
        Returns:
            Dict[str, Any]: Data dictionary for the visualization module.
            
        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if not hasattr(self, 'best_estimator'):
             raise RuntimeError("Model must be fitted before diagnostics.")

        y_pred = self.best_estimator.predict(X_test)
        y_proba = self.best_estimator.predict_proba(X_test)

        final_model = self.best_estimator.named_steps['model']

        return {
            'y_pred': y_pred,
            'y_proba': y_proba,
            'y_test': y_test,
            'model_name': self.name,
            'feature_log_prob': getattr(final_model, 'feature_log_prob_', None)
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Naive Bayes does not provide a global feature importance metric comparable to Trees or Linear models.
        """
        return {}

    def get_parameter_descriptions(self) -> Dict[str, Dict[str, str]]:
        """
        Returns descriptions of the most important tuned parameters.
        """
        best_params = self.model.best_params_
        if self.model_type == 'gaussian':
            return {
                'var_smoothing': {
                    'value': f"{best_params.get('model__var_smoothing'):.2e}",
                    'desc': 'Portion of the largest variance of all features that is added to variances for calculation stability.'
                }
            }
        else:
            return {
                'alpha': {
                    'value': str(best_params.get('model__alpha')),
                    'desc': 'Additive (Laplace/Lidstone) smoothing parameter. Helps handle features not present in the training set.'
                }
            }
