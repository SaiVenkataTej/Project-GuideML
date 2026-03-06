import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Union

# --- SKLEARN IMPORTS ---
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.base import clone

# --- PROJECT IMPORTS ---
from core_recommender.modeling.baseModel import BaseModel
from core_recommender.preprocessing import (
    get_imputer, 
    get_one_hot_encoder, 
    get_standard_scaler, 
    get_robust_scaler,
    get_log_transformer,
    get_box_cox_transformer,
    get_yeo_johnson_transformer,
    get_variance_threshold,
    get_select_k_best
)
from core_recommender.evaluation import (
    calculate_rmse, 
    calculate_mae, 
    calculate_r2_score, 
    calculate_adjusted_r2
)

# Import centralized logger
from core_recommender.logger import get_logger
logger = get_logger(__name__)

# --- DEFAULT CONFIGURATION ---
CONFIG = {
    'scaler': 'standard',      # 'standard' or 'robust'
    'transformation': 'none',  # 'none', 'log', 'box-cox', 'yeo-johnson'
    'feature_selection': 'none', # 'none', 'variance', 'k_best'
    'cv_folds': 3,
    'random_state': 42,
    'n_jobs': -1
}

# =========================================================================
# LinearRegressionModel Class
# =========================================================================

class LinearRegressionModel(BaseModel):
    """A concrete implementation of Linear Regression optimized for Regression tasks.
    
    Utilizes ElasticNet which generalizes Ridge (L2) and Lasso (L1) regularization,
    allowing for both variable selection and coefficient shrinkage.
    
    Attributes:
        model_instance (ElasticNet): The underlying Scikit-learn estimator.
        param_grid (Dict[str, List[Any]]): Hyperparameter grid for tuning.
        preprocessor (ColumnTransformer): The feature engineering pipeline.
        best_estimator (Pipeline): The fitted pipeline after tuning.
    """
    
    def __init__(self, config: Dict[str, Any] = CONFIG) -> None:
        """Initializes the Linear Regression model.

        Args:
            config (Dict[str, Any], optional): Dictionary containing hyperparameters. 
                Defaults to the global CONFIG dictionary.
        """
        super().__init__(
            name="Linear Regression (ElasticNet)",
            config=config
        )
        
        # Initialize the estimator (ElasticNet covers Ridge/Lasso/ElasticNet)
        self.model_instance = ElasticNet(random_state=config.get('random_state', 42))
        
        # Params for GridSearch
        self.param_grid = {
            'alpha': [0.01, 0.1, 1.0, 10.0],  # Regularization strength
            'l1_ratio': [0.1, 0.5, 0.7, 0.9, 1.0] # 1.0 = Lasso, 0.0 ~ Ridge. 
        }
        self.preprocessor: Optional[ColumnTransformer] = None
        self.best_estimator: Optional[Pipeline] = None

    def preprocess(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
        """Constructs and applies the feature pipeline optimized for Linear Regression.
        
        Pipeline Steps:
        1. Numerical: Imputation -> Transformation -> Scaling -> Selection.
        2. Categorical: Imputation -> One-Hot Encoding.
        
        Args:
            X (pd.DataFrame): Input features DataFrame.
            y (pd.Series): Target Series.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, ColumnTransformer]: 
                - Transformed feature array.
                - Transformed target array.
                - The fitted ColumnTransformer object.
            
        Raises:
            ValueError: If target variable 'y' contains NaNs.
        """
        logger.debug(f"[{self.name}] Entering preprocess()...")
        logger.debug(f"[{self.name}] Input shape: X={X.shape}, y={y.shape}")
        
        # Edge Case: Check for NaNs in target y before proceeding
        if pd.isna(y).any():
            logger.error(f"[{self.name}] Target variable contains {pd.isna(y).sum()} NaN values")
            raise ValueError("Target variable 'y' contains missing values (NaNs). Please handle missing targets before training.")

        # 1. Scaling Strategy
        scaler_type = self.config.get('scaler', 'standard')
        if scaler_type == 'robust':
            scaler = get_robust_scaler()
            logger.debug(f"[{self.name}] Using RobustScaler (resistant to outliers)")
        else:
            scaler = get_standard_scaler()
            logger.debug(f"[{self.name}] Using StandardScaler (zero mean, unit variance)")

        # 2. Numerical Pipeline
        num_steps: List[Tuple[str, Any]] = [
            ('imputer', get_imputer(strategy='median'))
        ]

        # Optional: Transformation
        trans_type = self.config.get('transformation')
        if trans_type == 'log':
            num_steps.append(('log_transform', get_log_transformer()))
            logger.debug(f"[{self.name}] Applying log transformation for normality")
        elif trans_type == 'box-cox':
            # Box-Cox requires strictly positive data
            num_steps.append(('box_cox', get_box_cox_transformer()))
            logger.debug(f"[{self.name}] Applying Box-Cox transformation")
        elif trans_type == 'yeo-johnson':
            num_steps.append(('yeo_johnson', get_yeo_johnson_transformer()))
            logger.debug(f"[{self.name}] Applying Yeo-Johnson transformation")
        
        num_steps.append(('scaler', scaler))

        # Optional: Feature Selection
        sel_type = self.config.get('feature_selection')
        if sel_type == 'variance':
            num_steps.append(('variance_thresh', get_variance_threshold(threshold=0.0)))
            logger.debug(f"[{self.name}] Applying variance threshold feature selection")
        elif sel_type == 'k_best':
            num_steps.append(('k_best', get_select_k_best(k=10, score_func='f_regression')))
            logger.debug(f"[{self.name}] Applying SelectKBest (k=10) feature selection")

        numerical_pipeline = Pipeline(steps=num_steps)
        logger.debug(f"[{self.name}] Numerical pipeline: {len(num_steps)} steps")

        # 3. Categorical Pipeline
        # Import local OneHotEncoder for strict drop='first' compliance if needed, 
        # or use the factory if it supports it. Factory handles unknown='ignore'.
        # Here we prioritize robustness over strict dummy trap avoidance for production safety,
        # but since this is Linear Regression, drop='first' is statistically preferred.
        # We'll use manual construction for specificity here.
        from sklearn.preprocessing import OneHotEncoder
        cat_pipeline = Pipeline(steps=[
            ('imputer', get_imputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
        ])

        # 4. ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, X.select_dtypes(include=np.number).columns.tolist()),
                ('cat', cat_pipeline, X.select_dtypes(include=['object', 'category']).columns.tolist())
            ],
            remainder='passthrough',
            n_jobs=self.config.get('n_jobs', -1)
        )

        self.preprocessor = preprocessor
        
        X_transformed = preprocessor.fit_transform(X, y)
        X_transformed = np.asarray(X_transformed)
        y_transformed = y.values if hasattr(y, 'values') else np.asarray(y)

        return X_transformed, y_transformed, preprocessor

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> None:
        """Trains the Linear Regression model using a Unified Pipeline.
        
        Incorporates GridSearchCV for hyperparameter optimization within the pipeline
        to prevent data leakage during preprocessing.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (np.ndarray): Training targets.
        """
        logger.info(f"[{self.name}] Starting training...")
        logger.debug(f"[{self.name}] Training data shape: X={X_train.shape}, y={y_train.shape}")
        
        if self.preprocessor is None:
             raise RuntimeError("Preprocessor not initialized. Call preprocess() before fit().")

        # 1. Pipeline Construction
        preprocessor_template = clone(self.preprocessor)
        pipe = Pipeline(steps=[
            ('pre', preprocessor_template),
            ('model', self.model_instance)
        ])

        # 2. Adjust Param Grid for Pipeline
        pipeline_param_grid = {f'model__{k}': v for k, v in self.param_grid.items()}

        # 3. K-Fold Cross Validation
        cv_strategy = KFold(
            n_splits=self.config.get('cv_folds', 5), 
            shuffle=True, 
            random_state=self.config.get('random_state', 42)
        )

        # 4. GridSearch on THE PIPELINE
        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=pipeline_param_grid,
            scoring='neg_mean_squared_error',
            cv=cv_strategy,
            n_jobs=self.config.get('n_jobs', -1),
            verbose=0
        )

        grid_search.fit(X_train, y_train)
        
        self.best_estimator = grid_search.best_estimator_
        # Store the grid_search object in self.model for parameter access if needed, 
        # though strictly self.model in base was generic.
        self.model = grid_search 
        
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_
        logger.info(f"✅ [{self.name}] Training complete")
        logger.info(f"[{self.name}] Best CV Score: {-best_score:.4f} (MSE)")
        logger.debug(f"[{self.name}] Best params: {best_params}")

    def calculate_metrics(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """Calculates regression performance metrics.
        
        Args:
            X_test (pd.DataFrame): Test features.
            y_test (np.ndarray): Test targets.

        Returns:
            Dict[str, float]: RMSE, MAE, R2, Adjusted R2.
        """
        if self.best_estimator is None:
             raise RuntimeError("Model must be fitted before calculating metrics.")

        y_pred = self.best_estimator.predict(X_test)
        
        metrics = {}
        metrics['RMSE'] = calculate_rmse(y_test, y_pred)
        metrics['MAE'] = calculate_mae(y_test, y_pred)
        metrics['R2 Score'] = calculate_r2_score(y_test, y_pred)
        
        # Access the processed data shape for adjusted R2
        n_samples = X_test.shape[0]
        # Transform strictly to get feature count
        n_features = self.best_estimator.named_steps['pre'].transform(X_test).shape[1]
        metrics['Adjusted R2'] = calculate_adjusted_r2(y_test, y_pred, n_samples, n_features)
        
        return metrics

    def get_diagnostic_data(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """Retrieves diagnostic data for visualization.
        
        Args:
            X_test (pd.DataFrame): Test features.
            y_test (np.ndarray): Test targets.

        Returns:
            Dict[str, Any]: Predictions, actuals, coefficients, and model name.
        """
        if self.best_estimator is None:
             raise RuntimeError("Model must be fitted before diagnostics.")
             
        y_pred = self.best_estimator.predict(X_test)
        final_model = self.best_estimator.named_steps['model']
        
        return {
            'y_pred': y_pred,
            'y_test': y_test,
            'coefficients': final_model.coef_ if hasattr(final_model, 'coef_') else None,
            'model_name': self.name
        }

    def get_feature_importance(self) -> Dict[str, Any]:
        """Retrieves feature importance based on model coefficients.
        
        Returns:
            Dict[str, Any]: Dictionary containing list of importances (coefficients).
        """
        if self.best_estimator is None:
            return {}
            
        final_model = self.best_estimator.named_steps['model']
        if hasattr(final_model, 'coef_'):
            return {'importances': final_model.coef_.tolist()}
        return {}

    def get_parameter_descriptions(self) -> Dict[str, Dict[str, str]]:
        """Returns descriptions of the most important tuned parameters.
        
        Returns:
            Dict[str, Dict[str, str]]: Parameter descriptions.
        """
        if hasattr(self.model, 'best_params_'):
            best_params = self.model.best_params_
            return {
                'alpha': {
                    'value': str(best_params.get('model__alpha')),
                    'desc': 'Regularization strength. Higher values increase the penalty for complex models.'
                },
                'l1_ratio': {
                    'value': str(best_params.get('model__l1_ratio')),
                    'desc': 'Balance between L1 (Lasso) and L2 (Ridge) regularization.'
                }
            }
        return {}
