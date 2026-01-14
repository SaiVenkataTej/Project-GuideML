import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

# --- SKLEARN IMPORTS ---
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, GridSearchCV

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
from core_recommender.visualization import plot_coefficient_bar_chart

# Import centralized logger
from core_recommender.logger import get_logger
logger = get_logger(__name__)

# --- DEFAULT CONFIGURATION ---
CONFIG = {
    'scaler': 'standard',      # 'standard' or 'robust'
    'transformation': 'none',  # 'none', 'log', 'box-cox', 'yeo-johnson'
    'feature_selection': 'none', # 'none', 'variance', 'k_best'
    'cv_folds': 5,
    'random_state': 42,
    'n_jobs': -1
}

# =========================================================================
# LinearRegressionModel Class
# =========================================================================

class LinearRegressionModel(BaseModel):
    """
    A concrete implementation of Linear Regression optimized for Regression tasks.
    
    Rationale:
    ----------
    - **ElasticNet Base**: We use ElasticNet because it generalizes Ridge (L2) and Lasso (L1) regularization.
    - **Multicollinearity Handling**: Regularization (L1/L2) is crucial when features are correlated, which is common in automated pipelines.
    - **Interpretability**: Coefficients provide a direct measure of feature impact (Magnitude and Direction).

    This model utilizes ElasticNet, which generalizes Ridge (L2 penalty) and Lasso (L1 penalty)
    regularization. This allows for both variable selection and coefficient shrinkage.
    """
    def __init__(self, config: Dict[str, Any] = CONFIG):
        """
        Initializes the Linear Regression model with configurable regularization.

        Args:
            config: Dictionary containing hyperparameters (e.g., 'alpha', 'l1_ratio').
                    Defaults to the global CONFIG dictionary.
        """
        
        # We use ElasticNet as the base estimator because it generalizes Lasso (l1_ratio=1) 
        # and Ridge (l1_ratio=0), allowing us to tune both via GridSearchCV.
        # However, for pure OLS, one could use LinearRegression(). 
        # Given the requirements ask for Ridge/Lasso/ElasticNet, ElasticNet is the best cover-all.
        
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

    def preprocess(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
        """
        Constructs and applies the feature pipeline optimized for Linear Regression.
        
        Rationale:
        ----------
        - **Normality Assumption**: Linear models assume residuals are normally distributed. Transformations (Log/Box-Cox) help achieve this.
        - **Scaling**: Essential for regularization (L1/L2) so that penalties are applied uniformly across features.
        - **Dummy Trap**: One-Hot Encoding with `drop='first'` prevents perfect collinearity, which breaks the normal equation (though less critical with regularization).
        
        Pipeline Steps:
        1. Numerical: Median imputation. Transformations (Log, Box-Cox, Yeo-Johnson). 
           Robust or Standard Scaling. Feature Selection.
        2. Categorical: Most frequent imputation. One-Hot encoding (drop='first').
        
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
        
        # Edge Case: Check for NaNs in target y before proceeding
        if y.isna().any():
            logger.error(f"[{self.name}] Target variable contains {y.isna().sum()} NaN values")
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
        # Steps: Impute -> Transform (Log/Power) -> Scale -> Select
        num_steps = [
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
        # Steps: Impute -> OneHot (drop='first')
        cat_steps = [
            ('imputer', get_imputer(strategy='most_frequent')), # Use most_frequent for cats if needed, though median is spec'd for nums
            ('onehot', get_one_hot_encoder(handle_unknown='ignore', sparse_output=False)) # drop='first' needs to be set manually if strict about dummy trap
        ]
        # Note: get_one_hot_encoder factory defaults to handle_unknown='ignore' (safer for prod). 
        # Using OneHotEncoder in pipeline with drop='first' and handle_unknown='ignore' can be conflict prone in older sklearn,
        # but modern versions handle it. If STRICT adherence to 'drop=first' is needed:
        # We'd need to modify the factory or override here. The factory call is compatible.
        
        # To strictly satisfy "drop='first' to avoid dummy variable trap":
        # We manually construct OneHot because the factory function in preprocessing.py might not expose drop param (let's check).
        # Checking preprocessing.py... it takes handle_unknown and sparse_output. It doesn't take 'drop'.
        # So we import OneHotEncoder class directly or modify the factory? 
        # I'll instantiate OneHotEncoder directly here to meet the strict requirement.
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
        
        # We still return the fitted version for back-compat or initial extraction, 
        # but the fit() method will use a fresh clone.
        X_transformed = preprocessor.fit_transform(X, y)
        X_transformed = np.asarray(X_transformed)
        y_transformed = y.values if hasattr(y, 'values') else np.asarray(y)

        return X_transformed, y_transformed, preprocessor

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """
        Trains the Linear Regression model using a Unified Pipeline to prevent data leakage.
        """
        logger.info(f"[{self.name}] Starting training...")
        logger.debug(f"[{self.name}] Training data shape: X={X_train.shape}, y={y_train.shape}")
        
        from sklearn.model_selection import GridSearchCV
        from sklearn.base import clone

        # 1. Pipeline Construction
        # We wrap the unfitted preprocessor and model in one object
        preprocessor_template = clone(self.preprocessor)
        pipe = Pipeline(steps=[
            ('pre', preprocessor_template),
            ('model', self.model_instance)
        ])

        # 2. Adjust Param Grid
        # GridSearchCV needs parameter names prefixed with the step name (e.g., 'model__alpha')
        pipeline_param_grid = {f'model__{k}': v for k, v in self.param_grid.items()}

        # 3. K-Fold Cross Validation
        cv_strategy = KFold(
            n_splits=self.config.get('cv_folds', 5), 
            shuffle=True, 
            random_state=self.config.get('random_state', 42)
        )

        # 4. GridSearch on THE PIPELINE
        # This ensures preprocessing is re-fit in every CV fold (Zero Leakage)
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
        self.model = grid_search # Store the tuner for parameter access
        
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_
        logger.info(f"✅ [{self.name}] Training complete")
        logger.info(f"[{self.name}] Best CV Score: {-best_score:.4f} (MSE)")
        logger.debug(f"[{self.name}] Best params: {best_params}")

    def calculate_metrics(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """
        Calculates standard regression performance metrics using the full Pipeline.
        """
        # The pipeline handles raw X_test (pre -> model)
        y_pred = self.best_estimator.predict(X_test)
        
        metrics = {}
        metrics['RMSE'] = calculate_rmse(y_test, y_pred)
        metrics['MAE'] = calculate_mae(y_test, y_pred)
        metrics['R2 Score'] = calculate_r2_score(y_test, y_pred)
        
        # Access the processed data shape for adjusted R2
        # We can transform temporarily to get the feature count
        n_samples = X_test.shape[0]
        n_features = self.best_estimator.named_steps['pre'].transform(X_test).shape[1]
        metrics['Adjusted R2'] = calculate_adjusted_r2(y_test, y_pred, n_samples, n_features)
        
        return metrics

    def get_diagnostic_data(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Retrieves diagnostic data for visualization using the Pipeline.
        """
        if not hasattr(self, 'best_estimator'):
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
        """
        Retrieves feature importance based on model coefficients.
        """
        final_model = self.best_estimator.named_steps['model']
        if hasattr(final_model, 'coef_'):
            return {'importances': final_model.coef_.tolist()}
        return {}

    def get_parameter_descriptions(self) -> Dict[str, Dict[str, str]]:
        """
        Returns descriptions of the most important tuned parameters.
        """
        # Access best params from GridSearch results
        best_params = self.model.best_params_
        descriptions = {
            'alpha': {
                'value': str(best_params.get('model__alpha')),
                'desc': 'Regularization strength. Higher values increase the penalty for complex models, helping to prevent overfitting.'
            },
            'l1_ratio': {
                'value': str(best_params.get('model__l1_ratio')),
                'desc': 'The balance between L1 (Lasso) and L2 (Ridge) regularization. 1.0 is full Lasso, 0.0 is full Ridge.'
            }
        }
        return descriptions

