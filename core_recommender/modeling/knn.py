import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import optuna
import time

# Import centralized logger
from core_recommender.logger import get_logger

logger = get_logger(__name__)

# --- PROJECT IMPORTS ---
from core_recommender.modeling.baseModel import BaseModel
from core_recommender.preprocessing import (
    get_imputer,
    get_one_hot_encoder,
    get_standard_scaler,
    get_minmax_scaler,
    get_pca_reducer,
    get_nca_reducer
)
from core_recommender.evaluation import (
    calculate_accuracy,
    calculate_f1_score,
    calculate_rmse,
    calculate_mae,
    measure_prediction_latency
)

# --- DEFAULT CONFIGURATION ---
CONFIG = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'scaler': 'minmax',        # 'standard', 'minmax'
    'reduction': 'pca',        # 'pca', 'nca', None
    'n_components': 0.95,      # float for var (PCA), int for components (NCA/PCA)
    'cv_folds': 5,
    'random_state': 42,
    'n_jobs': -1
}

# =========================================================================
# KNNModel Class
# =========================================================================

class KNNModel(BaseModel):
    """
    A concrete implementation of K-Nearest Neighbors (KNN) for both Classification and Regression tasks.
    
    Rationale:
    ----------
    - **Instance-Based Learning**: Makes no assumptions about the underlying data distribution (non-parametric).
    - **Distance Sensitivity**: Highly sensitive to feature scales, necessitating strict normalization.
    - **Curse of Dimensionality**: Performance degrades in high dimensions, making PCA/NCA integration critical.

    This model utilizes proximity-based predictions. It supports various distance metrics 
    (Euclidean, Manhattan, Minkowski) and weighting schemes (Uniform, Distance-weighted).
    """
    def __init__(self, is_classification: bool = True, config: Dict[str, Any] = CONFIG):
        """
        Initializes the KNN model with task-specific configurations.

        Args:
            is_classification: True for classification tasks, False for regression.
            config: Dictionary containing hyperparameters (e.g., 'n_neighbors', 'metric').
                    Defaults to the global CONFIG dictionary.
        """
        
        task_name = "Classification" if is_classification else "Regression"
        name = f"KNN ({task_name})"
        super().__init__(name=name, config=config)
        
        self.is_classification = is_classification
        
        # Initialize Model Instance (Scikit-Learn)
        if self.is_classification:
            self.model_instance = KNeighborsClassifier(n_jobs=config.get('n_jobs', -1))
        else:
            self.model_instance = KNeighborsRegressor(n_jobs=config.get('n_jobs', -1))

        self.param_grid = {
            'n_neighbors': config.get('n_neighbors', [3, 5, 7]),
            'weights': config.get('weights', ['uniform', 'distance']),
            'metric': config.get('metric', ['euclidean'])
        }

    def preprocess(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
        """
        Constructs and applies the feature pipeline optimized for KNN.
        
        Rationale:
        ----------
        - **Scaling is Mandatory**: KNN calculates distances between points. If one feature ranges from 0-1 and another from 0-1000, 
          the second will dominate the distance metric. Scaling ensures equal contribution.
        - **Dimensionality Reduction**: Removes noise and irrelevant features to improve neighbor quality.
        
        Pipeline Steps:
        1. Numerical: Median imputation. Scaling (Standard or MinMax). Dimensionality Reduction (PCA or NCA).
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
        logger.debug(f"[{self.name}] Entering preprocess()...")
        logger.debug(f"[{self.name}] Input shape: X={X.shape}, y={y.shape}")
        
        # 1. Pipeline Construction
        # ------------------------
        
        # Numerical Steps
        num_steps = []
        num_steps.append(('imputer', get_imputer(strategy='median'))) # Median Imput (Req)
        
        # Scaling (Req: MinMax or Standard)
        scaler_type = self.config.get('scaler', 'minmax')
        if scaler_type == 'standard':
            num_steps.append(('scaler', get_standard_scaler()))
            logger.debug(f"[{self.name}] Using StandardScaler for feature scaling")
        else:
            num_steps.append(('scaler', get_minmax_scaler()))
            logger.debug(f"[{self.name}] Using MinMaxScaler for feature scaling")

        # Dimensionality Reduction (Req: PCA or NCA)
        reduction_method = self.config.get('reduction', 'pca')
        n_components = self.config.get('n_components', 0.95)

        if reduction_method == 'nca':
            if self.is_classification:
                # NCA is supervised and requires y. Pipeline usually handles this if steps support fit(X, y).
                # Since we are building a step here, NCA(n_components) is fine.
                # However, NCA expects integer components, not float variance ratio.
                n_comps_nca = n_components if isinstance(n_components, int) else None 
                num_steps.append(('nca', get_nca_reducer(n_components=n_comps_nca, random_state=self.config.get('random_state', 42))))
                logger.debug(f"[{self.name}] Using NCA for dimensionality reduction (n_components={n_comps_nca})")
            else:
                # Fallback to PCA for Regression if NCA requested (NCA is supervised classif mostly)
                logger.warning("⚠️ NCA is for classification only. Falling back to PCA for regression task.")
                num_steps.append(('pca', get_pca_reducer(n_components=n_components)))
                logger.debug(f"[{self.name}] Fallback: Using PCA for dimensionality reduction (n_components={n_components})")
                
        elif reduction_method == 'pca':
            num_steps.append(('pca', get_pca_reducer(n_components=n_components)))
            logger.debug(f"[{self.name}] Using PCA for dimensionality reduction (n_components={n_components})")

        numerical_pipeline = Pipeline(steps=num_steps)

        # Categorical Steps
        cat_steps = []
        cat_steps.append(('imputer', get_imputer(strategy='most_frequent'))) 
        # OneHot Encoding (Req)
        cat_steps.append(('onehot', get_one_hot_encoder(handle_unknown='ignore', sparse_output=False)))
        
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

        self.preprocessor = preprocessor
        logger.debug(f"[{self.name}] Preprocessor pipeline constructed.")

        # We return fitted for initial summary, but fit() will clone and re-fit.
        X_transformed = preprocessor.fit_transform(X, y)
        X_transformed = np.asarray(X_transformed)
        logger.debug(f"[{self.name}] Features transformed. New shape: {X_transformed.shape}")

        # 4. Target Processing (Skip if already numeric/pre-encoded by Executor)
        if self.is_classification:
            if not np.issubdtype(y.dtype, np.number):
                le = LabelEncoder()
                y_transformed = le.fit_transform(y)
                self.label_encoder = le
                logger.debug(f"[{self.name}] Target variable LabelEncoded.")
            else:
                y_transformed = y.values
                self.label_encoder = None
                logger.debug(f"[{self.name}] Target variable already numeric.")
        else:
            y_transformed = y.values
            self.label_encoder = None
            logger.debug(f"[{self.name}] Target variable for regression (no encoding).")

        logger.debug(f"[{self.name}] Preprocessing complete. Output shapes: X={X_transformed.shape}, y={y_transformed.shape}")
        return X_transformed, y_transformed, preprocessor

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """
        Trains the KNN model using a Unified Pipeline to prevent data leakage.
        """
        logger.info(f"[{self.name}] Starting training...")
        logger.debug(f"[{self.name}] Training data shape: X={X_train.shape}, y={y_train.shape}")
        
        from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
        from sklearn.base import clone
        import optuna

        # 1. Setup Cross-Validation
        if self.is_classification:
            cv = StratifiedKFold(n_splits=self.config.get('cv_folds', 5), shuffle=True, random_state=self.config.get('random_state', 42))
            scoring = 'accuracy'
            base_cls = KNeighborsClassifier
            logger.debug(f"[{self.name}] Classification task: Using StratifiedKFold with scoring='{scoring}'")
        else:
            cv = KFold(n_splits=self.config.get('cv_folds', 5), shuffle=True, random_state=self.config.get('random_state', 42))
            scoring = 'neg_mean_absolute_error'
            base_cls = KNeighborsRegressor
            logger.debug(f"[{self.name}] Regression task: Using KFold with scoring='{scoring}'")

        # 2. Pipeline-based Tuning
        preprocessor_template = clone(self.preprocessor)
        logger.debug(f"[{self.name}] Cloned preprocessor for pipeline-based tuning.")
        
        def pipeline_objective(trial):
            params = self._get_optuna_space(trial)
            model_inst = base_cls(**params, n_jobs=self.config.get('n_jobs', -1))
            
            pipe = Pipeline(steps=[
                ('pre', preprocessor_template),
                ('model', model_inst)
            ])
            
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=self.config.get('n_jobs', -1))
            return scores.mean()

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logger.debug(f"[{self.name}] Starting Optuna optimization (n_trials={self.config.get('n_trials', 15)})...")
        study = optuna.create_study(direction='maximize')
        study.optimize(pipeline_objective, n_trials=self.config.get('n_trials', 15))
        
        # 3. Build Best Model
        best_params = study.best_params
        best_model_inst = base_cls(**best_params, n_jobs=self.config.get('n_jobs', -1))
        
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
        """Defines the search space for Optuna."""
        neighbors_conf = self.param_grid.get('n_neighbors', [3, 5, 7])
        n_min, n_max = min(neighbors_conf), max(neighbors_conf)
        
        return {
            'n_neighbors': trial.suggest_int('n_neighbors', n_min, max(n_max, 15)),
            'weights': trial.suggest_categorical('weights', self.param_grid.get('weights', ['uniform', 'distance'])),
            'metric': trial.suggest_categorical('metric', self.param_grid.get('metric', ['euclidean']))
        }

    def calculate_metrics(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """
        Calculates performance metrics using the full Pipeline.
        """
        y_pred = self.best_estimator.predict(X_test)
        
        metrics = {}
        
        # Latency (Access model directly for latency test if needed, or pipe)
        metrics['Prediction Latency (s)'] = measure_prediction_latency(self.best_estimator, X_test)

        if self.is_classification:
            metrics['Accuracy'] = calculate_accuracy(y_test, y_pred) 
            metrics['F1 Score'] = calculate_f1_score(y_test, y_pred, average='weighted')
            metrics['MAE'] = calculate_mae(y_test, y_pred) 
        else:
            metrics['MAE'] = calculate_mae(y_test, y_pred) 
            metrics['RMSE'] = calculate_rmse(y_test, y_pred)
        
        return metrics

    def get_diagnostic_data(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Retrieves diagnostic data for visualization using the Pipeline.
        """
        if not hasattr(self, 'best_estimator'):
             raise RuntimeError("Model must be fitted before diagnostics.")
        
        y_pred = self.best_estimator.predict(X_test)
        
        # Access steps
        final_model = self.best_estimator.named_steps['model']
        pre_step = self.best_estimator.named_steps['pre']
        X_test_proc = pre_step.transform(X_test)

        # Local Neighbor Inspection
        distances, indices = final_model.kneighbors(X_test_proc[:5])

        # Elbow Plot Data
        elbow_data = []
        best_k = None
        
        if hasattr(self, 'study'):
            trials_df = self.study.trials_dataframe()
            if 'params_n_neighbors' in trials_df.columns:
                elbow_df = trials_df[['params_n_neighbors', 'value']].rename(
                    columns={'params_n_neighbors': 'param_estimator__n_neighbors', 'value': 'mean_test_score'}
                )
                elbow_df['std_test_score'] = 0.0 
                elbow_data = elbow_df.sort_values(by='param_estimator__n_neighbors').to_dict(orient='records')
            
            best_k = self.study.best_params.get('n_neighbors')
        
        return {
            'y_pred': y_pred,
            'y_test': y_test,
            'model_name': self.name,
            'elbow_data': elbow_data,
            'neighbor_indices': indices.tolist(),
            'neighbor_distances': distances.tolist(),
            'best_k': best_k
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        KNN does not provide global feature importance scores as it is a distance-based, 
        instance-based learning algorithm (lazy learner).
        """
        return {}

    def get_parameter_descriptions(self) -> Dict[str, Dict[str, str]]:
        """
        Returns descriptions of the most important tuned parameters.
        """
        final_model = self.best_estimator.named_steps['model']
        params = final_model.get_params()
        descriptions = {
            'n_neighbors': {
                'value': str(params.get('n_neighbors')),
                'desc': 'The number of nearest neighbors used to make a prediction. Choosing the right value helps balance noise vs. local detail.'
            },
            'metric': {
                'value': str(params.get('metric')),
                'desc': 'The distance metric used to calculate similarity between points (e.g., Euclidean or Manhattan distance).'
            },
            'weights': {
                'value': str(params.get('weights')),
                'desc': 'How neighbors are weighted. "Uniform" treats all neighbors equally, while "distance" gives more weight to closer points.'
            }
        }
        return descriptions
