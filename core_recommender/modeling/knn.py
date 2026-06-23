import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Union

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
import optuna

# Import centralized logger
from core_recommender.logger import get_logger
logger = get_logger(__name__)

# --- PROJECT IMPORTS ---
from core_recommender.modeling.baseModel import BaseModel
from core_recommender.modeling.registry import register_model
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
    'n_neighbors': [3, 5, 7, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'scaler': 'minmax',        # 'standard', 'minmax'
    'reduction': 'pca',        # 'pca', 'nca', None
    'n_components': 0.95,      # float for var (PCA), int for components (NCA/PCA)
    'cv_folds': 3,
    'random_state': 42,
    'n_jobs': -1,
    'n_trials': 10
}

# =========================================================================
# KNNModel Class
# =========================================================================

@register_model(task='both')
class KNNModel(BaseModel):
    """A concrete implementation of K-Nearest Neighbors (KNN) for Classification and Regression.
    
    Supports various distance metrics and weighting schemes. Critically requires strict
    feature scaling and often benefits from dimensionality reduction (PCA/NCA).
    
    Attributes:
        is_classification (bool): Flag indicating the task type.
        label_encoder (LabelEncoder): Encoder for target variable (Classification only).
        best_estimator (Pipeline): The fitted pipeline after Optuna tuning.
        study (optuna.Study): The Optuna study object containing trial history.
    """
    
    def __init__(self, is_classification: bool = True, config: Dict[str, Any] = CONFIG) -> None:
        """Initializes the KNN model.

        Args:
            is_classification (bool): True for classification, False for regression. Defaults to True.
            config (Dict[str, Any], optional): Hyperparameters and settings. Defaults to GLOBAL config.
        """
        task_name = "Classification" if is_classification else "Regression"
        name = f"KNN ({task_name})"
        super().__init__(name=name, config=config)
        
        self.is_classification = is_classification
        self.label_encoder: Optional[LabelEncoder] = None
        self.best_estimator: Optional[Pipeline] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        
        # Initialize Base Estimator
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
        """Constructs and applies the feature pipeline optimized for KNN.
        
        Pipeline Steps:
        1. Numerical: Median Imputation -> Scaling (MinMax/Std) -> Dimensionality Reduction (PCA/NCA).
        2. Categorical: Most Frequent Imputation -> One-Hot Encoding.
        
        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, ColumnTransformer]: Transformed X, y, and fitted preprocessor.
        """
        logger.debug(f"[{self.name}] Entering preprocess()...")
        logger.debug(f"[{self.name}] Input shape: X={X.shape}, y={y.shape}")
        
        # 1. Numerical Pipeline
        num_steps: List[Tuple[str, Any]] = []
        num_steps.append(('imputer', get_imputer(strategy='median')))
        
        # Scaling
        scaler_type = self.config.get('scaler', 'minmax')
        if scaler_type == 'standard':
            num_steps.append(('scaler', get_standard_scaler()))
            logger.debug(f"[{self.name}] Using StandardScaler")
        else:
            num_steps.append(('scaler', get_minmax_scaler()))
            logger.debug(f"[{self.name}] Using MinMaxScaler")

        # Dimensionality Reduction
        reduction_method = self.config.get('reduction', 'pca')
        n_components = self.config.get('n_components', 0.95)

        if reduction_method == 'nca':
            if self.is_classification:
                # NCA requires integer components
                n_comps_nca = n_components if isinstance(n_components, int) else None 
                num_steps.append(('nca', get_nca_reducer(
                    n_components=n_comps_nca, 
                    random_state=self.config.get('random_state', 42)
                )))
                logger.debug(f"[{self.name}] Using NCA (n_components={n_comps_nca})")
            else:
                logger.warning("⚠️ NCA is for classification only. Falling back to PCA.")
                num_steps.append(('pca', get_pca_reducer(n_components=n_components)))
                logger.debug(f"[{self.name}] Fallback: Using PCA (n_components={n_components})")
                
        elif reduction_method == 'pca':
            num_steps.append(('pca', get_pca_reducer(n_components=n_components)))
            logger.debug(f"[{self.name}] Using PCA (n_components={n_components})")

        numerical_pipeline = Pipeline(steps=num_steps)

        # 2. Categorical Pipeline
        cat_steps: List[Tuple[str, Any]] = []
        cat_steps.append(('imputer', get_imputer(strategy='most_frequent'))) 
        cat_steps.append(('onehot', get_one_hot_encoder(handle_unknown='ignore', sparse_output=False)))
        
        cat_pipeline = Pipeline(steps=cat_steps)

        # 3. ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, X.select_dtypes(include=np.number).columns.tolist()),
                ('cat', cat_pipeline, X.select_dtypes(include=['object', 'category']).columns.tolist())
            ],
            remainder='drop',
            n_jobs=self.config.get('n_jobs', -1)
        )

        self.preprocessor = preprocessor
        
        X_transformed = preprocessor.fit_transform(X, y)
        X_transformed = np.asarray(X_transformed)
        
        # 4. Target Processing
        if self.is_classification:
            if not np.issubdtype(y.dtype, np.number):
                le = LabelEncoder()
                y_transformed = le.fit_transform(y)
                self.label_encoder = le
            else:
                y_transformed = y.values
                self.label_encoder = None
        else:
            y_transformed = y.values
            self.label_encoder = None

        return X_transformed, y_transformed, preprocessor

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> None:
        """Trains the KNN model using pipeline-based Optuna optimization.
        
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (np.ndarray): Training targets.
        """
        logger.info(f"[{self.name}] Starting training...")
        
        if self.preprocessor is None:
             raise RuntimeError("Preprocessor not initialized.")

        # 1. Setup Cross-Validation
        if self.is_classification:
            cv = StratifiedKFold(n_splits=self.config.get('cv_folds', 5), shuffle=True, random_state=self.config.get('random_state', 42))
            scoring = 'accuracy'
            base_cls = KNeighborsClassifier
        else:
            cv = KFold(n_splits=self.config.get('cv_folds', 5), shuffle=True, random_state=self.config.get('random_state', 42))
            scoring = 'neg_mean_absolute_error'
            base_cls = KNeighborsRegressor

        # 2. Pipeline-based Tuning
        preprocessor_template = clone(self.preprocessor)
        
        def pipeline_objective(trial):
            params = self._get_optuna_space(trial)
            model_inst = base_cls(**params, n_jobs=self.config.get('n_jobs', -1))
            
            pipe = Pipeline(steps=[
                ('pre', preprocessor_template),
                ('model', model_inst)
            ])
            
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)  # Sequential CV to avoid nested parallelism
            return scores.mean()

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logger.debug(f"[{self.name}] Starting Optuna optimization...")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(pipeline_objective, n_trials=self.config.get('n_trials', 10))
        
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
        
        logger.info(f"✅ [{self.name}] Training complete. Best Score: {study.best_value:.4f}")

    def _get_optuna_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Defines the search space for Optuna."""
        neighbors_conf = self.param_grid.get('n_neighbors', [3, 5, 7])
        n_min, n_max = min(neighbors_conf), max(neighbors_conf)
        
        return {
            'n_neighbors': trial.suggest_int('n_neighbors', n_min, max(n_max, 15)),
            'weights': trial.suggest_categorical('weights', self.param_grid.get('weights', ['uniform', 'distance'])),
            'metric': trial.suggest_categorical('metric', self.param_grid.get('metric', ['euclidean']))
        }

    def calculate_metrics(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """Calculates performance metrics.
        
        Args:
            X_test (pd.DataFrame): Test features.
            y_test (np.ndarray): Test targets.

        Returns:
            Dict[str, float]: Performance metrics (Accuracy/F1 for Classif, MAE/RMSE for Reg).
        """
        if self.best_estimator is None:
             raise RuntimeError("Model must be fitted before calculating metrics.")
             
        y_pred = self.best_estimator.predict(X_test)
        metrics = {}
        
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
        """Retrieves diagnostic data for visualization.
        
        Includes neighborhood inspection data and elbow plot data if available.
        """
        if self.best_estimator is None:
             raise RuntimeError("Model must be fitted before diagnostics.")
        
        y_pred = self.best_estimator.predict(X_test)
        final_model = self.best_estimator.named_steps['model']
        pre_step = self.best_estimator.named_steps['pre']
        X_test_proc = pre_step.transform(X_test)

        # Local Neighbor Inspection (for first 5 samples)
        distances, indices = final_model.kneighbors(X_test_proc[:5])
        
        # PROBA for Classification
        y_proba = None
        if self.is_classification and hasattr(final_model, 'predict_proba'):
             y_proba = final_model.predict_proba(X_test_proc)

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
            'y_true': y_test,
            'y_proba': y_proba,
            'model_name': self.name,
            'elbow_data': elbow_data,
            'neighbor_indices': indices.tolist(),
            'neighbor_distances': distances.tolist(),
            'best_k': best_k
        }
    
    def get_tailored_diagnostics(self) -> Dict[str, Any]:
        """KNN-specific diagnostics: elbow/tuning data and neighbourhood inspection.

        Returns:
            Dict[str, Any]:
                * ``elbow_data``         — per-trial (k, score) pairs from Optuna study
                  for elbow-plot visualisation.
                * ``best_k``             — the optimal number of neighbours chosen.
                * ``neighbor_indices``   — neighbour indices for the first 5 test samples.
                * ``neighbor_distances`` — corresponding distances (neighbourhood bounds).
        """
        if self.best_estimator is None:
            return {}

        final_model = self.best_estimator.named_steps['model']
        pre_step = self.best_estimator.named_steps['pre']

        diagnostics: Dict[str, Any] = {}

        # --- Elbow / tuning history ---
        if hasattr(self, 'study'):
            trials_df = self.study.trials_dataframe()
            if 'params_n_neighbors' in trials_df.columns:
                elbow_df = (
                    trials_df[['params_n_neighbors', 'value']]
                    .rename(columns={
                        'params_n_neighbors': 'param_estimator__n_neighbors',
                        'value': 'mean_test_score'
                    })
                )
                elbow_df['std_test_score'] = 0.0
                diagnostics['elbow_data'] = (
                    elbow_df.sort_values('param_estimator__n_neighbors')
                    .to_dict(orient='records')
                )
            diagnostics['best_k'] = self.study.best_params.get('n_neighbors')

        return diagnostics

    def get_parameter_descriptions(self) -> Dict[str, Dict[str, str]]:
        """Returns descriptions of the most important tuned parameters."""
        if self.best_estimator is None:
             return {}
             
        final_model = self.best_estimator.named_steps['model']
        params = final_model.get_params()
        
        return {
            'n_neighbors': {
                'value': str(params.get('n_neighbors')),
                'desc': 'The number of nearest neighbors used to make a prediction.'
            },
            'metric': {
                'value': str(params.get('metric')),
                'desc': 'The distance metric used (e.g., Euclidean).'
            },
            'weights': {
                'value': str(params.get('weights')),
                'desc': 'How neighbors are weighted (Uniform vs Distance).'
            }
        }
