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
    
    This model utilizes proximity-based predictions. It supports various distance metrics 
    (Euclidean, Manhattan, Minkowski) and weighting schemes (Uniform, Distance-weighted).
    Dimensionality reduction (PCA/NCA) is integrated to mitigate the "Curse of Dimensionality".
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
        # 1. Pipeline Construction
        # ------------------------
        
        # Numerical Steps
        num_steps = []
        num_steps.append(('imputer', get_imputer(strategy='median'))) # Median Imput (Req)
        
        # Scaling (Req: MinMax or Standard)
        if self.config.get('scaler') == 'standard':
            num_steps.append(('scaler', get_standard_scaler()))
        else:
             num_steps.append(('scaler', get_minmax_scaler()))

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
            else:
                # Fallback to PCA for Regression if NCA requested (NCA is supervised classif mostly)
                print("Warning: NCA is for classification. Using PCA for regression.")
                num_steps.append(('pca', get_pca_reducer(n_components=n_components)))
                
        elif reduction_method == 'pca':
            num_steps.append(('pca', get_pca_reducer(n_components=n_components)))

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

        # 3. Fit-Transform
        # ----------------
        # NCA needs y!
        # preprocessor.fit_transform(X, y) will pass y to underlying steps if they accept it.
        # ColumnTransformer passes y to steps. Pipeline passes y to steps.
        # So NCA step will receive y.
        X_transformed = preprocessor.fit_transform(X, y)
        X_transformed = np.asarray(X_transformed)

        # 4. Target Processing
        # --------------------
        if self.is_classification:
            le = LabelEncoder()
            y_transformed = le.fit_transform(y)
            self.label_encoder = le
        else:
            y_transformed = y.values
            self.label_encoder = None

        return X_transformed, y_transformed, preprocessor

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the KNN model using the configured optimization strategy.
        Delegates to core_recommender.tuning for modular execution.
        
        Args:
            X_train: Training features array.
            y_train: Training target array.
        """
        from core_recommender.tuning import (
            get_grid_search_tuner, 
            run_optuna_optimization
        )

        # 1. Setup Cross-Validation
        if self.is_classification:
            cv = StratifiedKFold(
                n_splits=self.config.get('cv_folds', 5),
                shuffle=True,
                random_state=self.config.get('random_state', 42)
            )
            scoring = 'accuracy'
        else:
            cv = KFold(
                n_splits=self.config.get('cv_folds', 5),
                shuffle=True,
                random_state=self.config.get('random_state', 42)
            )
            scoring = 'neg_mean_absolute_error'

        # 2. Select & Run Strategy
        strategy = self.config.get('tuning_strategy', 'optuna')
        print(f"[{self.name}] Starting Training with {strategy.upper()} strategy...")

        if strategy == 'optuna':
            # Use the functional executor from tuning.py
            self.best_estimator = run_optuna_optimization(
                estimator_class=KNeighborsClassifier if self.is_classification else KNeighborsRegressor,
                param_space_func=self._get_optuna_space,
                X=X_train,
                y=y_train,
                cv=cv,
                scoring=scoring,
                n_trials=self.config.get('n_trials', 20),
                n_jobs=self.config.get('n_jobs', -1),
                random_state=self.config.get('random_state', 42)
            )
            if hasattr(self.best_estimator, 'study_'):
                 self.study = self.best_estimator.study_

        elif strategy == 'grid':
            # Use the factory from tuning.py
            grid_search = get_grid_search_tuner(
                estimator=self.model_instance,
                param_grid=self.param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=self.config.get('n_jobs', -1)
            )
            grid_search.fit(X_train, y_train)
            self.best_estimator = grid_search.best_estimator_
            self.model = grid_search # For accessing cv_results_ if needed

        self.model = self.best_estimator

    def _get_optuna_space(self, trial):
        """Defines the search space for Optuna."""
        neighbors_conf = self.param_grid.get('n_neighbors', [3, 5, 7])
        n_min, n_max = min(neighbors_conf), max(neighbors_conf)
        
        return {
            'n_neighbors': trial.suggest_int('n_neighbors', n_min, max(n_max, 15)),
            'weights': trial.suggest_categorical('weights', self.param_grid.get('weights', ['uniform', 'distance'])),
            'metric': trial.suggest_categorical('metric', self.param_grid.get('metric', ['euclidean']))
        }

    def calculate_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Calculates performance metrics including Prediction Latency.

        Metrics Include:
        - Accuracy, F1 Score, MAE (Classification)
        - MAE, RMSE (Regression)
        - Prediction Latency (seconds)

        Args:
            X_test: Test features array.
            y_test: Test target array.
            
        Returns:
            Dict[str, float]: Dictionary of calculated metrics.
        """
        y_pred = self.best_estimator.predict(X_test)
        
        metrics = {}
        
        # Latency (Req)
        metrics['Prediction Latency (s)'] = measure_prediction_latency(self.best_estimator, X_test)

        if self.is_classification:
            metrics['Accuracy'] = calculate_accuracy(y_test, y_pred) # Req
            # Add F1 too as it's standard
            metrics['F1 Score'] = calculate_f1_score(y_test, y_pred, average='weighted')
            # Technically MAE can be calc for classif if encoded, but usually not primary.
            # User req table showed Accuracy and MAE.
            metrics['MAE'] = calculate_mae(y_test, y_pred) 
        else:
            metrics['MAE'] = calculate_mae(y_test, y_pred) # Req
            metrics['RMSE'] = calculate_rmse(y_test, y_pred)
        
        return metrics

    def get_diagnostic_data(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Retrieves diagnostic data for visualization.
        
        Includes data from Optuna study if available.
        """
        if not hasattr(self, 'best_estimator'):
             raise RuntimeError("Model must be fitted before diagnostics.")
        
        # Decision Boundary Data (for 2D projection later)
        y_pred = self.best_estimator.predict(X_test)

        # Local Neighbor Inspection (Req)
        # Find neighbors for a few test samples (first 5)
        distances, indices = self.best_estimator.kneighbors(X_test[:5])

        # Elbow Plot Data (for Visualization)
        elbow_data = []
        best_k = None
        
        if hasattr(self, 'study'):
            trials_df = self.study.trials_dataframe()
            # Visualize Score vs n_neighbors
            if 'params_n_neighbors' in trials_df.columns:
                elbow_df = trials_df[['params_n_neighbors', 'value']].rename(
                    columns={'params_n_neighbors': 'param_estimator__n_neighbors', 'value': 'mean_test_score'}
                )
                elbow_df['std_test_score'] = 0.0 
                elbow_data = elbow_df.sort_values(by='param_estimator__n_neighbors').to_dict(orient='records')
            
            best_k = self.study.best_params.get('n_neighbors')
        
        # If using GridSearch (fallback or explicit choice)
        elif hasattr(self.model, 'cv_results_'):
             results_df = pd.DataFrame(self.model.cv_results_)
             # Ensure 'param_estimator__n_neighbors' is present, or adapt if param_grid keys are used directly
             if 'param_n_neighbors' in results_df.columns: # For GridSearch, params are usually prefixed with 'param_'
                 elbow_df = results_df[['param_n_neighbors', 'mean_test_score', 'std_test_score']].rename(
                     columns={'param_n_neighbors': 'param_estimator__n_neighbors'}
                 )
                 elbow_data = elbow_df.sort_values(by='param_estimator__n_neighbors').to_dict(orient='records')
             best_k = self.model.best_params_.get('n_neighbors')


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

        Returns:
            Dict[str, float]: Empty dictionary.
        """
        return {}
