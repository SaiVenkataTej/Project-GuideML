import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
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
    A concrete implementation of K-Nearest Neighbors (KNN) for Classification and Regression.
    """
    def __init__(self, is_classification: bool = True, config: Dict[str, Any] = CONFIG):
        
        task_name = "Classification" if is_classification else "Regression"
        name = f"KNN ({task_name})"
        super().__init__(name=name, config=config)
        
        self.is_classification = is_classification
        
        # Initialize Model Instance
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
        Trains using GridSearchCV and appropriate CV.
        """
        if self.is_classification:
            cv = StratifiedKFold( # Stratified K-Fold (Req)
                n_splits=self.config.get('cv_folds', 5),
                shuffle=True,
                random_state=self.config.get('random_state', 42)
            )
            scoring = 'accuracy' # Optimize for Accuracy 
        else:
            cv = KFold(
                n_splits=self.config.get('cv_folds', 5),
                shuffle=True,
                random_state=self.config.get('random_state', 42)
            )
            scoring = 'neg_mean_absolute_error' # Optimize for MAE (Req)

        self.model = GridSearchCV( # GridSearchCV (Req)
            estimator=self.model_instance,
            param_grid=self.param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=self.config.get('n_jobs', -1),
            verbose=1
        )

        print(f"[{self.name}] Starting GridSearchCV...")
        self.model.fit(X_train, y_train)
        
        self.best_estimator = self.model.best_estimator_
        print(f"[{self.name}] Best parameters: {self.model.best_params_}")

    def calculate_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Calculates Accuracy/MAE and Prediction Latency.
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
        Returns data for diagnostics (Elbow Plot, Decision Boundary).
        """
        if not hasattr(self, 'best_estimator'):
             raise RuntimeError("Model must be fitted before diagnostics.")
        
        # Decision Boundary Data (for 2D projection later)
        y_pred = self.best_estimator.predict(X_test)

        # Local Neighbor Inspection (Req)
        # Find neighbors for a few test samples (first 5)
        distances, indices = self.best_estimator.kneighbors(X_test[:5])

        # Elbow Plot Data (Error vs K) from GridSearchCV results
        # accessing cv_results_
        results_df = pd.DataFrame(self.model.cv_results_)
        # Filter for relevant columns
        elbow_data = results_df[['param_estimator__n_neighbors', 'mean_test_score', 'std_test_score']]
        # Group by neighbor count if multiple other params
        # But we also have weights/metric. 
        # Ideally, we show the curve for the BEST other params.
        
        return {
            'y_pred': y_pred,
            'y_test': y_test,
            'model_name': self.name,
            'elbow_data': elbow_data.to_dict(orient='records'),
            'neighbor_indices': indices.tolist(),
            'neighbor_distances': distances.tolist(),
            'best_k': self.model.best_params_.get('estimator__n_neighbors')
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        KNN doesn't provide global feature importance.
        """
        return {}
