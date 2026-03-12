import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Union

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, KFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone

# --- PROJECT IMPORTS ---
from core_recommender.modeling.baseModel import BaseModel
from core_recommender.preprocessing import (
    get_imputer,
    get_one_hot_encoder,
    get_ordinal_encoder,
    get_variance_threshold,
    get_select_k_best
)
from core_recommender.evaluation import (
    calculate_accuracy,
    calculate_rmse,
    get_tree_depth,
    get_leaf_count,
    measure_prediction_latency
)

# Import centralized logger
from core_recommender.logger import get_logger
logger = get_logger(__name__)

# --- DEFAULT CONFIGURATION ---
CONFIG = {
    'criterion': 'gini',       
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'ccp_alpha': [0.0, 0.001, 0.01, 0.1], 
    'encoding': 'onehot',      
    'feature_selection': 'k_best', 
    'k_best': 10,
    'cv_folds': 3,
    'random_state': 42,
    'n_jobs': -1,
    'n_iter': 10
}

# =========================================================================
# DecisionTreeModel Class
# =========================================================================

class DecisionTreeModel(BaseModel):
    """A concrete implementation of Decision Trees for both Classification and Regression tasks.
    
    Attributes:
        is_classification (bool): Flag indicating task type.
        label_encoder (LabelEncoder): Encoder for target variable (Classification only).
        best_estimator (Pipeline): The fitted pipeline after tuning.
    """
    def __init__(self, is_classification: bool = True, config: Dict[str, Any] = CONFIG) -> None:
        """Initializes the Decision Tree model.

        Args:
            is_classification (bool, optional): True for classification. Defaults to True.
            config (Dict[str, Any], optional): Hyperparameters and settings. Defaults to GLOBAL config.
        """
        task_name = "Classification" if is_classification else "Regression"
        name = f"Decision Tree ({task_name})"
        super().__init__(name=name, config=config)
        
        self.is_classification = is_classification
        self.label_encoder: Optional[LabelEncoder] = None
        self.best_estimator: Optional[Pipeline] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        
        # Initialize Model Instance
        if self.is_classification:
            self.model_instance = DecisionTreeClassifier(random_state=config.get('random_state', 42))
        else:
            self.model_instance = DecisionTreeRegressor(random_state=config.get('random_state', 42))

        # Dynamic Param Grid based on task
        criterion_opts = config.get('criterion', ['gini'])
        if isinstance(criterion_opts, str): criterion_opts = [criterion_opts]
        
        if self.is_classification:
            valid_crit = [c for c in criterion_opts if c in ['gini', 'entropy', 'log_loss']]
            if not valid_crit: valid_crit = ['gini']
        else:
            valid_crit = [c for c in criterion_opts if c in ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']]
            if not valid_crit: valid_crit = ['squared_error']

        self.param_grid = {
            'max_depth': config.get('max_depth', [None, 10]),
            'min_samples_split': config.get('min_samples_split', [2, 5]),
            'min_samples_leaf': config.get('min_samples_leaf', [1, 2]),
            'ccp_alpha': config.get('ccp_alpha', [0.0, 0.01]), 
            'criterion': valid_crit
        }

    def preprocess(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
        """Constructs and applies the feature pipeline optimized for Decision Trees.
        
        Pipeline Steps:
        1. Numerical: Median imputation -> Feature Selection.
        2. Categorical: Most frequent imputation -> Encoding (OneHot/Ordinal).
        
        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, ColumnTransformer]: Transformed X, y, and fitted preprocessor.
        """
        logger.debug(f"[{self.name}] Entering preprocess()...")
        
        # 1. Numerical Pipeline
        num_steps: List[Tuple[str, Any]] = [('imputer', get_imputer(strategy='median'))]
        
        if self.config.get('feature_selection') == 'variance':
            num_steps.append(('variance_threshold', get_variance_threshold()))
            logger.debug(f"[{self.name}] Using VarianceThreshold")
        elif self.config.get('feature_selection') == 'k_best':
            score_func = 'f_classif' if self.is_classification else 'f_regression'
            num_steps.append(('select_k_best', get_select_k_best(k=self.config.get('k_best', 10), score_func=score_func)))
            logger.debug(f"[{self.name}] Using SelectKBest")

        numerical_pipeline = Pipeline(steps=num_steps)

        # 2. Categorical Pipeline
        cat_steps: List[Tuple[str, Any]] = [('imputer', get_imputer(strategy='most_frequent'))]
        
        if self.config.get('encoding') == 'ordinal':
            cat_steps.append(('ordinal', get_ordinal_encoder()))
            logger.debug(f"[{self.name}] Using Ordinal Encoding")
        else:
            cat_steps.append(('onehot', get_one_hot_encoder(handle_unknown='ignore', sparse_output=False)))
            logger.debug(f"[{self.name}] Using One-Hot Encoding")
        
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
        """Trains the Decision Tree model using a Unified Pipeline.
        
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (np.ndarray): Training targets.
        """
        logger.info(f"[{self.name}] Starting training...")
        
        if self.preprocessor is None:
             raise RuntimeError("Preprocessor not initialized.")

        # 1. Pipeline Construction
        preprocessor_template = clone(self.preprocessor)
        pipe = Pipeline(steps=[
            ('pre', preprocessor_template),
            ('model', self.model_instance)
        ])

        # 2. Adjust Param Grid
        pipeline_params = {f'model__{k}': v for k, v in self.param_grid.items()}

        # 3. CV Strategy
        if self.is_classification:
            cv = StratifiedKFold(n_splits=self.config.get('cv_folds', 5), shuffle=True, random_state=self.config.get('random_state', 42))
            scoring = 'accuracy'
        else:
            cv = KFold(n_splits=self.config.get('cv_folds', 5), shuffle=True, random_state=self.config.get('random_state', 42))
            scoring = 'neg_root_mean_squared_error'

        # 4. Randomized Search
        random_search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=pipeline_params,
            cv=cv,
            scoring=scoring,
            n_iter=self.config.get('n_iter', 10),
            n_jobs=self.config.get('n_jobs', -1),
            random_state=self.config.get('random_state', 42)
        )

        random_search.fit(X_train, y_train)
        
        self.best_estimator = random_search.best_estimator_
        self.model = random_search
        
        logger.info(f"✅ [{self.name}] Training complete. Best CV Score: {random_search.best_score_:.4f}")

    def calculate_metrics(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """Calculates performance metrics.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (np.ndarray): Test targets.

        Returns:
            Dict[str, float]: Tree complexity metrics + Accuracy/RMSE.
        """
        if self.best_estimator is None:
             raise RuntimeError("Model must be fitted before calculating metrics.")
             
        y_pred = self.best_estimator.predict(X_test)
        
        metrics = {}
        final_tree = self.best_estimator.named_steps['model']
        
        metrics['Tree Depth'] = get_tree_depth(final_tree)
        metrics['Leaf Count'] = get_leaf_count(final_tree)
        metrics['Prediction Latency (s)'] = measure_prediction_latency(self.best_estimator, X_test)

        if self.is_classification:
            metrics['Accuracy'] = calculate_accuracy(y_test, y_pred) 
        else:
            metrics['RMSE'] = calculate_rmse(y_test, y_pred) 
        
        return metrics

    def get_diagnostic_data(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """Retrieves diagnostic data for visualization.
        
        Includes Tree structure in Graphviz format.
        """
        if self.best_estimator is None:
             raise RuntimeError("Model must be fitted before diagnostics.")
        
        y_pred = self.best_estimator.predict(X_test)
        final_tree = self.best_estimator.named_steps['model']
        
        # Graphviz Source
        dot_data = export_graphviz(
            final_tree,
            out_file=None,
            filled=True,
            rounded=True,
            special_characters=True
        )

        return {
            'y_pred': y_pred,
            'y_true': y_test,
            'y_proba': self.best_estimator.predict_proba(X_test) if self.is_classification else None,
            'model_name': self.name,
            'feature_importances': self.get_feature_importance(),
            'tree_dot_data': dot_data
        }
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Retrieves the Gini Importance."""
        if self.best_estimator is None:
            return {}
            
        final_tree = self.best_estimator.named_steps['model']
        if hasattr(final_tree, 'feature_importances_'):
             # Convert array to list for JSON serialization
            return {'importances': final_tree.feature_importances_.tolist()}
        return {}

    def get_parameter_descriptions(self) -> Dict[str, Dict[str, str]]:
        """Returns descriptions of the most important tuned parameters."""
        if self.model and hasattr(self.model, 'best_params_'):
            best_params = self.model.best_params_
            return {
                'max_depth': {'value': str(best_params.get('model__max_depth')), 'desc': 'Maximum depth of the tree.'},
                'criterion': {'value': str(best_params.get('model__criterion')), 'desc': 'Function to measure split quality.'},
                'min_samples_leaf': {'value': str(best_params.get('model__min_samples_leaf')), 'desc': 'Minimum samples required at a leaf node.'}
            }
        return {}
