import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder

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
    get_leaf_count
)

# --- DEFAULT CONFIGURATION ---
CONFIG = {
    'criterion': 'gini',       # 'gini', 'entropy' (classif) / 'squared_error' (reg)
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'ccp_alpha': [0.0, 0.001, 0.01, 0.1], # Cost-Complexity Pruning
    'encoding': 'onehot',      # 'onehot', 'ordinal'
    'feature_selection': 'k_best', # 'k_best', 'variance', None
    'k_best': 10,
    'cv_folds': 5,
    'random_state': 42,
    'n_jobs': -1
}

# =========================================================================
# DecisionTreeModel Class
# =========================================================================

class DecisionTreeModel(BaseModel):
    """
    A concrete implementation of Decision Trees for both Classification and Regression tasks.
    
    This model supports Cost-Complexity Pruning (CCP) to control overfitting and 
    integrates specialized tree metrics such as depth and leaf count. It handles 
    data preprocessing, training, and evaluation within the standardized pipeline.
    """
    def __init__(self, is_classification: bool = True, config: Dict[str, Any] = CONFIG):
        """
        Initializes the Decision Tree model with task-specific configurations.

        Args:
            is_classification: True for classification tasks, False for regression.
            config: Dictionary containing hyperparameters (e.g., 'max_depth', 'ccp_alpha').
                    Defaults to the global CONFIG dictionary.
        """
        
        task_name = "Classification" if is_classification else "Regression"
        name = f"Decision Tree ({task_name})"
        super().__init__(name=name, config=config)
        
        self.is_classification = is_classification
        
        # Initialize Model Instance (Scikit-Learn)
        if self.is_classification:
            self.model_instance = DecisionTreeClassifier(random_state=config.get('random_state', 42))
        else:
            self.model_instance = DecisionTreeRegressor(random_state=config.get('random_state', 42))

        # Dynamic Param Grid based on task
        criterion_opts = config.get('criterion', ['gini'])
        if isinstance(criterion_opts, str): criterion_opts = [criterion_opts]
        
        # Filter criterion based on task compatibility if generic list passed
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
            'ccp_alpha': config.get('ccp_alpha', [0.0, 0.01]), # Pruning
            'criterion': valid_crit
        }

    def preprocess(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
        """
        Constructs and applies the feature pipeline optimized for Decision Trees.
        
        Pipeline Steps:
        1. Numerical: Median imputation. Variance threshold or SelectKBest feature selection.
        2. Categorical: Most frequent imputation. One-Hot or Ordinal encoding.
        
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
        
        # Feature Selection (Req: VarianceThreshold & SelectKBest)
        # Note: Trees perform internal feature selection, so 'variance' is usually enough to drop constants.
        # But 'SelectKBest' was explicitly requested.
        
        if self.config.get('feature_selection') == 'variance':
             num_steps.append(('variance_threshold', get_variance_threshold()))
        elif self.config.get('feature_selection') == 'k_best':
             # SelectKBest requires target y. fit_transform handles this.
             score_func = 'f_classif' if self.is_classification else 'f_regression'
             num_steps.append(('select_k_best', get_select_k_best(k=self.config.get('k_best', 10), score_func=score_func)))

        numerical_pipeline = Pipeline(steps=num_steps)

        # Categorical Steps
        cat_steps = []
        cat_steps.append(('imputer', get_imputer(strategy='most_frequent'))) 
        
        # Encoding (Req: OneHot or Ordinal)
        if self.config.get('encoding') == 'ordinal':
             cat_steps.append(('ordinal', get_ordinal_encoder()))
        else:
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
        Trains the Decision Tree model using GridSearchCV for hyperparameter tuning.
        
        It utilizes Cross-Validation (StratifiedKFold for classification, KFold for regression)
        to find the best parameters from the defined `param_grid`.
        
        Args:
            X_train: Training features array.
            y_train: Training target array.
        """
        if self.is_classification:
            cv = StratifiedKFold( # Stratified K-Fold (Req)
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
            scoring = 'neg_root_mean_squared_error'

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
        
        # Pruning check
        ccp_alpha = self.best_estimator.ccp_alpha
        if ccp_alpha > 0:
             print(f"[{self.name}] Tree pruned with ccp_alpha={ccp_alpha}")

    def calculate_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Calculates standard performance metrics and specific tree complexity metrics.

        Metrics Include:
        - Accuracy (Classification) or RMSE (Regression)
        - Tree Depth
        - Leaf Count

        Args:
            X_test: Test features array.
            y_test: Test target array.
            
        Returns:
            Dict[str, float]: Dictionary of calculated metrics.
        """
        y_pred = self.best_estimator.predict(X_test)
        
        metrics = {}
        
        # Tree Complexity Metrics (Req)
        metrics['Tree Depth'] = get_tree_depth(self.best_estimator)
        metrics['Leaf Count'] = get_leaf_count(self.best_estimator)

        if self.is_classification:
            metrics['Accuracy'] = calculate_accuracy(y_test, y_pred) # Req
        else:
            metrics['RMSE'] = calculate_rmse(y_test, y_pred) # Req
        
        return metrics

    def get_diagnostic_data(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Retrieves diagnostic data for visualization.
        
        Includes:
        - Predictions and Truth values
        - Feature Importance
        - Graphviz source code for tree visualization
        - Validation curve data (extracted from GridSearch results)
        
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
        
        # Graphviz Source (Req)
        dot_data = export_graphviz(
            self.best_estimator,
            out_file=None,
            filled=True,
            rounded=True,
            special_characters=True
        )

        # Validation Curve Data (Depth vs Score)
        # We can extract this from cv_results_ for 'max_depth' param
        results_df = pd.DataFrame(self.model.cv_results_)
        if 'param_estimator__max_depth' in results_df.columns:
             val_curve_data = results_df[['param_estimator__max_depth', 'mean_test_score', 'std_test_score']].to_dict(orient='records')
        else:
             val_curve_data = []

        return {
            'y_pred': y_pred,
            'y_test': y_test,
            'model_name': self.name,
            'feature_importances': self.get_feature_importance(),
            'tree_dot_data': dot_data,
            'validation_curve_data': val_curve_data
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Retrieves the Gini Importance (Feature Importance) from the trained tree.
        
        Returns:
            Dict[str, float]: Dictionary mapping feature indices to their importance scores.
        """
        if hasattr(self.best_estimator, 'feature_importances_'):
            # Note: We need feature names to make this mapped dict useful.
            # In this architecture, names are lost in numpy arrays unless preserved.
            # Returning raw list/dict with indices if possible.
            return dict(enumerate(self.best_estimator.feature_importances_))
        return {}
