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

# Import centralized logger
from core_recommender.logger import get_logger
logger = get_logger(__name__)

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
    
    Rationale:
    ----------
    - **Non-Linear Relationships**: Capable of capturing complex, non-linear patterns without explicit feature combinations.
    - **Interpretability**: One of the most explainable models; decisions can be visualized as a flowchart.
    - **No Scaling Required**: Trees are invariant to monotonic transformations, so scaling is not strictly necessary (though used here for pipeline consistency).

    This model supports Cost-Complexity Pruning (CCP) to control overfitting and integrates 
    specialized tree metrics such as depth and leaf count.
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
        
        Rationale:
        ----------
        - **Label Encoding**: Trees handle categorical data naturally, but scikit-learn requires numerical inputs.
        - **Imputation**: Missing values must be handled; Median/Mode is a robust baseline.
        
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
        logger.debug(f"[{self.name}] Entering preprocess()...")
        logger.debug(f"[{self.name}] Input shape: X={X.shape}, y={y.shape}")
        
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
            logger.debug(f"[{self.name}] Using VarianceThreshold feature selection")
        elif self.config.get('feature_selection') == 'k_best':
            # SelectKBest requires target y. fit_transform handles this.
            score_func = 'f_classif' if self.is_classification else 'f_regression'
            num_steps.append(('select_k_best', get_select_k_best(k=self.config.get('k_best', 10), score_func=score_func)))
            logger.debug(f"[{self.name}] Using SelectKBest feature selection (k={self.config.get('k_best', 10)})")

        numerical_pipeline = Pipeline(steps=num_steps)

        # Categorical Steps
        cat_steps = []
        cat_steps.append(('imputer', get_imputer(strategy='most_frequent'))) 
        
        # Encoding (Req: OneHot or Ordinal)
        if self.config.get('encoding') == 'ordinal':
            cat_steps.append(('ordinal', get_ordinal_encoder()))
            logger.debug(f"[{self.name}] Using Ordinal Encoding for categorical features")
        else:
            cat_steps.append(('onehot', get_one_hot_encoder(handle_unknown='ignore', sparse_output=False)))
            logger.debug(f"[{self.name}] Using One-Hot Encoding for categorical features")
        
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
        logger.debug(f"[{self.name}] ColumnTransformer created with numerical and categorical pipelines.")

        self.preprocessor = preprocessor

        X_transformed = preprocessor.fit_transform(X, y)
        X_transformed = np.asarray(X_transformed)
        logger.debug(f"[{self.name}] Features transformed. New shape: {X_transformed.shape}")

        if self.is_classification:
            if not np.issubdtype(y.dtype, np.number):
                le = LabelEncoder()
                y_transformed = le.fit_transform(y)
                self.label_encoder = le
                logger.debug(f"[{self.name}] Target variable LabelEncoded.")
            else:
                y_transformed = y.values
                self.label_encoder = None
        else:
            y_transformed = y.values
            self.label_encoder = None
        logger.debug(f"[{self.name}] Preprocessing complete.")
        return X_transformed, y_transformed, preprocessor

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """
        Trains the Decision Tree model using a Unified Pipeline to prevent data leakage.
        """
        logger.info(f"[{self.name}] Starting training...")
        logger.debug(f"[{self.name}] Training data shape: X={X_train.shape}, y={y_train.shape}")
        
        # 1. Pipeline Construction
        preprocessor_template = clone(self.preprocessor)
        pipe = Pipeline(steps=[
            ('pre', preprocessor_template),
            ('model', self.model_instance)
        ])
        logger.debug(f"[{self.name}] Pipeline created with preprocessor and model instance.")

        # 2. Adjust Param Grid
        pipeline_params = {f'model__{k}': v for k, v in self.param_grid.items()}
        logger.debug(f"[{self.name}] Parameter grid for RandomizedSearchCV: {pipeline_params}")

        # 3. CV Strategy
        if self.is_classification:
            cv = StratifiedKFold(n_splits=self.config.get('cv_folds', 5), shuffle=True, random_state=self.config.get('random_state', 42))
            scoring = 'accuracy'
            logger.debug(f"[{self.name}] Using StratifiedKFold for classification with {self.config.get('cv_folds', 5)} folds and 'accuracy' scoring.")
        else:
            cv = KFold(n_splits=self.config.get('cv_folds', 5), shuffle=True, random_state=self.config.get('random_state', 42))
            scoring = 'neg_root_mean_squared_error'
            logger.debug(f"[{self.name}] Using KFold for regression with {self.config.get('cv_folds', 5)} folds and 'neg_root_mean_squared_error' scoring.")

        # 4. Randomized Search on Pipeline
        random_search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=pipeline_params,
            cv=cv,
            scoring=scoring,
            n_iter=self.config.get('n_iter', 10),
            n_jobs=self.config.get('n_jobs', -1),
            random_state=self.config.get('random_state', 42)
        )
        logger.debug(f"[{self.name}] Starting RandomizedSearchCV with n_iter={self.config.get('n_iter', 10)}.")

        random_search.fit(X_train, y_train)
        
        self.best_estimator = random_search.best_estimator_
        self.model = random_search
        
        best_score = random_search.best_score_
        best_params = random_search.best_params_
        logger.info(f"✅ [{self.name}] Training complete")
        logger.info(f"[{self.name}] Best CV Score: {best_score:.4f}")
        logger.debug(f"[{self.name}] Best params: {best_params}")

    def calculate_metrics(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """
        Calculates performance metrics using the full Pipeline.
        """
        y_pred = self.best_estimator.predict(X_test)
        
        metrics = {}
        final_tree = self.best_estimator.named_steps['model']
        
        # Tree Complexity Metrics
        metrics['Tree Depth'] = get_tree_depth(final_tree)
        metrics['Leaf Count'] = get_leaf_count(final_tree)

        if self.is_classification:
            metrics['Accuracy'] = calculate_accuracy(y_test, y_pred) 
        else:
            metrics['RMSE'] = calculate_rmse(y_test, y_pred) 
        
        return metrics

    def get_diagnostic_data(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Retrieves diagnostic data for visualization using the Pipeline.
        """
        if not hasattr(self, 'best_estimator'):
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

        # Validation Curve Data
        results_df = pd.DataFrame(self.model.cv_results_)
        if 'param_model__max_depth' in results_df.columns:
             val_curve_data = results_df[['param_model__max_depth', 'mean_test_score', 'std_test_score']].to_dict(orient='records')
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
        """
        final_tree = self.best_estimator.named_steps['model']
        if hasattr(final_tree, 'feature_importances_'):
            return dict(enumerate(final_tree.feature_importances_))
        return {}
