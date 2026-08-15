from abc import ABC, abstractmethod
from joblib import dump
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union

# Import centralized logger
from core_recommender.logger import get_logger
from core_recommender.exceptions import ModelNotFittedError, ModelExportError

logger = get_logger(__name__)

# =========================================================================
# The BaseModel Abstract Class (F5)
# =========================================================================

class BaseModel(ABC):
    """Abstract Base Class (ABC) defining the standardised interface for all ML models.

    Enforces a strict core contract (preprocess / fit / calculate_metrics /
    get_diagnostic_data) while deliberately leaving model-specific capabilities
    optional via the concrete ``get_tailored_diagnostics`` hook.

    Design principles
    -----------------
    * **SRP** — one class, one responsibility: defining the model contract.
    * **OCP** — closed for modification; open for extension via subclassing.
    * **ISP** — only truly universal methods are abstract; optional features
      are exposed through the non-mandatory ``get_tailored_diagnostics`` hook.
    * **DIP** — consumers depend on this abstraction, never on concrete classes.

    Attributes:
        name (str): Human-readable identifier for the model.
        config (Dict[str, Any]): Configuration / hyperparameter dictionary.
        model (Any): Placeholder for the fitted Scikit-learn estimator.
        metrics (Dict[str, float]): Dictionary of computed performance metrics.
    """
    
    def __init__(self, name: str, config: Dict[str, Any], **kwargs) -> None:
        """Initializes the base model with a name and configuration.

        Args:
            name (str): A unique identifier for the model.
            config (Dict[str, Any]): A dictionary containing configuration parameters.
            **kwargs: Additional keyword arguments ignored by the base model.
        """
        self.name = name
        self.config = config
        self.model: Any = None
        self.best_estimator: Any = None
        self.metrics: Dict[str, float] = {}

    # ---------------------------------------------------------------------
    # ABSTRACT METHODS
    # ---------------------------------------------------------------------

    @abstractmethod
    def preprocess(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Any:
        """Prepares the data specifically for this model.
        
        Args:
            X (pd.DataFrame): Training features.
            y (Optional[pd.Series]): Training targets (optional).

        Returns:
            Any: Transformed features (and targets if applicable), and the fitted preprocessor.
        """
        pass 

    @abstractmethod
    def fit(self, X_train: Union[pd.DataFrame, np.ndarray], y_train: np.ndarray) -> None:
        """Trains the specific Scikit-learn model instance.
        
        Args:
            X_train (Union[pd.DataFrame, np.ndarray]): Training features (DataFrame or array).
            y_train (np.ndarray): Training target array.
        """
        pass 

    @abstractmethod
    def calculate_metrics(self, X_test: Union[pd.DataFrame, np.ndarray], y_test: np.ndarray) -> Dict[str, float]:
        """Calculates and returns a dictionary of performance metrics.

        Args:
            X_test (Union[pd.DataFrame, np.ndarray]): Test features (DataFrame or array).
            y_test (np.ndarray): Test target array.
            
        Returns:
            Dict[str, float]: Key-value pairs of metric names and their scores.
        """
        pass

    @abstractmethod
    def get_diagnostic_data(self, X_test: Union[pd.DataFrame, np.ndarray], y_test: np.ndarray) -> Dict[str, Any]:
        """Retrieves data for generating visualization plots.

        Args:
            X_test (Union[pd.DataFrame, np.ndarray]): Test features (DataFrame or array).
            y_test (np.ndarray): Test target array.

        Returns:
            Dict[str, Any]: Diagnostic data (predictions, probabilities, etc.).
        """
        pass

    # ---------------------------------------------------------------------
    # OPTIONAL EXTENSION HOOK  (ISP — not abstract, not mandatory)
    # ---------------------------------------------------------------------

    def get_tailored_diagnostics(self) -> Dict[str, Any]:
        """Returns model-specific advanced insights as a flat dictionary.

        Concrete models **may** override this method to surface any insight
        that is unique to their algorithm (e.g. feature importances for tree
        models, support vectors for SVMs, elbow data for KNN).  Models that
        do not override it silently return an empty dict, which is perfectly
        valid — the executor handles both cases transparently.

        Returns:
            Dict[str, Any]: Algorithm-specific diagnostic payload.  Keys and
            value types are defined by each concrete subclass.  Returns ``{}``
            by default.

        Example overrides
        -----------------
        * RandomForest  → ``{'feature_importances_mdi': [...], 'oob_score': 0.93}``
        * DecisionTree  → ``{'tree_dot_data': '...', 'feature_importances': [...]}``
        * Logistic/Linear Regression → ``{'coefficients': [...]}``
        * KNN           → ``{'elbow_data': [...], 'neighbor_indices': [...]}``
        * NaiveBayes    → ``{'feature_log_prob': [...]}``
        * SVM           → ``{'support_vectors': [...], 'n_support': [...]}``
        """
        return {}

    def get_parameter_descriptions(self) -> Dict[str, Dict[str, str]]:
        """Returns descriptive dictionary mapping hyperparameter names to their descriptions.

        Returns:
            Dict[str, Dict[str, str]]: Mapping of parameter name to its details.
        """
        return {}
    
    # ---------------------------------------------------------------------
    # CONCRETE METHOD
    # ---------------------------------------------------------------------

    def export(self, filepath: str) -> None:
        """Serializes and exports the trained model artifact using joblib.
        
        Args:
            filepath (str): The full path and filename for the exported model.
        
        Raises:
            ModelNotFittedError: If the model has not been trained yet.
            ModelExportError: If the export operation fails.
        """
        if self.model is None:
            raise ModelNotFittedError(
                "Cannot export model: it has not been trained yet. Call fit() first.",
                model_name=self.name
            )
        try:
            dump(self.model, filepath)
            logger.info(f"✅ Model {self.name} successfully exported to {filepath}")
        except OSError as exc:
            raise ModelExportError(
                f"Failed to write model to '{filepath}': {exc}",
                model_name=self.name,
                filepath=filepath
            ) from exc