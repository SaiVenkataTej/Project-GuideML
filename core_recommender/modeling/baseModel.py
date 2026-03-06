from abc import ABC, abstractmethod
from joblib import dump
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

# Import centralized logger
from core_recommender.logger import get_logger

logger = get_logger(__name__)

# =========================================================================
# The BaseModel Abstract Class (F5)
# =========================================================================

class BaseModel(ABC):
    """Abstract Base Class (ABC) defining the standardized interface for all ML models.

    Ensures modularity and consistent usage by concurrency orchestrators.
    Models must implement `fit` and `calculate_metrics`. 
    
    Attributes:
        name (str): Unique identifier for the model.
        config (Dict[str, Any]): Configuration parameters (hyperparameters).
        model (Any): Placeholder for the Scikit-learn model instance.
        metrics (Dict[str, float]): Dictionary storing performance metrics.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        """Initializes the base model with a name and configuration.

        Args:
            name (str): A unique identifier for the model.
            config (Dict[str, Any]): A dictionary containing configuration parameters.
        """
        self.name = name
        self.config = config
        self.model: Any = None
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
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Trains the specific Scikit-learn model instance.
        
        Args:
            X_train (np.ndarray): Training features array.
            y_train (np.ndarray): Training target array.
        """
        pass 

    @abstractmethod
    def calculate_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Calculates and returns a dictionary of performance metrics.

        Args:
            X_test (np.ndarray): Test features array.
            y_test (np.ndarray): Test target array.
            
        Returns:
            Dict[str, float]: Key-value pairs of metric names and their scores.
        """
        pass

    @abstractmethod
    def get_diagnostic_data(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Retrieves data for generating visualization plots.

        Args:
            X_test (np.ndarray): Test features array.
            y_test (np.ndarray): Test target array.

        Returns:
            Dict[str, Any]: Diagnostic data (predictions, probabilities, etc.).
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, Any]:
        """Retrieves the feature importance scores from the model.

        Returns:
            Dict[str, Any]: Map of feature names to importance scores.
        """
        pass
    
    # ---------------------------------------------------------------------
    # CONCRETE METHOD
    # ---------------------------------------------------------------------

    def export(self, filepath: str) -> None:
        """Serializes and exports the trained model artifact using joblib.
        
        Args:
            filepath (str): The full path and filename for the exported model.
        
        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Cannot export model: Model has not been trained (fit) yet.")
        
        dump(self.model, filepath)
        logger.info(f"✅ Model {self.name} successfully exported to {filepath}")