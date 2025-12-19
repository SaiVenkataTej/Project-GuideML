from abc import ABC, abstractmethod
from joblib import dump
import numpy as np
import pandas as pd
from typing import Dict, Any

# =========================================================================
# The BaseModel Abstract Class (F5)
# =========================================================================

class BaseModel(ABC):
    """
    Abstract Base Class (ABC) is used for defining the standardized interface 
    for all Machine Learning models in the GuideML recommender pipeline. 
    
    This class enforces modularity and ensures that concurrency orchestrators 
    (Joblib) can treat all model objects uniformly.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initializes the base model, storing its name and configuration."""
        self.name = name
        self.config = config
        self.model = None  # Placeholder for the Scikit-learn model instance
        self.metrics = {}  # Dictionary to store performance metrics (F8)

    # ---------------------------------------------------------------------
    # ABSTRACT METHODS (Must be implemented by every concrete model inheriting
    # the baseModel class)
    # ---------------------------------------------------------------------

    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepares the data specifically for this model.

        Used for feature engineering, scaling, encoding, handling missing 
        values, etc

        This method will call imported, granular functions from the 
        preprocessing.py module (F5).
        
        Args:
            data: The raw or partially processed DataFrame.
        Returns:
            The final feature array ready for the model.
        """
        pass # Concrete models must define this logic

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the specific Scikit-learn model instance.
        
        This method must incorporate logic for k-fold cross-validation (F6).
        
        Args:
            X_train: Training features array.
            y_train: Training target array.
        """
        pass # Concrete models must define this logic

    @abstractmethod
    def calculate_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Calculates and returns a dictionary of performance metrics (F8).
        
        This method must call imported, granular functions from evaluation.py.
        """
        pass

    @abstractmethod
    def get_diagnostic_data(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Retrieves the necessary data (predictions, probabilities) for generating 
        visualization plots (ROC, Confusion Matrix) (F9, F10).
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Retrieves the feature importance scores from the model, if supported (F9).
        """
        pass
    # ---------------------------------------------------------------------
    # CONCRETE METHOD (Reusable by all models)
    # ---------------------------------------------------------------------

    def export(self, filepath: str):
        """
        Serializes and exports the trained model artifact using joblib (F11).
        
        Args:
            filepath: The full path and filename for the exported model.
        """
        if self.model is None:
            raise ValueError("Cannot export model: Model has not been trained (fit) yet.")
        
        dump(self.model, filepath)
        print(f"Model {self.name} successfully exported to {filepath}")