"""
Modeling Package
================

This package contains the core machine learning models used in the recommendation engine.
It includes a variety of algorithms for regression, classification, and dimensionality reduction.

All models inherit from the `BaseModel` abstract base class to ensure a consistent API.

Available Models:
- Linear Models: LinearRegressionModel, LogisticRegressionModel
- Tree-Based: DecisionTreeModel, RandomForestModel
- Distance-Based: KNNModel
- Bayesian: NaiveBayesModel
- SVM: SVMModel
- Dimensionality Reduction: PCAModel
"""

# Base Class
from .base_model import BaseModel

# Classification & Regression Models
from .knn import KNNModel
from .svms import SVMModel
from .decision_trees import DecisionTreeModel
from .linear_regression import LinearRegressionModel
from .logistic_regression import LogisticRegressionModel
from .random_forest import RandomForestModel
from .naive_bayes import NaiveBayesModel

# Unsupervised Models
from .pca import PCAModel

