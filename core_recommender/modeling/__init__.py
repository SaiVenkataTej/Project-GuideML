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
from .baseModel import BaseModel

# Classification & Regression Models
from .knn import KNNModel
from .svms import SVMModel
from .decisionTrees import DecisionTreeModel
from .linearRegression import LinearRegressionModel
from .logisticRegression import LogisticRegressionModel
from .randomForest import RandomForestModel
from .naiveBayes import NaiveBayesModel

# Unsupervised Models
from .PCA import PCAModel

