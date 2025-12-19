import unittest
import pandas as pd
import numpy as np
from core_recommender.modeling.knn import KNNModel

class TestKNNModel(unittest.TestCase):
    def setUp(self):
        # Test default (Classification)
        self.clf_model = KNNModel(is_classification=True)
        # Test Regression
        self.reg_model = KNNModel(is_classification=False)

    def test_classification_flow(self):
        X = pd.DataFrame({
            'a': np.random.rand(20), 
            'b': np.random.rand(20)
        })
        y = pd.Series(np.random.randint(0, 2, 20))
        
        X_proc, y_proc, proc = self.clf_model.preprocess(X, y)
        self.clf_model.fit(X_proc, y_proc)
        
        # KNN might not have best_estimator if it doesn't use GridSearch or exposes model directly?
        # Checking knn.py: fit sets self.best_estimator via GridSearchCV
        self.assertIsNotNone(self.clf_model.best_estimator)
        metrics = self.clf_model.calculate_metrics(X_proc, y_proc)
        self.assertIn('Accuracy', metrics)

    def test_regression_flow(self):
        X = pd.DataFrame({'a': np.linspace(0, 10, 20), 'b': np.linspace(0, 10, 20)})
        y = pd.Series(np.linspace(0, 10, 20))
        
        X_proc, y_proc, proc = self.reg_model.preprocess(X, y)
        self.reg_model.fit(X_proc, y_proc)
        
        self.assertIsNotNone(self.reg_model.best_estimator)
        metrics = self.reg_model.calculate_metrics(X_proc, y_proc)
        self.assertIn('RMSE', metrics)

if __name__ == '__main__':
    unittest.main()
