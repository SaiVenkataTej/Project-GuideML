import unittest
import pandas as pd
import numpy as np
from core_recommender.modeling.svms import SVMModel

class TestSVMModel(unittest.TestCase):
    def setUp(self):
        self.clf = SVMModel(is_classification=True)
        self.reg = SVMModel(is_classification=False)

    def test_classification(self):
        # Need enough samples for 5-fold CV
        X = pd.DataFrame({
            'a': np.random.rand(20), 
            'b': np.random.rand(20)
        })
        y = pd.Series(np.random.randint(0, 2, 20)) # Binary class
        
        X_proc, y_proc, _ = self.clf.preprocess(X, y)
        self.clf.fit(X_proc, y_proc)
        
        # Verify fit by checking best_estimator and metrics
        self.assertIsNotNone(self.clf.best_estimator)
        metrics = self.clf.calculate_metrics(X_proc, y_proc)
        self.assertIn('Accuracy', metrics)

    def test_regression(self):
        X = pd.DataFrame({'a': np.linspace(0, 10, 20)})
        y = pd.Series(np.linspace(0, 10, 20) + np.random.randn(20) * 0.1)
        
        X_proc, y_proc, _ = self.reg.preprocess(X, y)
        self.reg.fit(X_proc, y_proc)
        
        self.assertIsNotNone(self.reg.best_estimator)
        metrics = self.reg.calculate_metrics(X_proc, y_proc)
        self.assertIn('RMSE', metrics)

if __name__ == '__main__':
    unittest.main()
