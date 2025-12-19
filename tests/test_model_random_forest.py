import unittest
import pandas as pd
import numpy as np
from core_recommender.modeling.randomForest import RandomForestModel

class TestRandomForestModel(unittest.TestCase):
    def setUp(self):
        self.clf = RandomForestModel(is_classification=True)
        self.reg = RandomForestModel(is_classification=False)

    def test_classification(self):
        # Small dataset to speed up
        X = pd.DataFrame({'a': np.random.rand(20), 'b': np.random.rand(20)})
        y = pd.Series(np.random.randint(0, 2, 20))
        
        X_proc, y_proc, _ = self.clf.preprocess(X, y)
        self.clf.fit(X_proc, y_proc)
        
        self.assertIsNotNone(self.clf.best_estimator)
        metrics = self.clf.calculate_metrics(X_proc, y_proc)
        self.assertIn('F1 Score', metrics)

    def test_regression(self):
        X = pd.DataFrame({'a': np.linspace(0, 1, 20)})
        y = pd.Series(np.linspace(0, 10, 20))
        
        X_proc, y_proc, _ = self.reg.preprocess(X, y)
        self.reg.fit(X_proc, y_proc)
        
        self.assertIsNotNone(self.reg.best_estimator)
        metrics = self.reg.calculate_metrics(X_proc, y_proc)
        self.assertIn('RMSE', metrics)

if __name__ == '__main__':
    unittest.main()
