import unittest
import pandas as pd
import numpy as np
from core_recommender.modeling.naiveBayes import NaiveBayesModel

class TestNaiveBayesModel(unittest.TestCase):
    def setUp(self):
        # Defaults to GaussianNB by task_type logic if not specified or inferred
        self.model = NaiveBayesModel()

    def test_fit_predict(self):
        # Need enough samples
        X = pd.DataFrame({'a': np.random.randn(20), 'b': np.random.randn(20)})
        y = pd.Series(np.random.randint(0, 2, 20))
        
        X_proc, y_proc, _ = self.model.preprocess(X, y)
        self.model.fit(X_proc, y_proc)
        
        self.assertIsNotNone(self.model.best_estimator)
        metrics = self.model.calculate_metrics(X_proc, y_proc)
        self.assertIn('Accuracy', metrics)
        
        # Verify it inferred Gaussian
        self.assertTrue('Gaussian' in self.model.name)

if __name__ == '__main__':
    unittest.main()
