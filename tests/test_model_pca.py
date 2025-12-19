import unittest
import pandas as pd
import numpy as np
from core_recommender.modeling.PCA import PCAModel

class TestPCAModel(unittest.TestCase):
    def setUp(self):
        self.model = PCAModel(config={'n_components': 2})
        self.data = pd.DataFrame({
            'a': np.random.rand(10),
            'b': np.random.rand(10),
            'c': np.random.rand(10)
        })

    def test_fit_metrics(self):
        # Preprocess usually returns X, None for unsupervised
        X_proc, y_proc, _ = self.model.preprocess(self.data, None)
        
        self.model.fit(X_proc, y_proc)
        
        # Verify model instance is set
        self.assertIsNotNone(self.model.model)
        
        # Test metrics
        metrics = self.model.calculate_metrics(X_proc)
        self.assertIn('Reconstruction RMSE', metrics)
        self.assertEqual(metrics['n_components'], 2)

if __name__ == '__main__':
    unittest.main()
