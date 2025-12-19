import unittest
import pandas as pd
import numpy as np
from core_recommender.execution import ModelExecutor

class TestModelExecutor(unittest.TestCase):
    def setUp(self):
        self.executor = ModelExecutor()
        # Small dataset
        self.df = pd.DataFrame({
            'f1': np.random.rand(20),
            'f2': np.random.rand(20),
            'target': np.random.randint(0, 2, 20)
        })

    def test_run_classification_smoke(self):
        # We assume auto task inference works
        # This will run GridSearch (might be slow if grids are large)
        # Ideally, we'd mock models or reduce grid size in config, 
        # but for end-to-end "unit" test, we just check it doesn't crash on small data
        
        # Override grids for speed?
        # The executors init models with default config globally.
        # We can pass a simplified config to executor? No, models get config from global/defaults.
        # We will trust that defaults on small data are fast enough (few seconds).
        
        results = self.executor.run(self.df, target_column='target')
        
        self.assertIn('best_model', results)
        self.assertIn('leaderboard', results)
        self.assertTrue(len(results['leaderboard']) > 0)

    def test_run_regression_smoke(self):
        df_reg = pd.DataFrame({
            'f1': np.random.rand(20),
            'f2': np.random.rand(20),
            'target': np.random.rand(20) * 10
        })
        results = self.executor.run(df_reg, target_column='target')
        
        self.assertIn('best_model', results)
        self.assertEqual(results['task_type'], 'regression')

if __name__ == '__main__':
    unittest.main()
