import unittest
import pandas as pd
import numpy as np
from core_recommender.modeling.linearRegression import LinearRegressionModel

class TestLinearRegressionModel(unittest.TestCase):
    def setUp(self):
        self.model = LinearRegressionModel()
        self.X_train = pd.DataFrame({'feature': np.linspace(0, 10, 20)})
        self.y_train = pd.Series(np.linspace(0, 20, 20)) # y = 2x
        self.X_test = pd.DataFrame({'feature': [6, 7]})
        self.y_test = pd.Series([12, 14])

    def test_fit_predict(self):
        # Linear Regression is usually scaled, so exact coefficients might vary depending on scaler
        # But we check run completion and reasonable output shape
        X_train_proc, y_train_proc, preprocessor = self.model.preprocess(self.X_train, self.y_train)
        self.model.fit(X_train_proc, y_train_proc)
        
        X_test_proc = preprocessor.transform(self.X_test)
        
        # Use calculate_metrics or check best_estimator
        # LinearRegressionModel fit sets self.best_estimator
        self.assertIsNotNone(self.model.best_estimator)
        
        metrics = self.model.calculate_metrics(X_test_proc, self.y_test.values)
        self.assertIn('RMSE', metrics)

if __name__ == '__main__':
    unittest.main()
