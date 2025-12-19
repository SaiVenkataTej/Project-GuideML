import unittest
import pandas as pd
import numpy as np
from core_recommender.modeling.logisticRegression import LogisticRegressionModel

class TestLogisticRegressionModel(unittest.TestCase):
    def setUp(self):
        self.model = LogisticRegressionModel()
        # Simple binary classification with enough samples
        self.X_train = pd.DataFrame({
            'f1': np.random.rand(20), 
            'f2': np.random.rand(20)
        })
        self.y_train = pd.Series(np.concatenate([np.zeros(10), np.ones(10)]))
        self.X_test = pd.DataFrame({'f1': [1, 2], 'f2': [10, 20]})
        self.y_test = pd.Series([0, 1])

    def test_fit_predict(self):
        X_train_proc, y_train_proc, preprocessor = self.model.preprocess(self.X_train, self.y_train)
        self.model.fit(X_train_proc, y_train_proc)
        
        X_test_proc = preprocessor.transform(self.X_test)
        
        # LogisticRegression fit uses GridSearchCV and sets self.best_estimator
        self.assertIsNotNone(self.model.best_estimator)
        
        # For metrics, we need transformed y_test because calculate_metrics usually expects it
        if self.model.label_encoder:
            y_test_proc = self.model.label_encoder.transform(self.y_test)
        else:
            y_test_proc = self.y_test
            
        metrics = self.model.calculate_metrics(X_test_proc, y_test_proc)
        self.assertIn('Accuracy', metrics)

if __name__ == '__main__':
    unittest.main()
