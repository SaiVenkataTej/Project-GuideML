import unittest
import pandas as pd
import numpy as np
from core_recommender.modeling.decisionTrees import DecisionTreeModel

class TestDecisionTreeModel(unittest.TestCase):
    def setUp(self):
        self.clf = DecisionTreeModel(is_classification=True)
        self.reg = DecisionTreeModel(is_classification=False)

    def test_classification(self):
        # Need enough samples for 5-fold CV
        X = pd.DataFrame({'f': np.random.rand(20)})
        y = pd.Series(np.random.randint(0, 2, 20))
        X_proc, y_proc, _ = self.clf.preprocess(X, y)
        self.clf.fit(X_proc, y_proc)
        preds = self.clf.best_estimator.predict(X_proc)
        self.assertEqual(len(preds), 20)
        metrics = self.clf.calculate_metrics(X_proc, y_proc)
        self.assertIn('Accuracy', metrics)

    def test_regression(self):
        X = pd.DataFrame({'f': np.linspace(0, 10, 20)})
        y = pd.Series(np.linspace(0, 100, 20))
        X_proc, y_proc, _ = self.reg.preprocess(X, y)
        self.reg.fit(X_proc, y_proc)
        preds = self.reg.best_estimator.predict(X_proc)
        self.assertEqual(len(preds), 20)
        metrics = self.reg.calculate_metrics(X_proc, y_proc)
        self.assertIn('RMSE', metrics)

if __name__ == '__main__':
    unittest.main()
