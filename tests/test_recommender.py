import unittest
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from core_recommender.profiling import DataProfiler
from core_recommender.visualization import plot_correlation_heatmap, plot_shap_summary
from core_recommender.modeling.linear_regression import LinearRegressionModel
from core_recommender.modeling.decision_trees import DecisionTreeModel
from core_recommender.execution import ModelExecutor


class TestRecommenderSystem(unittest.TestCase):
    def setUp(self):
        # Create temp directory for plot output tests
        self.test_dir = tempfile.mkdtemp()
        
        # 1. Create a classification dataset
        X_c, y_c = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        feature_names = [f"feature_{i}" for i in range(5)]
        self.df_classification = pd.DataFrame(X_c, columns=feature_names)
        self.df_classification['target'] = y_c
        
        # 2. Create a regression dataset
        X_r, y_r = make_regression(n_samples=100, n_features=5, random_state=42)
        self.df_regression = pd.DataFrame(X_r, columns=feature_names)
        self.df_regression['target'] = y_r

    def tearDown(self):
        # Clean up temp directory
        shutil.rmtree(self.test_dir)

    def test_profiler_with_nans(self):
        """Tests that DataProfiler doesn't crash when columns have NaNs (normality check fix)."""
        df = self.df_classification.copy()
        df['all_nan'] = np.nan
        df['mostly_nan'] = np.nan
        df.loc[:10, 'mostly_nan'] = np.random.randn(11)
        
        profiler = DataProfiler()
        profile = profiler.analyze(df.drop(columns=['target']), df['target'])['profile']
        
        self.assertIn('n_samples', profile)
        self.assertEqual(profile['n_samples'], 100)
        self.assertIn('sparsity', profile)

    def test_visualization_heatmap_name_error(self):
        """Tests that plot_correlation_heatmap runs without NameError (logger fix)."""
        df = self.df_classification.copy()
        df['target'] = df['target'].map({0: 'class_A', 1: 'class_B'})
        
        save_path = os.path.join(self.test_dir, 'heatmap.png')
        plot_correlation_heatmap(df, target_column='target', save_path=save_path)
        self.assertTrue(os.path.exists(save_path))

    def test_linear_regression_model(self):
        """Tests the Linear Regression concrete model implementation."""
        model = LinearRegressionModel(config={'cv_folds': 2, 'random_state': 42})
        X = self.df_regression.drop(columns=['target'])
        y = self.df_regression['target']
        
        X_trans, y_trans, preprocessor = model.preprocess(X, y)
        self.assertEqual(X_trans.shape[0], 100)
        self.assertEqual(y_trans.shape[0], 100)
        
        model.fit(X, y)
        self.assertIsNotNone(model.best_estimator)
        
        metrics = model.calculate_metrics(X, y)
        self.assertIn('RMSE', metrics)
        self.assertIn('MAE', metrics)
        self.assertIn('R2 Score', metrics)

        diagnostic_data = model.get_diagnostic_data(X, y)
        self.assertIn('y_pred', diagnostic_data)
        self.assertIn('y_true', diagnostic_data)

        # get_tailored_diagnostics replaces get_feature_importance (ISP refactor)
        tailored = model.get_tailored_diagnostics()
        self.assertIsInstance(tailored, dict)
        self.assertIn('coefficients', tailored)

    def test_decision_tree_model(self):
        """Tests the Decision Tree concrete model implementation."""
        model = DecisionTreeModel(is_classification=True, config={'cv_folds': 2, 'random_state': 42})
        X = self.df_classification.drop(columns=['target'])
        y = self.df_classification['target']
        
        X_trans, y_trans, preprocessor = model.preprocess(X, y)
        self.assertEqual(X_trans.shape[0], 100)
        
        model.fit(X, y)
        self.assertIsNotNone(model.best_estimator)
        
        metrics = model.calculate_metrics(X, y)
        self.assertIn('Accuracy', metrics)

        # get_tailored_diagnostics replaces get_feature_importance (ISP refactor)
        tailored = model.get_tailored_diagnostics()
        self.assertIsInstance(tailored, dict)
        self.assertIn('feature_importances', tailored)
        self.assertIn('tree_dot_data', tailored)

    def test_model_executor_end_to_end(self):
        """Tests the end-to-end execution flow including tailored diagnostics and feature names."""
        executor = ModelExecutor()
        
        summary = executor.run(
            self.df_regression,
            target_column='target',
            include_models=['Decision Tree', 'Linear Regression']
        )
        
        self.assertIn('task_type', summary)
        self.assertEqual(summary['task_type'].lower(), 'regression')
        self.assertIn('leaderboard', summary)
        self.assertIn('best_model', summary)
        
        best_model = summary['best_model']
        self.assertIn('name', best_model)
        self.assertIn('metrics', best_model)

        # New API: tailored_diagnostics replaces old feature_importance dict
        self.assertIn('tailored_diagnostics', best_model)
        self.assertIsInstance(best_model['tailored_diagnostics'], dict)

        # feature_names now exposed directly on best_model
        self.assertIn('feature_names', best_model)
        self.assertIsInstance(best_model['feature_names'], list)
        self.assertGreater(len(best_model['feature_names']), 0)


if __name__ == '__main__':
    unittest.main()
