"""
Unit Tests — core_recommender/execution.py (ModelExecutor)
============================================================
Tests every internal method and the main run() orchestration:
  __init__, _log_step, _infer_task_type, _detect_and_drop_leakage,
  _handle_outliers, _get_candidate_models, _train_single_model,
  _get_feature_names, run()
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from sklearn.datasets import make_classification, make_regression

from core_recommender.execution import ModelExecutor
from core_recommender.exceptions import DataValidationError
from core_recommender.modeling.registry import clear_registry


def _clf_df(n=150, n_features=5):
    X, y = make_classification(n_samples=n, n_features=n_features,
                                n_classes=2, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y
    return df


def _reg_df(n=150, n_features=5):
    X, y = make_regression(n_samples=n, n_features=n_features, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y.astype(float)
    return df


class TestModelExecutorInit(unittest.TestCase):
    def test_default_task_type(self):
        ex = ModelExecutor()
        self.assertEqual(ex.task_type, "auto")

    def test_default_random_state(self):
        ex = ModelExecutor()
        self.assertEqual(ex.random_state, 42)

    def test_results_starts_empty(self):
        ex = ModelExecutor()
        self.assertEqual(ex.results, [])

    def test_best_model_name_starts_none(self):
        ex = ModelExecutor()
        self.assertIsNone(ex.best_model_name)

    def test_pipeline_log_starts_empty(self):
        ex = ModelExecutor()
        self.assertEqual(ex.pipeline_log, [])

    def test_custom_params(self):
        ex = ModelExecutor(task_type="classification", random_state=7)
        self.assertEqual(ex.task_type, "classification")
        self.assertEqual(ex.random_state, 7)


class TestLogStep(unittest.TestCase):
    def test_appends_to_pipeline_log(self):
        ex = ModelExecutor()
        ex._log_step("Test Step", "detail message")
        self.assertEqual(len(ex.pipeline_log), 1)

    def test_log_entry_has_required_keys(self):
        ex = ModelExecutor()
        ex._log_step("My Step", "Some detail", icon="fas fa-cog")
        entry = ex.pipeline_log[0]
        for key in ("step", "details", "icon", "timestamp"):
            self.assertIn(key, entry)

    def test_log_entry_values(self):
        ex = ModelExecutor()
        ex._log_step("Step A", "Details A")
        entry = ex.pipeline_log[0]
        self.assertEqual(entry["step"], "Step A")
        self.assertEqual(entry["details"], "Details A")


class TestInferTaskType(unittest.TestCase):
    def setUp(self):
        self.ex = ModelExecutor()

    def test_float_target_is_regression(self):
        y = pd.Series([1.1, 2.2, 3.3])
        self.assertEqual(self.ex._infer_task_type(y), "regression")

    def test_object_target_is_classification(self):
        y = pd.Series(["cat", "dog", "cat"])
        self.assertEqual(self.ex._infer_task_type(y), "classification")

    def test_bool_target_is_classification(self):
        y = pd.Series([True, False, True])
        self.assertEqual(self.ex._infer_task_type(y), "classification")

    def test_integer_few_unique_is_classification(self):
        y = pd.Series([0, 1, 2, 0, 1])  # 3 unique values < 20
        self.assertEqual(self.ex._infer_task_type(y), "classification")

    def test_integer_many_unique_is_regression(self):
        y = pd.Series(list(range(30)))  # 30 unique values >= 20
        self.assertEqual(self.ex._infer_task_type(y), "regression")

    def test_forced_task_type_overrides_heuristic(self):
        ex = ModelExecutor(task_type="regression")
        y = pd.Series([0, 1, 2])  # would be classification by heuristic
        self.assertEqual(ex._infer_task_type(y), "regression")


class TestDetectAndDropLeakage(unittest.TestCase):
    def setUp(self):
        self.ex = ModelExecutor()

    def test_no_leakage_returns_same_shape(self):
        df = _clf_df()
        result = self.ex._detect_and_drop_leakage(df, "target")
        self.assertEqual(result.shape[1], df.shape[1])

    def test_perfect_correlation_drops_column(self):
        rng = np.random.RandomState(0)
        df = pd.DataFrame({"a": rng.randn(100)})
        df["target"] = df["a"]  # perfect correlation = 1.0
        df["noise"] = rng.randn(100)
        result = self.ex._detect_and_drop_leakage(df, "target", threshold=0.95)
        self.assertNotIn("a", result.columns)
        self.assertIn("noise", result.columns)

    def test_empty_df_returns_unchanged(self):
        df = pd.DataFrame()
        result = self.ex._detect_and_drop_leakage(df, "target")
        self.assertTrue(result.empty)

    def test_missing_target_returns_unchanged(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = self.ex._detect_and_drop_leakage(df, "missing_col")
        self.assertEqual(result.shape, df.shape)

    def test_suspicious_column_name_dropped(self):
        rng = np.random.RandomState(0)
        df = pd.DataFrame({
            "feature_a": rng.randn(100),
            "result_proxy": ["yes"] * 50 + ["no"] * 50,   # high-risk name
            "target": rng.randint(0, 2, 100)
        })
        result = self.ex._detect_and_drop_leakage(df, "target")
        # result_proxy has a high-risk name and should be dropped
        self.assertNotIn("result_proxy", result.columns)


class TestHandleOutliers(unittest.TestCase):
    def setUp(self):
        self.ex = ModelExecutor()

    def test_no_outliers_same_length(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "target": [1.0, 2.0, 3.0]})
        result = self.ex._handle_outliers(df, "target")
        self.assertEqual(len(result), 3)

    def test_outliers_removed(self):
        # IQR rule: Q1=2, Q3=4, IQR=2, lower=-1, upper=7
        df = pd.DataFrame({
            "x": list(range(10)) + [9999],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0,
                       6.0, 7.0, 8.0, 9.0, 10.0, 9999.0]
        })
        result = self.ex._handle_outliers(df, "target")
        self.assertLess(len(result), len(df))

    def test_non_numeric_target_unchanged(self):
        df = pd.DataFrame({"x": [1, 2, 3], "target": ["a", "b", "c"]})
        result = self.ex._handle_outliers(df, "target")
        self.assertEqual(len(result), 3)

    def test_empty_df_returned(self):
        df = pd.DataFrame()
        result = self.ex._handle_outliers(df, "target")
        self.assertTrue(result.empty)


class TestGetCandidateModels(unittest.TestCase):
    def test_returns_list(self):
        ex = ModelExecutor()
        models = ex._get_candidate_models("classification")
        self.assertIsInstance(models, list)

    def test_models_are_not_empty(self):
        ex = ModelExecutor()
        models = ex._get_candidate_models("classification")
        self.assertGreater(len(models), 0)

    def test_include_models_filter(self):
        ex = ModelExecutor()
        models = ex._get_candidate_models("regression",
                                           include_models=["Linear Regression"])
        names = [m.name for m in models]
        self.assertTrue(any("Linear Regression" in n for n in names))

    def test_nonexistent_include_falls_back_to_all(self):
        ex = ModelExecutor()
        models_all = ex._get_candidate_models("classification")
        models_filtered = ex._get_candidate_models("classification",
                                                    include_models=["NoSuchModel"])
        self.assertEqual(len(models_filtered), len(models_all))


class TestRunClassification(unittest.TestCase):
    def test_run_returns_dict_with_required_keys(self):
        df = _clf_df(n=150)
        ex = ModelExecutor(random_state=42)
        summary = ex.run(df, "target",
                          include_models=["Decision Tree"])
        for key in ("task_type", "leaderboard", "best_model", "total_time"):
            self.assertIn(key, summary)

    def test_task_type_is_classification(self):
        df = _clf_df(n=150)
        ex = ModelExecutor()
        summary = ex.run(df, "target",
                          include_models=["Logistic Regression"])
        self.assertEqual(summary["task_type"], "classification")

    def test_leaderboard_is_non_empty_list(self):
        df = _clf_df(n=150)
        ex = ModelExecutor()
        summary = ex.run(df, "target",
                          include_models=["Naive Bayes"])
        self.assertIsInstance(summary["leaderboard"], list)
        self.assertGreater(len(summary["leaderboard"]), 0)

    def test_best_model_has_name_and_metrics(self):
        df = _clf_df(n=150)
        ex = ModelExecutor()
        summary = ex.run(df, "target",
                          include_models=["Decision Tree"])
        best = summary["best_model"]
        self.assertIn("name", best)
        self.assertIn("metrics", best)

    def test_missing_target_raises_data_validation_error(self):
        df = _clf_df()
        ex = ModelExecutor()
        with self.assertRaises(DataValidationError):
            ex.run(df, "nonexistent_column",
                   include_models=["Decision Tree"])

    def test_total_time_is_positive(self):
        df = _clf_df(n=120)
        ex = ModelExecutor()
        summary = ex.run(df, "target", include_models=["Decision Tree"])
        self.assertGreater(summary["total_time"], 0.0)

    def test_tailored_diagnostics_in_best_model(self):
        df = _clf_df(n=120)
        ex = ModelExecutor()
        summary = ex.run(df, "target", include_models=["Decision Tree"])
        best = summary["best_model"]
        self.assertIn("tailored_diagnostics", best)
        self.assertIsInstance(best["tailored_diagnostics"], dict)

    def test_feature_names_in_best_model(self):
        df = _clf_df(n=120)
        ex = ModelExecutor()
        summary = ex.run(df, "target", include_models=["Decision Tree"])
        best = summary["best_model"]
        self.assertIn("feature_names", best)
        self.assertIsInstance(best["feature_names"], list)
        self.assertGreater(len(best["feature_names"]), 0)


class TestRunRegression(unittest.TestCase):
    def test_task_type_is_regression(self):
        df = _reg_df(n=150)
        ex = ModelExecutor()
        summary = ex.run(df, "target",
                          include_models=["Linear Regression"])
        self.assertEqual(summary["task_type"], "regression")

    def test_best_model_metrics_has_rmse(self):
        df = _reg_df(n=150)
        ex = ModelExecutor()
        summary = ex.run(df, "target",
                          include_models=["Linear Regression"])
        best = summary["best_model"]
        self.assertIn("RMSE", best["metrics"])


class TestRunProgressCallback(unittest.TestCase):
    def test_progress_callback_is_called(self):
        df = _clf_df(n=120)
        ex = ModelExecutor()
        calls = []
        ex.run(df, "target",
               include_models=["Decision Tree"],
               progress_callback=lambda p, m: calls.append((p, m)))
        self.assertGreater(len(calls), 0)


if __name__ == "__main__":
    unittest.main()
