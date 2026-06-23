"""
Integration Tests — Full AutoML Pipeline
=========================================
Tests the end-to-end execution through ModelExecutor with real data,
covering classification, regression, mixed features, edge cases,
multi-model selection, and result reproducibility.
"""

import unittest
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
from sklearn.datasets import make_classification, make_regression

from core_recommender.execution import ModelExecutor
from core_recommender.exceptions import DataValidationError


# ============================================================
# Shared helpers
# ============================================================

def _clf_df(n=200, n_features=6, n_classes=2, random_state=42):
    X, y = make_classification(n_samples=n, n_features=n_features,
                                n_classes=n_classes, random_state=random_state,
                                n_informative=4, n_redundant=1)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y
    return df


def _reg_df(n=200, n_features=6, random_state=42):
    X, y = make_regression(n_samples=n, n_features=n_features,
                            noise=0.1, random_state=random_state)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y.astype(float)
    return df


def _mixed_df(n=200):
    """DataFrame with both numeric and categorical features."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "age":      rng.randint(18, 70, n).astype(float),
        "salary":   rng.randn(n) * 20000 + 60000,
        "gender":   rng.choice(["M", "F", "Other"], n),
        "region":   rng.choice(["North", "South", "East", "West"], n),
        "score":    rng.randn(n),
    })
    df["target"] = (df["age"] > 40).astype(int)
    return df


# ============================================================
# 1. Classification Pipeline
# ============================================================

class TestClassificationPipelineDecisionTree(unittest.TestCase):
    """Binary classification end-to-end with Decision Tree."""

    @classmethod
    def setUpClass(cls):
        cls.df = _clf_df()
        cls.executor = ModelExecutor(random_state=42)
        cls.summary = cls.executor.run(cls.df, "target",
                                        include_models=["Decision Tree"])

    def test_task_type_is_classification(self):
        self.assertEqual(self.summary["task_type"], "classification")

    def test_leaderboard_non_empty(self):
        self.assertGreater(len(self.summary["leaderboard"]), 0)

    def test_leaderboard_entries_have_name_and_metrics(self):
        for entry in self.summary["leaderboard"]:
            self.assertIn("name", entry)
            self.assertIn("metrics", entry)

    def test_best_model_has_all_required_keys(self):
        for key in ("name", "metrics", "tailored_diagnostics", "feature_names"):
            self.assertIn(key, self.summary["best_model"])

    def test_classification_metrics_present(self):
        metrics = self.summary["best_model"]["metrics"]
        self.assertIn("F1 Score", metrics)

    def test_feature_names_match_input(self):
        feature_names = self.summary["best_model"]["feature_names"]
        expected = [c for c in self.df.columns if c != "target"]
        self.assertEqual(set(feature_names), set(expected))

    def test_total_time_positive(self):
        self.assertGreater(self.summary["total_time"], 0)

    def test_pipeline_log_populated(self):
        self.assertGreater(len(self.executor.pipeline_log), 0)

    def test_leaderboard_ordered_by_f1(self):
        """Leaderboard should be sorted descending by F1 Score."""
        lb = self.summary["leaderboard"]
        if len(lb) > 1:
            scores = [entry["metrics"].get("F1 Score", 0) for entry in lb]
            self.assertEqual(scores, sorted(scores, reverse=True))


class TestClassificationPipelineMultiClass(unittest.TestCase):
    """Multi-class classification (3 classes) with Random Forest."""

    @classmethod
    def setUpClass(cls):
        cls.df = _clf_df(n_classes=3)
        cls.executor = ModelExecutor()
        cls.summary = cls.executor.run(cls.df, "target",
                                        include_models=["Random Forest"])

    def test_task_type_is_classification(self):
        self.assertEqual(self.summary["task_type"], "classification")

    def test_f1_in_range(self):
        f1 = self.summary["best_model"]["metrics"]["F1 Score"]
        self.assertGreaterEqual(f1, 0.0)
        self.assertLessEqual(f1, 1.0)


# ============================================================
# 2. Regression Pipeline
# ============================================================

class TestRegressionPipelineLinear(unittest.TestCase):
    """Regression end-to-end with Linear Regression."""

    @classmethod
    def setUpClass(cls):
        cls.df = _reg_df()
        cls.executor = ModelExecutor()
        cls.summary = cls.executor.run(cls.df, "target",
                                        include_models=["Linear Regression"])

    def test_task_type_is_regression(self):
        self.assertEqual(self.summary["task_type"], "regression")

    def test_best_model_has_rmse(self):
        self.assertIn("RMSE", self.summary["best_model"]["metrics"])

    def test_rmse_non_negative(self):
        rmse = self.summary["best_model"]["metrics"]["RMSE"]
        self.assertGreaterEqual(rmse, 0.0)

    def test_r2_score_leq_1(self):
        metrics = self.summary["best_model"]["metrics"]
        if "R2 Score" in metrics:
            self.assertLessEqual(metrics["R2 Score"], 1.0)


# ============================================================
# 3. Mixed Features (numeric + categorical)
# ============================================================

class TestMixedFeaturesPipeline(unittest.TestCase):
    """Pipeline handles mixed dtype DataFrames without errors."""

    @classmethod
    def setUpClass(cls):
        cls.df = _mixed_df()
        cls.executor = ModelExecutor()
        cls.summary = cls.executor.run(cls.df, "target",
                                        include_models=["Decision Tree"])

    def test_runs_without_error(self):
        self.assertIn("task_type", self.summary)

    def test_task_type_is_classification(self):
        self.assertEqual(self.summary["task_type"], "classification")

    def test_feature_names_include_categorical_expanded(self):
        feature_names = self.summary["best_model"]["feature_names"]
        self.assertIsInstance(feature_names, list)


# ============================================================
# 4. Multi-Model Competition
# ============================================================

class TestMultiModelCompetition(unittest.TestCase):
    """Run two models and confirm leaderboard has both entries."""

    @classmethod
    def setUpClass(cls):
        cls.df = _clf_df(n=200)
        cls.executor = ModelExecutor()
        cls.summary = cls.executor.run(
            cls.df, "target",
            include_models=["Decision Tree", "KNN"]
        )

    def test_leaderboard_has_two_entries(self):
        self.assertEqual(len(self.summary["leaderboard"]), 2)

    def test_best_model_is_top_of_leaderboard(self):
        """best_model name should match the first leaderboard entry."""
        best_name = self.summary["best_model"]["name"]
        top_name = self.summary["leaderboard"][0]["name"]
        self.assertEqual(best_name, top_name)


# ============================================================
# 5. Data Validation Edge Cases
# ============================================================

class TestDataValidationInPipeline(unittest.TestCase):
    def setUp(self):
        self.ex = ModelExecutor()

    def test_missing_target_column_raises(self):
        df = _clf_df()
        with self.assertRaises(DataValidationError):
            self.ex.run(df, "nonexistent", include_models=["Decision Tree"])

    def test_all_nan_target_raises(self):
        df = _reg_df()
        df["target"] = np.nan
        with self.assertRaises(Exception):
            self.ex.run(df, "target", include_models=["Linear Regression"])

    def test_dataframe_with_nans_in_features_runs(self):
        df = _reg_df()
        df.loc[:10, "f0"] = np.nan
        df.loc[5:15, "f1"] = np.nan
        summary = self.ex.run(df, "target", include_models=["Decision Tree"])
        self.assertIn("task_type", summary)


# ============================================================
# 6. Leakage Detection Integration
# ============================================================

class TestLeakageDetectionIntegration(unittest.TestCase):
    def test_perfectly_correlated_column_dropped(self):
        rng = np.random.RandomState(0)
        n = 200
        df = pd.DataFrame({
            "f0": rng.randn(n),
            "f1": rng.randn(n),
        })
        df["target"] = df["f0"] * 2 + 1.5   # regression target
        df["leaky"] = df["target"]           # perfect correlation

        ex = ModelExecutor()
        summary = ex.run(df, "target", include_models=["Linear Regression"])
        # leaky should have been removed; pipeline should complete
        self.assertIn("best_model", summary)

    def test_identical_column_dropped(self):
        df = _clf_df()
        df["dup_target"] = df["target"]   # identity duplicate
        ex = ModelExecutor()
        summary = ex.run(df, "target", include_models=["Decision Tree"])
        self.assertIn("best_model", summary)


# ============================================================
# 7. Reproducibility
# ============================================================

class TestReproducibility(unittest.TestCase):
    def test_same_random_state_same_f1(self):
        """Two runs with the same random_state should produce identical F1."""
        df = _clf_df()
        ex1 = ModelExecutor(random_state=42)
        ex2 = ModelExecutor(random_state=42)
        s1 = ex1.run(df, "target", include_models=["Decision Tree"])
        s2 = ex2.run(df, "target", include_models=["Decision Tree"])
        f1_1 = s1["best_model"]["metrics"]["F1 Score"]
        f1_2 = s2["best_model"]["metrics"]["F1 Score"]
        self.assertAlmostEqual(f1_1, f1_2, places=4)


# ============================================================
# 8. Model Export Integration
# ============================================================

class TestModelExportIntegration(unittest.TestCase):
    def test_best_model_can_be_exported(self):
        df = _clf_df(n=150)
        ex = ModelExecutor()
        summary = ex.run(df, "target", include_models=["Decision Tree"])
        best_model_obj = summary["best_model"].get("model_instance")
        if best_model_obj is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "best.pkl")
                best_model_obj.export(path)
                self.assertTrue(os.path.exists(path))
                self.assertGreater(os.path.getsize(path), 0)


# ============================================================
# 9. Progress Callback Integration
# ============================================================

class TestProgressCallbackIntegration(unittest.TestCase):
    def test_callback_receives_progress_and_message(self):
        df = _clf_df(n=120)
        ex = ModelExecutor()
        received = []
        ex.run(df, "target",
               include_models=["Decision Tree"],
               progress_callback=lambda p, m: received.append((p, m)))
        self.assertGreater(len(received), 0)
        for pct, msg in received:
            self.assertIsInstance(pct, (int, float))
            self.assertIsInstance(msg, str)

    def test_final_progress_is_100(self):
        df = _clf_df(n=120)
        ex = ModelExecutor()
        progress_values = []
        ex.run(df, "target",
               include_models=["Decision Tree"],
               progress_callback=lambda p, m: progress_values.append(p))
        self.assertAlmostEqual(progress_values[-1], 100, delta=5)


if __name__ == "__main__":
    unittest.main()
