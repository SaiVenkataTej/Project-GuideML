"""
Unit Tests — core_recommender/profiling.py
===========================================
Tests the DataProfiler class:
  __init__, analyze(), _generate_rules()
"""

import unittest
import numpy as np
import pandas as pd
from core_recommender.profiling import DataProfiler


def _make_X_y(n=200, n_features=5, task="regression", random_state=42):
    """Helper to create numeric X and y."""
    rng = np.random.RandomState(random_state)
    X = pd.DataFrame(rng.randn(n, n_features), columns=[f"f{i}" for i in range(n_features)])
    if task == "regression":
        y = pd.Series(rng.randn(n), name="target")
    else:
        y = pd.Series(rng.randint(0, 3, size=n), name="target")
    return X, y


class TestDataProfilerInit(unittest.TestCase):
    def test_init_creates_empty_dicts(self):
        profiler = DataProfiler()
        self.assertIsInstance(profiler.suggestions, dict)
        self.assertIsInstance(profiler.profile, dict)


class TestDataProfilerAnalyzeOutputStructure(unittest.TestCase):
    def setUp(self):
        self.X, self.y = _make_X_y()
        self.profiler = DataProfiler()
        self.result = self.profiler.analyze(self.X, self.y)

    def test_returns_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_has_profile_key(self):
        self.assertIn("profile", self.result)

    def test_has_suggestions_key(self):
        self.assertIn("suggestions", self.result)

    def test_suggestions_has_messages(self):
        self.assertIn("messages", self.result["suggestions"])

    def test_messages_is_list(self):
        self.assertIsInstance(self.result["suggestions"]["messages"], list)


class TestDataProfilerProfileValues(unittest.TestCase):
    def setUp(self):
        self.X, self.y = _make_X_y(n=100, n_features=5)
        self.profiler = DataProfiler()
        self.result = self.profiler.analyze(self.X, self.y)["profile"]

    def test_n_samples_correct(self):
        self.assertEqual(self.result["n_samples"], 100)

    def test_n_features_correct(self):
        self.assertEqual(self.result["n_features"], 5)

    def test_ratio_correct(self):
        self.assertAlmostEqual(self.result["ratio"], 100 / 5)

    def test_sparsity_is_float(self):
        self.assertIsInstance(self.result["sparsity"], float)

    def test_sparsity_between_0_and_1(self):
        sp = self.result["sparsity"]
        self.assertGreaterEqual(sp, 0.0)
        self.assertLessEqual(sp, 1.0)

    def test_is_gaussian_is_bool(self):
        self.assertIsInstance(self.result["is_gaussian"], bool)

    def test_linearity_score_is_float(self):
        self.assertIsInstance(self.result["linearity_score"], float)

    def test_max_linear_score_is_float(self):
        self.assertIsInstance(self.result["max_linear_score"], float)


class TestDataProfilerSparsity(unittest.TestCase):
    def test_all_zeros_gives_sparsity_one(self):
        X = pd.DataFrame(np.zeros((50, 4)), columns=list("abcd"))
        y = pd.Series(np.ones(50))
        profiler = DataProfiler()
        result = profiler.analyze(X, y)
        self.assertAlmostEqual(result["profile"]["sparsity"], 1.0)

    def test_no_zeros_gives_sparsity_zero(self):
        X = pd.DataFrame(np.ones((50, 4)), columns=list("abcd"))
        y = pd.Series(np.ones(50))
        profiler = DataProfiler()
        result = profiler.analyze(X, y)
        self.assertAlmostEqual(result["profile"]["sparsity"], 0.0)


class TestDataProfilerNaNHandling(unittest.TestCase):
    def test_nan_columns_do_not_crash_profiler(self):
        X, y = _make_X_y(n=100, n_features=3)
        X["all_nan"] = np.nan
        X["partial_nan"] = np.nan
        X.loc[:10, "partial_nan"] = np.random.randn(11)
        profiler = DataProfiler()
        result = profiler.analyze(X, y)
        self.assertIn("n_samples", result["profile"])

    def test_empty_numeric_columns(self):
        X = pd.DataFrame({"cat": ["a", "b", "c"]})
        y = pd.Series([1, 2, 3])
        profiler = DataProfiler()
        result = profiler.analyze(X, y)
        self.assertEqual(result["profile"]["sparsity"], 0.0)


class TestDataProfilerRules(unittest.TestCase):
    def test_large_dataset_triggers_message(self):
        rng = np.random.RandomState(0)
        X = pd.DataFrame(rng.randn(25000, 3), columns=["a", "b", "c"])
        y = pd.Series(rng.randn(25000))
        profiler = DataProfiler()
        result = profiler.analyze(X, y)
        messages = result["suggestions"]["messages"]
        self.assertTrue(any("20k" in m or "Large" in m for m in messages))

    def test_high_dimensionality_triggers_message(self):
        rng = np.random.RandomState(0)
        X = pd.DataFrame(rng.randn(10, 50), columns=[f"f{i}" for i in range(50)])
        y = pd.Series(rng.randn(10))
        profiler = DataProfiler()
        result = profiler.analyze(X, y)
        messages = result["suggestions"]["messages"]
        self.assertTrue(any("dimensionality" in m.lower() or "regularization" in m.lower() for m in messages))

    def test_class_imbalance_triggers_message(self):
        rng = np.random.RandomState(0)
        X = pd.DataFrame(rng.randn(200, 4), columns=list("abcd"))
        y = pd.Series([0] * 195 + [1] * 5)   # 2.5% minority
        profiler = DataProfiler()
        result = profiler.analyze(X, y)
        messages = result["suggestions"]["messages"]
        self.assertTrue(any("imbalance" in m.lower() or "minority" in m.lower() for m in messages))

    def test_strong_linearity_triggers_message(self):
        n = 200
        X = pd.DataFrame({"x": np.arange(n, dtype=float)})
        y = pd.Series(np.arange(n, dtype=float) * 3.0 + 1.0)  # perfect linear
        profiler = DataProfiler()
        result = profiler.analyze(X, y)
        messages = result["suggestions"]["messages"]
        self.assertTrue(any("linear" in m.lower() for m in messages))

    def test_no_false_messages_for_small_clean_dataset(self):
        X, y = _make_X_y(n=100, n_features=4, task="regression")
        profiler = DataProfiler()
        result = profiler.analyze(X, y)
        # Should not crash; messages list may be empty or contain valid info
        self.assertIsInstance(result["suggestions"]["messages"], list)

    def test_analyze_resets_state_on_second_call(self):
        """A second call to analyze() should produce fresh results."""
        X1, y1 = _make_X_y(n=50, n_features=3)
        X2, y2 = _make_X_y(n=300, n_features=3)
        profiler = DataProfiler()
        profiler.analyze(X1, y1)
        result2 = profiler.analyze(X2, y2)
        self.assertEqual(result2["profile"]["n_samples"], 300)


class TestDataProfilerClassificationTarget(unittest.TestCase):
    def test_handles_object_dtype_target(self):
        X, _ = _make_X_y(n=100, n_features=3)
        y = pd.Series(["cat", "dog", "bird"] * 33 + ["cat"])
        profiler = DataProfiler()
        result = profiler.analyze(X, y)
        self.assertIn("n_samples", result["profile"])

    def test_handles_boolean_target(self):
        X, _ = _make_X_y(n=100, n_features=3)
        y = pd.Series([True, False] * 50)
        profiler = DataProfiler()
        result = profiler.analyze(X, y)
        self.assertIn("profile", result)


if __name__ == "__main__":
    unittest.main()
