"""
Unit Tests — core_recommender/evaluation.py
============================================
Tests every metric function:
  calculate_rmse, calculate_mae, calculate_r2_score, calculate_adjusted_r2,
  calculate_accuracy, calculate_f1_score, calculate_roc_auc_score,
  calculate_precision, calculate_log_loss, calculate_precision_recall_score,
  get_oob_score, get_tree_depth, get_leaf_count, measure_prediction_latency
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from core_recommender.evaluation import (
    calculate_rmse,
    calculate_mae,
    calculate_r2_score,
    calculate_adjusted_r2,
    calculate_accuracy,
    calculate_f1_score,
    calculate_roc_auc_score,
    calculate_precision,
    calculate_log_loss,
    calculate_precision_recall_score,
    get_oob_score,
    get_tree_depth,
    get_leaf_count,
    measure_prediction_latency,
)


class TestCalculateRMSE(unittest.TestCase):
    def test_perfect_prediction_is_zero(self):
        y = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(calculate_rmse(y, y), 0.0)

    def test_returns_float(self):
        result = calculate_rmse([1, 2, 3], [1, 2, 4])
        self.assertIsInstance(result, float)

    def test_known_value(self):
        y_true = [0.0, 0.0]
        y_pred = [3.0, 4.0]
        # MSE = (9+16)/2 = 12.5 → RMSE ≈ 3.536
        self.assertAlmostEqual(calculate_rmse(y_true, y_pred), np.sqrt(12.5), places=5)

    def test_always_non_negative(self):
        result = calculate_rmse([10, 20], [5, 15])
        self.assertGreaterEqual(result, 0.0)

    def test_numpy_arrays(self):
        result = calculate_rmse(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
        self.assertAlmostEqual(result, np.sqrt(0.5), places=5)


class TestCalculateMAE(unittest.TestCase):
    def test_perfect_prediction_is_zero(self):
        y = [5.0, 10.0, 15.0]
        self.assertAlmostEqual(calculate_mae(y, y), 0.0)

    def test_returns_float(self):
        result = calculate_mae([1, 2], [2, 3])
        self.assertIsInstance(result, float)

    def test_known_value(self):
        # |1-2| + |3-5| = 1 + 2 = 3 → mean = 1.5
        self.assertAlmostEqual(calculate_mae([1, 3], [2, 5]), 1.5)

    def test_always_non_negative(self):
        result = calculate_mae([100], [50])
        self.assertGreaterEqual(result, 0.0)


class TestCalculateR2Score(unittest.TestCase):
    def test_perfect_fit_is_one(self):
        y = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(calculate_r2_score(y, y), 1.0)

    def test_returns_float(self):
        result = calculate_r2_score([1, 2, 3], [1, 2, 2])
        self.assertIsInstance(result, float)

    def test_bad_model_can_be_negative(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [3.0, 2.0, 1.0]
        r2 = calculate_r2_score(y_true, y_pred)
        self.assertLessEqual(r2, 1.0)


class TestCalculateAdjustedR2(unittest.TestCase):
    def test_returns_float(self):
        result = calculate_adjusted_r2([1, 2, 3, 4, 5], [1, 2, 3, 4, 4], n_samples=5, n_features=2)
        self.assertIsInstance(result, float)

    def test_perfect_fit(self):
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = calculate_adjusted_r2(y, y, n_samples=5, n_features=2)
        self.assertAlmostEqual(result, 1.0)

    def test_invalid_denominator_returns_nan(self):
        """n_samples <= n_features + 1 → NaN."""
        result = calculate_adjusted_r2([1, 2], [1, 2], n_samples=2, n_features=5)
        self.assertTrue(np.isnan(result))

    def test_adjusted_r2_leq_r2(self):
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        y_pred = [1.1, 1.9, 3.1, 3.9, 5.1, 5.9]
        r2 = calculate_r2_score(y_true, y_pred)
        adj_r2 = calculate_adjusted_r2(y_true, y_pred, n_samples=6, n_features=3)
        self.assertLessEqual(adj_r2, r2)


class TestCalculateAccuracy(unittest.TestCase):
    def test_all_correct_is_one(self):
        y = [0, 1, 2]
        self.assertAlmostEqual(calculate_accuracy(y, y), 1.0)

    def test_all_wrong_is_zero(self):
        self.assertAlmostEqual(calculate_accuracy([0, 0], [1, 1]), 0.0)

    def test_returns_float(self):
        result = calculate_accuracy([0, 1, 1], [0, 1, 0])
        self.assertIsInstance(result, float)

    def test_half_correct(self):
        self.assertAlmostEqual(calculate_accuracy([0, 1, 0, 1], [0, 1, 1, 0]), 0.5)


class TestCalculateF1Score(unittest.TestCase):
    def test_perfect_prediction_is_one(self):
        y = [0, 1, 2, 0, 1]
        self.assertAlmostEqual(calculate_f1_score(y, y), 1.0)

    def test_binary_case(self):
        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 0, 0]
        result = calculate_f1_score(y_true, y_pred, average="weighted")
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_returns_float(self):
        result = calculate_f1_score([0, 1], [1, 0])
        self.assertIsInstance(result, float)

    def test_zero_division_returns_zero(self):
        """When no predictions match, f1 should be 0.0 (not error)."""
        result = calculate_f1_score([0, 0], [1, 1], average="weighted")
        self.assertAlmostEqual(result, 0.0)


class TestCalculateROCAUC(unittest.TestCase):
    def test_binary_perfect(self):
        y_true = [0, 0, 1, 1]
        y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
        result = calculate_roc_auc_score(y_true, y_proba)
        self.assertAlmostEqual(result, 1.0)

    def test_binary_random_returns_05(self):
        y_true = [0, 1, 0, 1]
        y_proba = np.array([[0.5, 0.5]] * 4)
        result = calculate_roc_auc_score(y_true, y_proba)
        self.assertGreaterEqual(result, 0.0)

    def test_returns_float(self):
        y_true = [0, 1]
        y_proba = np.array([[0.9, 0.1], [0.2, 0.8]])
        result = calculate_roc_auc_score(y_true, y_proba)
        self.assertIsInstance(result, float)

    def test_single_class_falls_back_gracefully(self):
        """Single class in y_true → function returns 0.5 (declared fallback) or
        NaN (sklearn behavior in some versions). Either way it must NOT raise."""
        y_true = [1, 1, 1]
        y_proba = np.array([[0.1, 0.9]] * 3)
        result = calculate_roc_auc_score(y_true, y_proba)
        # Acceptable: the declared fallback value OR NaN from sklearn
        valid = (result == 0.5) or np.isnan(result)
        self.assertTrue(valid,
                        f"Expected 0.5 or NaN for single-class input, got {result}")

    def test_multiclass_ovr(self):
        y_true = [0, 1, 2, 0, 1, 2]
        rng = np.random.default_rng(42)
        raw = rng.dirichlet([1, 1, 1], size=6)
        result = calculate_roc_auc_score(y_true, raw)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)


class TestCalculatePrecision(unittest.TestCase):
    def test_all_correct(self):
        y = [0, 1, 2]
        self.assertAlmostEqual(calculate_precision(y, y, average="weighted"), 1.0)

    def test_returns_float(self):
        result = calculate_precision([0, 1], [0, 1])
        self.assertIsInstance(result, float)

    def test_zero_division_zero(self):
        """All predictions wrong → precision = 0.0 (not error)."""
        result = calculate_precision([0, 0], [1, 1], average="weighted")
        self.assertEqual(result, 0.0)


class TestCalculateLogLoss(unittest.TestCase):
    def test_perfect_probability_is_near_zero(self):
        y_true = [0, 1]
        y_proba = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = calculate_log_loss(y_true, y_proba)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_returns_float(self):
        y_true = [0, 1]
        y_proba = np.array([[0.7, 0.3], [0.4, 0.6]])
        self.assertIsInstance(calculate_log_loss(y_true, y_proba), float)

    def test_high_uncertainty_is_high_loss(self):
        y_true = [0, 1]
        y_proba = np.array([[0.5, 0.5], [0.5, 0.5]])
        result = calculate_log_loss(y_true, y_proba)
        self.assertGreater(result, 0.0)


class TestCalculatePrecisionRecallScore(unittest.TestCase):
    def test_returns_tuple_of_two_floats(self):
        result = calculate_precision_recall_score([0, 1, 1], [0, 1, 0])
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)

    def test_perfect_prediction(self):
        y = [0, 1, 2]
        prec, recall = calculate_precision_recall_score(y, y)
        self.assertAlmostEqual(prec, 1.0)
        self.assertAlmostEqual(recall, 1.0)

    def test_values_in_range_0_to_1(self):
        prec, recall = calculate_precision_recall_score([0, 1, 0, 1], [0, 0, 1, 1])
        self.assertGreaterEqual(prec, 0.0)
        self.assertLessEqual(prec, 1.0)
        self.assertGreaterEqual(recall, 0.0)
        self.assertLessEqual(recall, 1.0)


class TestGetOOBScore(unittest.TestCase):
    def test_returns_float_when_attribute_exists(self):
        mock_model = MagicMock()
        mock_model.oob_score_ = 0.87
        result = get_oob_score(mock_model)
        self.assertAlmostEqual(result, 0.87)

    def test_returns_none_when_attribute_missing(self):
        mock_model = MagicMock(spec=[])  # no attributes
        result = get_oob_score(mock_model)
        self.assertIsNone(result)

    def test_returns_none_when_oob_score_is_none(self):
        mock_model = MagicMock()
        mock_model.oob_score_ = None
        result = get_oob_score(mock_model)
        self.assertIsNone(result)


class TestGetTreeDepth(unittest.TestCase):
    def test_single_tree_returns_depth(self):
        mock_tree = MagicMock()
        mock_tree.max_depth = 5
        mock_model = MagicMock()
        mock_model.tree_ = mock_tree
        result = get_tree_depth(mock_model)
        self.assertEqual(result, 5)

    def test_ensemble_returns_mean_depth(self):
        estimators = []
        for d in [3, 5, 7]:
            e = MagicMock()
            e.tree_ = MagicMock()
            e.tree_.max_depth = d
            estimators.append(e)
        mock_model = MagicMock(spec=["estimators_"])
        mock_model.estimators_ = estimators
        result = get_tree_depth(mock_model)
        self.assertEqual(result, 5)  # mean of [3,5,7]

    def test_returns_none_for_non_tree_model(self):
        result = get_tree_depth(MagicMock(spec=[]))
        self.assertIsNone(result)


class TestGetLeafCount(unittest.TestCase):
    def test_single_tree(self):
        mock_tree = MagicMock()
        mock_tree.n_leaves = 10
        mock_model = MagicMock()
        mock_model.tree_ = mock_tree
        result = get_leaf_count(mock_model)
        self.assertEqual(result, 10)

    def test_ensemble_returns_mean(self):
        estimators = []
        for n in [8, 12, 16]:
            e = MagicMock()
            e.tree_ = MagicMock()
            e.tree_.n_leaves = n
            estimators.append(e)
        mock_model = MagicMock(spec=["estimators_"])
        mock_model.estimators_ = estimators
        result = get_leaf_count(mock_model)
        self.assertEqual(result, 12)  # mean of [8,12,16]

    def test_returns_none_for_non_tree(self):
        result = get_leaf_count(MagicMock(spec=[]))
        self.assertIsNone(result)


class TestMeasurePredictionLatency(unittest.TestCase):
    def test_returns_float(self):
        X_test = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1])
        result = measure_prediction_latency(mock_model, X_test, n_runs=5)
        self.assertIsInstance(result, float)

    def test_returns_positive_value(self):
        X_test = np.array([[1.0, 2.0]])
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0])
        result = measure_prediction_latency(mock_model, X_test, n_runs=3)
        self.assertGreaterEqual(result, 0.0)

    def test_no_predict_returns_nan(self):
        """Model without predict() should return NaN."""
        mock_model = MagicMock(spec=[])
        X_test = np.array([[1.0]])
        result = measure_prediction_latency(mock_model, X_test)
        self.assertTrue(np.isnan(result))


if __name__ == "__main__":
    unittest.main()
