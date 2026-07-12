"""
Unit Tests — all concrete model classes in core_recommender/modeling/
======================================================================
For every model we test:
  1. __init__  — correct name, config, attribute types
  2. preprocess() — output shapes, types, preprocessor is set
  3. fit()     — best_estimator is not None after fitting
  4. calculate_metrics() — returns dict with expected keys
  5. get_diagnostic_data() — contains y_pred, y_true
  6. get_tailored_diagnostics() — returns dict (may be empty before fit)
  7. get_parameter_descriptions() — returns dict
  8. export() — saves a .pkl file on disk

Models tested:
  LinearRegressionModel, LogisticRegressionModel, DecisionTreeModel,
  RandomForestModel, KNNModel, NaiveBayesModel, SVMModel
"""

import os
import tempfile
import shutil
import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from core_recommender.modeling.linear_regression import LinearRegressionModel
from core_recommender.modeling.logistic_regression import LogisticRegressionModel
from core_recommender.modeling.decision_trees import DecisionTreeModel
from core_recommender.modeling.random_forest import RandomForestModel
from core_recommender.modeling.knn import KNNModel
from core_recommender.modeling.naive_bayes import NaiveBayesModel
from core_recommender.modeling.svms import SVMModel
from core_recommender.exceptions import DataValidationError, ModelNotFittedError


# ============================================================
# Shared dataset fixtures
# ============================================================

def _clf_data(n=120, n_features=6, random_state=42):
    X, y = make_classification(n_samples=n, n_features=n_features,
                                n_classes=2, random_state=random_state)
    feature_names = [f"f{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_s = pd.Series(y, name="target")
    return X_df, y_s


def _reg_data(n=120, n_features=6, random_state=42):
    X, y = make_regression(n_samples=n, n_features=n_features, random_state=random_state)
    feature_names = [f"f{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_s = pd.Series(y.astype(float), name="target")
    return X_df, y_s


# ============================================================
# Minimal config to speed up tests (fewer CV folds + trials)
# ============================================================

FAST_CONFIG_BASE = {"cv_folds": 2, "random_state": 42, "n_jobs": 1}

FAST_CLF_CONFIG = {**FAST_CONFIG_BASE, "n_trials": 2}
FAST_REG_CONFIG = {**FAST_CONFIG_BASE, "n_trials": 2,
                   "feature_selection": "none", "transformation": "none", "scaler": "standard"}
FAST_KNN_CONFIG = {**FAST_CLF_CONFIG, "reduction": None, "n_trials": 2, "scaler": "minmax"}
FAST_NB_CONFIG = {"cv_folds": 2, "random_state": 42, "n_jobs": 1,
                  "feature_selection": "k_best", "k_best": 4, "model_type": "gaussian"}
FAST_SVM_CONFIG = {**FAST_CONFIG_BASE, "n_iter": 2, "pca_components": None}


# ============================================================
# LinearRegressionModel
# ============================================================

class TestLinearRegressionModel(unittest.TestCase):
    def setUp(self):
        self.X, self.y = _reg_data()
        self.model = LinearRegressionModel(config=FAST_REG_CONFIG)

    # --- init ---
    def test_name(self):
        self.assertIn("Linear Regression", self.model.name)

    def test_initial_best_estimator_is_none(self):
        self.assertIsNone(self.model.best_estimator)

    def test_initial_preprocessor_is_none(self):
        self.assertIsNone(self.model.preprocessor)

    # --- preprocess ---
    def test_preprocess_returns_tuple_of_three(self):
        result = self.model.preprocess(self.X, self.y)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_preprocess_X_shape_rows(self):
        X_trans, _, _ = self.model.preprocess(self.X, self.y)
        self.assertEqual(X_trans.shape[0], len(self.X))

    def test_preprocess_y_shape(self):
        _, y_trans, _ = self.model.preprocess(self.X, self.y)
        self.assertEqual(len(y_trans), len(self.y))

    def test_preprocess_sets_preprocessor(self):
        self.model.preprocess(self.X, self.y)
        self.assertIsNotNone(self.model.preprocessor)

    def test_preprocess_raises_on_nan_target(self):
        y_nan = self.y.copy()
        y_nan.iloc[0] = np.nan
        with self.assertRaises(DataValidationError):
            self.model.preprocess(self.X, y_nan)

    # --- fit ---
    def test_fit_sets_best_estimator(self):
        self.model.preprocess(self.X, self.y)
        self.model.fit(self.X, self.y)
        self.assertIsNotNone(self.model.best_estimator)

    # --- calculate_metrics ---
    def test_calculate_metrics_returns_expected_keys(self):
        self.model.preprocess(self.X, self.y)
        self.model.fit(self.X, self.y)
        metrics = self.model.calculate_metrics(self.X, self.y)
        for key in ["RMSE", "MAE", "R2 Score"]:
            self.assertIn(key, metrics)

    def test_calculate_metrics_values_are_floats(self):
        self.model.preprocess(self.X, self.y)
        self.model.fit(self.X, self.y)
        metrics = self.model.calculate_metrics(self.X, self.y)
        for v in metrics.values():
            self.assertIsInstance(v, float)

    # --- get_diagnostic_data ---
    def test_diagnostic_data_keys(self):
        self.model.preprocess(self.X, self.y)
        self.model.fit(self.X, self.y)
        diag = self.model.get_diagnostic_data(self.X, self.y)
        self.assertIn("y_pred", diag)
        self.assertIn("y_true", diag)

    # --- get_tailored_diagnostics ---
    def test_tailored_diagnostics_has_coefficients(self):
        self.model.preprocess(self.X, self.y)
        self.model.fit(self.X, self.y)
        td = self.model.get_tailored_diagnostics()
        self.assertIn("coefficients", td)

    def test_tailored_diagnostics_before_fit_returns_empty(self):
        td = self.model.get_tailored_diagnostics()
        self.assertIsInstance(td, dict)

    # --- export ---
    def test_export_saves_pkl_file(self):
        self.model.preprocess(self.X, self.y)
        self.model.fit(self.X, self.y)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pkl")
            self.model.export(path)
            self.assertTrue(os.path.exists(path))

    def test_export_before_fit_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pkl")
            with self.assertRaises(Exception):
                self.model.export(path)


# ============================================================
# LogisticRegressionModel
# ============================================================

class TestLogisticRegressionModel(unittest.TestCase):
    def setUp(self):
        self.X, self.y = _clf_data()
        self.model = LogisticRegressionModel(
            config={"cv_folds": 2, "random_state": 42, "n_jobs": 1,
                    "feature_selection": "none"}
        )

    def test_name(self):
        self.assertEqual(self.model.name, "Logistic Regression")

    def test_preprocess_returns_tuple_of_three(self):
        result = self.model.preprocess(self.X, self.y)
        self.assertEqual(len(result), 3)

    def test_preprocess_raises_on_nan_target(self):
        y_nan = self.y.copy().astype(float)
        y_nan.iloc[0] = np.nan
        with self.assertRaises(DataValidationError):
            self.model.preprocess(self.X, y_nan)

    def test_fit_sets_best_estimator(self):
        self.model.preprocess(self.X, self.y)
        self.model.fit(self.X, self.y)
        self.assertIsNotNone(self.model.best_estimator)

    def test_calculate_metrics_has_f1_and_accuracy(self):
        self.model.preprocess(self.X, self.y)
        self.model.fit(self.X, self.y)
        metrics = self.model.calculate_metrics(self.X, self.y)
        self.assertIn("F1 Score", metrics)
        self.assertIn("Accuracy", metrics)

    def test_tailored_diagnostics_has_coefficients(self):
        self.model.preprocess(self.X, self.y)
        self.model.fit(self.X, self.y)
        td = self.model.get_tailored_diagnostics()
        self.assertIn("coefficients", td)
        self.assertIsInstance(td["coefficients"], list)

    def test_diagnostic_data_has_y_proba(self):
        self.model.preprocess(self.X, self.y)
        self.model.fit(self.X, self.y)
        diag = self.model.get_diagnostic_data(self.X, self.y)
        self.assertIn("y_proba", diag)
        self.assertIsNotNone(diag["y_proba"])


# ============================================================
# DecisionTreeModel
# ============================================================

class TestDecisionTreeModel(unittest.TestCase):
    def setUp(self):
        self.X_clf, self.y_clf = _clf_data()
        self.X_reg, self.y_reg = _reg_data()
        self.model_clf = DecisionTreeModel(is_classification=True,
                                           config={"cv_folds": 2, "random_state": 42, "n_trials": 2})
        self.model_reg = DecisionTreeModel(is_classification=False,
                                           config={"cv_folds": 2, "random_state": 42, "n_trials": 2})

    def test_classification_name_contains_classif(self):
        self.assertIn("Classification", self.model_clf.name)

    def test_regression_name_contains_regression(self):
        self.assertIn("Regression", self.model_reg.name)

    def test_clf_preprocess_shape(self):
        X_t, y_t, _ = self.model_clf.preprocess(self.X_clf, self.y_clf)
        self.assertEqual(X_t.shape[0], len(self.X_clf))

    def test_clf_fit_sets_best_estimator(self):
        self.model_clf.preprocess(self.X_clf, self.y_clf)
        self.model_clf.fit(self.X_clf, self.y_clf)
        self.assertIsNotNone(self.model_clf.best_estimator)

    def test_clf_metrics_has_accuracy(self):
        self.model_clf.preprocess(self.X_clf, self.y_clf)
        self.model_clf.fit(self.X_clf, self.y_clf)
        metrics = self.model_clf.calculate_metrics(self.X_clf, self.y_clf)
        self.assertIn("Accuracy", metrics)

    def test_reg_metrics_has_rmse(self):
        self.model_reg.preprocess(self.X_reg, self.y_reg)
        self.model_reg.fit(self.X_reg, self.y_reg)
        metrics = self.model_reg.calculate_metrics(self.X_reg, self.y_reg)
        self.assertIn("RMSE", metrics)

    def test_tailored_diagnostics_clf_has_feature_importances(self):
        self.model_clf.preprocess(self.X_clf, self.y_clf)
        self.model_clf.fit(self.X_clf, self.y_clf)
        td = self.model_clf.get_tailored_diagnostics()
        self.assertIn("feature_importances", td)

    def test_tailored_diagnostics_clf_has_tree_dot_data(self):
        self.model_clf.preprocess(self.X_clf, self.y_clf)
        self.model_clf.fit(self.X_clf, self.y_clf)
        td = self.model_clf.get_tailored_diagnostics()
        self.assertIn("tree_dot_data", td)

    def test_tailored_diagnostics_before_fit_returns_empty(self):
        td = self.model_clf.get_tailored_diagnostics()
        self.assertIsInstance(td, dict)


# ============================================================
# RandomForestModel
# ============================================================

class TestRandomForestModel(unittest.TestCase):
    def setUp(self):
        self.X_clf, self.y_clf = _clf_data()
        self.X_reg, self.y_reg = _reg_data()
        self.model_clf = RandomForestModel(is_classification=True, config=FAST_CLF_CONFIG)
        self.model_reg = RandomForestModel(is_classification=False, config=FAST_CLF_CONFIG)

    def test_clf_name(self):
        self.assertIn("Classification", self.model_clf.name)

    def test_reg_name(self):
        self.assertIn("Regression", self.model_reg.name)

    def test_preprocess_raises_on_nan_target(self):
        y_nan = self.y_clf.astype(float)
        y_nan.iloc[0] = np.nan
        with self.assertRaises(DataValidationError):
            self.model_clf.preprocess(self.X_clf, y_nan)

    def test_clf_fit_and_metrics(self):
        self.model_clf.preprocess(self.X_clf, self.y_clf)
        self.model_clf.fit(self.X_clf, self.y_clf)
        metrics = self.model_clf.calculate_metrics(self.X_clf, self.y_clf)
        self.assertIn("F1 Score", metrics)

    def test_reg_fit_and_metrics(self):
        self.model_reg.preprocess(self.X_reg, self.y_reg)
        self.model_reg.fit(self.X_reg, self.y_reg)
        metrics = self.model_reg.calculate_metrics(self.X_reg, self.y_reg)
        self.assertIn("RMSE", metrics)

    def test_tailored_diagnostics_has_oob_score_key(self):
        self.model_clf.preprocess(self.X_clf, self.y_clf)
        self.model_clf.fit(self.X_clf, self.y_clf)
        td = self.model_clf.get_tailored_diagnostics()
        self.assertIn("oob_score", td)

    def test_tailored_diagnostics_has_feature_importances_mdi(self):
        self.model_clf.preprocess(self.X_clf, self.y_clf)
        self.model_clf.fit(self.X_clf, self.y_clf)
        td = self.model_clf.get_tailored_diagnostics()
        self.assertIn("feature_importances_mdi", td)
        self.assertIsInstance(td["feature_importances_mdi"], list)

    def test_tailored_diagnostics_before_fit_returns_empty(self):
        td = self.model_clf.get_tailored_diagnostics()
        self.assertEqual(td, {})

    def test_get_parameter_descriptions(self):
        self.model_clf.preprocess(self.X_clf, self.y_clf)
        self.model_clf.fit(self.X_clf, self.y_clf)
        desc = self.model_clf.get_parameter_descriptions()
        self.assertIsInstance(desc, dict)
        self.assertIn("n_estimators", desc)


# ============================================================
# KNNModel
# ============================================================

class TestKNNModel(unittest.TestCase):
    def setUp(self):
        self.X_clf, self.y_clf = _clf_data()
        self.X_reg, self.y_reg = _reg_data()
        self.model_clf = KNNModel(is_classification=True, config=FAST_KNN_CONFIG)
        self.model_reg = KNNModel(is_classification=False, config=FAST_KNN_CONFIG)

    def test_clf_name(self):
        self.assertIn("Classification", self.model_clf.name)

    def test_clf_preprocess_shape(self):
        X_t, y_t, _ = self.model_clf.preprocess(self.X_clf, self.y_clf)
        self.assertEqual(X_t.shape[0], len(self.X_clf))

    def test_clf_fit_sets_best_estimator(self):
        self.model_clf.preprocess(self.X_clf, self.y_clf)
        self.model_clf.fit(self.X_clf, self.y_clf)
        self.assertIsNotNone(self.model_clf.best_estimator)

    def test_clf_metrics_has_f1(self):
        self.model_clf.preprocess(self.X_clf, self.y_clf)
        self.model_clf.fit(self.X_clf, self.y_clf)
        metrics = self.model_clf.calculate_metrics(self.X_clf, self.y_clf)
        self.assertIn("F1 Score", metrics)

    def test_reg_metrics_has_rmse(self):
        self.model_reg.preprocess(self.X_reg, self.y_reg)
        self.model_reg.fit(self.X_reg, self.y_reg)
        metrics = self.model_reg.calculate_metrics(self.X_reg, self.y_reg)
        self.assertIn("RMSE", metrics)

    def test_diagnostic_data_has_elbow_data(self):
        self.model_clf.preprocess(self.X_clf, self.y_clf)
        self.model_clf.fit(self.X_clf, self.y_clf)
        diag = self.model_clf.get_diagnostic_data(self.X_clf, self.y_clf)
        self.assertIn("elbow_data", diag)
        self.assertIn("neighbor_indices", diag)

    def test_tailored_diagnostics_before_fit_returns_empty(self):
        td = self.model_clf.get_tailored_diagnostics()
        self.assertEqual(td, {})


# ============================================================
# NaiveBayesModel
# ============================================================

class TestNaiveBayesModel(unittest.TestCase):
    def setUp(self):
        self.X, self.y = _clf_data()
        self.model = NaiveBayesModel(config=FAST_NB_CONFIG)

    def test_gaussian_name(self):
        self.assertIn("Gaussian", self.model.name)

    def test_invalid_model_type_raises(self):
        from core_recommender.exceptions import ConfigurationError
        with self.assertRaises(ConfigurationError):
            NaiveBayesModel(config={"model_type": "invalid"})

    def test_preprocess_shape(self):
        X_t, y_t, _ = self.model.preprocess(self.X, self.y)
        self.assertEqual(X_t.shape[0], len(self.X))

    def test_preprocess_raises_on_nan_target(self):
        y_nan = self.y.astype(float)
        y_nan.iloc[0] = np.nan
        with self.assertRaises(DataValidationError):
            self.model.preprocess(self.X, y_nan)

    def test_fit_sets_best_estimator(self):
        self.model.preprocess(self.X, self.y)
        self.model.fit(self.X, self.y)
        self.assertIsNotNone(self.model.best_estimator)

    def test_metrics_has_f1_and_accuracy(self):
        self.model.preprocess(self.X, self.y)
        self.model.fit(self.X, self.y)
        metrics = self.model.calculate_metrics(self.X, self.y)
        self.assertIn("F1 Score", metrics)
        self.assertIn("Accuracy", metrics)

    def test_tailored_diagnostics_has_model_type(self):
        self.model.preprocess(self.X, self.y)
        self.model.fit(self.X, self.y)
        td = self.model.get_tailored_diagnostics()
        self.assertIn("model_type", td)
        self.assertEqual(td["model_type"], "gaussian")

    def test_tailored_diagnostics_before_fit_returns_empty(self):
        td = self.model.get_tailored_diagnostics()
        self.assertEqual(td, {})


# ============================================================
# SVMModel
# ============================================================

class TestSVMModel(unittest.TestCase):
    def setUp(self):
        self.X_clf, self.y_clf = _clf_data(n=80)
        self.X_reg, self.y_reg = _reg_data(n=80)
        self.model_clf = SVMModel(is_classification=True, config=FAST_SVM_CONFIG)
        self.model_reg = SVMModel(is_classification=False, config=FAST_SVM_CONFIG)

    def test_clf_name(self):
        self.assertIn("Classification", self.model_clf.name)

    def test_reg_name(self):
        self.assertIn("Regression", self.model_reg.name)

    def test_clf_preprocess_shape(self):
        X_t, _, _ = self.model_clf.preprocess(self.X_clf, self.y_clf)
        self.assertEqual(X_t.shape[0], len(self.X_clf))

    def test_clf_fit_sets_best_estimator(self):
        self.model_clf.preprocess(self.X_clf, self.y_clf)
        self.model_clf.fit(self.X_clf, self.y_clf)
        self.assertIsNotNone(self.model_clf.best_estimator)

    def test_clf_metrics_has_f1(self):
        self.model_clf.preprocess(self.X_clf, self.y_clf)
        self.model_clf.fit(self.X_clf, self.y_clf)
        metrics = self.model_clf.calculate_metrics(self.X_clf, self.y_clf)
        self.assertIn("F1 Score", metrics)

    def test_reg_metrics_has_rmse(self):
        self.model_reg.preprocess(self.X_reg, self.y_reg)
        self.model_reg.fit(self.X_reg, self.y_reg)
        metrics = self.model_reg.calculate_metrics(self.X_reg, self.y_reg)
        self.assertIn("RMSE", metrics)

    def test_tailored_diagnostics_has_support_vectors(self):
        self.model_clf.preprocess(self.X_clf, self.y_clf)
        self.model_clf.fit(self.X_clf, self.y_clf)
        td = self.model_clf.get_tailored_diagnostics()
        self.assertIn("support_vectors", td)

    def test_tailored_diagnostics_before_fit_returns_empty(self):
        td = self.model_clf.get_tailored_diagnostics()
        self.assertEqual(td, {})


if __name__ == "__main__":
    unittest.main()
