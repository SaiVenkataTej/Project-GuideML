"""
Unit Tests — core_recommender/preprocessing.py
===============================================
Tests every factory function that returns a configured sklearn transformer:
  get_imputer, get_one_hot_encoder, get_ordinal_encoder,
  get_standard_scaler, get_minmax_scaler, get_robust_scaler,
  get_log_transformer, get_box_cox_transformer, get_yeo_johnson_transformer,
  get_variance_threshold, get_select_k_best, get_rfe_selector,
  get_select_from_model, get_pca_reducer, get_nca_reducer
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, OrdinalEncoder, FunctionTransformer, PowerTransformer
)
from sklearn.feature_selection import VarianceThreshold, SelectKBest, RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis

from core_recommender.preprocessing import (
    get_imputer,
    get_one_hot_encoder,
    get_ordinal_encoder,
    get_standard_scaler,
    get_minmax_scaler,
    get_robust_scaler,
    get_log_transformer,
    get_box_cox_transformer,
    get_yeo_johnson_transformer,
    get_variance_threshold,
    get_select_k_best,
    get_rfe_selector,
    get_select_from_model,
    get_pca_reducer,
    get_nca_reducer,
)
from core_recommender.exceptions import ConfigurationError


class TestGetImputer(unittest.TestCase):
    def test_returns_simple_imputer(self):
        imp = get_imputer()
        self.assertIsInstance(imp, SimpleImputer)

    def test_default_strategy_is_median(self):
        imp = get_imputer()
        self.assertEqual(imp.strategy, "median")

    def test_mean_strategy(self):
        imp = get_imputer(strategy="mean")
        self.assertEqual(imp.strategy, "mean")

    def test_most_frequent_strategy(self):
        imp = get_imputer(strategy="most_frequent")
        self.assertEqual(imp.strategy, "most_frequent")

    def test_constant_strategy_with_fill_value(self):
        imp = get_imputer(strategy="constant", fill_value=0)
        self.assertEqual(imp.strategy, "constant")
        self.assertEqual(imp.fill_value, 0)

    def test_imputer_can_transform_nans(self):
        imp = get_imputer(strategy="median")
        X = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 6.0]])
        imp.fit(X)
        result = imp.transform(X)
        self.assertFalse(np.isnan(result).any())


class TestGetOneHotEncoder(unittest.TestCase):
    def test_returns_one_hot_encoder(self):
        enc = get_one_hot_encoder()
        self.assertIsInstance(enc, OneHotEncoder)

    def test_default_handle_unknown_is_ignore(self):
        enc = get_one_hot_encoder()
        self.assertEqual(enc.handle_unknown, "ignore")

    def test_sparse_output_false_by_default(self):
        enc = get_one_hot_encoder()
        self.assertFalse(enc.sparse_output)

    def test_can_fit_transform(self):
        enc = get_one_hot_encoder()
        X = np.array([["cat"], ["dog"], ["cat"]])
        result = enc.fit_transform(X)
        self.assertEqual(result.shape[0], 3)


class TestGetOrdinalEncoder(unittest.TestCase):
    def test_returns_ordinal_encoder(self):
        enc = get_ordinal_encoder()
        self.assertIsInstance(enc, OrdinalEncoder)

    def test_default_handle_unknown(self):
        enc = get_ordinal_encoder()
        self.assertEqual(enc.handle_unknown, "use_encoded_value")

    def test_unknown_value_default(self):
        enc = get_ordinal_encoder()
        self.assertEqual(enc.unknown_value, -1)

    def test_can_fit_transform(self):
        enc = get_ordinal_encoder()
        X = np.array([["S"], ["M"], ["L"]])
        result = enc.fit_transform(X)
        self.assertEqual(result.shape, (3, 1))


class TestGetStandardScaler(unittest.TestCase):
    def test_returns_standard_scaler(self):
        scaler = get_standard_scaler()
        self.assertIsInstance(scaler, StandardScaler)

    def test_transforms_to_zero_mean(self):
        scaler = get_standard_scaler()
        X = np.array([[1.0], [2.0], [3.0]])
        result = scaler.fit_transform(X)
        self.assertAlmostEqual(result.mean(), 0.0, places=10)

    def test_transforms_to_unit_variance(self):
        scaler = get_standard_scaler()
        X = np.array([[1.0], [2.0], [3.0]])
        result = scaler.fit_transform(X)
        self.assertAlmostEqual(result.std(), 1.0, places=10)


class TestGetMinMaxScaler(unittest.TestCase):
    def test_returns_minmax_scaler(self):
        scaler = get_minmax_scaler()
        self.assertIsInstance(scaler, MinMaxScaler)

    def test_range_is_0_to_1(self):
        scaler = get_minmax_scaler()
        X = np.array([[0.0], [5.0], [10.0]])
        result = scaler.fit_transform(X)
        self.assertAlmostEqual(result.min(), 0.0, places=10)
        self.assertAlmostEqual(result.max(), 1.0, places=10)


class TestGetRobustScaler(unittest.TestCase):
    def test_returns_robust_scaler(self):
        scaler = get_robust_scaler()
        self.assertIsInstance(scaler, RobustScaler)

    def test_can_fit_transform(self):
        scaler = get_robust_scaler()
        X = np.array([[1.0], [2.0], [3.0], [100.0]])  # outlier
        result = scaler.fit_transform(X)
        self.assertEqual(result.shape, (4, 1))


class TestGetLogTransformer(unittest.TestCase):
    def test_returns_function_transformer(self):
        t = get_log_transformer()
        self.assertIsInstance(t, FunctionTransformer)

    def test_applies_log1p(self):
        t = get_log_transformer()
        X = np.array([[0.0], [1.0], [4.0]])
        result = t.fit_transform(X)
        expected = np.log1p(X)
        np.testing.assert_array_almost_equal(result, expected)

    def test_inverse_is_expm1(self):
        t = get_log_transformer()
        X = np.array([[0.0], [1.0], [9.0]])
        transformed = t.fit_transform(X)
        inverse = t.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(inverse, X)


class TestGetBoxCoxTransformer(unittest.TestCase):
    def test_returns_power_transformer(self):
        t = get_box_cox_transformer()
        self.assertIsInstance(t, PowerTransformer)

    def test_method_is_box_cox(self):
        t = get_box_cox_transformer()
        self.assertEqual(t.method, "box-cox")

    def test_can_fit_positive_data(self):
        t = get_box_cox_transformer()
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        result = t.fit_transform(X)
        self.assertEqual(result.shape, (4, 1))


class TestGetYeoJohnsonTransformer(unittest.TestCase):
    def test_returns_power_transformer(self):
        t = get_yeo_johnson_transformer()
        self.assertIsInstance(t, PowerTransformer)

    def test_method_is_yeo_johnson(self):
        t = get_yeo_johnson_transformer()
        self.assertEqual(t.method, "yeo-johnson")

    def test_handles_negative_values(self):
        t = get_yeo_johnson_transformer()
        X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
        result = t.fit_transform(X)
        self.assertEqual(result.shape, (5, 1))


class TestGetVarianceThreshold(unittest.TestCase):
    def test_returns_variance_threshold(self):
        vt = get_variance_threshold()
        self.assertIsInstance(vt, VarianceThreshold)

    def test_default_threshold_is_zero(self):
        vt = get_variance_threshold()
        self.assertEqual(vt.threshold, 0.0)

    def test_removes_zero_variance_column(self):
        vt = get_variance_threshold(threshold=0.0)
        X = np.array([[1, 0], [2, 0], [3, 0]])  # column 1 is constant
        result = vt.fit_transform(X)
        self.assertEqual(result.shape[1], 1)

    def test_custom_threshold(self):
        vt = get_variance_threshold(threshold=0.5)
        self.assertEqual(vt.threshold, 0.5)


class TestGetSelectKBest(unittest.TestCase):
    def test_returns_select_k_best(self):
        skb = get_select_k_best(k=5)
        self.assertIsInstance(skb, SelectKBest)

    def test_default_k_is_10(self):
        skb = get_select_k_best()
        self.assertEqual(skb.k, 10)

    def test_f_regression_score_func(self):
        skb = get_select_k_best(k=3, score_func="f_regression")
        self.assertIsInstance(skb, SelectKBest)

    def test_f_classif_score_func(self):
        skb = get_select_k_best(k=3, score_func="f_classif")
        self.assertIsInstance(skb, SelectKBest)

    def test_chi2_score_func(self):
        skb = get_select_k_best(k=3, score_func="chi2")
        self.assertIsInstance(skb, SelectKBest)

    def test_mutual_info_classif_score_func(self):
        skb = get_select_k_best(k=3, score_func="mutual_info_classif")
        self.assertIsInstance(skb, SelectKBest)

    def test_unknown_score_func_raises_configuration_error(self):
        with self.assertRaises(ConfigurationError):
            get_select_k_best(score_func="invalid_func")

    def test_callable_score_func(self):
        from sklearn.feature_selection import f_regression
        skb = get_select_k_best(k=2, score_func=f_regression)
        self.assertIsInstance(skb, SelectKBest)

    def test_selects_k_features(self):
        from sklearn.datasets import make_regression
        skb = get_select_k_best(k=3, score_func="f_regression")
        X, y = make_regression(n_samples=50, n_features=10, random_state=42)
        result = skb.fit_transform(X, y)
        self.assertEqual(result.shape[1], 3)


class TestGetRFESelector(unittest.TestCase):
    def test_returns_rfe(self):
        est = LinearRegression()
        rfe = get_rfe_selector(estimator=est, n_features_to_select=3)
        self.assertIsInstance(rfe, RFE)

    def test_n_features_to_select(self):
        est = LinearRegression()
        rfe = get_rfe_selector(estimator=est, n_features_to_select=5)
        self.assertEqual(rfe.n_features_to_select, 5)

    def test_step_parameter(self):
        est = LinearRegression()
        rfe = get_rfe_selector(estimator=est, n_features_to_select=3, step=2)
        self.assertEqual(rfe.step, 2)


class TestGetSelectFromModel(unittest.TestCase):
    def test_returns_select_from_model(self):
        est = LinearRegression()
        sfm = get_select_from_model(estimator=est)
        self.assertIsInstance(sfm, SelectFromModel)

    def test_default_threshold_is_median(self):
        est = LinearRegression()
        sfm = get_select_from_model(estimator=est)
        self.assertEqual(sfm.threshold, "median")

    def test_custom_threshold(self):
        est = LinearRegression()
        sfm = get_select_from_model(estimator=est, threshold=0.01)
        self.assertEqual(sfm.threshold, 0.01)


class TestGetPCAReducer(unittest.TestCase):
    def test_returns_pca(self):
        pca = get_pca_reducer()
        self.assertIsInstance(pca, PCA)

    def test_default_n_components_is_095(self):
        pca = get_pca_reducer()
        self.assertEqual(pca.n_components, 0.95)

    def test_integer_n_components(self):
        pca = get_pca_reducer(n_components=3)
        self.assertEqual(pca.n_components, 3)

    def test_can_reduce_dimensions(self):
        pca = get_pca_reducer(n_components=2)
        X = np.random.RandomState(0).randn(50, 10)
        result = pca.fit_transform(X)
        self.assertEqual(result.shape[1], 2)


class TestGetNCAReducer(unittest.TestCase):
    def test_returns_nca(self):
        nca = get_nca_reducer()
        self.assertIsInstance(nca, NeighborhoodComponentsAnalysis)

    def test_none_n_components(self):
        nca = get_nca_reducer()
        self.assertIsNone(nca.n_components)

    def test_integer_n_components(self):
        nca = get_nca_reducer(n_components=3)
        self.assertEqual(nca.n_components, 3)

    def test_random_state(self):
        nca = get_nca_reducer(random_state=42)
        self.assertEqual(nca.random_state, 42)


if __name__ == "__main__":
    unittest.main()
