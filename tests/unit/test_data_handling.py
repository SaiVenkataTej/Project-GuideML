"""
Unit Tests — core_recommender/dataHandling.py
=============================================
Tests every public function:
  apply_one_hot_encoder, apply_ordinal_encoder, apply_label_encoder_target
"""

import unittest
import numpy as np
import pandas as pd
from core_recommender.dataHandling import (
    apply_one_hot_encoder,
    apply_ordinal_encoder,
    apply_label_encoder_target,
)


class TestApplyOneHotEncoder(unittest.TestCase):
    """Tests for apply_one_hot_encoder()."""

    def setUp(self):
        self.df = pd.DataFrame({
            "color": ["red", "blue", "green", "red"],
            "size":  ["S", "M", "L", "M"]
        })

    def test_output_is_dataframe(self):
        result = apply_one_hot_encoder(self.df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_no_original_columns_remain(self):
        """Original string columns are replaced by binary columns."""
        result = apply_one_hot_encoder(self.df)
        self.assertNotIn("color", result.columns)
        self.assertNotIn("size", result.columns)

    def test_correct_number_of_columns_without_drop(self):
        """Without drop_first: 3 colors + 3 sizes = 6 binary columns."""
        result = apply_one_hot_encoder(self.df, drop_first=False)
        self.assertEqual(result.shape[1], 6)

    def test_correct_number_of_columns_with_drop_first(self):
        """With drop_first=True: (3-1) + (3-1) = 4 binary columns."""
        result = apply_one_hot_encoder(self.df, drop_first=True)
        self.assertEqual(result.shape[1], 4)

    def test_row_count_preserved(self):
        result = apply_one_hot_encoder(self.df)
        self.assertEqual(result.shape[0], len(self.df))

    def test_index_preserved(self):
        df = self.df.copy()
        df.index = [10, 20, 30, 40]
        result = apply_one_hot_encoder(df)
        self.assertListEqual(list(result.index), [10, 20, 30, 40])

    def test_values_are_binary(self):
        result = apply_one_hot_encoder(self.df)
        unique_vals = set(result.values.flatten().tolist())
        self.assertTrue(unique_vals.issubset({0.0, 1.0}))

    def test_single_column_dataframe(self):
        df = pd.DataFrame({"animal": ["cat", "dog", "cat"]})
        result = apply_one_hot_encoder(df)
        self.assertEqual(result.shape[0], 3)
        self.assertEqual(result.shape[1], 2)

    def test_handle_unknown_ignore(self):
        """Unknown categories during transform are silently set to 0."""
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(self.df)
        # The function always fits on the passed data — just verify no error
        result = apply_one_hot_encoder(self.df, handle_unknown_param="ignore")
        self.assertIsNotNone(result)


class TestApplyOrdinalEncoder(unittest.TestCase):
    """Tests for apply_ordinal_encoder()."""

    def setUp(self):
        self.df = pd.DataFrame({
            "size": ["S", "M", "L", "M", "S"]
        })

    def test_output_is_dataframe(self):
        result = apply_ordinal_encoder(self.df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_row_count_preserved(self):
        result = apply_ordinal_encoder(self.df)
        self.assertEqual(result.shape[0], len(self.df))

    def test_column_count_preserved(self):
        result = apply_ordinal_encoder(self.df)
        self.assertEqual(result.shape[1], self.df.shape[1])

    def test_column_names_preserved(self):
        result = apply_ordinal_encoder(self.df)
        self.assertEqual(list(result.columns), list(self.df.columns))

    def test_auto_mode_produces_integers(self):
        result = apply_ordinal_encoder(self.df, categories="auto")
        self.assertTrue(all(result["size"].apply(lambda x: x == int(x))))

    def test_custom_order_respected(self):
        """Passing explicit order: S=0, M=1, L=2."""
        result = apply_ordinal_encoder(self.df, categories=[["S", "M", "L"]])
        mapping = dict(zip(self.df["size"].tolist(), result["size"].tolist()))
        self.assertLess(mapping["S"], mapping["M"])
        self.assertLess(mapping["M"], mapping["L"])

    def test_flat_list_convenience_single_column(self):
        """Flat list ['S','M','L'] is auto-wrapped when 1 column."""
        result = apply_ordinal_encoder(self.df, categories=["S", "M", "L"])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, self.df.shape)

    def test_flat_list_convenience_multiple_columns(self):
        """Flat list is broadcast to all columns when multiple columns share the same ordinal."""
        df = pd.DataFrame({
            "q1": ["Low", "High", "Medium"],
            "q2": ["High", "Low", "Medium"]
        })
        result = apply_ordinal_encoder(df, categories=["Low", "Medium", "High"])
        self.assertEqual(result.shape, df.shape)

    def test_invalid_categories_type_raises(self):
        with self.assertRaises(TypeError):
            apply_ordinal_encoder(self.df, categories=123)

    def test_index_preserved(self):
        df = self.df.copy()
        df.index = [5, 10, 15, 20, 25]
        result = apply_ordinal_encoder(df)
        self.assertListEqual(list(result.index), [5, 10, 15, 20, 25])


class TestApplyLabelEncoderTarget(unittest.TestCase):
    """Tests for apply_label_encoder_target()."""

    def test_output_is_series(self):
        y = pd.Series(["cat", "dog", "cat", "bird"])
        result = apply_label_encoder_target(y)
        self.assertIsInstance(result, pd.Series)

    def test_values_are_integers(self):
        y = pd.Series(["cat", "dog", "cat", "bird"])
        result = apply_label_encoder_target(y)
        self.assertTrue(all(isinstance(v, (int, np.integer)) for v in result))

    def test_unique_labels_encoded_0_to_n(self):
        y = pd.Series(["a", "b", "c"])
        result = apply_label_encoder_target(y)
        self.assertEqual(set(result.tolist()), {0, 1, 2})

    def test_series_name_preserved(self):
        y = pd.Series(["yes", "no", "yes"], name="label")
        result = apply_label_encoder_target(y)
        self.assertEqual(result.name, "label")

    def test_index_preserved(self):
        y = pd.Series(["a", "b", "a"], index=[10, 20, 30])
        result = apply_label_encoder_target(y)
        self.assertListEqual(list(result.index), [10, 20, 30])

    def test_numeric_input(self):
        """Numeric categories are also encoded correctly."""
        y = pd.Series([3, 1, 2, 1, 3])
        result = apply_label_encoder_target(y)
        self.assertEqual(set(result.tolist()), {0, 1, 2})

    def test_single_class(self):
        y = pd.Series(["only"])
        result = apply_label_encoder_target(y)
        self.assertEqual(result.iloc[0], 0)

    def test_length_preserved(self):
        y = pd.Series(["x", "y", "z", "x", "y"])
        result = apply_label_encoder_target(y)
        self.assertEqual(len(result), len(y))


if __name__ == "__main__":
    unittest.main()
