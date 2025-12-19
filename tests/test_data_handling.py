import unittest
import pandas as pd
import numpy as np
from core_recommender.dataHandling import apply_one_hot_encoder, apply_ordinal_encoder, apply_label_encoder_target

class TestDataHandling(unittest.TestCase):
    def setUp(self):
        self.df_cat = pd.DataFrame({
            'color': ['red', 'blue', 'green', 'red'],
            'size': ['S', 'M', 'L', 'S']
        })
        self.target = pd.Series(['no', 'yes', 'no'], name='target')

    def test_apply_one_hot_encoder(self):
        # We assume the functions return a DataFrame
        df_encoded = apply_one_hot_encoder(self.df_cat, drop_first=False)
        self.assertIsInstance(df_encoded, pd.DataFrame)
        # Check simple dimensionality: color has 3 unique values -> 3 columns
        # size has 3 unique values -> 3 columns
        # Total columns = 6
        self.assertEqual(df_encoded.shape[1], 6)

    def test_apply_ordinal_encoder(self):
        # 'size' mapped to 0, 1, 2 typically if auto
        df_ord = apply_ordinal_encoder(self.df_cat, categories='auto')
        self.assertIsInstance(df_ord, pd.DataFrame)
        self.assertEqual(df_ord.shape, self.df_cat.shape)
        # Should be numeric now
        self.assertTrue(pd.api.types.is_numeric_dtype(df_ord.iloc[:, 0]))

    def test_apply_label_encoder_target(self):
        y_encoded = apply_label_encoder_target(self.target)
        self.assertIsInstance(y_encoded, pd.Series)
        # Should be integers 0, 1
        self.assertTrue(pd.api.types.is_integer_dtype(y_encoded))
        self.assertEqual(set(y_encoded.unique()), {0, 1})

if __name__ == '__main__':
    unittest.main()
