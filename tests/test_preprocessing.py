import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from core_recommender.preprocessing import (
    get_imputer,
    get_one_hot_encoder,
    get_standard_scaler,
    get_minmax_scaler
)

class TestPreprocessingFactories(unittest.TestCase):
    def test_get_imputer(self):
        imp = get_imputer(strategy='mean')
        self.assertEqual(imp.strategy, 'mean')
        
        imp_const = get_imputer(strategy='constant', fill_value=0)
        self.assertEqual(imp_const.strategy, 'constant')
        self.assertEqual(imp_const.fill_value, 0)

    def test_get_one_hot_encoder(self):
        ohe = get_one_hot_encoder(handle_unknown='ignore')
        self.assertEqual(ohe.handle_unknown, 'ignore')

    def test_get_standard_scaler(self):
        scaler = get_standard_scaler()
        self.assertIsInstance(scaler, StandardScaler)

    def test_get_minmax_scaler(self):
        scaler = get_minmax_scaler()
        self.assertIsInstance(scaler, MinMaxScaler)

if __name__ == '__main__':
    unittest.main()
