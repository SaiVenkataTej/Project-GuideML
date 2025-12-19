import unittest
import numpy as np
from core_recommender.evaluation import (
    calculate_accuracy, calculate_rmse, calculate_r2_score,
    calculate_f1_score, calculate_precision, calculate_log_loss,
    calculate_roc_auc_score
)

class TestEvaluation(unittest.TestCase):
    def test_calculate_accuracy(self):
        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 0, 0]
        acc = calculate_accuracy(y_true, y_pred)
        self.assertEqual(acc, 0.75)

    def test_calculate_rmse(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        rmse = calculate_rmse(y_true, y_pred)
        self.assertEqual(rmse, 0.0)

    def test_calculate_r2_score(self):
        y_true = [1, 2, 3]
        y_pred = [1, 2, 3]
        r2 = calculate_r2_score(y_true, y_pred)
        self.assertEqual(r2, 1.0)

    def test_calculate_f1_score(self):
        y_true = [0, 1, 1]
        y_pred = [0, 1, 0]
        # TP=1, FP=0, FN=1 -> Prec=1.0, Rec=0.5 -> F1=0.66...
        # Note: average='weighted' is default in the file logic? 
        # Checking logic defaults. File shows default='weighted'
        # With binary data and weighted average, it might differ.
        # Let's override to 'binary' if possible or check 'weighted'.
        f1 = calculate_f1_score(y_true, y_pred, average='binary')
        self.assertAlmostEqual(f1, 0.6666666666666666)

    def test_calculate_log_loss(self):
        y_true = [0, 1]
        # Proba: low for class 1 at index 0, high for class 1 at index 1
        y_proba = [[0.9, 0.1], [0.1, 0.9]] 
        loss = calculate_log_loss(y_true, y_proba)
        # Log loss should be low (roughly -log(0.9) ~= 0.105)
        self.assertTrue(loss < 0.2)

if __name__ == '__main__':
    unittest.main()
