"""
Unit Tests — core_recommender/knowledge_base.py
================================================
Tests the MODEL_KNOWLEDGE dictionary structure.
"""

import unittest
from core_recommender.knowledge_base import MODEL_KNOWLEDGE


class TestModelKnowledgeStructure(unittest.TestCase):
    EXPECTED_KEYS = [
        "KNN", "Random Forest", "Decision Tree",
        "SVM", "Naive Bayes", "Logistic Regression", "Linear Regression"
    ]
    REQUIRED_FIELDS = ["title", "story", "how_it_works", "pros", "cons", "best_for"]

    def test_is_dict(self):
        self.assertIsInstance(MODEL_KNOWLEDGE, dict)

    def test_all_expected_model_keys_present(self):
        for key in self.EXPECTED_KEYS:
            self.assertIn(key, MODEL_KNOWLEDGE, msg=f"Missing key: {key}")

    def test_each_entry_has_all_required_fields(self):
        for model_key, data in MODEL_KNOWLEDGE.items():
            for field in self.REQUIRED_FIELDS:
                self.assertIn(field, data,
                              msg=f"'{model_key}' missing field '{field}'")

    def test_pros_and_cons_are_lists(self):
        for model_key, data in MODEL_KNOWLEDGE.items():
            self.assertIsInstance(data["pros"], list,
                                   msg=f"'{model_key}'.pros should be list")
            self.assertIsInstance(data["cons"], list,
                                   msg=f"'{model_key}'.cons should be list")

    def test_pros_and_cons_non_empty(self):
        for model_key, data in MODEL_KNOWLEDGE.items():
            self.assertGreater(len(data["pros"]), 0,
                                msg=f"'{model_key}'.pros should not be empty")
            self.assertGreater(len(data["cons"]), 0,
                                msg=f"'{model_key}'.cons should not be empty")

    def test_string_fields_are_strings(self):
        for model_key, data in MODEL_KNOWLEDGE.items():
            for field in ["title", "story", "how_it_works", "best_for"]:
                self.assertIsInstance(data[field], str,
                                       msg=f"'{model_key}'.{field} should be str")

    def test_all_string_fields_non_empty(self):
        for model_key, data in MODEL_KNOWLEDGE.items():
            for field in ["title", "story", "how_it_works", "best_for"]:
                self.assertGreater(len(data[field].strip()), 0,
                                    msg=f"'{model_key}'.{field} should not be blank")

    def test_knn_entry_mentions_neighbors(self):
        story = MODEL_KNOWLEDGE["KNN"]["story"].lower()
        self.assertIn("neighbor", story)

    def test_random_forest_entry_mentions_trees(self):
        story = MODEL_KNOWLEDGE["Random Forest"]["story"].lower()
        self.assertTrue("tree" in story or "forest" in story or "expert" in story)

    def test_logistic_regression_pros_includes_item(self):
        pros = MODEL_KNOWLEDGE["Logistic Regression"]["pros"]
        self.assertGreater(len(pros), 0)


if __name__ == "__main__":
    unittest.main()
