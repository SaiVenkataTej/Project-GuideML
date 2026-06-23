"""
Unit Tests — core_recommender/modeling/registry.py
===================================================
Tests every public function:
  register_model (decorator), get_registered_models, clear_registry
"""

import unittest
from core_recommender.modeling.registry import (
    register_model,
    get_registered_models,
    clear_registry,
    _REGISTRY,
)
from core_recommender.exceptions import RegistryError


# ---------------------------------------------------------------------------
# Helper stub — a minimal fake BaseModel that does not import the full stack
# ---------------------------------------------------------------------------
class _FakeModel:
    """Minimal stub used only inside these tests."""
    pass


class _FakeModelB:
    pass


class _FakeModelC:
    pass


class TestRegisterModelDecorator(unittest.TestCase):
    def setUp(self):
        clear_registry()

    def tearDown(self):
        clear_registry()

    def test_classification_task_registers_in_classification_bucket(self):
        @register_model(task="classification")
        class TestClfModel(_FakeModel):
            pass

        models = get_registered_models("classification")
        self.assertIn(TestClfModel, models)

    def test_classification_task_not_in_regression_bucket(self):
        @register_model(task="classification")
        class TestOnlyClfModel(_FakeModel):
            pass

        models = get_registered_models("regression")
        self.assertNotIn(TestOnlyClfModel, models)

    def test_regression_task_registers_in_regression_bucket(self):
        @register_model(task="regression")
        class TestRegModel(_FakeModel):
            pass

        models = get_registered_models("regression")
        self.assertIn(TestRegModel, models)

    def test_regression_task_not_in_classification_bucket(self):
        @register_model(task="regression")
        class TestOnlyRegModel(_FakeModel):
            pass

        models = get_registered_models("classification")
        self.assertNotIn(TestOnlyRegModel, models)

    def test_both_task_registers_in_both_buckets(self):
        @register_model(task="both")
        class TestBothModel(_FakeModel):
            pass

        clf_models = get_registered_models("classification")
        reg_models = get_registered_models("regression")
        self.assertIn(TestBothModel, clf_models)
        self.assertIn(TestBothModel, reg_models)

    def test_decorator_returns_class_unchanged(self):
        @register_model(task="classification")
        class OriginalClass(_FakeModel):
            pass

        self.assertEqual(OriginalClass.__name__, "OriginalClass")
        self.assertTrue(issubclass(OriginalClass, _FakeModel))

    def test_duplicate_registration_does_not_add_twice(self):
        @register_model(task="classification")
        class UniqueModel(_FakeModel):
            pass

        # Simulate re-registering the same class
        register_model(task="classification")(UniqueModel)

        models = get_registered_models("classification")
        count = sum(1 for m in models if m is UniqueModel)
        self.assertEqual(count, 1)

    def test_invalid_task_raises_registry_error(self):
        with self.assertRaises(RegistryError):
            @register_model(task="invalid_task")
            class BadModel(_FakeModel):
                pass

    def test_multiple_classes_registered(self):
        @register_model(task="regression")
        class ModelA(_FakeModelB):
            pass

        @register_model(task="regression")
        class ModelB(_FakeModelC):
            pass

        models = get_registered_models("regression")
        self.assertIn(ModelA, models)
        self.assertIn(ModelB, models)


class TestGetRegisteredModels(unittest.TestCase):
    def setUp(self):
        clear_registry()

    def tearDown(self):
        clear_registry()

    def test_returns_list(self):
        result = get_registered_models("classification")
        self.assertIsInstance(result, list)

    def test_empty_after_clear(self):
        result = get_registered_models("classification")
        self.assertEqual(result, [])

    def test_invalid_task_raises_registry_error(self):
        with self.assertRaises(RegistryError):
            get_registered_models("unsupported_task")

    def test_returns_copy_not_reference(self):
        """Mutating the returned list should not affect the registry."""
        result = get_registered_models("classification")
        original_len = len(_REGISTRY["classification"])
        result.append(_FakeModel)
        self.assertEqual(len(_REGISTRY["classification"]), original_len)


class TestClearRegistry(unittest.TestCase):
    def test_clear_empties_both_buckets(self):
        @register_model(task="both")
        class TempModel(_FakeModel):
            pass

        clear_registry()
        self.assertEqual(get_registered_models("classification"), [])
        self.assertEqual(get_registered_models("regression"), [])

    def test_clear_does_not_raise_on_empty_registry(self):
        clear_registry()  # already empty
        try:
            clear_registry()
        except Exception as exc:
            self.fail(f"clear_registry() raised unexpectedly: {exc}")


if __name__ == "__main__":
    unittest.main()
