"""
Unit Tests — core_recommender/exceptions.py
============================================
Tests every exception class in the hierarchy:
  GuideMLError, DataValidationError, InsufficientDataError,
  ModelTrainingError, ModelNotFittedError, ModelExportError,
  ConfigurationError, RegistryError, PipelineError, VisualisationError
"""

import unittest
from core_recommender.exceptions import (
    GuideMLError,
    DataValidationError,
    InsufficientDataError,
    ModelTrainingError,
    ModelNotFittedError,
    ModelExportError,
    ConfigurationError,
    RegistryError,
    PipelineError,
    VisualisationError,
)


class TestGuideMLError(unittest.TestCase):
    """Tests for the root base exception."""

    def test_basic_raise_and_message(self):
        """GuideMLError stores message and can be raised."""
        with self.assertRaises(GuideMLError) as ctx:
            raise GuideMLError("root error occurred")
        self.assertEqual(str(ctx.exception), "root error occurred")

    def test_context_kwargs_in_str(self):
        """Extra kwargs appear in the string representation."""
        err = GuideMLError("something failed", module="execution", step=3)
        result = str(err)
        self.assertIn("something failed", result)
        self.assertIn("module=", result)
        self.assertIn("step=", result)

    def test_message_attribute(self):
        """The .message attribute matches what was passed."""
        err = GuideMLError("my message")
        self.assertEqual(err.message, "my message")

    def test_context_attribute(self):
        """The .context dict stores extra kwargs."""
        err = GuideMLError("ctx test", key="value", num=99)
        self.assertEqual(err.context["key"], "value")
        self.assertEqual(err.context["num"], 99)

    def test_no_context_str(self):
        """Without extra kwargs, only the message is shown."""
        err = GuideMLError("plain message")
        self.assertEqual(str(err), "plain message")

    def test_is_exception(self):
        """GuideMLError is a subclass of Exception."""
        self.assertTrue(issubclass(GuideMLError, Exception))


class TestDataValidationError(unittest.TestCase):
    """Tests for DataValidationError."""

    def test_inherits_from_guideml_error(self):
        self.assertTrue(issubclass(DataValidationError, GuideMLError))

    def test_raise_without_column(self):
        with self.assertRaises(DataValidationError):
            raise DataValidationError("target not found")

    def test_column_stored_in_context(self):
        err = DataValidationError("bad column", column="price")
        self.assertIn("column", err.context)
        self.assertEqual(err.context["column"], "price")

    def test_column_appears_in_str(self):
        err = DataValidationError("error", column="age")
        self.assertIn("age", str(err))

    def test_caught_as_guideml_error(self):
        """DataValidationError can be caught by GuideMLError handler."""
        with self.assertRaises(GuideMLError):
            raise DataValidationError("caught by parent")


class TestInsufficientDataError(unittest.TestCase):
    """Tests for InsufficientDataError."""

    def test_inherits_from_data_validation_error(self):
        self.assertTrue(issubclass(InsufficientDataError, DataValidationError))

    def test_raise_and_catch(self):
        with self.assertRaises(InsufficientDataError):
            raise InsufficientDataError("not enough rows")

    def test_caught_as_parent_types(self):
        with self.assertRaises(DataValidationError):
            raise InsufficientDataError("also caught by parent")
        with self.assertRaises(GuideMLError):
            raise InsufficientDataError("and by root")


class TestModelTrainingError(unittest.TestCase):
    """Tests for ModelTrainingError."""

    def test_inherits_from_guideml_error(self):
        self.assertTrue(issubclass(ModelTrainingError, GuideMLError))

    def test_model_name_stored_in_context(self):
        err = ModelTrainingError("fit failed", model_name="RandomForest")
        self.assertEqual(err.context["model"], "RandomForest")

    def test_no_model_name(self):
        err = ModelTrainingError("generic failure")
        self.assertNotIn("model", err.context)

    def test_raise(self):
        with self.assertRaises(ModelTrainingError):
            raise ModelTrainingError("boom", model_name="SVM")


class TestModelNotFittedError(unittest.TestCase):
    """Tests for ModelNotFittedError."""

    def test_inherits_from_model_training_error(self):
        self.assertTrue(issubclass(ModelNotFittedError, ModelTrainingError))

    def test_raise(self):
        with self.assertRaises(ModelNotFittedError):
            raise ModelNotFittedError("not fitted yet")

    def test_caught_as_model_training_error(self):
        with self.assertRaises(ModelTrainingError):
            raise ModelNotFittedError("caught by parent")


class TestModelExportError(unittest.TestCase):
    """Tests for ModelExportError."""

    def test_inherits_from_model_training_error(self):
        self.assertTrue(issubclass(ModelExportError, ModelTrainingError))

    def test_raise_with_context(self):
        with self.assertRaises(ModelExportError):
            raise ModelExportError("write failed", model_name="KNN", filepath="/tmp/m.pkl")


class TestConfigurationError(unittest.TestCase):
    """Tests for ConfigurationError."""

    def test_inherits_from_guideml_error(self):
        self.assertTrue(issubclass(ConfigurationError, GuideMLError))

    def test_raise_and_message(self):
        with self.assertRaises(ConfigurationError) as ctx:
            raise ConfigurationError("bad config value")
        self.assertIn("bad config value", str(ctx.exception))


class TestRegistryError(unittest.TestCase):
    """Tests for RegistryError."""

    def test_inherits_from_configuration_error(self):
        self.assertTrue(issubclass(RegistryError, ConfigurationError))

    def test_raise(self):
        with self.assertRaises(RegistryError):
            raise RegistryError("invalid task 'banana'")

    def test_caught_as_configuration_error(self):
        with self.assertRaises(ConfigurationError):
            raise RegistryError("caught by parent")


class TestPipelineError(unittest.TestCase):
    """Tests for PipelineError."""

    def test_inherits_from_guideml_error(self):
        self.assertTrue(issubclass(PipelineError, GuideMLError))

    def test_raise(self):
        with self.assertRaises(PipelineError):
            raise PipelineError("pipeline collapsed")


class TestVisualisationError(unittest.TestCase):
    """Tests for VisualisationError."""

    def test_inherits_from_guideml_error(self):
        self.assertTrue(issubclass(VisualisationError, GuideMLError))

    def test_raise(self):
        with self.assertRaises(VisualisationError):
            raise VisualisationError("plot failed")


if __name__ == "__main__":
    unittest.main()
