"""
Centralized Exception Hierarchy — core_recommender/exceptions.py

Rationale
---------
Previously, every module raised generic built-in exceptions (ValueError,
RuntimeError, AttributeError).  This made it impossible for callers to:
  - Distinguish a data problem from a model problem from a config problem.
  - Write targeted except-clauses that recover from specific failure modes.
  - Display meaningful, actionable error messages in the UI.

Design principles
-----------------
* **SRP** — one module owns all exception definitions.
* **OCP** — new exception types are added here, never in consumer modules.
* **Hierarchy** — all exceptions inherit from ``GuideMLError`` so callers can
  catch the entire family with a single ``except GuideMLError`` when needed.

Usage
-----
    from core_recommender.exceptions import DataValidationError, ModelTrainingError

    # Raise typed exception
    raise DataValidationError("Target column 'y' contains NaN values.", column='y')

    # Catch only data errors (ModelTrainingError still propagates)
    try:
        executor.run(df, target)
    except DataValidationError as exc:
        flash(str(exc), 'warning')
    except ModelTrainingError as exc:
        flash(str(exc), 'danger')
    except GuideMLError as exc:
        flash(f"Unexpected pipeline error: {exc}", 'danger')
"""

from __future__ import annotations
from typing import Optional


# =========================================================================
# Base
# =========================================================================

class GuideMLError(Exception):
    """Root exception for all GuideML errors.

    All application-specific exceptions inherit from this class so callers
    can catch the entire family with ``except GuideMLError``.

    Attributes:
        message (str): Human-readable description of the problem.
        context (dict): Optional key-value pairs for extra debugging context.
    """

    def __init__(self, message: str, **context):
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        if self.context:
            ctx_str = ', '.join(f'{k}={v!r}' for k, v in self.context.items())
            return f"{self.message} [{ctx_str}]"
        return self.message


# =========================================================================
# Data Layer
# =========================================================================

class DataValidationError(GuideMLError):
    """Raised when the input data fails a validation check.

    Examples
    --------
    * Target column not found in the DataFrame.
    * Target variable contains NaN values.
    * Dataset has fewer rows than required for cross-validation.
    * Non-numeric features where numeric are required.

    Args:
        message: Description of the validation failure.
        column: The specific column or feature involved, if applicable.
    """

    def __init__(self, message: str, column: Optional[str] = None, **context):
        if column:
            context['column'] = column
        super().__init__(message, **context)


class InsufficientDataError(DataValidationError):
    """Raised when there are not enough samples to proceed.

    Examples
    --------
    * Fewer samples than CV folds.
    * Empty DataFrame passed to executor.
    """
    pass


# =========================================================================
# Model / Training Layer
# =========================================================================

class ModelTrainingError(GuideMLError):
    """Raised when a model fails during fit / hyperparameter tuning.

    Examples
    --------
    * Optuna study fails to converge.
    * Scikit-learn estimator raises during fit().
    * All candidate models fail — no best model available.

    Args:
        message: Description of the training failure.
        model_name: The name of the model that failed, if applicable.
    """

    def __init__(self, message: str, model_name: Optional[str] = None, **context):
        if model_name:
            context['model'] = model_name
        super().__init__(message, **context)


class ModelNotFittedError(ModelTrainingError):
    """Raised when a method is called on a model that has not been trained yet.

    Examples
    --------
    * ``get_diagnostic_data()`` called before ``fit()``.
    * ``export()`` called before ``fit()``.
    * ``get_tailored_diagnostics()`` called before ``fit()``.
    """
    pass


class ModelExportError(ModelTrainingError):
    """Raised when a fitted model cannot be serialised to disk."""
    pass


# =========================================================================
# Configuration / Registry Layer
# =========================================================================

class ConfigurationError(GuideMLError):
    """Raised when an invalid configuration value is provided.

    Examples
    --------
    * Unknown task type passed to registry.
    * Unknown score_func string in feature selection.
    * Unsupported Naive Bayes model_type.
    * Invalid ``@register_model(task=...)`` value.
    """
    pass


class RegistryError(ConfigurationError):
    """Raised for problems with the ModelRegistry.

    Examples
    --------
    * Registering a model class with an invalid task string.
    * Requesting models for an unsupported task type.
    """
    pass


# =========================================================================
# Pipeline / Execution Layer
# =========================================================================

class PipelineError(GuideMLError):
    """Raised when the overall ML pipeline fails in a non-recoverable way.

    Examples
    --------
    * No models remain after filtering by ``include_models``.
    * Preprocessing step raises an unhandled error.
    * SHAP explainability step fails fatally.
    """
    pass


# =========================================================================
# Visualisation Layer
# =========================================================================

class VisualisationError(GuideMLError):
    """Raised when a plot cannot be generated.

    Examples
    --------
    * Image save path directory does not exist and cannot be created.
    * Input arrays have mismatched lengths.
    """
    pass
