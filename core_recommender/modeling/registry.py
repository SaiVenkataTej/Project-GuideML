"""
Model Registry — core_recommender/modeling/registry.py

Implements the Open/Closed Principle (OCP) for model management.

Adding a new model to the system requires ZERO changes to this file or the
ModelExecutor. Simply decorate the new class with @register_model and import
the module anywhere in the package (the registry in __init__.py handles this).

Usage:
    from core_recommender.modeling.registry import register_model, get_registered_models

    @register_model(task='classification')
    class MyNewModel(BaseModel):
        ...
"""

from typing import Dict, List, Optional, Type, TYPE_CHECKING
from core_recommender.exceptions import RegistryError

if TYPE_CHECKING:
    from core_recommender.modeling.base_model import BaseModel

# ---------------------------------------------------------------------------
# Internal registry store
# ---------------------------------------------------------------------------

# Structure: {'classification': [ClassA, ClassB], 'regression': [ClassA, ClassC]}
_REGISTRY: Dict[str, List[Type["BaseModel"]]] = {
    'classification': [],
    'regression': [],
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_model(task: str):
    """Class decorator that registers a model class into the global registry.

    Args:
        task (str): The task this model supports. Must be one of
                    'classification', 'regression', or 'both'.

    Returns:
        Callable: The unmodified class (decorator is non-destructive).

    Raises:
        ValueError: If an unrecognised task string is provided.

    Example:
        @register_model(task='both')
        class KNNModel(BaseModel):
            ...
    """
    valid_tasks = ('classification', 'regression', 'both')
    if task not in valid_tasks:
        raise RegistryError(
            f"Invalid task '{task}'. Must be one of {valid_tasks}."
        )

    def decorator(cls: Type["BaseModel"]) -> Type["BaseModel"]:
        if task in ('classification', 'both'):
            if cls not in _REGISTRY['classification']:
                _REGISTRY['classification'].append(cls)
        if task in ('regression', 'both'):
            if cls not in _REGISTRY['regression']:
                _REGISTRY['regression'].append(cls)
        return cls

    return decorator


def get_registered_models(task: str) -> List[Type["BaseModel"]]:
    """Returns all registered model classes for the given task.

    Args:
        task (str): 'classification' or 'regression'.

    Returns:
        List[Type[BaseModel]]: Registered model classes (not instances).

    Raises:
        ValueError: If task is not 'classification' or 'regression'.
    """
    if task not in ('classification', 'regression'):
        raise RegistryError(
            f"Invalid task '{task}'. Must be 'classification' or 'regression'."
        )
    return list(_REGISTRY[task])


def clear_registry() -> None:
    """Resets the registry. Intended for use in unit tests only."""
    _REGISTRY['classification'].clear()
    _REGISTRY['regression'].clear()
