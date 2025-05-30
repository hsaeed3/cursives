"""
✼ cursives.pydantic.models
"""

from typing import Any, Dict, Optional, Callable, Union
from functools import wraps
from pydantic import BaseModel, Field, ConfigDict, field_validator

__all__ = [
    "SubscriptableBaseModel",
    "DynamicModel",
    "FrozenModel",
    "CacheableModel",
    "FlexibleModel",
]


class SubscriptableBaseModel(BaseModel):
    """
    A base class for all Pydantic models that can be subscripted.
    """

    def __getitem__(self, key: str) -> Any:
        """Get field value using dict-like access.

        Usage:
            >>> msg = Message(role='user')
            >>> msg['role']
            'user'
        """
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set field value using dict-like access.

        Usage:
            >>> msg = Message(role='user')
            >>> msg['role'] = 'assistant'
            >>> msg['role']
            'assistant'
        """
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """Check if field exists using 'in' operator.

        Usage:
            >>> msg = Message(role='user')
            >>> 'role' in msg
            True
            >>> 'nonexistent' in msg
            False
        """
        if hasattr(self, key):
            return True
        if value := self.__class__.model_fields.get(key):
            return value.default is not None
        return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get field value with optional default.

        Usage:
            >>> msg = Message(role='user')
            >>> msg.get('role')
            'user'
            >>> msg.get('nonexistent', 'default')
            'default'
        """
        return getattr(self, key) if hasattr(self, key) else default


class DynamicModel(SubscriptableBaseModel):
    """
    A model that allows dynamic field assignment and access.
    Perfect for handling arbitrary JSON data or when schema is unknown at compile time.

    Usage:
        >>> data = DynamicModel()
        >>> data.name = "John"
        >>> data.age = 30
        >>> data.metadata = {"key": "value"}
        >>> print(data.name)  # John
        >>> print(data["age"])  # 30
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def __init__(self, **data: Any):
        super().__init__(**data)
        # Store extra fields for easy access
        self._dynamic_fields: Dict[str, Any] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name in self.__class__.model_fields:
            super().__setattr__(name, value)
        else:
            # Store in dynamic fields and set normally
            if hasattr(self, "_dynamic_fields"):
                self._dynamic_fields[name] = value
            super().__setattr__(name, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including all dynamic fields."""
        result = self.model_dump()
        if hasattr(self, "_dynamic_fields"):
            result.update(self._dynamic_fields)
        return result


class FrozenModel(SubscriptableBaseModel):
    """
    An immutable model that prevents modification after creation.
    Useful for creating read-only data structures and ensuring data integrity.

    Usage:
        >>> user = FrozenModel(name="Alice", age=25)
        >>> user.name  # "Alice"
        >>> user.name = "Bob"  # Raises FrozenInstanceError
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    def with_changes(self, **changes: Any) -> "FrozenModel":
        """Create a new instance with specified changes (functional approach)."""
        current_data = self.model_dump()
        current_data.update(changes)
        return self.__class__(**current_data)

    def evolve(self, **changes: Any) -> "FrozenModel":
        """Alias for with_changes for more intuitive API."""
        return self.with_changes(**changes)


class CacheableModel(SubscriptableBaseModel):
    """
    A model with built-in caching for computed properties.
    Automatically caches expensive computations and invalidates when dependencies change.

    Usage:
        >>> class MyModel(CacheableModel):
        ...     value: int
        ...     @CacheableModel.cached_property(dependencies=["value"])
        ...     def expensive_computation(self) -> int:
        ...         return self.value ** 2
        >>> model = MyModel(value=10)
        >>> model.expensive_computation  # Computed once
        >>> model.expensive_computation  # Returns cached value
    """

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._cache: Dict[str, Any] = {}
        self._cache_dependencies: Dict[str, list] = {}

    @classmethod
    def cached_property(cls, dependencies: Optional[list] = None):
        """Decorator for creating cached properties with optional dependencies."""

        def decorator(func: Callable) -> property:
            prop_name = func.__name__
            deps = dependencies or []

            @wraps(func)
            def wrapper(self) -> Any:
                # Check if cached and dependencies haven't changed
                if prop_name in self._cache:
                    if not deps or all(
                        getattr(self, dep) == self._cache.get(f"_{dep}_snapshot")
                        for dep in deps
                    ):
                        return self._cache[prop_name]

                # Compute and cache
                result = func(self)
                self._cache[prop_name] = result

                # Store dependency snapshots
                for dep in deps:
                    self._cache[f"_{dep}_snapshot"] = getattr(self, dep)

                return result

            return property(wrapper)

        return decorator

    def clear_cache(self, property_name: Optional[str] = None) -> None:
        """Clear cache for specific property or all cached properties."""
        if property_name:
            self._cache.pop(property_name, None)
        else:
            self._cache.clear()

    def __setattr__(self, name: str, value: Any) -> None:
        # Invalidate cache when dependencies change
        if hasattr(self, "_cache") and name in self.__class__.model_fields:
            # Clear cache for properties that depend on this field
            for prop_name, deps in getattr(self, "_cache_dependencies", {}).items():
                if name in deps:
                    self._cache.pop(prop_name, None)

        super().__setattr__(name, value)


class FlexibleModel(SubscriptableBaseModel):
    """
    A model combining multiple useful features: dynamic fields, validation, serialization.
    The Swiss Army knife of Pydantic models.

    Features:
    - Dynamic field assignment
    - Custom validation methods
    - Enhanced serialization options
    - Partial updates
    - Field aliasing support

    Usage:
        >>> class UserModel(FlexibleModel):
        ...     name: str
        ...     email: str
        >>> user = UserModel(name="John", email="john@example.com")
        >>> user.nickname = "Johnny"  # Dynamic field
        >>> user.partial_update(name="Jane")
    """

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._dynamic_fields: Dict[str, Any] = {}
        self._validation_errors: list = []

    def partial_update(self, **updates: Any) -> "FlexibleModel":
        """Update specific fields without full validation."""
        for key, value in updates.items():
            # NOTE:
            # update to .__class__.model_fields soon
            if key in self.__class__.model_fields:
                setattr(self, key, value)
            else:
                self._dynamic_fields[key] = value
                super().__setattr__(key, value)
        return self

    def safe_get(
        self, key: str, default: Any = None, validator: Optional[Callable] = None
    ) -> Any:
        """Safely get a field value with optional validation."""
        try:
            value = getattr(self, key, default)
            if validator and value != default:
                if not validator(value):
                    return default
            return value
        except (AttributeError, TypeError, ValueError):
            return default

    def to_json_safe(self) -> Dict[str, Any]:
        """Convert to JSON-safe dictionary (handles non-serializable types)."""
        result = {}

        # Regular fields
        for field_name, field_value in self.model_dump().items():
            try:
                # Test JSON serializability
                import json

                json.dumps(field_value)
                result[field_name] = field_value
            except (TypeError, ValueError):
                # Convert non-serializable types to string
                result[field_name] = str(field_value)

        # Dynamic fields
        for field_name, field_value in self._dynamic_fields.items():
            try:
                import json

                json.dumps(field_value)
                result[field_name] = field_value
            except (TypeError, ValueError):
                result[field_name] = str(field_value)

        return result

    def validate_field(self, field_name: str, validator: Callable[[Any], bool]) -> bool:
        """Validate a specific field with custom validator."""
        try:
            value = getattr(self, field_name)
            return validator(value)
        except (AttributeError, TypeError, ValueError) as e:
            self._validation_errors.append(f"Validation error for {field_name}: {e}")
            return False

    def get_validation_errors(self) -> list:
        """Get list of validation errors."""
        return self._validation_errors.copy()

    def clear_validation_errors(self) -> None:
        """Clear validation errors."""
        self._validation_errors.clear()

    @classmethod
    def from_dict(cls, data: Dict[str, Any], strict: bool = False) -> "FlexibleModel":
        """Create instance from dictionary with optional strict mode."""
        if strict:
            # Only use fields defined in the model
            filtered_data = {k: v for k, v in data.items() if k in cls.model_fields}
            return cls(**filtered_data)
        else:
            return cls(**data)
