"""
cursives.utils._cache

Contains resources that are used within `cursives` for caching
purposes.
"""

import hashlib
import inspect
import time
from collections import OrderedDict
from functools import wraps
from typing import (
    Any,
    Callable,
    TypeVar,
    Dict,
    Tuple,
    Optional,
    Union,
    Protocol,
    overload,
    ParamSpec,
)

__all__ = [
    "get_value",
    "make_hashable",
    "cached",
    "auto_cached",
    "_CURSIVES_CACHE",
    "TYPE_MAPPING",
    "_CursivesCache",
]


# ------------------------------------------------------------------------------
# TYPE VARIABLES
# ------------------------------------------------------------------------------

P = ParamSpec("P")
R = TypeVar("R")


class Hashable(Protocol):
    """Protocol for objects that can be hashed."""

    def __hash__(self) -> int: ...


# ------------------------------------------------------------------------------
# CACHE IMPLEMENTATION
# ------------------------------------------------------------------------------


class _CursivesCache:
    """
    Internal thread-safe TTL cache implementation with LRU eviction.

    Uses OrderedDict for efficient LRU tracking and automatic cleanup
    of expired entries on access.
    """

    def __init__(self, maxsize: int = 1000, ttl: int = 3600):
        """
        Initialize TTL cache.

        Args:
            maxsize: Maximum number of items to store
            ttl: Time-to-live in seconds for cached items
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()

    def __contains__(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp <= self.ttl:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return True
            else:
                del self._cache[key]
        return False

    def __getitem__(self, key: str) -> Any:
        """Get value for key if not expired."""
        if key in self:
            return self._cache[key][0]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set value with current timestamp."""
        # Remove expired entries if at capacity
        if len(self._cache) >= self.maxsize:
            self._cleanup_expired()

            # If still at capacity, remove least recently used
            if len(self._cache) >= self.maxsize:
                self._cache.popitem(last=False)

        self._cache[key] = (value, time.time())
        self._cache.move_to_end(key)

    def _cleanup_expired(self) -> None:
        """Remove all expired entries."""
        current_time = time.time()
        expired_keys = [
            k for k, (_, ts) in self._cache.items() if current_time - ts > self.ttl
        ]
        for k in expired_keys:
            del self._cache[k]

    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default if key doesn't exist or is expired."""
        try:
            return self[key]
        except KeyError:
            return default


# ------------------------------------------------------------------------------
# GLOBALS
# ------------------------------------------------------------------------------

_CURSIVES_CACHE = _CursivesCache(maxsize=1000, ttl=3600)
"""Global cache instance for the cursives package."""

TYPE_MAPPING = {
    int: ("integer", int),
    float: ("number", float),
    str: ("string", str),
    bool: ("boolean", bool),
    list: ("array", list),
    dict: ("object", dict),
    tuple: ("array", tuple),
    set: ("array", set),
    Any: ("any", Any),
}
"""Type mapping for JSON Schema generation."""


# ------------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------------------------


def get_value(obj: Any, key: str, default: Any = None) -> Any:
    """
    Safely retrieve a value from an object by attribute or key.

    Args:
        obj: Object to retrieve value from
        key: Attribute name or dictionary key
        default: Default value if not found

    Returns:
        Retrieved value or default
    """
    try:
        # Try attribute access first
        if hasattr(obj, key):
            return getattr(obj, key, default)
        # Try dictionary access
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default
    except Exception:
        return default


def make_hashable(obj: Any) -> str:
    """
    Convert any object to a stable hash string.

    Uses SHA-256 to generate consistent hash representations.
    Handles nested structures recursively.

    Args:
        obj: Object to hash

    Returns:
        Hexadecimal hash string
    """

    def _hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    # Handle None
    if obj is None:
        return _hash_bytes(b"None")

    # Handle primitives
    if isinstance(obj, (str, int, float, bool)):
        return _hash_bytes(str(obj).encode())

    if isinstance(obj, bytes):
        return _hash_bytes(obj)

    # Handle collections
    if isinstance(obj, (tuple, list)):
        items = ",".join(make_hashable(x) for x in obj)
        return _hash_bytes(f"[{items}]".encode())

    if isinstance(obj, set):
        # Sort for consistency
        items = ",".join(make_hashable(x) for x in sorted(obj, key=str))
        return _hash_bytes(f"{{{items}}}".encode())

    if isinstance(obj, dict):
        # Sort items for consistency
        items = ",".join(
            f"{k}:{make_hashable(v)}"
            for k, v in sorted(obj.items(), key=lambda x: str(x[0]))
        )
        return _hash_bytes(f"{{{items}}}".encode())

    # Handle types and functions
    if isinstance(obj, type):
        return _hash_bytes(f"type:{obj.__module__}.{obj.__qualname__}".encode())

    if callable(obj):
        # Include module and qualname for better uniqueness
        module = getattr(obj, "__module__", "unknown")
        name = getattr(obj, "__qualname__", getattr(obj, "__name__", str(obj)))
        return _hash_bytes(f"callable:{module}.{name}".encode())

    # Handle objects with __dict__
    if hasattr(obj, "__dict__"):
        return make_hashable({"__class__": type(obj), **obj.__dict__})

    # Fallback: use repr
    return _hash_bytes(repr(obj).encode())


# ------------------------------------------------------------------------------
# CACHING DECORATORS
# ------------------------------------------------------------------------------


@overload
def cached(function: Callable[P, R]) -> Callable[P, R]:
    """Decorator with automatic key generation."""
    ...


@overload
def cached(
    *,
    key: Optional[Callable[..., str]] = None,
    ttl: Optional[int] = None,
    maxsize: Optional[int] = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator with custom key function and cache settings."""
    ...


def cached(
    function: Optional[Callable[P, R]] = None,
    *,
    key: Optional[Callable[..., str]] = None,
    ttl: Optional[int] = None,
    maxsize: Optional[int] = None,
) -> Union[Callable[P, R], Callable[[Callable[P, R]], Callable[P, R]]]:
    """
    Flexible caching decorator that preserves type hints and signatures.

    Can be used with or without arguments:
    - @cached - Uses automatic key generation
    - @cached(key=lambda x: str(x)) - Custom key function
    - @cached(ttl=300) - Custom TTL

    Args:
        function: Function to cache (when used without parentheses)
        key: Custom key generation function
        ttl: Time-to-live override for this function
        maxsize: Max cache size override

    Returns:
        Decorated function with caching
    """
    # Create custom cache if settings provided
    cache_instance = _CURSIVES_CACHE
    if ttl is not None or maxsize is not None:
        cache_instance = _CursivesCache(
            maxsize=maxsize or _CURSIVES_CACHE.maxsize, ttl=ttl or _CURSIVES_CACHE.ttl
        )

    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        # Generate automatic key function if not provided
        if key is None:
            sig = inspect.signature(f)
            param_names = list(sig.parameters.keys())

            def auto_key(*args: P.args, **kwargs: P.kwargs) -> str:
                # Bind arguments to parameters
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                # Create stable key from all arguments
                key_parts = []
                for name, value in bound.arguments.items():
                    key_parts.append(f"{name}={make_hashable(value)}")

                return f"{f.__module__}.{f.__qualname__}({','.join(key_parts)})"

            key_func = auto_key
        else:
            key_func = key

        @wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                cache_key = key_func(*args, **kwargs)

                if cache_key in cache_instance:
                    return cache_instance[cache_key]

                result = f(*args, **kwargs)
                cache_instance[cache_key] = result
                return result

            except Exception:
                return f(*args, **kwargs)

        wrapper.__wrapped__ = f  # type: ignore
        return wrapper

    if function is None:
        return decorator
    else:
        return decorator(function)


def auto_cached(
    *,
    exclude: Optional[Tuple[str, ...] | list[str] | str] = None,
    include: Optional[Tuple[str, ...] | list[str] | str] = None,
    ttl: Optional[int] = None,
    maxsize: Optional[int] = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Advanced caching decorator with automatic parameter selection.

    Automatically generates cache keys based on function parameters,
    with options to include/exclude specific parameters.

    Args:
        exclude: Parameter names to exclude from cache key
        include: Only these parameters in cache key (exclusive with exclude)
        ttl: Time-to-live override
        maxsize: Max cache size override

    Returns:
        Decorator function

    Example:
        @auto_cached(exclude=('verbose', 'debug'))
        def process_data(data: dict, verbose: bool = False):
            ...
    """
    if exclude and include:
        raise ValueError("Cannot specify both 'exclude' and 'include'")

    if isinstance(exclude, str):
        exclude = (exclude,)
    elif isinstance(exclude, list):
        exclude = tuple(exclude)

    if isinstance(include, str):
        include = (include,)
    elif isinstance(include, list):
        include = tuple(include)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        sig = inspect.signature(func)

        def key_func(*args: P.args, **kwargs: P.kwargs) -> str:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Filter parameters
            params = bound.arguments
            if include:
                params = {k: v for k, v in params.items() if k in include}
            elif exclude:
                params = {k: v for k, v in params.items() if k not in exclude}

            # Generate key
            key_parts = [f"{k}={make_hashable(v)}" for k, v in sorted(params.items())]
            return f"{func.__module__}.{func.__qualname__}({','.join(key_parts)})"

        return cached(func, key=key_func, ttl=ttl, maxsize=maxsize)

    return decorator
