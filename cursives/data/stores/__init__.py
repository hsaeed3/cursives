"""
cursives.data.stores

```markdown
A collection of stores for storing and retrieving data.
```
"""

import sys
from importlib import import_module
from typing import Any, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .key_value_store import (
        DataKeyValueStore,
        DataKeyValueStoreConfig,
        create_data_key_value_store,
        StorageType,
    )
    from .model_store import (
        DataModelStore,
        DataModelStoreConfig,
        create_data_model_store,
    )


IMPORT_MAP: Dict[str, Tuple[str, str]] = {
    # ----------------------------
    # Key-Value Store
    # ----------------------------
    "DataKeyValueStore": (".key_value_store", "DataKeyValueStore"),
    "DataKeyValueStoreConfig": (".key_value_store", "DataKeyValueStoreConfig"),
    "create_data_key_value_store": (".key_value_store", "create_data_key_value_store"),
    "StorageType": (".key_value_store", "StorageType"),
    # ----------------------------
    # Model Store
    # ----------------------------
    "DataModelStore": (".model_store", "DataModelStore"),
    "DataModelStoreConfig": (".model_store", "DataModelStoreConfig"),
    "create_data_model_store": (".model_store", "create_data_model_store"),
}


def __getattr__(name: str) -> Any:
    """Handle dynamic imports for module attributes."""
    if name in IMPORT_MAP:
        module_path, attr_name = IMPORT_MAP[name]
        module = import_module(module_path, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> "list[str]":
    """Return list of module attributes for auto-completion."""
    return list(__all__)


if sys.version_info >= (3, 7):
    __getattr__.__module__ = __name__


__all__ = [
    "DataKeyValueStore",
    "DataKeyValueStoreConfig",
    "create_data_key_value_store",
    "StorageType",
    "DataModelStore",
    "DataModelStoreConfig",
    "create_data_model_store",
]
