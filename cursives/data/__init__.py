"""
cursives.data

```markdown
A collection of resources for working with persistent
data / data objects.
```
"""

from .stores import (
    DataKeyValueStore,
    DataKeyValueStoreConfig,
    create_data_key_value_store,
    StorageType,
    DataModelStore,
    DataModelStoreConfig,
    create_data_model_store,
)


__all__ = [
    # ----------------------------
    # Stores
    # ----------------------------
    "DataKeyValueStore",
    "DataKeyValueStoreConfig",
    "create_data_key_value_store",
    "StorageType",
    "DataModelStore",
    "DataModelStoreConfig",
    "create_data_model_store",
]
