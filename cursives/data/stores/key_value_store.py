"""
cursives.data.stores.key_value_store

```markdown
A service for storing and retrieving key-value pairs, with support for
in-memory and persistent (SQLAlchemy-based) backends, and TTL.
```
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Optional,
    Iterator,
    Literal,
    Tuple,
)

from sqlalchemy import (
    create_engine,
    Engine,
    event,
    pool,
    Column,
    String,
    DateTime,
    JSON as SQLAlchemyJSON,
)
from sqlalchemy.orm import Session as SQLAlchemySession, declarative_base
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

__all__ = [
    "DataKeyValueStore",
    "DataKeyValueStoreConfig",
    "PersistentKeyValueItem",
    "create_data_value_service",
    "StorageType",
]

StorageType = Literal["memory", "persistent"]

Base = declarative_base()


@dataclass
class DataKeyValueStoreConfig:
    """
    ```markdown
    Configuration for DataKeyValueStore initialization.
    ```
    """

    type: StorageType = "memory"
    location: Optional[str] = (
        None  # DB URL for persistent, or path for file-based if extended
    )
    echo_sql: bool = False
    default_ttl: Optional[int] = None  # Default Time-To-Live for keys in seconds

    # Simplified pool settings for this service, can be expanded if needed
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600


class PersistentKeyValueItem(Base):
    """
    ```markdown
    SQLAlchemy model for persistent key-value storage.
    ```
    """

    __tablename__ = "key_value_store"

    key = Column(String(255), primary_key=True, index=True)
    value = Column(SQLAlchemyJSON, nullable=False)  # Store complex values as JSON
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    expires_at = Column(DateTime, nullable=True, index=True)  # For TTL

    def __repr__(self):
        return f"<PersistentKeyValueItem(key='{self.key}', expires_at='{self.expires_at}')>"


class DataKeyValueStore:
    """
    ```markdown
    A service for storing and retrieving key-value pairs, with support for
    in-memory and persistent (SQLAlchemy-based) backends, and TTL.
    ```

    Example:
        ```python
        service = DataKeyValueStore(config=DataKeyValueStoreConfig(type="memory"))
        service.set("key1", "value1")
        print(service.get("key1"))

        service.set("key1", "value1", ttl=10)
        print(service.get("key1"))

        service.delete("key1")
        print(service.get("key1"))

        print(service.exists("key1"))
        ```
    """

    def __init__(self, config: DataKeyValueStoreConfig):
        """
        ```markdown
        Initialize the DataKeyValueStore.
        ```

        Args:
            config: Configuration object for the service.
        """
        self.config = config
        self._engine: Optional[Engine] = None

        # Memory store: Dict[key, Tuple[value, Optional[datetime_expiry_timestamp]]]
        self._memory_store: Dict[str, Tuple[Any, Optional[datetime]]] = {}

        if self.config.type == "persistent":
            self._initialize_persistent_storage()

        logger.info(f"DataKeyValueStore initialized with type: {self.config.type}")

    def _initialize_persistent_storage(self) -> None:
        """Initialize database connection and create tables if they don't exist."""
        if not self.config.location:
            raise ValueError(
                "Database location (URL) must be provided for persistent storage."
            )

        is_sqlite = self.config.location.startswith("sqlite")

        engine_kwargs = {
            "echo": self.config.echo_sql,
            "future": True,  # Recommended for SQLAlchemy 2.0 style
            "pool_pre_ping": True,
        }

        if is_sqlite:
            engine_kwargs.update(
                {
                    "connect_args": {"check_same_thread": False, "timeout": 15},
                    "poolclass": pool.StaticPool,
                }
            )
        else:
            engine_kwargs.update(
                {
                    "pool_size": self.config.pool_size,
                    "max_overflow": self.config.max_overflow,
                    "pool_timeout": self.config.pool_timeout,
                    "pool_recycle": self.config.pool_recycle,
                }
            )

        try:
            self._engine = create_engine(self.config.location, **engine_kwargs)

            if is_sqlite:

                @event.listens_for(self._engine, "connect")
                def set_sqlite_pragma(dbapi_connection, connection_record):
                    cursor = dbapi_connection.cursor()
                    try:
                        cursor.execute("PRAGMA journal_mode=WAL;")
                        cursor.execute("PRAGMA synchronous=NORMAL;")
                        # cursor.execute("PRAGMA foreign_keys=ON;") # If using foreign keys
                    finally:
                        cursor.close()

            # Create table if it doesn't exist
            Base.metadata.create_all(self._engine, checkfirst=True)
            logger.info(f"Persistent storage initialized at {self.config.location}")

        except Exception as e:
            logger.error(f"Failed to initialize persistent storage: {e}", exc_info=True)
            raise RuntimeError(f"Database initialization failed: {e}") from e

    @contextmanager
    def _get_db_session(self) -> Iterator[SQLAlchemySession]:
        """Provide a SQLAlchemy session for database operations."""
        if self.config.type != "persistent" or self._engine is None:
            raise RuntimeError(
                "Database session requested but service is not in persistent mode or engine not initialized."
            )

        session = SQLAlchemySession(self._engine)
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database transaction failed: {e}", exc_info=True)
            raise
        except Exception as e:
            session.rollback()
            logger.error(
                f"An unexpected error occurred during database transaction: {e}",
                exc_info=True,
            )
            raise
        finally:
            session.close()

    def _is_expired_memory(self, key: str) -> bool:
        """
        ```markdown
        Check if a key in the memory store is expired and delete if so.
        ```

        Args:
            key: The key to check.

        Returns:
            True if the key is expired and deleted, False otherwise.
        """
        if key not in self._memory_store:
            return False  # Or True, depending on desired behavior for non-existent keys

        _value, expires_at = self._memory_store[key]
        if expires_at and datetime.now(timezone.utc) >= expires_at:
            del self._memory_store[key]
            logger.debug(f"Key '{key}' expired and removed from memory store.")
            return True
        return False

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        ```markdown
        Store a value associated with a key, with an optional Time-To-Live (TTL).
        ```

        Args:
            key: The key to store the value under.
            value: The value to store.
            ttl: Optional Time-To-Live in seconds. If None, uses config.default_ttl.
        """
        if ttl is None:
            ttl = self.config.default_ttl

        expires_at: Optional[datetime] = None
        if ttl is not None:
            if ttl <= 0:
                logger.warning(f"TTL for key '{key}' must be positive. Ignoring TTL.")
            else:
                expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)

        if self.config.type == "memory":
            self._memory_store[key] = (value, expires_at)
            logger.debug(f"Set key '{key}' in memory store. Expires at: {expires_at}")
        elif self.config.type == "persistent":
            with self._get_db_session() as session:
                item = session.get(PersistentKeyValueItem, key)
                if item:
                    item.value = value
                    # Always store expires_at as UTC-aware
                    if expires_at is not None and expires_at.tzinfo is None:
                        expires_at = expires_at.replace(tzinfo=timezone.utc)
                    item.expires_at = expires_at
                    item.updated_at = datetime.now(timezone.utc)
                    logger.debug(
                        f"Updating key '{key}' in persistent store. Expires at: {expires_at}"
                    )
                else:
                    # Always store expires_at as UTC-aware
                    if expires_at is not None and expires_at.tzinfo is None:
                        expires_at = expires_at.replace(tzinfo=timezone.utc)
                    item = PersistentKeyValueItem(
                        key=key, value=value, expires_at=expires_at
                    )
                    session.add(item)
                    logger.debug(
                        f"Adding new key '{key}' to persistent store. Expires at: {expires_at}"
                    )
        else:
            raise ValueError(f"Unsupported storage type: {self.config.type}")

    def get(self, key: str) -> Optional[Any]:
        """
        ```markdown
        Retrieve a value associated with a key.
        ```

        Args:
            key: The key to retrieve the value for.

        Returns None if the key does not exist or is expired.
        """
        if self.config.type == "memory":
            if self._is_expired_memory(key) or key not in self._memory_store:
                return None
            return self._memory_store[key][0]
        elif self.config.type == "persistent":
            with self._get_db_session() as session:
                item = session.get(PersistentKeyValueItem, key)
                if item:
                    expires_at = item.expires_at
                    # Ensure expires_at is timezone-aware (UTC)
                    if expires_at is not None:
                        if expires_at.tzinfo is None:
                            expires_at = expires_at.replace(tzinfo=timezone.utc)
                    now_utc = datetime.now(timezone.utc)
                    if expires_at and now_utc >= expires_at:
                        logger.debug(
                            f"Key '{key}' expired in persistent store. Deleting."
                        )
                        session.delete(item)
                        # session.commit() is handled by context manager
                        return None
                    return item.value
                return None
        else:
            raise ValueError(f"Unsupported storage type: {self.config.type}")

    def delete(self, key: str) -> bool:
        """
        ```markdown
        Delete a key-value pair.
        ```

        Args:
            key: The key to delete.

        Returns True if the key existed and was deleted, False otherwise.
        """
        if self.config.type == "memory":
            self._is_expired_memory(
                key
            )  # Ensure it's removed if expired before checking presence
            if key in self._memory_store:
                del self._memory_store[key]
                logger.debug(f"Deleted key '{key}' from memory store.")
                return True
            return False
        elif self.config.type == "persistent":
            with self._get_db_session() as session:
                item = session.get(PersistentKeyValueItem, key)
                if item:
                    session.delete(item)
                    logger.debug(f"Deleted key '{key}' from persistent store.")
                    return True
                return False
        else:
            raise ValueError(f"Unsupported storage type: {self.config.type}")

    def exists(self, key: str) -> bool:
        """
        ```markdown
        Check if a key exists and is not expired.
        ```

        Args:
            key: The key to check.
        """
        return self.get(key) is not None

    def _increment_decrement_value(self, key: str, amount: int) -> int:
        """
        ```markdown
        Helper for increment and decrement operations.
        ```

        Args:
            key: The key to increment or decrement.
            amount: The amount to increment or decrement.

        Returns:
            The new value after the increment or decrement.
        """
        current_value_any = self.get(key)

        current_numeric_value: int
        if current_value_any is None:
            current_numeric_value = 0
        elif isinstance(current_value_any, (int, float)):
            current_numeric_value = int(current_value_any)
        else:
            raise TypeError(
                f"Cannot increment/decrement non-numeric value for key '{key}'. Found type: {type(current_value_any)}"
            )

        new_value = current_numeric_value + amount

        # Preserve original TTL if possible. Get the expiry time before `get` potentially deletes it.
        # This is a bit tricky as `get` might delete it. A more robust way would be to fetch without expiry check first.
        # For simplicity, we'll just reset TTL if default_ttl is set, or no TTL.
        # A truly robust TTL preservation would require fetching the expiry timestamp separately.
        self.set(key, new_value)  # Uses default TTL or no TTL
        return new_value

    def increment(self, key: str, amount: int = 1) -> int:
        """
        ```markdown
        Atomically increment a numerical value associated with a key.
        If the key does not exist, it's initialized to 0 before incrementing.
        If the current value is not a number, raises TypeError.

        Note: Atomicity is guaranteed for the memory store. For persistent store,
        it's a read-modify-write operation within a transaction, which is generally
        safe but not as strictly atomic as database-level atomic counters.
        ```

        Args:
            key: The key to increment or decrement.
            amount: The amount to increment or decrement.

        Returns:
            The new value after the increment or decrement.
        """
        if not isinstance(amount, int):
            raise TypeError("Increment amount must be an integer.")
        if self.config.type == "memory":
            # More direct manipulation for atomicity in memory
            self._is_expired_memory(key)
            raw_item = self._memory_store.get(key)

            current_value: int
            original_expires_at: Optional[datetime] = None

            if raw_item is None:
                current_value = 0
            else:
                value, original_expires_at = raw_item
                if not isinstance(value, (int, float)):
                    raise TypeError(
                        f"Cannot increment non-numeric value for key '{key}'. Found type: {type(value)}"
                    )
                current_value = int(value)

            new_value = current_value + amount
            self._memory_store[key] = (
                new_value,
                original_expires_at,
            )  # Preserve original expiry
            logger.debug(f"Incremented key '{key}' in memory to {new_value}.")
            return new_value

        elif self.config.type == "persistent":
            # For persistent store, we use a session to wrap the read-modify-write
            with self._get_db_session() as session:
                item = session.get(PersistentKeyValueItem, key)
                new_value: int

                if item:
                    # Ensure expires_at is timezone-aware before comparison
                    expires_at = item.expires_at
                    if expires_at is not None and expires_at.tzinfo is None:
                        expires_at = expires_at.replace(tzinfo=timezone.utc)

                    if expires_at and datetime.now(timezone.utc) >= expires_at:
                        # Expired, treat as non-existent for increment
                        current_value = 0
                        item.value = current_value + amount  # Set new value
                        item.expires_at = None  # Or re-apply default TTL
                        if self.config.default_ttl:
                            item.expires_at = datetime.now(timezone.utc) + timedelta(
                                seconds=self.config.default_ttl
                            )
                        item.updated_at = datetime.now(timezone.utc)
                    elif isinstance(item.value, (int, float)):
                        current_value = int(item.value)
                        item.value = current_value + amount
                        item.updated_at = datetime.now(timezone.utc)
                    else:
                        raise TypeError(
                            f"Cannot increment non-numeric value for key '{key}'. Found type: {type(item.value)}"
                        )
                    new_value = item.value
                else:
                    # Key does not exist, initialize
                    current_value = 0
                    new_value = current_value + amount
                    expires_at = None
                    if self.config.default_ttl:
                        expires_at = datetime.now(timezone.utc) + timedelta(
                            seconds=self.config.default_ttl
                        )
                    item = PersistentKeyValueItem(
                        key=key, value=new_value, expires_at=expires_at
                    )
                    session.add(item)

                logger.debug(
                    f"Incremented key '{key}' in persistent store to {new_value}."
                )
                return new_value
        else:
            raise ValueError(f"Unsupported storage type: {self.config.type}")

    def decrement(self, key: str, amount: int = 1) -> int:
        """
        ```markdown
        Atomically decrement a numerical value associated with a key.
        If the key does not exist, it's initialized to 0 before decrementing.
        If the current value is not a number, raises TypeError.
        ```
        """
        if not isinstance(amount, int):
            raise TypeError("Decrement amount must be an integer.")
        return self.increment(key, -amount)  # Decrement is just increment by negative


def create_data_key_value_store(
    type: StorageType = "memory",
    location: Optional[str] = None,
    echo_sql: bool = False,
    default_ttl: Optional[int] = None,
) -> DataKeyValueStore:
    """
    ```markdown
    Factory function to create a DataKeyValueStore instance.
    ```

    Args:
        type: Storage backend type ("memory" or "persistent").
        location: Database URL (for "persistent" type).
        echo_sql: Whether to echo SQL statements (for "persistent" type).
        default_ttl: Default Time-To-Live for keys in seconds.

    Example:
        ```python
        service = create_data_value_service(type="memory")
        service.set("key1", "value1")
        print(service.get("key1"))
        ```

    Returns:
        An instance of DataKeyValueStore.
    """
    config = DataKeyValueStoreConfig(
        type=type,
        location=location,
        echo_sql=echo_sql,
        default_ttl=default_ttl,
    )
    return DataKeyValueStore(config)


__all__ = [
    "DataKeyValueStore",
    "DataKeyValueStoreConfig",
    "PersistentKeyValueItem",
    "create_data_value_service",
    "StorageType",
]
