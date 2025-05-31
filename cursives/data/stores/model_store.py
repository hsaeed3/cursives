"""
cursives.data.stores.model_store

```markdown
Enhanced data service module providing flexible storage backends with
type-safe operations for user and session-based data management.
```
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    Generic,
    Iterator,
    Callable,
    cast,
    Literal,
)

from sqlalchemy import (
    Engine,
    MetaData,
    event,
    pool,
    Column,
    Integer,
    String,
    DateTime,
    JSON,
)
from sqlalchemy.orm import (
    Session as SQLAlchemySession,
    declarative_base,
)
from sqlmodel import (
    create_engine,
    and_,
)


__all__ = [
    "DataModelStore",
    "StorageType",
    "DataModelStoreConfig",
    "create_data_service",
]

logger = logging.getLogger(__name__)

# Type variables
T = TypeVar("T")
Base = declarative_base()

# Storage type literal
StorageType = Literal["memory", "persistent"]


class StorageLocation(str, Enum):
    """Storage backend options for the data service."""

    MEMORY = "memory"
    PERSISTENT = "persistent"


@dataclass
class DataModelStoreConfig:
    """Configuration for DataModelStore initialization."""

    type: StorageType = "memory"
    location: Optional[str] = None
    echo_sql: bool = False
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600
    enable_json_cache: bool = True
    enable_compression: bool = False


@dataclass
class BaseDataModel:
    """Base model for all persistent data with common fields."""

    id: Optional[int] = None
    user_id: str = ""
    session_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any] = field(default_factory=dict)


class PersistentDataModel(Base):
    """SQLAlchemy model for persistent storage."""

    __tablename__ = "data_storage"

    id = Column(Integer, primary_key=True)
    user_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    data = Column(JSON, nullable=False)


class DataModelStore(Generic[T]):
    """

    ```markdown
    Enhanced data service with flexible storage backends.

    Provides type-safe storage and retrieval of data partitioned by
    user and session IDs. Supports both in-memory and persistent
    (database) storage with automatic serialization.

    Features:
    - Type-safe operations with generics
    - Automatic schema validation
    - Connection pooling for persistent storage
    - JSON caching for complex objects
    - Transaction support
    - Bulk operations
    ```

    Example:
        ```python
        from dataclasses import dataclass
        from datetime import datetime

        @dataclass
        class ChatMessage:
            content: str
            timestamp: datetime
            role: str

        # Create service
        service = create_data_service(
            ChatMessage,
            type="persistent",
            location="sqlite:///chat.db"
        )

        # Use the service
        service.add("user123", "session456", ChatMessage(
            content="Hello",
            timestamp=datetime.now(),
            role="user"
        ))

        messages = service.get("user123", "session456")
        ```
    """

    def __init__(
        self,
        schema: Type[T],
        *,
        type: StorageType = "memory",
        location: Optional[str] = None,
        echo_sql: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        enable_json_cache: bool = True,
        enable_compression: bool = False,
        config: Optional[DataModelStoreConfig] = None,
    ):
        """
        Initalize a new data object service based on a specified schema
        and location.

        Args:
            schema: The data model class (dataclass or dict-like)
            config: Service configuration
        """
        self.schema = schema

        if config:
            self.config = config
        else:
            self.config = DataModelStoreConfig(
                type=type,
                location=location,
                echo_sql=echo_sql,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                pool_recycle=pool_recycle,
                enable_json_cache=enable_json_cache,
                enable_compression=enable_compression,
            )

        # Storage initialization
        self._memory_store: Dict[str, Dict[str, List[T]]] = {}
        self._engine: Optional[Engine] = None
        self._metadata: Optional[MetaData] = None

        # Caches
        self._json_cache: Dict[str, T] = {} if self.config.enable_json_cache else None

        # Initialize storage backend
        if self.config.type == "persistent":
            self._initialize_persistent_storage()

    def _initialize_persistent_storage(self) -> None:
        """Initialize database connection with optimized settings."""
        if not self.config.location:
            raise ValueError(
                "Database URL must be provided for persistent storage. "
                "Example: 'sqlite:///data.db' or 'postgresql://user:pass@host/db'"
            )

        # Determine if SQLite for optimization
        is_sqlite = str(self.config.location).startswith("sqlite")

        # Create engine with appropriate settings
        engine_kwargs = {
            "echo": self.config.echo_sql,
            "future": True,
            "pool_pre_ping": True,  # Verify connections before use
        }

        if is_sqlite:
            # SQLite optimizations
            engine_kwargs.update(
                {
                    "connect_args": {
                        "check_same_thread": False,
                        "timeout": 15,
                    },
                    "poolclass": pool.StaticPool,  # Better for SQLite
                }
            )
        else:
            # PostgreSQL/MySQL optimizations
            engine_kwargs.update(
                {
                    "pool_size": self.config.pool_size,
                    "max_overflow": self.config.max_overflow,
                    "pool_timeout": self.config.pool_timeout,
                    "pool_recycle": self.config.pool_recycle,
                }
            )

        try:
            self._engine = create_engine(str(self.config.location), **engine_kwargs)

            # SQLite-specific optimizations
            if is_sqlite:

                @event.listens_for(self._engine, "connect")
                def set_sqlite_pragma(dbapi_conn, connection_record):
                    cursor = dbapi_conn.cursor()
                    cursor.execute("PRAGMA journal_mode=WAL")  # Write-ahead logging
                    cursor.execute("PRAGMA synchronous=NORMAL")  # Faster writes
                    cursor.execute("PRAGMA cache_size=10000")  # Larger cache
                    cursor.execute("PRAGMA temp_store=MEMORY")  # Memory temp tables
                    cursor.close()

            # Create tables
            Base.metadata.create_all(self._engine)

            logger.info(f"Initialized persistent storage at {self.config.location}")

        except Exception as e:
            logger.error(f"Failed to initialize persistent storage: {e}")
            raise RuntimeError(
                f"Database initialization failed: {e}\nURL: {self.config.location}"
            ) from e

    @contextmanager
    def session(self) -> Iterator[SQLAlchemySession]:
        """
        Provide a database session with automatic cleanup.

        Example:
            ```python
            with service.session() as session:
                session.add(item)
            ```

        Returns:
            Session: SQLAlchemy session for database operations

        Raises:
            RuntimeError: If not in persistent mode
        """
        if self._engine is None:
            raise RuntimeError(
                "Database session requested but service is not in persistent mode. "
                "Initialize with type='persistent' and provide a database location."
            )

        session = SQLAlchemySession(self._engine)
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _serialize_item(self, item: T) -> Dict[str, Any]:
        """Serialize item for storage."""
        if hasattr(item, "__dataclass_fields__"):
            # Handle dataclass
            return asdict(item)
        elif hasattr(item, "model_dump"):
            return item.model_dump()
        elif hasattr(item, "dict"):
            return item.dict()
        elif isinstance(item, dict):
            return item
        else:
            # Try to convert to dict
            try:
                return vars(item)
            except TypeError:
                raise TypeError(f"Cannot serialize item of type {type(item)}")

    def _deserialize_item(self, data: Dict[str, Any]) -> T:
        """Deserialize item from storage."""
        if self._json_cache and str(data) in self._json_cache:
            return self._json_cache[str(data)]

        try:
            if hasattr(self.schema, "__dataclass_fields__"):
                # Handle dataclass
                item = self.schema(**data)
            elif hasattr(self.schema, "model_validate"):
                item = self.schema.model_validate(data)
            elif hasattr(self.schema, "parse_obj"):
                item = self.schema.parse_obj(data)
            elif callable(self.schema):
                item = self.schema(**data)
            else:
                item = cast(T, data)
        except Exception as e:
            logger.warning(f"Failed to deserialize {data}: {e}")
            item = cast(T, data)

        if self._json_cache:
            self._json_cache[str(data)] = item

        return item

    def _to_persistent(
        self, item: T, user_id: str, session_id: str
    ) -> PersistentDataModel:
        """Convert item to persistent model."""
        return PersistentDataModel(
            user_id=user_id,
            session_id=session_id,
            data=self._serialize_item(item),
            updated_at=datetime.now(timezone.utc),
        )

    def _from_persistent(self, model: PersistentDataModel) -> T:
        """Convert persistent model to item."""
        return self._deserialize_item(model.data)

    # Core operations
    def add(
        self,
        user_id: str,
        session_id: str,
        item: T,
        *,
        batch: Optional[List[T]] = None,
    ) -> None:
        """
        Add item(s) to the service.

        Args:
            user_id (str): The user ID to add data for
            session_id (str): The session ID to add data for
            item (T): The item to add
            batch (List[T] | None): Optional list of items for batch insert

        Example:
            ```python
            service.add("user123", "session456", message)
            service.add("user123", "session456", msg1, batch=[msg2, msg3])
            ```
        """
        items = [item] + (batch or [])

        if self.config.type == "memory":
            # Memory storage
            if user_id not in self._memory_store:
                self._memory_store[user_id] = {}
            if session_id not in self._memory_store[user_id]:
                self._memory_store[user_id][session_id] = []

            self._memory_store[user_id][session_id].extend(items)
            logger.debug(
                f"Added {len(items)} items to memory for {user_id}/{session_id}"
            )

        else:
            # Persistent storage
            with self.session() as session:
                models = [
                    self._to_persistent(item, user_id, session_id) for item in items
                ]
                session.add_all(models)
                logger.debug(
                    f"Added {len(items)} items to database for {user_id}/{session_id}"
                )

    def get(
        self,
        user_id: str,
        session_id: str,
        *,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[str] = None,
        filter_fn: Optional[Callable[[T], bool]] = None,
    ) -> List[T]:
        """
        Retrieve items for a user session.

        Args:
            user_id (str): The user ID to get data for
            session_id (str): The session ID to get data for
            limit (int | None): Maximum number of items to return
            offset (int | None): Number of items to skip
            order_by (str | None): Field name to order by (persistent only)
            filter_fn (Callable[[T], bool] | None): Optional filter function (memory only)

        Example:
            ```python
            items = service.get("user123", "session456")
            ```

        Returns:
            List[T]: List of items for the user session
        """
        if self.config.type == "memory":
            # Memory retrieval
            items = self._memory_store.get(user_id, {}).get(session_id, []).copy()

            # Apply filter
            if filter_fn:
                items = [item for item in items if filter_fn(item)]

            # Apply offset and limit
            if offset:
                items = items[offset:]
            if limit:
                items = items[:limit]

            return items

        else:
            # Persistent retrieval
            with self.session() as session:
                query = session.query(PersistentDataModel).filter(
                    and_(
                        PersistentDataModel.user_id == user_id,
                        PersistentDataModel.session_id == session_id,
                    )
                )

                # Apply ordering
                if order_by and hasattr(PersistentDataModel, order_by):
                    query = query.order_by(getattr(PersistentDataModel, order_by))
                else:
                    query = query.order_by(PersistentDataModel.created_at)

                # Apply pagination
                if offset:
                    query = query.offset(offset)
                if limit:
                    query = query.limit(limit)

                results = query.all()
                return [self._from_persistent(model) for model in results]

    def get_all_sessions(self, user_id: str) -> Dict[str, List[T]]:
        """
        Get all sessions and their data for a user.

        Args:
            user_id (str): The user ID to get data for

        Returns:
            Dict[str, List[T]]: Dictionary mapping session IDs to their items
        """
        if self.config.type == "memory":
            return self._memory_store.get(user_id, {}).copy()
        else:
            with self.session() as session:
                results = (
                    session.query(PersistentDataModel)
                    .filter(PersistentDataModel.user_id == user_id)
                    .all()
                )

                sessions: Dict[str, List[T]] = {}
                for model in results:
                    if model.session_id not in sessions:
                        sessions[model.session_id] = []
                    sessions[model.session_id].append(self._from_persistent(model))

                return sessions

    def count(self, user_id: str, session_id: str) -> int:
        """
        Get count of items for a user session.

        Args:
            user_id (str): The user ID to get data for
            session_id (str): The session ID to get data for

        Returns:
            int: Number of items for the user session
        """
        if self.config.type == "memory":
            return len(self._memory_store.get(user_id, {}).get(session_id, []))
        else:
            with self.session() as session:
                return (
                    session.query(PersistentDataModel)
                    .filter(
                        and_(
                            PersistentDataModel.user_id == user_id,
                            PersistentDataModel.session_id == session_id,
                        )
                    )
                    .count()
                )

    def exists(self, user_id: str, session_id: str) -> bool:
        """Check if any data exists for a user session."""
        return self.count(user_id, session_id) > 0

    def clear(self, user_id: str, session_id: str) -> int:
        """
        Clear all items for a user session.

        Args:
            user_id (str): The user ID to clear data for
            session_id (str): The session ID to clear data for

        Returns:
            int: Number of items cleared
        """
        count = self.count(user_id, session_id)

        if self.config.type == "memory":
            if (
                user_id in self._memory_store
                and session_id in self._memory_store[user_id]
            ):
                del self._memory_store[user_id][session_id]
        else:
            with self.session() as session:
                session.query(PersistentDataModel).filter(
                    and_(
                        PersistentDataModel.user_id == user_id,
                        PersistentDataModel.session_id == session_id,
                    )
                ).delete()

        logger.info(f"Cleared {count} items for {user_id}/{session_id}")
        return count

    def clear_user(self, user_id: str) -> int:
        """
        Clear all data for a user across all sessions.

        Args:
            user_id (str): The user ID to clear data for

        Returns:
            int: Number of sessions cleared
        """
        session_count = len(self.get_all_sessions(user_id))

        if self.config.type == "memory":
            if user_id in self._memory_store:
                del self._memory_store[user_id]
        else:
            with self.session() as session:
                session.query(PersistentDataModel).filter(
                    PersistentDataModel.user_id == user_id
                ).delete()

        logger.info(f"Cleared {session_count} sessions for user {user_id}")
        return session_count

    def migrate_to_persistent(
        self,
        target_location: str,
        *,
        batch_size: int = 100,
    ) -> int:
        """
        Migrate in-memory data to persistent storage.

        Args:
            target_location: Database URL for persistent storage
            batch_size: Number of items to migrate per transaction

        Returns:
            Total number of items migrated
        """
        if self.config.type != "memory":
            raise RuntimeError("Can only migrate from memory storage")

        # Create target service
        target_config = DataModelStoreConfig(
            type="persistent", location=target_location
        )
        target_service = DataModelStore(self.schema, target_config)

        # Migrate data
        total_migrated = 0
        for user_id, sessions in self._memory_store.items():
            for session_id, items in sessions.items():
                # Batch insert
                for i in range(0, len(items), batch_size):
                    batch = items[i : i + batch_size]
                    if batch:
                        target_service.add(
                            user_id, session_id, batch[0], batch=batch[1:]
                        )
                        total_migrated += len(batch)

        logger.info(f"Migrated {total_migrated} items to {target_location}")
        return total_migrated

    def export_json(self, user_id: str, session_id: str) -> str:
        """Export session data as JSON."""
        items = self.get(user_id, session_id)
        data = {
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "items": [self._serialize_item(item) for item in items],
        }
        return json.dumps(data, indent=2, default=str)

    def import_json(self, json_data: str) -> None:
        """Import session data from JSON."""
        data = json.loads(json_data)
        user_id = data["user_id"]
        session_id = data["session_id"]

        items = [self._deserialize_item(item_data) for item_data in data["items"]]
        if items:
            self.add(user_id, session_id, items[0], batch=items[1:])


def create_data_model_store(
    schema: Type[T],
    *,
    type: StorageType = "memory",
    location: Optional[str] = None,
    echo_sql: bool = False,
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_timeout: int = 30,
    pool_recycle: int = 3600,
    enable_json_cache: bool = True,
    enable_compression: bool = False,
) -> DataModelStore[T]:
    """
    Factory function to create a data service.

    Args:
        schema (Type[T]): Data model class (dataclass, BaseModel, or dict-like)
        type (StorageType): Storage backend type ("memory" or "persistent")
        location (str | None): Database URL/path for persistent storage
        echo_sql (bool): Whether to echo SQL statements
        pool_size (int): Maximum number of connections in the pool
        max_overflow (int): Maximum number of connections to allow beyond pool_size
        pool_timeout (int): Connection timeout in seconds
        pool_recycle (int): Recycle connections after this many seconds
        enable_json_cache (bool): Whether to enable JSON caching
        enable_compression (bool): Whether to enable compression

    Returns:
        DataModelStore[T]: Configured DataModelStore instance

    Example:
        ```python
        from dataclasses import dataclass

        @dataclass
        class MyModel:
            name: str
            value: int

        # Memory storage (default)
        service = create_data_service(MyModel)

        # Persistent storage
        service = create_data_service(
            MyModel,
            type="persistent",
            location="sqlite:///data.db",
            echo_sql=True
        )
        ```
    """
    # Create config
    config = DataModelStoreConfig(
        type=type,
        location=location,
        echo_sql=echo_sql,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        pool_recycle=pool_recycle,
        enable_json_cache=enable_json_cache,
        enable_compression=enable_compression,
    )

    return DataModelStore(schema, config=config)
