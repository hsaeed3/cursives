import pytest
from cursives.pydantic.models import (
    SubscriptableBaseModel,
    DynamicModel,
    FrozenModel,
    CacheableModel,
    FlexibleModel,
)


class TestSubscriptableBaseModel:
    def test_dict_like_access(self):
        class TestModel(SubscriptableBaseModel):
            name: str
            age: int

        model = TestModel(name="John", age=30)

        # Test __getitem__
        assert model["name"] == "John"
        assert model["age"] == 30
        with pytest.raises(KeyError):
            _ = model["nonexistent"]

        # Test __setitem__
        model["name"] = "Jane"
        assert model.name == "Jane"

        # Test __contains__
        assert "name" in model
        assert "age" in model
        assert "nonexistent" not in model

        # Test get method
        assert model.get("name") == "Jane"
        assert model.get("nonexistent", "default") == "default"


class TestDynamicModel:
    def test_dynamic_field_assignment(self):
        model = DynamicModel()

        # Test dynamic field assignment
        model.name = "John"
        model.age = 30
        model.metadata = {"key": "value"}

        assert model.name == "John"
        assert model.age == 30
        assert model.metadata == {"key": "value"}

        # Test dict-like access
        assert model["name"] == "John"
        assert model["age"] == 30

        # Test to_dict method
        data = model.to_dict()
        assert data["name"] == "John"
        assert data["age"] == 30
        assert data["metadata"] == {"key": "value"}


class TestFrozenModel:
    def test_immutability(self):
        class TestModel(FrozenModel):
            name: str
            age: int

        model = TestModel(name="John", age=30)

        # Test immutability
        with pytest.raises(Exception):
            model.name = "Jane"

        # Test with_changes
        new_model = model.with_changes(name="Jane")
        assert new_model.name == "Jane"
        assert new_model.age == 30

        # Test evolve (alias for with_changes)
        new_model = model.evolve(name="Jane")
        assert new_model.name == "Jane"
        assert new_model.age == 30


class TestCacheableModel:
    def test_cached_property(self):
        class TestModel(CacheableModel):
            value: int

            @CacheableModel.cached_property(dependencies=["value"])
            def squared(self) -> int:
                return self.value**2

        model = TestModel(value=5)

        # Test initial computation
        assert model.squared == 25

        # Test cache invalidation
        model.value = 6
        assert model.squared == 36

        # Test cache clearing
        model.clear_cache()
        assert model.squared == 36  # Should recompute


class TestFlexibleModel:
    def test_flexible_features(self):
        class TestModel(FlexibleModel):
            name: str
            age: int

        model = TestModel(name="John", age=30)

        # Test dynamic field
        model.nickname = "Johnny"
        assert model.nickname == "Johnny"

        # Test partial update
        model.partial_update(name="Jane", age=25)
        assert model.name == "Jane"
        assert model.age == 25

        # Test safe_get
        assert model.safe_get("name") == "Jane"
        assert model.safe_get("nonexistent", "default") == "default"

        # Test validation
        assert model.validate_field("age", lambda x: x > 0)
        assert not model.validate_field("age", lambda x: x > 30)

        # Test to_json_safe
        model.complex_data = {"key": lambda x: x}  # Non-serializable
        json_data = model.to_json_safe()
        assert json_data["name"] == "Jane"
        assert json_data["age"] == 25
        assert isinstance(json_data["complex_data"], str)

        # Test from_dict
        new_model = TestModel.from_dict({"name": "Bob", "age": 35})
        assert new_model.name == "Bob"
        assert new_model.age == 35

        # Test strict mode
        strict_model = TestModel.from_dict(
            {"name": "Bob", "age": 35, "extra": "field"}, strict=True
        )
        assert not hasattr(strict_model, "extra")


if __name__ == "__main__":
    pytest.main()
