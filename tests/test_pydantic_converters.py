import pytest
from pydantic.fields import FieldInfo
from typing import Literal

from cursives.pydantic.utils import (
    create_pydantic_model,
    create_selection_pydantic_model,
    create_confirmation_pydantic_model,
    convert_type_to_pydantic_field,
)


def test_convert_type_to_pydantic_field():
    # Test with basic type
    result = convert_type_to_pydantic_field(int)
    assert "value" in result
    assert result["value"][0] == int
    assert isinstance(result["value"][1], FieldInfo)

    # Test with description
    result = convert_type_to_pydantic_field(str, description="Test description")
    assert result["value"][1].description == "Test description"

    # Test with default value
    result = convert_type_to_pydantic_field(bool, default=True)
    assert result["value"][1].default is True

    # Test with index
    result = convert_type_to_pydantic_field(float, index=1)
    assert "value_1" in result


def test_create_selection_pydantic_model():
    # Test basic creation
    model = create_selection_pydantic_model(["option1", "option2"])
    assert model.__name__ == "Selection"
    assert "selection" in model.model_fields
    assert model.model_fields["selection"].annotation == Literal["option1", "option2"]

    # Test with custom name and description
    model = create_selection_pydantic_model(
        ["a", "b"], name="CustomSelection", description="Test description"
    )
    assert model.__name__ == "CustomSelection"
    assert model.__doc__ == "Test description"

    # Test empty fields list
    with pytest.raises(ValueError):
        create_selection_pydantic_model([])


def test_create_confirmation_pydantic_model():
    # Test basic creation
    model = create_confirmation_pydantic_model()
    assert model.__name__ == "Confirmation"
    assert "choice" in model.model_fields
    assert model.model_fields["choice"].annotation == bool

    # Test with custom name and description
    model = create_confirmation_pydantic_model(
        name="CustomConfirmation", description="Test description"
    )
    assert model.__name__ == "CustomConfirmation"
    assert model.__doc__ == "Test description"

    # Test with custom field name
    model = create_confirmation_pydantic_model(field_name="confirmed")
    assert "confirmed" in model.model_fields


def test_create_pydantic_model():
    # Test with single type
    model = create_pydantic_model(int)
    assert model.__name__ == "GeneratedModel"
    assert "value" in model.model_fields
    assert model.model_fields["value"].annotation == int

    # Test with dictionary
    data = {"name": "test", "age": 25}
    model = create_pydantic_model(data)
    assert "name" in model.model_fields
    assert "age" in model.model_fields
    assert model.model_fields["name"].annotation == str
    assert model.model_fields["age"].annotation == int

    # Test with sequence of types
    model = create_pydantic_model([str, int, bool])
    assert "value_0" in model.model_fields
    assert "value_1" in model.model_fields
    assert "value_2" in model.model_fields
    assert model.model_fields["value_0"].annotation == str
    assert model.model_fields["value_1"].annotation == int
    assert model.model_fields["value_2"].annotation == bool


if __name__ == "__main__":
    pytest.main()
