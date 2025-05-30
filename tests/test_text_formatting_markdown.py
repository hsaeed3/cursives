import pytest
from dataclasses import dataclass
from typing import Optional, List
from pydantic import BaseModel, Field

from cursives.text.formatting.markdown import (
    convert_to_markdown,
    convert_function_to_markdown, 
    convert_dataclass_to_markdown,
    convert_pydantic_model_to_markdown,
    convert_object_to_markdown,
    Markdown,
    MarkdownSection,
    MarkdownSettings,
)


# Test fixtures
def sample_function(param1: str, param2: int = 5) -> str:
    """
    A sample function for testing.
    
    This function demonstrates parameter handling and return values.
    
    Args:
        param1: A string parameter
        param2: An integer parameter with default value
        
    Returns:
        A formatted string
        
    Raises:
        ValueError: If param1 is empty
    """
    if not param1:
        raise ValueError("param1 cannot be empty")
    return f"{param1}: {param2}"


@dataclass
class SampleDataclass:
    """A sample dataclass for testing."""
    name: str
    age: int = 25
    email: Optional[str] = None


class SamplePydanticModel(BaseModel):
    """A sample Pydantic model for testing."""
    name: str = Field(description="The user's name")
    age: int = Field(default=25, description="The user's age")
    email: Optional[str] = Field(default=None, description="The user's email address")


class TestMarkdownModels:
    """Test the new Markdown and MarkdownSection models."""
    
    def test_markdown_section_creation(self):
        """Test MarkdownSection model creation."""
        section = MarkdownSection(
            heading="Test Section",
            content="Test content",
            level="h2",
            section_type="test",
            metadata={"key": "value"}
        )
        
        assert section.heading == "Test Section"
        assert section.content == "Test content"
        assert section.level == "h2"
        assert section.section_type == "test"
        assert section.metadata == {"key": "value"}

    def test_markdown_model_creation(self):
        """Test Markdown model creation."""
        section = MarkdownSection(heading="Section", content="Content")
        markdown = Markdown(
            title="Test Title",
            description="Test description",
            sections=[section],
            metadata={"type": "test"}
        )
        
        assert markdown.title == "Test Title"
        assert markdown.description == "Test description"
        assert len(markdown.sections) == 1
        assert markdown.sections[0] == section
        assert markdown.metadata == {"type": "test"}

    def test_markdown_to_string_basic(self):
        """Test basic to_string functionality."""
        section = MarkdownSection(heading="Section", content="Content", level="h2")
        markdown = Markdown(title="Test", sections=[section])
        
        result = markdown.to_string()
        
        assert "# Test" in result
        assert "## Section" in result
        assert "Content" in result

    def test_markdown_to_string_with_bullets(self):
        """Test to_string with bullet formatting."""
        section = MarkdownSection(content="item1: value1", section_type="field")
        markdown = Markdown(title="Test", sections=[section])
        
        result = markdown.to_string(show_bullets=True, bullet_style="-")
        
        assert "- item1: value1" in result

    def test_markdown_to_string_title_levels(self):
        """Test different title levels in to_string."""
        markdown = Markdown(title="Test Title")
        
        # Test different title levels
        h1_result = markdown.to_string(title_level="h1")
        assert "# Test Title" in h1_result
        
        h2_result = markdown.to_string(title_level="h2")
        assert "## Test Title" in h2_result
        
        bold_result = markdown.to_string(title_level="bold")
        assert "**Test Title**" in bold_result


class TestReturnStringParameter:
    """Test the return_string parameter functionality."""
    
    def test_return_string_false_default(self):
        """Test that return_string=False returns Markdown object by default."""
        result = convert_to_markdown(sample_function)
        assert isinstance(result, Markdown)
        assert result.title == "sample_function"

    def test_return_string_true(self):
        """Test that return_string=True returns string."""
        result = convert_to_markdown(sample_function, return_string=True)
        assert isinstance(result, str)
        assert "# sample_function" in result

    def test_return_string_backward_compatibility(self):
        """Test that string output matches previous behavior."""
        # Get structured output and convert to string
        structured = convert_to_markdown(sample_function, return_string=False)
        structured_as_string = structured.to_string()
        
        # Get direct string output
        direct_string = convert_to_markdown(sample_function, return_string=True)
        
        # They should be equivalent (allowing for minor formatting differences)
        assert "sample_function" in structured_as_string
        assert "sample_function" in direct_string
        assert "A sample function for testing" in structured_as_string
        assert "A sample function for testing" in direct_string


class TestFunctionConversion:
    """Test function conversion to structured markdown."""
    
    def test_function_basic_structure(self):
        """Test basic function conversion structure."""
        result = convert_function_to_markdown(sample_function)
        
        assert isinstance(result, Markdown)
        assert result.title == "sample_function"
        assert "A sample function for testing" in result.description
        assert result.metadata["type"] == "function"

    def test_function_docstring_sections(self):
        """Test that function docstring creates appropriate sections."""
        result = convert_function_to_markdown(sample_function)
        
        # Should have sections for parameters, returns, and raises
        section_headings = [s.heading for s in result.sections if s.heading]
        assert "Parameters" in section_headings
        assert "Returns" in section_headings
        assert "Raises" in section_headings

    def test_function_parameters_section(self):
        """Test function parameters section content."""
        result = convert_function_to_markdown(sample_function)
        
        params_section = next(s for s in result.sections if s.heading == "Parameters")
        assert "param1" in params_section.content
        assert "param2" in params_section.content
        assert "str" in params_section.content
        assert "int" in params_section.content

    def test_function_with_settings(self):
        """Test function conversion with various settings."""
        result = convert_function_to_markdown(
            sample_function, 
            {"show_docs": False, "override_title": "Custom Title"}
        )
        
        assert result.title == "Custom Title"
        assert not result.description  # docs disabled
        assert len(result.sections) == 0  # no docstring sections


class TestDataclassConversion:
    """Test dataclass conversion to structured markdown."""
    
    def test_dataclass_basic_structure(self):
        """Test basic dataclass conversion structure."""
        instance = SampleDataclass(name="John", age=30, email="john@example.com")
        result = convert_dataclass_to_markdown(instance)
        
        assert isinstance(result, Markdown)
        assert result.title == "SampleDataclass"
        assert "A sample dataclass for testing" in result.description
        assert result.metadata["type"] == "dataclass"
        assert result.metadata["is_class"] == False

    def test_dataclass_field_sections(self):
        """Test dataclass field sections."""
        instance = SampleDataclass(name="John")
        result = convert_dataclass_to_markdown(instance)
        
        # Should have sections for each field
        field_contents = [s.content for s in result.sections if s.section_type == "field"]
        assert len(field_contents) == 3  # name, age, email
        
        # Check field content includes types and values
        name_content = next(s.content for s in result.sections 
                          if s.metadata.get("field_name") == "name")
        assert "str" in name_content
        assert "'John'" in name_content

    def test_dataclass_split_mode(self):
        """Test dataclass split mode creates separate field sections."""
        instance = SampleDataclass(name="John")
        result = convert_dataclass_to_markdown(instance, {"split": True})
        
        # Should have separate sections with headings for each field
        field_sections = [s for s in result.sections if s.section_type == "field" and s.heading]
        assert len(field_sections) == 3
        
        field_headings = [s.heading for s in field_sections]
        assert "name" in field_headings
        assert "age" in field_headings
        assert "email" in field_headings

    def test_dataclass_natural_language(self):
        """Test dataclass natural language mode."""
        instance = SampleDataclass(name="John", age=30)
        result = convert_dataclass_to_markdown(instance, {"as_natural_language": True})
        
        nl_section = next(s for s in result.sections if s.section_type == "natural_language")
        assert "currently set with the following values" in nl_section.content
        assert "Name (A str) is defined as 'John'" in nl_section.content


class TestPydanticModelConversion:
    """Test Pydantic model conversion to structured markdown."""
    
    def test_pydantic_basic_structure(self):
        """Test basic Pydantic model conversion structure."""
        instance = SamplePydanticModel(name="Alice", age=28)
        result = convert_pydantic_model_to_markdown(instance)
        
        assert isinstance(result, Markdown)
        assert result.title == "SamplePydanticModel"
        assert "A sample Pydantic model for testing" in result.description
        assert result.metadata["type"] == "pydantic_model"

    def test_pydantic_field_descriptions(self):
        """Test that Pydantic field descriptions are included."""
        instance = SamplePydanticModel(name="Alice")
        result = convert_pydantic_model_to_markdown(instance, {"split": True})
        
        # Find the name field section
        name_section = next(s for s in result.sections 
                          if s.heading == "name" and s.section_type == "field")
        assert "The user's name" in name_section.content

    def test_pydantic_class_vs_instance(self):
        """Test difference between class and instance conversion."""
        class_result = convert_pydantic_model_to_markdown(SamplePydanticModel)
        instance_result = convert_pydantic_model_to_markdown(SamplePydanticModel(name="Test"))
        
        assert class_result.metadata["is_class"] == True
        assert instance_result.metadata["is_class"] == False
        
        # Instance should show values, class should not
        instance_name_content = next(s.content for s in instance_result.sections 
                                   if s.metadata.get("field_name") == "name")
        assert "'Test'" in instance_name_content


class TestGenericObjectConversion:
    """Test generic object conversion to structured markdown."""
    
    def test_list_conversion(self):
        """Test list object conversion."""
        test_list = [1, 2, "three", 4.0]
        result = convert_object_to_markdown(test_list)
        
        assert isinstance(result, Markdown)
        assert result.title == "list"
        assert result.metadata["type"] == "generic_object"
        
        collection_section = next(s for s in result.sections if s.section_type == "collection")
        assert "Item 0: 1" in collection_section.content
        assert "Item 2: 'three'" in collection_section.content

    def test_dict_conversion(self):
        """Test dictionary object conversion."""
        test_dict = {"key1": "value1", "key2": 42}
        result = convert_object_to_markdown(test_dict)
        
        dict_section = next(s for s in result.sections if s.section_type == "dictionary")
        assert "key1: 'value1'" in dict_section.content
        assert "key2: 42" in dict_section.content

    def test_empty_collections(self):
        """Test empty collection handling."""
        empty_list = []
        result = convert_object_to_markdown(empty_list)
        
        collection_section = next(s for s in result.sections if s.section_type == "collection")
        assert "Empty list" in collection_section.content
        assert collection_section.metadata["length"] == 0


class TestSettingsAndOptions:
    """Test various settings and options."""
    
    def test_exclude_fields(self):
        """Test excluding specific fields."""
        instance = SampleDataclass(name="John", age=30, email="john@example.com")
        result = convert_dataclass_to_markdown(instance, {"exclude": ["email"]})
        
        # Should not have email field
        field_names = [s.metadata.get("field_name") for s in result.sections 
                      if s.section_type == "field"]
        assert "email" not in field_names
        assert "name" in field_names
        assert "age" in field_names

    def test_hide_title(self):
        """Test hiding title."""
        result = convert_to_markdown(sample_function, show_title=False)
        assert result.title is None

    def test_hide_types(self):
        """Test hiding type information."""
        instance = SampleDataclass(name="John")
        result = convert_dataclass_to_markdown(instance, {"show_types": False})
        
        # Convert to string to check content
        result_str = result.to_string()
        assert "str" not in result_str  # type should be hidden

    def test_code_block_wrapping(self):
        """Test code block wrapping returns string."""
        result = convert_to_markdown(sample_function, as_code_block=True)
        
        assert isinstance(result, str)
        assert result.startswith("```python")
        assert result.endswith("```")

    def test_override_title_and_description(self):
        """Test overriding title and description."""
        result = convert_to_markdown(
            sample_function,
            override_title="Custom Function",
            override_description="Custom description"
        )
        
        assert result.title == "Custom Function"


class TestBackwardCompatibility:
    """Test that the new system maintains backward compatibility."""
    
    def test_string_output_equivalent(self):
        """Test that string output is equivalent to previous behavior."""
        test_objects = [
            sample_function,
            SampleDataclass(name="test"),
            SamplePydanticModel(name="test"),
            [1, 2, 3],
            {"key": "value"}
        ]
        
        for obj in test_objects:
            # Get structured result and convert to string
            structured = convert_to_markdown(obj, return_string=False)
            structured_string = structured.to_string()
            
            # Get direct string result
            direct_string = convert_to_markdown(obj, return_string=True)
            
            # Both should be strings and contain key elements
            assert isinstance(structured_string, str)
            assert isinstance(direct_string, str)
            
            # Should contain the object name/title
            obj_name = getattr(obj, '__name__', obj.__class__.__name__)
            assert obj_name in structured_string
            assert obj_name in direct_string

    def test_all_settings_work_with_both_modes(self):
        """Test that all settings work in both string and structured modes."""
        settings_to_test = [
            {"split": True},
            {"as_natural_language": True},
            {"show_types": False},
            {"show_values": False},
            {"exclude": ["age"]},
            {"bullet_style": "*"},
            {"title_level": "h2"},
        ]
        
        instance = SampleDataclass(name="Test", age=25)
        
        for settings in settings_to_test:
            # Both modes should work without errors
            structured = convert_dataclass_to_markdown(instance, {**settings, "return_string": False})
            string_result = convert_dataclass_to_markdown(instance, {**settings, "return_string": True})
            
            assert isinstance(structured, Markdown)
            assert isinstance(string_result, str)


if __name__ == "__main__":
    pytest.main()