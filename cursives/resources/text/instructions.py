"""
✼ cursives.text.instructions

Contains utilities for converting various types into natural language instructions.
"""

import json
from typing import Any, Dict, Union, Optional, Type, Literal, Callable, Sequence
from dataclasses import is_dataclass, fields as dataclass_fields, MISSING
from inspect import signature, Parameter

from pydantic import BaseModel

from .markdown import (
    convert_type_to_markdown,
    _parse_docstring,
)

__all__ = [
    "create_task_instructions",
    "create_output_instructions",
]


# ============================================================================
# Task Instructions - Natural Language Task Generation
# ============================================================================


def create_task_instructions(
    target: Union[
        Type, Sequence[Type], Dict[str, Any], BaseModel, Callable, list, dict
    ],
    name: Optional[str] = None,
    description: Optional[str] = None,
    numbered_steps: bool = True,
    include_types: bool = False,
    include_descriptions: bool = True,
) -> str:
    """
    Converts various input types into natural language task instructions.

    Args:
        target: The target to convert (type, sequence, dict, model, or function)
        name: Optional name for the task
        description: Optional description for the task
        numbered_steps: Whether to use numbered steps (vs bullet points)
        include_types: Whether to include type information in the instructions
        include_descriptions: Whether to include field descriptions

    Returns:
        A string containing natural language task instructions
    """

    # Handle Pydantic model instances and classes
    if isinstance(target, BaseModel) or (
        isinstance(target, type) and issubclass(target, BaseModel)
    ):
        is_instance = isinstance(target, BaseModel)
        model_class = target.__class__ if is_instance else target
        model_name = model_class.__name__

        # Build task header
        task_name = name or f"Complete {model_name} Information"
        parts = [f"## Task: {task_name}\n"]

        # Add description
        if description:
            parts.append(f"{description}\n")
        else:
            # Try to get from docstring
            doc_dict = _parse_docstring(model_class, use_getdoc=False)
            if doc_dict and doc_dict.get("short"):
                parts.append(f"{doc_dict['short']}\n")

        # Add instructions intro
        if is_instance:
            parts.append(
                f"Please update the following {model_name} fields with appropriate values:\n"
            )
        else:
            parts.append(
                f"Please provide values for the following {model_name} fields:\n"
            )

        # Process fields
        model_fields = model_class.model_fields
        step_num = 1

        for field_name, field_info in model_fields.items():
            # Build step prefix
            if numbered_steps:
                prefix = f"{step_num}. "
                step_num += 1
            else:
                prefix = "- "

            # Build field instruction
            field_parts = [f"**{field_name.replace('_', ' ').title()}**"]

            # Add type info if requested
            if include_types and field_info.annotation:
                type_name = convert_type_to_markdown(field_info.annotation)
                field_parts.append(f" ({type_name})")

            field_parts.append(": ")

            # Add current value for instances
            if is_instance:
                current_value = getattr(target, field_name, None)
                if current_value is not None:
                    field_parts.append(f"Currently set to `{repr(current_value)}`. ")

            # Add field description
            if include_descriptions and field_info.description:
                field_parts.append(field_info.description)
            elif include_descriptions:
                # Generate a generic instruction based on type
                type_name = convert_type_to_markdown(field_info.annotation).lower()
                if "str" in type_name or "string" in type_name:
                    field_parts.append(f"Enter a text value for {field_name}")
                elif "int" in type_name:
                    field_parts.append(f"Enter a whole number for {field_name}")
                elif "float" in type_name:
                    field_parts.append(f"Enter a decimal number for {field_name}")
                elif "bool" in type_name:
                    field_parts.append(f"Set to true or false")
                elif "list" in type_name:
                    field_parts.append(f"Provide a list of items")
                elif "dict" in type_name:
                    field_parts.append(f"Provide key-value pairs")
                else:
                    field_parts.append(f"Provide an appropriate value")

            # Add required/optional info
            if field_info.is_required():
                field_parts.append(" *(Required)*")
            else:
                field_parts.append(" *(Optional)*")

            parts.append(prefix + "".join(field_parts))

        # Add completion note
        parts.append("\n*Please ensure all required fields are completed accurately.*")

        return "\n".join(parts)

    # Handle callables (functions)
    elif callable(target) and not isinstance(target, type):
        func_name = target.__name__
        task_name = name or f"Call {func_name} Function"

        parts = [f"## Task: {task_name}\n"]

        # Add description
        if description:
            parts.append(f"{description}\n")
        else:
            doc_dict = _parse_docstring(target, use_getdoc=True)
            if doc_dict and doc_dict.get("short"):
                parts.append(f"{doc_dict['short']}\n")

        # Get function signature
        sig = signature(target)
        params = list(sig.parameters.values())

        if params:
            parts.append(
                f"Please provide the following parameters to call `{func_name}`:\n"
            )

            step_num = 1
            doc_dict = (
                _parse_docstring(target, use_getdoc=True)
                if include_descriptions
                else None
            )

            for param in params:
                if param.name == "self" or param.name == "cls":
                    continue

                if numbered_steps:
                    prefix = f"{step_num}. "
                    step_num += 1
                else:
                    prefix = "- "

                param_parts = [f"**{param.name.replace('_', ' ').title()}**"]

                # Add type annotation if available
                if include_types and param.annotation != Parameter.empty:
                    type_name = convert_type_to_markdown(param.annotation)
                    param_parts.append(f" ({type_name})")

                param_parts.append(": ")

                # Add parameter description from docstring
                if include_descriptions and doc_dict and doc_dict.get("params"):
                    for p_name, p_type, p_desc in doc_dict["params"]:
                        if p_name == param.name:
                            param_parts.append(p_desc)
                            break
                    else:
                        param_parts.append(f"Provide value for {param.name}")
                else:
                    param_parts.append(f"Provide value for {param.name}")

                # Add required/optional info
                if param.default == Parameter.empty:
                    param_parts.append(" *(Required)*")
                else:
                    param_parts.append(f" *(Optional, default: {param.default})*")

                parts.append(prefix + "".join(param_parts))
        else:
            parts.append(f"Call `{func_name}` with no parameters.")

        # Add return info if available
        if doc_dict and doc_dict.get("returns"):
            parts.append(f"\n**Expected Output:** {doc_dict['returns']}")

        return "\n".join(parts)

    # Handle lists
    elif isinstance(target, list):
        task_name = name or "Complete List Items"

        parts = [f"## Task: {task_name}\n"]

        if description:
            parts.append(f"{description}\n")
        else:
            parts.append("Please complete or update the following list items:\n")

        if not target:
            parts.append("*The list is currently empty. Add items as needed.*")
        else:
            for i, item in enumerate(target):
                if numbered_steps:
                    prefix = f"{i + 1}. "
                else:
                    prefix = "- "

                if isinstance(item, (BaseModel, dict)) and hasattr(item, "__dict__"):
                    # For complex objects, show a summary
                    item_type = type(item).__name__
                    parts.append(f"{prefix}Update {item_type} item")
                else:
                    # For simple values
                    if include_types:
                        item_type = convert_type_to_markdown(type(item))
                        parts.append(f"{prefix}Item ({item_type}): `{repr(item)}`")
                    else:
                        parts.append(f"{prefix}Item: `{repr(item)}`")

        parts.append("\n*Modify items as needed to complete the task.*")

        return "\n".join(parts)

    # Handle dictionaries
    elif isinstance(target, dict):
        task_name = name or "Complete Dictionary Values"

        parts = [f"## Task: {task_name}\n"]

        if description:
            parts.append(f"{description}\n")
        else:
            parts.append("Please provide or update values for the following fields:\n")

        if not target:
            parts.append(
                "*The dictionary is currently empty. Add key-value pairs as needed.*"
            )
        else:
            step_num = 1
            for key, value in target.items():
                if numbered_steps:
                    prefix = f"{step_num}. "
                    step_num += 1
                else:
                    prefix = "- "

                key_parts = [f"**{str(key).replace('_', ' ').title()}**"]

                if include_types:
                    type_name = convert_type_to_markdown(type(value))
                    key_parts.append(f" ({type_name})")

                key_parts.append(f": Currently `{repr(value)}`")

                if include_descriptions:
                    # Add generic instruction based on value type
                    if isinstance(value, str):
                        key_parts.append(" - Update with appropriate text")
                    elif isinstance(value, (int, float)):
                        key_parts.append(" - Update with appropriate number")
                    elif isinstance(value, bool):
                        key_parts.append(" - Set to true or false")
                    elif isinstance(value, list):
                        key_parts.append(" - Update list items as needed")
                    elif isinstance(value, dict):
                        key_parts.append(" - Update nested values")
                    else:
                        key_parts.append(" - Update as needed")

                parts.append(prefix + "".join(key_parts))

        parts.append("\n*Update all values to complete the task.*")

        return "\n".join(parts)

    # Handle single types and dataclasses
    elif isinstance(target, type):
        # Check if it's a dataclass
        if is_dataclass(target):
            class_name = target.__name__
            task_name = name or f"Complete {class_name} Information"
            parts = [f"## Task: {task_name}\n"]

            # Add description
            if description:
                parts.append(f"{description}\n")
            else:
                doc_dict = _parse_docstring(target, use_getdoc=False)
                if doc_dict and doc_dict.get("short"):
                    parts.append(f"{doc_dict['short']}\n")

            parts.append(
                f"Please provide values for the following {class_name} fields:\n"
            )

            # Process dataclass fields
            fields = dataclass_fields(target)
            step_num = 1

            for field in fields:
                if numbered_steps:
                    prefix = f"{step_num}. "
                    step_num += 1
                else:
                    prefix = "- "

                field_parts = [f"**{field.name.replace('_', ' ').title()}**"]

                if include_types:
                    type_name = convert_type_to_markdown(field.type)
                    field_parts.append(f" ({type_name})")

                field_parts.append(": ")

                # Add field description from metadata
                if (
                    include_descriptions
                    and hasattr(field, "metadata")
                    and field.metadata
                ):
                    desc = field.metadata.get("description")
                    if desc:
                        field_parts.append(desc)
                    else:
                        field_parts.append(f"Provide a value for {field.name}")
                else:
                    field_parts.append(f"Provide a value for {field.name}")

                parts.append(prefix + "".join(field_parts))

            parts.append("\n*Complete all fields as specified.*")
            return "\n".join(parts)

        # Handle simple types
        type_name = convert_type_to_markdown(target)
        task_name = name or f"Provide {type_name} Value"

        parts = [f"## Task: {task_name}\n"]

        if description:
            parts.append(f"{description}\n")
        else:
            parts.append(f"Please provide a value of type `{type_name}`.\n")

        # Add type-specific guidance
        if "str" in type_name or "string" in type_name:
            parts.append("Enter a text string value.")
        elif "int" in type_name:
            parts.append("Enter a whole number (integer).")
        elif "float" in type_name:
            parts.append("Enter a decimal number.")
        elif "bool" in type_name:
            parts.append("Enter either `true` or `false`.")
        elif "list" in type_name:
            parts.append("Provide a list/array of values.")
        elif "dict" in type_name:
            parts.append("Provide a dictionary/object with key-value pairs.")
        else:
            parts.append(f"Provide a valid {type_name} value.")

        return "\n".join(parts)

    # Fallback for any other type
    else:
        return f"## Task: {name or 'Complete Task'}\n\n{description or 'Please complete the required task.'}"


# ============================================================================
# Output Instructions - Schema Format Documentation
# ============================================================================


def create_output_instructions(
    target: Union[
        Type, Sequence[Type], Dict[str, Any], BaseModel, Callable, list, dict
    ],
    title: str = "Output Format",
    include_examples: bool = True,
    format_style: Literal["json", "yaml", "python", "markdown"] = "json",
    strict_mode: bool = False,
    include_descriptions: bool = True,
) -> str:
    """
    Converts various input types into well-defined output format instructions.

    Args:
        target: The target schema to document
        title: Title for the output format section
        include_examples: Whether to include example values
        format_style: The format style to use for examples
        strict_mode: Whether to enforce strict validation rules
        include_descriptions: Whether to include descriptions for each field

    Returns:
        A string containing output format documentation
    """

    # Handle Pydantic model instances and classes
    if isinstance(target, BaseModel) or (
        isinstance(target, type) and issubclass(target, BaseModel)
    ):
        is_instance = isinstance(target, BaseModel)
        model_class = target.__class__ if is_instance else target
        model_name = model_class.__name__

        parts = [f"# {title}\n"]

        # Add description
        doc_dict = _parse_docstring(model_class, use_getdoc=False)
        if doc_dict and doc_dict.get("short"):
            parts.append(f"_{doc_dict['short']}_\n")

        # Add schema structure
        parts.append("## Schema Structure\n")
        parts.append(
            f"The output must be a valid `{model_name}` object with the following fields:\n"
        )

        # Document fields
        model_fields = model_class.model_fields

        for field_name, field_info in model_fields.items():
            field_parts = [f"### `{field_name}`"]

            # Type
            type_name = convert_type_to_markdown(field_info.annotation)
            field_parts.append(f"- **Type:** `{type_name}`")

            # Required/Optional
            if field_info.is_required():
                field_parts.append("- **Required:** Yes")
            else:
                field_parts.append("- **Required:** No")
                if field_info.default is not None:
                    field_parts.append(f"- **Default:** `{repr(field_info.default)}`")

            # Description
            if field_info.description:
                field_parts.append(f"- **Description:** {field_info.description}")

            # Validation rules (if any)
            if strict_mode and hasattr(field_info, "constraints"):
                constraints = []
                if hasattr(field_info, "ge"):
                    constraints.append(f"≥ {field_info.ge}")
                if hasattr(field_info, "le"):
                    constraints.append(f"≤ {field_info.le}")
                if hasattr(field_info, "min_length"):
                    constraints.append(f"min length: {field_info.min_length}")
                if hasattr(field_info, "max_length"):
                    constraints.append(f"max length: {field_info.max_length}")

                if constraints:
                    field_parts.append(f"- **Constraints:** {', '.join(constraints)}")

            parts.append("\n".join(field_parts))
            parts.append("")  # Empty line between fields

        # Add example
        if include_examples:
            parts.append("## Example\n")

            if is_instance:
                example_data = target.model_dump()
            else:
                # Create example data
                example_data = {}
                for field_name, field_info in model_fields.items():
                    if (
                        field_info.default is not None
                        and str(field_info.default) != "PydanticUndefined"
                    ):
                        example_data[field_name] = field_info.default
                    else:
                        # Generate example based on type
                        type_str = str(field_info.annotation).lower()
                        if "str" in type_str:
                            example_data[field_name] = f"example_{field_name}"
                        elif "int" in type_str:
                            example_data[field_name] = 123
                        elif "float" in type_str:
                            example_data[field_name] = 123.45
                        elif "bool" in type_str:
                            example_data[field_name] = True
                        elif "list" in type_str:
                            example_data[field_name] = []
                        elif "dict" in type_str:
                            example_data[field_name] = {}
                        else:
                            example_data[field_name] = None

            # Format example based on style
            if format_style == "json":
                parts.append("```json")
                parts.append(json.dumps(example_data, indent=2))
                parts.append("```")
            elif format_style == "yaml":
                parts.append("```yaml")
                for key, value in example_data.items():
                    if isinstance(value, (list, dict)):
                        parts.append(f"{key}: {json.dumps(value)}")
                    else:
                        parts.append(f"{key}: {value}")
                parts.append("```")
            elif format_style == "python":
                parts.append("```python")
                parts.append(f"{model_name}(")
                for key, value in example_data.items():
                    parts.append(f"    {key}={repr(value)},")
                parts.append(")")
                parts.append("```")
            elif format_style == "markdown":
                parts.append("```")
                for key, value in example_data.items():
                    parts.append(f"{key}: {value}")
                parts.append("```")

        # Add validation notes if strict mode
        if strict_mode:
            parts.append("\n## Validation Rules\n")
            parts.append("- All required fields must be present")
            parts.append("- Field types must match exactly as specified")
            parts.append("- Any constraints listed must be satisfied")
            parts.append("- No additional fields are allowed unless explicitly stated")

        return "\n".join(parts)

    # Handle callables (functions) - document their return type
    elif callable(target) and not isinstance(target, type):
        func_name = target.__name__

        parts = [f"# {title}\n"]
        parts.append(f"Output format for function `{func_name}`:\n")

        # Get return type annotation
        sig = signature(target)
        return_annotation = sig.return_annotation

        if return_annotation != Parameter.empty:
            return_type = convert_type_to_markdown(return_annotation)
            parts.append(f"## Return Type\n")
            parts.append(f"The function returns a value of type `{return_type}`.\n")

            # If return type is a model, recurse
            if isinstance(return_annotation, type) and (
                issubclass(return_annotation, BaseModel)
                or is_dataclass(return_annotation)
            ):
                # Get the output instructions for the return type
                nested_instructions = create_output_instructions(
                    return_annotation,
                    title="Return Value Structure",
                    include_examples=include_examples,
                    format_style=format_style,
                    strict_mode=strict_mode,
                )
                # Remove the title from nested instructions
                nested_lines = nested_instructions.split("\n")
                if nested_lines[0].startswith("#"):
                    nested_lines = nested_lines[2:]  # Skip title and empty line
                parts.append("\n".join(nested_lines))
            else:
                # Add description from docstring
                doc_dict = _parse_docstring(target, use_getdoc=True)
                if doc_dict and doc_dict.get("returns"):
                    parts.append(f"**Description:** {doc_dict['returns']}\n")

                # Add generic example
                if include_examples:
                    parts.append("## Example Return Value\n")
                    parts.append(f"```{format_style}")

                    type_str = str(return_annotation).lower()
                    if "str" in type_str:
                        parts.append('"result string"')
                    elif "int" in type_str:
                        parts.append("42")
                    elif "float" in type_str:
                        parts.append("3.14")
                    elif "bool" in type_str:
                        parts.append("true")
                    elif "list" in type_str:
                        parts.append('["item1", "item2"]')
                    elif "dict" in type_str:
                        parts.append('{"key": "value"}')
                    else:
                        parts.append("null")

                    parts.append("```")
        else:
            parts.append("## Return Type\n")
            parts.append("The function's return type is not specified.")

            # Check docstring for return info
            doc_dict = _parse_docstring(target, use_getdoc=True)
            if doc_dict and doc_dict.get("returns"):
                parts.append(f"\n**From Documentation:** {doc_dict['returns']}")

        return "\n".join(parts)

    # Handle lists - document as array schema
    elif isinstance(target, list):
        parts = [f"# {title}\n"]
        parts.append("The output must be an array/list.\n")

        if target:
            # Infer item type from first element
            first_item = target[0]
            item_type = convert_type_to_markdown(type(first_item))

            parts.append("## Array Schema\n")
            parts.append(f"- **Item Type:** `{item_type}`")
            parts.append(f"- **Current Length:** {len(target)}")

            if strict_mode:
                parts.append(f"- **Required Length:** Exactly {len(target)} items")

            # Check if all items are same type
            all_same_type = all(type(item) == type(first_item) for item in target)
            if all_same_type:
                parts.append("- **Homogeneous:** Yes (all items must be same type)")
            else:
                parts.append("- **Heterogeneous:** Yes (items can be different types)")
                # Document each position if heterogeneous
                parts.append("\n### Item Types by Position:")
                for i, item in enumerate(target):
                    parts.append(
                        f"- Index {i}: `{convert_type_to_markdown(type(item))}`"
                    )
        else:
            parts.append("## Array Schema\n")
            parts.append("- **Item Type:** Any")
            parts.append("- **Current Length:** 0 (empty)")

        # Add example
        if include_examples:
            parts.append("\n## Example\n")

            if format_style == "json":
                parts.append("```json")
                if target:
                    # Use actual values if simple types
                    if all(
                        isinstance(item, (str, int, float, bool, type(None)))
                        for item in target
                    ):
                        parts.append(json.dumps(target, indent=2))
                    else:
                        # Create simplified example
                        example_list = []
                        for item in target[:3]:  # Show first 3 items
                            if isinstance(item, (str, int, float, bool, type(None))):
                                example_list.append(item)
                            else:
                                example_list.append(f"<{type(item).__name__} object>")
                        if len(target) > 3:
                            example_list.append("...")
                        parts.append(json.dumps(example_list, indent=2))
                else:
                    parts.append("[]")
                parts.append("```")
            else:
                parts.append("```")
                if target:
                    for i, item in enumerate(target[:5]):  # Show first 5 items
                        if isinstance(item, (str, int, float, bool, type(None))):
                            parts.append(f"- {item}")
                        else:
                            parts.append(f"- <{type(item).__name__} object>")
                    if len(target) > 5:
                        parts.append("- ...")
                else:
                    parts.append("(empty list)")
                parts.append("```")

        return "\n".join(parts)

    # Handle dictionaries - document as object schema
    elif isinstance(target, dict):
        parts = [f"# {title}\n"]
        parts.append(
            "The output must be an object/dictionary with the following structure:\n"
        )

        if not target:
            parts.append("*Currently empty - add key-value pairs as needed.*")
        else:
            parts.append("## Object Schema\n")

            for key, value in target.items():
                parts.append(f"### `{key}`")

                value_type = convert_type_to_markdown(type(value))
                parts.append(f"- **Type:** `{value_type}`")

                if value is not None:
                    if isinstance(value, (str, int, float, bool)):
                        parts.append(f"- **Current Value:** `{repr(value)}`")
                    else:
                        parts.append(
                            f"- **Current Value:** <{type(value).__name__} object>"
                        )

                if strict_mode:
                    parts.append("- **Required:** Yes")

                parts.append("")  # Empty line between fields

        if strict_mode and target:
            parts.append("## Validation Rules\n")
            parts.append("- All keys shown above must be present")
            parts.append("- No additional keys are allowed")
            parts.append("- Value types must match exactly")

        # Add example
        if include_examples:
            parts.append("\n## Example\n")

            if format_style == "json":
                parts.append("```json")
                if target:
                    # Create example with actual values if possible
                    example_dict = {}
                    for key, value in target.items():
                        if isinstance(value, (str, int, float, bool, type(None))):
                            example_dict[key] = value
                        elif isinstance(value, list):
                            example_dict[key] = []
                        elif isinstance(value, dict):
                            example_dict[key] = {}
                        else:
                            example_dict[key] = f"<{type(value).__name__}>"
                    parts.append(json.dumps(example_dict, indent=2))
                else:
                    parts.append("{}")
                parts.append("```")
            else:
                parts.append("```")
                if target:
                    for key, value in target.items():
                        if isinstance(value, (str, int, float, bool, type(None))):
                            parts.append(f"{key}: {value}")
                        else:
                            parts.append(f"{key}: <{type(value).__name__}>")
                else:
                    parts.append("(empty object)")
                parts.append("```")

        return "\n".join(parts)

    # Handle single types (including dataclasses)
    elif isinstance(target, type):
        # Check if it's a dataclass
        if is_dataclass(target):
            class_name = target.__name__

            parts = [f"# {title}\n"]

            # Add description
            doc_dict = _parse_docstring(target, use_getdoc=False)
            if doc_dict and doc_dict.get("short"):
                parts.append(f"_{doc_dict['short']}_\n")

            parts.append("## Schema Structure\n")
            parts.append(
                f"The output must be a valid `{class_name}` object with the following fields:\n"
            )

            # Document dataclass fields
            fields = dataclass_fields(target)

            for field in fields:
                field_parts = [f"### `{field.name}`"]

                # Type
                type_name = convert_type_to_markdown(field.type)
                field_parts.append(f"- **Type:** `{type_name}`")

                # Default value
                if hasattr(field, "default") and field.default is not MISSING:
                    field_parts.append(f"- **Default:** `{repr(field.default)}`")
                elif (
                    hasattr(field, "default_factory")
                    and field.default_factory is not MISSING
                ):
                    field_parts.append("- **Default:** (factory function)")
                else:
                    field_parts.append("- **Required:** Yes")

                # Description from metadata
                if (
                    include_descriptions
                    and hasattr(field, "metadata")
                    and field.metadata
                ):
                    desc = field.metadata.get("description")
                    if desc:
                        field_parts.append(f"- **Description:** {desc}")

                parts.append("\n".join(field_parts))
                parts.append("")

            # Add example
            if include_examples:
                parts.append("## Example\n")

                # Generate example data
                example_data = {}
                for field in fields:
                    if hasattr(field, "default") and field.default is not MISSING:
                        example_data[field.name] = field.default
                    else:
                        # Generate based on type name
                        type_str = str(field.type).lower()
                        if "str" in type_str:
                            example_data[field.name] = f"example_{field.name}"
                        elif "int" in type_str:
                            example_data[field.name] = 123
                        elif "float" in type_str:
                            example_data[field.name] = 123.45
                        elif "bool" in type_str:
                            example_data[field.name] = True
                        elif "list" in type_str:
                            example_data[field.name] = []
                        elif "dict" in type_str:
                            example_data[field.name] = {}
                        else:
                            example_data[field.name] = None

                # Format example
                if format_style == "json":
                    parts.append("```json")
                    parts.append(json.dumps(example_data, indent=2))
                    parts.append("```")
                elif format_style == "python":
                    parts.append("```python")
                    parts.append(f"{class_name}(")
                    for key, value in example_data.items():
                        parts.append(f"    {key}={repr(value)},")
                    parts.append(")")
                    parts.append("```")
                else:
                    parts.append("```")
                    for key, value in example_data.items():
                        parts.append(f"{key}: {value}")
                    parts.append("```")

            return "\n".join(parts)

        # Handle simple types
        type_name = convert_type_to_markdown(target)

        parts = [f"# {title}\n"]
        parts.append(f"The output must be a value of type `{type_name}`.\n")

        # Add type-specific guidance
        if "str" in type_name:
            parts.append("## Format Requirements")
            parts.append("- Must be a valid string value")
            parts.append("- Can contain any UTF-8 characters")
            if strict_mode:
                parts.append("- Empty strings are not allowed")
        elif "int" in type_name:
            parts.append("## Format Requirements")
            parts.append("- Must be a whole number (integer)")
            parts.append("- No decimal points allowed")
            if strict_mode:
                parts.append("- Must be within valid integer range")
        elif "float" in type_name:
            parts.append("## Format Requirements")
            parts.append("- Must be a numeric value")
            parts.append("- Can include decimal points")
            parts.append("- Scientific notation is allowed (e.g., 1.23e-4)")
        elif "bool" in type_name:
            parts.append("## Format Requirements")
            parts.append("- Must be exactly `true` or `false`")
            parts.append("- Case sensitive")
        elif "list" in type_name:
            parts.append("## Format Requirements")
            parts.append("- Must be a valid array/list")
            parts.append("- Can be empty unless otherwise specified")
        elif "dict" in type_name:
            parts.append("## Format Requirements")
            parts.append("- Must be a valid object/dictionary")
            parts.append("- Keys must be strings")

        # Add example
        if include_examples:
            parts.append("\n## Example\n")

            if format_style == "json":
                parts.append("```json")
                if "str" in type_name:
                    parts.append('"example string value"')
                elif "int" in type_name:
                    parts.append("42")
                elif "float" in type_name:
                    parts.append("3.14159")
                elif "bool" in type_name:
                    parts.append("true")
                elif "list" in type_name:
                    parts.append('["item1", "item2", "item3"]')
                elif "dict" in type_name:
                    parts.append('{\n  "key1": "value1",\n  "key2": "value2"\n}')
                else:
                    parts.append("null")
                parts.append("```")
            else:
                parts.append("```")
                if "str" in type_name:
                    parts.append("example string value")
                elif "int" in type_name:
                    parts.append("42")
                elif "float" in type_name:
                    parts.append("3.14159")
                elif "bool" in type_name:
                    parts.append("true")
                elif "list" in type_name:
                    parts.append("- item1\n- item2\n- item3")
                elif "dict" in type_name:
                    parts.append("key1: value1\nkey2: value2")
                else:
                    parts.append("null")
                parts.append("```")

        return "\n".join(parts)

    # Fallback
    else:
        return f"# {title}\n\nOutput format documentation not available for this type."
