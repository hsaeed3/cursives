"""
✼ cursives.pydantic.utils

Contains converters for converting various types to Pydantic models using plum-dispatch.
"""

import json
import logging
from pathlib import Path
from dataclasses import is_dataclass
from docstring_parser import parse
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    Union,
    get_type_hints,
)

from pydantic import BaseModel, Field, create_model
from beartype.typing import Callable, Sequence
from plum import dispatch

from .._cache import cached, make_hashable, TYPE_MAPPING

logger = logging.getLogger(__name__)

__all__ = [
    "is_pydantic_model_class",
    "extract_function_fields",
    "convert_type_to_pydantic_field",
    "create_pydantic_model",
    "create_selection_pydantic_model",
]


def is_pydantic_model_class(obj: Any) -> bool:
    """
    Checks if an object is a Pydantic model class.

    Args:
        obj: The object to check

    Returns:
        True if the object is a Pydantic model class, False otherwise
    """
    return isinstance(obj, type) and issubclass(obj, BaseModel)


def extract_function_fields(func: Callable) -> Dict[str, Any]:
    """
    Extract fields from a function's signature and docstring.

    Args:
        func: The function to extract fields from

    Returns:
        Dictionary mapping field names to (type, Field) tuples
    """
    try:
        hints = get_type_hints(func)
        fields = {}

        # Parse docstring if available
        docstring = func.__doc__
        doc_info = None
        if docstring:
            doc_info = parse(docstring)

        for param_name, param_type in hints.items():
            if param_name == "return":
                continue

            description = ""
            if doc_info and doc_info.params:
                description = next(
                    (
                        p.description
                        for p in doc_info.params
                        if p.arg_name == param_name
                    ),
                    "",
                )

            fields[param_name] = (
                param_type,
                Field(default=..., description=description),
            )

        return fields
    except Exception as e:
        logger.debug(f"Error extracting function fields: {e}")
        return {}


__all__ = [
    "create_pydantic_model",
    "create_selection_pydantic_model", 
    "create_confirmation_pydantic_model",
    "convert_type_to_pydantic_field",
]


def convert_type_to_pydantic_field(
    type_hint: Type,
    index: Optional[int] = None,
    description: Optional[str] = None,
    default: Any = ...,
) -> Dict[str, Any]:
    """
    Creates a Pydantic field mapping from a type hint.

    Args:
        type_hint: The Python type to convert
        index: Optional index to append to field name for uniqueness
        description: Optional field description
        default: Optional default value

    Returns:
        Dictionary mapping field name to (type, Field) tuple
    """

    @cached(
        lambda type_hint, index=None, description=None, default=...: make_hashable(
            (type_hint, index, description, default)
        )
    )
    def _create_field_mapping(
        type_hint: Type,
        index: Optional[int] = None,
        description: Optional[str] = None,
        default: Any = ...,
    ) -> Dict[str, Any]:
        try:
            base_name, _ = TYPE_MAPPING.get(type_hint, ("value", type_hint))
            # Always use "value" as the base name unless an index is provided
            if index is not None:
                field_name = f"value_{index}"
            else:
                field_name = "value"
            return {
                field_name: (
                    type_hint,
                    Field(default=default, description=description),
                )
            }
        except Exception as e:
            logger.debug(f"Error creating field mapping: {e}")
            raise

    return _create_field_mapping(type_hint, index, description, default)


# ============================================================================
# Dispatched create_pydantic_model implementations
# ============================================================================

@dispatch.abstract
def create_pydantic_model(
    target: Union[Type, Sequence[Type], Dict[str, Any], BaseModel, Callable],
    init: bool = False,
    name: Optional[str] = None,
    description: Optional[str] = None,
    field_name: Optional[str] = None,
    default: Any = ...,
) -> Union[Type[BaseModel], BaseModel]:
    """
    Converts various input types into a pydantic model class or instance.

    Args:
        target: The target to convert (type, sequence, dict, model, or function)
        init: Whether to initialize the model with values (for dataclasses/dicts)
        name: Optional name for the generated model
        description: Optional description for the model/field
        field_name: Optional field name for the generated model (If the target is a single type)
        default: Optional default value for single-type models

    Returns:
        A pydantic model class or instance if init=True
    """


@dispatch
def create_pydantic_model(
    target: type,
    init: bool = False,
    name: Optional[str] = None,
    description: Optional[str] = None,
    field_name: Optional[str] = None,
    default: Any = ...,
) -> Union[Type[BaseModel], BaseModel]:
    """Handle single types and BaseModel subclasses."""
    
    @cached(
        lambda target, init=False, name=None, description=None, field_name=None, default=...: make_hashable(
            (target, init, name, description, field_name, default)
        )
    )
    def _create_from_type(target, init, name, description, field_name, default):
        model_name = name or "GeneratedModel"
        
        # Handle existing Pydantic models
        if issubclass(target, BaseModel):
            return target
            
        # Handle dataclasses
        if is_dataclass(target):
            hints = get_type_hints(target)
            fields = {}

            # Parse docstring if available
            docstring = target.__doc__
            doc_info = None
            if docstring:
                doc_info = parse(docstring)

            for field_name_dc, hint in hints.items():
                description_dc = ""
                if doc_info and doc_info.params:
                    description_dc = next(
                        (
                            p.description
                            for p in doc_info.params
                            if p.arg_name == field_name_dc
                        ),
                        "",
                    )

                fields[field_name_dc] = (
                    hint,
                    Field(
                        default=getattr(target, field_name_dc) if init else ...,
                        description=description_dc,
                    ),
                )

            model_class = create_model(
                model_name,
                __doc__=description
                or (doc_info.short_description if doc_info else None),
                **fields,
            )

            if init and isinstance(target, type):
                return model_class
            elif init:
                return model_class(
                    **{field_name_dc: getattr(target, field_name_dc) for field_name_dc in hints}
                )
            return model_class

        # Handle regular single types
        field_mapping = convert_type_to_pydantic_field(
            target, description=description, default=default
        )
        # If field_name is provided, override the default field name
        if field_name:
            # Get the first (and only) key-value pair from field_mapping
            _, field_value = next(iter(field_mapping.items()))
            field_mapping = {field_name: field_value}
        return create_model(model_name, __doc__=description, **field_mapping)
    
    return _create_from_type(target, init, name, description, field_name, default)


@dispatch
def create_pydantic_model(
    target: Callable,
    init: bool = False,
    name: Optional[str] = None,
    description: Optional[str] = None,
    field_name: Optional[str] = None,
    default: Any = ...,
) -> Union[Type[BaseModel], BaseModel]:
    """Handle callable (functions)."""
    
    @cached(
        lambda target, init=False, name=None, description=None, field_name=None, default=...: make_hashable(
            (target, init, name, description, field_name, default)
        )
    )
    def _create_from_callable(target, init, name, description, field_name, default):
        # Skip if it's a type constructor
        if isinstance(target, type):
            return create_pydantic_model(target, init, name, description, field_name, default)
            
        fields = extract_function_fields(target)

        # Extract just the short description from the docstring
        doc_info = parse(target.__doc__ or "")
        clean_description = doc_info.short_description if doc_info else None

        return create_model(
            name or target.__name__,
            __doc__=description or clean_description,
            **fields,
        )
    
    return _create_from_callable(target, init, name, description, field_name, default)


@dispatch
def create_pydantic_model(
    target: Sequence,
    init: bool = False,
    name: Optional[str] = None,
    description: Optional[str] = None,
    field_name: Optional[str] = None,
    default: Any = ...,
) -> Union[Type[BaseModel], BaseModel]:
    """Handle sequences of types."""
    
    @cached(
        lambda target, init=False, name=None, description=None, field_name=None, default=...: make_hashable(
            (tuple(target), init, name, description, field_name, default)
        )
    )
    def _create_from_sequence(target, init, name, description, field_name, default):
        model_name = name or "GeneratedModel"
        field_mapping = {}
        
        for i, type_hint in enumerate(target):
            if not isinstance(type_hint, type):
                raise ValueError("Sequence elements must be types")
            # If field_name is provided and this is the first type, use it
            if field_name and i == 0:
                field_mapping.update(
                    {
                        field_name: convert_type_to_pydantic_field(
                            type_hint,
                            description=description,
                            default=default,
                        )[
                            next(
                                iter(
                                    convert_type_to_pydantic_field(type_hint).keys()
                                )
                            )
                        ]
                    }
                )
            else:
                field_mapping.update(
                    convert_type_to_pydantic_field(type_hint, index=i)
                )
        return create_model(model_name, __doc__=description, **field_mapping)
    
    return _create_from_sequence(target, init, name, description, field_name, default)


@dispatch  
def create_pydantic_model(
    target: dict,
    init: bool = False,
    name: Optional[str] = None,
    description: Optional[str] = None,
    field_name: Optional[str] = None,
    default: Any = ...,
) -> Union[Type[BaseModel], BaseModel]:
    """Handle dictionaries."""
    
    @cached(
        lambda target, init=False, name=None, description=None, field_name=None, default=...: make_hashable(
            (tuple(sorted(target.items())), init, name, description, field_name, default)
        )
    )
    def _create_from_dict(target, init, name, description, field_name, default):
        model_name = name or "GeneratedModel"
        
        if init:
            model_class = create_model(
                model_name,
                __doc__=description,
                **{k: (type(v), Field(default=v)) for k, v in target.items()},
            )
            return model_class(**target)
        # Create proper field definitions from dict values
        field_mapping = {
            k: (type(v), Field(default=...)) for k, v in target.items()
        }
        return create_model(model_name, __doc__=description, **field_mapping)
    
    return _create_from_dict(target, init, name, description, field_name, default)


@dispatch
def create_pydantic_model(
    target: BaseModel,
    init: bool = False,
    name: Optional[str] = None,
    description: Optional[str] = None,
    field_name: Optional[str] = None,
    default: Any = ...,
) -> Union[Type[BaseModel], BaseModel]:
    """Handle model instances."""
    
    @cached(
        lambda target, init=False, name=None, description=None, field_name=None, default=...: make_hashable(
            (target.__class__, tuple(sorted(target.model_dump().items())), init, name, description, field_name, default)
        )
    )
    def _create_from_instance(target, init, name, description, field_name, default):
        model_name = name or "GeneratedModel"
        
        # Parse docstring from the model's class
        docstring = target.__class__.__doc__
        doc_info = None
        if docstring:
            doc_info = parse(docstring)

        if init:
            fields = {}
            for k, v in target.model_dump().items():
                description_field = ""
                if doc_info and doc_info.params:
                    description_field = next(
                        (p.description for p in doc_info.params if p.arg_name == k),
                        "",
                    )
                fields[k] = (
                    type(v),
                    Field(default=v, description=description_field),
                )

            model_class = create_model(
                model_name,
                __doc__=description
                or (doc_info.short_description if doc_info else None),
                **fields,
            )
            return model_class(**target.model_dump())
        return target.__class__
    
    return _create_from_instance(target, init, name, description, field_name, default)


def create_selection_pydantic_model(
    fields: List[str] = [],
    name: str = "Selection",
    description: str | None = None,
) -> Type[BaseModel]:
    """
    Creates a Pydantic model for making a selection from a list of string options.

    The model will have a single field named `selection`. The type of this field
    will be `Literal[*fields]`, meaning its value must be one of the strings
    provided in the `fields` list.

    Args:
        name: The name for the created Pydantic model. Defaults to "Selection".
        description: An optional description for the model (becomes its docstring).
        fields: A list of strings representing the allowed choices for the selection.
                This list cannot be empty.

    Returns:
        A new Pydantic BaseModel class with a 'selection' field.

    Raises:
        ValueError: If the `fields` list is empty, as Literal requires at least one option.
    """
    if not fields:
        raise ValueError(
            "`fields` list cannot be empty for `create_selection_model` "
            "as it defines the possible selections for the Literal type."
        )

    # Create the Literal type from the list of field strings.
    # We can't use unpacking syntax directly with Literal, so we need to handle it differently
    if len(fields) == 1:
        selection_type = Literal[fields[0]]
    else:
        # For multiple fields, we need to use eval to create the Literal type
        # This is because Literal needs to be constructed with the actual string values
        # as separate arguments, not as a list
        literal_str = f"Literal[{', '.join(repr(f) for f in fields)}]"
        selection_type = eval(literal_str)

    # Define the field for the model. It's required (...).
    model_fields_definitions = {
        "selection": (
            selection_type,
            Field(
                ...,
                description="The selected value from the available options.",
            ),
        )
    }

    # Determine the docstring for the created model
    model_docstring = description
    if model_docstring is None:
        if fields:
            model_docstring = (
                f"A model for selecting one option from: {', '.join(fields)}."
            )
        else:  # Should not be reached due to the check above, but for completeness
            model_docstring = "A selection model."

    NewModel: Type[BaseModel] = create_model(
        name,
        __base__=BaseModel,
        __doc__=model_docstring,
        **model_fields_definitions,
    )
    return NewModel


def create_confirmation_pydantic_model(
    name: str = "Confirmation",
    description: str | None = None,
    field_name: str = "choice",
) -> Type[BaseModel]:
    """
    Creates a Pydantic model for boolean confirmation/response.

    The model will have a single field named `confirmed`. The type of this field
    will be `bool`, meaning its value must be either True or False.

    Args:
        name: The name for the created Pydantic model. Defaults to "Confirmation".
        description: An optional description for the model (becomes its docstring).

    Returns:
        A new Pydantic BaseModel class with a 'confirmed' field.
    """
    # Define the field for the model. It's required (...).
    model_fields_definitions = {
        field_name: (
            bool,
            Field(..., description="The boolean confirmation value."),
        )
    }

    # Determine the docstring for the created model
    model_docstring = description
    if model_docstring is None:
        model_docstring = "A model for boolean confirmation."

    NewModel: Type[BaseModel] = create_model(
        name,
        __base__=BaseModel,
        __doc__=model_docstring,
        **model_fields_definitions,
    )
    return NewModel