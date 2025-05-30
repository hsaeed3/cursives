"""
✼ cursives.text
"""

import sys
from importlib import import_module
from typing import Any, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .markdown import (
        Markdown,
        MarkdownSection,
        MarkdownSettings,
        convert_to_markdown,
        convert_type_to_markdown,
        convert_function_to_markdown,
        convert_dataclass_to_markdown,
        convert_pydantic_model_to_markdown,
        convert_object_to_markdown,
    )
    from .instructions import (
        create_task_instructions,
        create_output_instructions,
    )


IMPORT_MAP: Dict[str, Tuple[str, str]] = {
    # ----------------------------
    # Markdown
    # ----------------------------
    "Markdown": (".markdown", "Markdown"),
    "MarkdownSection": (".markdown", "MarkdownSection"),
    "MarkdownSettings": (".markdown", "MarkdownSettings"),
    "convert_to_markdown": (".markdown", "convert_to_markdown"),
    "convert_type_to_markdown": (".markdown", "convert_type_to_markdown"),
    "convert_function_to_markdown": (".markdown", "convert_function_to_markdown"),
    "convert_dataclass_to_markdown": (".markdown", "convert_dataclass_to_markdown"),
    "convert_pydantic_model_to_markdown": (
        ".markdown",
        "convert_pydantic_model_to_markdown",
    ),
    "convert_object_to_markdown": (".markdown", "convert_object_to_markdown"),
    # ----------------------------
    # Instructions
    # ----------------------------
    "create_task_instructions": (".instructions", "create_task_instructions"),
    "create_output_instructions": (".instructions", "create_output_instructions"),
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
    "Markdown",
    "MarkdownSection",
    "MarkdownSettings",
    "convert_to_markdown",
    "convert_type_to_markdown",
    "convert_function_to_markdown",
    "convert_dataclass_to_markdown",
    "convert_pydantic_model_to_markdown",
    "convert_object_to_markdown",
    "create_task_instructions",
    "create_output_instructions",
]
