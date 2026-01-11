"""
Type aliases and base definitions for VarFrame.
================================================

This module provides type aliases used throughout the library.

Type Aliases:
    - VariableType: Union type for any variable class
    - VariableList: List of variable classes
"""

from __future__ import annotations

from typing import List, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from varframe.variables import BaseVariable, DerivedVariable
    from varframe.models import ModelVariable

# Type representing any variable class
VariableType = Type["BaseVariable | DerivedVariable | ModelVariable"]

# List of variable classes
VariableList = List[VariableType]
