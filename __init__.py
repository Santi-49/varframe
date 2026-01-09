"""
VarFrame - A DataFrame Variable Management Library
===================================================

A declarative, class-based framework for defining, computing, and managing
variables (columns) in pandas DataFrames, with optional ML model integration.

Modules:
    - **config**: Configuration and warning management (VFConfig, ImplicitOperation)
    - **types**: Type aliases and base definitions
    - **variables**: Core variable classes (BaseVariable, DerivedVariable)
    - **dependencies**: DAG-based dependency resolution
    - **dataframe**: VarFrame class
    - **models**: ML model integration (BaseModel, ModelVariable, ModelRegistry)

Quick Start:
    >>> from varframe import (
    ...     BaseVariable, DerivedVariable, VarFrame,
    ...     VFConfig, BaseModel, ModelVariable
    ... )
    >>>
    >>> class Lap(BaseVariable):
    ...     name = "lap"
    ...     raw_column = "lap_num"
    ...     dtype = "int"
    >>>
    >>> class Gap(BaseVariable):
    ...     name = "gap"
    ...     raw_column = "gap"
    >>>
    >>> class GapDelta(DerivedVariable):
    ...     name = "gap_delta"
    ...     dependencies = [Gap]
    ...
    ...     @classmethod
    ...     def calculate(cls, df):
    ...         return df["gap"] - df["gap"].shift(1)
    >>>
    >>> vf = VarFrame(df_raw, [Lap, Gap, GapDelta])

Author: Santiago Romagosa
Version: 4.0.0
"""

from varframe.config import (
    ImplicitOperation,
    VFConfig,
)
from varframe.types import (
    VariableList,
    VariableType,
)
from varframe.variables import (
    BaseVariable,
    DerivedVariable,
)
from varframe.dependencies import (
    resolve_dependencies,
)
from varframe.dataframe import (
    VarFrame,
)
from varframe.models import (
    BaseModel,
    ModelVariable,
    ModelRegistry,
    ModelType,
    ModelList,
)

__all__ = [
    # Config
    "ImplicitOperation",
    "VFConfig",
    # Types
    "VariableType",
    "VariableList",
    # Variables
    "BaseVariable",
    "DerivedVariable",
    # Dependencies
    "resolve_dependencies",
    # DataFrame
    "VarFrame",
    # Models
    "BaseModel",
    "ModelVariable",
    "ModelRegistry",
    "ModelType",
    "ModelList",
]

__version__ = "4.0.0"
