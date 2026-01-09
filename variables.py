"""
Core variable classes for the VarFrame library.
================================================

This module provides the foundational variable classes for defining
columns in pandas DataFrames in a declarative, class-based style.

Classes:
    - BaseVariable: Maps raw DataFrame columns to processed variables.
    - DerivedVariable: Computed variables with dependencies.

Example:
    >>> from varframe import BaseVariable, DerivedVariable
    >>>
    >>> class Lap(BaseVariable):
    ...     name = "lap"
    ...     raw_column = "lap_num"
    ...     dtype = "int"
    >>>
    >>> class GapDelta(DerivedVariable):
    ...     name = "gap_delta"
    ...     dependencies = [Gap]
    ...
    ...     @classmethod
    ...     def calculate(cls, df):
    ...         return df["gap"] - df["gap"].shift(1)
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from varframe.types import VariableType

__all__ = [
    "BaseVariable",
    "DerivedVariable",
]


# ------------------- Base Variable -------------------


class BaseVariable:
    """
    A declarative variable that maps to a column in a raw DataFrame.

    Define subclasses with class attributes to create variable definitions.
    The class itself represents the variable - no instantiation needed.

    Class Attributes:
        **name (str)**: The name of the variable in the processed DataFrame.
        **raw_column (str)**: The column name in the raw DataFrame to extract.
        **dtype (str)**: The pandas dtype to cast the column to. Defaults to 'float'.

    Note:
        Use the class docstring as the variable description.

    Example:
        >>> class LapNumber(BaseVariable):
        ...     '''The current lap number in the race.'''
        ...     name = "lap"
        ...     raw_column = "lap_num"
        ...     dtype = "int"
        ...
        >>> # Use the class directly, not an instance
        >>> series = LapNumber.compute(raw_dataframe)
    """

    # Class attributes to be overridden by subclasses
    name: ClassVar[str] = ""
    raw_column: ClassVar[str] = ""
    dtype: ClassVar[str] = "float"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate subclass definition on class creation."""
        super().__init_subclass__(**kwargs)
        # Auto-generate name from class name if not specified
        if not cls.name and cls.__name__ not in ("BaseVariable", "DerivedVariable"):
            cls.name = cls.__name__.lower()

    @classmethod
    def get_description(cls) -> str:
        """
        Get the variable description from the class docstring.

        Returns:
            The first line of the class docstring, or empty string if none.
        """
        if cls.__doc__:
            return cls.__doc__.strip().split("\n")[0]
        return ""

    @classmethod
    def compute(cls, df_raw: pd.DataFrame) -> pd.Series:
        """
        Extract and transform the column from a raw DataFrame.

        Args:
            df_raw: The raw DataFrame containing the source column.

        Returns:
            A pandas Series with the extracted column cast to the specified dtype.

        Raises:
            KeyError: If `raw_column` does not exist in `df_raw`.
            ValueError: If the column cannot be cast to the specified `dtype`.
        """
        return df_raw[cls.raw_column].astype(cls.dtype)

    @classmethod
    def info(cls) -> Dict[str, Any]:
        """
        Get variable metadata as a dictionary.

        Returns:
            Dict with name, type, raw_column, dtype, and description.
        """
        return {
            "name": cls.name,
            "type": "base",
            "raw_column": cls.raw_column,
            "dtype": cls.dtype,
            "description": cls.get_description(),
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"


# ------------------- Derived Variable -------------------


class DerivedVariable:
    """
    A declarative variable computed from other variables.

    Define subclasses with class attributes and a `calculate()` classmethod
    to create computed variable definitions. Dependencies are resolved
    automatically using the DataFrame as the single source of truth.

    Class Attributes:
        name (str): The name of the variable in the processed DataFrame.
        dependencies (List[Type]): List of variable classes this depends on.
        dtype (str): The pandas dtype for the result. Defaults to 'float'.

    Note:
        - Use the class docstring as the variable description.
        - Override the `calculate()` classmethod to define computation logic.
        - Access dependencies directly from df (e.g., `df["gap"]`), not memo.

    Example:
        >>> class GapDelta(DerivedVariable):
        ...     '''Change in gap from previous measurement.'''
        ...     name = "gap_delta"
        ...     dependencies = [Gap]
        ...
        ...     @classmethod
        ...     def calculate(cls, df: pd.DataFrame) -> pd.Series:
        ...         return df["gap"] - df["gap"].shift(1)
    """

    # Class attributes to be overridden by subclasses
    name: ClassVar[str] = ""
    dependencies: ClassVar[List["VariableType"]] = []
    dtype: ClassVar[str] = "float"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate subclass definition and set defaults."""
        super().__init_subclass__(**kwargs)
        # Auto-generate name from class name if not specified
        if not cls.name and cls.__name__ not in ("BaseVariable", "DerivedVariable"):
            cls.name = cls.__name__.lower()
        # Ensure dependencies is a new list (not shared reference)
        if "dependencies" not in cls.__dict__:
            cls.dependencies = []

    @classmethod
    def get_description(cls) -> str:
        """
        Get the variable description from the class docstring.

        Returns:
            The first line of the class docstring, or empty string if none.
        """
        if cls.__doc__:
            return cls.__doc__.strip().split("\n")[0]
        return ""

    @classmethod
    def calculate(cls, df: pd.DataFrame) -> pd.Series:
        """
        Calculate the derived variable's values.

        Override this method in subclasses to define the computation logic.
        Access dependency values directly from df (e.g., `df["gap"]`).

        Args:
            df: The current processed DataFrame containing all computed
                variables so far (including dependencies).

        Returns:
            A pandas Series with the computed values.

        Raises:
            NotImplementedError: If not overridden in subclass.
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement the 'calculate' classmethod"
        )

    @classmethod
    def compute(cls, df: pd.DataFrame) -> pd.Series:
        """
        Compute the derived variable from the DataFrame.

        Dependencies must already exist in df before calling this method.
        Use `VarFrame` to handle dependency ordering automatically.

        Args:
            df: The DataFrame containing all dependency columns.

        Returns:
            A pandas Series containing the computed values.

        Raises:
            KeyError: If a dependency column is missing from df.
        """
        # Verify dependencies exist in df
        for dep in cls.dependencies:
            if dep.name not in df.columns:
                raise KeyError(
                    f"Dependency '{dep.name}' not found in DataFrame. "
                    f"Ensure dependencies are computed before '{cls.name}'."
                )
        return cls.calculate(df)

    @classmethod
    def info(cls) -> Dict[str, Any]:
        """
        Get variable metadata as a dictionary.

        Returns:
            Dict with name, type, dependencies, dtype, and description.
        """
        return {
            "name": cls.name,
            "type": "derived",
            "dependencies": [d.name for d in cls.dependencies],
            "dtype": cls.dtype,
            "description": cls.get_description(),
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"
