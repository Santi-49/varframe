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

import hashlib
import inspect
import json

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
    def get_hash_components(cls) -> Dict[str, str]:
        """
        Get the hash components of the variable definition.

        Returns:
            Dict with keys: calc, deps, attrs, meta.
        """
        # Calc: raw_column
        calc_hash = hashlib.sha256(cls.raw_column.encode("utf-8")).hexdigest()
        
        # Deps: None for BaseVariable
        deps_hash = hashlib.sha256(b"").hexdigest()
        
        # Attrs: dtype
        attrs_hash = hashlib.sha256(cls.dtype.encode("utf-8")).hexdigest()
        
        # Meta: description
        meta_hash = hashlib.sha256(cls.get_description().encode("utf-8")).hexdigest()

        return {
            "calc": calc_hash,
            "deps": deps_hash,
            "attrs": attrs_hash,
            "meta": meta_hash,
        }

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
        lazy (bool): If True, computed on access and not stored in DataFrame. Defaults to False.

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
    lazy: ClassVar[bool] = False

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
            # If dependency is in columns, we are good
            if dep.name in df.columns:
                continue

            # If dependency is Lazy, we assume it can be computed on-demand by __getitem__
            # so we don't raise KeyError here.
            is_lazy_dep = getattr(dep, "lazy", False)
            if is_lazy_dep:
                continue

            raise KeyError(
                f"Dependency '{dep.name}' not found in DataFrame. "
                f"Ensure dependencies are computed before '{cls.name}'."
            )
        return cls.calculate(df)

    @classmethod
    def get_hash_components(cls) -> Dict[str, str]:
        """
        Get the hash components of the variable definition.

        Returns:
            Dict with keys: calc, deps, attrs, meta.
        """
        # Calc: Source code of calculate method
        try:
            source = inspect.getsource(cls.calculate)
        except (OSError, TypeError):
            source = ""
        calc_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        
        # Deps: Recursive hash of dependencies
        # We combine the full hashes of all dependencies
        dep_hashes = []
        for dep in cls.dependencies:
            # Recursively get hash (we use a flattened version generally, 
            # but here we just need a signature of the dependency state).
            # To avoid infinite recursion in malformed cyclic graphs (though resolve handles that),
            # we rely on the fact that dependencies must be solved variables.
            # Ideally, we want the hash of the dependency variable itself.
            
            # Note: A dependency change should ripple up. 
            # We use the dependency's class name + its own component hashes
            try:
                d_comps = dep.get_hash_components()
                # Encapsulate strictly functional components for dependency hash
                # We exclude 'meta' so that docstring changes don't invalidate downstream calculations
                functional_comps = {k: v for k, v in d_comps.items() if k != "meta"}
                d_combined = hashlib.sha256(json.dumps(functional_comps, sort_keys=True).encode("utf-8")).hexdigest()
                dep_hashes.append(f"{dep.__name__}:{d_combined}")
            except Exception:
                # If dependency is broken or not a Variable class
                dep_hashes.append(f"{dep}:{str(dep)}")
                
        deps_str = ",".join(sorted(dep_hashes))
        deps_hash = hashlib.sha256(deps_str.encode("utf-8")).hexdigest()
        
        # Attrs: dtype
        attrs_hash = hashlib.sha256(cls.dtype.encode("utf-8")).hexdigest()
        
        # Meta: description, lazy
        meta_str = f"{cls.get_description()}|{cls.lazy}"
        meta_hash = hashlib.sha256(meta_str.encode("utf-8")).hexdigest()

        return {
            "calc": calc_hash,
            "deps": deps_hash,
            "attrs": attrs_hash,
            "meta": meta_hash,
        }

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
            "lazy": cls.lazy,
            "description": cls.get_description(),
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"
