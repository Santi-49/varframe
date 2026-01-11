"""
VarFrame - A pandas DataFrame subclass with variable metadata.
===============================================================

This module provides VarFrame, a pandas DataFrame subclass that
maintains a registry of variable class definitions for type-aware operations.

Classes:
    - VarFrame: Main DataFrame class with variable metadata.
    - _VarFrameInternal: Internal class for pandas operations.

Example:
    >>> from varframe import VarFrame, BaseVariable, DerivedVariable
    >>>
    >>> vf = VarFrame(df_raw, [Lap, Gap, GapDelta])
    >>> vf.head()              # Normal DataFrame operations work
    >>> vf.filter_by_type(DerivedVariable)  # Variable-aware filtering
    >>> vf[Gap]                # Access by variable class
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, Union

import pandas as pd

from varframe.config import ImplicitOperation, VFConfig
from varframe.dependencies import resolve_dependencies
from varframe.types import VariableList, VariableType
from varframe.variables import BaseVariable, DerivedVariable

__all__ = [
    "VarFrame",
]


class VarFrame(pd.DataFrame):
    """
    A pandas DataFrame subclass with variable metadata and helper methods.

    VarFrame behaves exactly like a pandas DataFrame, but also
    maintains a registry of variable class definitions. This enables:
    - Filtering columns by variable type (BaseVariable vs DerivedVariable)
    - Accessing variable metadata and descriptions
    - Indexing by variable class (e.g., ``vf[Gap]``)

    Attributes:
        _variables (List[Type]): List of variable classes (stored in _metadata).

    Example:
        >>> vf = VarFrame(df_raw, [Lap, Gap, GapDelta])
        >>> vf.head()              # Normal DataFrame operations work
        >>> vf.filter_by_type(DerivedVariable)  # Variable-aware filtering
        >>> vf[Gap]                # Access by variable class
    """

    # Tell pandas to preserve these attributes during operations
    _metadata = ["_variables", "_df_raw"]

    def __init__(
        self,
        df_raw: Optional[pd.DataFrame] = None,
        variables: Optional[VariableList] = None,
        compute: bool = True,
        auto_resolve: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a VarFrame from raw data.

        Args:
            df_raw: The raw DataFrame to process. Stored internally for
                future resolve() calls. If None, creates an empty DataFrame.
            variables: List of variable classes to compute. If None or empty,
                no variables are computed initially (use resolve() later).
            compute: If True (default), compute the specified variables immediately.
                If False, just store df_raw without computing anything.
            auto_resolve: If True (default) and compute=True, automatically
                resolve all dependencies. If False, variables must be in
                correct dependency order.
            **kwargs: Additional arguments passed to pd.DataFrame.__init__.

        Example:
            >>> # Compute specific variables
            >>> vf = VarFrame(df_raw, [Lap, Gap, GapDelta])
            >>>
            >>> # Auto-resolve: just specify final variables
            >>> vf = VarFrame(df_raw, [PredictedGapDelta])
            >>>
            >>> # Store df_raw, compute nothing (use resolve() later)
            >>> vf = VarFrame(df_raw, compute=False)
            >>> vf.resolve(PredictedGapDelta)
        """
        # Store raw DataFrame
        self._df_raw: Optional[pd.DataFrame] = df_raw
        self._variables: VariableList = []

        # Determine which variables to compute
        vars_to_compute: VariableList = []
        if variables and compute:
            if auto_resolve:
                vars_to_compute = resolve_dependencies(variables)
            else:
                vars_to_compute = list(variables)

            # Filter out lazy variables from initial computation
            vars_to_compute = [
                v
                for v in vars_to_compute
                if not (issubclass(v, DerivedVariable) and v.lazy)
            ]

        # Initialize DataFrame with proper index
        if df_raw is not None:
            super().__init__(index=df_raw.index, **kwargs)
        else:
            super().__init__(**kwargs)

        # Compute variables if requested
        if vars_to_compute and df_raw is not None:
            for var in vars_to_compute:
                if issubclass(var, BaseVariable):
                    self[var.name] = var.compute(df_raw)
                else:
                    self[var.name] = var.compute(self)
                
                if var not in self._variables:
                    self._variables.append(var)
            
            # Register remaining variables (lazy ones or non-computed) that were passed in
            if variables:
                 for var in variables:
                     if var not in self._variables:
                         self._variables.append(var)

        elif variables:
            # Store variable list even if not computing
            self._variables = list(variables) if variables else []
            # Ensure no duplicates if variables were passed
            self._variables = list(dict.fromkeys(self._variables))

    @property
    def _constructor(self) -> Type[VarFrame]:
        """Return the constructor for DataFrame operations to preserve type."""
        return _VarFrameInternal

    @property
    def variables(self) -> VariableList:
        """Get the list of variable classes."""
        return self._variables

    @variables.setter
    def variables(self, value: VariableList) -> None:
        """Set the list of variable classes."""
        self._variables = value

    @property
    def df_raw(self) -> Optional[pd.DataFrame]:
        """Get the raw DataFrame (if stored)."""
        return self._df_raw

    @df_raw.setter
    def df_raw(self, value: Optional[pd.DataFrame]) -> None:
        """Set the raw DataFrame."""
        self._df_raw = value

    def filter_by_type(
        self, variable_type: Type[Union[BaseVariable, DerivedVariable]]
    ) -> VarFrame:
        """
        Filter to include only columns of a specific variable type.

        Args:
            variable_type: The base class to filter by (BaseVariable or DerivedVariable).

        Returns:
            A new VarFrame containing only columns whose variables
            are subclasses of the specified type.

        Example:
            >>> derived_df = vf.filter_by_type(DerivedVariable)
            >>> base_df = vf.filter_by_type(BaseVariable)
        """
        filtered_vars = [v for v in self._variables if issubclass(v, variable_type)]
        filtered_names = [v.name for v in filtered_vars]
        result = self[filtered_names].copy()
        result._variables = filtered_vars
        return result

    def get_variable(self, name: str) -> Optional[VariableType]:
        """
        Retrieve a variable class by its name.

        Args:
            name: The name of the variable to retrieve.

        Returns:
            The variable class if found, None otherwise.
        """
        for var in self._variables:
            if var.name == name:
                return var
        return None

    def list_variables(self) -> List[str]:
        """
        Get a list of all variable names.

        Returns:
            List of variable names in order.
        """
        return [v.name for v in self._variables]

    def describe_variables(self) -> pd.DataFrame:
        """
        Generate a summary DataFrame describing all variables.

        Returns:
            A DataFrame with columns: name, type, dtype, description, etc.
        """
        data = [var.info() for var in self._variables]
        return pd.DataFrame(data)

    def view(
        self,
        include: Optional[List[str]] = None,
        variables: Optional[VariableList] = None,
    ) -> pd.DataFrame:
        """
        Create a DataFrame view containing only specific variables.

        Args:
            include: List of variable categories to include.
                Options: 'base', 'derived', 'lazy', 'model', 'all'.
                Defaults to ['all'] if both include and variables are None.
            variables: Explicit list of variable classes to include.

        Returns:
            A pandas DataFrame with the requested variables. 
            Lazy variables will be computed on-demand for this view.
        
        Example:
            >>> vf.view(include=['base', 'lazy'])
            >>> vf.view(variables=[LazySum])
        """
        target_vars = []

        # 1. Handle 'variables' argument
        if variables:
            target_vars.extend(variables)

        # 2. Handle 'include' argument
        if include:
            for category in include:
                if category == "all":
                    target_vars.extend(self._variables)
                elif category == "base":
                    target_vars.extend(
                        [v for v in self._variables if issubclass(v, BaseVariable)]
                    )
                elif category == "derived":
                    # Non-lazy derived
                    target_vars.extend(
                        [
                            v
                            for v in self._variables
                            if issubclass(v, DerivedVariable) and not v.lazy
                        ]
                    )
                elif category == "lazy":
                    # Lazy derived
                    target_vars.extend(
                        [
                            v
                            for v in self._variables
                            if issubclass(v, DerivedVariable) and v.lazy
                        ]
                    )
                elif category == "model":
                    target_vars.extend(
                        [
                            v
                            for v in self._variables
                            if hasattr(v, "model_class") and v.model_class
                        ]
                    )

        # Default to 'all' if nothing specified
        if not target_vars:
            target_vars = list(self._variables)

        # Remove duplicates while preserving order
        target_vars = list(dict.fromkeys(target_vars))

        # 3. Construct DataFrame
        data = {}
        for var in target_vars:
            # If standard column, use it (zero copy if possible)
            if var.name in self.columns:
                data[var.name] = self[var.name]
            else:
                # Must be lazy or missing -> Compute it
                data[var.name] = var.compute(self)

        return pd.DataFrame(data, index=self.index)

    def __getitem__(self, key: Union[str, VariableType, List, slice, pd.Series]) -> Any:
        """
        Access columns by name, variable class, or standard DataFrame indexing.

        Extends pandas indexing to support variable classes directly.

        Args:
            key: Column name (str), variable class, list of columns,
                 boolean Series, or slice.

        Returns:
            Series for single column, DataFrame for multiple columns.

        Example:
            >>> vf["gap"]      # By string name
            >>> vf[Gap]        # By variable class
            >>> vf[[Lap, Gap]] # Multiple variable classes
        """
        # Handle variable class
        if isinstance(key, type) and issubclass(key, (BaseVariable, DerivedVariable)):
            # Check if it's lazy and missing (Compute just-in-time)
            if (
                issubclass(key, DerivedVariable)
                and key.lazy
                and key.name not in self.columns
            ):
                 return key.compute(self)
            
            return super().__getitem__(key.name)

        # Handle list of variable classes
        if isinstance(key, list) and len(key) > 0:
            if isinstance(key[0], type) and issubclass(
                key[0], (BaseVariable, DerivedVariable)
            ):
                names = [k.name for k in key]
                return super().__getitem__(names)

        # Default pandas behavior
        try:
            return super().__getitem__(key)
        except KeyError:
            # Check for lazy variable (string access)
            if isinstance(key, str):
                var = self.get_variable(key)

                # Case 1: Variable is registered but not in columns (Lazy or missing)
                if var and issubclass(var, DerivedVariable) and var.lazy:
                    return var.compute(self)
                
            # Re-raise if not handled
            raise

    def __repr__(self) -> str:
        """Return a string representation."""
        var_info = f"Variables: {len(self._variables)} ({self.list_variables()})"
        df_repr = super().__repr__()
        return f"{var_info}\n{df_repr}"

    # ------------------- ML Compatibility Methods -------------------

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert to a plain pandas DataFrame.

        Use this before passing to ML libraries that may have issues
        with DataFrame subclasses (e.g., pickle, joblib, some sklearn pipelines).

        Returns:
            A plain pandas DataFrame with the same data (no variable metadata).

        Example:
            >>> plain_df = vf.to_pandas()
            >>> model.fit(plain_df, y)
        """
        return pd.DataFrame(self)

    def to_ml(self) -> pd.DataFrame:
        """
        Alias for to_pandas(). Explicit conversion for ML pipelines.

        Use this to clearly indicate the DataFrame is being prepared
        for machine learning operations.

        Returns:
            A plain pandas DataFrame suitable for ML libraries.

        Example:
            >>> X_train = vf.to_ml()
            >>> model.fit(X_train, y_train)
        """
        return pd.DataFrame(self)

    def add_variables(
        self,
        *variables: VariableType,
        compute: bool = True,
        suppress_warnings: bool = False,
    ) -> VarFrame:
        """
        Register and optionally compute new variables (in-place).

        Args:
            *variables: Variable classes to add.
            compute: If True (default), compute variables immediately.
                If False, just register them (must already exist in DataFrame).
            suppress_warnings: If True, suppress warnings for this call only.

        Returns:
            Self, for method chaining.

        Raises:
            KeyError: If compute=True and dependencies are missing.
            RuntimeError: If implicit usage is disabled in VFConfig.
            ValueError: If compute=True and adding a BaseVariable (must come from raw).
        """
        for var in variables:
            if compute:
                # --- Compute Mode ---
                if var.name in self.columns:
                    continue  # Already exists

                # Note: Explicit addition via add_variables does not trigger implicit warnings.

                if issubclass(var, BaseVariable):
                    raise ValueError(
                        f"Cannot add BaseVariable '{var.name}' to existing DataFrame. "
                        "BaseVariables can only be computed from raw data."
                    )
                else:
                    self[var.name] = var.compute(self)

            else:
                # --- No Compute Mode (Registration Only) ---
                # Explicit registration - no warning needed.
                pass

            # Common: Add to registry
            if var not in self._variables:
                self._variables.append(var)

        return self

    def add_variable(
        self,
        *variables: VariableType,
        compute: bool = True,
        suppress_warnings: bool = False,
    ) -> VarFrame:
        """
        Alias for add_variables.
        """
        return self.add_variables(
            *variables, compute=compute, suppress_warnings=suppress_warnings
        )

    def resolve(
        self,
        *target_variables: VariableType,
        suppress_warnings: bool = False,
    ) -> VarFrame:
        """
        Resolve and compute target variables with automatic dependency resolution.

        This method automatically determines all missing dependencies for the
        target variables, computes them in the correct DAG order, and adds
        them to the DataFrame.

        Args:
            *target_variables: Variable classes to resolve and compute.
            suppress_warnings: If True, suppress warnings for this call only.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If circular dependencies are detected.
            ValueError: If a BaseVariable is needed but df_raw is not available.
            RuntimeError: If implicit operations are disabled in VFConfig.

        Example:
            >>> vf = VarFrame(df_raw, [Lap, Gap])
            >>> vf.resolve(PredictedGapDelta)
        """
        # Resolve all dependencies
        all_vars = resolve_dependencies(list(target_variables))

        # Filter to only missing variables
        # Filter to only missing variables
        missing_vars = [v for v in all_vars if v.name not in self.columns]
        
        # Split into compute-now vs lazy
        lazy_vars = [
            v for v in missing_vars if issubclass(v, DerivedVariable) and v.lazy
        ]
        missing_vars = [v for v in missing_vars if v not in lazy_vars]

        # Check for missing BaseVariables - need df_raw to compute them
        missing_base = [v for v in missing_vars if issubclass(v, BaseVariable)]
        if missing_base and self._df_raw is None:
            names = [v.name for v in missing_base]
            raise ValueError(
                f"Cannot resolve BaseVariables {names}: no raw DataFrame stored. "
                "Pass df_raw to VarFrame() or set vf.df_raw to enable."
            )

        # Warn about implicit computation of missing variables
        if missing_vars and not suppress_warnings:
            var_names = [v.name for v in missing_vars]
            VFConfig.check_permission(
                ImplicitOperation.ADD_VARIABLE_COMPUTE,
                f"Cannot resolve variables {var_names}. "
                "Set VFConfig.allow_implicit_compute = True to enable.",
            )
            VFConfig.warn(
                ImplicitOperation.ADD_VARIABLE_COMPUTE,
                f"Implicitly computing {len(missing_vars)} variable(s) not in _variables: {var_names}",
                stacklevel=3,
            )

        # Register lazy variables (so get_variable works)
        for var in lazy_vars:
            if var not in self._variables:
                self._variables.append(var)

        if suppress_warnings:
            context = VFConfig.suppress_warnings()
        else:
            context = VFConfig.null_context()

        with context:
            # Compute missing variables in order
            for var in missing_vars:
                if issubclass(var, BaseVariable):
                    self[var.name] = var.compute(self._df_raw)
                else:
                    self[var.name] = var.compute(self)

                if var not in self._variables:
                    self._variables.append(var)

        return self

    def explain_calculation(
        self,
        target_variables: VariableList,
        legend: bool = False,
    ) -> None:
        """
        Explain the calculation plan for the given variables given the current DataFrame state.

        Prints a color-coded list indicating:
        - Variable Type (Base, Derived, Model)
        - Calculation Status (Ready vs Needs Calculation)
        - Warnings (Implicit computation, Auto-training, Inference)

        Args:
            target_variables: List of variable classes to explain.
            legend: If True, print a legend for the warning codes.
        """
        from varframe.variables import BaseVariable
        from varframe.config import VFConfig, ImplicitOperation

        # ANSI color codes
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        PURPLE = "\033[95m"
        ORANGE = "\033[33m"
        GREY = "\033[90m"
        RESET = "\033[0m"

        resolved = resolve_dependencies(target_variables)

        names = ", ".join(v.name for v in target_variables)
        print(f"Calculation Plan for {names}:")

        used_codes = set()
        # Mapping: Code -> (Description, ConfigCheck)
        warning_defs = {
            "I": ("Implicit Compute", VFConfig.warn_add_variable_compute),
            "T": ("Auto-Train Model", VFConfig.warn_train_model),
            "M": ("Model Inference", VFConfig.warn_infer_model),
        }

        for i, v in enumerate(resolved, 1):
            # 1. Determine Type
            if hasattr(v, "model_class") and v.model_class:
                var_type = "model_prediction"
                color = PURPLE
            elif issubclass(v, BaseVariable):
                var_type = "base"
                color = GREEN
            else:
                var_type = "derived"
                color = YELLOW

            # 2. Determine Status
            exists = v.name in self.columns
            status_icon = "✅" if exists else "⏳"
            
            # 3. Predict Warnings
            current_codes = []
            if not exists:
                # Implicit Variable Warning
                if v not in self._variables and warning_defs["I"][1]:
                     current_codes.append("I")

                # Model Warnings
                if var_type == "model_prediction" and v.model_class:
                    if not v.model_class.is_trained and warning_defs["T"][1]:
                         current_codes.append("T")
                    if warning_defs["M"][1]:
                        current_codes.append("M")
            
            used_codes.update(current_codes)
            
            # Formatting
            line = f"  {i}. {status_icon} {color}{v.name}{RESET} ({var_type})"
            if current_codes:
                codes_str = ",".join(current_codes)
                # Subtle grey for the warning hints
                line += f" {GREY}⚠ [{codes_str}]{RESET}"
            
            print(line)
        
        if legend and used_codes:
            print("\nLegend:")
            for code in sorted(used_codes):
                desc = warning_defs[code][0]
                print(f"  {GREY}⚠ [{code}]{RESET} : {desc}")
        print()

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        variables: Optional[VariableList] = None,
        df_raw: Optional[pd.DataFrame] = None,
    ) -> VarFrame:
        """
        Create a VarFrame from a plain pandas DataFrame.

        Use this to re-wrap a DataFrame after ML operations or when
        loading data that was previously converted with to_pandas().

        Args:
            df: The pandas DataFrame to convert (already processed data).
            variables: List of variable classes to associate with the DataFrame.
            df_raw: Optional raw DataFrame to store for future resolve() calls.

        Returns:
            A new VarFrame with the given data and variables.

        Example:
            >>> plain_df = pd.read_pickle("data.pkl")
            >>> vf = VarFrame.from_pandas(plain_df, variables=[Lap, Gap])
        """
        vf = cls(df_raw=None, variables=None, compute=False)
        for col in df.columns:
            vf[col] = df[col].values
        vf.index = df.index
        vf._variables = list(variables) if variables else []
        vf._df_raw = df_raw
        return vf


class _VarFrameInternal(VarFrame):
    """
    Internal subclass for pandas operations (slicing, head, etc.).

    This class has a simpler __init__ that doesn't try to compute variables,
    allowing pandas operations like .head(), .copy(), slicing to work correctly.
    """

    def __init__(
        self,
        data: Any = None,
        **kwargs: Any,
    ) -> None:
        # Bypass VarFrame.__init__ and go directly to DataFrame
        pd.DataFrame.__init__(self, data, **kwargs)
        # Initialize metadata with defaults
        if not hasattr(self, "_variables"):
            self._variables = []
        if not hasattr(self, "_df_raw"):
            self._df_raw = None
