"""
DAG-based dependency resolution for VarFrame.
==============================================

This module provides functions for resolving variable dependencies using
directed acyclic graph (DAG) traversal and topological sorting.

Functions:
    - resolve_dependencies: Resolve all dependencies for target variables
    - _topological_sort: Internal Kahn's algorithm implementation
    - _get_all_deps: Get direct dependencies of a variable

Example:
    >>> from varframe.dependencies import resolve_dependencies
    >>> all_vars = resolve_dependencies([PredictedGapDelta])
    >>> # Returns: [Lap, Gap, TireAge, GapDelta, PredictedGapDelta]
"""

from __future__ import annotations

from typing import Dict, List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from varframe.types import VariableType, VariableList

__all__ = [
    "resolve_dependencies",
]


def resolve_dependencies(
    target_variables: "VariableList",
    include_model_training_deps: bool = True,
) -> "VariableList":
    """
    Resolve all dependencies for the given variables using DAG traversal.

    Performs a topological sort to determine the correct computation order,
    ensuring all dependencies are computed before their dependents.

    Args:
        target_variables: List of variable classes to compute.
        include_model_training_deps: If True, includes dependencies needed
            for model training (target_var of models).

    Returns:
        A topologically sorted list of all variables needed to compute
        the target variables, with dependencies before dependents.

    Raises:
        ValueError: If a circular dependency is detected.

    Example:
        >>> # Only specify final variables - dependencies auto-resolved!
        >>> all_vars = resolve_dependencies([PredictedGapDelta])
        >>> # Returns: [Lap, Gap, TireAge, GapDelta, PredictedGapDelta]
    """
    # Collect all dependencies recursively
    all_vars: Set["VariableType"] = set()
    visiting: Set["VariableType"] = set()  # For cycle detection

    def collect(var: "VariableType") -> None:
        """Recursively collect all dependencies for a variable."""
        if var in all_vars:
            return
        if var in visiting:
            raise ValueError(f"Circular dependency detected involving '{var.name}'")

        visiting.add(var)

        # Get direct dependencies
        deps: List["VariableType"] = []

        # DerivedVariable dependencies
        if hasattr(var, "dependencies") and var.dependencies:
            deps.extend(var.dependencies)

        # ModelVariable: also needs model's input_vars for prediction
        # and target_var for training (if include_model_training_deps)
        if hasattr(var, "model_class") and var.model_class is not None:
            model_cls = var.model_class
            if hasattr(model_cls, "input_vars") and model_cls.input_vars:
                deps.extend(model_cls.input_vars)
            if include_model_training_deps:
                if hasattr(model_cls, "target_var") and model_cls.target_var:
                    deps.append(model_cls.target_var)

        # Recursively collect dependencies
        for dep in deps:
            collect(dep)

        visiting.remove(var)
        all_vars.add(var)

    # Collect from all target variables
    for var in target_variables:
        collect(var)

    # Topological sort using Kahn's algorithm
    return _topological_sort(all_vars)


def _topological_sort(variables: Set["VariableType"]) -> "VariableList":
    """
    Perform topological sort on variables using Kahn's algorithm.

    Args:
        variables: Set of variable classes to sort.

    Returns:
        List of variables in dependency order (dependencies first).
    """
    # Build dependency graph
    in_degree: Dict["VariableType", int] = {v: 0 for v in variables}
    dependents: Dict["VariableType", List["VariableType"]] = {v: [] for v in variables}

    for var in variables:
        deps = _get_all_deps(var)
        for dep in deps:
            if dep in variables:
                in_degree[var] += 1
                dependents[dep].append(var)

    # Start with variables that have no dependencies (in_degree == 0)
    queue = [v for v in variables if in_degree[v] == 0]
    sorted_vars: "VariableList" = []

    while queue:
        # Sort by name for deterministic ordering
        queue.sort(key=lambda v: v.name)
        var = queue.pop(0)
        sorted_vars.append(var)

        for dependent in dependents[var]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(sorted_vars) != len(variables):
        raise ValueError("Circular dependency detected in variables")

    return sorted_vars


def _get_all_deps(var: "VariableType") -> List["VariableType"]:
    """
    Get all direct dependencies of a variable (including model deps).

    Args:
        var: The variable class to get dependencies for.

    Returns:
        List of direct dependency variable classes.
    """
    deps: List["VariableType"] = []

    if hasattr(var, "dependencies") and var.dependencies:
        deps.extend(var.dependencies)

    if hasattr(var, "model_class") and var.model_class is not None:
        model_cls = var.model_class
        if hasattr(model_cls, "input_vars") and model_cls.input_vars:
            deps.extend(model_cls.input_vars)
        if hasattr(model_cls, "target_var") and model_cls.target_var:
            deps.append(model_cls.target_var)

    return deps


def explain_dependencies(
    target_variables: "VariableList",
    include_model_training_deps: bool = True,
) -> None:
    """
    Print a color-coded dependency list for the given variables.

    Displays the variables in computation order (dependencies first),
    with color coding for different variable types:
    - BaseVariable: Green (Input)
    - DerivedVariable: Yellow (Computed)
    - ModelVariable (or has model_class): Purple (Model Prediction)

    Args:
        target_variables: List of variable classes to explain.
        include_model_training_deps: Whether to include model training dependencies.
    """
    from varframe.variables import BaseVariable

    # ANSI color codes
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    PURPLE = "\033[95m"
    RESET = "\033[0m"

    resolved = resolve_dependencies(
        target_variables, include_model_training_deps=include_model_training_deps
    )

    names = ", ".join(v.name for v in target_variables)
    print(f"To compute {names}, we need:")

    for i, v in enumerate(resolved, 1):
        # Determine variable type and color
        if hasattr(v, "model_class") and v.model_class:
            var_type = "model_prediction"
            color = PURPLE
        elif issubclass(v, BaseVariable):
            var_type = "base"
            color = GREEN
        else:
            var_type = "derived"
            color = YELLOW

        print(f"  {i}. {color}{v.name}{RESET} ({var_type})")
    print()
