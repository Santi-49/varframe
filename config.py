"""
Configuration and warning management for VarFrame.
===================================================

This module provides global configuration for controlling implicit operations
and their associated warnings throughout the library.

Classes:
    - ImplicitOperation: Enum for categorizing implicit operations
    - VFConfig: Global configuration singleton for warnings and permissions

Example:
    >>> from varframe.config import VFConfig, ImplicitOperation
    >>>
    >>> # Disable all warnings
    >>> VFConfig.warnings_enabled = False
    >>>
    >>> # Block implicit model training
    >>> VFConfig.allow_implicit_train = False
    >>>
    >>> # Context manager for temporary suppression
    >>> with VFConfig.suppress_warnings():
    ...     vf.resolve(PredictedGapDelta)
"""

from __future__ import annotations

import warnings
from enum import Enum, auto
from typing import Any, ClassVar

__all__ = [
    "ImplicitOperation",
    "VFConfig",
]


# ------------------- Implicit Operation Types -------------------


class ImplicitOperation(Enum):
    """
    Enum for different implicit operations that can be warned or blocked.

    These represent operations that happen automatically in the library,
    which users may want to be notified about or prevent entirely.

    Attributes:
        ADD_VARIABLE_NO_COMPUTE: Adding a variable to registry without computing it.
        ADD_VARIABLE_COMPUTE: Computing a variable not explicitly in the registry.
        TRAIN_MODEL: Automatically training a model that hasn't been trained.
        INFER_MODEL: Performing inference with a model.
    """

    ADD_VARIABLE_NO_COMPUTE = auto()
    ADD_VARIABLE_COMPUTE = auto()
    TRAIN_MODEL = auto()
    INFER_MODEL = auto()


# ------------------- Configuration Class -------------------


class VFConfig:
    """
    Global configuration for VarFrame warnings and implicit operations.

    Controls whether implicit operations (like auto-training models or computing
    variables not in _variables) issue warnings or are blocked entirely.

    This is a static configuration class - all attributes are class-level.

    Class Attributes:
        warn_add_variable_no_compute (bool): Warn when adding variable without computing.
        warn_add_variable_compute (bool): Warn when computing variable not in _variables.
        warn_train_model (bool): Warn when auto-training a model.
        warn_infer_model (bool): Warn when inferring with a model implicitly.
        allow_implicit_train (bool): Allow implicit model training.
        allow_implicit_infer (bool): Allow implicit model inference.
        allow_implicit_compute (bool): Allow implicit variable computation.
        warnings_enabled (bool): Master switch for all warnings.

    Example:
        >>> # Disable all warnings
        >>> VFConfig.warnings_enabled = False
        >>>
        >>> # Block implicit model training (raises RuntimeError)
        >>> VFConfig.allow_implicit_train = False
        >>>
        >>> # Suppress specific warning type
        >>> VFConfig.warn_train_model = False
        >>>
        >>> # Context manager for temporary suppression
        >>> with VFConfig.suppress_warnings():
        ...     vf.resolve(PredictedGapDelta)
        >>>
        >>> # Reset to defaults
        >>> VFConfig.reset()
    """

    # Warning flags - control which operations emit warnings
    warn_add_variable_no_compute: ClassVar[bool] = True
    warn_add_variable_compute: ClassVar[bool] = True
    warn_train_model: ClassVar[bool] = True
    warn_infer_model: ClassVar[bool] = True

    # Permission flags - if False, raises error instead of warning
    allow_implicit_train: ClassVar[bool] = True
    allow_implicit_infer: ClassVar[bool] = True
    allow_implicit_compute: ClassVar[bool] = True

    # Master switches
    warnings_enabled: ClassVar[bool] = True

    # Internal state for context manager
    _suppressed: ClassVar[bool] = False

    @classmethod
    def warn(
        cls,
        operation: ImplicitOperation,
        message: str,
        stacklevel: int = 3,
    ) -> None:
        """
        Issue a warning for an implicit operation if configured.

        Args:
            operation: The type of implicit operation.
            message: The warning message.
            stacklevel: Stack level for the warning (default 3 for typical call depth).
        """
        if cls._suppressed or not cls.warnings_enabled:
            return

        should_warn = {
            ImplicitOperation.ADD_VARIABLE_NO_COMPUTE: cls.warn_add_variable_no_compute,
            ImplicitOperation.ADD_VARIABLE_COMPUTE: cls.warn_add_variable_compute,
            ImplicitOperation.TRAIN_MODEL: cls.warn_train_model,
            ImplicitOperation.INFER_MODEL: cls.warn_infer_model,
        }.get(operation, True)

        if should_warn:
            # Use orange/yellow color for visibility in terminal
            colored_msg = f"\033[93m⚠ {message}\033[0m"
            warnings.warn(colored_msg, UserWarning, stacklevel=stacklevel)

    @classmethod
    def check_permission(cls, operation: ImplicitOperation, context: str = "") -> None:
        """
        Check if an implicit operation is allowed.

        Args:
            operation: The type of implicit operation.
            context: Additional context for the error message.

        Raises:
            RuntimeError: If the operation is not allowed.
        """
        permission_map = {
            ImplicitOperation.TRAIN_MODEL: (
                cls.allow_implicit_train,
                "Implicit model training is disabled",
            ),
            ImplicitOperation.INFER_MODEL: (
                cls.allow_implicit_infer,
                "Implicit model inference is disabled",
            ),
            ImplicitOperation.ADD_VARIABLE_COMPUTE: (
                cls.allow_implicit_compute,
                "Implicit variable computation is disabled",
            ),
            ImplicitOperation.ADD_VARIABLE_NO_COMPUTE: (
                cls.allow_implicit_compute,
                "Implicit variable addition is disabled",
            ),
        }

        allowed, base_msg = permission_map.get(operation, (True, ""))
        if not allowed:
            full_msg = f"{base_msg}. {context}" if context else base_msg
            raise RuntimeError(full_msg)

    @classmethod
    def suppress_warnings(cls) -> "_WarningSuppressionContext":
        """
        Context manager for temporarily suppressing all VF warnings.

        Returns:
            A context manager that suppresses warnings while active.

        Example:
            >>> with VFConfig.suppress_warnings():
            ...     vf.resolve(PredictedGapDelta)  # No warnings issued
        """
        return _WarningSuppressionContext()

    @classmethod
    def reset(cls) -> None:
        """Reset all configuration to default values."""
        cls.warn_add_variable_no_compute = True
        cls.warn_add_variable_compute = True
        cls.warn_train_model = True
        cls.warn_infer_model = True
        cls.allow_implicit_train = True
        cls.allow_implicit_infer = True
        cls.allow_implicit_compute = True
        cls.warnings_enabled = True
        cls._suppressed = False


# ------------------- Context Manager -------------------


class _WarningSuppressionContext:
    """
    Context manager for temporarily suppressing VF warnings.

    Used internally by VFConfig.suppress_warnings().
    """

    def __enter__(self) -> "_WarningSuppressionContext":
        self._previous_state = VFConfig._suppressed
        VFConfig._suppressed = True
        return self

    def __exit__(self, *args: Any) -> None:
        VFConfig._suppressed = self._previous_state
