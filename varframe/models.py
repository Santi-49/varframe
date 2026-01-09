"""
Machine Learning Model Integration for VarFrame.
=================================================

This module provides classes for integrating machine learning models
into the varframe pipeline.

Classes:
    - BaseModel: Declarative class for defining ML models.
    - ModelVariable: A derived variable computed via model inference.
    - ModelRegistry: Central registry for managing trained models.

Example:
    >>> from varframe import BaseModel, ModelVariable
    >>> from sklearn.ensemble import RandomForestRegressor
    >>>
    >>> class GapPredictor(BaseModel):
    ...     name = "gap_predictor"
    ...     input_vars = [Lap, Gap, TireAge]
    ...     target_var = GapDelta
    ...     model_class = RandomForestRegressor
    ...     model_params = {"n_estimators": 100}
    >>>
    >>> class PredictedGap(ModelVariable):
    ...     name = "predicted_gap"
    ...     model_class = GapPredictor
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Type, Union

import pandas as pd

from varframe.config import ImplicitOperation, VFConfig
from varframe.types import VariableList, VariableType
from varframe.variables import BaseVariable, DerivedVariable

# Forward reference for VarFrame to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from varframe.dataframe import VarFrame

__all__ = [
    "BaseModel",
    "ModelVariable",
    "ModelRegistry",
    "ModelType",
    "ModelList",
]


# ------------------- Type Aliases -------------------

ModelType = Type["BaseModel"]
ModelList = List[ModelType]


# ------------------- Base Model -------------------


class BaseModel:
    """
    A declarative base class for defining ML models.

    Define subclasses with class attributes to specify input features,
    target variable, model type, and hyperparameters.

    Class Attributes:
        name (str): Unique identifier for the model.
        input_vars (List[VariableType]): Variables used as features (X).
        target_var (VariableType): Variable to predict (y).
        model_class (Type): The model class (e.g., RandomForestRegressor).
        model_params (Dict): Hyperparameters passed to model_class().
        model (Any): The trained model instance (set after training).
        is_trained (bool): Whether the model has been trained.

    Example:
        >>> class GapPredictor(BaseModel):
        ...     name = "gap_predictor"
        ...     input_vars = [Lap, Gap, TireAge]
        ...     target_var = GapDelta
        ...     model_class = RandomForestRegressor
        ...     model_params = {"n_estimators": 100}
    """

    name: ClassVar[str] = ""
    input_vars: ClassVar[List[VariableType]] = []
    target_var: ClassVar[Optional[VariableType]] = None
    model_class: ClassVar[Optional[Type[Any]]] = None
    model_params: ClassVar[Dict[str, Any]] = {}
    model: ClassVar[Any] = None
    is_trained: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize subclass with defaults."""
        super().__init_subclass__(**kwargs)
        if not cls.name and cls.__name__ != "BaseModel":
            cls.name = cls.__name__.lower()
        if "input_vars" not in cls.__dict__:
            cls.input_vars = []
        if "model_params" not in cls.__dict__:
            cls.model_params = {}
        cls.model = None
        cls.is_trained = False

    @classmethod
    def get_description(cls) -> str:
        """Get model description from class docstring."""
        if cls.__doc__:
            return cls.__doc__.strip().split("\n")[0]
        return ""

    @classmethod
    def get_input_names(cls) -> List[str]:
        """Get list of input variable names."""
        return [v.name for v in cls.input_vars]

    @classmethod
    def get_target_name(cls) -> Optional[str]:
        """Get target variable name."""
        return cls.target_var.name if cls.target_var else None

    @classmethod
    def prepare_X(cls, vf: Union["VarFrame", pd.DataFrame]) -> pd.DataFrame:
        """Extract input features from a DataFrame."""
        input_names = cls.get_input_names()
        missing = [n for n in input_names if n not in vf.columns]
        if missing:
            raise KeyError(f"Missing input variables: {missing}")

        X = vf[input_names]
        if hasattr(X, "to_ml"):
            X = X.to_ml()
        return X

    @classmethod
    def prepare_y(cls, vf: Union["VarFrame", pd.DataFrame]) -> pd.Series:
        """Extract target variable from a DataFrame."""
        if cls.target_var is None:
            raise ValueError(f"{cls.__name__} has no target_var defined")

        target_name = cls.get_target_name()
        if target_name not in vf.columns:
            raise KeyError(f"Missing target variable: {target_name}")

        return vf[target_name]

    @classmethod
    def train(
        cls,
        vf: Union["VarFrame", pd.DataFrame],
        **fit_kwargs: Any,
    ) -> None:
        """Train the model on the provided data."""
        if cls.model_class is None:
            raise ValueError(
                f"{cls.__name__} must define 'model_class'. "
                "Example: model_class = RandomForestRegressor"
            )

        cls.model = cls.model_class(**cls.model_params)
        X = cls.prepare_X(vf)
        y = cls.prepare_y(vf)
        cls.model.fit(X, y, **fit_kwargs)
        cls.is_trained = True

    @classmethod
    def train_with_validation(
        cls,
        train_vf: Union["VarFrame", pd.DataFrame],
        val_vf: Union["VarFrame", pd.DataFrame],
        **fit_kwargs: Any,
    ) -> None:
        """Train with a validation set (for XGBoost, LightGBM, etc.)."""
        if cls.model_class is None:
            raise ValueError(f"{cls.__name__} must define 'model_class'.")

        cls.model = cls.model_class(**cls.model_params)
        X_train = cls.prepare_X(train_vf)
        y_train = cls.prepare_y(train_vf)
        X_val = cls.prepare_X(val_vf)
        y_val = cls.prepare_y(val_vf)

        fit_params = fit_kwargs.copy()
        model_name = cls.model_class.__name__.lower()

        if any(name in model_name for name in ["xgb", "lgbm", "lightgbm", "catboost"]):
            fit_params.setdefault("eval_set", [(X_val, y_val)])

        cls.model.fit(X_train, y_train, **fit_params)
        cls.is_trained = True

    @classmethod
    def predict(
        cls,
        vf: Union["VarFrame", pd.DataFrame],
        add_to_df: bool = True,
        column_name: Optional[str] = None,
    ) -> pd.Series:
        """Generate predictions."""
        if not cls.is_trained or cls.model is None:
            raise ValueError(f"{cls.__name__} must be trained before calling predict()")

        X = cls.prepare_X(vf)
        predictions = cls.model.predict(X)

        col_name = column_name or f"{cls.name}_pred"
        result = pd.Series(predictions, index=vf.index, name=col_name)

        if add_to_df and hasattr(vf, "_variables"):
            vf[col_name] = result

        return result

    @classmethod
    def evaluate(
        cls,
        vf: Union["VarFrame", pd.DataFrame],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        if metrics is None:
            metrics = {
                "mse": mean_squared_error,
                "mae": mean_absolute_error,
                "r2": r2_score,
            }

        y_true = cls.prepare_y(vf)
        y_pred = cls.predict(vf, add_to_df=False)

        results = {}
        for name, metric_fn in metrics.items():
            mask = ~(y_true.isna() | pd.Series(y_pred).isna())
            results[name] = metric_fn(y_true[mask], y_pred[mask])

        return results

    @classmethod
    def info(cls) -> Dict[str, Any]:
        """Get model metadata as a dictionary."""
        return {
            "name": cls.name,
            "description": cls.get_description(),
            "input_vars": cls.get_input_names(),
            "target_var": cls.get_target_name(),
            "model_class": cls.model_class.__name__ if cls.model_class else None,
            "model_params": cls.model_params,
            "is_trained": cls.is_trained,
        }

    @classmethod
    def save(cls, path: str) -> None:
        """Save the trained model to disk."""
        import joblib

        if not cls.is_trained:
            raise ValueError(f"{cls.__name__} must be trained before saving")

        joblib.dump(
            {
                "model": cls.model,
                "input_vars": cls.get_input_names(),
                "target_var": cls.get_target_name(),
                "name": cls.name,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> None:
        """Load a trained model from disk."""
        import joblib

        data = joblib.load(path)
        cls.model = data["model"]
        cls.is_trained = True

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name} (trained={self.is_trained})>"


# ------------------- Model Variable -------------------


class ModelVariable(DerivedVariable):
    """
    A derived variable computed via model inference.

    Class Attributes:
        name (str): The name of the prediction variable.
        model_class (Type[BaseModel]): The model class to use for predictions.
        dependencies: Auto-populated from model's input_vars.

    Example:
        >>> class PredictedGap(ModelVariable):
        ...     name = "predicted_gap"
        ...     model_class = GapPredictor
    """

    model_class: ClassVar[Optional[Type[BaseModel]]] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize and validate ModelVariable subclass."""
        super().__init_subclass__(**kwargs)

        if cls.model_class is not None:
            cls.dependencies = list(cls.model_class.input_vars)

    @classmethod
    def calculate(
        cls,
        df: pd.DataFrame,
        suppress_warnings: bool = False,
    ) -> pd.Series:
        """
        Calculate predictions using the associated model.

        If the model is not trained, it will be automatically trained
        using the available data in the DataFrame (with a warning).
        """
        if cls.model_class is None:
            raise ValueError(f"{cls.__name__} must define 'model_class' attribute")

        model_name = cls.model_class.name

        # Auto-train if model is not trained
        if not cls.model_class.is_trained:
            target_name = (
                cls.model_class.target_var.name
                if cls.model_class.target_var
                else "unknown"
            )

            if cls.model_class.target_var is None:
                raise ValueError(
                    f"Cannot auto-train model '{model_name}': no target_var defined"
                )

            if target_name not in df.columns:
                raise ValueError(
                    f"Cannot auto-train model '{model_name}': "
                    f"target variable '{target_name}' not in DataFrame."
                )

            VFConfig.check_permission(
                ImplicitOperation.TRAIN_MODEL,
                f"Cannot auto-train model '{model_name}'. "
                "Set VFConfig.allow_implicit_train = True to enable.",
            )

            if not suppress_warnings:
                VFConfig.warn(
                    ImplicitOperation.TRAIN_MODEL,
                    f"Auto-training model '{model_name}' (target: {target_name}).",
                    stacklevel=4,
                )

            train_data = df.dropna()
            cls.model_class.train(train_data)

        # Check permission and warn for inference
        VFConfig.check_permission(
            ImplicitOperation.INFER_MODEL,
            f"Cannot perform inference with model '{model_name}'.",
        )

        if not suppress_warnings:
            VFConfig.warn(
                ImplicitOperation.INFER_MODEL,
                f"Performing inference with model '{model_name}'.",
                stacklevel=4,
            )

        return cls.model_class.predict(df, add_to_df=False)

    @classmethod
    def info(cls) -> Dict[str, Any]:
        """Get variable metadata including model info."""
        base_info = super().info()
        base_info["model"] = cls.model_class.name if cls.model_class else None
        base_info["type"] = "model_prediction"
        return base_info


# ------------------- Model Registry -------------------


class ModelRegistry:
    """
    Central registry for managing multiple models.

    Example:
        >>> registry = ModelRegistry()
        >>> registry.register(GapPredictor)
        >>> registry.train_all(training_vf)
        >>> registry.save_all("models/")
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._models: Dict[str, Type[BaseModel]] = {}

    def register(self, model_class: Type[BaseModel]) -> None:
        """Register a model class."""
        self._models[model_class.name] = model_class

    def get(self, name: str) -> Optional[Type[BaseModel]]:
        """Get a model class by name."""
        return self._models.get(name)

    def list_models(self) -> List[str]:
        """Get list of registered model names."""
        return list(self._models.keys())

    def train_all(
        self,
        vf: Union["VarFrame", pd.DataFrame],
        models: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, bool]:
        """Train all registered models."""
        models = models or {}
        results = {}

        for name, model_class in self._models.items():
            try:
                model_class.train(vf)
                results[name] = True
            except Exception as e:
                print(f"Failed to train {name}: {e}")
                results[name] = False

        return results

    def evaluate_all(
        self, vf: Union["VarFrame", pd.DataFrame]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models."""
        results = {}
        for name, model_class in self._models.items():
            if model_class.is_trained:
                results[name] = model_class.evaluate(vf)
        return results

    def save_all(self, directory: str) -> None:
        """Save all trained models to a directory."""
        import os

        os.makedirs(directory, exist_ok=True)

        for name, model_class in self._models.items():
            if model_class.is_trained:
                path = os.path.join(directory, f"{name}.joblib")
                model_class.save(path)

    def load_all(self, directory: str) -> None:
        """Load all models from a directory."""
        import os

        for name, model_class in self._models.items():
            path = os.path.join(directory, f"{name}.joblib")
            if os.path.exists(path):
                model_class.load(path)

    def describe(self) -> pd.DataFrame:
        """Generate a summary DataFrame of all registered models."""
        data = [model_class.info() for model_class in self._models.values()]
        return pd.DataFrame(data)
