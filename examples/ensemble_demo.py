"""
Ensemble Model Demo
====================

This example demonstrates VarFrame's DAG-based dependency resolution
with ensemble models. It showcases:

1. Defining BaseVariables (mapping raw columns)
2. Defining DerivedVariables (computed columns)
3. Defining ML models declaratively (BaseModel)
4. Using model predictions as variables (ModelVariable)
5. Building ensemble models that use other model predictions as inputs
6. Automatic dependency resolution via DAG traversal

Requirements:
    pip install varframe[ml]

Run this example (from project root):
    pip install -e ".[ml]"
    python examples/ensemble_demo.py
"""

import pandas as pd

try:
    from varframe import (
        BaseModel,
        BaseVariable,
        DerivedVariable,
        ModelVariable,
        VarFrame,
        resolve_dependencies,
        explain_dependencies,
    )
except ImportError:
    print("Error: varframe package not found. Please install it first:")
    print("    pip install -e .")
    exit(1)


# ==========================================
# 1. Define Base Variables
# ==========================================


class Lap(BaseVariable):
    """Current lap number."""

    name = "lap"
    raw_column = "lap_num"
    dtype = "int"


class Gap(BaseVariable):
    """Time gap to leader."""

    name = "gap"
    raw_column = "gap"


class TireAge(BaseVariable):
    """Tire age in laps."""

    name = "tire_age"
    raw_column = "tire_age"
    dtype = "int"


# ==========================================
# 2. Define Derived Variables
# ==========================================


class GapDelta(DerivedVariable):
    """Actual gap change (target for prediction)."""

    name = "gap_delta"
    dependencies = [Gap]

    @classmethod
    def calculate(cls, df: pd.DataFrame) -> pd.Series:
        return df["gap"] - df["gap"].shift(1)


# ==========================================
# 3. Define Models (DECLARATIVE - model defined in class!)
# ==========================================

# Import sklearn models (requires varframe[ml])
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Install with: pip install varframe[ml]")


if SKLEARN_AVAILABLE:

    class GapDeltaPredictorRF(BaseModel):
        """Predicts gap delta using Random Forest."""

        name = "gap_delta_predictor_rf"
        input_vars = [Lap, Gap, TireAge]
        target_var = GapDelta
        model_class = RandomForestRegressor
        model_params = {"n_estimators": 50, "max_depth": 5, "random_state": 42}

    class GapDeltaLinear(BaseModel):
        """Predicts gap delta using Linear Regression."""

        name = "gap_delta_linear"
        input_vars = [Lap, Gap, TireAge]
        target_var = GapDelta
        model_class = LinearRegression
        model_params = {}

    class GapDeltaRidge(BaseModel):
        """Predicts gap delta using Ridge Regression."""

        name = "gap_delta_ridge"
        input_vars = [Lap, Gap, TireAge]
        target_var = GapDelta
        model_class = Ridge
        model_params = {"alpha": 1.0}

    # ==========================================
    # 4. Define Model Variables (predictions as variables)
    # ==========================================

    class PredictedGapDeltaRF(ModelVariable):
        """Random Forest prediction for gap delta."""

        name = "predicted_gap_delta_rf"
        model_class = GapDeltaPredictorRF

    class PredictedGapDeltaLinear(ModelVariable):
        """Linear model prediction for gap delta."""

        name = "predicted_gap_delta_linear"
        model_class = GapDeltaLinear

    class PredictedGapDeltaRidge(ModelVariable):
        """Ridge model prediction for gap delta."""

        name = "predicted_gap_delta_ridge"
        model_class = GapDeltaRidge

    # ==========================================
    # 5. Define Ensemble Model (uses predictions as input!)
    # ==========================================

    class GapDeltaEnsemble(BaseModel):
        """Ensemble model combining RF, Linear, and Ridge predictions."""

        name = "gap_delta_ensemble"
        input_vars = [
            PredictedGapDeltaRF,
            PredictedGapDeltaLinear,
            PredictedGapDeltaRidge,
        ]
        target_var = GapDelta
        model_class = Ridge
        model_params = {"alpha": 0.1}

    class PredictedGapDeltaEnsemble(ModelVariable):
        """Ensemble prediction combining all base models."""

        name = "predicted_gap_delta_ensemble"
        model_class = GapDeltaEnsemble


def main():
    """Run the ensemble demo."""
    if not SKLEARN_AVAILABLE:
        print("Cannot run demo without scikit-learn.")
        return

    # ==========================================
    # 6. Raw Data
    # ==========================================

    df_raw = pd.DataFrame(
        {
            "lap_num": [1, 2, 3, 4, 5, 6, 7, 8],
            "gap": [1.2, 1.1, 1.05, 0.9, 0.85, 0.82, 0.78, 0.75],
            "tire_age": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    print("=" * 60)
    print("AUTOMATIC DAG-BASED DEPENDENCY RESOLUTION DEMO")
    print("=" * 60)
    print()

    # ==========================================
    # CASE 1: Direct VarFrame instantiation
    # Just specify the final variable you need!
    # ==========================================

    print("=== CASE 1: VarFrame(df_raw, [FinalVar]) ===")
    print("Request: VarFrame(df_raw, [PredictedGapDeltaRF])")
    print()

    # Show what dependencies are resolved
    resolved = resolve_dependencies([PredictedGapDeltaRF])
    print(f"Resolved variables: {[v.name for v in resolved]}")
    print()

    # This AUTOMATICALLY:
    # 1. Computes all BaseVariables (Lap, Gap, TireAge)
    # 2. Computes GapDelta (dependency)
    # 3. Auto-trains the model (if not trained) with WARNING
    # 4. Makes prediction
    vf = VarFrame(df_raw, [PredictedGapDeltaRF])

    print("Result DataFrame:")
    print(vf)
    print()

    # ==========================================
    # CASE 2: Start with EMPTY vf, resolve EVERYTHING via df_raw
    # ==========================================

    print("=== CASE 2: Start empty, resolve ALL via stored df_raw ===")
    print("vf = VarFrame(df_raw, compute=False)")
    print("vf.resolve(PredictedGapDeltaEnsemble)")
    print()

    # Start with EMPTY vf - just store df_raw, compute nothing yet!
    vf2 = VarFrame(df_raw, compute=False)
    print(f"Empty vf2 - has df_raw stored: {vf2.df_raw is not None}")
    print(f"Initial columns: {list(vf2.columns)}")
    print()

    # Now resolve the ENSEMBLE prediction - starting from NOTHING!
    # This will compute ALL BaseVariables from stored df_raw, then DerivedVariables
    print("Calling: vf2.resolve(PredictedGapDeltaEnsemble)")
    print("This resolves EVERYTHING - even BaseVariables from stored df_raw!")
    print()

    vf2.resolve(PredictedGapDeltaEnsemble)

    print()
    print("Final vf2 columns:", list(vf2.columns))
    print()
    print("Result DataFrame:")
    print(vf2)
    print()

    # ==========================================
    # CASE 3: Show dependency resolution for ensemble
    # ==========================================

    print("Final vf2 columns:", list(vf2.columns))
    print()
    print("Result DataFrame:")
    print(vf2)
    print()

    # ==========================================
    # CASE 3: Show dependency resolution for ensemble
    # ==========================================

    print("=== CASE 3: Show Ensemble DAG Resolution ===")
    explain_dependencies([PredictedGapDeltaEnsemble])

    # ==========================================
    # CASE 4: Show State-Aware Calculation Plan
    # ==========================================

    print("=== CASE 4: State-Aware Calculation Plan ===")
    print("Start with new empty VarFrame (vf3) and ask for plan:")

    # Reset model training state to trigger warnings (since they were trained in previous cases)
    GapDeltaPredictorRF.is_trained = False
    GapDeltaLinear.is_trained = False
    GapDeltaRidge.is_trained = False
    GapDeltaEnsemble.is_trained = False

    vf3 = VarFrame(df_raw, variables=[Gap, GapDelta])
    vf3.add_variable(TireAge, compute=False)
    vf3.explain_calculation([PredictedGapDeltaEnsemble], legend=True)

    # ==========================================
    # CASE 5: Partial Partial State (Linear Model Ready)
    # ==========================================

    print("=== CASE 5: Partial State (Linear Model Ready) ===")
    print("Pre-calculating PredictedGapDeltaLinear, then asking for Ensemble plan:\n")

    # Reset all models first to simulate fresh start
    GapDeltaPredictorRF.is_trained = False
    GapDeltaLinear.is_trained = False
    GapDeltaRidge.is_trained = False
    GapDeltaEnsemble.is_trained = False

    vf4 = VarFrame(df_raw, compute=False)
    
    # Pre-calculate one dependency (Linear Model)
    # This acts as an explicit compute, so it will train the Linear model
    vf4.resolve(PredictedGapDeltaLinear, suppress_warnings=True)

    # Now explain the full ensemble
    # Linear should be Ready ✅
    # Others should be Needs Calculation ⏳
    vf4.explain_calculation([PredictedGapDeltaEnsemble], legend=True)


if __name__ == "__main__":
    main()
