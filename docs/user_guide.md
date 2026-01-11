# User Guide

This guide details the core components and workflows of VarFrame. We will build a complete pipeline to predict **Customer Lifetime Value (LTV)**.

You can follow along step-by-step. A complete, runnable script is provided at the bottom of this page.

## 1. Setup & Imports

First, we import the necessary components. VarFrame integrates seamlessly with pandas and scikit-learn.

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from varframe import VarFrame, BaseVariable, DerivedVariable, BaseModel, ModelVariable
```

## 2. Base Variables

**Base Variables** represent the columns in your raw input data. They act as the "contract" for your pipeline, ensuring the input data has the expected structure.

*   `raw_column`: The exact name of the column in your input DataFrame.
*   `dtype`: (Optional) The type to cast the data to.

```python
class AvgOrderValue(BaseVariable):
    raw_column = "avg_order_value"
    dtype = "float"

class PurchaseFrequency(BaseVariable):
    raw_column = "purchase_frequency"
    dtype = "float"
```

## 3. Derived Variables

**Derived Variables** are calculated from other variables. This is where you define your feature engineering logic.

*   `dependencies`: A list of other variable classes this variable needs. VarFrame uses this to determine execution order.
*   `calculate(cls, df)`: The method that performs the transformation.

```python
class AnnualRevenue(DerivedVariable):
    dependencies = [AvgOrderValue, PurchaseFrequency]

    @classmethod
    def calculate(cls, df):
        # We access data using .name to be safe and consistent
        return df[AvgOrderValue.name] * df[PurchaseFrequency.name]
```

## 4. Lazy Loading

Sometimes a variable is expensive to compute (e.g., API calls, complex simulations) and not always needed. You can mark these as **Lazy**.

*   `lazy = True`: The variable is **not** computed by default when you run `resolve()`.
*   It is computed **only** when you explicitly ask for it (e.g., `vf[CustomerScore]`).

```python
class CustomerScore(DerivedVariable):
    dependencies = [AnnualRevenue]
    lazy = True

    @classmethod
    def calculate(cls, df):
        print("Performing expensive calculation for Customer Score...")
        return df[AnnualRevenue.name] * 0.1
```

## 5. Machine Learning Models

VarFrame treats ML models as first-class citizens in the dependency graph.

### Defining the Model
`BaseModel` defines the schema of your model: what goes in, what comes out, and what algorithm to use.

```python
# The target variable we want to predict (for training)
class LifetimeValue(BaseVariable):
    name = "lifetime_value"
    raw_column = "lifetime_value"

class LTVPredictor(BaseModel):
    name = "ltv_predictor"
    input_vars = [AvgOrderValue, PurchaseFrequency, AnnualRevenue]
    target_var = LifetimeValue
    model_class = LinearRegression
```

### Getting Predictions
`ModelVariable` represents the *output* of the model.

```python
class PredictedLTV(ModelVariable):
    name = "predicted_ltv"
    model_class = LTVPredictor
```

**Auto-Training:** If `PredictedLTV` is requested but `LTVPredictor` hasn't been trained, VarFrame will automatically train the model using the current DataFrame (if `target_var` is present).

## 6. Execution

Now we initialize the `VarFrame` with some raw data and resolve our target variables.

```python
# 1. Initialize with raw data
data = pd.DataFrame({
    "avg_order_value": [50.0, 100.0, 20.0, 80.0],
    "purchase_frequency": [4, 2, 10, 1],
    "lifetime_value": [220.0, 210.0, 250.0, 90.0]
})

vf = VarFrame(data)

# 2. Resolve Dependencies
# We asks for PredictedLTV. VarFrame figures out the rest:
# AnnualRevenue -> Train LTVPredictor -> Predict LTV
vf.resolve(PredictedLTV)

print(vf[[AvgOrderValue, AnnualRevenue, PredictedLTV]])
```

## 7. Using Lazy Variables

Lazy variables remain uncomputed until accessed.

```python
print(f"Is score computed? {'customerscore' in vf.columns}")

# Accessing triggers the calculation
score = vf[CustomerScore]

print(f"Is score computed now? {'customerscore' in vf.columns}")
```

## 8. Import / Export

When exporting, you can force all lazy variables to be computed using `include=['all']`.

```python
# Export everything, including lazy variables
vf.to_csv("customer_ltv.csv", index=False, include=['all'])
```

---

## Full Runnable Code

Here is the complete script combining all the steps above.

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from varframe import VarFrame, BaseVariable, DerivedVariable, BaseModel, ModelVariable

# --- Variables ---
class AvgOrderValue(BaseVariable):
    raw_column = "avg_order_value"

class PurchaseFrequency(BaseVariable):
    raw_column = "purchase_frequency"

class AnnualRevenue(DerivedVariable):
    dependencies = [AvgOrderValue, PurchaseFrequency]
    @classmethod
    def calculate(cls, df):
        return df[AvgOrderValue.name] * df[PurchaseFrequency.name]

class CustomerScore(DerivedVariable):
    dependencies = [AnnualRevenue]
    lazy = True
    @classmethod
    def calculate(cls, df):
        return df[AnnualRevenue.name] * 0.1

# --- Model ---
class LifetimeValue(BaseVariable):
    name = "lifetime_value"
    raw_column = "lifetime_value"

class LTVPredictor(BaseModel):
    name = "ltv_predictor"
    input_vars = [AvgOrderValue, PurchaseFrequency, AnnualRevenue]
    target_var = LifetimeValue
    model_class = LinearRegression

class PredictedLTV(ModelVariable):
    name = "predicted_ltv"
    model_class = LTVPredictor

# --- Execution ---
if __name__ == "__main__":
    data = pd.DataFrame({
        "avg_order_value": [50.0, 100.0, 20.0, 80.0],
        "purchase_frequency": [4, 2, 10, 1],
        "lifetime_value": [220.0, 210.0, 250.0, 90.0]
    })
    
    vf = VarFrame(data)
    
    # 1. Resolve & Auto-Train
    vf.resolve(PredictedLTV)
    print("Predictions:\n", vf[PredictedLTV])
    
    # 2. Lazy Load
    print("\nCustomer Score (JIT):\n", vf[CustomerScore])
    
    # 3. Export
    vf.to_csv("output.csv", include=['all'])
    print("\nExported pipeline results.")
```
