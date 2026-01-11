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
    print("Running Customer Lifetime Value (LTV) Demo...")
    
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
    print("\nExported pipeline results to 'output.csv'.")
