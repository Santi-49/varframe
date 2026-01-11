import pandas as pd
from varframe import VarFrame, BaseVariable, DerivedVariable

# 1. Setup Data
df_raw = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})

# 2. Define Variables
class VarA(BaseVariable):
    raw_column = "a"

class VarB(BaseVariable):
    raw_column = "b"

class EagerSum(DerivedVariable):
    dependencies = [VarA, VarB]
    @classmethod
    def calculate(cls, df):
        print("Calculating EagerSum...")
        return df[VarA.name] + df[VarB.name]

class LazySum(DerivedVariable):
    dependencies = [VarA, VarB]
    lazy = True  # <--- THE NEW FEATURE
    
    @classmethod
    def calculate(cls, df):
        print("Calculating LazySum (Just in Time)...")
        return df[VarA.name] + df[VarB.name]

class LazyChain(DerivedVariable):
    dependencies = [LazySum]
    lazy = True

    @classmethod
    def calculate(cls, df):
        print("Calculating LazyChain (Depends on LazySum)...")
        return df[LazySum.name] * 2

# 3. Initialize VarFrame
print("\n--- Initializing VarFrame ---")
vf = VarFrame(df_raw, [VarA, VarB, EagerSum, LazySum, LazyChain])

print(f"\nColumns present: {vf.columns.tolist()}")


# 5. Access Lazy Variable
print("\n--- Accessing Lazy Variable ---")
result = vf[LazySum]
print(f"Result:\n{result}")

# 6. Verify Transient Nature
print("\n--- Verifying Transient Nature ---")
if "lazysum" in vf.columns:
    print("FAIL: LazySum should NOT be stored after access!")
else:
    print("SUCCESS: LazySum remained transient (not stored).")

# 7. Verify String Access
print("\n--- Accessing by String ---")
try:
    result_str = vf["lazysum"]
    print(f"Result via string:\n{result_str}")
except Exception as e:
    import traceback
    traceback.print_exc()

# 8. Verify Chained Lazy Variables
print("\n--- Accessing Chained Lazy Variable ---")
try:
    result_chain = vf[LazyChain]
    print(f"Result Chain:\n{result_chain}")
except Exception as e:
    import traceback
    traceback.print_exc()

# 9. Access LazyChain (no LazySum)
print("\n--- Accessing LazyChain (no LazySum) ---")
print(vf.view(variables=[VarA, VarB, EagerSum, LazyChain]))

