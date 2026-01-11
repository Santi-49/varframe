import pandas as pd
import os
import shutil
import json
try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from varframe import VarFrame, BaseVariable, DerivedVariable

print("--- VarFrame Import/Export Demo ---")

# 1. Setup Data & Variables
df_raw = pd.DataFrame({"id": [1, 2, 3], "val": [10.0, 20.0, 30.0]})

class RawVal(BaseVariable):
    raw_column = "val"

class ComputedVal(DerivedVariable):
    dependencies = [RawVal]
    @classmethod
    def calculate(cls, df):
        return df[RawVal.name] * 2

class LazyVal(DerivedVariable):
    dependencies = [RawVal]
    lazy = True
    @classmethod
    def calculate(cls, df):
        # Quiet calculation for demo clarity
        return df[RawVal.name] * 10

# Initialize
vf = VarFrame(df_raw, [RawVal, ComputedVal, LazyVal], name="demo_vf")

print(f"Initialized VarFrame '{vf.name}' with columns: {vf.columns.tolist()}")

# ---------------------------------------------------------
# 2. CSV Export & Load (Auto-Discovery)
# ---------------------------------------------------------
print("\n--- CSV Export & Load (Auto-Discovery) ---")

# Export (computing lazy vars)
csv_path = "demo_full.csv"
print(f"Exporting to {csv_path} (including lazy variables)...")
vf.to_csv(csv_path, include=['all'])

# Load (Auto-Discovery)
print(f"Loading from {csv_path}...")
vf_loaded = VarFrame.load_csv(csv_path)

print(f"  -> Discovered Variables: {vf_loaded.list_variables()}")
if LazyVal in vf_loaded.variables:
    print("  -> SUCCESS: LazyVal correctly linked via auto-discovery!")
else:
    print("  -> FAIL: LazyVal not linked.")

# ---------------------------------------------------------
# 3. Parquet Export & Load (Metadata)
# ---------------------------------------------------------
print("\n--- Parquet Export & Load (Metadata) ---")

if pq:
    pq_path = "demo_meta.parquet"
    print(f"Exporting to {pq_path} (embedding metadata)...")
    vf.to_parquet(pq_path, include=['all'])
    
    # Verify Metadata Presence
    meta = pq.read_metadata(pq_path)
    if b'varframe_variables' in meta.metadata:
        print("  -> Metadata confirmed in file header.")
    
    # Load (Metadata-driven)
    print(f"Loading from {pq_path}...")
    vf_pq = VarFrame.load_parquet(pq_path)
    
    print(f"  -> Variables: {vf_pq.list_variables()}")
    if LazyVal in vf_pq.variables:
        print("  -> SUCCESS: LazyVal correctly linked via metadata!")
    else:
        print("  -> FAIL: LazyVal not linked.")

    print("\nVariables in loaded VarFrame:")
    print(vf_pq.describe_variables())

else:
    print("Skipping Parquet tests (pyarrow not installed).")

# ---------------------------------------------------------
# Cleanup
# ---------------------------------------------------------
print("\n--- Cleanup ---")
if os.path.exists(csv_path):
    os.remove(csv_path)
    print(f"Removed {csv_path}")

if pq and os.path.exists(pq_path):
    os.remove(pq_path)
    print(f"Removed {pq_path}")

print("\nDemo Complete.")
