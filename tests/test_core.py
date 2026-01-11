import pandas as pd
import pytest
from varframe import VarFrame, BaseVariable, DerivedVariable

class Radius(BaseVariable):
    raw_column = "radius"
    dtype = "float"

class Area(DerivedVariable):
    dependencies = [Radius]
    
    @classmethod
    def calculate(cls, df):
        import math
        return math.pi * (df[Radius.name] ** 2)

def test_varframe_basic_resolution():
    df_raw = pd.DataFrame({"radius": [1.0, 2.0]})
    vf = VarFrame(df_raw)
    
    vf.resolve(Area)
    
    assert Radius.name in vf.columns
    assert Area.name in vf.columns
    
    # Check computation correctness
    assert vf[Area.name].iloc[0] == pytest.approx(3.14159, abs=1e-5)
    assert vf[Area.name].iloc[1] == pytest.approx(12.56637, abs=1e-5)

def test_lazy_execution():
    class ExpensiveVar(DerivedVariable):
        dependencies = [Radius]
        lazy = True
        
        @classmethod
        def calculate(cls, df):
            return df[Radius.name] * 2

    df_raw = pd.DataFrame({"radius": [10.0]})
    vf = VarFrame(df_raw)
    
    vf.resolve(ExpensiveVar) 
    # Lazy var should NOT be in columns yet
    assert ExpensiveVar.name not in vf.columns
    
    # Access triggers compute
    val = vf[ExpensiveVar]
    assert val.iloc[0] == 20.0
    # Should still not be stored in columns (transient by default for lazy? 
    # Wait, check implementation: lazy vars are NOT added to self unless cached? 
    # Docs say "computed on access and not stored in DataFrame" implied or explicit?
    # Let's check DataFrame behavior. Usually lazy vars are returned but not persisted unless explicitly set.
    # Actually, looking at code: __getitem__ computes but doesn't set item unless we manually do it.
    
    assert ExpensiveVar.name not in vf.columns
