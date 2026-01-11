"""
Tests for VarFrame loading functionality.
Covers: load_csv, load_parquet, column selection, ambiguity handling.
"""
import pytest
import pandas as pd
from varframe import VarFrame, BaseVariable, DerivedVariable, AmbiguityError


# =============================================================================
# BASIC LOADING
# =============================================================================

class TestBasicLoading:
    """Test basic load/save roundtrip."""
    
    def test_csv_roundtrip(self, tmp_path):
        """CSV export and import preserves data."""
        class Col1(BaseVariable):
            name = "col1"
            raw_column = "a"
        
        df = pd.DataFrame({"a": [1, 2, 3]})
        vf = VarFrame(df, [Col1])
        vf.to_csv(tmp_path / "test.csv", index=False)
        
        vf2 = VarFrame.load_csv(tmp_path / "test.csv")
        assert "col1" in vf2.columns
        assert list(vf2["col1"]) == [1, 2, 3]
    
    def test_parquet_roundtrip(self, tmp_path):
        """Parquet export and import preserves data and variables."""
        class Col2(BaseVariable):
            name = "col2"
            raw_column = "b"
        
        df = pd.DataFrame({"b": [10, 20]})
        vf = VarFrame(df, [Col2])
        vf.to_parquet(tmp_path / "test.parquet")
        
        vf2 = VarFrame.load_parquet(tmp_path / "test.parquet")
        assert "col2" in vf2.columns
        assert vf2.get_variable("col2") == Col2


# =============================================================================
# COLUMN SELECTION
# =============================================================================

class TestColumnSelection:
    """Test variables/exclude/discard_unmatched parameters."""
    
    def test_variables_whitelist(self, tmp_path):
        """Only whitelisted variables are loaded."""
        class VarA(BaseVariable):
            name = "var_a"
            raw_column = "a"
        
        class VarB(BaseVariable):
            name = "var_b"
            raw_column = "b"
        
        df = pd.DataFrame({"a": [1], "b": [2]})
        vf = VarFrame(df, [VarA, VarB])
        vf.to_csv(tmp_path / "test.csv", index=False)
        
        vf2 = VarFrame.load_csv(tmp_path / "test.csv", variables=[VarA])
        assert "var_a" in vf2.columns
        assert "var_b" not in vf2.columns
    
    def test_exclude_blacklist(self, tmp_path):
        """Excluded variables are not loaded."""
        class VarC(BaseVariable):
            name = "var_c"
            raw_column = "c"
        
        class VarD(BaseVariable):
            name = "var_d"
            raw_column = "d"
        
        df = pd.DataFrame({"c": [1], "d": [2]})
        vf = VarFrame(df, [VarC, VarD])
        vf.to_parquet(tmp_path / "test.parquet")
        
        vf2 = VarFrame.load_parquet(tmp_path / "test.parquet", exclude=[VarD])
        assert "var_c" in vf2.columns
        assert "var_d" not in vf2.columns
    
    def test_variables_exclude_mutual_exclusion(self, tmp_path):
        """Cannot use variables and exclude together."""
        class Dummy(BaseVariable):
            name = "dummy"
            raw_column = "x"
        
        df = pd.DataFrame({"x": [1]})
        df.to_csv(tmp_path / "test.csv", index=False)
        
        with pytest.raises(ValueError):
            VarFrame.load_csv(tmp_path / "test.csv", variables=[Dummy], exclude=[Dummy])
    
    def test_discard_unmatched_true(self, tmp_path):
        """Unmatched columns are dropped by default."""
        class Known(BaseVariable):
            name = "known"
            raw_column = "known"
        
        df = pd.DataFrame({"known": [1], "unknown": [99]})
        df.to_csv(tmp_path / "test.csv", index=False)
        
        vf = VarFrame.load_csv(tmp_path / "test.csv")
        assert "known" in vf.columns
        assert "unknown" not in vf.columns
    
    def test_discard_unmatched_false(self, tmp_path):
        """Unmatched columns are kept when discard_unmatched=False."""
        class Known2(BaseVariable):
            name = "known2"
            raw_column = "known2"
        
        df = pd.DataFrame({"known2": [1], "extra": [99]})
        df.to_csv(tmp_path / "test.csv", index=False)
        
        vf = VarFrame.load_csv(tmp_path / "test.csv", discard_unmatched=False)
        assert "known2" in vf.columns
        assert "extra" in vf.columns


# =============================================================================
# AMBIGUITY HANDLING
# =============================================================================

class TestAmbiguity:
    """Test handling of multiple variable definitions with same name."""
    
    def test_ambiguity_raises_error(self, tmp_path):
        """Multiple candidates without disambiguation raises AmbiguityError."""
        class Root(BaseVariable):
            name = "amb_root"
            raw_column = "r"
        
        class TargetV1(DerivedVariable):
            name = "amb_target"
            dependencies = [Root]
            @classmethod
            def calculate(cls, df): return df["amb_root"] + 1
        
        df = pd.DataFrame({"r": [10]})
        vf = VarFrame(df, [Root, TargetV1])
        vf.to_parquet(tmp_path / "test.parquet")
        
        # Define second candidate with same name
        class TargetV2(DerivedVariable):
            name = "amb_target"
            dependencies = [Root]
            @classmethod
            def calculate(cls, df): return df["amb_root"] * 2
        
        with pytest.raises(AmbiguityError) as exc:
            VarFrame.load_parquet(tmp_path / "test.parquet")
        
        assert "amb_target" in str(exc.value)
    
    def test_ambiguity_resolution(self, tmp_path):
        """Ambiguity parameter resolves conflicts."""
        class Root2(BaseVariable):
            name = "res_root"
            raw_column = "r"
        
        class ResV1(DerivedVariable):
            name = "res_target"
            dependencies = [Root2]
            @classmethod
            def calculate(cls, df): return 1
        
        df = pd.DataFrame({"r": [10]})
        vf = VarFrame(df, [Root2, ResV1])
        vf.to_parquet(tmp_path / "test.parquet")
        
        class ResV2(DerivedVariable):
            name = "res_target"
            dependencies = [Root2]
            @classmethod
            def calculate(cls, df): return 2
        
        vf2 = VarFrame.load_parquet(
            tmp_path / "test.parquet",
            ambiguity={"res_target": ResV2}
        )
        assert vf2.get_variable("res_target") == ResV2


# =============================================================================
# INTEGRITY WARNINGS
# =============================================================================

class TestIntegrityWarnings:
    """Test hash-based integrity checking on load."""
    
    def test_metadata_change_warning(self, tmp_path, capsys):
        """Changing docstring triggers metadata warning."""
        class IntegRoot(BaseVariable):
            name = "integ_root"
            raw_column = "val"
        
        class IntegTarget(DerivedVariable):
            """Original docstring"""
            name = "integ_target"
            dependencies = [IntegRoot]
            @classmethod
            def calculate(cls, df): return df["integ_root"] + 1
        
        df = pd.DataFrame({"val": [10]})
        vf = VarFrame(df, [IntegRoot, IntegTarget])
        vf.to_parquet(tmp_path / "test.parquet")
        
        # Modify docstring
        IntegTarget.__doc__ = "Changed docstring"
        
        VarFrame.load_parquet(tmp_path / "test.parquet")
        out, _ = capsys.readouterr()
        
        assert "Metadata (desc/lazy) changed" in out
