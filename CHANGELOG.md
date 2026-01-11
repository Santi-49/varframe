# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.1] - 2026-01-11

### Added
- **Enhanced Load Functions**: `load_csv` and `load_parquet` now support advanced column selection and disambiguation:
    - `variables`: Whitelist of variable classes to load (implicit disambiguation)
    - `exclude`: Blacklist of variable classes to exclude
    - `discard_unmatched`: Drop columns not matching any variable (default: `True`)
    - `ambiguity`: Dict mapping variable names to specific classes for explicit disambiguation
- **AmbiguityError Exception**: Raised when multiple variable definitions match the same column name without disambiguation
- **Parquet Column Selection**: Uses `pyarrow.parquet.read_table(columns=[...])` for efficient I/O when loading specific columns
- **Validation**: `variables` and `exclude` parameters are mutually exclusive

### Changed
- **CSV/Parquet Loading**: Now raises `AmbiguityError` when multiple variable definitions exist with the same name (removed score-based matching)
- **Default `discard_unmatched=True`**: Unmatched columns are now dropped by default

### Fixed
- Fixed pyarrow schema API compatibility (`meta.schema.names` instead of deprecated `.field()/.length`)

## [1.4.0] - 2026-01-11

### Added
- **Granular Integrity Checks**: Parquet import/export now includes robust, recursive hashing of variable definitions to ensure data consistency.
    - Hashes calculation logic (source code), dependencies (recursive), attributes (dtype), and metadata (description/lazy).
    - Stores hashes in Parquet metadata (`varframe_hashes`).
    - Compares stored hashes with current environment on load.
    - **Color-Coded Warnings**:
        - 🔴 **Red**: Critical changes (Calculation logic, Dependencies, Raw Column).
        - 🟠 **Orange**: Type changes (Dtype).
        - ⚪ **White**: Metadata changes (Description, Lazy flag).
    - Metadata encapsulation prevents trivial documentation changes from invalidating downstream dependencies.

## [1.3.0] - 2026-01-11

### Added
- **Smart Data Persistence**:
    - `VarFrame.load_csv(path)`: Reconstructs VarFrames from CSV, automatically discovering and linking variable classes from the environment.
    - `VarFrame.load_parquet(path)`: Reconstructs VarFrames from Parquet, using embedded metadata for precise linking (fallback to discovery).
- **Enhanced Export**:
    - `to_csv` and `to_parquet` now support `include` and `variables` arguments (like `view`) to compute lazy variables during export.
    - Added safety warnings if registered variables are excluded from the export.
    - Added `VarFrame.name` attribute; exports default to `{name}.csv/.parquet` if no path is provided.
- **Parquet Metadata**: Variable definitions are now saved in the Parquet file metadata for reliable restoration.
- `examples/export_demo.py`: Comprehensive demo of the new Import/Export capabilities.

## [1.2.1] - 2026-01-11

### Added
- **Lazy Loading**: `DerivedVariable` can now be marked with `lazy = True`. These variables are computed on-demand and are not stored in the DataFrame, effectively optimizing memory usage for transient calculations.
- **Flexible Views**: New `VarFrame.view()` method to export dataframes with robust filtering logic:
    - Filter by category: `vf.view(include=["base", "lazy"])`
    - Filter by variable list: `vf.view(variables=[MyVar])`
    - Automatically computes lazy variables required for the view.

## [1.1.0] - 2026-01-10

### Added
- `vf.explain_calculation()`: New method to visualize the calculation plan for variables, showing availability status (✅/⏳) and warnings.
- `explain_dependencies()`: Helper function to visualize the DAG structure.
- `VFConfig.null_context()`: Helper for warning suppression.
- `examples/ensemble_demo.py`: Expanded with Case 4 (Calculation Plan) and Case 5 (Partial State).

### Changed
- Refactored `add_variable` and `add_variables`:
    - `add_variables(*vars, compute=True)` is now the primary method.
    - `add_variable` is an alias for `add_variables`.
- Warning suppression in `resolve()` now properly silences downstream model operations (auto-training, inference) using a context manager.
- Warnings in `explain_calculation` are now concise codes (`[I, T, M]`) with an optional legend.

### Fixed
- Fixed issue where explicit variable addition could trigger implicit operation warnings.
- Fixed issue where `resolve(suppress_warnings=True)` leaked model training warnings.
