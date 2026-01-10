# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
