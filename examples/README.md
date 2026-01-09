# VarFrame Examples

This directory contains example scripts demonstrating VarFrame features.

## Running Examples

From the project root directory:

```bash
# Run ensemble demo
python -m examples.ensemble_demo

# Or directly
python examples/ensemble_demo.py
```

## Available Examples

### `ensemble_demo.py`

Demonstrates DAG-based dependency resolution with ensemble models:

- **BaseVariables**: Mapping raw columns (`Lap`, `Gap`, `TireAge`)
- **DerivedVariables**: Computed columns (`GapDelta`)
- **BaseModel**: Declarative ML model definitions (RF, Linear, Ridge)
- **ModelVariable**: Model predictions as DataFrame columns
- **Ensemble**: Model that uses other model predictions as inputs

Key concepts shown:
1. Automatic dependency resolution via topological sort
2. Lazy computation with `compute=False` + `resolve()`
3. Auto-training of untrained models (with warnings)
4. Building ensembles from multiple model predictions

## Requirements

Most examples require the ML extras:

```bash
pip install varframe[ml]
```
