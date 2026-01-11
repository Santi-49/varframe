# Welcome to VarFrame

**Declarative DataFrame variable management with automatic DAG dependency resolution.**

[![PyPI version](https://badge.fury.io/py/varframe.svg)](https://badge.fury.io/py/varframe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/santi-49/varframe)

VarFrame is a Python library designed to bring structure, reproducibility, and maintainability to your data science and machine learning pipelines. It transforms your data processing from a linear script into a robust, self-documenting graph of dependencies.

---

## Why VarFrame?

Data pipelines often start as simple scripts but quickly grow into unmanageable "spaghetti code."

*   **The Problem:**
    *   **Implicit Dependencies:** "Why did column `X` change? Oh, because I modified column `Y` 200 lines above."
    *   **Execution Order Fragility:** "You have to run cell 4 before cell 2, but only if you skipped cell 3."
    *   **Copy-Paste Logic:** Feature engineering logic gets duplicated across training and inference pipelines, leading to drift.

*   **The VarFrame Solution:**
    *   **Explicit Dependencies:** Every variable declares exactly what it needs to be computed.
    *   **Automatic Resolution:** VarFrame builds a Directed Acyclic Graph (DAG) and executes transformations in the mathematically correct order.
    *   **Single Source of Truth:** Your variable classes *are* your pipeline. Use the exact same classes for training, evaluation, and production inference.

## Traditional vs. VarFrame

| Feature | Traditional (Pandas Script) | VarFrame |
| :--- | :--- | :--- |
| **Logic Definition** | Imperative (`df['new'] = df['old'] * 2`) | Declarative (Class `New` depends on `Old`) |
| **Execution Order** | Manually managed (top-to-bottop) | Automatic (DAG-based topological sort) |
| **Reusability** | Low (Copy-paste code blocks) | High (Import classes anywhere) |
| **Lazy Loading** | Manual caching or re-computation | Built-in (Compute only when accessed) |
| **Documentation** | Comments scattered in code | Self-documenting class structure |

---

## Key Features

*   **Declarative Syntax**: Define variables as Python classes.
*   **Automatic Dependency Resolution**: Never worry about execution order again.
*   **Lazy Loading**: Defer expensive computations until the data is actually needed.
*   **ML Integration**: Built-in `BaseModel` and `ModelVariable` to treat model predictions just like any other column.
*   **Metadata Preserved**: Export to Parquet with variable metadata intact.

## Installation

VarFrame is available on PyPI:

```bash
pip install varframe
```

## Quick Start

Here is a simple example showing how VarFrame automatically resolves dependencies.

```python
import pandas as pd
from varframe import VarFrame, BaseVariable, DerivedVariable

# 1. Define your variables as classes
class Radius(BaseVariable):
    raw_column = "radius" # Maps to input DataFrame column
    dtype = "float"

class Area(DerivedVariable):
    # Dependencies are specific pointers to other variable classes
    dependencies = [Radius]

    @classmethod
    def calculate(cls, df):
        import math
        # Access column by variable name
        return math.pi * (df[Radius.name] ** 2)

# 2. Initialize with data
data = pd.DataFrame({"radius": [1, 2, 5]})
vf = VarFrame(data)

# 3. Resolve dependencies
# You ask for 'Area', VarFrame knows it needs 'Radius'
vf.resolve(Area)

print(vf.df)
#    radius       area
# 0     1.0   3.141593
# 1     2.0  12.566371
# 2     5.0  78.539816
```

## Next Steps

*   Check out the **[User Guide](user_guide.md)** for deep dives into Lazy Loading, Machine Learning integration, and Data I/O.
*   See the **[API Reference](api_reference.md)** for detailed class documentation.
