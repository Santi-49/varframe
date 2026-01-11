# Project Roadmap

## ⚙️ Core Logic & Performance
- [x] **Lazy Loading**: Optimize loading of heavy variables only when needed.
- [ ] **Sparse Data**: Investigate/implement sparse data structures for memory efficiency.
- [ ] **Inter-VF Dependencies**: Allow VarFrames to depend on variables from other VarFrames.
- [ ] **Deep Resolution**: Ensure `resolve(Model)` automatically resolves all its input variable dependencies recursively.

## 🤖 ML Integration
- [ ] **Model linkage**: Formalize how models link to the underlying dataframe and raw data.
- [ ] **Data Splitting**: Add utilities for train/test/val splitting within the VarFrame context.

## 🚀 New Features
- [x] **Export**: Add `to_csv` and `to_parquet` methods that handle computed variables correctly.
- [ ] **Testing**: Expand test suite (unit and integration tests).