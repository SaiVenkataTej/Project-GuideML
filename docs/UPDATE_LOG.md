# Documentation Update Log

**Date**: 2026-03-12

## Changes Made

### Architectural Standardization
- **Uniform Diagnostics**: Enforced `y_true`/`y_proba` naming across all model classes.
- **Model DNA**: Implemented `get_parameter_descriptions` protocol to expose tuned hyperparameters to the UI.

### Explainability Integration
- **SHAP Layer**: Added `plot_shap_summary` to `visualization.py` and integrated it into the `ModelExecutor` lifecycle.
- **UI Enhancements**: Added SHAP global influence cards to the dashboard.

### Stability & UX
- **Windows IO Fix**: Implemented stable temporary paths for `joblib` to prevent multiprocessing errors.
- **UI Integrity**: Resolved CSS injection bugs in `dashboard.html` and added "Best For" storytelling from `knowledge_base.py`.

---

**Date**: 2026-02-15

## Changes Made

### API_REFERENCE.md
- **Removed**: Section 4 "Scripts & Tests" which referenced:
  - `repro_clone_error.py` (file no longer exists)
  - `test_app.py` (file no longer exists)
  
**Reason**: These test/debugging files were removed from the project at some point, but the documentation still referenced them. Cleaned up to match current codebase state.

## Files Verified
✅ All 21 current Python source files are documented
✅ No ghost references remain
✅ Documentation matches current project structure
