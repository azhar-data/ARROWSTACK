# Validation record

Verified on 14 September 2026 with Python 3.12.14, NumPy 2.3.5, pandas 2.2.3,
Matplotlib 3.10.8, scikit-learn 1.8.0, and joblib 1.5.3.

## Commands

Run from the repository root in the environment containing `requirements.txt`:

```bash
python run_all.py
python scripts/build_notebooks.py
python -m unittest discover -s tests -v
python scripts/verify_deliverables.py
```

## Evidence

| Check | Observed result / evidence |
| --- | --- |
| Raw source hashes | Match the SHA-256 values in `DATA_SOURCES.md` |
| EDA source/clean rows | 344 / 344; no dropped rows or imputed measurements |
| EDA complete cases | 333; one `.` sex marker normalized; 19 missing cells remain |
| Model source rows | 398; six missing horsepower values retained |
| Train/test split | 321 / 77 rows; 244 / 61 exact-name groups |
| Group leakage checks | Zero train/test name overlap and zero overlap in each of five CV folds |
| Preprocessing | Numeric imputer medians equal training-only medians; predicting changed test data does not refit them |
| Model selection | Lowest mean training CV RMSE selects Gradient Boosting |
| Reported metrics | Recomputed from saved test predictions and matched to `metrics.json` |
| Model reload | Predictions match the original fitted pipeline to numerical tolerance |
| Edge cases | Missing/unseen categories, blank horsepower, invalid schema, infinities, bad years/cylinder counts, protected output paths |
| Automated tests | 20 tests passed using built-in `unittest` |
| Notebooks | 2 notebooks; 8 executed Python cells each; 7 embedded plot outputs total |
| Figure review | All 7 generated PNG files visually reviewed for labels, clipping, units, and legends |
| Documentation | Local Markdown links and README metric rounding checked by verification script |
| Existing work | Original `Data Analysis/` files preserved unchanged |

## Reproduction and execution limits

The initial command-line run and the notebook rebuild produced identical numeric
metrics. Figure files were inspected visually, and notebook output images were
decoded and checked. The custom notebook builder executes ordinary Python cells
sequentially and captures their real text and figures. It is not a Jupyter kernel
and does not emulate magics/widgets; these notebooks do not require them.

A full Jupyter frontend/kernel session and a clean Windows dependency installation
were not exercised in the build environment. Optional notebook dependencies and
instructions are provided for those workflows. No test is evidence of internship
approval, unseen rubric compliance, or modern-vehicle model validity.

The metric tables are deliberately preserved as a measured run snapshot. If code,
data, or dependencies change, rerun the workflow and update README summaries after
review; `verify_deliverables.py` checks their rounded metric values for consistency.
