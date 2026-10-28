# Validation record

The repository provides Python 3.12-pinned dependencies and reproducible command-line
entry points. Run these checks from the repository root after installing
`requirements.txt`.

## Commands

```bash
python run_all.py
python -m unittest discover -s tests -v
python scripts/verify_deliverables.py
```

## Evidence recorded in the project

| Check | Expected evidence |
| --- | --- |
| Raw source hashes | Match SHA-256 values in `DATA_SOURCES.md` |
| EDA source/clean rows | 344 / 344; no dropped rows or imputed measurements |
| EDA complete cases | 333; one `.` sex marker normalized |
| Model source rows | 398; six missing horsepower values retained |
| Train/test split | 321 / 77 rows; no exact car-name group overlap |
| Group leakage checks | No train/test name overlap and no overlap in CV folds |
| Preprocessing | Imputation and scaling are fitted inside training folds |
| Model selection | Selected by mean training CV RMSE |
| Reported metrics | Saved holdout metrics and README values are reconciled |
| Model reload | Saved pipeline can be reloaded for inference |
| Deliverables | Local Markdown links, Python entry points, report figures and SVG banner checked |
| Existing work | Original `Data Analysis/` exercises preserved |

## Limits

These are project checks, not proof of official internship approval or compliance
with an unseen rubric. The model is an educational historical benchmark for 1970–1982
vehicles, not a validated estimator for modern cars. A clean Windows dependency
installation was not independently exercised in the original build environment.
Re-run the commands after changing code, data, or dependencies.
