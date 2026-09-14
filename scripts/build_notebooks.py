"""Build and execute plain-Python Jupyter notebooks using the installed runtime.

This small runner supports these notebooks' ordinary Python cells and captures
stdout/stderr plus figures shown with plt.show(). It is not a general Jupyter
kernel implementation: no magics, widgets, shell commands, or async cells.
Every stored output comes from execution; failures stop the build.

Optional full Jupyter execution:
  pip install -r requirements-notebooks.txt
  jupyter nbconvert --to notebook --execute --inplace PATH.ipynb
"""
import base64
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "source": text.strip().splitlines(keepends=True),
            "execution_count": None, "outputs": []}


SETUP = '''from pathlib import Path
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "run_all.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
pd.set_option("display.max_columns", 20)

def show_saved_figure(path):
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.imshow(plt.imread(path))
    ax.axis("off")
    fig.tight_layout()
    plt.show()

print("Project root found. Included source data will be used offline.")'''


def eda_cells():
    return [
        md('''# Task 1 · Palmer Penguins EDA

**Arrowstack Data Science · Azhar**

Explore sample composition, missingness, morphology, and possible confounding in
the Kaggle Palmer Archipelago dataset. This notebook follows the screenshot's
Kaggle EDA task title. It does not claim compliance with an unseen assignment PDF.

Run cells from top to bottom after installing `requirements-notebooks.txt`.
The source CSV is included; no Kaggle account or network access is required.

[Kaggle dataset](https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data)
· [Original research data documentation](https://allisonhorst.github.io/palmerpenguins/)

**Questions:** Which species are represented? Which measurements vary by species?
Does the pooled size relationship persist within species? Can island effects be
separated from species? What changes when incomplete records are excluded?'''),
        code(SETUP),
        md("## 1. Inspect the original data\n\nOne row is an observation; the simplified file has no unique animal ID."),
        code('''from task1_eda.analysis import load_data, clean_data, summarize, correlations, outlier_review, run, NUMERIC, CATEGORICAL
raw = load_data()
print("Shape:", raw.shape)
print(raw.head(8).to_string(index=False))
print("\\nTypes:\\n", raw.dtypes.to_string())
print("\\nOriginal missing values:\\n", raw.isna().sum().to_string())
print("\\nSex values including nonstandard markers:\\n", raw.sex.value_counts(dropna=False).to_string())'''),
        md('''## 2. Clean with explicit decisions

Normalize `.` as missing sex, lowercase the labels, rename culmen to bill, and
validate positive numeric measurements. Preserve missing measurements rather than
inventing observations. Report identical measurement rows; do not equate them
with duplicate animals. `source_row` is traceability metadata, not a predictor.'''),
        code('''clean, audit = clean_data(raw)
print(json.dumps(audit, indent=2))
print("\\nDescriptive numeric statistics:\\n", clean[NUMERIC].describe().round(2).to_string())
assert len(clean) == len(raw)
assert audit["imputed_values"] == 0'''),
        md("## 3. Reproduce all report files\n\nThe same tested Python functions power the command-line workflow and this notebook."),
        code('''summary_result = run()
print(summary_result)
REPORTS = ROOT / "task1_eda/reports"
show_saved_figure(REPORTS / "figures/01_quality_and_composition.png")'''),
        md("## 4. Compare measurements by species\n\nShow sample size and available mass observations with every group mean."),
        code('''species_summary = summarize(clean)
print(species_summary.round(2).to_string())
show_saved_figure(REPORTS / "figures/02_species_and_measurements.png")'''),
        md('''## 5. Check confounding and pooled relationships

Correlation does not establish causation. If species occur on different islands,
an island comparison also changes the species composition. Check the cross-tab
before interpreting geographic differences.'''),
        code('''print("Species × island counts:\\n", pd.crosstab(clean.species, clean.island).to_string())
print("\\nPaired flipper-length/body-mass correlations:\\n", correlations(clean).round(3).to_string(index=False))
show_saved_figure(REPORTS / "figures/03_confounding_and_correlations.png")'''),
        md('''## 6. Review outliers and uncertainty

Within-species 1.5-IQR flags identify measurements for review, not automatic
deletion. Bootstrap mean intervals use 2,000 resamples and seed 42. They assume
independent observations and do not establish population representativeness.'''),
        code('''review = outlier_review(clean)
print(review.loc[review.flagged_rows > 0].round(2).to_string(index=False))
show_saved_figure(REPORTS / "figures/04_bootstrap_intervals.png")'''),
        md("## 7. Missingness sensitivity\n\nCompare available mass records with complete cases. A small difference is not proof that data are missing at random."),
        code('''complete = clean.dropna(subset=NUMERIC + CATEGORICAL)
print(f"Available mass: n={clean.body_mass_g.count()}, mean={clean.body_mass_g.mean():.1f} g")
print(f"Complete cases: n={len(complete)}, mean={complete.body_mass_g.mean():.1f} g")
print(f"Mean difference: {complete.body_mass_g.mean()-clean.body_mass_g.mean():.1f} g")'''),
        md('''## 8. Conclusions and next steps

- Compare body measurements within species; pooled relationships mix species differences.
- Report sample counts, missingness, and uncertainty alongside means.
- Species and island are linked in this sample, so avoid causal geographic claims.
- Investigate unusual measurements using field records before changing them.
- Collect balanced samples and individual identifiers before generalizing beyond this dataset.

Read `reports/final_report.md` for the generated numerical findings,
`docs/data_dictionary.md` for variables, and `../docs/DATA_SOURCES.md` for citations.
This notebook is descriptive EDA; it does not train a prediction model.'''),
    ]


def regression_cells():
    return [
        md('''# Task 2 · Auto MPG regression pipeline

**Arrowstack Data Science · Azhar**

Predict historical city-cycle fuel efficiency using vehicle specifications.
The continuous target is `mpg`. This is regression, not classification.

[UCI Auto MPG source](https://archive.ics.uci.edu/dataset/9/auto+mpg)
· Quinlan, R. (1993), [DOI](https://doi.org/10.24432/C5859H), CC BY 4.0.

Protocol: hold out exact normalized car-name groups, compare fixed candidate
pipelines using five training-only group folds, freeze the selection, then evaluate
the selected model and naive baseline on the holdout. No model decisions use the
held-out test scores. This is a small historical benchmark, not a modern-car service.'''),
        code(SETUP),
        md("## 1. Understand the target and predictors"),
        code('''from task2_regression.train import load_data, prepare_features, group_split, make_pipeline, candidate_models, train, FEATURES, TARGET
data = load_data()
X, y = prepare_features(data), data[TARGET]
print("X:", X.shape, "y:", y.shape)
print("Predictors:", FEATURES)
print(data.head().to_string(index=False))
print("\\nMissing values:\\n", data.isna().sum().to_string())
assert TARGET not in X.columns
assert "car_name" not in X.columns'''),
        md('''## 2. Reserve the grouped test set

The dataset contains repeated names. Exact names must stay in one partition.
The 20% test fraction applies to groups, so row counts need not be exactly 80/20.
Typos and related model names can still create dependence; this grouping is not
a guarantee that all vehicle families are independent.'''),
        code('''train_ids, test_ids = group_split(data)
train_names = set(data.iloc[train_ids].car_name)
test_names = set(data.iloc[test_ids].car_name)
print("Training rows:", len(train_ids), "Test rows:", len(test_ids))
print("Training names:", len(train_names), "Test names:", len(test_names))
print("Name overlap:", len(train_names & test_names))
assert not train_names & test_names'''),
        md('''## 3. Build the preprocessing and candidate models

Numeric branch: median imputation, missing indicators, and scaling.
Categorical branch: most-frequent imputation and one-hot encoding with unknown
categories handled safely. All learned statistics fit inside each CV training fold.
The naive baseline predicts the training mean. Fixed candidates include Linear
Regression, Ridge, Random Forest, and Gradient Boosting.'''),
        code('''candidates = candidate_models()
for name, estimator in candidates.items():
    print(name, "→", estimator)
print("\\nExample complete pipeline:\\n", make_pipeline(candidates["Ridge"]))'''),
        md('''## 4. Run the full training workflow

Selection uses lowest mean training-fold RMSE. The code saves fold scores,
split assignments, test predictions, diagnostics, metadata, and the pipeline.
The saved model stays fitted on the training partition only. Results below are
computed by this run, not hardcoded examples.'''),
        code('''results = train()
REPORTS = ROOT / "task2_regression/reports"
print("Selected:", results["selected_model"])
print("Selection rule:", results["selection_metric"])
print(pd.read_csv(REPORTS / "model_comparison.csv").round(4).to_string(index=False))
print("\\nCV partition checks:\\n", pd.read_csv(REPORTS / "cv_split_checks.csv").to_string(index=False))
show_saved_figure(REPORTS / "figures/01_model_comparison.png")'''),
        md('''## 5. Final holdout metrics

MAE: typical absolute error, in mpg. RMSE: error in mpg that emphasizes larger
misses. MSE: squared mpg. R²: variance-based performance relative to a mean
reference, not a classification accuracy percentage. The baseline uses the training
mean, while the R² definition uses the observed test mean in its denominator.'''),
        code('''print("Selected model test metrics:\\n", json.dumps(results["test_metrics"], indent=2))
print("Training-mean baseline test metrics:\\n", json.dumps(results["test_baseline_metrics"], indent=2))
show_saved_figure(REPORTS / "figures/02_holdout_diagnostics.png")'''),
        md('''## 6. Post-selection diagnostics

Inspect origin-specific errors and permutation importance after freezing the model.
Do not retune it using these holdout diagnostics and still call the same holdout
untouched. Correlated features can share importance; importance is not causation.'''),
        code('''print(pd.read_csv(REPORTS / "errors_by_origin.csv").round(3).to_string(index=False))
print("\\nFeature importance:\\n", pd.read_csv(REPORTS / "permutation_importance.csv").round(3).to_string(index=False))
show_saved_figure(REPORTS / "figures/03_permutation_importance.png")'''),
        md('''## 7. Reload the locally trained model and predict

The following input rows are **illustrative vehicle specifications**, not another
measured dataset. One horsepower value is deliberately blank to demonstrate the
trained imputer. Load only trusted joblib files: they can execute code.'''),
        code('''import joblib
artifact = joblib.load(ROOT / "task2_regression/models/auto_mpg_pipeline.joblib")
example = pd.read_csv(ROOT / "task2_regression/examples/new_cars.csv")
example["predicted_mpg"] = artifact["pipeline"].predict(prepare_features(example))
print(example.round(3).to_string(index=False))
example.to_csv(REPORTS / "example_predictions.csv", index=False)'''),
        md('''## 8. Limitations and next steps

- Model years are historical two-digit codes 70–82. Reject modern years.
- This random group split is not a forecast evaluation; consider a separate
  time-ordered study before making claims about future vehicles.
- A small holdout has uncertainty, and exact names do not identify every related model family.
- Unseen categories and out-of-range numeric inputs require review.
- No prediction intervals, deployment monitoring, or modern-car validation are included.

Read `reports/final_report.md`, `docs/data_dictionary.md`, and the repository's
beginner guide. Run `python -m unittest discover -s tests -v` for validation.
The binary model is generated locally and excluded from Git; the source data,
metadata, notebook outputs, and prediction results are versioned.'''),
    ]


def execute_notebook(notebook, name):
    namespace = {"__name__": "__main__"}
    count = 0
    original_show = plt.show
    try:
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            count += 1
            outputs = []
            buffer = io.StringIO()
            def flush_text():
                value = buffer.getvalue()
                if value:
                    outputs.append({"output_type": "stream", "name": "stdout", "text": value.splitlines(keepends=True)})
                    buffer.seek(0)
                    buffer.truncate(0)
            def show(*args, **kwargs):
                flush_text()
                for number in list(plt.get_fignums()):
                    figure = plt.figure(number)
                    png = io.BytesIO()
                    figure.savefig(png, format="png", dpi=100, bbox_inches="tight")
                    outputs.append({"output_type": "display_data", "metadata": {}, "data": {
                        "image/png": base64.b64encode(png.getvalue()).decode("ascii"),
                        "text/plain": ["<Matplotlib figure generated by the preceding cell>"]}})
                    plt.close(figure)
            plt.show = show
            with redirect_stdout(buffer), redirect_stderr(buffer):
                exec(compile("".join(cell["source"]), f"{name}:cell{count}", "exec"), namespace)
            flush_text()
            cell["execution_count"] = count
            cell["outputs"] = outputs
    finally:
        plt.show = original_show
        plt.close("all")
    return count


def build(cells, path):
    for number, cell in enumerate(cells):
        cell["id"] = hashlib.sha256(f"{number}:{''.join(cell['source'])}".encode()).hexdigest()[:12]
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": platform.python_version()},
                             "execution_method": "scripts/build_notebooks.py; sequential plain-Python cells with captured stdout and figures"},
                "cells": cells}
    executed = execute_notebook(notebook, str(path.relative_to(ROOT)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Built {path.relative_to(ROOT)}: {executed} executed Python cells")


if __name__ == "__main__":
    build(eda_cells(), ROOT / "task1_eda/notebooks/penguins_eda.ipynb")
    build(regression_cells(), ROOT / "task2_regression/notebooks/auto_mpg_regression.ipynb")
