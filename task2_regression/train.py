"""Run: python -m task2_regression.train [--output DIRECTORY]."""
import argparse
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from project_utils import ROOT, NAVY, TEAL, ORANGE, markdown_table, save_figure, save_json, set_style, sha256

PROJECT = ROOT / "task2_regression"
RAW_PATH = PROJECT / "data/raw/auto-mpg.data"
NUMERIC = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
CATEGORICAL = ["origin"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "mpg"
SEED = 42
RAW_COLUMNS = [TARGET] + NUMERIC + CATEGORICAL + ["car_name"]


def load_data(path=RAW_PATH):
    df = pd.read_csv(path, sep=r"\s+", names=RAW_COLUMNS, na_values="?", quotechar='"')
    origin = pd.to_numeric(df.origin, errors="raise")
    if not origin.isin([1, 2, 3]).all():
        raise ValueError("Raw UCI origin code must be 1, 2, or 3")
    df["origin"] = origin.map({1: "usa", 2: "europe", 3: "japan"})
    df["car_name"] = df.car_name.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    if df.car_name.isna().any() or df.car_name.eq("").any():
        raise ValueError("Every row needs a car name for group-aware evaluation")
    prepare_features(df)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise")
    if df[TARGET].isna().any() or not np.isfinite(df[TARGET]).all() or df[TARGET].le(0).any():
        raise ValueError("Targets must be positive and finite; missing targets cannot be imputed")
    df.insert(0, "source_row", np.arange(1, len(df) + 1))
    return df


def prepare_features(frame):
    """Validate inference/training schema; leave fitting of statistics to the pipeline."""
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("At least one row is required")
    X = frame[FEATURES].copy()
    for column in NUMERIC:
        X[column] = pd.to_numeric(X[column], errors="raise")
        present = X[column].dropna()
        if not np.isfinite(present).all() or present.le(0).any():
            raise ValueError(f"{column} must be positive and finite, or blank for imputation")
    years = X.model_year.dropna()
    if ((years < 70) | (years > 82) | (years % 1 != 0)).any():
        raise ValueError("model_year must be an integer two-digit code, 70 through 82")
    if not X.cylinders.dropna().isin([3, 4, 5, 6, 8]).all():
        raise ValueError("cylinders must be one of 3, 4, 5, 6, 8, or blank")
    # Use np.nan/object rather than pandas.NA to match SimpleImputer's default marker.
    origin = X.origin.astype("string").str.strip().str.lower().replace("", pd.NA)
    X["origin"] = origin.astype(object).where(origin.notna(), np.nan)
    return X


def make_pipeline(model):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL),
    ], remainder="drop")
    return Pipeline([("preprocess", preprocess), ("model", model)])


def candidate_models():
    """Fixed candidates declared before inspecting held-out test results."""
    return {
        "Dummy mean": DummyRegressor(strategy="mean"),
        "Linear regression": LinearRegression(),
        "Ridge": Ridge(alpha=10.0),
        "Random forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                                 random_state=SEED, n_jobs=1),
        "Gradient boosting": GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                                         learning_rate=.05, random_state=SEED),
    }


def group_split(df):
    splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED)
    train_ids, test_ids = next(splitter.split(df[FEATURES], df[TARGET], groups=df.car_name))
    if set(df.iloc[train_ids].car_name) & set(df.iloc[test_ids].car_name):
        raise AssertionError("Car-name groups overlap between train and test")
    return train_ids, test_ids


def metric_dict(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1 or y_true.shape != y_pred.shape or y_true.size < 2:
        raise ValueError("Metrics require aligned one-dimensional arrays with at least two rows")
    mse = float(mean_squared_error(y_true, y_pred))
    return {"mae_mpg": float(mean_absolute_error(y_true, y_pred)), "mse_mpg_squared": mse,
            "rmse_mpg": float(np.sqrt(mse)), "r2": float(r2_score(y_true, y_pred))}


def train(output=PROJECT):
    output = Path(output)
    reports = output / "reports"
    models_dir = output / "models"
    processed = output / "data/processed"
    for path in (reports, models_dir, processed):
        path.mkdir(parents=True, exist_ok=True)
    df = load_data()
    X, y = prepare_features(df), df[TARGET]
    train_ids, test_ids = group_split(df)
    X_train, X_test = X.iloc[train_ids], X.iloc[test_ids]
    y_train, y_test = y.iloc[train_ids], y.iloc[test_ids]
    train_groups = df.iloc[train_ids].car_name
    cv = list(GroupKFold(n_splits=5).split(X_train, y_train, groups=train_groups))
    fold_details = []
    for fold, (fit_ids, valid_ids) in enumerate(cv, 1):
        overlap = set(train_groups.iloc[fit_ids]) & set(train_groups.iloc[valid_ids])
        if overlap:
            raise AssertionError("CV car-name groups overlap")
        fold_details.append({"fold": fold, "fit_rows": len(fit_ids), "valid_rows": len(valid_ids),
                             "group_overlap": len(overlap)})
    candidates = candidate_models()
    rows, fold_rows = [], []
    for name, estimator in candidates.items():
        scores = cross_validate(make_pipeline(clone(estimator)), X_train, y_train, cv=cv,
                                scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error",
                                         "r2": "r2"}, n_jobs=1, error_score="raise")
        rmse = -scores["test_rmse"]
        rows.append({"model": name, "cv_rmse_mean_mpg": float(rmse.mean()),
                     "cv_rmse_std_mpg": float(rmse.std(ddof=1)),
                     "cv_mae_mean_mpg": float((-scores["test_mae"]).mean()),
                     "cv_r2_mean": float(scores["test_r2"].mean())})
        for i in range(5):
            fold_rows.append({"model": name, "fold": i+1, "rmse_mpg": float(rmse[i]),
                              "mae_mpg": float(-scores["test_mae"][i]), "r2": float(scores["test_r2"][i])})
    comparison = pd.DataFrame(rows).sort_values("cv_rmse_mean_mpg").reset_index(drop=True)
    chosen_name = comparison.iloc[0]["model"]
    selected = make_pipeline(clone(candidates[chosen_name])).fit(X_train, y_train)
    baseline = make_pipeline(DummyRegressor(strategy="mean")).fit(X_train, y_train)
    # The selected configuration is now frozen. Only it and the reference baseline
    # are evaluated on the holdout; other candidates are not selected by test score.
    predictions = selected.predict(X_test)
    baseline_predictions = baseline.predict(X_test)
    test_metrics = metric_dict(y_test, predictions)
    baseline_metrics = metric_dict(y_test, baseline_predictions)
    split_table = df[["source_row", "car_name"]].copy()
    split_table["split"] = "train"
    split_table.loc[test_ids, "split"] = "test"
    split_table.to_csv(reports / "split_assignments.csv", index=False)
    df.to_csv(processed / "auto_mpg.csv", index=False)
    pd.DataFrame(fold_details).to_csv(reports / "cv_split_checks.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(reports / "cv_fold_metrics.csv", index=False)
    comparison.to_csv(reports / "model_comparison.csv", index=False)
    predicted = df.iloc[test_ids][["source_row", "car_name", "origin", TARGET]].copy()
    predicted["predicted_mpg"] = predictions
    predicted["baseline_mpg"] = baseline_predictions
    predicted["residual_mpg"] = y_test.to_numpy() - predictions
    predicted.to_csv(reports / "test_predictions.csv", index=False)
    # Descriptive post-selection diagnostics only: never used to change the model.
    importance = permutation_importance(selected, X_test, y_test, n_repeats=15,
                                        scoring="neg_root_mean_squared_error", random_state=SEED, n_jobs=1)
    importance_table = pd.DataFrame({"feature": FEATURES,
                                    "rmse_increase_mean_mpg": importance.importances_mean,
                                    "rmse_increase_std_mpg": importance.importances_std}).sort_values("rmse_increase_mean_mpg")
    importance_table.to_csv(reports / "permutation_importance.csv", index=False)
    group_error_rows = []
    for origin, subset in predicted.groupby("origin"):
        group_error_rows.append({"origin": origin, "test_rows": len(subset),
                                 "mae_mpg": float(subset.residual_mpg.abs().mean())})
    group_errors = pd.DataFrame(group_error_rows)
    group_errors.to_csv(reports / "errors_by_origin.csv", index=False)
    metadata = {
        "selected_model": chosen_name, "selection_metric": "lowest mean 5-fold group CV RMSE on training rows",
        "seed": SEED, "rows": len(df), "train_rows": len(train_ids), "test_rows": len(test_ids),
        "train_groups": int(train_groups.nunique()), "test_groups": int(df.iloc[test_ids].car_name.nunique()),
        "group_key": "exact whitespace-normalized lowercase car name", "train_test_group_overlap": 0,
        "cv_folds": 5, "test_group_fraction_requested": .2,
        "missing_horsepower": int(df.horsepower.isna().sum()),
        "exact_duplicate_rows": int(df.drop(columns="source_row").duplicated().sum()),
        "test_metrics": test_metrics, "test_baseline_metrics": baseline_metrics,
        "features": FEATURES, "excluded": ["source_row", "car_name", TARGET],
        "numeric_training_ranges": {c: {"min": float(X_train[c].min()), "max": float(X_train[c].max())} for c in NUMERIC},
        "raw_sha256": sha256(RAW_PATH),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "model_trained_on": "training partition only; not refitted on holdout",
    }
    save_json(metadata, reports / "metrics.json")
    save_json(metadata, models_dir / "model_metadata.json")
    joblib.dump({"pipeline": selected, "metadata": metadata}, models_dir / "auto_mpg_pipeline.joblib", compress=3)
    reloaded = joblib.load(models_dir / "auto_mpg_pipeline.joblib")
    np.testing.assert_allclose(reloaded["pipeline"].predict(X_test), predictions, rtol=1e-12, atol=1e-12)
    set_style()
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    view = comparison.iloc[::-1]
    ax.barh(view.model, view.cv_rmse_mean_mpg, xerr=view.cv_rmse_std_mpg,
            color=[TEAL if n == chosen_name else "#9faebd" for n in view.model], capsize=4)
    ax.set(title="Model selection uses training folds only", xlabel="Mean cross-validation RMSE (mpg); lower is better")
    ax.text(.01, -.2, "Error bars: sample standard deviation across 5 group folds, not confidence intervals.",
            transform=ax.transAxes, fontsize=9)
    save_figure(fig, reports / "figures/01_model_comparison.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    axes[0].scatter(y_test, predictions, color=TEAL, alpha=.7, s=30)
    bounds = [float(min(y_test.min(), predictions.min())-1), float(max(y_test.max(), predictions.max())+1)]
    axes[0].plot(bounds, bounds, "--", color=NAVY, linewidth=1)
    axes[0].set(title="Untouched holdout: actual vs predicted", xlabel="Actual fuel efficiency (mpg)",
                ylabel="Predicted fuel efficiency (mpg)", xlim=bounds, ylim=bounds)
    axes[1].scatter(predictions, predicted.residual_mpg, color=ORANGE, alpha=.7, s=30)
    axes[1].axhline(0, color=NAVY, linestyle="--", linewidth=1)
    axes[1].set(title="Residuals reveal error patterns", xlabel="Predicted fuel efficiency (mpg)",
                ylabel="Actual minus predicted (mpg)")
    save_figure(fig, reports / "figures/02_holdout_diagnostics.png")
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.barh(importance_table.feature.str.replace("_", " "), importance_table.rmse_increase_mean_mpg,
            xerr=importance_table.rmse_increase_std_mpg, color=TEAL, capsize=3)
    ax.axvline(0, color=NAVY, linewidth=.8)
    ax.set(title="Permutation importance: selected model", xlabel="Holdout RMSE increase when shuffled (mpg)")
    save_figure(fig, reports / "figures/03_permutation_importance.png")
    reduction = 100 * (1-test_metrics["rmse_mpg"]/baseline_metrics["rmse_mpg"])
    test_table = pd.DataFrame([{"model": chosen_name, **test_metrics}, {"model": "Dummy mean", **baseline_metrics}])
    content = f"""# Auto MPG: regression results

## Problem

Estimate historical city-cycle fuel efficiency (mpg) from seven vehicle
attributes. `mpg` is a continuous target, so this is regression, not classification.
Source: [UCI Auto MPG](https://archive.ics.uci.edu/dataset/9/auto+mpg).
See [data dictionary](../docs/data_dictionary.md) for feature units and coding.

## Evaluation protocol

- {len(df)} source records; {metadata['missing_horsepower']} missing horsepower values are retained.
- Split by exact normalized car name with seed 42: {len(train_ids)} training rows
  across {metadata['train_groups']} names; {len(test_ids)} test rows across {metadata['test_groups']} names.
- Test size is 20% of names, not necessarily 20% of rows. Name overlap is zero.
- Five-fold GroupKFold runs exclusively on training records with zero name overlap
  in every fold. Related models with different names may still be correlated.
- Median imputation, scaling, and categorical encoding fit inside each training fold.
- Fixed model configurations are compared by mean CV RMSE. No test-based selection.
- `car_name`, `source_row`, and the target are excluded from predictors.

## Cross-validation model comparison

{markdown_table(comparison)}

The selected model is **{chosen_name}**. Fold standard deviations describe fold
variation; they are not confidence intervals. These CV scores are used for selection
and may be optimistic after comparing multiple candidates.

## Final holdout evaluation

{markdown_table(test_table)}

The selected model reduces holdout RMSE by {reduction:.1f}% relative to a constant
training-mean prediction. R² is not a classification accuracy percentage.
MAE and RMSE use mpg; MSE uses squared mpg. These are measured outputs of this run,
not illustrative scores or a guarantee of future performance.

## Diagnostic subgroup errors

{markdown_table(group_errors)}

Subgroups can be small. These values are descriptive and were not used to retune
the model. Residuals and permutation importance are also post-selection diagnostics.
Correlated predictors can share importance; importance is not causal influence.

## Limitations and next steps

- Historical 1970–1982 vehicles only. Do not use this model for modern vehicles,
  EVs, safety decisions, regulatory fuel-economy claims, or price prediction.
- A grouped random split evaluates held-out names from this historical data, not
  forecasting a future year. A time-ordered evaluation is a separate next step.
- Exact-name grouping cannot resolve typos, shared manufacturers, or model families.
- One small holdout has uncertainty. Evaluate additional external data before deployment.
- No prediction intervals are provided. Out-of-range inputs and unseen origins need review.
- The saved pipeline is fitted on training records only, and reload predictions were
  checked against the original fit. Load joblib files only from trusted sources.

## Figures

![CV comparison](figures/01_model_comparison.png)
![Holdout diagnostics](figures/02_holdout_diagnostics.png)
![Permutation importance](figures/03_permutation_importance.png)
"""
    (reports / "final_report.md").write_text(content, encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT)
    result = train(parser.parse_args().output)
    print(result["selected_model"], result["test_metrics"])
