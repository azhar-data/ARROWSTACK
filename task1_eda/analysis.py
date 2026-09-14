"""Run: python -m task1_eda.analysis [--output DIRECTORY]."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from project_utils import ROOT, COLORS, NAVY, TEAL, markdown_table, save_figure, save_json, set_style, sha256

PROJECT = ROOT / "task1_eda"
RAW_PATH = PROJECT / "data/raw/penguins_size.csv"
NUMERIC = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL = ["species", "island", "sex"]
RENAME = {"culmen_length_mm": "bill_length_mm", "culmen_depth_mm": "bill_depth_mm"}
SPECIES = ["Adelie", "Chinstrap", "Gentoo"]


def load_data(path=RAW_PATH):
    return pd.read_csv(path)


def clean_data(raw):
    """Normalize known missing tokens, validate schema, and never impute EDA values.

    Identical measurements are not proof of duplicate penguins (no unique ID in
    this simplified file). They are reported, not automatically removed.
    """
    df = raw.copy(deep=True).rename(columns=RENAME)
    required = set(NUMERIC + CATEGORICAL)
    if set(df.columns) != required:
        raise ValueError(f"Expected exactly these columns: {sorted(required)}")
    duplicate_count = int(df.duplicated().sum())
    converted = {}
    for column in CATEGORICAL:
        values = df[column].astype("string").str.strip()
        invalid = values.isin(["", ".", "NA", "N/A", "?"])
        converted[column] = int(invalid.sum())
        values = values.mask(invalid)
        df[column] = values.str.lower() if column == "sex" else values.str.title()
    allowed = {"species": set(SPECIES), "island": {"Biscoe", "Dream", "Torgersen"},
               "sex": {"male", "female"}}
    for column, levels in allowed.items():
        unknown = set(df[column].dropna()) - levels
        if unknown:
            raise ValueError(f"Unexpected {column} values: {sorted(unknown)}")
    for column in NUMERIC:
        df[column] = pd.to_numeric(df[column], errors="raise")
        present = df[column].dropna()
        if not np.isfinite(present).all() or (present <= 0).any():
            raise ValueError(f"{column} must contain positive, finite measurements or missing values")
    if df[["species", "island"]].isna().any().any():
        raise ValueError("Species and island are required for the grouped analysis")
    df.insert(0, "source_row", np.arange(1, len(df) + 1))
    audit = {
        "raw_rows": len(raw), "clean_rows": len(df), "rows_removed": 0,
        "exact_duplicate_measurement_rows": duplicate_count,
        "known_missing_tokens_normalized": converted,
        "missing_before": {k: int(v) for k, v in raw.isna().sum().items()},
        "missing_after": {k: int(v) for k, v in df.isna().sum().items()},
        "complete_case_rows": int(df[NUMERIC + CATEGORICAL].notna().all(axis=1).sum()),
        "imputed_values": 0,
    }
    return df, audit


def bootstrap_mean_ci(values, seed=42, repetitions=2000):
    values = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [.025, .975]))


def summarize(df):
    summary = df.groupby("species", observed=True).agg(
        rows=("source_row", "size"), measured_mass=("body_mass_g", "count"),
        mean_mass_g=("body_mass_g", "mean"), median_mass_g=("body_mass_g", "median"),
        mean_flipper_mm=("flipper_length_mm", "mean"),
    ).reindex(SPECIES)
    intervals = [bootstrap_mean_ci(df.loc[df.species == s, "body_mass_g"]) for s in SPECIES]
    summary["mass_ci_low_g"] = [x[0] for x in intervals]
    summary["mass_ci_high_g"] = [x[1] for x in intervals]
    return summary


def correlations(df):
    rows = []
    for name, subset in [("All species", df)] + [(s, df[df.species == s]) for s in SPECIES]:
        complete = subset[["flipper_length_mm", "body_mass_g"]].dropna()
        rows.append({"group": name, "paired_n": len(complete),
                     "pearson_r": complete.corr().iloc[0, 1]})
    return pd.DataFrame(rows)


def outlier_review(df):
    """Within-species 1.5-IQR flags: review signals, never automatic deletion."""
    rows = []
    for species in SPECIES:
        subset = df.loc[df.species == species]
        for column in NUMERIC:
            q1, q3 = subset[column].quantile([.25, .75])
            low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            flagged = subset[column].lt(low) | subset[column].gt(high)
            rows.append({"species": species, "feature": column, "lower_fence": low,
                         "upper_fence": high, "flagged_rows": int(flagged.sum())})
    return pd.DataFrame(rows)


def make_figures(df, summary, folder):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].bar(SPECIES, summary.rows, color=COLORS, width=.65)
    axes[0].set(title="Sample composition", ylabel="Penguins (count)", ylim=(0, 175))
    for i, n in enumerate(summary.rows):
        axes[0].text(i, n + 3, str(n), ha="center", fontweight="bold")
    missing = df[NUMERIC + CATEGORICAL].isna().sum().sort_values()
    axes[1].barh(missing.index.str.replace("_", " "), missing.values, color=TEAL)
    axes[1].set(title="Missing values after cleaning", xlabel="Missing records (count)", xlim=(0, 13))
    for i, n in enumerate(missing):
        axes[1].text(n + .2, i, str(n), va="center")
    save_figure(fig, folder / "01_quality_and_composition.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    boxes = axes[0].boxplot([df.loc[df.species == s, "body_mass_g"].dropna() for s in SPECIES],
                            tick_labels=SPECIES, patch_artist=True, widths=.55)
    for patch, color in zip(boxes["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(.65)
    axes[0].set(title="Body mass varies by species", ylabel="Body mass (g)")
    for species, color in zip(SPECIES, COLORS):
        subset = df[df.species == species]
        axes[1].scatter(subset.flipper_length_mm, subset.body_mass_g, s=25,
                        color=color, alpha=.65, label=species)
    axes[1].set(title="Flipper length and body mass", xlabel="Flipper length (mm)", ylabel="Body mass (g)")
    axes[1].legend(frameon=False)
    save_figure(fig, folder / "02_species_and_measurements.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    counts = pd.crosstab(df.species, df.island).reindex(SPECIES)
    heat = axes[0].imshow(counts.values, cmap="Blues", vmin=0)
    axes[0].set(xticks=range(len(counts.columns)), xticklabels=counts.columns,
                yticks=range(3), yticklabels=SPECIES, title="Species and island are linked")
    axes[0].grid(False)
    for i in range(3):
        for j in range(len(counts.columns)):
            axes[0].text(j, i, str(counts.iloc[i, j]), ha="center", va="center",
                         color="white" if counts.iloc[i, j] > 80 else NAVY)
    fig.colorbar(heat, ax=axes[0], shrink=.8, label="Count")
    corr = df[NUMERIC].corr()
    labels = ["Bill length", "Bill depth", "Flipper length", "Body mass"]
    heat2 = axes[1].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set(xticks=range(4), xticklabels=labels, yticks=range(4), yticklabels=labels,
                title="Pooled Pearson correlations")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(False)
    for i in range(4):
        for j in range(4):
            axes[1].text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center",
                         color="white" if abs(corr.iloc[i,j]) > .6 else NAVY)
    fig.colorbar(heat2, ax=axes[1], shrink=.8, label="Pearson r")
    save_figure(fig, folder / "03_confounding_and_correlations.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    means = summary.mean_mass_g.values
    ax.errorbar(SPECIES, means, yerr=[means-summary.mass_ci_low_g, summary.mass_ci_high_g-means],
                fmt="o", capsize=6, color=TEAL, markersize=8, elinewidth=2)
    ax.set(title="Mean body mass with bootstrap 95% intervals", ylabel="Body mass (g)")
    for i, row in enumerate(summary.itertuples()):
        ax.annotate(f"n = {row.measured_mass}", (i, row.mean_mass_g), xytext=(12, 0), textcoords="offset points")
    save_figure(fig, folder / "04_bootstrap_intervals.png")


def run(output=PROJECT):
    output = Path(output)
    raw = load_data()
    df, audit = clean_data(raw)
    summary, corr, outliers = summarize(df), correlations(df), outlier_review(df)
    report = output / "reports"
    figures = report / "figures"
    processed = output / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed / "penguins_clean.csv", index=False)
    summary.to_csv(report / "species_summary.csv")
    corr.to_csv(report / "correlations.csv", index=False)
    outliers.to_csv(report / "outlier_review.csv", index=False)
    df[NUMERIC].describe().to_csv(report / "descriptive_statistics.csv")
    pd.crosstab(df.species, df.island).to_csv(report / "species_island_counts.csv")
    audit["raw_sha256"] = sha256(RAW_PATH)
    save_json(audit, report / "data_quality.json")
    make_figures(df, summary, figures)
    missing = sum(audit["missing_after"].values())
    gentoo = summary.loc["Gentoo", "mean_mass_g"]
    adelie = summary.loc["Adelie", "mean_mass_g"]
    content = f"""# Palmer Penguins: EDA findings

## Questions and scope

Which species are represented? Which measurements differ between species? How
does flipper length relate to mass within species? Can island differences be
interpreted separately from species? How sensitive are summaries to missingness?

Source: [Kaggle Palmer Archipelago dataset](https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data),
`penguins_size.csv`. See [data dictionary](../docs/data_dictionary.md) for units
and [sources](../../docs/DATA_SOURCES.md) for provenance and attribution.

## Data quality

- {len(raw)} raw records; {len(df)} retained. No values were invented or imputed.
- {audit['known_missing_tokens_normalized']['sex']} nonstandard sex token(s) normalized to missing.
- {missing} missing cells after normalization; {audit['complete_case_rows']} complete cases.
- {audit['exact_duplicate_measurement_rows']} exact repeated measurement rows found.
- Each descriptive statistic uses its own available observations. Sex-only comparisons
  would exclude unknown sex; mass summaries retain records with known mass.
- Within-species IQR flags are exported for review. No statistical outliers were deleted.

## Findings supported by the data

{markdown_table(summary.reset_index(), 1)}

1. Gentoo mean body mass is {gentoo:,.0f} g versus {adelie:,.0f} g for Adelie,
   a sample difference of {gentoo-adelie:,.0f} g. The bootstrap intervals describe
   resampling uncertainty under an independent-observation assumption, not population coverage.
2. Pooled flipper-length/body-mass correlation is {corr.iloc[0].pearson_r:.3f}.
   Within-species correlations below show why pooled relationships should not be
   interpreted as a single species-independent biological effect.
3. Gentoo observations occur only on Biscoe and Chinstrap only on Dream. Species
   and island are confounded in this sample; an island difference is not an island effect.
4. Species counts are unequal. These counts describe the supplied sample, not
   the proportion of penguins in Antarctica.

{markdown_table(corr)}

## Missingness sensitivity

The complete-case mass mean is {df.dropna(subset=NUMERIC+CATEGORICAL).body_mass_g.mean():,.1f} g;
the available-mass mean is {df.body_mass_g.mean():,.1f} g. This descriptive check
does not prove that missing values are random.

## Recommendations and limitations

- Report measurements by species and show sample sizes with uncertainty intervals.
- Review flagged measurements against original field records before changing them.
- Collect a more balanced species/island sample before estimating island effects.
- The simplified dataset has no unique individual identifier or sampling-design
  metadata. Identical measurements alone are not a safe deduplication key, and
  clustered/repeated observations could make bootstrap intervals too narrow.
- This is observational EDA, not a causal study, population census, or predictive model.
- Raw data are included and checksummed so another reviewer can reproduce every table.

## Figures

![Data quality and composition](figures/01_quality_and_composition.png)
![Measurements by species](figures/02_species_and_measurements.png)
![Confounding and correlations](figures/03_confounding_and_correlations.png)
![Uncertainty intervals](figures/04_bootstrap_intervals.png)
"""
    (report / "final_report.md").write_text(content, encoding="utf-8")
    return {"rows": len(df), "complete_cases": audit["complete_case_rows"],
            "pooled_flipper_mass_r": float(corr.iloc[0].pearson_r)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT)
    print(run(parser.parse_args().output))
