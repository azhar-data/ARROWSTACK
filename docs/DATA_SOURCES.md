# Data sources, licenses, and provenance

The new projects use public research data, not generated records. The original
files are included so normal analysis/training runs require no download, API key,
Kaggle account, or network connection. Downloads were obtained on 14 September 2026.

## Task 1: Kaggle Palmer Archipelago penguins

- Dataset listing: [Palmer Archipelago (Antarctica) penguin data, Parul Pandey](https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data).
- Download: [Kaggle dataset API archive](https://www.kaggle.com/api/v1/datasets/download/parulpandey/palmer-archipelago-antarctica-penguin-data).
- Included original file: `task1_eda/data/raw/penguins_size.csv` (344 rows, 7 columns).
- This is the CSV actually extracted from Kaggle, not a synthetic substitute or a
  manually retyped version. The larger `penguins_lter.csv` archive member is not used.
- Dataset documentation and original attribution:
  [palmerpenguins](https://allisonhorst.github.io/palmerpenguins/).
- Data collected and made available by Kristen Gorman and Palmer Station LTER.
- Citation: Horst AM, Hill AP, Gorman KB (2020), *palmerpenguins: Palmer Archipelago
  (Antarctica) penguin data*. [DOI: 10.5281/zenodo.3960218](https://doi.org/10.5281/zenodo.3960218).
- Related study: Gorman KB, Williams TD, Fraser WR (2014), *Ecological Sexual
  Dimorphism and Environmental Variability within a Community of Antarctic Penguins*.
  [DOI: 10.1371/journal.pone.0090081](https://doi.org/10.1371/journal.pone.0090081).
- Data license: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/), as
  described by the upstream project. Attribution is retained as a scholarly courtesy.

Transformations: normalize sex casing, convert the literal `.` sex value to missing,
rename `culmen_*` to `bill_*`, and add `source_row` for traceability. Measurements
are not imputed, statistically trimmed, or treated as duplicate animals without IDs.
No upstream artwork or third-party notebook code is copied.

## Task 2: UCI Auto MPG

- Canonical page: [UCI Auto MPG](https://archive.ics.uci.edu/dataset/9/auto+mpg).
- Download: [UCI data archive](https://archive.ics.uci.edu/static/public/9/auto+mpg.zip).
- Included original files: `task2_regression/data/raw/auto-mpg.data` and `auto-mpg.names`.
- Citation: Quinlan, R. (1993). *Auto MPG* [Dataset]. UCI Machine Learning Repository.
  [DOI: 10.24432/C5859H](https://doi.org/10.24432/C5859H).
- Data license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- The original source is the StatLib library at Carnegie Mellon University.

Transformations: parse whitespace-delimited records, treat horsepower `?` as missing,
map origin codes 1/2/3 to usa/europe/japan, normalize car-name whitespace/case, and
add `source_row`. The published UCI 398-row version already excludes eight records
whose target was missing; this project does not invent those targets. The six
missing horsepower entries remain until training-fold median imputation.

## Integrity checks

| Original file | SHA-256 |
| --- | --- |
| `task1_eda/data/raw/penguins_size.csv` | `aa728597b2228a2637e39c6f08e40a80971f4cdac7faf7bc21ff4481ee3e3ae9` |
| `task2_regression/data/raw/auto-mpg.data` | `48b830e11feee5572525f8f1691ddb9d38d3d7b7063edcd8fca672c2a5e17d8d` |

Tests check these hashes. Analysis reports repeat the source hash for auditability.
Do not replace raw files without updating provenance, expectations, and tests.

## Technical references

- [scikit-learn: preventing data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
- [scikit-learn: cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).
- [scikit-learn: GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html).
- [scikit-learn: GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).
- [scikit-learn: model persistence](https://scikit-learn.org/stable/model_persistence.html).
