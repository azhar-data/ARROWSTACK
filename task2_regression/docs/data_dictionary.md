# Auto MPG data dictionary

One row is a vehicle record from the historical Auto MPG dataset. Names can repeat
across observations. There are 398 records in the published target-complete file.

| Column | Role | Unit/coding | Processing |
| --- | --- | --- | --- |
| `mpg` | Target `y` | City-cycle miles per gallon | Never a predictor; no target imputation |
| `cylinders` | Numeric `X` | Cylinder count: 3, 4, 5, 6, 8 | Training-fold median + scaling |
| `displacement` | Numeric `X` | Engine displacement, cubic inches | Training-fold median + scaling |
| `horsepower` | Numeric `X` | Horsepower | `?` → missing; training-fold median |
| `weight` | Numeric `X` | Vehicle weight, pounds | Training-fold median + scaling |
| `acceleration` | Numeric `X` | Acceleration time, seconds | Training-fold median + scaling |
| `model_year` | Numeric `X` | Two-digit historical model year, 70–82 | Preserve source coding, not current year |
| `origin` | Categorical `X` | 1=usa, 2=europe, 3=japan | Source code mapping; most-frequent imputation + one-hot encoding |
| `car_name` | Split group only | Lowercase, normalized vehicle name | Excluded from predictors |
| `source_row` | Audit only | One-based source record number | Excluded from predictors |

`X` has seven columns. `y` is the one-dimensional `mpg` Series. Numeric
missingness indicators are created within each fitted preprocessing pipeline.
All learned preprocessing parameters come only from the current training fold.

Prediction CSVs use textual origins, as shown in `examples/new_cars.csv`. Blank
feature cells are permitted and imputed, but every feature column must be present.
Unexpected text, nonpositive values, infinities, invalid cylinder counts, and model
years outside 70–82 are rejected. Unseen origin labels are accepted with a warning
and handled by the encoder; this does not establish model reliability for that group.

Source attribution and modifications: [DATA_SOURCES.md](../../docs/DATA_SOURCES.md).
