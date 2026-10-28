# Demonstration guide

Use this outline for a short walkthrough or screen recording. It is a guide, not
evidence that a recording or social post has already been produced.

## 1. Introduce the repository (30 seconds)

Show the README project table and Python entry points. Explain that one project
explores data and the other trains a regression model. Point to data attribution
and the assignment-scope note.

## 2. Demonstrate EDA (2 minutes)

- Open `task1_eda/analysis.py` and explain the original shape and missing-value checks.
- Explain the nonstandard `.` sex marker and why missing measurements were not imputed.
- Show the species/mass chart and sample sizes.
- Compare pooled and within-species correlations.
- Explain why species and island effects may be confounded.
- End with one limitation and one next-step recommendation.

## 3. Demonstrate regression (3 minutes)

- Name the seven `X` columns and continuous `mpg` target `y`.
- Explain repeated-name grouping and the zero-overlap checks.
- Show the preprocessing/model Pipeline in `task2_regression/train.py`.
- Explain the five CV candidates and selection rule.
- Show MAE, RMSE, R², the baseline, and residual plot.
- Demonstrate prediction on the illustrative CSV with missing horsepower.

## 4. Demonstrate reproducibility (1 minute)

Run `python -m unittest discover -s tests -v` and
`python scripts/verify_deliverables.py`. Show that source data, Python files,
reports, figures, tests, and dependency pins are included. Explain that the model
binary is regenerated locally, not downloaded from an unknown source.

## Questions to be ready for

1. Why preserve missing values in EDA but impute them for modeling?
2. What is the difference between training, validation, and final test data?
3. Why is a random row split weaker when names repeat?
4. Why is `car_name` excluded from `X`?
5. Why is R² not classification accuracy?
6. Why does permutation importance not prove causation?
7. What validation would be needed for modern vehicles?
8. Which official task specification does the repository implement?

Before submitting or posting, compare the official PDF and follow the program's
requirements for attribution and disclosure. No social post is created or sent by
this project, and no internship certificate is claimed.
