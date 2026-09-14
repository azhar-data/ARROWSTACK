# Demonstration guide

Use this outline for a short walkthrough or screen recording. It is a guide,
not evidence that a recording or LinkedIn post has already been produced.

## 1. Introduce the repository (30 seconds)

State the selected Data Science task titles and show the README project table.
Explain that one project explores data and the other trains a regression model.
Point to data attribution and the scope note.

## 2. Demonstrate EDA (2 minutes)

- Open the EDA notebook and show the original shape and missing-value checks.
- Explain the literal `.` sex marker and why missing measurements were not imputed.
- Show the species/mass chart and its group sample sizes.
- Compare pooled and within-species correlations.
- Show species/island counts and explain confounding.
- End with one limitation and one next-step recommendation.

## 3. Demonstrate regression (3 minutes)

- Name the seven `X` columns and continuous `mpg` target `y`.
- Explain repeated-name grouping and show the zero-overlap checks.
- Show the complete preprocessing/model Pipeline.
- Show all five CV candidates and the selection rule.
- Explain why the holdout did not determine model choice.
- Show MAE, RMSE, R², the baseline, and residual plot.
- Demonstrate prediction on the illustrative CSV with a missing horsepower value.

## 4. Demonstrate reproducibility (1 minute)

Run `python -m unittest discover -s tests -v`. Show that raw datasets, source
hashes, reports, executed notebooks, and dependency pins are included. Explain
that the model binary is regenerated locally, not downloaded from an unknown source.

## Questions to be ready for

1. Why did you preserve missing values in EDA but impute them for modeling?
2. What is the difference between a training fold, validation fold, and final test set?
3. Why is a random row split weaker when names repeat?
4. Why is `car_name` excluded from `X`?
5. Why is R² not classification accuracy?
6. Why does higher permutation importance not prove causation?
7. What additional validation would be needed for future or modern vehicles?
8. Which official task specification does this repository implement?

Before submitting or posting, compare the official PDF and follow the program's
requirements for attribution and disclosure. No social post is created or sent by
this project, and no internship certificate is claimed.
