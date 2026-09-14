# Assignment scope and deliverables

## Confirmed scope

The selected track is **Data Science — Intermediate**. These two deliverables
follow the titles visible in the supplied Data Science dashboard screenshot:

1. Comprehensive Exploratory Data Analysis (EDA) on Kaggle Dataset.
2. End-to-End Predictive Regression Pipeline with Model Metrics.

The optional customer-segmentation project is not included. A different handbook
discussed earlier used Customer Churn / Retail Demand titles. That handbook is not
the specification used here. The full official domain PDF was not available for
verification in this implementation. Therefore these are implementations of the
confirmed screenshot titles, not a claim that an unseen rubric has been satisfied.

## Acceptance mapping

| Requirement | Implementation/evidence |
| --- | --- |
| Kaggle dataset EDA | Original Kaggle CSV, source links, analysis notebook |
| Dataset understanding | Per-project data dictionaries and explicit questions |
| Cleaning and quality checks | Validation functions, audit JSON, hashes, unit tests |
| Statistical exploration | Distributions, group summaries, correlations, IQR review, bootstrap intervals |
| Evidence-based conclusions | Generated Markdown reports tied to measured tables |
| End-to-end regression | Input validation, split, preprocessing, CV, selection, fitting, metrics, persistence, prediction CLI |
| Proper evaluation | Group-disjoint holdout and GroupKFold; preprocessing in Pipeline |
| Model metrics | MAE, MSE, RMSE, R²; baseline and CV comparison |
| Documentation | Root README, task READMEs, setup, beginner walkthrough, demo guide |
| Reproducibility | Included data, pinned dependencies, fixed seed, run scripts |
| Notebook delivery | Two valid Python notebooks with executed cell outputs |
| Repository hygiene | Existing `Data Analysis/` files preserved; credentials and binary model excluded |

## Decisions

- Use two independent datasets so Task 1's full-data exploration cannot influence
  the held-out labels in Task 2.
- Use a compact, real historical dataset so learners can run all models on a CPU.
- Treat exact normalized car names as groups because names repeat in Auto MPG.
- Fix candidate hyperparameters before holdout evaluation. Select by training CV
  only and evaluate the selected configuration plus a naive baseline on the holdout.
- Use built-in `unittest` for checks so no additional testing dependency is needed.
- Keep model binaries local and reproducible. Never ask users to load untrusted pickle files.
- Preserve the user's existing examples, even where their original paths or style
  differ from the new projects. They are outside the new test/requirements scope.

## Before submitting

Compare the current official PDF with this scope. Confirm whether the internship
requires screenshot-title projects or handbook-specific projects. Review and
understand the notebooks, run them on your machine, and prepare your own explanation.
Follow any internship policy on assistance and attribution. This repository is
not a claim of certificate eligibility, internship completion, or official approval.
