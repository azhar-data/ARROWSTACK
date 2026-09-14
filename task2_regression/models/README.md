# Model artifacts

Run `python -m task2_regression.train` from the repository root to create
`auto_mpg_pipeline.joblib`. The full preprocessing pipeline and selected estimator
are serialized together. `model_metadata.json` records data hash, dependency
versions, features, training ranges, split sizes, and measured metrics.

The binary model is intentionally excluded from Git to avoid version-sensitive
pickle artifacts in the source repository. Training reproduces it locally using
the included dataset and fixed seed. Use the pinned requirements for compatibility.

Only load joblib/pickle files you generated or explicitly trust: loading them can
execute code. The model is an educational historical benchmark, not a deployed service.
