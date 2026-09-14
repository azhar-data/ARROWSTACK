"""Predict from a CSV using the saved, trusted local training pipeline."""
import argparse
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from task2_regression.train import PROJECT, NUMERIC, prepare_features


def predict_file(input_path, output_path, model_path=PROJECT / "models/auto_mpg_pipeline.joblib"):
    input_path, output_path, model_path = map(Path, (input_path, output_path, model_path))
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output must not overwrite the input CSV")
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}. Choose a new output path.")
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run: python -m task2_regression.train")
    raw = pd.read_csv(input_path)
    features = prepare_features(raw)
    # Pickle/joblib can execute code. Only load a model you generated or explicitly trust.
    artifact = joblib.load(model_path)
    for col in NUMERIC:
        bounds = artifact["metadata"]["numeric_training_ranges"][col]
        if (features[col].lt(bounds["min"]) | features[col].gt(bounds["max"])).any():
            warnings.warn(f"{col}: values outside training range {bounds}; review predictions", stacklevel=2)
    unknown = set(features.origin.dropna()) - {"usa", "europe", "japan"}
    if unknown:
        warnings.warn(f"Unseen origins {sorted(unknown)} use an all-zero one-hot encoding; review predictions", stacklevel=2)
    predictions = artifact["pipeline"].predict(features)
    if not np.isfinite(predictions).all():
        raise ValueError("Model produced a non-finite prediction")
    result = raw.copy()
    result["predicted_mpg"] = predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=PROJECT / "models/auto_mpg_pipeline.joblib")
    args = parser.parse_args()
    print(predict_file(args.input, args.output, args.model).to_string(index=False))
