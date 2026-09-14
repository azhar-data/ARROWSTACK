"""Offline checks. Run from the root: python -m unittest discover -s tests -v"""
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from project_utils import ROOT, sha256
from task1_eda import analysis as eda
from task2_regression import train as reg
from task2_regression.predict import predict_file


class TestEDA(unittest.TestCase):
    def setUp(self):
        self.raw = eda.load_data()
        self.clean, self.audit = eda.clean_data(self.raw)

    def test_source_integrity(self):
        self.assertEqual(sha256(eda.RAW_PATH), "aa728597b2228a2637e39c6f08e40a80971f4cdac7faf7bc21ff4481ee3e3ae9")
        self.assertEqual(self.raw.shape, (344, 7))

    def test_missing_values_preserved_and_token_normalized(self):
        self.assertEqual(len(self.clean), 344)
        self.assertEqual(self.clean.sex.isna().sum(), 11)
        self.assertEqual(self.clean.body_mass_g.isna().sum(), 2)
        self.assertEqual(self.audit["complete_case_rows"], 333)
        self.assertEqual(self.audit["known_missing_tokens_normalized"]["sex"], 1)
        self.assertEqual(self.audit["imputed_values"], 0)

    def test_input_not_mutated(self):
        before = self.raw.copy(deep=True)
        eda.clean_data(self.raw)
        pd.testing.assert_frame_equal(before, self.raw)

    def test_missing_schema_rejected(self):
        with self.assertRaises(ValueError):
            eda.clean_data(self.raw.drop(columns="species"))

    def test_unrecognized_category_rejected(self):
        self.raw.loc[0, "species"] = "Invented species"
        with self.assertRaises(ValueError):
            eda.clean_data(self.raw)

    def test_invalid_measurements_rejected(self):
        for invalid in [0, -1, np.inf, "bad"]:
            copy = self.raw.astype({"body_mass_g": object})
            copy.loc[0, "body_mass_g"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                eda.clean_data(copy)

    def test_no_automatic_deduplication_of_measurements(self):
        duplicated = pd.concat([self.raw, self.raw.iloc[[0]]], ignore_index=True)
        clean, audit = eda.clean_data(duplicated)
        self.assertEqual(len(clean), 345)
        self.assertEqual(audit["exact_duplicate_measurement_rows"], 1)

    def test_summary_denominators(self):
        summary = eda.summarize(self.clean)
        self.assertEqual(summary.rows.sum(), 344)
        self.assertEqual(summary.measured_mass.sum(), 342)
        self.assertEqual(summary.loc["Adelie", "rows"], 152)
        self.assertEqual(summary.loc["Gentoo", "rows"], 124)
        self.assertEqual(summary.loc["Chinstrap", "rows"], 68)

    def test_bootstrap_reproducible(self):
        values = pd.Series([1, 2, 3, 4, np.nan])
        self.assertEqual(eda.bootstrap_mean_ci(values), eda.bootstrap_mean_ci(values))

    def test_paired_correlation_denominator(self):
        table = eda.correlations(self.clean)
        self.assertEqual(table.iloc[0].paired_n, 342)
        self.assertTrue(table.pearson_r.between(-1, 1).all())


class TestRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = reg.load_data()

    def test_source_integrity(self):
        self.assertEqual(sha256(reg.RAW_PATH), "48b830e11feee5572525f8f1691ddb9d38d3d7b7063edcd8fca672c2a5e17d8d")
        self.assertEqual(len(self.df), 398)
        self.assertEqual(self.df.horsepower.isna().sum(), 6)

    def test_features_exclude_target_and_identifiers(self):
        features = reg.prepare_features(self.df)
        self.assertEqual(features.columns.tolist(), reg.FEATURES)
        self.assertTrue(set(["mpg", "car_name", "source_row"]).isdisjoint(features.columns))

    def test_group_split_reproducible_and_disjoint(self):
        train, test = reg.group_split(self.df)
        again, _ = reg.group_split(self.df)
        np.testing.assert_array_equal(train, again)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(train) + len(test), len(self.df))
        self.assertFalse(set(self.df.iloc[train].car_name) & set(self.df.iloc[test].car_name))

    def test_cv_groups_disjoint(self):
        train, _ = reg.group_split(self.df)
        subset = self.df.iloc[train]
        for fit, valid in GroupKFold(5).split(subset, groups=subset.car_name):
            self.assertFalse(set(subset.iloc[fit].car_name) & set(subset.iloc[valid].car_name))

    def test_imputer_fits_training_only(self):
        train, test = reg.group_split(self.df)
        X = reg.prepare_features(self.df)
        model = reg.make_pipeline(LinearRegression()).fit(X.iloc[train], self.df.mpg.iloc[train])
        imputer = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"]
        np.testing.assert_allclose(imputer.statistics_, X.iloc[train][reg.NUMERIC].median().to_numpy())
        before = imputer.statistics_.copy()
        changed_test = X.iloc[test].copy()
        changed_test["horsepower"] = 100000.0
        model.predict(changed_test)
        np.testing.assert_array_equal(before, imputer.statistics_)

    def test_missing_features_and_unseen_category_predict(self):
        X = reg.prepare_features(self.df)
        model = reg.make_pipeline(LinearRegression()).fit(X.iloc[:300], self.df.mpg.iloc[:300])
        sample = X.iloc[[300, 301]].copy()
        sample.loc[:, "horsepower"] = np.nan
        sample.loc[sample.index[0], "origin"] = "unseen"
        sample.loc[sample.index[1], "origin"] = np.nan
        self.assertTrue(np.isfinite(model.predict(sample)).all())

    def test_input_schema_validation(self):
        with self.assertRaises(ValueError):
            reg.prepare_features(self.df.drop(columns="weight"))
        with self.assertRaises(ValueError):
            reg.prepare_features(self.df.iloc[:0])
        for column, invalid in [("weight", -1), ("weight", np.inf), ("horsepower", "bad"),
                                ("model_year", 2026), ("model_year", 70.5), ("cylinders", 7)]:
            sample = self.df.iloc[[0]].astype({column: object})
            sample.loc[0, column] = invalid
            with self.subTest(column=column, invalid=invalid), self.assertRaises(ValueError):
                reg.prepare_features(sample)

    def test_metrics_against_hand_calculation(self):
        result = reg.metric_dict([1, 2, 3], [1, 2, 4])
        self.assertAlmostEqual(result["mae_mpg"], 1/3)
        self.assertAlmostEqual(result["mse_mpg_squared"], 1/3)
        self.assertAlmostEqual(result["rmse_mpg"], np.sqrt(1/3))
        self.assertAlmostEqual(result["r2"], .5)
        with self.assertRaises(ValueError):
            reg.metric_dict([[1], [2]], [[1], [2]])

    def test_serialization_and_prediction_cli_function(self):
        X = reg.prepare_features(self.df)
        fitted = reg.make_pipeline(LinearRegression()).fit(X.iloc[:300], self.df.mpg.iloc[:300])
        metadata = {"numeric_training_ranges": {
            c: {"min": float(X[c].min()), "max": float(X[c].max())} for c in reg.NUMERIC}}
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            model_path, input_path, out_path = folder/"model.joblib", folder/"input.csv", folder/"predictions.csv"
            joblib.dump({"pipeline": fitted, "metadata": metadata}, model_path)
            X.iloc[:3].to_csv(input_path, index=False)
            output = predict_file(input_path, out_path, model_path)
            np.testing.assert_allclose(output.predicted_mpg, fitted.predict(X.iloc[:3]))
            with self.assertRaises(FileExistsError):
                predict_file(input_path, out_path, model_path)
            with self.assertRaises(ValueError):
                predict_file(input_path, input_path, model_path)

    def test_reported_holdout_metrics_reconcile(self):
        report_dir = reg.PROJECT / "reports"
        self.assertTrue((report_dir / "metrics.json").exists(), "Run python run_all.py first")
        metadata = json.loads((report_dir / "metrics.json").read_text())
        predictions = pd.read_csv(report_dir / "test_predictions.csv")
        actual = reg.metric_dict(predictions.mpg, predictions.predicted_mpg)
        for key, value in actual.items():
            self.assertAlmostEqual(value, metadata["test_metrics"][key], places=9)
        cv = pd.read_csv(report_dir / "model_comparison.csv")
        self.assertEqual(metadata["selected_model"], cv.iloc[0].model)
        self.assertEqual(metadata["test_rows"], len(predictions))


if __name__ == "__main__":
    unittest.main()
