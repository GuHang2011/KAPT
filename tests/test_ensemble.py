import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from kapt_lab import (
    ensemble_predictions, evaluate_ensemble, evaluate_predictions,
    read_label_csv, read_prediction_csv, write_prediction_csv,
)
from kapt_lab.__main__ import main
from kapt_lab.ensemble import normalized_weights


class EnsembleTests(unittest.TestCase):
    def test_alignment_and_weighted_average(self):
        combined = ensemble_predictions([{"a": 0.1, "b": 0.8}, {"b": 0.4, "a": 0.9}], [1, 3])
        self.assertEqual(list(combined), ["a", "b"])
        self.assertAlmostEqual(combined["a"], 0.7)
        self.assertAlmostEqual(combined["b"], 0.5)

    def test_missing_extra_and_empty_tables(self):
        for tables in ([], [{}], [{"a": 0.1}, {}], [{"a": 0.1}, {"b": 0.1}],
                       [{"a": 0.1}, {"a": 0.1, "b": 0.2}]):
            with self.subTest(tables=tables), self.assertRaises(ValueError):
                ensemble_predictions(tables)

    def test_invalid_probabilities_ids_and_weights(self):
        for value in (-0.1, 1.1, float("nan"), float("inf"), True, "0.2", 10 ** 1000):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ensemble_predictions([{"a": value}])
        for sample in ("", " a", "a ", 1):
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                ensemble_predictions([{sample: 0.2}])
        for weights in ([], [0, 0], [1], [1, -1], [1, float("inf")], [1, float("nan")], [True, 1], [1, 10 ** 1000]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                ensemble_predictions([{"a": 0.1}, {"a": 0.9}], weights)
        self.assertEqual(normalized_weights(2, [1e308, 1e308]), [0.5, 0.5])

    def test_metrics_use_aligned_labels_and_inclusive_threshold(self):
        metrics = evaluate_predictions({"a": 0.5, "b": 0.1}, {"b": 0, "a": 1})
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["brier_score"], 0.13)
        for threshold in (-0.1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                evaluate_predictions({"a": 0.5}, {"a": 1}, threshold)

    def test_labels_are_strict_and_identical(self):
        for labels in ({"a": 2}, {"a": True}, {"a": 0.0}, {"b": 1}, {"a": 1, "b": 0}):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                evaluate_predictions({"a": 0.5}, labels)

    def test_one_class_metrics_are_explicit(self):
        metrics = evaluate_predictions({"a": 0.1, "b": 0.2}, {"a": 0, "b": 0})
        self.assertIsNone(metrics["roc_auc"])
        self.assertIsNone(metrics["average_precision"])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1"], 0.0)

    def test_model_metric_mean_is_not_ensemble_performance(self):
        report = evaluate_ensemble([{"a": 0.9, "b": 0.8}, {"b": 0.1, "a": 0.8}], {"a": 1, "b": 0})
        self.assertEqual(report["ensemble_metrics"]["accuracy"], 1.0)
        self.assertEqual(report["mean_individual_metrics"]["accuracy"], 0.75)
        self.assertEqual(report["normalized_weights"], [0.5, 0.5])

    def test_csv_roundtrip_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            write_prediction_csv(path, {"sample,quoted": 0.2, "b": 0.8})
            self.assertEqual(read_prediction_csv(path), {"sample,quoted": 0.2, "b": 0.8})
            for text in ("sample_id,probability\na,0.1\na,0.2\n", "sample_id,probability\n",
                         "sample_id,probability\na\n", "sample_id,probability\na,0.1,extra\n",
                         "id,probability\na,0.1\n", "sample_id,probability\na,nan\n"):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    read_prediction_csv(path)
            path.write_text("sample_id,label\na,1\nb,0\n", encoding="utf-8")
            self.assertEqual(read_label_csv(path), {"a": 1, "b": 0})
            path.write_text("sample_id,label\na,1.0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_label_csv(path)

    def test_cli_synthetic_fixtures(self):
        examples = Path(__file__).resolve().parents[1] / "examples"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output" / "predictions" / "ensemble.csv"
            report = Path(directory) / "output" / "reports" / "report.json"
            args = ["ensemble", "--predictions", *(str(examples / f"model_{name}.csv") for name in "abc"),
                    "--weights", "1", "1", "2", "--labels", str(examples / "labels.csv"),
                    "--output", str(output), "--report", str(report)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            result = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(result["sample_count"], 8)
            self.assertEqual(result["model_count"], 3)
            self.assertEqual(result["normalized_weights"], [0.25, 0.25, 0.5])
            self.assertEqual(result["ensemble_metrics"]["accuracy"], 1.0)
            predictions = read_prediction_csv(output)
            self.assertEqual(list(predictions)[0], "synthetic-001")
            self.assertAlmostEqual(predictions["synthetic-004"], 0.625)
            self.assertAlmostEqual(predictions["synthetic-001"], 0.4125)

    def test_cli_does_not_overwrite_inputs_or_claim_unlabeled_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            source.write_text("sample_id,probability\na,0.2\n", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(["ensemble", "--predictions", str(source), "--output", str(source)])
            self.assertEqual(error.exception.code, 2)
            rejected_output = Path(directory) / "must-not-create" / "out.csv"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(["ensemble", "--predictions", str(source), "--output", str(rejected_output),
                      "--report", str(source)])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(rejected_output.parent.exists())
            self.assertEqual(source.read_text(encoding="utf-8"), "sample_id,probability\na,0.2\n")
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                main(["ensemble", "--predictions", str(source), "--output", str(Path(directory) / "out.csv")])
            self.assertNotIn("ensemble_metrics", json.loads(stream.getvalue()))


if __name__ == "__main__":
    unittest.main()
