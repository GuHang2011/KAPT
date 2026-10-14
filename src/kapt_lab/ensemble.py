"""Binary probability ensembling with explicit sample identity and evaluation."""

import csv
from math import fsum, isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Mapping, Sequence


def _sample_id(value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("sample_id must be a nonempty string without outer whitespace")
    return value


def _probability(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("probability must be a finite number in [0, 1]")
    if not 0 <= value <= 1 or not isfinite(value):
        raise ValueError("probability must be a finite number in [0, 1]")
    return float(value)


def _read_csv(path: str | Path, column: str) -> dict[str, str]:
    result: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["sample_id", column]:
            raise ValueError(f"{Path(path).name}: expected CSV header sample_id,{column}")
        for line, row in enumerate(reader, start=2):
            if None in row or row[column] is None:
                raise ValueError(f"{Path(path).name}: malformed CSV row {line}")
            sample = _sample_id(row["sample_id"])
            if sample in result:
                raise ValueError(f"{Path(path).name}: duplicate sample_id at row {line}")
            result[sample] = row[column]
    if not result:
        raise ValueError(f"{Path(path).name}: CSV must contain at least one sample")
    return result


def read_prediction_csv(path: str | Path) -> dict[str, float]:
    """Read exactly ``sample_id,probability``; probability is P(label=1)."""
    raw = _read_csv(path, "probability")
    result = {}
    for sample, text in raw.items():
        try:
            result[sample] = _probability(float(text))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{Path(path).name}: invalid probability") from exc
    return result


def read_label_csv(path: str | Path) -> dict[str, int]:
    """Read exactly ``sample_id,label`` with labels 0 or 1."""
    raw = _read_csv(path, "label")
    if any(value not in ("0", "1") for value in raw.values()):
        raise ValueError(f"{Path(path).name}: labels must be literal 0 or 1")
    return {sample: int(value) for sample, value in raw.items()}


def normalized_weights(count: int, weights: Sequence[float] | None = None) -> list[float]:
    """Normalize nonnegative finite weights, requiring positive total weight."""
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
        raise ValueError("at least one prediction table is required")
    if weights is None:
        return [1.0 / count] * count
    if len(weights) != count:
        raise ValueError("number of weights must equal number of prediction tables")
    values = []
    for weight in weights:
        if isinstance(weight, bool) or not isinstance(weight, Real):
            raise ValueError("weights must be finite nonnegative numbers")
        try:
            weight = float(weight)
        except (OverflowError, ValueError) as exc:
            raise ValueError("weights must be finite nonnegative numbers") from exc
        if not isfinite(weight) or weight < 0:
            raise ValueError("weights must be finite nonnegative numbers")
        values.append(float(weight))
    scale = max(values)
    if scale == 0:
        raise ValueError("at least one weight must be positive")
    # Scaling before summing also handles large but finite user weights.
    scaled = [value / scale for value in values]
    total = fsum(scaled)
    return [value / total for value in scaled]


def _check_ids(reference: Mapping[str, object], other: Mapping[str, object], name: str) -> None:
    missing = set(reference) - set(other)
    extra = set(other) - set(reference)
    if missing or extra:
        raise ValueError(f"{name}: sample identity mismatch ({len(missing)} missing, {len(extra)} extra)")


def ensemble_predictions(
    predictions: Sequence[Mapping[str, float]], weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """Average aligned positive-class probabilities; retain the first table's order.

    All tables must represent the same examples and same positive class. Weights
    must be chosen independently of the held-out evaluation labels.
    """
    normalized = normalized_weights(len(predictions), weights)
    validated = []
    for index, table in enumerate(predictions):
        if not table:
            raise ValueError(f"prediction table {index + 1} must not be empty")
        checked = {_sample_id(sample): _probability(value) for sample, value in table.items()}
        if validated:
            _check_ids(validated[0], checked, f"prediction table {index + 1}")
        validated.append(checked)
    return {
        sample: min(1.0, max(0.0, fsum(weight * table[sample] for weight, table in zip(normalized, validated))))
        for sample in validated[0]
    }


def evaluate_predictions(
    probabilities: Mapping[str, float], labels: Mapping[str, int], threshold: float = 0.5,
) -> dict[str, float | None]:
    """Compute metrics from predictions, classifying probability >= threshold as 1.

    AUC and average precision are returned as None for a one-class label set.
    Other metrics remain defined. The threshold must be fixed without test labels.
    """
    threshold = _probability(threshold)
    checked = ensemble_predictions([probabilities])
    _check_ids(checked, labels, "labels")
    for value in labels.values():
        if isinstance(value, bool) or not isinstance(value, Integral) or value not in (0, 1):
            raise ValueError("labels must be integers 0 or 1")
    from sklearn.metrics import (
        accuracy_score, average_precision_score, brier_score_loss, f1_score,
        log_loss, precision_score, recall_score, roc_auc_score,
    )
    targets = [int(labels[sample]) for sample in checked]
    scores = list(checked.values())
    classes = [int(score >= threshold) for score in scores]
    two_classes = len(set(targets)) == 2
    return {
        "accuracy": float(accuracy_score(targets, classes)),
        "precision": float(precision_score(targets, classes, zero_division=0)),
        "recall": float(recall_score(targets, classes, zero_division=0)),
        "f1": float(f1_score(targets, classes, zero_division=0)),
        "roc_auc": float(roc_auc_score(targets, scores)) if two_classes else None,
        "average_precision": float(average_precision_score(targets, scores)) if two_classes else None,
        "log_loss": float(log_loss(targets, scores, labels=[0, 1])),
        "brier_score": float(brier_score_loss(targets, scores)),
    }


def evaluate_ensemble(
    predictions: Sequence[Mapping[str, float]], labels: Mapping[str, int],
    weights: Sequence[float] | None = None, threshold: float = 0.5,
) -> dict[str, object]:
    """Report ensemble metrics separately from the arithmetic mean of model metrics."""
    combined = ensemble_predictions(predictions, weights)
    individual = [evaluate_predictions(table, labels, threshold) for table in predictions]
    metric_mean = {
        name: fsum(float(row[name]) for row in individual) / len(individual)
        if all(row[name] is not None for row in individual) else None
        for name in individual[0]
    }
    return {
        "sample_count": len(combined),
        "model_count": len(predictions),
        "normalized_weights": normalized_weights(len(predictions), weights),
        "threshold": float(threshold),
        "ensemble_metrics": evaluate_predictions(combined, labels, threshold),
        "individual_metrics": individual,
        "mean_individual_metrics": metric_mean,
        "metric_note": "mean_individual_metrics is an unweighted descriptive mean, not ensemble performance",
        "undefined_metric_note": "roc_auc and average_precision are null when labels contain only one class",
    }


def write_prediction_csv(path: str | Path, predictions: Mapping[str, float]) -> None:
    """Write validated probabilities in their existing sample order."""
    checked = ensemble_predictions([predictions])
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "probability"])
        writer.writerows(checked.items())
