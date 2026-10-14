"""Command-line access to CSV ensembling and validation-only early stopping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .early_stopping import EarlyStopping
from .ensemble import (
    ensemble_predictions, evaluate_ensemble, normalized_weights,
    read_label_csv, read_prediction_csv, write_prediction_csv,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auditable KAPT study utilities; no checkpoint inference")
    commands = parser.add_subparsers(dest="command", required=True)
    ensemble = commands.add_parser("ensemble", help="combine positive-class probabilities by sample_id")
    ensemble.add_argument("--predictions", nargs="+", required=True, type=Path)
    ensemble.add_argument("--weights", nargs="+", type=float)
    ensemble.add_argument("--labels", type=Path)
    ensemble.add_argument("--threshold", type=float, default=0.5)
    ensemble.add_argument("--output", type=Path, required=True)
    ensemble.add_argument("--report", type=Path)
    stopping = commands.add_parser("early-stop", help="inspect a sequence of validation scores")
    stopping.add_argument("--scores", nargs="+", required=True, type=float)
    stopping.add_argument("--mode", choices=("min", "max"), default="max")
    stopping.add_argument("--patience", type=int, default=3)
    stopping.add_argument("--min-delta", type=float, default=0.0)
    args = parser.parse_args(argv)
    try:
        if args.command == "early-stop":
            tracker = EarlyStopping(args.patience, args.min_delta, args.mode)
            events = []
            for epoch, score in enumerate(args.scores):
                tracker(score, epoch)
                events.append({"epoch": epoch, "score": score, "best_epoch": tracker.best_epoch,
                               "best_score": tracker.best_score, "bad_epochs": tracker.bad_epochs,
                               "stopped": tracker.stopped})
                if tracker.stopped:
                    break
            report = {"source": "user-supplied validation scores", "events": events}
        else:
            tables = [read_prediction_csv(path) for path in args.predictions]
            combined = ensemble_predictions(tables, args.weights)
            # Validate the threshold even if labels were not supplied.
            if not 0 <= args.threshold <= 1:
                raise ValueError("threshold must be a finite number in [0, 1]")
            report = {"sample_count": len(combined), "model_count": len(tables),
                      "normalized_weights": normalized_weights(len(tables), args.weights),
                      "threshold": args.threshold, "evaluation": "no labels supplied; metrics not computed"}
            if args.labels is not None:
                report = evaluate_ensemble(tables, read_label_csv(args.labels), args.weights, args.threshold)
            destinations = [path.resolve() for path in [args.output, args.report] if path is not None]
            inputs = [path.resolve() for path in args.predictions]
            if args.labels is not None:
                inputs.append(args.labels.resolve())
            if len(set(destinations)) != len(destinations) or set(destinations) & set(inputs):
                raise ValueError("output/report paths must be distinct and cannot overwrite inputs")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.report is not None:
                args.report.parent.mkdir(parents=True, exist_ok=True)
            write_prediction_csv(args.output, combined)
        text = json.dumps(report, indent=2, allow_nan=False)
        if args.command == "ensemble" and args.report is not None:
            args.report.write_text(text + "\n", encoding="utf-8")
        print(text)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
