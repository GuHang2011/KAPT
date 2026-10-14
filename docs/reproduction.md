# Reproduction

Run from the repository root. All CSV inputs are synthetic. No download of an original dataset or model checkpoint is needed.

## Base Environment

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .
python -m kapt_lab ensemble --predictions examples/model_a.csv examples/model_b.csv examples/model_c.csv --labels examples/labels.csv --weights 1 1 2 --threshold 0.5 --output output/ensemble.csv --report output/report.json
python -m unittest discover -s tests -p test_early_stopping.py -v
python -m unittest discover -s tests -p test_ensemble.py -v
```

`torch` is optional for CSV evaluation. Install it for prompt tests and `examples/prompt_smoke.py`, using the command in the README. The base Python 3.12 build environment did not contain PyTorch; prompt tests and the smoke example passed in the local Anaconda base environment with Python 3.8.5 and PyTorch 2.0.1+cu118. The intended clean CPU check is Python 3.12, NumPy 2.2.6, scikit-learn 1.7.2 and PyTorch 2.6.0.

## Input Contract

Prediction files contain `sample_id,probability`; the labels file contains `sample_id,label`. Each ID is unique and nonempty. All prediction files and labels must contain the same sample IDs, regardless of row order. Labels are `0` or `1`. Probability values are finite and in `[0,1]`.

Weights are finite and nonnegative; their sum is positive. The decision threshold is configured explicitly. Single-class labels cannot produce a defined ROC-AUC; read the metric output and warnings rather than replacing that situation with an invented score.

The output CSV records aligned probabilities. The JSON report records configuration, individual-model metrics and metrics computed on the ensemble. It does not reconstruct checkpoints or make predictions from SMILES.

## Local Verification Record

The release's measured environment and check results are recorded in [verification.json](verification.json). Test totals and example output will be added only after the checks have run. The GitHub Actions workflow reruns the CPU checks on clean Python environments; a local pass is separate from the status of that workflow.

## Unverified Scope

Molecular training, external datasets, GPU throughput, KANO checkpoint compatibility and molecular benchmark performance have not been verified by this release.
