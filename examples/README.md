# Synthetic prediction fixtures

These eight invented IDs, labels, and probabilities are small teaching fixtures.
They do not represent molecules, a trained model, a benchmark, or KANO/KAPT results.
The three model tables disagree and have different row orders to exercise identity
alignment. Each probability means `P(label=1)` for the same binary task and examples.

After installing the package from the repository root:

```console
python -m kapt_lab ensemble --predictions examples/model_a.csv examples/model_b.csv examples/model_c.csv --weights 1 1 2 --labels examples/labels.csv --output output/ensemble.csv --report output/report.json
python -m kapt_lab early-stop --scores 0.70 0.73 0.75 0.74 0.74 0.73 --patience 3 --mode max --min-delta 0.001
```

For ensemble output, rows follow the first model's order. Labels are aligned by ID,
not row position. `probability >= threshold` predicts class 1; the default threshold
is 0.5. A label file is optional, and without labels no metrics are claimed.
Output and report parent directories are created automatically; neither destination
can overwrite an input file, and the two destinations must be distinct.

`ensemble_metrics` evaluates averaged probabilities. `mean_individual_metrics`
is the unweighted arithmetic mean of each model's evaluated metrics; it is a
separate descriptive statistic and is not ensemble performance. AUC and average
precision are `null` if the labels contain only one class. Precision, recall and
F1 use zero when their denominator is zero. Log loss uses scikit-learn's clipping
of endpoint probabilities.

In a real experiment, models must share the exact held-out sample IDs, label meaning,
and split. Select weights and thresholds using training/validation data, then fix
them before the held-out evaluation. Sample identity equality alone cannot establish
that experiment protocol. These utilities do not load checkpoints or run inference.
