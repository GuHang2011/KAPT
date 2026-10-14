# KAPT Reproducibility Lab

**Prompt-learning components, reliable early stopping, and sample-aligned prediction ensembles.**

[Case & interactive demo](https://guhang2011.github.io/cases/kapt/) · [Reviewer entry](REVIEWER_ENTRY.md) · [Source & changes](docs/source-and-changes.md) · [Study notes](docs/model-study-notes.md)

This is a small learning and engineering project curated from a local **KANO-based** workspace. [KANO](https://github.com/ZJU-Fangyin/KANO) is the upstream molecular representation-learning project, released under MIT by its authors. This repository is not the official KANO implementation and does not claim ownership of its method or paper.

本仓库整理提示学习组件与评估工具，便于阅读、运行和检查。重点是接口契约、训练控制和评估正确性；原有分子训练链尚未在此完成适配，也未复现分子基准成绩。

## What is runnable?

| Module | Behavior | Inspect |
| --- | --- | --- |
| Prompt components | Task-conditioned tokens, graph mean features, fusion, gated node residuals; CPU forward/backward example | [prompts.py](src/kapt_lab/prompts.py) |
| Early stopping | Explicit min/max rules, finite scores, ordered epochs, sticky stopping and reset | [early_stopping.py](src/kapt_lab/early_stopping.py) |
| Prediction ensemble | Align model probabilities by sample ID, validate weights, average predictions and evaluate against labels | [ensemble.py](src/kapt_lab/ensemble.py) |

The graph example pools **node features**. It does not read edges, encode molecules, or establish a structure-aware molecular model. CSV inputs contain positive-class probabilities, not logits or serialized model checkpoints.

## Quick Start

Use Python 3.10 or later; the checked local environment is Python 3.12. Run commands from the repository root.

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .
python -m kapt_lab ensemble --predictions examples/model_a.csv examples/model_b.csv examples/model_c.csv --labels examples/labels.csv --weights 1 1 2 --threshold 0.5 --output output/ensemble.csv --report output/report.json
python -m unittest discover -s tests -p test_early_stopping.py -v
python -m unittest discover -s tests -p test_ensemble.py -v
```

For the optional PyTorch components, install a wheel suitable for your machine using the [official installer](https://pytorch.org/get-started/locally/). The local CPU check uses `torch==2.6.0`:

```sh
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python examples/prompt_smoke.py
python -m unittest discover -s tests -v
```

The CSV fixtures are deliberately small and synthetic. Their different row orders exercise ID alignment. Metrics from these fixtures illustrate behavior and are not research results. See [reproduction notes](docs/reproduction.md) for the input schema and verification record. Prompt tests require the optional PyTorch dependency.

## Read the Work

1. [Reviewer entry](REVIEWER_ENTRY.md): question, implementation, verification and limits.
2. [Model study notes](docs/model-study-notes.md): tensor flow, leakage, selection and ensemble reasoning.
3. [Source and changes](docs/source-and-changes.md): upstream attribution and the scope of this engineering revision.
4. Source files and tests: small enough to read alongside each other.

## Public Scope

The release contains source, documentation, tests and synthetic CSV fixtures. It excludes original datasets, pretrained weights, checkpoints, logs, private machine settings, paper PDFs and old result screenshots. Dataset licensing and molecular benchmark reproduction are separate work.

The original local workspace has no verifiable Git history. Its existing KAPT extensions are treated as source material, not attributed as a new personal research contribution. Public curation and the engineering revisions in this repository are dated and described in [the change record](docs/source-and-changes.md).

## Upstream Reference

- [KANO source and original installation/data instructions](https://github.com/ZJU-Fangyin/KANO)
- [Original paper, Nature Machine Intelligence (2023)](https://www.nature.com/articles/s42256-023-00654-0)
- [MIT license, including the upstream copyright notice](LICENSE)

Maintained as a learning project by [Hang Gu](https://guhang2011.github.io/). Questions and reproducible bug reports can use GitHub Issues.
