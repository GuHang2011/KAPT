# Reviewer Entry / 阅读入口

## Question

Can a prompt-learning experiment expose its tensor contracts, stopping decisions and evaluation inputs clearly enough for another reader to check?

这是工程整理与模型自学项目。它展示如何把实验脚本中的含糊行为改为可检查的接口，不能替代完整分子模型的训练复现。

## A Five-Minute Reading Path

| Read | What to check |
| --- | --- |
| [Source and changes](docs/source-and-changes.md) | Which ideas came from KANO and the local workspace; what this release actually changes |
| [prompts.py](src/kapt_lab/prompts.py) | Node/task shapes, pooling, residual injection and device/dtype rules |
| [early_stopping.py](src/kapt_lab/early_stopping.py) | Direction, strict improvement, patience, best epoch and stopped state |
| [ensemble.py](src/kapt_lab/ensemble.py) | ID alignment, positive-class probabilities and label agreement |
| [Tests](tests) | Numerical examples, backward propagation and rejected malformed inputs |
| [Reproduction](docs/reproduction.md) | Exact commands and the tested environment |

The [online case](https://guhang2011.github.io/cases/kapt/) includes a synthetic early-stopping replay. Code and tests are the primary evidence; the animation is a visual explanation.

## What Has Been Changed

- Separated reusable tools from the legacy Chemprop/RDKit training pipeline.
- Replaced a checkpoint-inference placeholder with a functioning CSV prediction ensemble.
- Made early-stopping configuration and state explicit.
- Unified the task prompt, node-feature pooling, fusion and residual injection into one composable Tensor interface.
- Added small fixtures and checks instead of copying local performance reports.

## What This Does Not Establish

- No molecular dataset or KANO checkpoint is included.
- The extracted graph pooling does not use graph edges.
- No molecular accuracy, AUC improvement, novelty or paper acceptance is claimed.
- CPU component tests do not verify GPU performance or a complete Chemprop adapter.
- The prediction CLI accepts prepared probabilities; it does not run a model or load a pickle file.

## Next Research Exercise

Before adding a molecular benchmark, define the split, model-selection rule and budget; verify that the checkpoint loader reconstructs the same model; check probability outputs; freeze the protocol; and evaluate on the test split once. A useful report would include the baseline, identical seeds/splits, ablations, uncertainty and failure examples.
