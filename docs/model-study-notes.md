# Model Study Notes / 模型自学笔记

These notes explain the runnable components and a protocol for extending them. They are learning notes, not completed experimental results.

## 1. Trace the Tensors Before Naming the Model

Let node features be `X: [N, H]`, graph assignments be `batch: [N]`, and the number of graphs be `B`. Every graph must have at least one node. A node MLP `phi` maps each node to the prompt dimension; then the encoded graph mean is:

`m_b = sum(phi(X_i) for batch_i == b) / count(batch_i == b)`

A task selects learned tokens `P_t: [K, D]`; self-attention followed by token mean pooling gives a task prompt. A small network fuses `m_b` with the selected task prompt. The node update is a gated residual:

`X'_i = X_i + gate(X_i, prompt_b) * projection(prompt_b)`, where `b = batch_i`.

The output remains `[N, H]`. A shared task ID conditions the entire batch; one task ID per graph permits mixed tasks. Gradients should reach the selected tokens and the fusion/injection parameters.

这里的均值池化不读取边。同一个节点特征集合即使边连接不同，也会得到相同池化结果，因此不能把它称为完整的结构感知分子网络。接入 GNN 时，需要单独说明编码器如何利用键和邻接关系。

**Check:** inspect [the prompt tests](../tests/test_prompts.py) for unequal graph sizes, task-conditioned gradients and malformed inputs. Compare the average of encoded nodes to a manually calculated example; nonlinear encoding before averaging is different from encoding an averaged raw feature.

## 2. Early Stopping Is Part of Model Selection

Choose `min` for a loss and `max` for a score such as AUC. The first finite score establishes the best epoch. An improvement must strictly exceed `min_delta`; equality does not count. Every non-improving validation step consumes one patience unit. Once stopped, the state stays stopped until reset.

Example in `max` mode, `patience=2`, `min_delta=0`:

| Epoch | Validation score | Best epoch | Bad epochs | Stop? |
| --- | --- | --- | --- | --- |
| 1 | 0.60 | 1 | 0 | No |
| 2 | 0.64 | 2 | 0 | No |
| 3 | 0.63 | 2 | 1 | No |
| 4 | 0.64 | 2 | 2 | Yes |

Select the checkpoint from epoch 2; stopping at epoch 4 does not make epoch 4 the chosen model. Nonfinite scores should raise an error, so a numerical failure is not silently interpreted as a plateau.

**Check:** [early-stopping tests](../tests/test_early_stopping.py) and [interactive replay](https://guhang2011.github.io/cases/kapt/).

## 3. Mean Metrics and Prediction Ensembles Answer Different Questions

For sample `i`, weighted prediction averaging is:

`p_i = sum(w_j * p_ij) / sum(w_j)`.

Compute the final metrics from this `p_i`. A mean of each model's AUC is a description of individual models; it is not the AUC of the ensemble. AUC depends on ordering, and accuracy depends on the threshold, so nonlinear operations cannot be moved outside the averaging step.

Two CSVs with different row orders must be joined by `sample_id` before averaging. Equal row counts alone do not establish that two rows refer to the same example. Reject duplicate IDs, extra IDs and missing IDs. A probability must be finite and between 0 and 1; a logit must be converted by the producing model before it reaches this CLI.

阈值也属于评估协议。若需要调整阈值，应使用验证集；不能看到测试标签以后再选择使测试成绩最好的阈值。标签文件与预测文件必须对应同一批样本。

**Check:** the synthetic model files have deliberately shuffled rows. [The ensemble tests](../tests/test_ensemble.py) check ID alignment and show why individual metric summaries differ from ensemble metrics.

## 4. Test-Set Discipline

Use train data to fit model parameters, validation data to choose the checkpoint and threshold, and a held-out test set to evaluate the frozen choice. Viewing test metrics every epoch creates an opportunity to tune choices to the test set even when the code says it selects by validation.

For repeated runs, keep the evaluation sample IDs fixed before averaging predictions. A seed may affect the split as well as the optimizer. Models evaluated on different test samples cannot be combined by matching row positions.

## 5. A Checkpoint Must Reconstruct the Same Model

Before a training adapter is considered working:

1. Save both architecture configuration and state.
2. Reconstruct the exact prompt-aware architecture.
3. Load strictly, or enumerate and justify every missing/unexpected key.
4. Compare eval predictions before and after reloading on the same input.
5. Check the output contract: logits in training versus probabilities when required by evaluation.

`strict=False` without checking matched keys can report success while loading no relevant parameters. The current public package avoids model serialization; a complete molecular training adapter remains future work.

## 6. A Useful Ablation Plan

Treat the following as a future protocol, not a reported result:

| Variant | Purpose |
| --- | --- |
| Fixed encoder + task head | Establish the simplest baseline |
| Task tokens only | Isolate task conditioning |
| Mean node features only | Isolate pooled feature conditioning |
| Fused prompt | Check interaction between the two |
| Fused prompt + residual injection | Check the injection path |

Keep data splits, training budget and selection criteria identical. Record parameters, seeds, dependency versions, convergence and failure cases. Report uncertainty across runs; do not advertise a single best seed as the expected improvement.

## Reading

- [KANO source](https://github.com/ZJU-Fangyin/KANO) and [original paper](https://www.nature.com/articles/s42256-023-00654-0)
- [PyTorch autograd](https://pytorch.org/docs/stable/autograd.html)
- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
