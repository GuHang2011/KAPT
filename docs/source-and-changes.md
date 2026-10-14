# Source and Engineering Changes

Updated: 2026-10-14

## Provenance

The source workspace was named `KAPT-main`. Its README identified the upstream implementation as [KANO by ZJU-Fangyin](https://github.com/ZJU-Fangyin/KANO), with an MIT license carrying `Copyright (c) 2022 Fangyin`. This public repository preserves that notice. The source workspace did not contain Git history, so the authorship of its added KAPT files cannot be independently established here.

KANO and its paper remain work of the original authors. This repository is an independently curated learning package; it is not an official release or evidence of a new published molecular model.

## Revision Map

| Local source material | Finding | Public engineering revision |
| --- | --- | --- |
| `kapt_components.py` | Implicit dimensions, unused refiner, graph pooling without edges | One explicit Tensor interface; shape/task checks; mean-feature naming; forward/backward tests |
| `early_stopping.py` | Invalid modes and nonfinite scores had unclear behavior; stopped state could change | Validated configuration, explicit best epoch, strict ordering, sticky stop and reset |
| `predict_ensemble.py` | Checkpoint model reconstruction and prediction loop were unfinished | Real CSV probability input, exact sample-ID alignment, weighted mean and evaluation |
| `collect_results.py` | Averaging per-run metrics was presented as prediction ensemble | Separate individual metric summary from metrics computed on averaged predictions |
| Legacy training scripts | Competing model paths, incomplete custom-model wiring and different checkpoint reconstruction | Training scripts excluded; component scope made explicit |

The revision unifies and rewrites the small interfaces. It does not attempt to fix or redistribute all of Chemprop, KANO or the original training environment.

## Excluded Material

- Original `data/`, `dumped/`, `ckpt/`, `initial/` and `logs/`.
- The legacy `chemprop/` and `KGembedding/` trees, which include other third-party dependencies and licenses.
- Serialized models and parameter files (`.pt`, `.pth`, `.pkl`, `.ckpt`).
- PDF/PPT paper attachments, copied overview figures, previous result screenshots and optimization claims.
- IDE settings, environment inventories and machine-specific paths.

The published CSV fixtures are newly assembled synthetic examples. No original sample or learned weight is needed to run them.

## Attribution

- Upstream method and official implementation: [KANO](https://github.com/ZJU-Fangyin/KANO).
- Public curation, documentation and the engineering revisions listed above: this repository's 2026-10-14 release.
- Numerical metrics: [scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html).
- Tensor operations and autograd: [PyTorch](https://pytorch.org/docs/stable/index.html).
