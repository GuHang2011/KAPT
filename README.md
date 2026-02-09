# KAPT: Knowledge-Aware Prompt Tuning

**KAPT** is a deep learning framework designed for molecular property prediction. It introduces a novel **Knowledge-Aware Prompt Tuning** mechanism to enhance the representation capability of graph neural networks (CMPNN) for downstream tasks.

Environment Setup

To reproduce the experiments, please set up the environment using the provided `environment.yaml` file.

```bash
# 1. Create the conda environment
conda env create -f environment.yaml

# 2. Activate the environment
conda activate kapt

```

> **Note:** Please ensure the pre-trained graph encoder weights are placed correctly at:
> `./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl`

## Usage & Testing Commands

Below are the scripts to run KAPT on benchmark datasets.

### Classification Tasks

#### 1. BBBP Dataset

Standard classification test with fixed seed.

```bash
python train.py \
    --use_kapt \
    --data_path ./data/bbbp.csv \
    --metric auc \
    --dataset_type classification \
    --epochs 100 \
    --num_runs 5 \
    --ensemble_size 5 \
    --gpu 0 \
    --batch_size 50 \
    --seed 4 \
    --init_lr 1e-4 --max_lr 1e-3 --final_lr 1e-4 \
    --warmup_epochs 2.0 \
    --split_type scaffold_balanced \
    --exp_name bbbp_kapt \
    --exp_id bbbp_kapt \
    --checkpoint_path ./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl \
    --prompt_dim 256 \
    --prompt_lr 1e-4 \
    --kapt_alpha 0.1 \
    --structure_noise_scale 0

```

#### 2. Tox21 Dataset

Multi-task classification test.

```bash
python train.py \
    --use_kapt \
    --data_path ./data/tox21.csv \
    --metric auc \
    --dataset_type classification \
    --epochs 100 \
    --num_runs 3 \
    --ensemble_size 3 \
    --gpu 0 \
    --batch_size 50 \
    --seed 43 \
    --init_lr 1e-4 --max_lr 1e-3 --final_lr 1e-4 \
    --warmup_epochs 2.0 \
    --split_type scaffold_balanced \
    --exp_name tox21_kapt \
    --exp_id tox21_kapt \
    --checkpoint_path ./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl \
    --prompt_dim 256 \
    --prompt_lr 1e-4 \
    --kapt_alpha 0.1 \
    --structure_noise_scale 0

```

#### 3. SIDER Dataset

**Auto-Seed Mode:** By setting `--seed -1`, the model generates a random seed for each run.

```bash
python train.py \
    --use_kapt \
    --kapt_alpha 0.1 \
    --data_path ./data/sider.csv \
    --dataset_type classification \
    --metric auc \
    --epochs 100 \
    --num_runs 3 \
    --ensemble_size 3 \
    --gpu 0 \
    --batch_size 50 \
    --seed -1 \
    --init_lr 1e-4 --max_lr 1e-3 --final_lr 1e-4 \
    --warmup_epochs 2.0 \
    --split_type scaffold_balanced \
    --exp_name sider_kapt \
    --exp_id sider_kapt \
    --checkpoint_path ./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl \
    --prompt_dim 256 \
    --prompt_lr 1e-4

```

### Regression Tasks (Quantum Mechanics)

#### 4. QM7 Dataset

Regression task measuring Mean Absolute Error (MAE) with multi-layer prompt injection.

```bash
python train.py \
    --use_kapt \
    --data_path ./data/qm7.csv \
    --metric mae \
    --dataset_type regression \
    --epochs 100 \
    --num_runs 3 \
    --ensemble_size 3 \
    --gpu 0 \
    --batch_size 256 \
    --seed 43 \
    --init_lr 1e-4 --max_lr 1e-3 --final_lr 1e-4 \
    --warmup_epochs 2.0 \
    --split_type scaffold_balanced \
    --split_sizes 0.8 0.1 0.1 \
    --exp_name kapt_qm7 \
    --exp_id scaffold \
    --checkpoint_path ./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl \
    --prompt_dim 128 \
    --num_prompt_tokens 10 \
    --prompt_lr 1e-3 \
    --kano_lr 1e-5 \
    --kapt_dropout 0.1 \
    --weight_decay 1e-5

```

#### 5. QM8 Dataset

Regression task for electronic spectra properties.

```bash
python train.py \
    --use_kapt \
    --data_path ./data/qm8.csv \
    --metric mae \
    --dataset_type regression \
    --epochs 100 \
    --num_runs 3 \
    --ensemble_size 3 \
    --gpu 0 \
    --batch_size 256 \
    --seed 43 \
    --init_lr 1e-4 --max_lr 1e-3 --final_lr 1e-4 \
    --warmup_epochs 2.0 \
    --split_type scaffold_balanced \
    --split_sizes 0.8 0.1 0.1 \
    --exp_name kapt_qm8 \
    --exp_id kapt_qm8 \
    --checkpoint_path ./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl \
    --prompt_dim 128 \
    --num_prompt_tokens 10 \
    --prompt_lr 1e-3 \
    --kano_lr 1e-5 \
    --kapt_dropout 0.1 \
    --weight_decay 1e-5

```

## Key Arguments

* **`--use_kapt`**: Activates the KAPT module.
* **`--seed`**: Random seed (`-1` for random).
* **`--dataset_type`**: `classification` or `regression`.
* **`--metric`**: `auc` for classification, `mae`/`rmse` for regression.
