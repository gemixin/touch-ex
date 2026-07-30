# Touch-Ex

Touch-Ex provides a reproducible classification pipeline for the [Touch-Ex dataset](https://huggingface.co/datasets/gemixin/touch-ex), a visuo-tactile dataset collected with a [DIGIT](https://digit.ml/) tactile sensor. It supports baseline CNN and pretrained visual/tactile encoders, standard and unseen-set evaluation, and saved experiment results and plots.

## Project structure

- `configs/` contains data, training, and normalisation-cache configurations
- `data/` contains dataset loading, preprocessing, splitting, and validation code
- `models/` contains model definitions, training, evaluation, and visualisation code
- `notebooks/` contains exploratory and configuration notebooks
- `scripts/` contains runnable experiment entry points

## Installation

Clone the repository:

```bash
git clone https://github.com/gemixin/touch-ex.git
cd touch-ex
```

Install the dependencies with pip:

```bash
python3 -m pip install -r requirements.txt
```

Or create the provided Conda environment:

```bash
conda env create -f environment.yml
conda activate touch-ex
```

The pipeline downloads the Touch-Ex dataset from Hugging Face when it is first used. A CUDA-enabled PyTorch installation is recommended for training.

## Running an experiment

Configure the constants at the top of [`scripts/classify.py`](scripts/classify.py), then run:

```bash
python3 -m scripts.classify
```

The main settings are:

- `MODEL_TYPES`: one or more models to train and compare
- `TARGET_LABEL`: `object`, `object_region`, `force_level`, or `motion`
- `EXPERIMENT_NAME`: a descriptive name recorded with the results
- `SEED` and `DETERMINISTIC`: reproducibility settings
- `FREEZE_BACKBONE`: train only the task-specific head of pretrained models when `True`
- `PLOT_TSNE` and `TSNE_MAX_SAMPLES`: test-set t-SNE plot settings; use `-1` to include every test example

Available model types are:

- `baseline`: a CNN trained from scratch
- `resnet18`: ImageNet-pretrained ResNet-18
- `efficientnet_b0`: ImageNet-pretrained EfficientNet-B0
- `vit_b_16`: ImageNet-pretrained ViT-B/16
- `deit_tiny`: ImageNet-pretrained DeiT-Tiny
- `t3_tiny`: T3-Tiny tactile encoder pretrained on DIGIT data

For pretrained models, `FREEZE_BACKBONE=True` retains the pretrained backbone and trains only the classifier. With `False`, the entire model is fine-tuned. The baseline is always trained end-to-end.

## Running data ablations

[`scripts/ablate.py`](scripts/ablate.py) runs one selected model over several named data filters and compares the conditions in the resulting plots. Its default conditions are all data, force levels 1–3, sliding, and rotation. Each result row keeps the same `model_type`; the applied filter is recorded in its saved `data_config`.

```bash
python3 -m scripts.ablate
```

## Configurations

`default_data_config.json` controls splitting, preprocessing, normalisation, and DataLoader settings. The experiment seed and target label are applied automatically so that they are recorded consistently.

Training uses three configurations:

- `finetuned_train_config.json` for pretrained models trained end-to-end
- `frozen_train_config.json` for pretrained models with frozen backbones
- `baseline_train_config.json` for the baseline CNN

Training configs define the optimiser, learning rate, linear warmup, cosine decay, weight decay, and epoch count. When a baseline is included in a multi-model experiment, it automatically receives its own training config while the pretrained models use the selected frozen or fine-tuned config.

## Outputs

Each run saves the best validation checkpoint for every model under:

```text
checkpoints/<target>_classify/<experiment_number>/
```

Experiment metadata, data/training configs, histories, predictions, and evaluation metrics are appended to:

```text
results/<target>_classify/experiments.parquet
```

Plots are saved under:

```text
results/<target>_classify/plots/<experiment_number>/
```

This includes training curves, confusion matrices for the standard and unseen test sets, comparison plots for multi-model runs, and optional static and interactive t-SNE plots for the standard test set.

## Citations

If you use Touch-Ex, please cite DIGIT:

**DIGIT: A Novel Design for a Low-Cost Compact High-Resolution Tactile Sensor with Application to In-Hand Manipulation**
Mike Lambeta, Po-Wei Chou, Stephen Tian, Brian Yang, Benjamin Maloon, Victoria Rose Most, Dave Stroud, Raymond Santos, Ahmad Byagowi, Gregg Kammerer, Dinesh Jayaraman, Roberto Calandra
_IEEE Robotics and Automation Letters (RA-L), vol. 5, no. 3, pp. 3838–3845, 2020_
[https://doi.org/10.1109/LRA.2020.2977257](https://doi.org/10.1109/LRA.2020.2977257)

If you use the T3-Tiny encoder, also cite:

**Transferable Tactile Transformers for Representation Learning Across Diverse Sensors and Tasks**
Jialiang Zhao, Yuxiang Ma, Lirui Wang, Edward Adelson
_Proceedings of The 8th Conference on Robot Learning, Proceedings of Machine Learning Research, vol. 270, pp. 3766–3779, 2025_
[https://proceedings.mlr.press/v270/zhao25c.html](https://proceedings.mlr.press/v270/zhao25c.html)
