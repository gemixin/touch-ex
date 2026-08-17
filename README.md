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

## Running configuration sweeps

[`scripts/classify_sweep.py`](scripts/classify_sweep.py) runs one selected model over every combination of named data and training configuration variants. Each result row keeps the same `model_type` and records its complete `data_config` and `train_config`.

Add variants to `DATA_CONFIG_VARIANTS` and `TRAIN_CONFIG_VARIANTS` at the top of the
script. For example, two data variants and two training variants produce four runs,
named `<data_variant>__<train_variant>`. Each run applies the corresponding pair of
variant dictionaries over the base JSON configuration files.

```bash
python3 -m scripts.classify_sweep
```

## Configurations

`default_data_config.json` controls splitting, preprocessing, normalisation, and DataLoader settings. The experiment seed and target label are applied automatically so that they are recorded consistently.

Training uses three configurations:

- `finetuned_train_config.json` for pretrained models trained end-to-end
- `frozen_train_config.json` for pretrained models with frozen backbones
- `baseline_train_config.json` for the baseline CNN

Training configs define the optimiser, learning rate, linear warmup, cosine decay, weight decay, and maximum epoch count. They can also enable early stopping using `early_stopping_patience`; set it to a positive number of consecutive non-improving validation epochs, or `null` to disable it. `early_stopping_min_delta` is the minimum validation-accuracy improvement (in percentage points) that resets patience. When a baseline is included in a multi-model experiment, it automatically receives its own training config while the pretrained models use the selected frozen or fine-tuned config.

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

If you use Touch-Ex in your research, please cite:

```
@dataset{mclean2026touchex,
  author = {McLean, Gemma and Hao, Zhou Daniel},
  title = {Touch-Ex: A Region-Level, Force-Annotated Visuo-Tactile Dataset},
  year = {2026}
}
```

Touch-Ex was collected using the DIGIT vision-based tactile sensor. If you use this dataset in your research, please additionally cite the original DIGIT paper:

```
@article{lambeta2020digit,
  title = {DIGIT: A Novel Design for a Low-Cost Compact High-Resolution Tactile Sensor with Application to In-Hand Manipulation},
  author = {Lambeta, Mike and Chou, Po-Wei and Tian, Stephen and Yang, Brian and Maloon, Benjamin and Most, Victoria Rose and Stroud, Dave and Santos, Raymond and Byagowi, Ahmad and Kammerer, Gregg and Jayaraman, Dinesh and Calandra, Roberto},
  journal = {IEEE Robotics and Automation Letters},
  volume = {5},
  number = {3},
  pages = {3838--3845},
  year = {2020},
  doi = {10.1109/LRA.2020.2977257}
}
```

If you use the T3-Tiny encoder, also cite:

```
@article{zhao2024transferable,
  title={Transferable Tactile Transformers for Representation Learning Across Diverse Sensors and Tasks}, 
  author={Jialiang Zhao and Yuxiang Ma and Lirui Wang and Edward H. Adelson},
  year={2024},
  eprint={2406.13640},
  archivePrefix={arXiv},
}
```
