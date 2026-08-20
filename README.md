# Touch-Ex

Touch-Ex provides a reproducible classification pipeline for the [Touch-Ex dataset](https://huggingface.co/datasets/gemixin/touch-ex), a visuo-tactile dataset collected with a [DIGIT](https://digit.ml/) tactile sensor. It supports baseline CNN and pretrained visual/tactile encoders, standard and unseen-set evaluation, and saved experiment results and plots.

## Project Structure

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

## Running an Experiment

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

## Running Configuration Sweeps

[`scripts/classify_sweep.py`](scripts/classify_sweep.py) runs one selected model over every combination of named data and training configuration variants. Each result row keeps the same `model_type` and records its complete `data_config` and `train_config`.

Add variants to `DATA_CONFIG_VARIANTS` and `TRAIN_CONFIG_VARIANTS` at the top of the
script. For example, two data variants and two training variants produce four runs,
named `<data_variant>__<train_variant>`. Each run applies the corresponding pair of
variant dictionaries over the base JSON configuration files.

```bash
python3 -m scripts.classify_sweep
```

## Configurations

The scripts start with a base JSON file and apply any variant overrides defined in the
script. The experiment's `SEED` and `TARGET_LABEL` take precedence over
`random_state` and `stratify_label`, respectively, so every run in an experiment uses
the same split and records it consistently.

### Data

[`configs/default_data_config.json`](configs/default_data_config.json) is the standard
starting point. [`configs/t3_data_config.json`](configs/t3_data_config.json) is the
T3-oriented alternative: it uses centre-crop evaluation, no background subtraction,
and training augmentation.

| Setting | Purpose and accepted values |
| --- | --- |
| `split_size` | Total fraction held out from training, split equally between validation and the standard test set. |
| `filtered_force_level` | Restrict samples to `"1"`, `"2"`, or `"3"`; use `null` for all force levels. |
| `filtered_motion` | Restrict samples to `"sliding"` or `"rotation"`; use `null` for both. |
| `transform_name` | Base 224×224 image preparation: `pad_224` pads then resizes; `center_crop_224` resizes then centre-crops. Used for validation, test, and training unless `random_resized_crop` is enabled. |
| `bg_path` | Path to a `.jpg` background image for subtraction, or `null` to disable it. |
| `norm_type` | `dataset` computes/loads statistics from the training split; `imagenet` uses ImageNet statistics; `null` disables normalisation. `norm_cache_path` is required unless normalisation is disabled. |
| `batch_size`, `num_workers`, `shuffle_map` | DataLoader batch size, worker count, and per-split shuffling. |
| `train_augmentations` | Training-only settings: `random_resized_crop` replaces `transform_name` with `RandomResizedCrop(224)` when `true`; `color_jitter` is a ColorJitter dictionary or `null` and runs before background subtraction; `horizontal_flip` is a probability from `0` to `1` or `null`, applied before normalisation. Validation and test data remain deterministic and unaugmented. |

The default data config disables all augmentation. The reusable
[`configs/ssvtp_color_jitter_settings.json`](configs/ssvtp_color_jitter_settings.json)
contains the SSVTP ColorJitter values.

### Training

Choose the base training file that matches the model mode:

| File | Use for |
| --- | --- |
| [`finetuned_train_config.json`](configs/finetuned_train_config.json) | Pretrained models trained end-to-end. |
| [`frozen_train_config.json`](configs/frozen_train_config.json) | Pretrained models with frozen backbones. |
| [`baseline_train_config.json`](configs/baseline_train_config.json) | The baseline CNN, which is always trained end-to-end. |

All training configs set `optimizer` (`adam`, `adamw`, or `sgd`), learning-rate
schedule (`learning_rate`, `warmup_epochs`, `warmup_start_factor`, and
`min_learning_rate`), `weight_decay`, and `num_epochs`. For SGD, `momentum` controls
the optimiser momentum. To enable early stopping, set
`early_stopping_patience` to a positive number of consecutive non-improving validation
epochs; leave it as `null` to train for every epoch. `early_stopping_min_delta` is the
minimum validation-accuracy improvement, in percentage points, that resets patience.
When a baseline is part of a multi-model experiment, it automatically uses the
baseline config while pretrained models use the selected frozen or fine-tuned config.

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

The ColorJitter settings were taken from SSVTP:

```
@misc{kerr2023selfsupervisedvisuotactilepretraininglocate,
      title={Self-Supervised Visuo-Tactile Pretraining to Locate and Follow Garment Features}, 
      author={Justin Kerr and Huang Huang and Albert Wilcox and Ryan Hoque and Jeffrey Ichnowski and Roberto Calandra and Ken Goldberg},
      year={2023},
      eprint={2209.13042},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2209.13042}, 
}
```
