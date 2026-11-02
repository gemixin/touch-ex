# Touch-Ex

Touch-Ex provides reproducible classification and regression pipelines for the
[Touch-Ex dataset](https://huggingface.co/datasets/gemixin/touch-ex), a visuo-tactile
dataset collected with a [DIGIT](https://digit.ml/) tactile sensor. It supports a
baseline CNN, pretrained visual and tactile encoders, continuous force and voltage
regression, standard and unseen-set evaluation, and saved experiment results and plots.

Results from my own experiments using these pipelines are available in the
[Touch-Ex results repository](https://github.com/gemixin/touch-ex-results).

## Project Structure

- `configs/` contains data, training, and normalisation-cache configurations
- `data/` contains dataset loading, preprocessing, splitting, and validation code
- `ml/` contains experiment management, training, evaluation, visualisation, and utilities
- `models/` contains model architecture definitions
- `notebooks/` contains exploratory notebooks
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

## Running a Classification Experiment

Configure the constants at the top of [`scripts/classify.py`](scripts/classify.py), then run:

```bash
python3 -m scripts.classify
```

The main settings are:

- `MODEL_TYPES`: one or more models to train and compare
- `TARGET_LABEL`: `object`, `object_region`, `force_level`, or `motion`
- `EXPERIMENT_NAME`: a descriptive name recorded with the results
- `EXPERIMENT_PURPOSE`: `model_selection` or `final_evaluation`
- `SEED` and `DETERMINISTIC`: reproducibility settings
- `FREEZE_BACKBONE`: train only the task-specific head of pretrained models when `True`
- `PLOT_TSNE` and `TSNE_MAX_SAMPLES`: t-SNE settings for every evaluated split; use `-1` to include every example in each split

Available model types are:

- `baseline`: a CNN trained from scratch
- `resnet18`: ImageNet-pretrained ResNet-18
- `resnet50`: ImageNet-pretrained ResNet-50
- `efficientnet_b0`: ImageNet-pretrained EfficientNet-B0
- `vit_s_16`: ImageNet-pretrained ViT-S/16
- `deit_tiny`: ImageNet-pretrained DeiT-Tiny
- `t3_tiny`: T3-Tiny tactile encoder pretrained on DIGIT data

The `baseline` architecture is implemented by `BaselineCNNClassifier`. All configurable
pretrained backbones are wrapped by `PretrainedClassifier`.

For pretrained models, `FREEZE_BACKBONE=True` retains the pretrained backbone and trains only the classifier. With `False`, the entire model is fine-tuned. The baseline is always trained end-to-end.

### Running Classification Configuration Sweeps

Configure the constants at the top of
[`scripts/classify_sweep.py`](scripts/classify_sweep.py), then run:

```bash
python3 -m scripts.classify_sweep
```

The script runs one selected model over every combination of named data and training
configuration variants. Each result row keeps the same `model_type` and records its
complete `data_config` and `train_config`.

Add variants to `DATA_CONFIG_VARIANTS` and `TRAIN_CONFIG_VARIANTS` at the top of the
script. For example, two data variants and two training variants produce four runs,
named `<data_variant>__<train_variant>`. Each run applies the corresponding pair of
variant dictionaries over the base JSON configuration files. Sweeps default to
`model_selection`.

### Running Conditioned Classification

Configure the constants at the top of
[`scripts/classify_conditioned.py`](scripts/classify_conditioned.py),
then run:

```bash
python3 -m scripts.classify_conditioned
```

Set `MODEL_TYPE` to a supported pretrained backbone and `TARGET_LABEL` to either
`object` or `object_region`. The model concatenates its image representation with the
ground-truth force level, normalised to `0.0`, `0.5`, or `1.0`, before applying a
lightweight fusion layer and classification head.
`FUSION_HIDDEN_DIM` and `FUSION_DROPOUT` configure the fusion architecture. All other
data, training, backbone-freezing, and t-SNE settings follow the standard
classification pipeline.

## Running a Regression Experiment

Configure `REGRESSION_TARGET` in [`scripts/regress.py`](scripts/regress.py) as either
`force_n` or `fsr_voltage`, then run:

```bash
python3 -m scripts.regress
```

The main settings are:

- `REGRESSION_TARGET`: `force_n` or `fsr_voltage`
- `MODEL_TYPE`: any supported pretrained backbone
- `EXPERIMENT_NAME`: a descriptive name recorded with the results
- `EXPERIMENT_PURPOSE`: `model_selection` or `final_evaluation`
- `SEED` and `DETERMINISTIC`: reproducibility settings
- `FREEZE_BACKBONE`: train only the regression head when `True`
- `DATA_CONFIG_OVERRIDES` and `TRAIN_CONFIG_OVERRIDES`: values applied over the selected
  base JSON configurations

The regression workflow uses the selected pretrained backbone, optimises Huber loss on
training-split-normalised targets, and reports MAE, RMSE, and R² in the original target
units. With `FREEZE_BACKBONE=True`, the pretrained backbone remains frozen and
only the regression head is trained; with `False`, the entire model is fine-tuned.

## Configurations

The scripts start with base JSON files and apply the overrides defined in the script.
Classification sweeps apply every combination of their named data and training variants.
Classification experiments set `stratify_label` from `TARGET_LABEL`; regression
experiments retain the `stratify_label` selected in their data configuration.

### Data

[`configs/default_data_config.json`](configs/default_data_config.json) is the standard
starting point. Store the configuration selected from validation experiments in
[`configs/chosen_data_config.json`](configs/chosen_data_config.json) and set
`DATA_CONFIG_PATH` in the script to the file you want to use.

| Setting | Purpose and accepted values |
| --- | --- |
| `split_size` | Total fraction held out from training, split equally between internal validation and test. |
| `stratify_label` | Train/validation stratification for regression. Classification uses `TARGET_LABEL`. |
| `transform_name` | Base 224×224 image preparation: `pad_224` pads then resizes; `center_crop_224` resizes then centre-crops. Used for validation, test, and training unless `random_resized_crop` is enabled. |
| `bg_path` | Path to a `.jpg` background image for subtraction, or `null` to disable it. |
| `norm_type` | `dataset` computes/loads statistics from the training split; `imagenet` uses ImageNet statistics; `null` disables normalisation. `norm_cache_path` is required unless normalisation is disabled. |
| `batch_size`, `num_workers`, `shuffle_map` | DataLoader batch size, worker count, and per-split shuffling. |
| `train_augmentations` | Training-only settings: `random_resized_crop` replaces `transform_name` with `RandomResizedCrop(224)` when `true`; `color_jitter` is a ColorJitter dictionary or `null` and runs before background subtraction; `horizontal_flip` is a probability from `0` to `1` or `null`, applied before normalisation. Validation and test data remain deterministic and unaugmented. |

The default data config disables all augmentation. Reusable ColorJitter settings in
[`configs/ssvtp_color_jitter_settings.json`](configs/ssvtp_color_jitter_settings.json)
use the SSVTP values from the official paper (cited below).

### Splitting and Randomness

The source `train` split is partitioned by interaction, keeping all frames from an
interaction together. With `split_size=0.3`, the ratio is approximately 70:15:15.
Partitioning uses fixed seed **129**. The internal test partition is reserved first,
stratified by `object_region`, so it stays the same across tasks for a fixed dataset
and split size. Train/validation stratification uses the classification target or
regression's configured `stratify_label`.

The experiment `SEED` defaults to **156** and controls model initialisation, training
shuffle, and augmentation independently of dataset partitioning. It resets before each run.

### Training

Choose the base training file that matches the model mode:

| File | Use for |
| --- | --- |
| [`finetuned_train_config.json`](configs/finetuned_train_config.json) | Pretrained models trained end-to-end. |
| [`frozen_train_config.json`](configs/frozen_train_config.json) | Pretrained models with frozen backbones. |
| [`baseline_train_config.json`](configs/baseline_train_config.json) | The baseline CNN, which is always trained end-to-end. |

All training configs set `optimizer` (`adam` or `adamw`), learning-rate
schedule (`learning_rate`, `warmup_epochs`, `warmup_start_factor`, and
`min_learning_rate`), `weight_decay`, and `num_epochs`. To enable early stopping, set
`early_stopping_patience` to a positive number of consecutive non-improving validation
epochs; leave it as `null` to train for every epoch. `early_stopping_min_delta` is the
minimum improvement required to reset patience: validation-accuracy improvement (in
percentage points) for classification, or validation-loss reduction for regression.
When a baseline is part of a multi-model classification experiment, it automatically
uses the baseline config while pretrained models use the selected frozen or fine-tuned
config.

## Evaluation Purposes

Set `EXPERIMENT_PURPOSE` to choose which splits are evaluated:

| Purpose | Evaluation splits |
| --- | --- |
| `model_selection` | Internal `val` and `validation_unseen_matched` |
| `final_evaluation` | Internal `test`, `test_unseen_matched`, `test_unseen_related`, and `validation_unseen_matched` |

During final evaluation, `validation_unseen_matched` is included as an additional
diagnostic tool, not an independent generalisation evaluation.

Use validation results for model and configuration choices, then run final evaluation.
Internal validation selects checkpoints and controls early stopping in both modes,
using accuracy for classification and normalised Huber loss for regression. Selection
runs exclude the internal test partition and do not load the unseen test splits.

Metrics are calculated per frame: accuracy and macro F1 for classification, and MAE,
RMSE, and R² for regression. Accuracy is a percentage; macro F1 is on a 0–1 scale and
averages the ground-truth classes present in each split. Regression errors use the
original target units.

On `test_unseen_related`, object and object-region identities have no training classes.
Their predictions are shown against original source identities without loss, accuracy,
or F1. Force-level, motion, and regression targets receive the usual metrics.

## Outputs

| Workflow | Task folder | Run name |
| --- | --- | --- |
| Classification and sweeps | `<target>_classification` | Model type or sweep variant name |
| Conditioned classification | `<target>_conditioned_classification` | `conditioned_<MODEL_TYPE>` |
| Regression | `<target>_regression` | `<MODEL_TYPE>_regressor` |

Results use this layout:

```text
<RESULTS_DIR>/<experiment_purpose>/<task>/
├── experiments.parquet
├── plots/<NNN>/
└── tables/<NNN>/
    ├── summary.csv
    └── <run_name>/classification_report_<split>.csv
```

`<NNN>` is the experiment number within the task and purpose. `EXPERIMENT_NAME` is
recorded as metadata and does not affect paths. Parquet stores model metadata,
configurations, histories, predictions, and metrics. The CSV summary has one row per
run; per-class reports are saved only for scored classification splits.

Checkpoints are saved separately:

```text
<CHECKPOINT_DIR>/<experiment_purpose>/<task>/<NNN>/<run_name>.pth
```

Classification plots include training curves, confusion/prediction matrices, and
comparisons for multiple runs. Per-run plots have run subfolders within the numbered
plot folder. `PLOT_TSNE` enables PNG and interactive HTML t-SNE plots for every evaluated
split; `TSNE_MAX_SAMPLES` limits samples per split (`-1` uses all). Each split has its own
embedding, and related object and object-region plots use original source labels.

Regression plots include training curves, predicted-versus-true plots, and residual
plots for the evaluated splits. These are saved directly in the numbered plot folder.

## Citations

If you use Touch-Ex in your research, please cite:

```
@dataset{mclean2026touchex,
  author = {McLean, Gemma and Hao, Zhou Daniel},
  title = {Touch-Ex: A Region-Level, Force-Annotated Visuo-Tactile Dataset},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/gemixin/touch-ex}
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

SSVTP (used for ColorJitter settings):

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
