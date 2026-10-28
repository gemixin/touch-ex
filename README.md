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
- `ml/` contains experiment orchestration, training, evaluation, visualisation, and utilities
- `models/` contains model architecture definitions
- `notebooks/` contains dataset exploration notebooks
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

The pipeline downloads the Touch-Ex dataset from Hugging Face when it is first used. A
CUDA-enabled PyTorch installation is recommended for training. Run the commands below
from the repository root so the relative configuration and data paths resolve correctly.

## Running a Classification Experiment

Configure the constants at the top of [`scripts/classify.py`](scripts/classify.py), then run:

```bash
python3 -m scripts.classify
```

The main settings are:

- `MODEL_TYPES`: one or more models to train and compare
- `TARGET_LABEL`: `object`, `object_region`, `force_level`, or `motion`
- `EXPERIMENT_NAME`: a descriptive name recorded with the results
- `EXPERIMENT_PURPOSE`: `model_selection` (default) or `final_evaluation`
- `SEED` and `DETERMINISTIC`: reproducibility settings
- `FREEZE_BACKBONE`: train only the task-specific head of pretrained models when `True`
- `PLOT_TSNE` and `TSNE_MAX_SAMPLES`: t-SNE plot settings for internal validation or test; use `-1` to include every example

Available model types are:

- `baseline`: a CNN trained from scratch
- `resnet18`: ImageNet-pretrained ResNet-18
- `resnet50`: ImageNet-pretrained ResNet-50
- `efficientnet_b0`: ImageNet-pretrained EfficientNet-B0
- `vit_b_16`: ImageNet-pretrained ViT-B/16
- `deit_tiny`: ImageNet-pretrained DeiT-Tiny
- `t3_tiny`: T3-Tiny tactile encoder pretrained on DIGIT data

The `baseline` architecture is implemented by `BaselineCNNClassifier`. All configurable
pretrained backbones are wrapped by `PretrainedClassifier`.

For pretrained models, `FREEZE_BACKBONE=True` retains the pretrained backbone and trains only the classifier. With `False`, the entire model is fine-tuned. The baseline is always trained end-to-end.

### Running Classification Configuration Sweeps

[`scripts/classify_sweep.py`](scripts/classify_sweep.py) runs one selected model over every combination of named data and training configuration variants. Each result row keeps the same `model_type` and records its complete `data_config` and `train_config`.

Add variants to `DATA_CONFIG_VARIANTS` and `TRAIN_CONFIG_VARIANTS` at the top of the
script. For example, two data variants and two training variants produce four runs,
named `<data_variant>__<train_variant>`. Each run applies the corresponding pair of
variant dictionaries over the base JSON configuration files.

```bash
python3 -m scripts.classify_sweep
```

### Running Conditioned Classification

Configure the constants at the top of
[`scripts/classify_conditioned.py`](scripts/classify_conditioned.py),
then run:

```bash
python3 -m scripts.classify_conditioned
```

Set `TARGET_LABEL` to either `object` or `object_region`. `MODEL_TYPE` selects the
pretrained backbone (default `resnet18`). The model concatenates its image representation
with the ground-truth force level, normalised to `0.0`, `0.5`, or `1.0`, before applying
a lightweight fusion layer and classification head.
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
- `MODEL_TYPE`: any supported pretrained backbone (default `resnet18`)
- `EXPERIMENT_NAME`: a descriptive name recorded with the results
- `EXPERIMENT_PURPOSE`: `model_selection` (default) or `final_evaluation`
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

### Data

All experiments use every force level and both motion types in their selected splits.

[`configs/default_data_config.json`](configs/default_data_config.json) is the standard
starting point. Update [`configs/chosen_data_config.json`](configs/chosen_data_config.json)
with the configuration selected from validation experiments; the experiment
scripts use this file by default.

| Setting | Purpose and accepted values |
| --- | --- |
| `split_size` | Combined validation/test fraction, divided equally between them (`0.3` by default). |
| `stratify_label` | Train/validation stratification label for regression. Classification uses `TARGET_LABEL`. |
| `transform_name` | Base 224×224 image preparation: `pad_224` pads then resizes; `center_crop_224` resizes then centre-crops. Used for validation, test, and training unless `random_resized_crop` is enabled. |
| `bg_path` | Path to a `.jpg` background image for subtraction, or `null` to disable it. |
| `norm_type` | `dataset` computes/loads statistics from the training split; `imagenet` uses ImageNet statistics; `null` disables normalisation. `norm_cache_path` is required unless normalisation is disabled. |
| `batch_size`, `num_workers`, `shuffle_map` | DataLoader batch size, worker count, and per-split shuffling. |
| `train_augmentations` | Training-only settings: `random_resized_crop` replaces `transform_name` with `RandomResizedCrop(224)` when `true`; `color_jitter` is a ColorJitter dictionary or `null` and runs before background subtraction; `horizontal_flip` is a probability from `0` to `1` or `null`, applied before normalisation. Validation and test data remain deterministic and unaugmented. |

The default data config disables all augmentation. Reusable ColorJitter settings in
[`configs/ssvtp_color_jitter_settings.json`](configs/ssvtp_color_jitter_settings.json)
use the SSVTP values from the official paper (cited below).

### Splitting and Randomness

Hugging Face `train` is split by interaction using fixed seed **129**. The default
`split_size=0.3` gives approximately **70% train / 15% validation / 15% test**.
Test interactions are reserved first, stratified by `object_region`, and remain the
same across tasks when the dataset and split size are unchanged. Train/validation
stratification uses the classification target or regression's configured
`stratify_label` (default `object_region`). This preserves the selected label's
approximate proportions, without guaranteeing balance across other labels.

The experiment `SEED` controls model initialisation, training shuffle, and augmentation.
It resets before each run, so changing run order does not change its randomness or
dataset partitions.

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
config. Regression uses the frozen or fine-tuned config according to
`FREEZE_BACKBONE`.

## Evaluation Purposes

Set `EXPERIMENT_PURPOSE` to choose which splits are evaluated:

| Purpose | Evaluation splits |
| --- | --- |
| `model_selection` (default) | Internal `val` and Hugging Face `validation_unseen_matched` |
| `final_evaluation` | Internal `test`, `test_unseen_matched`, and `test_unseen_related` |

Use validation results for model/configuration choices, then switch to `final_evaluation`
after those choices are fixed. Selection runs exclude the common internal test
interactions and do not load the unseen test splits. Internal validation selects
checkpoints and controls early stopping in both modes.

Matched splits share the training class mapping. On `test_unseen_related`, object and
object-region labels are unknown (`-1`): save predictions and a matrix of original
source identities versus predicted training classes, without loss, accuracy, or F1.
Force, motion, and regression targets receive normal metrics on related data.

Workflows return trained model(s), training histories, and evaluation results keyed by
split name.

## Outputs

All outputs include an `<experiment_purpose>` folder. `<NNN>` is the zero-padded
experiment number within that task and purpose. Only the selected evaluation splits
are recorded; model-selection results have no test-result fields.

### Classification

Each classification run saves the best-validation-accuracy checkpoint for every model
under:

```text
<CHECKPOINT_DIR>/<target>_classify/<experiment_purpose>/<NNN>/<run_name>.pth
```

Experiment metadata, data/training configs, histories, predictions, and evaluation
metrics are appended to:

```text
<RESULTS_DIR>/<target>_classify/<experiment_purpose>/experiments.parquet
```

Plots are saved under:

```text
<RESULTS_DIR>/<target>_classify/<experiment_purpose>/plots/<NNN>/
```

The output includes per-run training curves and confusion matrices for the evaluated
splits, comparison plots for multi-run experiments, and optional static and interactive
t-SNE plots on internal validation for selection or internal test for final evaluation.
Per-run plots go in `<run_name>/` subfolders; comparison plots sit alongside them.

### Conditioned Classification

Conditioned classification uses force level as its additional input and follows the same
output structure, with separate folders for the object and object-region tasks:

```text
<CHECKPOINT_DIR>/<target>_conditioned_classify/<experiment_purpose>/<NNN>/<run_name>.pth
<RESULTS_DIR>/<target>_conditioned_classify/<experiment_purpose>/experiments.parquet
<RESULTS_DIR>/<target>_conditioned_classify/<experiment_purpose>/plots/<NNN>/
```

Run names use `conditioned_<MODEL_TYPE>`. The conditioned result table additionally
records the conditioning label, ordered force labels, fusion hidden dimension, and
fusion dropout probability.

### Regression

Each regression run saves its lowest-validation-loss checkpoint under:

```text
<CHECKPOINT_DIR>/<target>_regress/<experiment_purpose>/<NNN>/<MODEL_TYPE>_regressor.pth
```

Its metadata, configurations, target normalisation statistics, training history,
predictions, and metrics for the evaluated splits are appended to:

```text
<RESULTS_DIR>/<target>_regress/<experiment_purpose>/experiments.parquet
```

Regression plots are saved under:

```text
<RESULTS_DIR>/<target>_regress/<experiment_purpose>/plots/<NNN>/
```

They include loss and MAE curves, plus predicted-versus-true and residual plots for
the evaluated splits.

### CSV Tables

Each experiment also saves a summary and, for classification, per-class reports:

```text
<RESULTS_DIR>/<task>/<experiment_purpose>/tables/<NNN>/summary.csv
<RESULTS_DIR>/<task>/<experiment_purpose>/tables/<NNN>/<run_name>/classification_report_<split>.csv
```

`<task>` is `<target>_classify`, `<target>_conditioned_classify`, or `<target>_regress`.
The summary has one row per run, with split-prefixed accuracy and macro F1 for
classification, or MAE/RMSE/R² for regression. Classification reports include per-class
precision, recall, F1, and support for each scored split; recall is the fraction of a
class's frames classified correctly.

Accuracy is a percentage; F1, precision, and recall are on a 0–1 scale. Macro F1 averages
the ground-truth classes present in each split. Saved targets and predictions allow
other metrics to be calculated later.

### Plot Formatting

Static plots omit titles for LaTeX captions and retain consistent axis labels and
legends. Interactive t-SNE has its own title and Plotly styling. Paths and filenames
identify the task, run, and split.

### Checkpoint Use

For inference, rebuild the model, load its `model_state_dict`, and call `eval()`.

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
