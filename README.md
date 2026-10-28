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
- `SEED` and `DETERMINISTIC`: model and training reproducibility settings; dataset partitions always use seed `129`
- `FREEZE_BACKBONE`: train only the task-specific head of pretrained models when `True`
- `PLOT_TSNE` and `TSNE_MAX_SAMPLES`: t-SNE settings for internal validation or test, according to the purpose; use `-1` to include every example

Available model types are:

- `baseline`: a CNN trained from scratch
- `resnet18`: ImageNet-pretrained ResNet-18
- `resnet50`: ImageNet-pretrained ResNet-50 (`ResNet50_Weights.IMAGENET1K_V2`)
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

Set `TARGET_LABEL` to either `object` or `object_region`. The ImageNet-pretrained
ResNet-18 concatenates its image representation with the ground-truth force level,
normalised to `0.0`, `0.5`, or `1.0`, before applying a lightweight fusion layer and
classification head.
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
- `EXPERIMENT_NAME`: a descriptive name recorded with the results
- `EXPERIMENT_PURPOSE`: `model_selection` (default) or `final_evaluation`
- `SEED` and `DETERMINISTIC`: model and training reproducibility settings; dataset partitions always use seed `129`
- `FREEZE_BACKBONE`: train only the regression head when `True`
- `DATA_CONFIG_OVERRIDES` and `TRAIN_CONFIG_OVERRIDES`: values applied over the selected
  base JSON configurations

The regression workflow uses an ImageNet-pretrained ResNet-18, optimises Huber loss on
training-split-normalised targets, and reports MAE, RMSE, and R² in the original target
units. With `FREEZE_BACKBONE=True`, the ImageNet-pretrained backbone remains frozen and
only the regression head is trained; with `False`, the entire model is fine-tuned.

## Configurations

The scripts start with base JSON files and apply the overrides defined in the script.
Classification sweeps apply every combination of their named data and training variants.
Dataset partitioning always uses `SPLIT_SEED = 129`, independently of `SEED`. The
experiment `SEED` controls model initialisation, training shuffle, and augmentation
randomness. For the same split size and stratification, changing `SEED`
preserves the train/validation/test partition membership.

Classification experiments, including sweeps and conditioned models, always override
`stratify_label` with `TARGET_LABEL`: object classification stratifies by `object`,
object-region classification by `object_region`, force classification by `force_level`,
and motion classification by `motion`. Regression uses the data config's
`stratify_label`, including any overrides; the supplied configs use `object`.
Different classification targets can therefore produce different interaction partitions
despite using the same split seed. Stratification approximately preserves proportions
for the selected label, without guaranteeing balance across other labels.

### Data

All experiments use every force level and both motion types in their selected splits.

[`configs/default_data_config.json`](configs/default_data_config.json) is the standard
starting point. [`configs/pad_jitter_data_config.json`](configs/pad_jitter_data_config.json)
is the selected configuration from the configuration experiments; it uses the same
padding-based transform with mild training-only colour jitter.

| Setting | Purpose and accepted values |
| --- | --- |
| `split_size` | Total fraction held out from training, split equally between validation and the standard test set. |
| `stratify_label` | Interaction-level stratification column used by regression (`object` by default). Classification always overrides this with `TARGET_LABEL`. |
| `transform_name` | Base 224×224 image preparation: `pad_224` pads then resizes; `center_crop_224` resizes then centre-crops. Used for validation, test, and training unless `random_resized_crop` is enabled. |
| `bg_path` | Path to a `.jpg` background image for subtraction, or `null` to disable it. |
| `norm_type` | `dataset` computes/loads statistics from the training split; `imagenet` uses ImageNet statistics; `null` disables normalisation. `norm_cache_path` is required unless normalisation is disabled. |
| `batch_size`, `num_workers`, `shuffle_map` | DataLoader batch size, worker count, and per-split shuffling. |
| `train_augmentations` | Training-only settings: `random_resized_crop` replaces `transform_name` with `RandomResizedCrop(224)` when `true`; `color_jitter` is a ColorJitter dictionary or `null` and runs before background subtraction; `horizontal_flip` is a probability from `0` to `1` or `null`, applied before normalisation. Validation and test data remain deterministic and unaugmented. |

The default data config disables all augmentation. Reusable ColorJitter settings in
[`configs/ssvtp_color_jitter_settings.json`](configs/ssvtp_color_jitter_settings.json)
use the SSVTP values from the official paper (cited below).

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

## Evaluation purposes

All scripts expose `EXPERIMENT_PURPOSE`, passed to the Python workflows as
`experiment_purpose`. It defaults to `model_selection`.

| Purpose | Evaluation splits |
| --- | --- |
| `model_selection` | Internal `val` and Hugging Face `validation_unseen_matched` |
| `final_evaluation` | Internal `test`, `test_unseen_matched`, and `test_unseen_related` |

Hugging Face `train` is always partitioned by interaction into internal train, val,
and test using seed `129`. Internal val selects the best checkpoint and controls early
stopping in both modes. Selection runs reserve internal test interactions without
creating their loader, and request only the unseen validation source split. Use
`final_evaluation` after fixing the model and configuration choices.

Matched splits use the training class indices directly. For related objects, object
and object-region targets are `unknown` (`-1`); classifiers still predict known training
classes. Those evaluations save predictions and original source identities, with no
loss, accuracy, or F1. One matrix shows predictions by original source identity, with
training classes as columns. Force-level, motion, and regression targets remain
labelled and receive normal metrics on related data.

The public workflows return `(models, histories, evaluation_results)` (singular model
and history for conditioned classification and regression). `evaluation_results` is
keyed by split name; classification and sweeps contain a list of results per split in
run order, while conditioned classification and regression contain one result per split.
Classifier result keys are `loss`, `accuracy` (percent), `macro_f1_avg`, `y_true`,
and `y_pred`. Unknown identity results contain only `y_true`, `y_pred`, and
`source_labels`. Regression results contain `loss`, `mae`, `rmse`, `r2`, `y_true`, and
`y_pred`. The low-level evaluators accept `data_loader`.
Macro F1 gives equal weight to each ground-truth class represented in the evaluation
split. Predictions outside those classes still count as errors. Saved `y_true` and
`y_pred` allow other F1 averages to be calculated later.

## Outputs

Results, checkpoints, and plots use the existing task folders with a purpose subfolder:

```text
<RESULTS_DIR>/<task>/<experiment_purpose>/experiments.parquet
<RESULTS_DIR>/<task>/<experiment_purpose>/tables/<NNN>/summary.csv
<RESULTS_DIR>/<task>/<experiment_purpose>/tables/<NNN>/<run_name>/classification_report_<split>.csv
<RESULTS_DIR>/<task>/<experiment_purpose>/plots/<NNN>/
<CHECKPOINT_DIR>/<task>/<experiment_purpose>/<NNN>/<run_name>.pth
```

Here `<task>` is `<target>_classify`, `<target>_conditioned_classify`, or
`<target>_regress`; `<NNN>` is the zero-padded experiment number within that purpose.
Regression uses `resnet18_regressor` as its run name.
Classification checkpoints maximise internal validation accuracy; regression checkpoints
minimise internal validation loss.

Tables record the experiment purpose, training seed, fixed split seed, configurations,
histories, and evaluated split results. Evaluation columns are prefixed by the actual
split, for example `val_accuracy` or `test_unseen_matched_macro_f1_avg`. Selection
tables have no test evaluation columns. Training histories still include validation
metrics in both modes.
Conditioned tables also record the conditioning label, ordered force labels, fusion
hidden dimension, and fusion dropout.

Each experiment's `summary.csv` has one row per `run_name`, with `model_type` and
split-prefixed accuracy/macro F1 for classification or MAE/RMSE/R² for regression.
It contains only that experiment's runs and applicable metrics. Accuracy is a percentage;
F1 is on a 0–1 scale. Related identity predictions have no summary metric columns.

Classification also saves a report for each run and scored split, with columns `class`,
`precision`, `recall`, `f1`, and `support`. Recall is the fraction of a class's frames
classified correctly (often called per-class accuracy); precision, recall, and F1 are
on a 0–1 scale. Reports cover ground-truth classes present in that split, matching the
macro F1 calculation. Related identity predictions have diagnostic plots instead of
classification reports; related force and motion predictions receive normal reports.

Classification plots include training curves, confusion matrices, scored-model
comparisons, and optional static/interactive t-SNE on internal val for selection or
internal test for final evaluation. Regression plots include training curves and
predicted-versus-true/residual plots for the evaluated splits.
Plot titles are omitted for use with LaTeX captions. Axis labels and legends are
retained, with shared DejaVu Sans typography: 12-point axis labels and 11-point ticks,
legends, and heatmap annotations. Interactive t-SNE uses the same font and equivalent
sizes in pixels. Filenames and folder paths identify the task, run, and split.

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
