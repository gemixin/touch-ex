# Touch-Ex

Classification and regression pipelines for the
[Touch-Ex dataset](https://huggingface.co/datasets/gemixin/touch-ex), collected with a
[DIGIT](https://digit.ml/) tactile sensor. The project supports a baseline CNN,
pretrained visual and tactile encoders, force-conditioned classification, and
force/voltage regression.

Experiment results are available in the
[Touch-Ex results repository](https://github.com/gemixin/touch-ex-results).

## Installation

```bash
git clone https://github.com/gemixin/touch-ex.git
cd touch-ex
python3 -m pip install -r requirements.txt
```

Alternatively, use Conda:

```bash
conda env create -f environment.yml
conda activate touch-ex
```

Run commands from the repository root. Dataset splits and pretrained weights are
downloaded when first needed. A CUDA-enabled PyTorch installation is recommended for
training.

## Running Experiments

Edit the settings at the top of the relevant script, including `RESULTS_DIR` and
`CHECKPOINT_DIR` for your machine.

| Workflow | Command | Main settings |
| --- | --- | --- |
| [Classification](scripts/classify.py) | `python3 -m scripts.classify` | `MODEL_TYPES`, `TARGET_LABEL` |
| [Configuration sweep](scripts/classify_sweep.py) | `python3 -m scripts.classify_sweep` | `MODEL_TYPE`, `TARGET_LABEL`, `DATA_CONFIG_VARIANTS`, `TRAIN_CONFIG_VARIANTS` |
| [Conditioned classification](scripts/classify_conditioned.py) | `python3 -m scripts.classify_conditioned` | `MODEL_TYPE`, `TARGET_LABEL`, `FUSION_HIDDEN_DIM`, `FUSION_DROPOUT` |
| [Regression](scripts/regress.py) | `python3 -m scripts.regress` | `MODEL_TYPE`, `REGRESSION_TARGET` |

Classification targets are `object`, `object_region`, `force_level`, and `motion`.
Conditioned classification supports `object` and `object_region`, using ground-truth
force level as an additional input. Set its `MODEL_TYPE` before running.
Regression targets are `force_n` (approximate calibrated force in Newtons) and
`fsr_voltage` (volts).

Sweeps run every combination of named data and training variants, applied as overrides
to the base JSON configurations. Run names use the variant names, combining them as
`<data_variant>__<train_variant>` when both vary; a single pair is named `default`.
Sweeps default to `model_selection`.

### Models

| Model type | Pretraining |
| --- | --- |
| `baseline` | CNN trained from scratch |
| `resnet18`, `resnet50`, `efficientnet_b0`, `vit_b_16`, `deit_tiny` | ImageNet |
| `t3_tiny` | T3-Tiny tactile encoder pretrained on DIGIT data |

Regression and conditioned classification support the pretrained models.
`FREEZE_BACKBONE=True` trains only the task-specific head, including the fusion layer
for conditioned classification. `False` fine-tunes the whole model. The baseline is
always trained end-to-end.

## Configuration

Scripts use [`configs/chosen_data_config.json`](configs/chosen_data_config.json) and a
training JSON file, with `DATA_CONFIG_OVERRIDES` and `TRAIN_CONFIG_OVERRIDES` applied
on top. Sweeps use their named variant dictionaries instead. The unaugmented
[`default_data_config.json`](configs/default_data_config.json) is also available as a
starting point.

### Data

| Setting | Purpose |
| --- | --- |
| `split_size` | Combined internal validation/test fraction, divided equally; `0.3` gives approximately 70:15:15 train/validation/test. |
| `stratify_label` | Train/validation stratification for regression; classification uses its target label. |
| `transform_name` | `pad_224` pads then resizes; `center_crop_224` resizes then centre-crops. Both produce 224×224 images. |
| `bg_path` | Background `.jpg` for subtraction, or `null` to disable it. |
| `norm_type` | `dataset` uses training-partition statistics, `imagenet` uses ImageNet statistics, and `null` disables normalisation. `norm_cache_path` sets the statistics cache path. |
| `train_augmentations` | Training-only colour jitter, random resized cropping, and horizontal flipping. Random resized cropping replaces the configured spatial transform. |
| `batch_size`, `num_workers`, `shuffle_map` | Batch size, loading workers, and shuffling by split. |

Reusable ColorJitter settings are provided in
[`ssvtp_color_jitter_settings.json`](configs/ssvtp_color_jitter_settings.json).

### Training

| Configuration | Use |
| --- | --- |
| [finetuned_train_config.json](configs/finetuned_train_config.json) | Pretrained models trained end-to-end |
| [frozen_train_config.json](configs/frozen_train_config.json) | Pretrained models with frozen backbones |
| [baseline_train_config.json](configs/baseline_train_config.json) | Baseline CNN |

The scripts select the frozen or fine-tuned config using `FREEZE_BACKBONE`; baseline
runs use the baseline config. Training settings cover the optimiser (`adam` or
`adamw`), learning-rate schedule, weight decay, epochs, and early stopping.
Set `early_stopping_patience` to `null` to disable early stopping.
`early_stopping_min_delta` measures improvement in accuracy percentage points for
classification or normalised Huber loss for regression.

## Evaluation

Set `EXPERIMENT_PURPOSE` to select the evaluation splits:

| Purpose | Evaluated splits |
| --- | --- |
| `model_selection` | Internal `val` and `validation_unseen_matched` |
| `final_evaluation` | Internal `test`, `test_unseen_matched`, and `test_unseen_related` |

Use validation results to select models and configurations, then run final evaluation.
Internal validation selects checkpoints and controls early stopping in both modes:
classification uses accuracy, and regression uses Huber loss on targets normalised with
training-partition statistics. Selection runs exclude the internal test partition and
load only the training and matched-validation source splits.

The source `train` split is partitioned by interaction with fixed seed **129**, keeping
all frames from an interaction together. The internal test partition is reserved first,
stratified by `object_region`, and remains the same across tasks for a fixed dataset and
split size. Train/validation stratification uses the classification target or regression's
configured label (default `object_region`). `SEED` defaults to **156** and controls model
initialisation, shuffling, and augmentation independently of partitioning. It is reset
before each run. `DETERMINISTIC` controls whether PyTorch requires deterministic algorithms.

Metrics are calculated per frame:

- **Classification:** accuracy (%) and macro F1 (0–1). Macro F1 averages the ground-truth
  classes present in the evaluated split.
- **Regression:** MAE, RMSE, and R². MAE and RMSE use the original target units.

Related-test object and object-region identities have no training classes, so these
targets produce matrices of source identities versus predicted training classes, without
classification metrics.
Force-level, motion, and regression targets receive their usual metrics.

## Outputs

| Workflow | Task folder | Run name |
| --- | --- | --- |
| Classification and sweeps | `<target>_classify` | Model type or sweep variant name |
| Conditioned classification | `<target>_conditioned_classify` | `conditioned_<MODEL_TYPE>` |
| Regression | `<target>_regress` | `<MODEL_TYPE>_regressor` |

Results use this layout:

```text
<RESULTS_DIR>/<task>/<experiment_purpose>/
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
<CHECKPOINT_DIR>/<task>/<experiment_purpose>/<NNN>/<run_name>.pth
```

Classification plots include training curves, confusion/prediction matrices, and
comparisons for multiple runs. `PLOT_TSNE` enables PNG and interactive HTML t-SNE plots
for every evaluated split; `TSNE_MAX_SAMPLES` limits samples per split (`-1` uses all).
Related object and object-region plots use original source labels. Each split has its own
embedding. Per-run classification plots have run subfolders; regression saves training curves,
predicted-versus-true plots, and residual plots in the numbered plot folder.

## Project Structure

- `scripts/`: experiment entry points
- `configs/`: data, training, and normalisation-cache configurations
- `data/`: loading, preprocessing, splitting, and validation
- `models/`: model architectures
- `ml/`: training, evaluation, experiment management, and plotting
- `notebooks/`: dataset exploration

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
