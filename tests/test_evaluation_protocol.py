"""Small, offline checks for split selection, seeds, and saved evaluations."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import builder, utils
from ml import experiments
from ml.torch_functions import set_random_seed
from ml.train_eval import eval_classifier, eval_regressor


class ToyClassifier(nn.Module):
    def forward(self, images, force=None, return_features=False):
        return images


class EvaluationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path('configs/default_data_config.json').read_text())
        self.config.update(norm_type=None, bg_path=None, num_workers=0)
        rows = [
            dict(object=obj, region='body', object_region=f'{obj}_body',
                 force_level=str(i % 3 + 1), motion='sliding' if i % 2 else 'rotation',
                 interaction_id=f'{obj}_{i}', frame_num=str(frame), interaction_num=str(i),
                 force_n=float(i), fsr_voltage=float(i) / 10,
                 hardness='hard', material='plastic', description='example', image={})
            for obj in ['b', 'a'] for i in range(30) for frame in range(2)
        ]
        self.frames = {'train': pd.DataFrame(rows)}
        for split in ['validation_unseen_matched', 'test_unseen_matched', 'test_unseen_related']:
            df = self.frames['train'].iloc[-6:].copy()
            if split == 'test_unseen_related':
                df['object'], df['object_region'] = 'new', 'new_body'
            self.frames[split] = df
        self.load = patch('data.utils.load_dataset', side_effect=lambda _, split: Mock(
            to_pandas=lambda: self.frames[split].copy()))
        self.loader = self.load.start()
        self.addCleanup(self.load.stop)
        self.images = patch('data.dataset.process_tactile_image', return_value=torch.tensor([2., 0.]))
        self.images.start()
        self.addCleanup(self.images.stop)

    def test_fixed_partitions_and_separate_training_seed(self):
        set_random_seed(17)
        first, _ = builder.get_datasets(self.config)
        weights = nn.Linear(2, 2).weight.detach().clone()
        set_random_seed(23)
        second, _ = builder.get_datasets(self.config)
        self.assertFalse(torch.equal(weights, nn.Linear(2, 2).weight))
        self.assertEqual(set(first), {'train', 'val', 'validation_unseen_matched'})
        self.assertEqual([c.kwargs['split'] for c in self.loader.call_args_list],
                         ['train', 'validation_unseen_matched'] * 2)
        for split in first:
            self.assertEqual(first[split].df.interaction_id.tolist(), second[split].df.interaction_id.tolist())
        orders = [list(builder.create_dataloaders(first, self.config, seed=s)['train'].sampler)
                  for s in [17, 17, 23]]
        self.assertEqual(orders[0], orders[1])
        self.assertNotEqual(orders[0], orders[2])
        final, _ = builder.get_datasets(self.config, 'final_evaluation')
        ids = [set(final[s].df.interaction_id) for s in ['train', 'val', 'test']]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertEqual(set.union(*ids), set(self.frames['train'].interaction_id))

    def test_shared_matched_encoding_and_unknown_related_identities(self):
        datasets, labels = builder.get_datasets(self.config, 'final_evaluation')
        matched = datasets['test_unseen_matched']
        self.assertEqual(matched.label2idx, datasets['train'].label2idx)
        self.assertEqual(matched[0]['object'].item(), labels['train']['object'].index('a'))
        related = datasets['test_unseen_related'][0]
        self.assertEqual(related['object'].item(), -1)
        self.assertEqual(related['object_region'].item(), -1)
        self.assertEqual(related['source_object'], 'new')
        self.assertGreaterEqual(related['force_level'].item(), 0)
        self.assertNotIn('expected_object', related)

    def test_unknown_predictions_skip_loss_but_known_targets_are_scored(self):
        features = [dict(image=torch.tensor([2., 0.]), object=torch.tensor(-1),
                         source_object='new', motion=torch.tensor(0)) for _ in range(3)]
        loader = DataLoader(features, batch_size=2)
        with patch('ml.train_eval.nn.CrossEntropyLoss', side_effect=AssertionError('loss called')):
            result = eval_classifier(ToyClassifier(), 'toy', 'cpu', loader, 'object', compute_metrics=False)
        self.assertEqual(set(result), {'y_true', 'y_pred', 'source_labels'})
        self.assertEqual(result['y_true'], [-1] * 3)
        scored = eval_classifier(ToyClassifier(), 'toy', 'cpu', loader, 'motion', conditioning_label='motion')
        self.assertEqual(scored['accuracy'], 100)
        self.assertEqual(scored['macro_f1_avg'], 1)
        rows = [dict(image=torch.tensor(float(i)), force_n=torch.tensor(float(i))) for i in range(3)]
        regression = eval_regressor(nn.Identity(), 'toy', 'cpu', DataLoader(rows, batch_size=2),
                                    'force_n', {'mean': 0., 'std': 1.})
        self.assertEqual(regression['mae'], 0)
        self.assertEqual(regression['loss'], 0)

    def test_macro_f1_averages_ground_truth_classes_and_keeps_all_predictions(self):
        truths, predictions = [0, 0, 0, 1], [0, 0, 2, 1]
        features = []
        for truth, prediction in zip(truths, predictions):
            logits = torch.full((4,), -3.)
            logits[prediction] = 3.
            features.append(dict(image=logits, object=torch.tensor(truth)))
        result = eval_classifier(
            ToyClassifier(), 'toy', 'cpu', DataLoader(features, batch_size=2), 'object'
        )
        # Class F1s are 0.8 and 1.0: macro=0.9, weighted=0.85, union-of-labels macro=0.6.
        self.assertAlmostEqual(result['macro_f1_avg'], 0.9)
        self.assertNotIn('weighted_f1_avg', result)
        self.assertEqual(result['y_true'], truths)
        self.assertEqual(result['y_pred'], predictions)
        with tempfile.TemporaryDirectory() as directory:
            experiments.mv.plot_model_comparison([result], ['toy'], directory, 'val')
            self.assertTrue(Path(directory, 'comparison_val.png').is_file())

    def test_classification_workflows_route_and_save_both_purposes(self):
        def train(context):
            context['models'] = [ToyClassifier() for _ in context['run_names']]
            context['histories'] = [[{'epoch': 1, 'val_acc': 50.}] for _ in context['run_names']]

        with tempfile.TemporaryDirectory() as directory, patch.object(experiments, 'train_classifiers', train), \
                patch.object(experiments, 'train_conditioned_classifier', train), \
                patch.object(experiments, 'generate_classification_plots'):
            config = Path(directory, 'data.json')
            config.write_text(json.dumps(self.config))
            common = dict(target_label='object', experiment_name='example', seed=17, deterministic=True,
                          freeze_backbone=False, data_config_overrides={}, train_config_overrides={},
                          plot_tsne=False, tsne_max_samples=-1, data_config_path=str(config),
                          train_config_path='configs/baseline_train_config.json',
                          baseline_train_config_path='configs/baseline_train_config.json',
                          results_dir=directory, checkpoint_dir=directory)
            for purpose, splits in utils.EVALUATION_SPLITS.items():
                with self.subTest(purpose=purpose):
                    _, _, results = experiments.classify(model_types=['baseline'], experiment_purpose=purpose, **common)
                    self.assertEqual(tuple(results), splits)
                    path = Path(directory, 'object_classify', purpose, 'experiments.parquet')
                    saved = pd.read_parquet(path)
                    self.assertEqual(saved.iloc[0].split_seed, 129)
                    self.assertEqual(saved.iloc[0].experiment_purpose, purpose)
                    self.assertIn(f'{splits[0]}_macro_f1_avg', saved)
                    self.assertFalse(any('weighted_f1' in column for column in saved.columns))
                    tables = path.parent / 'tables' / '001'
                    summary = pd.read_csv(tables / 'summary.csv')
                    self.assertEqual(summary.run_name.tolist(), ['baseline'])
                    expected_columns = ['run_name', 'model_type'] + [
                        f'{split}_{metric}' for split in splits if split != 'test_unseen_related'
                        for metric in ['accuracy', 'macro_f1_avg']
                    ]
                    self.assertEqual(summary.columns.tolist(), expected_columns)
                    report = pd.read_csv(tables / 'baseline' / f'classification_report_{splits[0]}.csv')
                    self.assertEqual(report.columns.tolist(), ['class', 'precision', 'recall', 'f1', 'support'])
                    self.assertAlmostEqual(report.f1.mean(), summary.iloc[0][f'{splits[0]}_macro_f1_avg'])
                    self.assertEqual(report.support.sum(), len(results[splits[0]][0]['y_true']))
                    matched = pd.read_csv(tables / 'baseline' / f'classification_report_{splits[1]}.csv')
                    self.assertEqual(matched['class'].tolist(), ['a'])
                    self.assertEqual(matched.f1.tolist(), [0.])
                    if purpose == 'model_selection':
                        self.assertFalse(any(c.startswith('test_') for c in saved.columns))
                    else:
                        self.assertNotIn('test_unseen_related_accuracy', saved)
                        self.assertIn('test_unseen_related_source_labels', saved)
                        self.assertFalse((tables / 'baseline' / 'classification_report_test_unseen_related.csv').exists())
                    # Append to the same table and retain the applicable schema.
                    experiments.classify(model_types=['baseline'], experiment_purpose=purpose, **common)
                    self.assertEqual(len(pd.read_parquet(path)), 2)
                    self.assertEqual(len(pd.read_csv(path.parent / 'tables' / '002' / 'summary.csv')), 1)
                    conditioned = {k: v for k, v in common.items() if k != 'baseline_train_config_path'}
                    _, _, results = experiments.classify_conditioned(
                        fusion_hidden_dim=4, fusion_dropout=0., experiment_purpose=purpose, **conditioned)
                    self.assertEqual(tuple(results), splits)
                    conditioned_tables = Path(directory, 'object_conditioned_classify', purpose, 'tables', '001')
                    self.assertEqual(pd.read_csv(conditioned_tables / 'summary.csv').run_name.tolist(),
                                     ['conditioned_resnet18'])
                    sweep = {k: v for k, v in common.items() if k not in {'data_config_overrides', 'train_config_overrides'}}
                    _, _, results = experiments.classify_sweep(
                        model_type='baseline', data_config_variants={'first': {}, 'second': {}},
                        train_config_variants={'default': {}}, experiment_purpose=purpose, **sweep)
                    self.assertEqual(tuple(results), splits)
                    sweep_tables = path.parent / 'tables' / '003'
                    self.assertEqual(pd.read_csv(sweep_tables / 'summary.csv').run_name.tolist(), ['first', 'second'])
                    for run in ['first', 'second']:
                        self.assertTrue((sweep_tables / run / f'classification_report_{splits[0]}.csv').is_file())
                    if purpose == 'final_evaluation':
                        experiments.classify(model_types=['baseline'], experiment_purpose=purpose,
                                             **{**common, 'target_label': 'motion'})
                        motion_tables = Path(directory, 'motion_classify', purpose, 'tables', '001')
                        self.assertTrue((motion_tables / 'baseline' / 'classification_report_test_unseen_related.csv').is_file())
                        self.assertIn('test_unseen_related_macro_f1_avg', pd.read_csv(motion_tables / 'summary.csv'))

    def test_regression_workflow_routes_and_saves_both_purposes(self):
        # Regressors normally squeeze their scalar output, so use a tiny equivalent.
        class Regressor(nn.Module):
            def forward(self, images):
                return images[:, 0]

        def train(context):
            context['model'] = Regressor()
            context['history'] = [{'epoch': 1, 'val_loss': 1.}]

        with tempfile.TemporaryDirectory() as directory, patch.object(experiments, 'train_regression_model', train), \
                patch.object(experiments, 'generate_regression_plots'):
            config = Path(directory, 'data.json')
            config.write_text(json.dumps(self.config))
            for purpose, splits in utils.EVALUATION_SPLITS.items():
                _, _, results = experiments.regress(
                    regression_target='force_n', experiment_name='example', seed=23,
                    deterministic=True, freeze_backbone=False, data_config_overrides={},
                    train_config_overrides={}, data_config_path=str(config),
                    train_config_path='configs/baseline_train_config.json',
                    results_dir=directory, checkpoint_dir=directory, experiment_purpose=purpose,
                )
                self.assertEqual(tuple(results), splits)
                saved = pd.read_parquet(Path(directory, 'force_n_regress', purpose, 'experiments.parquet'))
                self.assertEqual(saved.iloc[0].split_seed, 129)
                self.assertTrue(all(f'{split}_mae' in saved for split in splits))
                summary = pd.read_csv(Path(directory, 'force_n_regress', purpose, 'tables', '001', 'summary.csv'))
                self.assertEqual(summary.run_name.tolist(), ['resnet18_regressor'])
                self.assertEqual(summary.columns.tolist(), ['run_name', 'model_type'] + [
                    f'{split}_{metric}' for split in splits for metric in ['mae', 'rmse', 'r2']
                ])
                for split in splits:
                    self.assertAlmostEqual(summary.iloc[0][f'{split}_mae'], saved.iloc[0][f'{split}_mae'])
                if purpose == 'model_selection':
                    self.assertFalse(any(c.startswith('test_') for c in saved.columns))

    def test_plots_use_selected_splits_and_related_source_rows(self):
        scored = dict(y_true=[0, 1], y_pred=[0, 0], accuracy=50., macro_f1_avg=1/3)
        related = dict(y_true=[-1, -1], y_pred=[0, 1], source_labels=['new_a', 'new_b'])
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(experiments.mv, 'plot_classification_training_curves'), \
                patch.object(experiments.mv, 'plot_tsne_features') as tsne, \
                patch.object(experiments.mv, 'plot_confusion_matrix',
                             wraps=experiments.mv.plot_confusion_matrix) as matrices:
            for purpose, splits in utils.EVALUATION_SPLITS.items():
                context = dict(
                    results_path=directory, experiment_number=1, run_names=['toy'],
                    models=[ToyClassifier()], histories=[[]], train_labels=[['a', 'b']],
                    target_label='object', plot_tsne=True, seed=23, device='cpu',
                    tsne_max_samples=10, experiment_purpose=purpose,
                    dataloaders=[{splits[0]: 'primary_loader'}],
                    evaluation_results={s: [related if s == 'test_unseen_related' else scored] for s in splits},
                )
                experiments.generate_classification_plots(context)
                self.assertEqual(tsne.call_args.kwargs['split_name'], splits[0])
                self.assertEqual(tsne.call_args.kwargs['dataloader'], 'primary_loader')
            plots = Path(directory, 'plots', '001', 'toy')
            self.assertTrue((plots / 'confusion_matrix_test_unseen_related.png').is_file())
            related_calls = [call.kwargs for call in matrices.call_args_list
                             if call.kwargs['split_name'] == 'test_unseen_related']
            self.assertEqual(len(related_calls), 1)
            self.assertEqual(related_calls[0]['row_labels'], ['new_a', 'new_b'])
            self.assertEqual(related_calls[0]['results']['y_true'], [0, 1])
            self.assertEqual(related['y_true'], [-1, -1])


if __name__ == '__main__':
    unittest.main()
