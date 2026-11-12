"""Functional tests only: no real-data tuning or claimed performance improvements."""
import copy
import tempfile
import unittest
from unittest.mock import patch
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from main import Framework
from continual.core import batches, evaluate, digest, read_json, write_json
from continual.dynamic import PromptMemory, DynamicPromptModel
from run_dynamic import execute, validate_config, save_checkpoint


def settings():
    return {'capacity': 3, 'initial_prompts': 1, 'source_quantile': .95,
            'minimum_distance': .001, 'distance_margin': 0., 'novel_fraction': .5,
            'patience': 2, 'novel_buffer': 8, 'minimum_coherence': .8,
            'allocation_cooldown': 1, 'replacement_grace': 1}


class DynamicTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)

    def test_persistent_novelty_reuse_capacity_and_optimizer_reset(self):
        memory = PromptMemory(3, 2, settings())
        memory.initialize(np.asarray([[1., 0.], [1., .0001]]), 1, 42)
        optimizer = torch.optim.Adam(memory.parameters(), lr=.01)
        new = torch.tensor([[0., 1.], [0., 1.]])
        event, _ = memory.observe(new, optimizer)
        self.assertIsNone(event)  # One unusual batch is not enough.
        restored = PromptMemory(3, 2, settings())
        restored.load_state_dict(copy.deepcopy(memory.state_dict()))
        restored_event, _ = restored.observe(new, None)
        self.assertEqual(restored_event['type'], 'allocate')  # Pending detector state survives.
        event, _ = memory.observe(new, optimizer)
        self.assertEqual(event['type'], 'allocate')
        self.assertEqual(event['slot'], 1)
        self.assertEqual(int(memory.active.sum()), 2)
        optimizer.zero_grad(set_to_none=True)
        memory.prompts[1].sum().backward(); optimizer.step()
        self.assertIn(memory.prompts[1], optimizer.state)
        source_before = memory.prompts[0].detach().clone()
        event, ids = memory.observe(new, optimizer)
        self.assertIsNone(event)
        self.assertTrue((ids == 1).all())
        self.assertGreater(int(memory.reuse_samples), 0)
        third = torch.tensor([[-1., 0.], [-1., 0.]])
        memory.observe(third, optimizer)
        event, _ = memory.observe(third, optimizer)
        self.assertEqual(event['type'], 'allocate')
        self.assertEqual(event['slot'], 2)
        fourth = torch.tensor([[0., -1.], [0., -1.]])
        memory.observe(fourth, optimizer)
        event, _ = memory.observe(fourth, optimizer)
        self.assertEqual(event['type'], 'replace')
        self.assertEqual(event['slot'], 1)  # Least recently used non-source expert.
        self.assertEqual(int(memory.active.sum()), 3)
        self.assertNotIn(memory.prompts[1], optimizer.state)
        self.assertTrue(torch.equal(source_before, memory.prompts[0]))
        self.assertTrue(memory.protected[0])

    def test_capacity_deferral_and_fixed_mode(self):
        s = settings(); s['replacement_grace'] = 100
        memory = PromptMemory(2, 2, s)
        memory.initialize(np.asarray([[1., 0.], [1., .0001]]), 1, 42)
        for query in [torch.tensor([[0., 1.]])] * 2:
            memory.observe(query, None)
        memory.observe(torch.tensor([[-1., 0.]]), None)
        event, _ = memory.observe(torch.tensor([[-1., 0.]]), None)
        self.assertEqual(event['type'], 'capacity_deferred')
        self.assertEqual(int(memory.active.sum()), 2)
        fixed = PromptMemory(3, 2, settings())
        fixed.initialize(np.asarray([[1., 0.], [1., .0001]]), 1, 42)
        for _ in range(5):
            fixed.observe(torch.tensor([[0., 1.]]), None, dynamic=False)
        self.assertEqual(int(fixed.active.sum()), 1)

    def test_evaluation_is_read_only_and_gradient_reaches_only_prompts(self):
        backbone = Framework(8, 8, 2, dropout=.1).eval()
        source = np.random.default_rng(42).normal(size=(8, 8)).astype('float32')
        model = DynamicPromptModel(backbone, source, settings(), 42)
        rows = [{'smiles': 'CCO', 'substructure': "{'CCO'}", 'cls_label': 0},
                {'smiles': 'CCN', 'substructure': "{'CCN'}", 'cls_label': 1}]
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        evaluate(model, batches(rows, 2), 'cpu')
        self.assertTrue(all(torch.equal(initial[k], v) for k, v in model.state_dict().items()))
        batch = next(iter(batches(rows, 2)))
        model.eval()
        self.assertTrue(torch.allclose(model(batch)[0], model.backbone(batch), atol=1e-6))
        model.train()
        logits, regularization, ids = model(batch)
        (torch.nn.functional.cross_entropy(logits, batch.y) + regularization).backward()
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        for slot, prompt in enumerate(model.memory.prompts):
            if slot in ids.tolist():
                self.assertIsNotNone(prompt.grad)
            else:
                self.assertIsNone(prompt.grad)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.memory.prompts if p.grad is not None), 0)

    def test_full_suite_three_scenarios_resume_and_restore(self):
        base = Framework(8, 8, 2, dropout=.1).eval()
        rows = [{'smiles': 'C' * (i + 1), 'substructure': str({'C' * (i + 1)}),
                 'cls_label': i % 2} for i in range(24)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original, source = root / 'original.json', root / 'source.json'
            write_json(original, {'split': {'train': rows}})
            write_json(source, {'split': {'train': rows[:8]}})
            write_json(root / 'args.json', {'train_path': str(source), 'dropout': .1})
            checkpoint = root / 'best_model.pth'
            torch.save({'main': base.state_dict()}, checkpoint)
            plan = {'scenario': 'abrupt', 'partition': {
                'source_indices': list(range(8)), 'remaining_indices': list(range(8, 24)),
                'seed': 42, 'data_path': str(original), 'data_sha256': digest(original),
                'source_sha256': digest(source)},
                'checkpoint_path': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
                'environments': [
                    {'environment': 0, 'adapt_indices': [8, 9, 10, 11], 'eval_indices': [12, 13, 14, 15]},
                    {'environment': 1, 'adapt_indices': [16, 17, 18, 19], 'eval_indices': [20, 21, 22, 23]}],
                'episodes': [{'environment': 0, 'indices': [8, 9, 10, 11]},
                             {'environment': 1, 'indices': [16, 17, 18, 19]}]}
            manifest = root / 'manifest.json'; write_json(manifest, plan)
            config = {'manifest': str(manifest), 'output': str(root / 'suite'), 'device': 'cpu',
                      'seeds': [42], 'scenarios': ['abrupt', 'gradual', 'repeat'],
                      'methods': ['frozen', 'finetune', 'fixed', 'dynamic'], 'batch_size': 2,
                      'cpu_threads': 1, 'gradual_windows': 4, 'eval_every_batches': 2,
                      'learning_rate': .001, 'finetune_learning_rate': .0001,
                      'teacher_weight': .1, 'anchor_weight': 1., 'gradient_clip': 1.,
                      'memory': settings(), 'plots': False}
            results = execute(config)
            self.assertEqual(len(results), 12)
            self.assertTrue((root / 'suite/COMPLETE.json').exists())
            for result in results:
                if result['method'] == 'frozen':
                    self.assertAlmostEqual(result['auc_change'], 0., places=6)
                if result['memory']:
                    self.assertLessEqual(result['memory']['active'], settings()['capacity'])
            self.assertEqual(results, execute(config, resume=True))
            # Simulate a failure after a completed episode save, before result finalization.
            run = root / 'suite/abrupt/seed_42/dynamic'
            status = read_json(run / 'status.json'); status['complete'] = False
            write_json(run / 'status.json', status)
            restored = execute(config, resume=True)
            original_result = next(r for r in results if r['scenario'] == 'abrupt' and r['method'] == 'dynamic')
            restored_result = next(r for r in restored if r['scenario'] == 'abrupt' and r['method'] == 'dynamic')
            self.assertEqual(original_result['final_auc'], restored_result['final_auc'])
            self.assertEqual(original_result['memory'], restored_result['memory'])
            # An actual interrupted episode sequence resumes from the saved first boundary.
            interrupted = copy.deepcopy(config)
            interrupted.update(output=str(root / 'interrupted'), scenarios=['abrupt'], methods=['dynamic'])

            def stop_after_first_episode(path, model, optimizer, progress, signature):
                save_checkpoint(path, model, optimizer, progress, signature)
                if progress['completed_episode'] == 1:
                    raise RuntimeError('Simulated interruption')

            with patch('run_dynamic.save_checkpoint', side_effect=stop_after_first_episode):
                with self.assertRaisesRegex(RuntimeError, 'Simulated interruption'):
                    execute(interrupted)
            resumed = execute(interrupted, resume=True)
            self.assertEqual(resumed[0]['final_auc'], original_result['final_auc'])
            reference_state = torch.load(run / 'latest.pt', weights_only=True)['model']
            resumed_path = root / 'interrupted/abrupt/seed_42/dynamic/latest.pt'
            resumed_state = torch.load(resumed_path, weights_only=True)['model']
            self.assertTrue(all(torch.equal(value, resumed_state[key]) for key, value in reference_state.items()))
            # The prediction-only CLI restores the same result and does not update memory.
            cli = subprocess.run([sys.executable, 'evaluate_dynamic.py', '--run', str(run),
                                  '--device', 'cpu', '--output', str(root / 'reevaluated.json')],
                                 cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
            self.assertEqual(cli.returncode, 0, cli.stdout + cli.stderr)
            exported = read_json(root / 'reevaluated.json')
            last_metrics = read_json(run / 'metrics.json')[-1]['metrics']
            self.assertEqual(exported['metrics'], last_metrics)
            altered = copy.deepcopy(config); altered['learning_rate'] = .01
            with self.assertRaises(FileExistsError):
                execute(altered, resume=True)

    def test_invalid_capacity_is_rejected(self):
        config = read_json(Path(__file__).resolve().parents[1] / 'configs/dynamic_prompt.json')
        config['memory']['capacity'] = config['memory']['initial_prompts']
        with self.assertRaises(ValueError):
            validate_config(config)


if __name__ == '__main__':
    unittest.main()
