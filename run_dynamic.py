"""One-command, fixed-budget dynamic Prompt experiments and reporting.

No hyperparameter search, test-based early stopping, or task IDs for learners.
Resume is supported at completed episode boundaries (an interrupted episode
is replayed from its saved beginning). All outputs go to a separate directory.
"""
import argparse
import copy
import csv
import hashlib
import io
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from continual.core import (read_json, write_json, digest, batches, load_backbone,
                            encode_rows, PromptLearner, evaluate, source_class_weights)
from continual.dynamic import DynamicPromptModel
from continual.protocol import (groups_for, stratified_groups,
                                validate_source_checkpoint, validate_stream)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_config(config):
    required = ['manifest', 'output', 'device', 'seeds', 'scenarios', 'methods', 'memory']
    if any(key not in config for key in required):
        raise ValueError(f'Configuration requires {required}')
    for key, allowed in [('scenarios', {'abrupt', 'gradual', 'repeat'}),
                         ('methods', {'frozen', 'head', 'finetune', 'shared', 'fixed', 'dynamic'})]:
        values = config[key]
        if not values or len(values) != len(set(values)) or not set(values) <= allowed:
            raise ValueError(f'Invalid or duplicate {key}')
    if not config['seeds'] or len(config['seeds']) != len(set(config['seeds'])):
        raise ValueError('Provide distinct integer seeds')
    if any(not isinstance(seed, int) for seed in config['seeds']):
        raise ValueError('Seeds must be integers')
    for key in ['batch_size', 'cpu_threads', 'gradual_windows', 'eval_every_batches']:
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ['learning_rate', 'finetune_learning_rate', 'gradient_clip']:
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if not all(np.isfinite(config[key]) and config[key] >= 0 for key in ['teacher_weight', 'anchor_weight']):
        raise ValueError('Loss weights must be nonnegative')
    memory = config['memory']
    for key in ['initial_prompts', 'capacity', 'patience', 'novel_buffer',
                'allocation_cooldown', 'replacement_grace']:
        if not isinstance(memory[key], int) or memory[key] < 1:
            raise ValueError(f'memory.{key} must be a positive integer')
    if memory['capacity'] <= memory['initial_prompts']:
        raise ValueError('Capacity must exceed initial_prompts to allow dynamic growth')
    if not 0 < memory['source_quantile'] < 1 or not 0 < memory['novel_fraction'] <= 1:
        raise ValueError('Invalid quantile or novelty fraction')
    if not 0 <= memory['minimum_coherence'] <= 1 or memory['minimum_distance'] <= 0:
        raise ValueError('Invalid coherence or minimum distance')
    if memory['distance_margin'] < 0:
        raise ValueError('distance_margin must be nonnegative')


def schedules(base, rows, scenarios, windows):
    """Curator-only environment metadata; the model never receives these IDs."""
    plans = {}
    seed = base['partition']['seed']
    for scenario in scenarios:
        plan = copy.deepcopy(base)
        rng = np.random.default_rng(seed)
        if scenario == 'abrupt':
            episodes = [{'environment': env['environment'], 'indices': list(env['adapt_indices'])}
                        for env in base['environments']]
        elif scenario == 'repeat':
            first, second = [], []
            for env in base['environments']:
                indices = env['adapt_indices']
                records = [rows[i] for i in indices]
                a, b = stratified_groups(records, groups_for(records), .5, seed)
                first.append({'environment': env['environment'], 'indices': [indices[i] for i in a]})
                second.append({'environment': env['environment'], 'indices': [indices[i] for i in b]})
            episodes = first + second
        else:
            ordered = [(position + rng.uniform(-.75, .75), index)
                       for position, env in enumerate(base['environments']) for index in env['adapt_indices']]
            indices = [i for _, i in sorted(ordered)]
            episodes = [{'environment': None, 'indices': block.tolist()}
                        for block in np.array_split(indices, min(windows, len(indices)))]
        if scenario != 'gradual':
            for episode in episodes:
                rng.shuffle(episode['indices'])
        plan.update(scenario=scenario, episodes=episodes)
        validate_stream(plan, rows)
        plans[scenario] = plan
    return plans


def construct_model(checkpoint, features, config, seed, method, device):
    backbone, _ = load_backbone(checkpoint)
    if method in ('dynamic', 'fixed', 'shared'):
        model = DynamicPromptModel(backbone, features, config['memory'], seed, method)
    else:
        model = PromptLearner(backbone, mode=method, pool_size=1)
    return model.to(device)


def metric_summary(metrics, history, seen):
    valid = [v['auc'] for v in metrics.values() if v['auc'] is not None]
    seen_values = [metrics[e]['auc'] for e in seen if metrics[e]['auc'] is not None]
    forgetting = []
    for env in seen:
        values = [entry['metrics'][env]['auc'] for entry in history
                  if env in entry['seen_environments'] and entry['metrics'][env]['auc'] is not None]
        current = metrics[env]['auc']
        if current is not None:
            forgetting.append(max(values + [current]) - current)
    return {'mean_auc': float(np.mean(valid)) if valid else None,
            'mean_balanced_accuracy': float(np.mean([v['balanced_accuracy'] for v in metrics.values()])),
            'seen_mean_auc': float(np.mean(seen_values)) if seen_values else None,
            'seen_forgetting': float(np.mean(forgetting)) if forgetting else None}


def tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def save_checkpoint(path, model, optimizer, progress, run_signature):
    payload = {'format': 'moleood_dynamic_v1', 'model': model.state_dict(),
               'optimizer': optimizer.state_dict() if optimizer else None,
               'progress': progress, 'signature': run_signature,
               'torch_rng': torch.get_rng_state(),
               'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def run_one(plan, rows, features, config, method, seed, directory, resume):
    signature = fingerprint({'config': config, 'plan': plan, 'method': method, 'seed': seed})
    status_path = directory / 'status.json'
    if directory.exists():
        if not resume:
            raise FileExistsError(directory)
        if read_json(directory / 'run.json')['signature'] != signature:
            raise ValueError(f'Cannot resume changed configuration: {directory}')
        if status_path.exists() and read_json(status_path)['complete']:
            return read_json(directory / 'result.json')
    else:
        directory.mkdir(parents=True)
        write_json(directory / 'run.json', {'signature': signature, 'method': method,
                                           'seed': seed, 'config': config, 'plan': plan})
    seed_all(seed)
    device = torch.device(config['device'])
    model = construct_model(plan['checkpoint_path'], features, config, seed, method, device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    lr = config['finetune_learning_rate'] if method == 'finetune' else config['learning_rate']
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=0) if parameters else None
    weights = source_class_weights(rows, plan['partition']['source_indices'], device)
    write_json(directory / 'loss_config.json', {
        'class_weights': weights.cpu().tolist(), 'weight_fit_split': 'source_only',
        'learning_rate': lr, 'teacher_weight': config['teacher_weight'] if isinstance(model, DynamicPromptModel) else 0.,
        'anchor_weight': config['anchor_weight'] if isinstance(model, DynamicPromptModel) else 0.,
        'property_head_frozen': isinstance(model, DynamicPromptModel) or method == 'frozen',
        'one_pass': True, 'optimizer_steps_per_batch': 1 if optimizer else 0})
    evaluations = {str(env['environment']): batches([rows[i] for i in env['eval_indices']], config['batch_size'])
                   for env in plan['environments']}
    owner = {index: str(env['environment']) for env in plan['environments'] for index in env['adapt_indices']}
    progress = {'completed_episode': 0, 'batches': 0, 'samples': 0, 'seen': [],
                'history': [], 'events': [], 'online_labels': [], 'online_probabilities': [],
                'elapsed_seconds': 0., 'training_loss_sum': 0., 'training_loss_count': 0}
    checkpoint_path = directory / 'latest.pt'
    if resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if state['signature'] != signature:
            raise ValueError('Checkpoint signature mismatch')
        model.load_state_dict(state['model'], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(state['optimizer'])
        progress = state['progress']
        torch.set_rng_state(state['torch_rng'])
        if torch.cuda.is_available() and state['cuda_rng']:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        # Construction and source clustering cannot perturb training dropout RNG differently by method.
        seed_all(seed)
    started = time.perf_counter()

    def record(episode, position):
        metrics = {env: evaluate(model, loader, device) for env, loader in evaluations.items()}
        summary = metric_summary(metrics, progress['history'], set(progress['seen']))
        entry = {'episode': episode, 'position': position, 'samples': progress['samples'],
                 'batches': progress['batches'], 'seen_environments': list(progress['seen']),
                 'metrics': metrics, 'summary': summary,
                 'training_loss_since_previous_evaluation': (
                     progress['training_loss_sum'] / progress['training_loss_count']
                     if progress['training_loss_count'] else None),
                 'memory': model.memory.snapshot() if isinstance(model, DynamicPromptModel) else None}
        progress['history'].append(entry)
        progress['training_loss_sum'] = 0.
        progress['training_loss_count'] = 0
        write_json(directory / 'metrics.json', progress['history'])
        print(f'{plan["scenario"]}/{method}/seed{seed}: samples={progress["samples"]}, '
              f'AUC={summary["mean_auc"]}, '
              f'active={entry["memory"]["active"] if entry["memory"] else 0}', flush=True)

    if not progress['history']:
        record(0, 'initial')
        save_checkpoint(checkpoint_path, model, optimizer, progress, signature)
    for number, episode in enumerate(plan['episodes'], 1):
        if number <= progress['completed_episode']:
            continue
        loader = batches([rows[i] for i in episode['indices']], config['batch_size'])
        offset = 0
        for data in loader:
            data = data.to(device)
            # Predict before learning this batch. These are online scores, not held-out test scores.
            model.eval()
            with torch.no_grad():
                before, _, _ = model(data)
                progress['online_probabilities'].extend(before.softmax(-1)[:, 1].cpu().tolist())
                progress['online_labels'].extend(data.y.view(-1).cpu().tolist())
            if isinstance(model, DynamicPromptModel):
                query, _, teacher = model.context(data)
                event, _ = model.memory.observe(query, optimizer, dynamic=(method == 'dynamic'))
                if event:
                    progress['events'].append(event)
                    print(f'  Memory event: {event}', flush=True)
            if optimizer is not None:
                model.train()
                logits, anchor, _ = model(data)
                loss = F.cross_entropy(logits, data.y.view(-1), weight=weights)
                if isinstance(model, DynamicPromptModel):
                    consistency = F.kl_div(logits.log_softmax(-1), teacher.softmax(-1), reduction='batchmean')
                    loss = loss + config['teacher_weight'] * consistency + config['anchor_weight'] * anchor
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite loss at {directory}, batch {progress["batches"]}')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip'])
                # Unselected prompt Parameters have grad=None, so Adam leaves them untouched.
                optimizer.step()
                progress['training_loss_sum'] += float(loss.detach()) * len(data.y)
                progress['training_loss_count'] += len(data.y)
            current_indices = episode['indices'][offset:offset + len(data.y)]
            progress['seen'] = sorted(set(progress['seen']) | {owner[i] for i in current_indices})
            offset += len(data.y)
            progress['samples'] += len(data.y)
            progress['batches'] += 1
            if progress['batches'] % config['eval_every_batches'] == 0:
                record(number, 'interval')
        if progress['history'][-1]['samples'] != progress['samples']:
            record(number, 'episode_end')
        progress['completed_episode'] = number
        progress['elapsed_seconds'] += time.perf_counter() - started
        started = time.perf_counter()
        save_checkpoint(checkpoint_path, model, optimizer, progress, signature)
        write_json(directory / 'events.json', progress['events'])
        write_json(status_path, {'complete': False, 'completed_episode': number})
    initial = progress['history'][0]['summary']
    final = progress['history'][-1]['summary']
    memory = model.memory.snapshot() if isinstance(model, DynamicPromptModel) else None
    stats = {'total_parameters': sum(p.numel() for p in model.parameters()),
             'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
             'full_checkpoint_bytes': checkpoint_path.stat().st_size,
             'elapsed_seconds': progress['elapsed_seconds'],
             'optimizer_tensor_bytes': tensor_bytes(optimizer.state_dict()) if optimizer else 0}
    if memory:
        torch.save({'memory': model.memory.state_dict(), 'settings': config['memory'],
                    'checkpoint_sha256': plan['checkpoint_sha256']}, directory / 'prompt_memory.pt')
        stats.update(active_prompt_parameters=memory['active'] * model.memory.keys.shape[1],
                     reserved_prompt_parameters=sum(p.numel() for p in model.memory.prompts),
                     memory_tensor_bytes=tensor_bytes(model.memory.state_dict()),
                     memory_file_bytes=(directory / 'prompt_memory.pt').stat().st_size)
    online_auc = roc_auc_score(progress['online_labels'], progress['online_probabilities'])
    result = {'scenario': plan['scenario'], 'method': method, 'seed': seed,
              'initial_auc': initial['mean_auc'], 'final_auc': final['mean_auc'],
              'auc_change': final['mean_auc'] - initial['mean_auc'],
              'final_balanced_accuracy': final['mean_balanced_accuracy'],
              'seen_forgetting': final['seen_forgetting'], 'prequential_auc': float(online_auc),
              'time_average_eval_auc': float(np.mean([h['summary']['mean_auc'] for h in progress['history'][1:]])),
              'allocations': sum(e['type'] == 'allocate' for e in progress['events']),
              'replacements': sum(e['type'] == 'replace' for e in progress['events']),
              'capacity_deferrals': sum(e['type'] == 'capacity_deferred' for e in progress['events']),
              'memory': memory, 'resources': stats,
              'note': 'Fixed protocol development experiment; no automatic performance claim'}
    switching, visited = [], set()
    for number, episode in enumerate(plan['episodes'], 1):
        entries = [h for h in progress['history'] if h['episode'] == number]
        previous = [h for h in progress['history'] if h['episode'] < number][-1]
        environments = sorted({owner[i] for i in episode['indices']})
        for env in environments:
            before = previous['metrics'][env]['auc']
            first = entries[0]['metrics'][env]['auc']
            end = entries[-1]['metrics'][env]['auc']
            switching.append({'episode': number, 'environment': env,
                              'is_revisit': env in visited,
                              'before_auc': before, 'first_evaluation_auc': first, 'end_auc': end,
                              'drop_at_first_evaluation': before - first if before is not None and first is not None else None,
                              'recovery_from_first_evaluation': end - first if end is not None and first is not None else None,
                              'first_evaluation_after_samples': entries[0]['samples'] - previous['samples'],
                              'note': 'Measured at actual evaluation points, not an instantaneous switch response'})
        visited.update(environments)
    write_json(directory / 'switching.json', switching)
    write_json(directory / 'result.json', result)
    write_json(status_path, {'complete': True, 'completed_episode': len(plan['episodes'])})
    return result


def report(root, results, plots=True):
    fields = ['scenario', 'method', 'seed', 'initial_auc', 'final_auc', 'auc_change',
              'final_balanced_accuracy', 'seen_forgetting', 'prequential_auc',
              'allocations', 'replacements', 'capacity_deferrals']
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=fields, extrasaction='ignore')
    writer.writeheader(); writer.writerows(results)
    (root / 'summary.csv').write_text(text.getvalue(), encoding='utf-8-sig')
    write_json(root / 'summary.json', results)
    lines = ['# Dynamic Prompt 实验结果', '',
             '以下为固定配置运行结果，不按评价成绩自动调参。AUC 与变化量均以百分点显示。', '',
             '| 数据流 | 方法 | 种子 | 初始 AUC | 最终 AUC | 变化 | 新增 | 替换 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for result in results:
        lines.append(f'| {result["scenario"]} | {result["method"]} | {result["seed"]} | '
                     f'{100*result["initial_auc"]:.2f} | {100*result["final_auc"]:.2f} | '
                     f'{100*result["auc_change"]:+.2f} | {result["allocations"]} | {result["replacements"]} |')
    groups = {}
    for item in results:
        groups.setdefault((item['scenario'], item['method']), []).append(item['final_auc'])
    lines += ['', '## 按适应随机种子汇总', '', '| 数据流 | 方法 | n | 最终 AUC 均值 ± 样本标准差 |',
              '|---|---|---:|---:|']
    for (scenario, method), values in groups.items():
        std = f'{100*np.std(values, ddof=1):.2f}' if len(values) > 1 else '不适用（单种子）'
        lines.append(f'| {scenario} | {method} | {len(values)} | {100*np.mean(values):.2f} ± {std} |')
    lines += ['', '## 阅读限制', '',
              '- 比较同一数据流、同一初始检查点的方法；不能与原论文 OOD AUC 直接比较。',
              '- 若动态方法没有优于不适应或固定池，应报告没有观察到收益。',
              '- 新增/替换计数为零也是有效现象，不能人为强制新增来制造方法有效的证据。',
              '- 多种子仅改变适应过程；初始模型与数据划分固定，未覆盖初始预训练的方差。',
              '- 当前评价集已多次用于开发且负样本少，不是独立最终确认性测试。',
              '- prequential_auc 是适应样本在更新前的预测成绩，不能与独立评价 AUC 混称。',
              '- time_average_eval_auc 对保存的评价时点等权，不是按时间/样本积分。',
              '- 模型记忆容量有上限；日志、阶段评价记录和完整检查点的磁盘占用不属于该上限。']
    (root / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if not plots:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for scenario in sorted({r['scenario'] for r in results}):
            fig, axis = plt.subplots(figsize=(8, 4))
            for item in results:
                if item['scenario'] != scenario:
                    continue
                history = read_json(root / scenario / f'seed_{item["seed"]}' / item['method'] / 'metrics.json')
                axis.plot([h['samples'] for h in history], [100*h['summary']['mean_auc'] for h in history],
                          marker='o', markersize=3, label=f'{item["method"]}/s{item["seed"]}')
            axis.set(xlabel='Adaptation samples processed', ylabel='Macro ROC-AUC (%)', title=scenario)
            axis.legend(); axis.grid(alpha=.2); fig.tight_layout()
            fig.savefig(root / f'{scenario}_auc.png', dpi=160); plt.close(fig)
            fig, axis = plt.subplots(figsize=(8, 3))
            for item in results:
                if item['scenario'] != scenario or item['memory'] is None:
                    continue
                history = read_json(root / scenario / f'seed_{item["seed"]}' / item['method'] / 'metrics.json')
                axis.step([h['samples'] for h in history], [h['memory']['active'] for h in history],
                          where='post', label=f'{item["method"]}/s{item["seed"]}')
            axis.set(xlabel='Adaptation samples processed', ylabel='Active prompts', title=scenario)
            if any(i['scenario'] == scenario and i['memory'] is not None for i in results):
                axis.legend()
            axis.grid(alpha=.2); fig.tight_layout()
            fig.savefig(root / f'{scenario}_memory.png', dpi=160); plt.close(fig)
    except ImportError:
        (root / 'plots_unavailable.txt').write_text('Install matplotlib to generate optional plots.', encoding='utf-8')


def execute(config, resume=False, check_only=False):
    validate_config(config)
    root = Path(config['output']).resolve()
    base_path = Path(config['manifest']).resolve()
    base = read_json(base_path)
    if digest(base['partition']['data_path']) != base['partition']['data_sha256']:
        raise ValueError('Dataset fingerprint changed')
    if digest(base['checkpoint_path']) != base['checkpoint_sha256']:
        raise ValueError('Initial checkpoint fingerprint changed')
    validate_source_checkpoint(base['checkpoint_path'], base['partition'])
    rows = read_json(base['partition']['data_path'])['split']['train']
    validate_stream(base, rows)
    ids = [env['environment'] for env in base['environments']]
    if len(ids) < 2 or len(ids) != len(set(ids)):
        raise ValueError('Require at least two distinct evaluation environments')
    for env in base['environments']:
        for split in ('adapt_indices', 'eval_indices'):
            if {int(rows[i]['cls_label']) for i in env[split]} != {0, 1}:
                raise ValueError(f'Environment {env["environment"]}/{split} needs both classes')
    source = [rows[i] for i in base['partition']['source_indices']]
    if len(source) < config['memory']['capacity']:
        raise ValueError('Capacity exceeds available source records')
    plans = schedules(base, rows, config['scenarios'], config['gradual_windows'])
    if check_only:
        backbone, _ = load_backbone(base['checkpoint_path'])
        del backbone
        print(f'Preflight passed: source={len(source)}, environments={len(base["environments"])}, '
              f'fixed runs={len(plans)*len(config["methods"])*len(config["seeds"])}. '
              'No output directory created and no training started.', flush=True)
        return []
    torch.set_num_threads(config['cpu_threads'])
    device = torch.device(config['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; activate piptorch or set device=cpu')
    identity = {'config': config, 'base_manifest_sha256': digest(base_path)}
    if root.exists():
        if not resume or read_json(root / 'suite.json') != identity:
            raise FileExistsError('Use a new output directory or --resume with unchanged settings')
    else:
        root.mkdir(parents=True)
        write_json(root / 'suite.json', identity)
    if (root / 'COMPLETE.json').exists():
        (root / 'COMPLETE.json').unlink()
    for name, plan in plans.items():
        write_json(root / 'streams' / f'{name}.json', plan)
    print(f'Fixed run budget: {len(plans)*len(config["methods"])*len(config["seeds"])}. '
          'No automatic search or extra runs.', flush=True)
    backbone, _ = load_backbone(base['checkpoint_path'], device)
    features = encode_rows(backbone, source, device, config['batch_size'])
    del backbone
    results = []
    for scenario, plan in plans.items():
        for seed in config['seeds']:
            for method in config['methods']:
                directory = root / scenario / f'seed_{seed}' / method
                result = run_one(plan, rows, features, config, method, seed, directory, resume)
                results.append(result)
                report(root, results, config.get('plots', True))
    write_json(root / 'COMPLETE.json', {'runs': len(results), 'complete': True})
    print(f'All fixed runs complete. Read {root / "SUMMARY.md"}', flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default='configs/dynamic_prompt.json')
    parser.add_argument('--output', help='Override output directory')
    parser.add_argument('--device', help='Override device, e.g. cpu or cuda:0')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check', action='store_true', help='Read-only input/shape/protocol validation')
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    import os
    os.chdir(project)
    config = read_json(args.config)
    if args.output:
        config['output'] = args.output
    if args.device:
        config['device'] = args.device
    execute(config, args.resume, args.check)
