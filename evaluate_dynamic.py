"""Reload a saved complete-method checkpoint for prediction-only evaluation."""
import argparse
import json
from pathlib import Path

import torch

from continual.core import read_json, write_json, digest, batches, encode_rows, load_backbone, evaluate
from run_dynamic import construct_model


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', required=True, help='Directory containing run.json and latest.pt')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--split', choices=['environments', 'iid_test', 'ood_test'], default='environments')
    parser.add_argument('--output', help='Optional NEW JSON file; otherwise print only')
    args = parser.parse_args()
    if args.output and Path(args.output).exists():
        raise FileExistsError(args.output)
    info = read_json(Path(args.run) / 'run.json')
    config, plan = info['config'], info['plan']
    if digest(plan['partition']['data_path']) != plan['partition']['data_sha256']:
        raise ValueError('Dataset changed')
    if digest(plan['checkpoint_path']) != plan['checkpoint_sha256']:
        raise ValueError('Initial checkpoint changed')
    state = torch.load(Path(args.run) / 'latest.pt', map_location='cpu', weights_only=True)
    if state['signature'] != info['signature']:
        raise ValueError('Run/checkpoint mismatch')
    torch.set_num_threads(config['cpu_threads'])
    splits = read_json(plan['partition']['data_path'])['split']
    rows = splits['train']
    device = torch.device(args.device)
    backbone, _ = load_backbone(plan['checkpoint_path'], device)
    source = [rows[i] for i in plan['partition']['source_indices']]
    features = encode_rows(backbone, source, device, config['batch_size'])
    del backbone
    model = construct_model(plan['checkpoint_path'], features, config, info['seed'], info['method'], device)
    model.load_state_dict(state['model'], strict=True)
    groups = {str(env['environment']): [rows[i] for i in env['eval_indices']]
              for env in plan['environments']} if args.split == 'environments' else {args.split: splits[args.split]}
    # No memory.observe and no optimizer: evaluation cannot allocate, replace or train prompts.
    metrics = {key: evaluate(model, batches(records, config['batch_size']), device)
               for key, records in groups.items()}
    report = {'run': str(Path(args.run).resolve()), 'split': args.split,
              'completed_episode': state['progress']['completed_episode'], 'metrics': metrics,
              'training_performed': False}
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
