"""Full-train EC50-Scaffold MoleOOD PyG baseline: no TTA, prompts or warm start.

Restores the two-stage training from MoleOOD-L2P/main.py. This reproduces
its local PyG variant, not an exact copy of the official DrugOOD backend.
"""
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import torch
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch_geometric.seed import seed_everything

from models.dataset import LBAPDatasetWithSub

from models import KLDist, MeanLoss, DeviationLoss, discrete_gaussian, evaluate
from torch_geometric.data import Batch
from torch_geometric.nn import MLP
from models.mygin import MyGIN
from models.Framework import AttentionAgger, ConditionalGnn, DomainClassifier
from models.utils import pyg_batch_from_subgraphs, get_subgraph_batch_index


def collate_molecules(items):
    """Keep each molecule's variable-length list of subgraphs intact."""
    batch = Batch.from_data_list(items, exclude_keys=['subs'])
    batch.subs = [item.subs for item in items]
    return batch


class Framework(torch.nn.Module):
    def __init__(self, base_dim, sub_dim, num_class, dropout=0.1):
        super().__init__()
        if base_dim != sub_dim:
            raise ValueError('Graph and substructure dimensions must match')
        settings = dict(num_node_emb_list=[39], num_edge_emb_list=[10],
                        num_layers=4, JK='last', dropout=0.1, readout='mean')
        self.base_model = MyGIN(emb_dim=base_dim, **settings)
        self.sub_model = MyGIN(emb_dim=sub_dim, **settings)
        self.aggr = AttentionAgger(base_dim, sub_dim, max(base_dim, sub_dim))
        self.predictor = MLP([sub_dim, sub_dim, num_class],
                             dropout=float(dropout), norm=None)

    def forward(self, data):
        _, main = self.base_model(data.x, data.edge_index, data.edge_attr, data.batch)
        sub_batch = pyg_batch_from_subgraphs(data.subs)
        if sub_batch is None:
            return self.predictor(main)
        sub_batch = sub_batch.to(data.x.device)
        _, features = self.sub_model(sub_batch.x, sub_batch.edge_index,
                                     sub_batch.edge_attr, sub_batch.batch)
        owners = get_subgraph_batch_index(data.subs).to(data.x.device)
        sums = features.new_zeros(main.shape[0], features.shape[-1])
        sums.index_add_(0, owners, features)
        counts = torch.bincount(owners, minlength=main.shape[0]).clamp_min(1)
        return self.predictor(main + sums / counts[:, None])


def init_args():
    parser = argparse.ArgumentParser('Experiment for Drugood Dataset (PyG Version)')

    parser.add_argument('--train_path', type=str, default='data/ec50/lbap_core_ec50_scaffold_brics.json')
    parser.add_argument('--val_path', type=str, default='data/ec50/lbap_core_ec50_scaffold_brics.json')
    parser.add_argument('--test_path', type=str, default='data/ec50/lbap_core_ec50_scaffold_brics.json')

    parser.add_argument('--emb_dim', default=128, type=int, help='GNN hidden dimension')
    parser.add_argument('--num_class', default=2, type=int, help='number of classes')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--batch_size', default=128, type=int)

    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--num_domain', default=20, type=int)
    parser.add_argument('--epoch_main', default=50, type=int)
    parser.add_argument('--epoch_ast', default=20, type=int)
    parser.add_argument('--lambda_loss', default=1.0, type=float)
    parser.add_argument('--dist', default='uniform', type=str)

    parser.add_argument('--output_dir', default=None, help='New directory for this run; existing results are protected')
    return parser.parse_args()


def make_log(args):
    from datetime import datetime
    from pathlib import Path
    dataset_name = Path(args.train_path).stem
    run_name = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_dir = Path(args.output_dir) if args.output_dir else (
        Path('log') / dataset_name / 'baseline_no_tta' / f'seed_{args.seed}_{run_name}'
    )
    if log_dir.exists() and any(log_dir.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {log_dir}')
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / 'args.json').write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(f'[INFO] Output directory: {log_dir.resolve()}')
    return str(log_dir), 'metrics.json'


def get_prior(num_domain, dtype='uniform'):
    if dtype == 'uniform':
        return torch.ones(num_domain) / num_domain
    else:
        return discrete_gaussian(num_domain)


def eval_one_epoch(model, loader, device, verbose=True):
    model.eval()
    result_all, gt_all = [], []

    for data in (tqdm(loader) if verbose else loader):
        data = data.to(device)

        with torch.no_grad():
            logits = model(data)
            result = torch.softmax(logits, dim=-1)

            result_all.append(result.detach().cpu())
            gt_all.append(data.y.cpu().view(-1))

    result_all = torch.cat(result_all, dim=0)
    gt_all = torch.cat(gt_all, dim=0)
    return evaluate(pred=result_all, gt=gt_all, metric=['auc', 'accuracy'])


if __name__ == '__main__':
    args = init_args()
    if args.epoch_ast < 1 or args.epoch_main < 1:
        raise ValueError('epoch_ast and epoch_main must both be positive')
    if args.batch_size < 2 or args.num_domain < 1:
        raise ValueError('batch_size must be >= 2 and num_domain must be positive')
    from pathlib import Path
    project = Path(__file__).resolve().parent
    for key in ('train_path', 'val_path', 'test_path'):
        value = Path(getattr(args, key))
        value = value if value.is_absolute() else project / value
        if not value.is_file():
            raise FileNotFoundError(value)
        setattr(args, key, str(value))
    if args.output_dir is not None:
        value = Path(args.output_dir)
        args.output_dir = str(value if value.is_absolute() else project / value)
    os.chdir(project)
    if args.device >= 0 and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use --device -1 for CPU training.')
    print(args)
    seed_everything(args.seed)

    log_dir, log_name = make_log(args)
    device = torch.device('cpu') if args.device < 0 else torch.device(f'cuda:{args.device}')

    print("[INFO] Building Datasets...")
    train_set = LBAPDatasetWithSub(args.train_path, split='train')
    valid_set = LBAPDatasetWithSub(args.val_path, split='ood_val')
    test_set  = LBAPDatasetWithSub(args.test_path, split='ood_test')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_molecules)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_molecules)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_molecules)

    print("[INFO] Building Models...")
    main_model = Framework(
        dropout=args.dropout, base_dim=args.emb_dim, sub_dim=args.emb_dim, num_class=args.num_class
    ).to(device)

    conditional_gnn = ConditionalGnn(
        emb_dim=args.emb_dim, backend_dim=args.emb_dim, num_domain=args.num_domain, num_class=args.num_class
    ).to(device)

    domain_classifier = DomainClassifier(
        backend_dim=args.emb_dim, num_task=1, num_domain=args.num_domain
    ).to(device)

    optimizer_main = AdamW(main_model.parameters(), lr=args.lr)
    optimizer_dom = AdamW(domain_classifier.parameters(), lr=args.lr)
    optimizer_con = AdamW(conditional_gnn.parameters(), lr=args.lr)

    CLSLoss = torch.nn.CrossEntropyLoss()
    mean_loss = MeanLoss(CLSLoss)
    dev_loss = DeviationLoss(activation='abs', reduction='mean')
    prior = get_prior(args.num_domain, args.dist).to(device)

    loss_curv, min_loss, best_ep, best_para = [], None, None, {}

    for ep in range(args.epoch_ast):
        print(f'\n[INFO] Training Assistant Models - Epoch {ep}')
        conditional_gnn.train()
        domain_classifier.train()
        Eqs, ELs = [], []

        for data in tqdm(train_loader):
            data = data.to(device) # PyG 自动把图、标签等全搬到 GPU
            batch_size = data.y.size(0)

            q_e = torch.softmax(domain_classifier(data), dim=-1)
            losses = []

            for dom in range(args.num_domain):
                domain_info = torch.ones(batch_size, dtype=torch.long, device=device) * dom
                p_ye = conditional_gnn(data, domain_info)
                loss = cross_entropy(p_ye, data.y.view(-1), reduction='none')
                losses.append(loss)

            losses = torch.stack(losses, dim=1)
            Eq = torch.mean(torch.sum(q_e * losses, dim=-1))
            ELBO = Eq + KLDist(q_e, prior)

            optimizer_con.zero_grad()
            optimizer_dom.zero_grad()
            ELBO.backward()
            optimizer_con.step()
            optimizer_dom.step()

            Eqs.append(Eq.item())
            ELs.append(ELBO.item())

        mean_ELBO = np.mean(ELs)
        print(f'[INFO] Eq: {np.mean(Eqs):.4f}, ELBO: {mean_ELBO:.4f}')
        loss_curv.append((np.mean(Eqs), mean_ELBO))

        if best_ep is None or mean_ELBO < min_loss:
            min_loss, best_ep = mean_ELBO, ep
            best_para = {
                'con': deepcopy(conditional_gnn.state_dict()),
                'dom': deepcopy(domain_classifier.state_dict())
            }

    print(f'[INFO] Using the best Assistant Model from Epoch {best_ep}')
    domain_classifier.load_state_dict(best_para['dom'])
    conditional_gnn.load_state_dict(best_para['con'])
    domain_classifier.eval()
    conditional_gnn.eval()

    valid_curv, test_curv, max_valid_auc = {}, {}, None
    model_path = os.path.join(log_dir, f'best_model.pth')

    for ep in range(args.epoch_main):
        print(f'\n[INFO] Training Main Model - Epoch {ep}')
        main_model.train()

        for data in tqdm(train_loader):
            data = data.to(device)
            batch_size = data.y.size(0)
            cond_result = []

            with torch.no_grad():
                for dom in range(args.num_domain):
                    domain_info = torch.ones(batch_size, dtype=torch.long, device=device) * dom
                    cond_term = cross_entropy(
                        conditional_gnn(data, domain_info),
                        data.y.view(-1), reduction='none'
                    )
                    cond_result.append(cond_term)
                cond_result = torch.stack(cond_result, dim=0)
                cond_result = torch.matmul(prior, cond_result)

                p_e = domain_classifier(data)
                group = torch.argmax(p_e, dim=-1)

            pred = main_model(data)

            mean_term = mean_loss(pred, data.y.view(-1), group)
            this_loss = cross_entropy(pred, data.y.view(-1), reduction='none')

            dev_term = dev_loss(this_loss, cond_result)
            loss = args.lambda_loss * mean_term + dev_term

            optimizer_main.zero_grad()
            loss.backward()
            optimizer_main.step()

        val_perf = eval_one_epoch(main_model, valid_loader, device)
        test_perf = eval_one_epoch(main_model, test_loader, device)

        for k, v in val_perf.items():
            if k not in valid_curv:
                valid_curv[k], test_curv[k] = [], []
            valid_curv[k].append(val_perf[k])
            test_curv[k].append(test_perf[k])

        print(f'[INFO] Valid: {val_perf}')
        print(f'[INFO] Test : {test_perf}')

        if max_valid_auc is None or val_perf['auc'] > max_valid_auc:
            torch.save({
                'main': main_model.state_dict(),
                'dom': best_para['dom'],
                'con': best_para['con']
            }, model_path)
            max_valid_auc = val_perf['auc']

        with open(os.path.join(log_dir, log_name), 'w', encoding='utf-8') as fout:
            json.dump({'valid': valid_curv, 'test': test_curv}, fout, indent=2)

    best_result = {}
    for k, v in valid_curv.items():
        pos = int(np.argmax(v))
        best_result[k] = [pos, v[pos], test_curv[k][pos]]
    print('\n[FINAL RESULT] Test performance at the best validation epoch:')
    print(best_result)
    with open(os.path.join(log_dir, 'result.json'), 'w', encoding='utf-8') as fout:
        json.dump(best_result, fout, indent=2)
