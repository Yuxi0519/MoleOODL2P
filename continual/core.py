"""Model and data utilities; no changes to the running baseline modules."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import from_smiles

from main import Framework, collate_molecules
from models.mygin import MyGIN
from models.utils import pyg_batch_from_subgraphs, get_subgraph_batch_index


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def molecule_id(smiles):
    """Group alternative SMILES of the same stereochemical molecule together."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


class Molecules(Dataset):
    """Cache graphs on CPU, safely parse substructures, preserve record order."""
    def __init__(self, rows):
        self.rows = rows
        self.cache = {}

    @staticmethod
    def graph(smiles):
        graph = from_smiles(smiles)
        graph.x = graph.x[:, :1].long().clamp(max=38)
        graph.edge_attr = graph.edge_attr[:, :1].long().clamp(max=9)
        return graph

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if index not in self.cache:
            row = self.rows[index]
            graph = self.graph(row['smiles'])
            # literal_eval avoids executing arbitrary code from a JSON field.
            subs = row.get('substructure', [])
            subs = ast.literal_eval(subs) if isinstance(subs, str) else subs
            subs = sorted(subs) if subs else [row['smiles']]
            graph.subs = [self.graph(smile) for smile in subs]
            graph.y = torch.tensor([int(row['cls_label'])], dtype=torch.long)
            # Do not attach domain_id: the learner never receives environment IDs.
            self.cache[index] = graph
        return self.cache[index].clone()


def batches(rows, batch_size=64, shuffle=False):
    return DataLoader(Molecules(rows), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_molecules)


def load_backbone(path, device='cpu'):
    # New baseline checkpoints contain tensors only. Do not unpickle arbitrary objects.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = checkpoint['main']
    dim = state['base_model.node_embeddings.0.linears.0.weight'].shape[0]
    if state['base_model.node_embeddings.0.linears.0.weight'].shape[1] != 39:
        raise ValueError('Expected the new no-TTA checkpoint with 39 input categories')
    args_path = Path(path).parent / 'args.json'
    if not args_path.is_file():
        raise ValueError('Checkpoint must have its baseline args.json beside it')
    args = read_json(args_path)
    model = Framework(dim, dim, 2, dropout=args['dropout'])
    model.load_state_dict(state, strict=True)
    return model.to(device), args


def sub_features(backbone, data, count):
    sub_batch = pyg_batch_from_subgraphs(data.subs)
    if sub_batch is None:
        return data.x.new_zeros((count, backbone.predictor.in_channels), dtype=torch.float)
    sub_batch = sub_batch.to(data.x.device)
    _, features = backbone.sub_model(sub_batch.x, sub_batch.edge_index,
                                     sub_batch.edge_attr, sub_batch.batch)
    owners = get_subgraph_batch_index(data.subs).to(data.x.device)
    sums = features.new_zeros(count, features.shape[-1])
    sums.index_add_(0, owners, features)
    counts = torch.bincount(owners, minlength=count).clamp_min(1)
    return sums / counts[:, None]


@torch.no_grad()
def encode_rows(backbone, rows, device, batch_size=64):
    backbone.eval()
    outputs = []
    for data in batches(rows, batch_size):
        data = data.to(device)
        _, graph = backbone.base_model(data.x, data.edge_index, data.edge_attr, data.batch)
        outputs.append((graph + sub_features(backbone, data, len(graph))).cpu())
    return torch.cat(outputs).numpy()


class PromptGIN(MyGIN):
    """The baseline's last-layer/mean GIN, with additive node-embedding prompts."""
    def forward(self, x, edge_index, edge_attr, batch, graph_prompt=None):
        if self.JK != 'last' or self.readout_type != 'mean':
            raise ValueError('This adapter supports the baseline last/mean settings only')
        embedded = [layer(F.one_hot(x[:, i], self.num_node_emb_list[i]).float())
                    for i, layer in enumerate(self.node_embeddings)]
        nodes = torch.stack(embedded).sum(0)
        if graph_prompt is not None:
            # Every atom in a molecule shares its selected vector; the graph is unchanged.
            nodes = nodes + graph_prompt[batch]
        for layer in self.gnn_layers:
            nodes = self.dropout(layer(nodes, edge_index, edge_attr))
        return nodes, global_mean_pool(nodes, batch)


class PromptLearner(torch.nn.Module):
    def __init__(self, backbone, mode='pool', pool_size=10, freeze_head=False):
        super().__init__()
        if mode not in ('frozen', 'head', 'shared', 'pool', 'finetune'):
            raise ValueError(mode)
        dim = backbone.base_model.node_embeddings[0].linears[0].out_features
        adapter = PromptGIN([39], [10], num_layers=4, emb_dim=dim,
                            JK='last', dropout=.1, readout='mean')
        adapter.load_state_dict(backbone.base_model.state_dict(), strict=True)
        backbone.base_model = adapter
        self.backbone, self.mode = backbone, mode
        self.backbone.requires_grad_(mode == 'finetune')
        self.backbone.aggr.requires_grad_(False)  # Historical unused attention module.
        if mode in ('head', 'shared', 'pool') and not freeze_head:
            self.backbone.predictor.requires_grad_(True)
        size = pool_size if mode == 'pool' else 1
        self.prompts = torch.nn.Parameter(torch.zeros(size, dim),
                                          requires_grad=mode in ('shared', 'pool'))
        self.keys = torch.nn.Parameter(torch.empty(size, dim), requires_grad=mode == 'pool')
        torch.nn.init.uniform_(self.keys, -.02, .02)
        # Zero prompts preserve the source predictor exactly before adaptation.
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        if self.mode != 'finetune':
            # Frozen means both weights AND BatchNorm statistics remain fixed.
            self.backbone.base_model.eval()
            self.backbone.sub_model.eval()
        return self

    def forward(self, data):
        if self.mode in ('frozen', 'head', 'finetune'):
            return self.backbone(data), data.x.new_zeros((), dtype=torch.float), None
        active = self.mode in ('shared', 'pool')
        # Routing does not use labels, environment IDs, or the current stage number.
        with torch.no_grad():
            _, query_graph = self.backbone.base_model(
                data.x, data.edge_index, data.edge_attr, data.batch)
            query_sub = sub_features(self.backbone, data, len(query_graph))
            query = F.normalize(query_graph + query_sub, dim=-1).detach()
        ids = torch.zeros(len(query), dtype=torch.long, device=query.device)
        key_loss = query.new_zeros(())
        if self.mode == 'pool':
            similarity = query @ F.normalize(self.keys, dim=-1).T
            ids = similarity.argmax(-1)
            key_loss = (1 - similarity.gather(1, ids[:, None])).mean()
        # Keep autograd through the frozen GIN: prompts still need its input gradients.
        _, graph = self.backbone.base_model(data.x, data.edge_index, data.edge_attr,
                                            data.batch, self.prompts[ids] if active else None)
        sub = query_sub
        return self.backbone.predictor(graph + sub), key_loss, ids if active else None


def selected_step(optimizer, model, ids):
    """Adam momentum must not move rows that this batch did not select."""
    saved = []
    if ids is not None:
        inactive = torch.ones(len(model.prompts), dtype=torch.bool, device=ids.device)
        inactive[ids.unique()] = False
        for parameter in (model.prompts, model.keys):
            state = optimizer.state.get(parameter, {})
            moments = {k: v[inactive].clone() for k, v in state.items()
                       if torch.is_tensor(v) and v.shape == parameter.shape}
            saved.append((parameter, inactive, parameter.detach()[inactive].clone(), moments))
    optimizer.step()
    with torch.no_grad():
        for parameter, inactive, values, moments in saved:
            parameter[inactive] = values
            for key, value in optimizer.state.get(parameter, {}).items():
                if torch.is_tensor(value) and value.shape == parameter.shape:
                    value[inactive] = moments.get(key, torch.zeros_like(value[inactive]))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probabilities, labels = [], []
    counts = [0] * len(model.prompts)
    for data in loader:
        data = data.to(device)
        logits, _, ids = model(data)
        probabilities.extend(logits.softmax(-1)[:, 1].cpu().tolist())
        labels.extend(data.y.view(-1).cpu().tolist())
        if ids is not None:
            for key, value in enumerate(torch.bincount(ids, minlength=len(counts)).tolist()):
                counts[key] += value
    if not labels:
        raise ValueError('Empty evaluation set')
    auc = float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None
    predicted = (np.asarray(probabilities) >= .5).astype(int)
    truth = np.asarray(labels)
    tn = int(((truth == 0) & (predicted == 0)).sum())
    fp = int(((truth == 0) & (predicted == 1)).sum())
    fn = int(((truth == 1) & (predicted == 0)).sum())
    tp = int(((truth == 1) & (predicted == 1)).sum())
    return {'auc': auc, 'accuracy': float(np.mean(predicted == labels)),
            'balanced_accuracy': float(balanced_accuracy_score(labels, predicted)),
            'confusion_matrix': [[tn, fp], [fn, tp]],
            'negative_recall': tn / (tn + fp) if tn + fp else None,
            'positive_recall': tp / (tp + fn) if tp + fn else None,
            'predicted_positive_fraction': float(predicted.mean()),
            'mean_positive_probability': float(np.mean(probabilities)),
            'n': len(labels), 'prompt_counts': counts}


def source_class_weights(rows, indices, device='cpu'):
    """Inverse frequency weights fit on source only, never on future/evaluation labels."""
    counts = torch.bincount(torch.tensor([int(rows[i]['cls_label']) for i in indices]),
                            minlength=2).float()
    if len(counts) != 2 or (counts == 0).any():
        raise ValueError('Source requires both binary classes')
    return (counts.sum() / (2 * counts)).to(device)
