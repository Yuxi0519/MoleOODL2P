"""Bounded dynamic prompt memory for supervised, task-ID-free graph streams.

The allocator observes only frozen graph features. Labels are used solely by
the supervised loss. All decisions, including capacity failures, are logged.
"""
import torch
from sklearn.cluster import KMeans
from torch.nn import functional as F

from .core import PromptGIN, sub_features


class PromptMemory(torch.nn.Module):
    def __init__(self, capacity, dim, settings):
        super().__init__()
        self.settings = dict(settings)
        # Reserve a hard upper bound; logical activation is dynamic, physical memory bounded.
        # Separate Parameters give each prompt independent Adam moments and step counters.
        self.prompts = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(dim)) for _ in range(capacity)])
        for name, value in {
            'keys': torch.zeros(capacity, dim), 'anchors': torch.zeros(capacity, dim),
            'active': torch.zeros(capacity, dtype=torch.bool),
            'protected': torch.zeros(capacity, dtype=torch.bool),
            'usage': torch.zeros(capacity, dtype=torch.long),
            'last_used': torch.full((capacity,), -1, dtype=torch.long),
            'birth': torch.zeros(capacity, dtype=torch.long),
            'generation': torch.zeros(capacity, dtype=torch.long),
            'clock': torch.tensor(0, dtype=torch.long),
            'streak': torch.tensor(0, dtype=torch.long),
            'last_allocation': torch.tensor(-100000, dtype=torch.long),
            'pending': torch.zeros(settings['novel_buffer'], dim),
            'pending_count': torch.tensor(0, dtype=torch.long),
            'novel_distance': torch.tensor(0., dtype=torch.float),
            'reuse_samples': torch.tensor(0, dtype=torch.long),
            'novel_batches': torch.tensor(0, dtype=torch.long),
            'novel_samples': torch.tensor(0, dtype=torch.long),
        }.items():
            self.register_buffer(name, value)

    @torch.no_grad()
    def initialize(self, features, count, seed):
        features = F.normalize(torch.as_tensor(features, dtype=torch.float32), dim=-1)
        if len(features) < count or count < 1 or count > len(self.prompts):
            raise ValueError('Invalid initial prompt count')
        centers = KMeans(count, random_state=seed, n_init=10).fit(features.cpu().numpy()).cluster_centers_
        centers = F.normalize(torch.as_tensor(centers, device=self.keys.device), dim=-1)
        self.keys[:count].copy_(centers)
        self.active[:count] = True
        self.protected[:count] = True
        distance = 1 - (features.to(self.keys.device) @ centers.T).max(-1).values
        # Calibration uses source features only, never stream or evaluation outcomes.
        radius = torch.quantile(distance, self.settings['source_quantile'])
        self.novel_distance.copy_(radius.clamp_min(self.settings['minimum_distance']))

    def route(self, query):
        similarity = query @ F.normalize(self.keys, dim=-1).T
        similarity = similarity.masked_fill(~self.active[None, :], -torch.inf)
        scores, ids = similarity.max(-1)
        return ids, scores

    def selected(self, ids):
        # Only actually selected Parameters enter the autograd graph.
        return torch.stack([self.prompts[int(i)] for i in ids])

    def regularization(self, ids):
        unique = ids.unique()
        values = torch.stack([self.prompts[int(i)] for i in unique])
        return (values - self.anchors[unique]).square().mean()

    @torch.no_grad()
    def observe(self, query, optimizer, dynamic=True):
        """One causal allocator update per arriving minibatch; no task boundary input."""
        self.clock.add_(1)
        step = int(self.clock)
        ids, scores = self.route(query)
        novel = (1 - scores) > self.novel_distance + self.settings['distance_margin']
        fraction = float(novel.float().mean())
        self.novel_samples.add_(int(novel.sum()))
        if fraction >= self.settings['novel_fraction']:
            self.novel_batches.add_(1)
        event = None
        if dynamic and fraction >= self.settings['novel_fraction']:
            self.streak.add_(1)
            old = self.pending[:int(self.pending_count)]
            buffered = torch.cat([old, query[novel]], dim=0)[-len(self.pending):]
            self.pending[:len(buffered)].copy_(buffered)
            self.pending_count.fill_(len(buffered))
        else:
            self.streak.zero_()
            self.pending_count.zero_()
        ready = (dynamic and int(self.streak) >= self.settings['patience'] and
                 step - int(self.last_allocation) >= self.settings['allocation_cooldown'])
        if ready:
            buffered = self.pending[:int(self.pending_count)]
            candidate = F.normalize(buffered.mean(0), dim=0)
            coherence = float((buffered @ candidate).mean())
            parent_ids, parent_scores = self.route(candidate[None])
            parent = int(parent_ids[0])
            still_novel = float(1 - parent_scores[0]) > float(self.novel_distance) + self.settings['distance_margin']
            base = {'step': step, 'novel_fraction': fraction, 'coherence': coherence,
                    'active_before': int(self.active.sum())}
            if not still_novel or coherence < self.settings['minimum_coherence']:
                event = {**base, 'type': 'novelty_rejected', 'reason': 'incoherent_or_already_covered'}
            else:
                free = torch.where(~self.active)[0].tolist()
                eligible = [i for i in torch.where(self.active & ~self.protected)[0].tolist()
                            if step - int(self.last_used[i]) >= self.settings['replacement_grace']]
                if free:
                    slot, action = free[0], 'allocate'
                elif eligible:
                    # Bounded LRU replacement; initial source slots are never evicted.
                    slot = min(eligible, key=lambda i: (int(self.last_used[i]), int(self.usage[i]), i))
                    action = 'replace'
                else:
                    slot, action = None, 'capacity_deferred'
                event = {**base, 'type': action, 'parent': parent}
                if slot is not None:
                    inherited = self.prompts[parent].detach().clone()
                    previous_generation = int(self.generation[slot])
                    self.prompts[slot].copy_(inherited)
                    self.prompts[slot].grad = None
                    if optimizer is not None:
                        # A reused slot is a new expert, not the previous expert's Adam state.
                        optimizer.state.pop(self.prompts[slot], None)
                    self.keys[slot].copy_(candidate)
                    self.anchors[slot].copy_(inherited)
                    self.active[slot] = True
                    self.usage[slot] = 0
                    self.birth[slot] = step
                    self.last_used[slot] = step
                    self.generation[slot] += 1
                    event.update(slot=slot, previous_generation=previous_generation,
                                 generation=int(self.generation[slot]))
                    self.last_allocation.fill_(step)
            # No persistent growth from a single sustained outlier burst.
            self.streak.zero_()
            self.pending_count.zero_()
        ids, _ = self.route(query)
        for slot in ids.unique().tolist():
            if int(self.usage[slot]) > 0:
                self.reuse_samples.add_(int((ids == slot).sum()))
            self.usage[slot] += int((ids == slot).sum())
            self.last_used[slot] = step
        if event is not None:
            event['active_after'] = int(self.active.sum())
        return event, ids

    def snapshot(self):
        return {'active': int(self.active.sum()), 'capacity': len(self.prompts),
                'source_distance_threshold': float(self.novel_distance),
                'reuse_samples': int(self.reuse_samples),
                'novel_batches': int(self.novel_batches), 'novel_samples': int(self.novel_samples),
                'slots': [{'slot': i, 'generation': int(self.generation[i]),
                           'usage': int(self.usage[i]), 'last_used': int(self.last_used[i]),
                           'protected_source': bool(self.protected[i])}
                          for i in torch.where(self.active)[0].tolist()]}


class DynamicPromptModel(torch.nn.Module):
    def __init__(self, backbone, source_features, settings, seed, mode='dynamic'):
        super().__init__()
        dim = backbone.base_model.node_embeddings[0].linears[0].out_features
        adapter = PromptGIN([39], [10], num_layers=4, emb_dim=dim, dropout=.1,
                            JK='last', readout='mean')
        adapter.load_state_dict(backbone.base_model.state_dict(), strict=True)
        backbone.base_model = adapter
        self.backbone = backbone.requires_grad_(False).eval()
        self.mode = mode
        capacity = 1 if mode == 'shared' else settings['capacity']
        initial = settings['initial_prompts'] if mode == 'dynamic' else capacity
        self.memory = PromptMemory(capacity, dim, settings)
        self.memory.initialize(source_features, initial, seed)

    @property
    def prompts(self):
        return self.memory.prompts

    def train(self, mode=True):
        super().train(mode)
        # The property head is also frozen: keep all its stochastic layers deterministic.
        self.backbone.eval()
        return self

    @torch.no_grad()
    def context(self, data):
        _, graph = self.backbone.base_model(data.x, data.edge_index, data.edge_attr, data.batch)
        sub = sub_features(self.backbone, data, len(graph))
        query = F.normalize(graph + sub, dim=-1)
        return query, sub, self.backbone.predictor(graph + sub)

    def forward(self, data):
        query, sub, _ = self.context(data)
        ids, _ = self.memory.route(query)
        # Frozen weights, but input gradients must reach the selected prompts.
        _, graph = self.backbone.base_model(data.x, data.edge_index, data.edge_attr,
                                            data.batch, self.memory.selected(ids))
        logits = self.backbone.predictor(graph + sub)
        return logits, self.memory.regularization(ids), ids
