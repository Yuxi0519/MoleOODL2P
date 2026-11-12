"""Shared data protocol for the final delivery.

Functions relocated unchanged from development scripts so final code does
not depend on archived experiment entry points.
"""
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from .core import read_json, digest, molecule_id


def groups_for(rows):
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[molecule_id(row['smiles'])].append(index)
    for indices in grouped.values():
        if len({int(rows[i]['cls_label']) for i in indices}) != 1:
            raise ValueError('Equivalent molecules have conflicting labels')
    return list(grouped.values())


def stratified_groups(rows, groups, fraction, seed):
    """Split molecule groups, never individual duplicate records."""
    if not 0 < fraction < 1:
        raise ValueError('Fraction must be between zero and one')
    labels = [int(rows[group[0]]['cls_label']) for group in groups]
    try:
        left, right = train_test_split(groups, train_size=fraction, random_state=seed,
                                       stratify=labels)
    except ValueError as error:
        raise ValueError('Too few molecule groups per class; use fewer environments') from error
    left = [i for group in left for i in group]
    right = [i for group in right for i in group]
    if any({int(rows[i]['cls_label']) for i in indices} != {0, 1}
           for indices in (left, right)):
        raise ValueError('Each partition needs both classes; use fewer environments or more data')
    return left, right


def validate_source_checkpoint(path, partition):
    """Reject the full-train reproduction checkpoint to avoid future-data pretraining."""
    args_path = Path(path).resolve().parent / 'args.json'
    trained_args = read_json(args_path)
    source_path = Path(trained_args['train_path'])
    if not source_path.is_file() or digest(source_path) != partition['source_sha256']:
        raise ValueError('Checkpoint was not trained on this source.json. '
                         'The full-train reproduction checkpoint is not a valid CL initialization.')


def validate_stream(manifest, rows):
    """Validate record and molecular separation before any learning takes place."""
    partition = manifest['partition']
    source = partition['source_indices']
    later = partition['remaining_indices']
    if sorted(source + later) != list(range(len(rows))):
        raise ValueError('Source/remaining partition is invalid')
    adaptation = [i for env in manifest['environments'] for i in env['adapt_indices']]
    evaluation = [i for env in manifest['environments'] for i in env['eval_indices']]
    stream = [i for episode in manifest['episodes'] for i in episode['indices']]
    if sorted(adaptation + evaluation) != sorted(later) or sorted(stream) != sorted(adaptation):
        raise ValueError('Stream must visit every adaptation record exactly once')
    ownership = {}
    for split, indices in [('source', source), ('adapt', adaptation), ('eval', evaluation)]:
        for index in indices:
            key = molecule_id(rows[index]['smiles'])
            if key in ownership and ownership[key] != split:
                raise ValueError('Equivalent molecule crosses source/adapt/eval partitions')
            ownership[key] = split
