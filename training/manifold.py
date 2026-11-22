"""Refit the gesture-conditioned phase filter from raw AIRI training recordings.

The pretrained encoder must correspond to exactly the included recording days.
"""
import argparse
import csv
import os
from pathlib import Path
import pickle
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from trajectory_filter import ACTIONS, Encoder, fit as fit_filter


def fit(data, checkpoint, excluded_days, output_pkl):
    """Fit from data/train, excluding day identifiers or sorted zero-based indices."""
    train = Path(data) / 'train'
    with (train / 'metadata.csv').open(newline='') as metadata:
        rows = list(csv.DictReader(metadata))
    days = sorted({row['day_id'] for row in rows})
    if isinstance(excluded_days, str):
        excluded_days = [excluded_days]
    held = set()
    for day in excluded_days:
        if isinstance(day, int) and 0 <= day < len(days):
            held.add(days[day])
        elif day in days:
            held.add(day)
        else:
            raise ValueError(f'Unknown day {day!r}')
    included = set(days) - held
    if not included:
        raise ValueError('No training days remain')
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state['seed'] != 4827 or state['use_context'] or state['extra_steps'] != 480:
        raise ValueError('Expected the long_480 encoder with seed 4827')
    if set(state['training_days']) != included:
        raise ValueError('Encoder training days differ from the requested training split')
    encoder = Encoder(checkpoint)
    training = []
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for row in rows:
            if row['day_id'] in held:
                continue
            rid = row['recording_id']
            signal = np.load(train / 'signals' / f'{rid}.npy', allow_pickle=False)
            target = np.load(train / 'targets' / f'{rid}.npy', allow_pickle=False)
            n = int(row['n_samples'])
            if signal.shape != (n,6) or target.shape != (n,20):
                raise ValueError(f'Unexpected signal or target shape for {rid}')
            if not np.isfinite(signal).all() or not np.isfinite(target).all():
                raise ValueError(f'Nonfinite values in {rid}')
            encoded = encoder(np.ascontiguousarray(signal, dtype=np.float32))
            training.append((ACTIONS.index(row['action']), encoded,
                             np.ascontiguousarray(target[::25], dtype=np.float32)))
        parameters = fit_filter(training)
    finally:
        torch.set_num_threads(old_threads)
    destination = Path(output_pkl)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
        path = Path(tmp.name)
    try:
        with path.open('wb') as stream:
            pickle.dump(parameters, stream, protocol=5)
        os.replace(path, destination)
    finally:
        path.unlink(missing_ok=True)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--held-days', nargs='*', default=[])
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    print(fit(args.data, args.checkpoint, args.held_days, args.out))
