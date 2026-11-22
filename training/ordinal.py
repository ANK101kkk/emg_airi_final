"""Train a compact ordinal rhythm model from the official AIRI recordings."""
import argparse
import csv
import os
from pathlib import Path

# Reproduce the single-threaded fit used for the released coefficients.
for variable in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
from ordinal_emg import OrdinalRhythm
from rhythm_emg import feature_pair


def train(data, output, excluded_days=()):
    folder = Path(data) / 'train'
    with (folder / 'metadata.csv').open(newline='', encoding='utf-8') as metadata:
        rows = sorted(csv.DictReader(metadata), key=lambda row: row['recording_id'])
    available_days = {row['day_id'] for row in rows}
    excluded = set(excluded_days)
    unknown = excluded - available_days
    if unknown:
        raise ValueError(f'Unknown day IDs: {sorted(unknown)}')
    if excluded == available_days:
        raise ValueError('Training requires at least one included day')
    output = Path(output).resolve()
    if output == folder.resolve() or output.is_relative_to(folder.resolve()):
        raise ValueError('Save the model outside the training data directory')
    if output.suffix != '.npz':
        raise ValueError('Use an .npz filename for the output weights')

    recordings = []
    for row in rows:
        if row['day_id'] in excluded:
            continue
        filename = row['recording_id'] + '.npy'
        signal = np.load(folder / 'signals' / filename, allow_pickle=False)
        target = np.load(folder / 'targets' / filename, allow_pickle=False)
        if signal.shape != (len(target), 6) or target.shape != (len(signal), 20):
            raise ValueError(f'Unexpected signal or angle shape for {filename}')
        features, gate = feature_pair(signal)
        recordings.append((features, gate, target[::25].copy(), row['action']))

    model = OrdinalRhythm().fit(recordings)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    return len(recordings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True,
                        help='Directory containing train/metadata.csv, train/signals and train/targets')
    parser.add_argument('--out', type=Path, required=True, help='Output .npz weights')
    parser.add_argument('--exclude-day', action='append', default=[], metavar='DAY_ID',
                        help='Omit one day_id from training. This option can be repeated.')
    args = parser.parse_args()
    count = train(args.data, args.out, args.exclude_day)
    print(f'Trained on {count} recordings. Saved {args.out}.')
