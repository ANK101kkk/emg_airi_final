"""Fit the released three-branch predictor on the labelled AIRI recordings."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

# Match the CPU linear algebra configuration of the reference fit.
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np

RUNTIME = ('predict.py', 'submission.py', 'literature_models.py', 'ordinal_emg.py',
           'physiology_tds.py', 'pose_retrieval.py', 'rhythm_emg.py', 'spatial_tds.py',
           'spectral_expert.py', 'trajectory_filter.py', 'requirements.txt')


def fit(project, data, work, out, held_days):
    sys.path.insert(0, str(project))
    from training.manifold import fit as fit_manifold
    from training.ordinal import train as fit_ordinal
    from training.portable_spatial import precache
    from training.spectral import fit as fit_spectral
    from training.trainkit import prepare_start
    from training.trainkit.official_baseline.utils import ensure_output_outside_data

    ensure_output_outside_data(out, [data])
    ensure_output_outside_data(work, [data])
    if out.exists():
        raise FileExistsError(f'Output directory already exists: {out}')

    with (data / 'train/metadata.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    days = {r['day_id'] for r in rows}
    if not set(held_days) < days:
        raise ValueError(f'Excluded days must be a proper subset of {sorted(days)}')
    train_rows = [r for r in rows if r['day_id'] not in held_days]
    if {r['action'] for r in train_rows} != {'Index','Middle','Little','Ring','Thumb','Close','Fist','Open'}:
        raise ValueError('The training subset must contain all eight action classes')

    config = json.loads((project / 'model_config.json').read_text())
    expected = {'models': ['spectral_expert', 'manifold_hmm', 'ordinal_rhythm'],
                'weights': [0.375, 0.375, 0.25], 'gain': 1.6, 'bias': None}
    if config['recipe'] != expected:
        raise ValueError('Expected the released spectral, HMM and ordinal recipe with gain 1.6')
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Training on {len(train_rows)} recordings from {len(days - set(held_days))} days.", flush=True)
    with TemporaryDirectory(prefix='.airi-train-', dir=out.parent) as temporary:
        stage = Path(temporary) / 'predictor'
        weights = stage / 'weights'
        weights.mkdir(parents=True)
        for name in RUNTIME:
            shutil.copy2(project / name, stage / name)

        print('Fitting TDS-LSTM backbone and spectral experts (seed 4831).', flush=True)
        spectral_start, _ = prepare_start(data, work, 4831, held_days=held_days)
        from training.trainkit import prepare_cache
        feature_dir = prepare_cache(data, work).parent / 'spatial'
        precache(data, feature_dir)
        fit_spectral(data, feature_dir, spectral_start, held_days, weights / 'model.pt')

        print('Fitting the HMM encoder (seed 4827) and gesture-phase filter.', flush=True)
        hmm_start, _ = prepare_start(data, work, 4827, held_days=held_days)
        shutil.copy2(hmm_start, weights / 'tds_long_480_1.pt')
        fit_manifold(data, hmm_start, held_days, weights / 'manifold.pkl')

        print('Fitting ordinal rhythm coefficients and angle bounds.', flush=True)
        fit_ordinal(data, weights / 'rhythm_weights.npz', held_days)
        extrema = []
        for row in train_rows:
            y = np.load(data / 'train/targets' / (row['recording_id'] + '.npy'),
                        mmap_mode='r', allow_pickle=False)[1000:]
            extrema.append((y.min(0), y.max(0)))
        config['bounds'] = [np.min([a for a,_ in extrema], axis=0).tolist(),
                            np.max([b for _,b in extrema], axis=0).tolist()]
        (stage/'model_config.json').write_text(json.dumps(config, indent=2) + '\n')
        stage.rename(out)
    print(f'Saved predictor to {out}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path, help='Directory containing train/metadata.csv')
    parser.add_argument('--work', required=True, type=Path, help='Reusable feature / backbone cache')
    parser.add_argument('--out', required=True, type=Path, help='New directory for the fitted predictor')
    parser.add_argument('--exclude-day', action='append', default=[], help='Held-out day_id, repeatable')
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parent,
                        help='Inference sources and model_config.json (default: next to train.py)')
    args = parser.parse_args()
    fit(args.project.resolve(), args.data.resolve(), args.work.resolve(), args.out.resolve(), args.exclude_day)
