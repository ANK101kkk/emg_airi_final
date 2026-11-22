"""Predict 20 signed hand angles from six synchronized 500 Hz EMG channels."""
import argparse
import importlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from submission import output_csv

ROOT = Path(__file__).resolve().parent
CONTEXT_SAMPLES = 1000


def load_member(settings, root):
    module, constructor = settings['loader'].split(':')
    factory = importlib.import_module(module)
    for attribute in constructor.split('.'):
        factory = getattr(factory, attribute)
    paths = [root / name for name in settings['files']]
    if settings.get('separate_paths', False):
        return factory(*paths, *settings.get('args', []))
    source = paths if settings.get('takes_list', False) else paths[0]
    return factory(source, *settings.get('args', []))


class HandPose:
    def __init__(self):
        config = json.loads((ROOT / 'model_config.json').read_text())
        recipe = config['recipe']
        self.models = [load_member(config['models'][key], ROOT) for key in recipe['models']]
        weights = np.asarray(recipe['weights'], dtype=np.float32)
        if weights.ndim == 1:
            weights = weights[:, None]
        if (weights.shape not in ((len(self.models), 1), (len(self.models), 20))
                or not np.isfinite(weights).all() or np.any(weights < 0)
                or not np.allclose(weights.sum(axis=0), 1., atol=1.e-5)):
            raise ValueError('Invalid ensemble weights.')
        self.weights = weights
        self.gain = float(recipe['gain'])
        self.bias = None if recipe.get('bias') is None else np.asarray(recipe['bias'], dtype=np.float32)
        self.low, self.high = np.asarray(config['bounds'], dtype=np.float32)
        if self.low.shape != (20,) or self.high.shape != (20,) or np.any(self.low > self.high):
            raise ValueError('Invalid joint bounds.')

    def predict(self, signal):
        """Return angles in radians, shape (T, 20). The first 1000 rows are context."""
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim != 2 or signal.shape[1] != 6 or len(signal) <= CONTEXT_SAMPLES:
            raise ValueError('Expected a (T, 6) recording with more than 1000 samples.')
        if not np.isfinite(signal).all():
            raise ValueError('EMG contains non-finite values.')
        frame_count = (len(signal) + 24) // 25
        mixed = np.zeros((frame_count, 20), dtype=np.float32)
        for model, weight in zip(self.models, self.weights):
            frames = model.predict_frames(signal)
            if frames.shape != mixed.shape or not np.isfinite(frames).all():
                raise ValueError('A model returned invalid angle frames.')
            mixed += frames * weight

        # Estimate movement since the last 20 Hz frame using only earlier frames.
        samples = np.arange(len(signal))
        frame = samples // 25
        previous = np.maximum(frame - 2, 0)
        spans = np.maximum(1, np.minimum(frame, 2)).astype(np.float32)
        phase = ((samples % 25) / 25.).astype(np.float32)
        change = mixed[frame] - mixed[previous]
        correction = ((self.gain * phase / spans)[:, None] * change).clip(-.12, .12)
        result = mixed[frame] + correction
        if self.bias is not None:
            result += self.bias
        return np.clip(result, self.low, self.high)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='A recording or signal directory')
    parser.add_argument('--template', type=Path, help='Official template for a signal directory')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    source, dest = args.input.resolve(), args.output.resolve()
    if dest == source or source in dest.parents or (args.template and dest == args.template.resolve()):
        parser.error('Choose an output path separate from input data.')
    if not source.exists() or source.is_dir() != bool(args.template):
        parser.error('Use a recording alone or a signal directory with --template.')
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch.set_num_threads(2)
    model = HandPose()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        if args.template:
            output_csv(model, source, args.template, temporary)
        else:
            prediction = model.predict(np.load(source, allow_pickle=False))
            with temporary.open('wb') as handle:
                np.save(handle, prediction[CONTEXT_SAMPLES:])
        os.replace(temporary, dest)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
