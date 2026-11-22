"""Prepare fold-specific backbone checkpoints from an unmodified AIRI dataset."""
import csv
import errno
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from . import fit_literature, refine_tds, train_physiology_tds


def _data_root(data):
    root = Path(data).expanduser().resolve()
    if not (root / 'train' / 'metadata.csv').is_file():
        raise ValueError(f'Expected train/metadata.csv inside {root}')
    return root


def _publish(destination, populate):
    """Publish one fully written directory, supporting concurrent identical requests."""
    if destination.is_dir():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.stage-', dir=destination.parent) as tmp:
        candidate = Path(tmp) / 'result'
        candidate.mkdir()
        populate(candidate)
        try:
            candidate.rename(destination)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY, errno.EACCES) or not destination.is_dir():
                raise


def prepare_cache(data, workdir):
    """Return the shared, dataset-specific 20-Hz feature and label cache."""
    data = _data_root(data)
    train = data / 'train'
    # Include all input file identities. The metadata bytes determine recording order.
    inputs = [train / 'metadata.csv', train / 'angle_info.json']
    with inputs[0].open(newline='') as file:
        metadata = list(csv.DictReader(file))
    required = [train / group / (r['recording_id']+'.npy') for r in metadata for group in ('signals','targets')]
    fingerprint = hashlib.sha256(str(data).encode())
    for item in inputs:
        fingerprint.update(item.read_bytes())
    for path in required:
        stat = path.stat()
        fingerprint.update(f'{path.relative_to(train)}:{stat.st_size}:{stat.st_mtime_ns}\n'.encode())
    destination = Path(workdir).expanduser().resolve() / fingerprint.hexdigest()[:20] / 'cache'
    _publish(destination, lambda temporary: fit_literature.save_cache(data,temporary))
    return destination


def prepare_start(data, workdir, seed, held_days=(), need_context=False):
    """Return (long_480_path, optional_context_960_path); days may be day IDs or integer indices."""
    data = _data_root(data)
    cache = prepare_cache(data,workdir)
    rows = json.loads((cache / 'rows.json').read_text())
    days = sorted({row['day'] for row in rows})
    if isinstance(held_days,str):
        held_days=[held_days]
    exclude = []
    for day in held_days:
        if isinstance(day,int) and 0 <= day < len(days):
            exclude.append(day)
        elif day in days:
            exclude.append(days.index(day))
        else:
            raise ValueError(f'Unknown day {day!r}. Available identifiers: {days}')
    exclude = sorted(set(exclude))
    if len(exclude) == len(days):
        raise ValueError('At least one training day is needed')
    name = '_'.join(map(str,exclude)) or 'all'
    destination = cache.parent / 'backbones' / name / str(seed)
    base = destination / 'base'
    refined = destination / 'refined'
    long = destination / 'long'
    context = destination / 'context'
    _publish(base, lambda p: fit_literature.fit(cache,data,p,exclude,1,1080,(1080,),True,seed))
    _publish(refined, lambda p: refine_tds.train(cache,base/'model_1080.pt',p,'combined',seed))
    _publish(long, lambda p: train_physiology_tds.train(cache,refined/'last.pt',p,'long',seed))
    if need_context:
        def fit_context(p):
            train_physiology_tds.train(cache,refined/'last.pt',p,'context',seed)
            train_physiology_tds.train(cache,p/'480.pt',p,'context',seed)
        _publish(context, fit_context)
    return long / '480.pt', context / '960.pt' if need_context else None
