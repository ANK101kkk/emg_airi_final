"""Prepare per-record spatial descriptors from the raw EMG before training a fold."""
import csv
from tempfile import NamedTemporaryFile
from pathlib import Path
import numpy as np
from literature_models import features
from spatial_tds import spatial


def records(data):
    with (Path(data)/'train/metadata.csv').open(newline='') as source:
        return list(csv.DictReader(source))


def precache(data, folder):
    train=Path(data)/'train'
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    for r in records(data):
        rid=r['recording_id']
        path=folder/(rid+'.npz')
        if path.exists():
            continue
        signal=np.load(train/'signals'/(rid+'.npy'),allow_pickle=False)
        y=np.load(train/'targets'/(rid+'.npy'),allow_pickle=False)
        with NamedTemporaryFile(dir=folder, suffix='.npz', delete=False) as tmp:
            staging = Path(tmp.name)
        try:
            np.savez(staging, x=features(signal,(25,50,100)), s=spatial(signal),
                     y=y[::25].astype('f'))
            staging.replace(path)
        finally:
            staging.unlink(missing_ok=True)
