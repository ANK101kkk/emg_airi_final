"""Fit one candidate without using angles from its held-out days."""
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from literature_models import (STRIDE, SEED, build, features, fit_norm, grid, transform)
from .official_baseline import utils

CHECKPOINTS = (180, 360, 720, 1080)
BATCH = 12
CONTEXT = 40
SCORED = 80


def save_cache(data, cache):
    cache.mkdir(exist_ok=True, parents=True)
    records = utils.load_train_recordings(data)
    for r in records:
        target = r.target[::STRIDE].copy()
        raw = {str(w[0]): features(r.signal, w) for w in {(25,50,100),(12,25,50)}}
        np.savez(cache / (r.recording_id+'.npz'), target=target, **raw)
    rows = [dict(id=r.recording_id, day=r.day_id, session=r.session_id, action=r.action) for r in records]
    (cache/'rows.json').write_text(json.dumps(rows))


def fit(cache, data, output, exclude, config_id, steps, checkpoints=CHECKPOINTS, save_models=False, seed=SEED):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    cfg = grid()[config_id]
    rng = np.random.default_rng(seed)
    rows = json.loads((cache/'rows.json').read_text())
    days = sorted({r['day'] for r in rows})
    if any(i<0 or i>=len(days) for i in exclude): raise ValueError('Unexpected day index')
    held = [days[i] for i in exclude]
    train = [i for i,r in enumerate(rows) if r['day'] not in held]
    test = [i for i,r in enumerate(rows) if r['day'] in held]
    raw, ys, weights = [], [], []
    for r in rows:
        with np.load(cache/(r['id']+'.npz')) as z:
            ys.append(torch.from_numpy(np.ascontiguousarray(z['target'].T)))
            raw.append(z[str(cfg.windows[0])])
        w = np.full(20, .05, dtype='f')
        if r['action'] in utils.FINGER_NAMES:
            pos = utils.FINGER_NAMES.index(r['action'])*4
            w[:] = 1/32
            w[pos:pos+4] = 1/8
        weights.append(w)
    norm = fit_norm([raw[i] for i in train])
    xs = [transform(a,norm) for a in raw]
    net = build(cfg)
    opt = torch.optim.AdamW(net.parameters(),lr=cfg.lr,weight_decay=1e-4)
    output.mkdir(exist_ok=True, parents=True)
    split = dict(held=held, train_days=sorted({rows[i]['day'] for i in train}), config=asdict(cfg), seed=seed)
    (output/'split.json').write_text(json.dumps(split))
    score_rows = []
    for step in range(1, steps+1):
        ids = rng.choice(train,size=BATCH)
        # Multiples of four preserve the downsampled time grid in the U-Net.
        starts = [4 * rng.integers((xs[i].shape[1]-CONTEXT-SCORED)//4+1) for i in ids]
        x = torch.stack([xs[i][:,s:s+CONTEXT+SCORED] for i,s in zip(ids,starts)])
        y = torch.stack([ys[i][:,s+CONTEXT:s+CONTEXT+SCORED] for i,s in zip(ids,starts)])
        weight = torch.from_numpy(np.stack([weights[i] for i in ids]))
        net.train()
        opt.zero_grad(set_to_none=True)
        pred = net(x)[:,:,CONTEXT:]
        loss = ((pred-y).abs().mean(-1)*weight).sum(1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt.step()
        if step not in checkpoints: continue
        net.eval()
        if save_models:
            torch.save(dict(config=asdict(cfg),norm=norm,state=net.state_dict(),steps=step,**{'training_days':split['train_days']}),output/f'model_{step}.pt')
        arrays = {}
        for i in test:
            r = rows[i]
            with torch.inference_mode():
                frames = net(xs[i][None])[0].T.contiguous().numpy()
            target = np.load(data/'train/targets'/(r['id']+'.npy'),mmap_mode='r')
            p = frames.repeat(STRIDE,axis=0)[:len(target)]
            score = utils.recording_weighted_mae(p,target,r['action'])
            score_rows.append(dict(record=r['id'],day=r['day'],session=r['session'],step=step,weighted_mae=score))
            arrays[r['id']] = frames
        if test and save_models:
            np.savez_compressed(output/f'pred_{step}.npz',**arrays)
    if score_rows:
        with (output/'records.csv').open('w',newline='') as fp:
            writer = csv.DictWriter(fp,fieldnames=score_rows[0])
            writer.writeheader(); writer.writerows(score_rows)
    (output/'done').write_text(str(steps))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--output',type=Path)
    p.add_argument('--exclude',nargs='*',type=int,default=[])
    p.add_argument('--config',type=int,default=0)
    p.add_argument('--steps',type=int,default=360)
    p.add_argument('--seed',type=int,default=SEED)
    p.add_argument('--save',action='store_true')
    args = p.parse_args()
    if args.prepare: save_cache(args.data,args.cache)
    else: fit(args.cache,args.data,args.output,args.exclude,args.config,args.steps,sorted({v for v in CHECKPOINTS if v<=args.steps}|{args.steps}),args.save,args.seed)
