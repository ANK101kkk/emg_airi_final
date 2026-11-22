"""Fine-tune a fitted AIRI TDS model, using only its original training days."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from literature_models import Config, build
from .official_baseline import utils

STEPS = 360
SNAPSHOTS = (0, 120, 240, 360)
BATCH = 12
CONTEXT = 40
SCORED = 80
POLICIES = ('control','velocity','gain','relative_mse','combined')


def train(cache, checkpoint, output, policy, seed, write_weights=True):
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
    rows = json.loads((cache/'rows.json').read_text())
    days = set(saved['training_days'])
    train_ids = [i for i,row in enumerate(rows) if row['day'] in days]
    held_ids = [i for i,row in enumerate(rows) if row['day'] not in days]
    cfg = Config(**saved['config'])
    if cfg.family != 'tds_lstm' or cfg.width != 64 or saved['steps'] != 1080:
        raise ValueError('Expected the original width-64 TDS checkpoint trained for 1080 steps')
    norm = saved['norm']
    scale,mean,std = [torch.tensor(norm[key],dtype=torch.float32).view(1,-1,1) for key in ('scale','mean','std')]
    xs, ys, joint_weights = [], [], []
    train_set = set(train_ids)
    for i, row in enumerate(rows):
        with np.load(cache/(row['id']+'.npz')) as z:
            xs.append(torch.from_numpy(np.ascontiguousarray(z[str(cfg.windows[0])].T)))
            ys.append(torch.from_numpy(np.ascontiguousarray(z['target'].T)) if i in train_set else None)
        weights = np.full(20,1/20,dtype=np.float32)
        if row['action'] in utils.FINGER_NAMES:
            index = 4 * utils.FINGER_NAMES.index(row['action'])
            weights[:] = 1/32
            weights[index:index+4] = 1/8
        joint_weights.append(torch.from_numpy(weights))
    sd = torch.cat([ys[i][:,CONTEXT:] for i in train_ids], dim=1).std(1,unbiased=False).clamp_min(.04)
    use_velocity = policy in ('velocity','combined')
    use_gain = policy in ('gain','combined')
    use_relative = policy in ('relative_mse','combined')
    torch.manual_seed(seed+9001)
    sample = np.random.default_rng(seed+9001)
    augment = np.random.default_rng(seed+4417)
    net = build(cfg)
    net.load_state_dict(saved['state'])
    opt = torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=1e-4)
    accum = {key:p.detach().clone() for key,p in net.state_dict().items()}
    for step in range(1,STEPS+1):
        ids = sample.choice(train_ids,size=BATCH)
        starts = [4 * sample.integers((xs[i].shape[1]-CONTEXT-SCORED)//4+1) for i in ids]
        x = torch.stack([xs[i][:,s:s+CONTEXT+SCORED] for i,s in zip(ids,starts)])
        if use_gain:
            log_gain = augment.normal(0,.10,(BATCH,1)) + augment.normal(0,.12,(BATCH,6))
            gains = torch.from_numpy(np.exp(log_gain).astype(np.float32)).repeat(1,9)
            x = x * gains[:,:,None]
        x = (torch.log1p(x/scale)-mean)/std
        y = torch.stack([ys[i][:,s+CONTEXT:s+CONTEXT+SCORED] for i,s in zip(ids,starts)])
        w = torch.stack([joint_weights[i] for i in ids])
        opt.zero_grad(set_to_none=True)
        pred = net(x)[:,:,CONTEXT:]
        diff = pred-y
        loss = (diff.abs().mean(-1)*w).sum(1).mean()
        if use_relative:
            relative = (diff / sd[None,:,None]).square().mean()
            loss = .75*loss + .25*sd.median()*relative
        if use_velocity:
            motion_error = diff[:,:,2:] - diff[:,:,:-2]
            loss = loss + .35*(motion_error.abs().mean(-1)*w).sum(1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt.step()
        if step in SNAPSHOTS:
            with torch.no_grad():
                for key,value in net.state_dict().items():
                    accum[key].add_(value)
    output.mkdir(parents=True,exist_ok=True)
    for label in ('last','average'):
        if label=='average':
            net.load_state_dict({key:value/len(SNAPSHOTS) for key,value in accum.items()})
        net.eval()
        if write_weights:
            torch.save(dict(saved,state=net.state_dict(),refinement=policy,refinement_steps=STEPS,
                            average_snapshots=list(SNAPSHOTS) if label=='average' else [STEPS],seed=seed),output/(label+'.pt'))
        pred={}
        with torch.inference_mode():
            for i in held_ids:
                x=(torch.log1p(xs[i][None]/scale)-mean)/std
                pred[rows[i]['id']]=net(x)[0].T.contiguous().numpy()
        if held_ids:
            np.savez_compressed(output/(label+'.npz'),**pred)
    (output/'split.json').write_text(json.dumps(dict(training_days=sorted(days),
        held_days=sorted({rows[i]['day'] for i in held_ids}),seed=seed,policy=policy,steps=STEPS)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True,type=Path)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--policy',choices=POLICIES,default='combined')
    p.add_argument('--seed',type=int,required=True)
    args=p.parse_args()
    train(args.cache,args.checkpoint,args.out,args.policy,args.seed)
