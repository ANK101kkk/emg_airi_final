"""Fit the two continued TDS recipes from held-day-specific checkpoints."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from literature_models import Config
from physiology_tds import ContextTDS, reference_envelopes
from .official_baseline import utils

STEPS = 480
SAVE_AT = (240, 480)
REGIMES = ('long', 'context')


def train(cache, checkpoint, output, recipe, seed):
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
    prior_steps = saved.get('extra_steps', 0)
    if prior_steps:
        if prior_steps != 480 or not saved['use_context'] or recipe != 'context':
            raise ValueError('Extended training requires a 480-step context model')
    elif saved['refinement'] != 'combined' or saved['refinement_steps'] != 360:
        raise ValueError('Expected a previously refined TDS checkpoint')
    if saved['seed'] != seed:
        raise ValueError('Starting checkpoint seed differs')
    config = Config(**saved['config'])
    days = set(saved['training_days'])
    rows = json.loads((cache / 'rows.json').read_text())
    training = [i for i, r in enumerate(rows) if r['day'] in days]
    held = [i for i, r in enumerate(rows) if r['day'] not in days]
    raw, envelopes, y, weights = [], [], {}, []
    norm = [torch.tensor(saved['norm'][key], dtype=torch.float32).view(1,-1,1) for key in ('scale','mean','std')]
    enabled = recipe == 'context'
    for i, row in enumerate(rows):
        with np.load(cache / (row['id'] + '.npz')) as z:
            frame = z[str(config.windows[0])]
            raw.append(torch.from_numpy(frame.T.copy()))
            if enabled:
                envelopes.append(torch.from_numpy(reference_envelopes(frame).T.copy()))
            if i in training:
                y[i] = torch.from_numpy(z['target'].T.copy())
        weights_i = torch.full((20,), .05)
        if row['action'] in utils.FINGER_NAMES:
            start = 4 * utils.FINGER_NAMES.index(row['action'])
            weights_i[:] = 1/32
            weights_i[start:start+4] = 1/8
        weights.append(weights_i)
    sd = torch.cat([y[i][:,40:] for i in training],dim=1).std(1,unbiased=False).clamp_min(.04)
    torch.manual_seed(seed + 25019 + prior_steps)
    samples = np.random.default_rng(seed + 25019 + prior_steps)
    augment = np.random.default_rng(seed + 62439 + prior_steps)
    net = ContextTDS(config, enabled)
    if prior_steps:
        net.load_state_dict(saved['state'])
    else:
        net.original.load_state_dict(saved['state'])
    opt = torch.optim.AdamW(net.parameters(), lr=.00014 if prior_steps else .00024, weight_decay=.0001)
    batch, warm, scored = 12, 40, 80
    length = warm + scored
    output.mkdir(parents=True, exist_ok=True)
    for step in range(1, STEPS+1):
        if step == 361:
            for group in opt.param_groups:
                group['lr'] = .00008 if prior_steps else .00013
        ids = samples.choice(training,size=batch)
        starts = [int(4*samples.integers((raw[i].shape[1]-length)//4+1)) for i in ids]
        inputs = torch.stack([raw[i][:,s:s+length] for i,s in zip(ids, starts)])
        reference = torch.stack([envelopes[i][:,s:s+length] for i,s in zip(ids,starts)]) if enabled else None
        targets = torch.stack([y[i][:,s+warm:s+length] for i,s in zip(ids,starts)])
        w = torch.stack([weights[i] for i in ids])
        common, individual = (.17,.22) if enabled else (.10,.12)
        gain = augment.normal(0,common,(batch,1)) + augment.normal(0,individual,(batch,6))
        gain = torch.tensor(np.exp(gain),dtype=torch.float32)
        inputs = inputs * gain.repeat(1,9)[:,:,None]
        if enabled:
            reference = reference * gain.repeat(1,3)[:,:,None]
        net.train()
        opt.zero_grad(set_to_none=True)
        outputs = net(inputs, reference, norm)[:,:,warm:]
        error = outputs - targets
        position = (error.abs().mean(-1)*w).sum(1).mean()
        relative = (error / sd[None,:,None]).square().mean()
        loss = .75*position + .25*sd.median()*relative
        lags = ((1,.10),(3,.15),(6,.10)) if enabled else ((2,.35),)
        for lag, coef in lags:
            displacement = (error[:,:,lag:] - error[:,:,:-lag]).abs().mean(-1)
            loss = loss + coef * (displacement*w).sum(1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt.step()
        if step in ((STEPS,) if prior_steps else SAVE_AT):
            total_steps = step + prior_steps
            net.eval()
            torch.save(dict(config=saved['config'], norm=saved['norm'], state=net.state_dict(),
                            use_context=enabled, training_days=sorted(days), seed=seed, extra_steps=total_steps),
                       output/f'{total_steps}.pt')
            predictions = {}
            with torch.inference_mode():
                for i in held:
                    refs = envelopes[i][None] if enabled else None
                    predictions[rows[i]['id']] = net(raw[i][None], refs, norm)[0].T.contiguous().numpy()
            if held:
                np.savez_compressed(output/f'{total_steps}.npz', **predictions)
    marker = 'split_960.json' if prior_steps else 'split.json'
    (output/marker).write_text(json.dumps(dict(train_days=sorted(days),
        held_days=sorted({rows[i]['day'] for i in held}),recipe=recipe,seed=seed,steps=STEPS+prior_steps)))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--recipe',choices=REGIMES,required=True)
    p.add_argument('--seed',type=int,required=True)
    a=p.parse_args()
    train(a.cache,a.checkpoint,a.out,a.recipe,a.seed)
