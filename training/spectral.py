"""Final training stage for spectral_expert."""
import numpy as np
import torch
from torch.nn import functional as F
from .portable_spatial import records
from spectral_expert import Config, SpectralExpert, summaries
FINGERS=["Index","Middle","Little","Ring","Thumb"]
ACTIONS=FINGERS+["Close","Fist","Open"]
STEPS=900

def fit(data, cache, checkpoint, held_days, output):
    torch.set_num_threads(1)
    ROWS=records(data)
    DAYS=sorted({r["day_id"] for r in ROWS})
    held=set(held_days)
    CACHE=cache
    train=[i for i,r in enumerate(ROWS) if r['day_id'] not in held]
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert set(ck['training_days'])==set(DAYS)-held
    scale,mean,std=(np.asarray(ck['norm'][k],dtype='f') for k in ('scale','mean','std'))
    x=[];raw=[];gr=[];ys={};ww=[];acts=[]
    for i,r in enumerate(ROWS):
        with np.load(CACHE/(r['recording_id']+'.npz')) as zd:
            x.append(torch.from_numpy((np.log1p(zd['x']/scale)-mean)/std))
            raw.append(zd['s']);gr.append(summaries(raw[-1]))
            if i in train:ys[i]=torch.from_numpy(zd['y'])
        w=np.full(20,.05,'f')
        if r['action'] in FINGERS:
            f=4*FINGERS.index(r['action']);w[:]=1/32;w[f:f+4]=1/8
        ww.append(torch.from_numpy(w));acts.append(ACTIONS.index(r['action']))
    def standardize(v):
        sample=np.concatenate([v[i][40::3] for i in train])
        mu=sample.mean(0,dtype='d').astype('f');std=sample.std(0,dtype='d').clip(.10).astype('f')
        return [torch.from_numpy(np.clip((val-mu)/std,-6,6)) for val in v],mu,std
    s,sm,sd=standardize(raw);g,gm,gd=standardize(gr)
    torch.manual_seed(962);rng=np.random.default_rng(962)
    cfg=Config(**ck['config']);net=SpectralExpert(cfg,len(sm),len(gm))
    net.temporal.load_state_dict({k.removeprefix('original.'):v for k,v in ck['state'].items()})
    params=[(net.temporal,.00012),(net.adapter,.00065),(net.gate,.0008),(net.experts,.0002)]
    opt=torch.optim.AdamW([dict(params=m.parameters(),lr=lr) for m,lr in params],weight_decay=.0003)
    best={}
    for t in range(1,STEPS+1):
        if t==600:
            for group in opt.param_groups:group['lr']*=.48
        ids=rng.choice(train,size=12)
        sl=[slice(p,p+110) for p in [int(rng.integers(0,len(x[i])-110+1)) for i in ids]]
        def gather(rows):return torch.stack([rows[i][v] for i,v in zip(ids,sl)])
        xx=gather(x);zz=gather(s);gg=gather(g)
        xx+=torch.randn(len(ids),1,54)*.14
        zz+=torch.randn(len(ids),1,len(sm))*.13
        gg+=torch.randn(len(ids),1,len(gm))*.19
        pose,logits=net(xx,zz,gg)
        err=pose[:,32:]-gather(ys)[:,32:];w=torch.stack([ww[i] for i in ids])[:,None,:]
        def cost(e):return ((e.square()+.000025).sqrt()*w).sum(dim=2).mean()
        lab=torch.tensor([acts[i] for i in ids])[:,None].expand(-1,logits.shape[1]-32)
        loss=cost(err)+.17*cost(err[:,2:]-err[:,:-2])+.08*cost(err[:,6:]-err[:,:-6])+ .028*F.cross_entropy(logits[:,32:].transpose(1,2),lab,label_smoothing=.05)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
        if t in (600,700,800,900):
            for k,v in net.state_dict().items():
                if k not in best:best[k]=v.detach().clone()/4
                else:best[k].add_(v.detach()/4)
    net.load_state_dict(best);net.eval()
    torch.save(dict(config=ck['config'],norm=ck['norm'],smean=sm.tolist(),sstd=sd.tolist(),gmean=gm.tolist(),gstd=gd.tolist(),model=best,training_days=sorted(set(DAYS)-held)),output)
