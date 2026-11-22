"""Causal phase-space filter over a TDS–LSTM's current predicted hand pose."""
from pathlib import Path
import pickle
import numpy as np
from scipy.special import softmax
from pose_retrieval import ACTIONS, Encoder, lag, running_mean, fit as _fit, features_from_encoder, ridge, transform, make_basis


def descriptors(enc):
    pose=enc[0]
    return np.column_stack((pose,(pose-lag(pose,2)),(pose-lag(pose,5))*.6)).astype('f')


def fit(records):
    groups=[[] for _ in ACTIONS]; gates=[]; labels=[]
    for a,enc,y in records:
        g=features_from_encoder(enc)[0][10::6]
        gates.append(g); labels.append(np.broadcast_to(np.eye(8,dtype='f')[a],(len(g),8)))
        groups[a].append((descriptors(enc)[40:],y[40:]))
    g=np.concatenate(gates)
    gb=make_basis(g,48,4901)
    gc=ridge(transform(g,gb), np.concatenate(labels),.04)
    models=[]
    for a,rs in enumerate(groups):
        ys=np.concatenate([y-y.mean(0) for _,y in rs]); means=ys.mean(0)
        eig, vec=np.linalg.eigh(ys.T @ ys)
        direction=vec[:,-1]
        absolute=np.concatenate([y@direction for _,y in rs])
        velocity=np.concatenate([(y-lag(y,3))@direction for _,y in rs])
        count=14
        edges=np.quantile(absolute,np.linspace(0,1,count+1)[1:-1])
        s1=np.searchsorted(edges,absolute)
        s2=(velocity>0).astype('i')
        states=2*s1+s2
        n=2*count
        trans=np.full((n,n),.05)
        i=0
        for desc,y in rs:
            seq=states[i:i+len(y)]; i+=len(y)
            np.add.at(trans,(seq[:-1],seq[1:]),1)
        trans/=trans.sum(1,keepdims=True)
        prior=np.bincount(states,minlength=n)+.5; prior/=prior.sum()
        o=np.concatenate([d for d,_ in rs]); y=np.concatenate([v for _,v in rs]); residual=y-o[:,:20]
        # Scale within-state precision from actual fitted encoder errors plus a noise floor.
        group_o=[]; group_y=[]; group_e=[]
        for j in range(n):
            ix=(states==j)
            group_o.append((o[ix].sum(0)+o.mean(0))/(ix.sum()+1))
            group_y.append((y[ix].sum(0)+y.mean(0))/(ix.sum()+1))
            group_e.append((residual[ix].sum(0)+4*residual.mean(0))/(ix.sum()+4))
        group_o=np.array(group_o); group_y=np.array(group_y); group_e=np.array(group_e)
        err=o-group_o[states]
        cov=np.cov(err.T)
        floors=np.tile(np.r_[np.full(20,.11),np.full(20,.08),np.full(20,.09)],1)
        cov+=np.diag(floors**2)
        eig,vec=np.linalg.eigh(cov)
        whitening=(vec/np.sqrt(eig)[None,:]) @ vec.T
        models.append((prior.astype('f'),trans.astype('f'),(group_o@whitening).astype('f'),whitening.astype('f'),group_y.astype('f'),group_e.astype('f')))
    return gb,gc,models


def from_encoder(enc,meta):
    gb,gc,models=meta
    gate=transform(features_from_encoder(enc)[0],gb)@gc
    weights=softmax(10*(.85*gate+.15*running_mean(gate)),axis=1)
    pose=enc[0]; obs=descriptors(enc)
    correct=np.zeros_like(pose); phase=np.zeros_like(pose)
    for i,(prior,transition,centers,precision,goal,residual) in enumerate(models):
        x=obs@precision
        distance=(x*x).sum(1)[:,None]+(centers*centers).sum(1)-2*x@centers.T
        likelihood=np.exp(np.clip(-.48*(distance-distance.min(1,keepdims=True)),-60,0))
        state=prior; posterior=np.empty_like(likelihood)
        for t,row in enumerate(likelihood):
            state=(.92*(state@transition)+.08*prior)*row
            state/=state.sum()
            posterior[t]=state
        correct+=weights[:,i,None]*(posterior@residual)
        phase+=weights[:,i,None]*(posterior@goal)
    return {'manifold_hmm':.50*(pose+correct)+.50*phase,'manifold_residual':.8*(pose+correct)+.2*phase}


class ManifoldPredictor:
    def __init__(self, checkpoint, metadata, variant='manifold_hmm'):
        self.encoder=Encoder(checkpoint)
        with Path(metadata).open('rb') as f:self.meta=pickle.load(f)
        self.variant=variant

    def predict_frames(self,signal):
        return from_encoder(self.encoder(signal),self.meta)[self.variant]
