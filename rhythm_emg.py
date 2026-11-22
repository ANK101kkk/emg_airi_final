"""Causal EMG envelope, rhythm and action-conditioned hand pose at 20 Hz."""
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.signal import butter, sosfilt, lfilter
from scipy.special import softmax

ACTIONS = ('Close','Fist','Index','Little','Middle','Open','Ring','Thumb')
STRIDE = 25


def trailing(x, width, end=None):
    if end is None:
        end = np.arange(len(x))
    stop = end+1
    start = np.maximum(0, stop-width)
    prefix = np.vstack([np.zeros((1,x.shape[1])), np.cumsum(x,axis=0,dtype='d')])
    return (prefix[stop]-prefix[start]) / (stop-start)[:,None]


def lag(x, frames):
    return x[np.maximum(0,np.arange(len(x))-frames)]


def feature_pair(signal):
    x=np.asarray(signal,dtype='d')*1e5
    if not len(x) or x.ndim!=2 or x.shape[1]!=6 or not np.isfinite(x).all():
        raise ValueError("Expected a finite nonempty six-channel EMG recording")
    fr=np.arange(0,len(x),25)
    wide=sosfilt(butter(2,(18,230),fs=500,btype='bandpass',output='sos'),x,axis=0)
    fast=sosfilt(butter(2,(80,200),fs=500,btype='bandpass',output='sos'),x,axis=0)
    spatial=wide-wide.mean(axis=1,keepdims=True)
    power=np.column_stack([wide*wide,spatial*spatial,fast*fast])
    env=.5*np.log(np.maximum(2e-4,trailing(power,64,fr)))
    s=env[:,:12]
    ix=np.arange(len(s))
    sm=trailing(s,9)
    baseline=trailing(s,len(s))
    dev=np.sqrt(np.maximum(.035,trailing(s*s,len(s))-baseline**2))
    standardized=np.clip((s-baseline)/dev,-4,4)
    drift=s-trailing(s,52)
    spatial_s=s.reshape(-1,2,6)-s.reshape(-1,2,6).mean(2,keepdims=True)
    ii,jj=np.triu_indices(6,1)
    cross=trailing(wide[:,ii]*wide[:,jj],125,fr)
    broad=trailing(power[:,:6],125,fr)
    corr=cross/np.sqrt(np.maximum(1e-7,broad[:,ii]*broad[:,jj]))
    band=env[:,12:]-env[:,:6]
    # Causal complex resonators have separate in-phase and quadrature channels.
    # Their amplitude and sign convey recent tempo and movement phase.
    rhythm=[]
    for hz,tau in ((.38,.8),(.60,.85),(1.,.55)):
        damp=np.exp(-1/(20*tau))
        pole=damp*np.exp(2j*np.pi*hz/20)
        phase=lfilter([1-damp],[1,-pole],standardized,axis=0)
        rhythm.extend((phase.real,phase.imag))
    dh=[s-sm,sm,spatial_s.reshape(-1,12),band,corr,baseline,dev,drift,standardized,
        *(lag(standardized,k) for k in (2,5,10,17,26)),*rhythm]
    # Running level, modulation and correlation supply evidence of the action.
    mod=standardized[:,:6]
    modcorr=mod[:,ii]*mod[:,jj]
    env_stat=np.column_stack([s,spatial_s.reshape(-1,12),band,corr,modcorr,
        abs(mod-lag(mod,4)),mod*lag(mod,12),abs(rhythm[0][:,:6]+1j*rhythm[1][:,:6])])
    gate=np.column_stack([trailing(env_stat,len(ix)),trailing(env_stat,64),dev,
        np.sqrt(np.maximum(0,trailing(env_stat[:,:30]**2,len(ix))-trailing(env_stat[:,:30],len(ix))**2))])
    return np.asarray(np.column_stack(dh),dtype='f'),np.asarray(gate,dtype='f')


def kernel_fit(x,n,scale,seed):
    rng=np.random.default_rng(seed)
    center=np.median(x,0).astype('f')
    spread=np.maximum(x.std(0),.05).astype('f')
    anchor=(x[rng.choice(len(x),min(len(x),n),replace=False)]-center)/spread
    anchor=np.clip(anchor,-4,4)
    return (center,spread,anchor,np.array(scale,dtype='f'))


def kernel(x,params):
    center,spread,anchor,scale=params
    z=np.clip((x-center)/spread,-4,4)
    d=np.maximum((z*z).sum(1)[:,None]+(anchor*anchor).sum(1)-2*z@anchor.T,0)/anchor.shape[1]
    return np.column_stack([np.ones(len(z),dtype='f'),z*.15,np.exp(-scale*d)]).astype('f')


def ridge(x,y,lam,weight=None):
    if weight is None: weight=np.ones(len(x),dtype='f')
    xw=x.astype('d')*weight[:,None]
    gram=xw.T@x
    gram.flat[::len(gram)+1]+=lam
    gram[0,0]-=lam-.001
    return cho_solve(cho_factor(gram,lower=True),xw.T@y).astype('f')


class RhythmPose:
    def fit(self,recs):
        x,g,y,ac=[np.concatenate(z) for z in zip(*[(a[40::2],b[40::2],c[40::2],
            np.full(len(a[40::2]),ACTIONS.index(act),'i')) for a,b,c,act in recs])]
        self.gate_basis=kernel_fit(g,160,.8,3105)
        self.gate_weights=ridge(kernel(g,self.gate_basis),np.eye(len(ACTIONS),dtype='f')[ac],3.)
        self.bases=[]
        self.experts=[]
        for i in range(len(ACTIONS)):
            sel=(ac==i)
            par=kernel_fit(x[sel],220,.67,3356+i)
            xx=kernel(x[sel],par)
            yy=y[sel]
            pred=ridge(xx,yy,1.8)
            err=np.abs(xx@pred-yy)/np.maximum(.1,yy.std(0))
            w=1/np.maximum(1.,np.mean(err,axis=1)/.62)
            self.bases.append(par)
            self.experts.append(ridge(xx,yy,1.8,w))
        return self

    def from_features(self,x,g):
        logits=kernel(g,self.gate_basis)@self.gate_weights
        probs=softmax((logits*.65+trailing(logits,len(g))*.35)*12.,axis=1).astype('f')
        out=np.zeros((len(x),20),'f')
        for j,(basis,w) in enumerate(zip(self.bases,self.experts)):
            out+=probs[:,j,None]*(kernel(x,basis)@w)
        return out

    def predict_frames(self,signal):
        return self.from_features(*feature_pair(signal))

    def save(self,path):
        fields={}
        for name,weights,basis in [('gate',self.gate_weights,self.gate_basis)]+[
                (str(i),w,b) for i,(w,b) in enumerate(zip(self.experts,self.bases))]:
            fields[name+'_weights']=weights
            for i,v in enumerate(basis):
                fields[name+'_'+str(i)]=v
        np.savez_compressed(path,**fields)

    @classmethod
    def load(cls,path):
        obj=cls()
        with np.load(Path(path)) as z:
            unpack=lambda key: (tuple(z[key+'_'+str(i)] for i in range(4)),z[key+'_weights'])
            obj.gate_basis,obj.gate_weights=unpack('gate')
            experts=[unpack(str(i)) for i in range(8)]
        obj.bases,obj.experts=map(list,zip(*experts))
        return obj
