"""Causal rhythm features with trainable, signed conditional angle quantiles."""
import numpy as np
from scipy.special import softmax,expit
from rhythm_emg import ACTIONS,kernel_fit,kernel,ridge,trailing,feature_pair

QUANTILES=(.04,.13,.28,.5,.72,.87,.96)


def fit_small_basis(x,n,scale,seed):
    params=kernel_fit(x,n,scale,seed)
    return tuple(v.astype('f2').astype('f') for v in params)


class OrdinalRhythm:
    def fit(self,recs):
        x,g,y,ac=[np.concatenate(z) for z in zip(*[(a[40::2],b[40::2],c[40::2],
            np.full(len(a[40::2]),ACTIONS.index(act),'i')) for a,b,c,act in recs])]
        self.gate_basis=fit_small_basis(g,140,.8,3105)
        self.gate_weights=ridge(kernel(g,self.gate_basis),np.eye(len(ACTIONS),dtype='f')[ac],3.)
        self.parts=[]
        for i in range(len(ACTIONS)):
            sel=ac==i
            basis=fit_small_basis(x[sel],144,.7,3356+i)
            xx=kernel(x[sel],basis)
            yy=y[sel]
            knots=np.quantile(yy,QUANTILES,axis=0).T.astype('f')
            width=np.maximum(.025,yy.std(0)*.12)
            prob=expit((knots[None]-yy[:,:,None])/width[None,:,None])
            target=np.column_stack((yy,prob.reshape(len(yy),-1)))
            # Compress redundant signed angle and ordinal outputs into one common basis.
            mean=target.mean(0)
            cent=target-mean
            _,vectors=np.linalg.eigh((cent.T@cent).astype('d'))
            vectors=vectors[:,-26:].astype('f')
            coef=ridge(xx,cent@vectors,1.7)
            self.parts.append((basis,knots,mean,vectors,coef))
        return self

    def from_features(self,x,g):
        logits=kernel(g,self.gate_basis)@self.gate_weights
        probs=softmax((logits*.65+trailing(logits,len(g))*.35)*12.,axis=1).astype('f')
        out=np.zeros((len(x),20),'f')
        for j,(basis,knots,center,vectors,coef) in enumerate(self.parts):
            decoded=(kernel(x,basis)@coef)@vectors.T+center
            angles=decoded[:,:20]
            cdf=np.maximum.accumulate(decoded[:,20:].reshape(len(x),20,-1),axis=-1)
            # Smooth approximation to the median from a monotone estimated CDF.
            gap=np.diff(knots,axis=-1)
            tails=expit((.5-(cdf[:,:,1:]+cdf[:,:,:-1])*.5)/.095)
            median=knots[:,0][None]+(tails*gap[None]).sum(-1)
            out+=probs[:,j,None]*(.30*angles+.70*median)
        return out

    def predict_frames(self,signal):
        return self.from_features(*feature_pair(signal))

    def save(self,path):
        fields={'gatew':self.gate_weights}
        for k,a in enumerate(self.gate_basis):
            fields[f'gate{k}']=a.astype('f2')
        for j,(basis,knots,m,v,c) in enumerate(self.parts):
            for k,a in enumerate(basis):
                fields[f'b{j}_{k}']=a.astype('f2')
            for key,val in [('k',knots),('m',m),('v',v),('c',c)]:
                fields[f'{key}{j}']=val
        np.savez_compressed(path,**fields)

    @classmethod
    def load(cls,path):
        obj=cls()
        with np.load(path) as z:
            obj.gate_basis=tuple(z[f'gate{k}'].astype('f') for k in range(4))
            obj.gate_weights=z['gatew']
            obj.parts=[]
            for j in range(len(ACTIONS)):
                basis=tuple(z[f'b{j}_{k}'].astype('f') for k in range(4))
                obj.parts.append((basis,*(z[f'{key}{j}'] for key in ('k','m','v','c'))))
        return obj
