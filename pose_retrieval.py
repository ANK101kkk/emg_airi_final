"""A causal gesture-conditioned readout for one frozen TDS–LSTM."""
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from scipy.special import softmax
from literature_models import features
from physiology_tds import ContinuedTDSPredictor

ACTIONS = ('Index', 'Middle', 'Little', 'Ring', 'Thumb', 'Fist', 'Open', 'Close')


def running_mean(x, length=None):
    end = 1 + np.arange(len(x))
    start = np.zeros_like(end) if length is None else np.maximum(0, end - length)
    sums = np.vstack((np.zeros((1, x.shape[1])), np.cumsum(x, axis=0, dtype=np.float64)))
    return ((sums[end] - sums[start]) / (end - start)[:, None]).astype('f')


def lag(x, frames):
    return x[np.maximum(0, np.arange(len(x)) - frames)]


class Encoder:
    def __init__(self, checkpoint):
        member = ContinuedTDSPredictor(checkpoint)
        self.net, self.norm = member.members[0]
        if self.net.use_context:
            raise ValueError('A plain TDS model is required')
        self.windows = member.windows

    @torch.inference_mode()
    def from_raw(self, raw):
        x = torch.from_numpy(np.ascontiguousarray(raw.T))[None]
        scale, mean, std = self.norm
        stem = self.net.original.encoder((torch.log1p(x / scale) - mean) / std).transpose(1, 2)
        memory, _ = self.net.original.recurrent(stem)
        z = torch.cat((stem, memory), dim=-1)
        heads = self.net.original.readout
        latent = heads[1](heads[0](z))
        forecast = heads[2](latent)[0].numpy()
        # The hidden activations are EMG observations, not inferred joint-angle labels.
        return (forecast.copy(), z[0].numpy().copy(), latent[0].numpy().copy())

    def __call__(self, signal):
        return self.from_raw(features(signal, self.windows))


def features_from_encoder(encoded):
    pose, hidden, latent = encoded
    mean = running_mean(pose)
    rms = np.sqrt(np.maximum(0, running_mean(pose**2) - mean**2))
    hid_mean = running_mean(hidden)
    gate = np.column_stack((hid_mean, running_mean(hidden, 40), running_mean(latent), mean, rms))
    dynamics = np.column_stack((pose, pose-lag(pose,1), pose-lag(pose,3), latent, hidden, mean))
    return gate.astype('f'), dynamics.astype('f')


def ridge(x, y, alpha, w=None):
    n = len(x)
    w = np.ones(n, dtype='f') if w is None else w
    gram = x.T @ (w[:, None] * x)
    diag = np.arange(1, len(gram))
    gram[diag, diag] += alpha * n
    gram[0, 0] += 1e-3
    return cho_solve(cho_factor(gram, lower=True, check_finite=False), x.T @ (w[:, None]*y), check_finite=False)


def make_basis(x, nc, seed):
    mu, scale = x.mean(0), np.maximum(x.std(0), .08)
    standardized = np.clip((x-mu) / scale, -5, 5)
    rng = np.random.default_rng(seed)
    anchors = standardized[rng.choice(len(x), size=min(nc,len(x)), replace=False)]
    return mu, scale, anchors


def transform(x, basis):
    mu, scale, anchors = basis
    z = np.clip((x-mu) / scale, -5, 5)
    distance = np.maximum(0, (z*z).sum(1)[:,None] + (anchors*anchors).sum(1) - 2*z@anchors.T) / z.shape[1]
    return np.column_stack((np.ones(len(z), dtype='f'), z, 2.5*np.exp(-distance))).astype('f')


def fit(records):
    # The encoder and both supervised readouts use the same permitted training days.
    gx, gy = [], []
    local = {i: [] for i in range(len(ACTIONS))}
    for a, encoded, y in records:
        gate, x = features_from_encoder(encoded)
        # Include prefix snippets in gate learning, but the pose model scores after 2 s.
        ix = np.arange(10, len(x), 7)
        gx.append(gate[ix])
        gy.append(np.broadcast_to(np.eye(8,dtype='f')[a],(len(ix),8)))
        ix = np.arange(40,len(x),2)
        local[a].append((x[ix], y[ix], encoded[0][ix]))
    gx, gy = np.concatenate(gx), np.concatenate(gy)
    gb = make_basis(gx,80,4901)
    gcoef = ridge(transform(gx,gb), gy, .025)
    experts=[]
    for a, examples in local.items():
        x, y, p = (np.concatenate(items) for items in zip(*examples))
        basis = make_basis(x,72,4902+a)
        z=transform(x,basis)
        center=y.mean(0)
        coef=ridge(z,y-center,.06)
        # Three IRLS steps for MAE robustness; fixed ridge shrinks small effects on new days.
        for _ in range(2):
            residual = np.abs(y - center - z @ coef)
            norm = np.maximum(.05, np.std(y,0))
            error = np.median(np.clip(residual/norm, 0, 5),axis=1)
            coef=ridge(z,y-center,.06,np.clip(.3/np.maximum(.08,error),.45,2.0))
        # Nearest postures are learnt exclusively from observed train-set kinematics.
        # Coordinates use equal significance for active joint and one stable floor.
        scale=np.maximum(.13, y.std(0))
        thin=y[::4]
        sub = thin[np.linspace(0,len(thin)-1,144).astype(int)]
        experts.append((basis,coef,center,scale,sub))
    return (gb,gcoef,experts)


def from_encoder(encoded, meta):
    pose=encoded[0]; gx,x=features_from_encoder(encoded)
    gb, gc, specialists=meta
    logits = transform(gx,gb) @ gc
    prob=softmax(11*(.8*logits+.2*running_mean(logits)),axis=1)
    out=np.zeros_like(pose); closest=np.zeros_like(pose)
    for i,(basis,coef,center,scale,bank) in enumerate(specialists):
        direct=transform(x,basis) @ coef + center
        out+=prob[:,i,None]*direct
        pred=(.5*direct+.5*pose)/scale
        ref=bank/scale
        d=np.maximum(0,(pred**2).sum(1)[:,None]+(ref**2).sum(1)-2*pred@ref.T)
        ix=np.argpartition(d,8,axis=1)[:,:8]
        dist=np.take_along_axis(d,ix,axis=1)
        local=np.sum(softmax(-.65*(dist-dist.min(1,keepdims=True)),axis=1)[:,:,None]*bank[ix],axis=1)
        closest+=prob[:,i,None]*local
    return {'conditioned':.85*out+.15*pose, 'retrieval':.5*out+.25*closest+.25*pose}


class ReadoutPredictor:
    def __init__(self, checkpoint, metadata, variant='conditioned'):
        import pickle
        self.encoder=Encoder(checkpoint)
        with Path(metadata).open('rb') as f:
            self.meta = pickle.load(f)
        self.variant=variant

    def predict_frames(self,signal):
        return from_encoder(self.encoder(signal),self.meta)[self.variant]
