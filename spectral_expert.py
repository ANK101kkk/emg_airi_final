"""Causal spatial TDS with an EMG-only gesture posterior and per-gesture angle residual."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from spatial_tds import spatial, trailing, Config, SpatialTDS, features


def summaries(z):
    ids=np.arange(len(z))
    # Each 27-column block contains six log energies followed by spatial contrasts and correlations.
    core=z[:,np.r_[0:6,12:27,54:60,81:87,108:114]]
    mean=trailing(core,ids,len(z))
    var=np.maximum(0,trailing(core**2,ids,len(z))-mean*mean)
    return np.ascontiguousarray(np.column_stack([mean,np.sqrt(var),trailing(core,ids,40)]),dtype='f')


class SpectralExpert(SpatialTDS):
    def __init__(self, cfg, width, context):
        super().__init__(cfg,width)
        self.gate=nn.Sequential(nn.Linear(context,48),nn.LayerNorm(48),nn.SiLU(),nn.Dropout(.1),nn.Linear(48,8))
        self.experts=nn.Linear(cfg.width*2,8*20)
        nn.init.zeros_(self.experts.weight); nn.init.zeros_(self.experts.bias)
    def forward(self,x,z,g):
        seq=self.temporal.encoder(x.transpose(1,2)).transpose(1,2)+self.adapter(z)
        memory,_=self.temporal.recurrent(seq)
        hidden=torch.cat([seq,memory],dim=2)
        logits=self.gate(g)
        experts=self.experts(hidden).reshape(*hidden.shape[:2],8,20)
        return self.temporal.readout(hidden)+(logits.softmax(-1)[:,:,:,None]*experts).sum(2),logits


class SpectralPredictor:
    def __init__(self,path):
        ck=torch.load(Path(path),map_location='cpu',weights_only=True)
        self.config=Config(**ck['config'])
        self.norm=[np.asarray(ck['norm'][k],dtype='f') for k in ('scale','mean','std')]
        self.snorm=[np.asarray(ck[k],dtype='f') for k in ('smean','sstd')]
        self.gnorm=[np.asarray(ck[k],dtype='f') for k in ('gmean','gstd')]
        self.net=SpectralExpert(self.config,len(self.snorm[0]),len(self.gnorm[0])).eval()
        self.net.load_state_dict(ck['model'])
    @torch.inference_mode()
    def predict_frames(self,signal):
        x=features(signal,self.config.windows);z=spatial(signal);g=summaries(z)
        scale,mean,std=self.norm
        x=torch.from_numpy((np.log1p(x/scale)-mean)/std)[None]
        def transform(value,norm):
            return torch.from_numpy(np.clip((value-norm[0])/norm[1],-6,6))[None]
        return self.net(x,transform(z,self.snorm),transform(g,self.gnorm))[0][0].numpy()
