"""One causal TDS-LSTM with an adapter for relative and cross-electrode activity."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.signal import butter, sosfilt
from literature_models import Config, TDSLSTM, features


def trailing(values, endpoints, width):
    cumulative = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0, dtype=np.float64)])
    end = endpoints + 1
    start = np.maximum(0, end - width)
    return (cumulative[end] - cumulative[start]) / (end - start)[:, None]


def spatial(signal):
    signal = np.asarray(signal, dtype=np.float64) * 1e5
    frame = np.arange(0, len(signal), 25)
    left, right = np.triu_indices(6, 1)
    blocks = []
    for edges in (None, (12,48), (48,105), (105,215)):
        band = signal if edges is None else sosfilt(butter(2, edges, fs=500, btype='bandpass', output='sos'), signal, axis=0)
        products = np.column_stack([band**2, band[:,left]*band[:,right]])
        for width in ((50,150) if edges is None else (100,)):
            stats = trailing(products, frame, width)
            power = stats[:,:6].clip(0)
            log_power = np.log(power + .04)
            corr = stats[:,6:] / np.sqrt((power[:,left]+.04)*(power[:,right]+.04))
            blocks.append(np.column_stack([log_power, log_power-log_power.mean(axis=1,keepdims=True), corr]))
    activity = blocks[1][:,:6]
    k = np.arange(len(frame))
    mean = trailing(activity,k,1000000000)
    std = np.sqrt(np.maximum(0,trailing(activity**2,k,1000000000)-mean**2))
    blocks.extend([mean,std,activity-trailing(activity,k,20)])
    return np.ascontiguousarray(np.column_stack(blocks), dtype=np.float32)


class SpatialTDS(nn.Module):
    def __init__(self, config, n_spatial):
        super().__init__()
        self.temporal = TDSLSTM(config)
        self.adapter = nn.Sequential(nn.Linear(n_spatial,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,config.width))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, original, sp):
        seq = self.temporal.encoder(original.transpose(1,2)).transpose(1,2) + self.adapter(sp)
        memory,_ = self.temporal.recurrent(seq)
        return self.temporal.readout(torch.cat([seq,memory],dim=2))


class SpatialPredictor:
    def __init__(self, path):
        ck = torch.load(Path(path),map_location='cpu',weights_only=True)
        self.config = Config(**ck['config'])
        self.norm = [np.asarray(ck['norm'][k], dtype='f') for k in ('scale','mean','std')]
        self.smean = np.asarray(ck['smean'], dtype='f')
        self.sstd = np.asarray(ck['sstd'], dtype='f')
        self.net = SpatialTDS(self.config,len(self.smean)).eval()
        self.net.load_state_dict(ck['model'])

    @torch.inference_mode()
    def predict_frames(self, signal):
        x=features(signal,self.config.windows)
        z=spatial(signal)
        scale,mean,std=self.norm
        x=torch.from_numpy((np.log1p(x/scale)-mean)/std)[None]
        z=torch.from_numpy(np.clip((z-self.smean)/self.sstd,-6,6))[None]
        return self.net(x,z)[0].numpy()
