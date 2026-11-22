"""Causal adaptations of TDS/LSTM and NeuroPose for six-channel AIRI EMG."""
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

STRIDE = 25
SHARE = .25
SEED = 4827


@dataclass(frozen=True)
class Config:
    family: str
    width: int = 32
    kernel: int = 5
    lr: float = .001
    windows: tuple = (25, 50, 100)


def grid():
    variations = ({}, {'width': 64}, {'kernel': 3}, {'lr': .0005},
                  {'windows': (12, 25, 50)})
    return [Config(family, **v) for family in ('tds_lstm', 'neuropose') for v in variations]


def features(signal, windows):
    x = np.asarray(signal, dtype=np.float64)
    increments = np.abs(np.diff(x, axis=0, prepend=x[:1]))
    values = np.concatenate((np.abs(x), x*x, increments), axis=1)
    sums = np.vstack((np.zeros((1, 18)), values.cumsum(axis=0)))
    stop = np.arange(1, len(x)+1, STRIDE)
    blocks = []
    for window in windows:
        start = np.maximum(0, stop-window)
        mean = (sums[stop] - sums[start]) / (stop-start)[:, None]
        mean[:, 6:12] = np.sqrt(np.maximum(0, mean[:, 6:12]))
        blocks.append(mean)
    return np.ascontiguousarray(np.concatenate(blocks, axis=1), dtype=np.float32)


def fit_norm(training):
    x = np.concatenate(training)
    scale = np.maximum(x.mean(axis=0, dtype=np.float64), 1e-10).astype('f')
    logged = np.log1p(x/scale)
    mean = logged.mean(axis=0, dtype=np.float64).astype('f')
    std = np.maximum(logged.std(axis=0, dtype=np.float64), .02).astype('f')
    return dict(scale=scale.tolist(), mean=mean.tolist(), std=std.tolist())


def transform(x, norm):
    scale, mean, std = (np.asarray(norm[key], dtype=np.float32) for key in ('scale','mean','std'))
    return torch.from_numpy(np.ascontiguousarray(((np.log1p(x / scale)-mean)/std).T))


class CausalConv(nn.Conv1d):
    def forward(self, x):
        return super().forward(F.pad(x, ((self.kernel_size[0]-1)*self.dilation[0], 0)))


class TDSBlock(nn.Module):
    def __init__(self, width, kernel):
        super().__init__()
        self.kernel = kernel
        self.time = nn.Conv2d(4, 4, (1, kernel))
        self.temporal_norm = nn.LayerNorm(width)
        self.feature = nn.Sequential(nn.Linear(width,width), nn.ReLU(), nn.Linear(width,width))
        self.feature_norm = nn.LayerNorm(width)

    def forward(self, x):
        b, c, t = x.shape
        groups = x.reshape(b, 4, c//4, t)
        convolved = self.time(F.pad(groups,(self.kernel-1,0))).relu().reshape(b,c,t)
        seq = self.temporal_norm((x+convolved).transpose(1,2))
        return self.feature_norm(seq+self.feature(seq)).transpose(1,2)


class TDSLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        w, k = config.width, config.kernel
        self.encoder = nn.Sequential(CausalConv(54,w,k), nn.SiLU(), TDSBlock(w,k), TDSBlock(w,k))
        self.recurrent = nn.LSTM(w,w,batch_first=True)
        self.readout = nn.Sequential(nn.Linear(2*w,w), nn.SiLU(), nn.Linear(w,20))

    def forward(self, x):
        seq = self.encoder(x).transpose(1,2)
        memory, _ = self.recurrent(seq)
        return self.readout(torch.cat((seq,memory),dim=-1)).transpose(1,2)


class NeuroPose(nn.Module):
    def __init__(self, config):
        super().__init__()
        w, k = config.width, config.kernel
        self.stem = CausalConv(54,w,k)
        self.down1 = CausalConv(w,w,k,stride=2)
        self.down2 = CausalConv(w,w,k,stride=2)
        self.bottleneck = nn.Sequential(CausalConv(w,w,k), nn.SiLU(), CausalConv(w,w,k))
        self.up1 = CausalConv(2*w,w,k)
        self.up0 = CausalConv(2*w,w,k)
        self.out = nn.Conv1d(w,20,1)

    def forward(self, x):
        first = F.silu(self.stem(x))
        second = F.silu(self.down1(first))
        third = F.silu(self.down2(second))
        third = third + self.bottleneck(third)
        def join(coarse, skip):
            past = coarse.repeat_interleave(2,dim=-1)[:,:,:skip.shape[-1]]
            return torch.cat((past,skip),dim=1)
        up = F.silu(self.up1(join(third, second)))
        return self.out(F.silu(self.up0(join(up, first))))


def build(config):
    return {'tds_lstm': TDSLSTM, 'neuropose': NeuroPose}[config.family](config)


class LiteraturePredictor:
    def __init__(self, path):
        saved = torch.load(Path(path), weights_only=True, map_location='cpu')
        self.config = Config(**saved['config'])
        self.norm = saved['norm']
        self.net = build(self.config)
        self.net.load_state_dict(saved['state'])
        self.net.eval()

    @torch.inference_mode()
    def predict_frames(self, signal):
        raw_features = features(signal, self.config.windows)
        raw = torch.from_numpy(np.ascontiguousarray(raw_features.T))[None]
        scale, mean, std = [
            torch.tensor(self.norm[key], dtype=torch.float32).view(1, -1, 1)
            for key in ('scale', 'mean', 'std')
        ]
        normalized = (torch.log1p(raw / scale) - mean) / std
        return self.net(normalized)[0].T.contiguous().numpy()

    def predict(self, signal):
        return self.predict_frames(signal).repeat(STRIDE, axis=0)[:len(signal)]
