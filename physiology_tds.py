"""Continued TDS–LSTM with optional causal muscle-activation context."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from literature_models import Config, build, features


def reference_envelopes(raw):
    """RMS ratios use only samples available at the frame anchor."""
    rms = np.asarray(raw[:, 24:30], dtype=np.float64)
    sums = np.vstack((np.zeros((1, 6)), rms.cumsum(axis=0)))
    stop = np.arange(1, len(rms) + 1)
    calibration_stop = np.minimum(stop, 40)
    calibration = sums[calibration_stop] / calibration_stop[:, None]
    windows = [calibration]
    for width in (20, 60):
        start = np.maximum(0, stop - width)
        windows.append((sums[stop] - sums[start]) / (stop - start)[:, None])
    return np.ascontiguousarray(np.concatenate(windows, axis=1), dtype=np.float32)


class ContextTDS(nn.Module):
    def __init__(self, config, use_context):
        super().__init__()
        self.original = build(config)
        self.use_context = use_context
        if use_context:
            self.adapter = nn.Sequential(nn.Linear(30, config.width), nn.SiLU(),
                                         nn.Linear(config.width, config.width))
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)

    @staticmethod
    def extra_features(x, refs, scale):
        rms = x[:, 24:30] / scale[:, 24:30]
        calibrated, short, long = (v / scale[:, 24:30] for v in refs.split(6, dim=1))
        floor = .06
        logs = torch.log(rms + floor)
        ratios = [(logs - torch.log(v + floor)).clamp(-4, 4) for v in (calibrated, short, long)]
        distribution = logs - logs.mean(dim=1, keepdim=True)
        return torch.cat((*ratios, distribution.clamp(-4, 4), torch.log1p(calibrated)), dim=1)

    def forward(self, x, refs, normalization):
        scale, mean, std = normalization
        norm_x = (torch.log1p(x / scale) - mean) / std
        if not self.use_context:
            return self.original(norm_x)
        seq = self.original.encoder(norm_x).transpose(1, 2)
        seq = seq + self.adapter(self.extra_features(x, refs, scale).transpose(1, 2))
        memory, _ = self.original.recurrent(seq)
        return self.original.readout(torch.cat((seq, memory), dim=-1)).transpose(1, 2)


class ContinuedTDSPredictor:
    """One seed, or an equal-weight ensemble of two independently fitted seeds."""
    def __init__(self, checkpoints):
        if isinstance(checkpoints, (str, Path)):
            checkpoints = [checkpoints]
        self.members = []
        for path in checkpoints:
            payload = torch.load(path, weights_only=True, map_location='cpu')
            config = Config(**payload['config'])
            network = ContextTDS(config, payload['use_context']).eval()
            network.load_state_dict(payload['state'])
            norm = [torch.tensor(payload['norm'][k], dtype=torch.float32).view(1,-1,1)
                    for k in ('scale', 'mean', 'std')]
            self.members.append((network, norm))
            self.windows = config.windows

    @torch.inference_mode()
    def predict_frames(self, signal):
        x = features(np.asarray(signal, dtype=np.float32), self.windows)
        refs = torch.from_numpy(reference_envelopes(x).T.copy())[None]
        raw = torch.from_numpy(x.T.copy())[None]
        result = [net(raw, refs, norm)[0].T.contiguous().numpy() for net, norm in self.members]
        return np.mean(result, axis=0, dtype=np.float32)
