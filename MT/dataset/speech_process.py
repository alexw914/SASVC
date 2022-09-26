import math, torch, torchaudio, librosa
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import random

def repeat_padding_Tensor(spec, ref_len):
    mul = int(np.ceil(ref_len / spec.shape[1]))
    spec = spec.repeat(1, mul)[:, :ref_len]
    return spec

def load_wav(audio_file, num_frames=398):
    audio, sr = librosa.load(audio_file, sr=16000)
    length = num_frames * 160 + 240
    if audio.shape[0] <= length:
        shortage = length - audio.shape[0]
        audio = np.pad(audio, (0, shortage), 'wrap')
    start_frame = np.int64(random()*(audio.shape[0]-length))
    audio = audio[start_frame:start_frame + length]
    return torch.FloatTensor(audio)

def load_pt(feat_path, num_frames=400):
    data_x = torch.load(feat_path)
    if data_x.shape[1] > num_frames:    
        start_frame = np.int64(random() * (data_x.shape[1]-num_frames))
        data_x = data_x[:, start_frame: start_frame+num_frames]
    if data_x.shape[1] < num_frames:
        data_x = repeat_padding_Tensor(data_x, num_frames)
    return data_x


class PreEmphasis(nn.Module):
    
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class FbankAug(nn.Module):

    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):    
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x

