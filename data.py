import torch
from torch.utils.data import Dataset
from torchaudio import load
from torchaudio.functional import resample

import numpy
from extraUtils.misc import get_leaf_files

BASE_PATH = 'Data'

class VCTKCorpus(Dataset):
    
    def __init__(self, path:str=BASE_PATH, dsf:int=2, limit:int=2e4)->None:
        super().__init__()
        
        self.dsf = dsf
        self.files = get_leaf_files(path=path, ender=('.mp4', '.wav'))
        numpy.random.shuffle(self.files)
        self.files = self.files[:min(limit,len(self.files))]
        self.target_sr = 48000
        self.segment_size = int(self.target_sr * 0.3)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        
        hrw, sr = load(self.files[index], channels_first=True)
        if sr != self.target_sr:
            hrw = resample(hrw, sr, self.target_sr)
        
        if hrw.shape[-1] > self.segment_size:
            start = torch.randint(0, hrw.shape[-1] - self.segment_size + 1, (1,))
            hrw = hrw[:, start : start + self.segment_size]
        else:
            hrw = torch.nn.functional.pad(hrw, (0, self.segment_size - hrw.shape[-1]))
        
        lrw = resample(hrw, sr, sr//self.dsf)
        lrw = lrw[:, :(self.target_sr//self.dsf)]
        maxAmp = torch.clamp(lrw.abs().max(), 1e-6)
        
        return lrw / maxAmp, hrw / maxAmp