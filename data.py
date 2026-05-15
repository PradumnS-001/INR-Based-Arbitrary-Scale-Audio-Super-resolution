import torch
from torch.utils.data import Dataset
from torchaudio.functional import resample
import soundfile as sf
import numpy
from extraUtils.misc import get_leaf_files

BASE_PATH = 'Data'

class VCTKCorpus(Dataset):
    
    def __init__(self, path:str=BASE_PATH, limit:int=200e4)->None:
        super().__init__()
        
        self.files = get_leaf_files(path=path, ender=('.mp4', '.wav'))
        numpy.random.shuffle(self.files)
        self.files = self.files[:int(min(limit,len(self.files)))]
        self.target_sr = 48000
        self.segment_size = int(self.target_sr * 0.5)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        
        dsf = numpy.random.uniform(2.0,5.0)
        data, sr = sf.read(self.files[index])
        hrw = torch.from_numpy(data).float()
        if hrw.ndim == 1:
            hrw = hrw.unsqueeze(0)
        else:
            hrw = hrw.transpose(0, 1)

        if sr != self.target_sr:
            hrw = resample(hrw, sr, self.target_sr)
        
        if hrw.shape[-1] > self.segment_size:
            start = torch.randint(0, hrw.shape[-1] - self.segment_size + 1, (1,))
            hrw = hrw[:, start : start + self.segment_size]
        else:
            hrw = torch.nn.functional.pad(hrw, (0, self.segment_size - hrw.shape[-1]))
        
        lrw = resample(hrw, int(dsf * 1000), 1000)
        lrw = lrw[:, :int(self.target_sr/dsf)]
        maxAmp = torch.clamp(lrw.abs().max(), 1e-6)
        
        return lrw / maxAmp, hrw / maxAmp
    
dataset2x = VCTKCorpus()