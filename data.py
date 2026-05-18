import torch
from torch.utils.data import Dataset
from torchaudio.functional import resample
import soundfile as sf
import numpy
from extraUtils.misc import get_leaf_files

BASE_PATH = 'Data'

class VCTKCorpus(Dataset):
    
    def __init__(self, path:str=BASE_PATH, limit:int=200e4, hsr:int=24000)->None:
        super().__init__()
        
        self.files = get_leaf_files(path=path, ender=('.mp4', '.wav'))
        numpy.random.shuffle(self.files)
        self.files = self.files[:int(min(limit,len(self.files)))]
        self.lsr = 8000
        self.hsr = hsr
        self.segment_size = int(self.hsr * 0.5)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index)->tuple[torch.Tensor]:
        
        data, sr = sf.read(self.files[index])
        hrw = torch.from_numpy(data).float()
        if hrw.ndim == 1:
            hrw = hrw.unsqueeze(0)
        else:
            hrw = hrw.transpose(0, 1)

        orig_segment_size = int(sr * 0.5)
        if hrw.shape[-1] > orig_segment_size:
            start = torch.randint(0, hrw.shape[-1] - orig_segment_size + 1, (1,))
            hrw = hrw[:, start : start + orig_segment_size]
        else:
            hrw = torch.nn.functional.pad(hrw, (0, orig_segment_size - hrw.shape[-1]))
        
        if sr != self.hsr:
            hrw = resample(hrw, sr, self.hsr)
        lrw = resample(hrw, self.hsr, self.lsr)
        
        return lrw, hrw
    
dataset2x:tuple[torch.Tensor] = VCTKCorpus()