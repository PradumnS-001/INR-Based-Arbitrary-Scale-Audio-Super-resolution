import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchaudio.functional import resample
import soundfile as sf
import numpy as np
from extraUtils.misc import get_leaf_files
from configs import *

np.random.seed(seed)
generator = torch.Generator().manual_seed(seed)
    
class AudioCorpus(Dataset):
    
    def __init__(self, 
                path:str=BASE_PATH, 
                limit:int | None=limit, 
                lsr:int=low_sampling_rate, 
                hsr:int=high_sampling_rate, 
                trunc:bool=True, 
                file_list:list[str]=None)->None:
        super().__init__()
        
        if file_list is None:
            _,self.files = get_leaf_files(path=path, ender=('.mp4', '.wav', '.flac'))
            np.random.shuffle(self.files)
            kk = int(min(limit,len(self.files))) if limit is not None else len(self.files)
            self.files = self.files[:kk]
        else:
            self.files = file_list
        self.lsr = lsr
        self.hsr = hsr
        self.trunc = trunc
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index)->tuple[torch.Tensor]:
        
        data, sr = sf.read(self.files[index])
        hrw = torch.from_numpy(data).float()
        if hrw.ndim == 1:
            hrw = hrw.unsqueeze(0)
        else:
            hrw = hrw.transpose(0, 1)

        if self.trunc:
            orig_segment_size = int(sr * clip_size_sec)
            if hrw.shape[-1] > orig_segment_size:
                start = torch.randint(0, hrw.shape[-1] - orig_segment_size + 1, (1,))
                hrw = hrw[:, start : start + orig_segment_size]
            else:
                hrw = torch.nn.functional.pad(hrw, (0, orig_segment_size - hrw.shape[-1]))
        
        if sr != self.hsr:
            hrw = resample(hrw, sr, self.hsr)
        lrw = resample(hrw, self.hsr, self.lsr)
        
        return lrw, hrw
    
dataset = AudioCorpus()
tr_len = int(0.9 * len(dataset))
val_len = len(dataset) - tr_len
tr_set, val_set = random_split(dataset, [tr_len, val_len], generator=generator)

tr_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True, prefetch_factor=4,drop_last=True)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4, persistent_workers=True, pin_memory=True, prefetch_factor=4)

val12_indices = val_set.indices[:12]
val12_files = [dataset.files[i] for i in val12_indices]
val12_set = AudioCorpus(file_list=val12_files, trunc=False)
val12_loader = DataLoader(val12_set, batch_size=1, shuffle=False, num_workers=0)

@torch.no_grad()
def calc_opcs(data: AudioCorpus) -> torch.Tensor:
    count = 0
    total_sum = 0.0
    total_sq_sum = 0.0
    total_elements = 0
    
    for lr_wav, hr_wav in data:
        lr_wav = resample(lr_wav, low_sampling_rate, high_sampling_rate)
        min_len = min(lr_wav.shape[-1], hr_wav.shape[-1])
        diff = hr_wav[..., :min_len] - lr_wav[..., :min_len]
        total_sum += diff.sum().item()
        total_sq_sum += (diff ** 2).sum().item()
        total_elements += diff.numel()
        
        count += 1
        if count >= (4096 / batch_size): 
            break
    true_variance = (total_sq_sum / total_elements) - (total_sum / total_elements) ** 2
    true_std = torch.tensor(true_variance).sqrt()
    
    return true_std