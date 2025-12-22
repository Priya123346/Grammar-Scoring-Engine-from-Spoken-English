import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self,csv_path,audio_dir,max_len=16000*5):
        self.df=pd.read_csv(csv_path)
        self.audio_dir=audio_dir
        self.max_len=max_len
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        row=self.df.iloc[idx]
        audio_path=os.path.join(self.audio_dir,row["filename"])
        waveform,sr=torchaudio.load(audio_path)

        if sr!=16000:
            waveform=torchaudio.functional.resample(waveform,sr,16000)
        waveform=waveform.mean(dim=0)
        if waveform.size(0)>self.max_len:
            waveform=waveform[:self.max_len]
        else:
            waveform=torch.nn.functional.pad(waveform(0,self.max_len-waveform.size(0)))
        label=torch.tensor(row["label"],dtype=torch.float)
        return waveform,label

