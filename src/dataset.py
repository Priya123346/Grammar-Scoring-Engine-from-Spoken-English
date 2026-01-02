import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self,csv_path,audio_dir,max_len=16000*5,has_labels=True):
        self.has_labels=has_labels
        self.df=pd.read_csv(csv_path)
        self.audio_dir=audio_dir
        self.max_len=max_len
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        row=self.df.iloc[idx]
        filename = row["filename"]
        if not filename.endswith(".wav"):
            filename = filename + ".wav"
        audio_path = os.path.join(self.audio_dir, filename)
        waveform,sr=torchaudio.load(audio_path)
        #waveform represents (channels, samples),channels=1 for mono,2 for stereo which defines number of audio channels
        if sr!=16000:
            waveform=torchaudio.functional.resample(waveform,sr,16000)
        waveform=waveform.mean(dim=0)
        if waveform.size(0)>self.max_len:
            waveform=waveform[:self.max_len]
        else:
            waveform=torch.nn.functional.pad(waveform(0,self.max_len-waveform.size(0)))
        if not self.has_labels:
            return waveform
        label=torch.tensor(row["label"],dtype=torch.float)
        return waveform,label

