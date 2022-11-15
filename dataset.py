from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torch


class GenderDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path)

    def __getitem__(self, idx):
        waveform, _ = torchaudio.load(self.data['path'][idx])
        x = waveform
        y = torch.tensor(self.data['label'] == 'M', dtype=torch.float32)
        return x, y[idx].unsqueeze(-1)

    def __len__(self):
        return len(self.data)
