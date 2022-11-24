from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torch


class GenderDataset(Dataset):
    def __init__(self, paths):
        self.data = pd.concat((pd.read_csv(data) for data in paths), ignore_index=True)

    def __getitem__(self, idx):
        waveform, _ = torchaudio.load(self.data['path'][idx])
        x = waveform.view(-1)
        y = torch.tensor(self.data['label'] == 'M', dtype=torch.float32)
        return x, y[idx].unsqueeze(-1)

    def __len__(self):
        return len(self.data)
