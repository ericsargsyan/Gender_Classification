import os
from tqdm import tqdm
import librosa
import soundfile as sf
import numpy as np
import pandas as pd


class CVImporter:
    def __init__(self, config):
        self.target_dir = os.path.join(config['target_dir'], "CV")
        os.makedirs(self.target_dir, exist_ok=True)
        self.sr = config['sr']
        self.max_duration = config['max_duration']
        self.source_path = config['datasets']['cv']['source_path']
        self.csv_path = os.path.join(config['target_dir'], "labels")

    def import_dataset(self):
        for split in ['dev', 'test']:
            os.makedirs(os.path.join(self.target_dir, split.upper()), exist_ok=True)
            dataset = pd.read_csv(os.path.join(self.source_path, f'cv-valid-{split}.csv'))[['filename', 'gender']]
            dataset = dataset[dataset["gender"].isin(('female', 'male'))]
            dataset['path'] = [os.path.join(self.source_path, f'cv-valid-{split}{os.sep}')] + dataset['filename']
            dataset["gender"] = np.where(dataset["gender"] == "male", 'M', 'F')
            dataset.rename(columns={'gender': 'label'})
            dataset = dataset.reset_index()

            for i in range(len(dataset)):
                dataset.loc[i, 'name'] = dataset.loc[0, 'filename'].split(os.sep)[-1]

            paths = []
            labels = []

            for file_name in tqdm(dataset['path'].values):
                data_raw, samplerate = sf.read(file_name)
                data = librosa.resample(data_raw, target_sr=self.sr, orig_sr=samplerate)
                duration = data.shape[0] / self.sr

                if duration > self.max_duration:
                    data = data[:self.sr * self.max_duration]
                else:
                    diff = self.max_duration * self.sr - data.shape[0]
                    data = np.pad(data, pad_width=(0, diff))

                new_audio_filepath = os.path.join(self.target_dir, f"{split}{os.sep}{file_name.split(os.sep)[-1]}")
                new_audio_filepath = new_audio_filepath.split('.')[0]
                paths.append(f'{new_audio_filepath}.wav')
                labels.append(dataset['gender'].values)

                sf.write(f'{new_audio_filepath}.wav', data, self.sr)

            pd.DataFrame({'path': paths,
                          'label': labels[0]}).to_csv(os.path.join(self.csv_path, f'CVoice_{split}.csv'), index=False)
