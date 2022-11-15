import soundfile as sf
import os
from tqdm import tqdm
from glob import glob
import pandas as pd
from collections import defaultdict
import numpy as np


class TimitImporter:
    def __init__(self, config):
        self.target_dir = os.path.join(config['target_dir'], "TIMIT")
        os.makedirs(self.target_dir, exist_ok=True)
        self.sr = config['sr']
        self.max_duration = config['max_duration']
        self.source_path = config['datasets']['timit']['source_path']
        self.csv_path = os.path.join(config['target_dir'], "labels")

    def _write_csv(self):
        pass

    def import_dataset(self):
        for split in ["TRAIN", "TEST"]:
            path = os.path.join(self.source_path, 'data', split, '*', '*', '*')

            audios = []
            for file_name in glob(path):
                if file_name[-3:].upper() == 'WAV':
                    audios.append(file_name)

            if split == "TRAIN":
                names = defaultdict(lambda: [])
                for audio in audios:
                    name = f"{audio.split(os.sep)[9]}{os.sep}{audio.split(os.sep)[10]}"
                    names[name].append(audio)

                np.random.seed(12)
                train_audio_names = np.random.choice(np.array([i for i in names.keys()]),
                                                     size=int(len(names) * 0.9),
                                                     replace=False)
                train_audios = []
                for name in train_audio_names:
                    train_audios.append(names[name])

                val_audios = [audio for audio in audios if audio not in np.array(train_audios).flatten()]

                self.process_data(np.array(train_audios).flatten(), "TRAIN")
                self.process_data(val_audios, "VAL")
            else:
                self.process_data(audios, split)

    def process_data(self, audios, split):
        os.makedirs(os.path.join(self.target_dir, split), exist_ok=True)
        ind = 0
        paths = []
        labels = []
        for audio in tqdm(audios):
            data, samplerate = sf.read(audio)
            new_audio_filepath = os.path.join(self.target_dir, f"{split}{os.sep}{ind}.wav")

            if (data.shape[0] / samplerate) > self.max_duration:
                data = data[:samplerate * self.max_duration]
            else:
                diff = self.max_duration * self.sr - data.shape[0]
                data = np.pad(data, pad_width=(0, diff))
            sf.write(new_audio_filepath, data, self.sr)

            paths.append(new_audio_filepath)
            labels.append(audio.split(os.sep)[-2][0])

            ind += 1

        pd.DataFrame({'path': paths,
                      'label': labels}).to_csv(os.path.join(self.csv_path, f'timit_{split}.csv'), index=False)
