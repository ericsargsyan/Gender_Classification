import os
import torch
from dataset import GenderDataset
from model import GenderClassificator
import argparse
from dataflow.utils import read_yaml
import soundfile as sf
import librosa
import torchaudio


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        type=str,
                        required=True)
    parser.add_argument('--audio_path',
                        type=str,
                        required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parser = arg_parser()
    config = read_yaml(parser.config_path)
    audio_path = parser.audio_path

    data_raw, samplerate = sf.read(audio_path)
    data = librosa.resample(data_raw, target_sr=config['sr'], orig_sr=samplerate)
    duration = data.shape[0] / config['sr']
    data = torch.tensor(data[:config['sr'] * 2], dtype=torch.float32).view(1, -1)

    # data = torch.cat([data, data]) for multiple people

    check_path = '/Users/eric/Desktop/DL/Gender_Classification/logs/version_2/checkpoints/epoch==00-val_acc_epoch=0.944107.ckpt'
    model = GenderClassificator.load_from_checkpoint(check_path)
    model.eval()
    y = model(data)

    if y >= 0.5:
        print(f'Label of {audio_path.split(os.sep)[-1]} - M')
    else:
        print(f'Label of {audio_path.split(os.sep)[-1]} - F')
