import os
from dataset import GenderDataset
from model import GenderClassificator
from conv_model import GenderClassificatorConvModel
import argparse
from dataflow.utils import read_yaml
import soundfile as sf
import librosa
import torch


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        type=str,
                        required=True)
    parser.add_argument('--audio_path',
                        type=str,
                        nargs='+',
                        required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parser = arg_parser()
    config = read_yaml(parser.config_path)
    audio_paths = parser.audio_path

    audios = []

    for audio_path in audio_paths:
        data_raw, samplerate = sf.read(audio_path)
        data = librosa.resample(data_raw, target_sr=config['sr'], orig_sr=samplerate)
        duration = data.shape[0] / config['sr']
        data = torch.tensor(data[:config['sr'] * 2], dtype=torch.float32).view(1, -1)
        audios.append(data)

    data = torch.cat(audios)

    check_path = '/Users/eric/Desktop/DL/Gender_Classification/model_logs/version_2/checkpoints/epoch=07-val_acc_epoch=0.974079.ckpt'
    # model = GenderClassificator.load_from_checkpoint(check_path)
    model = GenderClassificatorConvModel.load_from_checkpoint(check_path)
    model.eval()
    y_prob = model(data)
    print(y_prob)

    for idx, y in enumerate(y_prob):
        if y >= 0.5:
            print(f'Label of {audio_paths[idx].split(os.sep)[-1]} - M')
        else:
            print(f'Label of {audio_paths[idx].split(os.sep)[-1]} - F')
