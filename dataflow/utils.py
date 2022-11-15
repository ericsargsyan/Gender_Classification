import yaml
import soundfile as sf


def read_audio(path):
    with sf.SoundFile(path, 'r') as f:
        sample_rate = f.samplerate
        samples = f.read(dtype='float32')
    return samples


def read_yaml(path):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config
