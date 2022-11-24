from dataset import GenderDataset
from model import GenderClassificator
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import argparse
from dataflow.utils import read_yaml


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        type=str,
                        required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parser = arg_parser()
    config = read_yaml(parser.config_path)

    test_dataset = GenderDataset(config['data']['test_path'])
    test_dataloader = DataLoader(test_dataset, batch_size=config['dataloader']['batch_size'],
                                 shuffle=True, num_workers=config['dataloader']['num_workers'])

    path = '/Users/eric/Desktop/DL/Gender_Classification/logs/version_2/checkpoints/epoch==00-val_acc_epoch=0.944107.ckpt'

    model = GenderClassificator.load_from_checkpoint(path)
    trainer = Trainer(max_epochs=config['pl_trainer']['max_epochs'])
    trainer.test(model, dataloaders=test_dataloader)
