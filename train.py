import os
from dataset import GenderDataset
from model import GenderClassificator
from conv_model import GenderClassificatorConvModel
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import argparse
from dataflow.utils import read_yaml
from utils import get_last_version_number
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        type=str,
                        required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parser = arg_parser()
    config = read_yaml(parser.config_path)
    dataloader_config = config['dataloader']

    train_path = config['data']['train_path']
    val_path = config['data']['val_path']

    train_dataset = GenderDataset(train_path)
    val_dataset = GenderDataset(val_path)

    train_dataloader = DataLoader(train_dataset, batch_size=dataloader_config['batch_size'],
                                  shuffle=True, num_workers=config['dataloader']['num_workers'])
    val_dataloader = DataLoader(val_dataset, batch_size=dataloader_config['batch_size'],
                                shuffle=False, num_workers=config['dataloader']['num_workers'])

    model = GenderClassificatorConvModel()
    version_number = get_last_version_number(config['log_dir'])

    logger = TensorBoardLogger(os.path.join(config['log_dir'], version_number), name='', version='')
    checkpoints_dir = os.path.join(config['log_dir'], version_number, 'checkpoints')

    checkpoint_callback = ModelCheckpoint(save_top_k=5,
                                          filename="{epoch=:02d}-{val_acc_epoch:.6f}",
                                          dirpath=checkpoints_dir,
                                          monitor='val_acc_epoch')
    trainer = Trainer(callbacks=[checkpoint_callback], logger=logger, **config['pl_trainer'])
    trainer.fit(model, train_dataloader, val_dataloader)
    # checkpoint_callback.best_model_path
