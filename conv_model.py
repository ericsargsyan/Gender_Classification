import pytorch_lightning as pl
import torch.optim
from torch import nn
import torchmetrics
from torchaudio.transforms import MFCC
from torch.nn import functional as F


class GenderClassificatorConvModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.mfcc = MFCC()
        self.model = nn.Sequential(
            # nn.Flatten(),
            nn.Conv1d(40, kernel_size=9, out_channels=32),
            nn.ReLU(),
            nn.Conv1d(32, kernel_size=9, out_channels=16),
            nn.Flatten(),


        )
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        x = self.mfcc(x)
        x = self.model(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred_probs = self(x)
        loss = F.binary_cross_entropy(pred_probs, y)
        pred = pred_probs > 0.5
        batch_accuracy = self.train_accuracy(pred.int(), y.int())

        self.log("train_acc", batch_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        pred_probs = self(x)
        pred = pred_probs > 0.5
        self.val_accuracy(pred.int(), y.int())
        loss = F.binary_cross_entropy(self(x), y)
        self.log("val_loss", loss)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        pred_probs = self(x)
        pred = pred_probs > 0.5
        loss = F.binary_cross_entropy(self(x), y)
        acc = self.test_accuracy(pred.int(), y.int())
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_epoch_end(self, outputs):
        test_epoch_acc = self.test_accuracy.compute()
        self.test_accuracy.reset()
        self.log("test_acc_epoch", test_epoch_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def train_epoch_end(self, outputs):  # ???
        train_epoch_acc = self.train_accuracy.compute()
        self.train_accuracy.reset()
        self.log("train_acc_epoch", train_epoch_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def validation_epoch_end(self, outputs):
        val_epoch_acc = self.val_accuracy.compute()
        self.val_accuracy.reset()
        self.log("val_acc_epoch", val_epoch_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)
