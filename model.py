import pytorch_lightning as pl
import torch.optim
from torch import nn
import torchmetrics
from torchaudio.transforms import MFCC
from torch.nn import functional as F


class GenderClassificator(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.mfcc = MFCC()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(161*40, 161*40),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=161*40),
            nn.Linear(161*40, 1),
            nn.Sigmoid()
        )
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.train_f1score = torchmetrics.classification.BinaryF1Score()
        self.val_f1score = torchmetrics.classification.BinaryF1Score()
        self.test_f1score = torchmetrics.classification.BinaryF1Score()
        self.train_auc = torchmetrics.classification.BinaryAUROC()
        self.test_auc = torchmetrics.classification.BinaryAUROC()
        self.val_auc = torchmetrics.classification.BinaryAUROC()

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
        train_f1score = self.train_f1score(pred.int(), y.int())
        train_auroc = self.train_auc(pred.int(), y.int())

        self.log("train_acc", batch_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_f1', train_f1score, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_auroc', train_auroc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        pred_probs = self(x)
        pred = pred_probs > 0.5
        self.val_accuracy(pred.int(), y.int())
        loss = F.binary_cross_entropy(self(x), y)
        val_f1score = self.val_f1score(pred.int(), y.int())
        val_auroc = self.val_auc(pred.int(), y.int())

        self.log("val_loss", loss, logger=True)
        self.log('train_f1', val_f1score, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_auroc', val_auroc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        pred_probs = self(x)
        pred = pred_probs > 0.5
        acc = self.test_accuracy(pred.int(), y.int())
        loss = F.binary_cross_entropy(self(x), y)
        f1score = self.test_f1score(pred.int(), y.int())
        roc_curve = self.test_auc(pred.int(), y.int())
        metrics = {"Loss": loss,
                   "Accuracy": acc}

        # self.log_dict(metrics)

        return metrics

    def test_epoch_end(self, outputs):
        test_epoch_acc = self.test_accuracy.compute()
        self.test_accuracy.reset()
        test_epoch_f1 = self.test_f1score.compute()
        self.test_f1score.reset()
        test_epoch_auroc = self.test_auc.compute()
        self.test_auc.reset()
        self.log("Accuracy_epoch", test_epoch_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("F1_epoch", test_epoch_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("AUC ROC_epoch", test_epoch_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def train_epoch_end(self, outputs):
        train_epoch_acc = self.train_accuracy.compute()
        self.train_accuracy.reset()
        train_epoch_f1 = self.train_f1score.compute()
        self.train_f1score.reset()
        train_epoch_auc = self.train_auc.compute()
        self.train_auc.reset() ### train_stepum nayev logger-tensorboard
        self.log("train_acc_epoch", train_epoch_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1-score_epoch", train_epoch_f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_auc_epoch", train_epoch_auc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def validation_epoch_end(self, outputs):
        val_epoch_acc = self.val_accuracy.compute()
        self.val_accuracy.reset()
        val_epoch_f1 = self.val_f1score.compute()
        self.val_f1score.reset()
        val_epoch_auc = self.val_auc.compute()
        self.val_auc.reset()
        self.log("val_acc_epoch", val_epoch_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1-score_epoch", val_epoch_f1, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_auc_epoch", val_epoch_auc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)
