
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from learn_quant.training import MV_LSTM

class LightningModel(L.LightningModule):
    def __init__(self, model, criterion, optimizer):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        if isinstance(self.model, MV_LSTM):
            self.model.init_hidden(x.size(0))
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('global_step', self.global_step, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the train loop.
        x, y = batch
        if isinstance(self.model, MV_LSTM):
            self.model.init_hidden(x.size(0))
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('global_step', self.global_step, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer