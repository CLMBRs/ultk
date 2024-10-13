
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L

class LightningModel(L.LightningModule):
    def __init__(self, model, criterion, optimizer):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        self.model.init_hidden(x.size(0))
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the train loop.
        x, y = batch
        self.model.init_hidden(x.size(0))
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer