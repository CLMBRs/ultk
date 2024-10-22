
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from learn_quant.training import MV_LSTM
from lightning.pytorch.callbacks import EarlyStopping

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
    
class ThresholdEarlyStopping(EarlyStopping):
    def __init__(self, threshold, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def on_validation_end(self, trainer, pl_module):
        # Get the current value of the monitored metric
        current = trainer.callback_metrics.get(self.monitor)

        # Proceed only if the current metric is available
        if current is None:
            return

        # Check the threshold condition based on the mode
        if self.mode == 'min' and current > self.threshold:
            # Do not consider early stopping if the metric is above the threshold
            return
        elif self.mode == 'max' and current < self.threshold:
            # Do not consider early stopping if the metric is below the threshold
            return

        # If threshold condition is met, proceed with the usual early stopping checks
        super().on_validation_end(trainer, pl_module)