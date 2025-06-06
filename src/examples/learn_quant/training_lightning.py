import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import Dataset, DataLoader, DataLoader
from learn_quant.training import MV_LSTM
from learn_quant.tracking.optionals import get_mlflow
from lightning.pytorch.callbacks import EarlyStopping, Callback
from lightning.pytorch.loggers import MLFlowLogger
from collections import deque
from lightning.pytorch.callbacks import Timer
import logging
import time
import socket

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LightningModel(L.LightningModule):
    def __init__(self, model, criterion, optimizer):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.validation_losses = deque(maxlen=50)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        if isinstance(self.model, MV_LSTM):
            self.model.init_hidden(x.size(0))
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "global_step", self.global_step, on_step=True, on_epoch=False, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the train loop.
        x, y = batch
        if isinstance(self.model, MV_LSTM):
            self.model.init_hidden(x.size(0))
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "global_step", self.global_step, on_step=True, on_epoch=False, prog_bar=True
        )

        self.validation_losses.append(loss.detach().cpu().item())
        # Compute running average over the last 50 losses if deque is full
        if len(self.validation_losses) == 50:
            running_avg_loss = sum(self.validation_losses) / len(self.validation_losses)
            self.log(
                "val_loss_running_avg50",
                running_avg_loss,
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )
            return {
                "global_step": self.global_step,
                "val_loss": loss,
                "val_loss_running_avg50": running_avg_loss,
            }
        else:
            # If deque is not full, do not log 'val_loss_running_avg50' to avoid misleading values
            return {
                "global_step": self.global_step,
                "val_loss": loss,
                "val_loss_running_avg50": None,
            }

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
        if self.mode == "min" and current > self.threshold:
            # Do not consider early stopping if the metric is above the threshold
            return
        elif self.mode == "max" and current < self.threshold:
            # Do not consider early stopping if the metric is below the threshold
            return

        # If threshold condition is met, proceed with the usual early stopping checks
        super().on_validation_end(trainer, pl_module)


def is_mlflow_server_up(mlflow_tracking_uri, timeout=5):
    """Checks if the MLflow server is reachable."""

    if not mlflow_tracking_uri or not isinstance(mlflow_tracking_uri, str):
        # If it's None or not a string, assume we're in dummy mode and skip
        return True  # or False, or short-circuit however you like

    try:
        # Extract hostname and port from the URI
        if mlflow_tracking_uri.startswith("http://"):
            host, port = mlflow_tracking_uri[7:].split(":")
        elif mlflow_tracking_uri.startswith("https://"):
            host, port = mlflow_tracking_uri[8:].split(":")
        else:
            raise ValueError(f"Unsupported tracking URI format: {mlflow_tracking_uri}")
        port = int(port)
        # Attempt a socket connection
        socket.create_connection((host, port), timeout=timeout)
        return True
    except (socket.timeout, socket.gaierror, ConnectionRefusedError, ValueError) as e:
        logger.warning(f"MLflow server check failed: {e}")
        return False


def wait_for_mlflow(max_retries=30, retry_delay=30):
    """Waits for the MLflow server to become available."""
    mlflow = get_mlflow()
    mlflow_tracking_uri = mlflow.get_tracking_uri()
    if mlflow.active_run() is None:
        logger.warning("No active MLflow run. Skipping connectivity check.")
        return
    if not mlflow_tracking_uri:
        raise ValueError("Tracking URI not set.")
    for attempt in range(max_retries):
        if is_mlflow_server_up(mlflow_tracking_uri):
            logger.info("MLflow server is up. Continuing training.")
            return
        else:
            logger.warning(
                f"MLflow server is down. Retrying in {retry_delay} seconds (Attempt {attempt+1}/{max_retries})."
            )
            time.sleep(retry_delay)
    raise RuntimeError("MLflow server is unavailable after multiple retries.")


class MLFlowConnectivityCallback(Callback):
    def __init__(self, retry_delay=30, max_retries=5):
        mlflow = get_mlflow()
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.mlflow_tracking_uri = mlflow.get_tracking_uri()
        logger.info(
            f"MLFlowConnectivityCallback initialized: {self.mlflow_tracking_uri}"
        )
        if not self.mlflow_tracking_uri:
            raise ValueError("Tracking URI not set.")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        mlflow = get_mlflow()
        mlflow_tracking_uri = mlflow.get_tracking_uri()

        #  If there's no real URI (meaning mlflow is dummy), skip connectivity checks
        if not mlflow_tracking_uri or mlflow_tracking_uri.startswith("dummy"):
            logger.debug(
                "MLflow is disabled or dummy; skipping server connectivity check."
            )
            return

        if not is_mlflow_server_up(self.mlflow_tracking_uri):
            logger.warning("MLflow server is down. Attempting to reconnect...")
            try:
                wait_for_mlflow(
                    max_retries=self.max_retries, retry_delay=self.retry_delay
                )
                logger.info("MLflow server reconnected. Resuming logging.")
            except RuntimeError:
                logger.error("MLflow server remains unavailable. Stopping training.")
                trainer.should_stop = True
                exit(1)
            except Exception as e:
                logger.error(f"An unexpected error occurred during reconnection: {e}")
                trainer.should_stop = True
                exit(1)


def get_logger(cfg, mainrun, mlflow):
    if "mlflow" in cfg.tracking:
        from lightning.pytorch.loggers import MLFlowLogger

        mlf_logger = MLFlowLogger(
            experiment_name=f"{cfg.experiment_name}",
            log_model=True,
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=mainrun.info.run_id,
        )
    else:
        from lightning.pytorch.loggers.logger import DummyLogger

        mlf_logger = DummyLogger()
    return mlf_logger


def train_lightning(
    cfg: DictConfig,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    mlf_logger: MLFlowLogger | None,
):

    selected_model = instantiate(cfg.model)
    selected_optimizer = instantiate(cfg.optimizer)

    model = selected_model(device=cfg.training.device)

    optimizer = selected_optimizer(model.parameters())
    lightning = LightningModel(
        model, criterion=instantiate(cfg.criterion), optimizer=optimizer
    )
    timer_callback = Timer()
    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.training.device,
        val_check_interval=0.1,
        logger=mlf_logger,
        callbacks=[
            timer_callback,
            # EarlyStopping(monitor="val_loss_epoch", verbose=True, mode="min", min_delta=.01, patience=3),
            # ThresholdEarlyStopping(
            #            threshold=cfg.training.early_stopping.threshold,
            #            monitor=cfg.training.early_stopping.monitor,
            #            patience=cfg.training.early_stopping.patience,
            #            min_delta=cfg.training.early_stopping.min_delta,
            #            mode=cfg.training.early_stopping.mode,
            #            check_on_train_epoch_end=cfg.training.early_stopping.check_on_train_epoch_end, # Check at the step level, not at the epoch level
            #            ),
            MLFlowConnectivityCallback(),
        ],
    )
    trainer.fit(lightning, train_dataloader, validation_dataloader)
    total_time = timer_callback.time_elapsed("train")
    print(f"Total training time: {total_time:.2f} seconds")
    print(trainer.callback_metrics)
    print("_______")
