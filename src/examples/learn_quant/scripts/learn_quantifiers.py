import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import time
import mlflow

from ultk.util.io import read_grammatical_expressions

from ..quantifier import QuantifierModel
from ultk.language.grammar import GrammaticalExpression
from ..util import read_expressions
from ..grammar import quantifiers_grammar
from ..training import QuantifierDataset, train_loop, MV_LSTM


@hydra.main(version_base=None, config_path="../conf", config_name="learn")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("learn_quantifiers")

    quantifiers_grammar.add_indices_as_primitives(4)
    expressions_path = cfg.expressions.output_dir + "X" + str(cfg.expressions.x_size) + "/M" + str(cfg.expressions.m_size) + "/d" + str(cfg.expressions.depth) + "/" + "generated_expressions.yml"
    print("Reading expressions from: ", expressions_path)
    expressions, _ = read_grammatical_expressions(expressions_path, quantifiers_grammar)

    mlflow.log_params(cfg)
    mlflow.set_tag("Notes", cfg.notes)

    if cfg.training.device == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            x = torch.ones(1, device=device)
            print ("MPS device found.")
        else:
            print ("MPS device not found.")
    elif cfg.training.device == "cpu":
        print("Using CPU.")
        device = torch.device("cpu")

    for expression in tqdm(expressions[1:1+cfg.expressions.n_limit]):

        for run in range(cfg.n_runs):
            print("Expression: ", expression)
            print("Run: ", run)
            dataset = QuantifierDataset(expression, representation=cfg.expressions.representation, downsampling=cfg.expressions.downsampling)

            dataset.inputs = dataset.inputs.to(device)
            dataset.targets = dataset.targets.to(device)
            n_features = dataset[0][0].shape[1] # this is number of parallel inputs
            n_timesteps = dataset[0][0].shape[0] # this is number of timesteps

            # split the dataset into train and validation sets
            train_data, validation_data = torch.utils.data.random_split(dataset, [cfg.expressions.split, 1-cfg.expressions.split])

            print("Training set size: ", len(train_data))
            print("Validation set size: ", len(validation_data))

            train_dataloader = DataLoader(train_data, batch_size=cfg.expressions.batch_size, shuffle=True)
            validation_dataloader = DataLoader(validation_data, batch_size=cfg.expressions.batch_size, shuffle=True)

            # create NN
            model = MV_LSTM(n_features, n_timesteps, device=cfg.training.device)
            criterion = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
            model.to(device)

            start = time.time()
            train_loop(train_dataloader, model, criterion, optimizer, cfg.training.epochs)
            end = time.time()
            print("Training time: ", end-start)
            model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                running_vloss = 0.0
                for i, vdata in enumerate(validation_dataloader):
                    v_inputs, v_targets = vdata
                    model.init_hidden(v_inputs.size(0))
                    v_outputs = model(v_inputs)
                    vloss = criterion(v_outputs, v_targets)
                    running_vloss += vloss
            print("Validation loss: ", running_vloss.item())

if __name__ == "__main__":
    main()