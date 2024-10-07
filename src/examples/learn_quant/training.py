
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ultk.language.grammar import GrammaticalExpression
from typing import Iterable

import torch
from torch.utils.data import Dataset

from learn_quant.quantifier import QuantifierModel

import mlflow

class ExpressionDataset(Dataset):

    """Creates a PyTorch dataset from a collection of GrammaticalExpressions.
    Adds all QuantifierModels from all Grammatical Expressions to the dataset.
    Mode can be any of the levels associated with the `binarize` function of the QuantifierModel class.
    """

    def __init__(self, expressions: Iterable[GrammaticalExpression], representation="one_hot"):
        """
        Args:
            data (list or array-like): A collection of objects to be transformed into vector representations.
            metadata (list or array-like, optional): Additional information or labels tied to each data object.
            transform (callable, optional): A function/transform that converts an object to its vector representation.
        """
        self.expressions = expressions
        inputs = []
        targets = []
        for expression in expressions:
            for model in expression.meaning.mapping:
                in_meaning_value = expression.meaning.mapping[model]
                if in_meaning_value:
                    y_vector = torch.tensor([1, 0], dtype=torch.float32)
                else:
                    y_vector = torch.tensor([0, 1], dtype=torch.float32)
                inputs.append(torch.tensor(model.to_numpy(mode=representation), dtype=torch.float32))
                targets.append(y_vector)
        self.inputs = torch.stack(inputs, dim=0)
        self.targets = torch.stack(targets, dim=0)

        assert len(self.inputs) == len(self.targets), "Data and metadata must have the same length"

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Return both the transformed data and the corresponding metadata
        return self.inputs[idx], self.targets[idx]
    
    def __get_label(self, idx):
        return self.targets[idx]

def downsample_quantifier_models(expression: GrammaticalExpression):
    print("Downsampling to the smallest truth-value class for:", expression)

    # Get the truth value counts of the expressions
    value_array = expression.meaning.get_binarized_meaning()
    values, counts = np.unique(value_array, return_counts=True)
    value_counts = dict(zip(values, counts))
    sorted_counts = sorted(value_counts.items(), key=lambda x: x[1])
    assert len(sorted_counts) == 2, "Only two classes are supported"

    # Get the indices of the minimum and maximum classes
    minimum_class_indices = np.where(value_array == sorted_counts[0][0])[0]
    max_indices = np.where(value_array == sorted_counts[1][0])[0]
    maximum_class_indices = np.random.choice(max_indices, sorted_counts[0][1], replace=False)
    meaning = np.array(list(expression.meaning.mapping))

    # Combine the expressions indexed by the sampled expressions from the minimum and maximum classes
    sampled_expressions = np.concatenate([meaning[minimum_class_indices], meaning[maximum_class_indices]])
    # Shuffle the expressions
    sampled_expressions = np.random.permutation(sampled_expressions)
    return sampled_expressions

class QuantifierDataset(Dataset):

    """Creates a PyTorch dataset from a collection of QuantifierModels.
    Mode can be any of the levels associated with the `binarize` function of the QuantifierModel class.
    """

    def __init__(self, expression: GrammaticalExpression, representation="one_hot", downsampling=True):
        """
        Args:
            data (list or array-like): A collection of objects to be transformed into vector representations.
            metadata (list or array-like, optional): Additional information or labels tied to each data object.
            transform (callable, optional): A function/transform that converts an object to its vector representation.
        """
        self.expression = expression
        inputs = []
        targets = []

        if downsampling:
            models = downsample_quantifier_models(expression)
        else:
            models = expression.meaning.mapping

        for model in models:
            in_meaning_value = expression.meaning.mapping[model]
            if in_meaning_value:
                y_vector = torch.tensor([1, 0], dtype=torch.float32)
            else:
                y_vector = torch.tensor([0, 1], dtype=torch.float32)
            inputs.append(torch.tensor(model.to_numpy(mode=representation), dtype=torch.float32))
            targets.append(y_vector)
        self.inputs = torch.stack(inputs, dim=0)
        self.targets = torch.stack(targets, dim=0)

        assert len(self.inputs) == len(self.targets), "Data and metadata must have the same length"

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Return both the transformed data and the corresponding metadata
        return self.inputs[idx], self.targets[idx]
    
    def __get_label(self, idx):
        return self.targets[idx]
    
class MV_LSTM(torch.nn.Module):
    def __init__(self,n_features,seq_length, device="cpu"):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 20 # number of hidden states
        self.n_layers = 3 # number of LSTM layers (stacked)
        self.device=device
    
        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True,
                                 device=device)
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden*self.seq_len, 2, device=device)
        
    
    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden, device=self.device)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden, device=self.device)
        self.hidden = (hidden_state, cell_state)
    
    
    def forward(self, x):        
        batch_size, seq_len, _ = x.size()
        
        lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        # lstm_out(with batch_first = True) is 
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size,-1)
        return self.l_linear(x)
    
def compute_accuracy(outputs, targets):
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(outputs)
    
    # Apply a threshold of 0.5 to convert probabilities to binary predictions (0 or 1)
    preds = (probs > 0.5).float()

    # Compare predictions with targets
    correct = (preds == targets).float()

    # Compute accuracy
    accuracy = correct.sum() / torch.numel(correct)

    return accuracy
    
def train_loop(dataloader, model, criterion, optimizer, epochs):

    def check_conditions(ledger, metrics):
        if metrics["accuracy"] > 0.99:
            ledger['accuracy_tally'] += 1
        if metrics["loss"] < 0.01:
            ledger['loss_tally'] += 1
        if ledger['accuracy_tally'] >= 100 or ledger['loss_tally'] >= 100:
            return True
        return False

    size = len(dataloader.dataset)
    # Loop over epochs
    ledger = {"loss_tally": 0, "accuracy_tally": 0}
    terminate = False
    while not terminate:
        for epoch in range(epochs):
            metrics = {"loss": 0, "accuracy": 0, "batch": 0, "epoch": 0}
            running_loss = 0.0
            running_accuracy = 0.0
            print(f"Epoch {epoch+1}/{epochs}")
            # Set the model to training mode
            model.train()
            for batch, (X, y) in enumerate(dataloader):

                if check_conditions(ledger, metrics):
                    print("Conditions met. Stopping training.")
                    return

                # Initialize the hidden state (assuming this is an RNN-based model)
                model.init_hidden(X.size(0))
                
                # Zero out gradients before the backward pass
                optimizer.zero_grad()

                # Compute prediction and loss
                pred = model(X)
                loss = criterion(pred, y)

                # Backpropagation
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Compute minibatch accuracy
                accuracy = compute_accuracy(pred, y)
                running_accuracy += accuracy.item()

                # Print progress
                print(f"Epoch {epoch+1}, Batch {batch}, Cumulative Batch {epoch*len(dataloader)+batch}, Accuracy: {accuracy.item()}, Loss: {loss.item():.4f}")
                metrics = {"loss": loss.item(), "accuracy": accuracy.item(), "batch": batch, "epoch": epoch}

            # Average training loss and accuracy per epoch
            avg_train_loss = running_loss / len(dataloader)
            avg_train_accuracy = running_accuracy / len(dataloader)

            # Optional: print overall progress at the end of the epoch
            print(f"Epoch {epoch+1} completed\n")