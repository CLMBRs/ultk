import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import Dataset, DataLoader, DataLoader
import lightning as L
from sklearn.model_selection import KFold
from lightning.pytorch.loggers import MLFlowLogger

import numpy as np
from tqdm import tqdm
import mlflow

from ultk.util.io import read_grammatical_expressions

from ultk.language.grammar import GrammaticalExpression
from ..util import set_vars, print_vars, determine_start_index, define_index_bounds, reorder_by_index_file
from ..tracking.optionals import set_mlflow, get_mlflow
from ..grammar import add_indices
from ..sampling import DatasetInitializationError, get_data_loaders
from ..training import QuantifierDataset, train_loop, MV_LSTM, set_device, train_base_pytorch
from ..training_lightning import (
    train_lightning,
    LightningModel,
    MLFlowConnectivityCallback,
)
from ..measures import (
    load_grammar,
    load_universe,
    filter_universe,
    calculate_measure
)
import random
from collections.abc import MutableMapping

# HYDRA_FULL_ERROR=1 python -m learn_quant.scripts.script training.lightning=true training.strategy=multirun training.device=cpu model=mvlstm grammar.indices=false

def flatten(dictionary, parent_key="", separator="_"):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def set_and_log_seeds(mainrun=False):
    # Set the seeds
    mlflow = get_mlflow()
    seed = random.randint(0, 999999)

    # Log the seed in MLFlow
    if mainrun:
        mlflow.log_param("mainrun_seed", seed)
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mlflow.log_param("childrun_seed", seed)


def train(
    cfg: DictConfig,
    expression: GrammaticalExpression,
    dataset: Dataset,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    mlf_logger: MLFlowLogger | None,
):
    if cfg.training.lightning:
        train_lightning(
            cfg,
            train_dataloader,
            validation_dataloader,
            mlf_logger,
        )
    else:
        train_base_pytorch(
            cfg, train_dataloader, validation_dataloader
        )

def set_mlflow_experiment(cfg):
    if "mlflow" in cfg.tracking:
        if cfg.tracking.mlflow.active:
            set_mlflow(True)
    else:
        set_mlflow(False)
    mlflow = get_mlflow()
    mlflow.set_tracking_uri(f"http://{cfg.tracking.mlflow.host}:{cfg.tracking.mlflow.port}")
    mlflow.set_experiment(f"{cfg.experiment_name}")
    mlflow.pytorch.autolog()
    return mlflow

@hydra.main(version_base=None, config_path="../conf", config_name="learn")
def main(cfg: DictConfig) -> None:
    import sys
    import traceback

    # This try-except is required to circumvent an stderr flushing bug in Hydra
    # See https://github.com/facebookresearch/hydra/issues/2664

    try:

        # Set tracking experiment if tracking is on
        mlflow = set_mlflow_experiment(cfg)

        for key, _ in cfg.tracking.items():
            if "vars":
                set_vars(cfg.tracking[key])

        # Verify settings by printing
        print_vars(cfg)

        # Load grammar

        grammar = load_grammar(cfg.expressions)

        # Add indices as primitives to the grammar if specified. 
        # If expressions.grammar.indices is set to False, no index primitives are added to the grammar.
        grammar, indices_tag = add_indices(
            grammar=grammar,
            indices=cfg.expressions.grammar.indices,
            m_size=cfg.expressions.universe.m_size,
            weight=cfg.expressions.grammar.index_weight,
        )

        expressions_path = cfg.expressions.output_dir + cfg.expressions.target + f"/generated_expressions{indices_tag}.yml"
        expressions, _ = read_grammatical_expressions(expressions_path, grammar)
        print("Read expressions from: ", expressions_path)
        print("Number of expressions: ", len(expressions))

        device = set_device(cfg.training.device)

        print("Loading universe...")
        universe = load_universe(cfg)
        universe = filter_universe(cfg, universe) # This is ensuring that the universe does not have "M-only" or "X-only" subreferents, necessary for monotonicity calculation

        # If an expression is set for training.resume.term_expression, the start index will be set at that expression's index.
        # For running expressions one at a time, use expressions.index.
        # Determine index bounds will use expressions.n_limit if set to determine the number of expressions to run.
        start_index = determine_start_index(cfg, expressions)
        start, end = define_index_bounds(cfg, start_index)
        print(start, end)

        # If index_file is set, make sure to reorder the expressions according to the index file. Useful for getting random samples.
        if "index_file" in cfg.expressions:
            original_index_list = reorder_by_index_file(cfg.expressions.index_file)
        else:
            original_index_list = list(range(len(expressions)))

        for original_index in tqdm(original_index_list[start:end]):

            expression = expressions[original_index]
            run_name = f"{expression.term_expression}"
            print("Running experiment: ", run_name)

            with mlflow.start_run(
                log_system_metrics=False, run_name=run_name
            ) as mainrun:

                set_and_log_seeds(mainrun=True)

                # Calculate specified measures
                if "measures" in cfg:
                    for measure in cfg.measures:
                        calculate_measure(cfg, measure, expression, universe)

                mlflow.log_params(flatten(OmegaConf.to_container(cfg)))
                mlflow.log_param("expression", expression.term_expression)
                mlflow.set_tag("Notes", cfg.notes)

                if "mlflow" in cfg.tracking:
                    if cfg.tracking.mlflow.active:
                        mlf_logger = MLFlowLogger(
                            experiment_name=f"{cfg.experiment_name}",
                            log_model=True,
                            tracking_uri=mlflow.get_tracking_uri(),
                            run_id=mainrun.info.run_id,
                        )
                    else:
                        mlf_logger = None

                print("Expression: ", expression.term_expression)
                try:
                    if cfg.expressions.generation_args:
                        print(
                            "Using generation args: ", cfg.expressions.generation_args
                        )
                        dataset = QuantifierDataset(
                            expression,
                            representation=cfg.expressions.representation,
                            downsampling=cfg.expressions.downsampling,
                            generation_args=cfg.expressions.generation_args,
                        )
                    else:
                        print("No generation args provided")
                        dataset = QuantifierDataset(
                            expression,
                            representation=cfg.expressions.representation,
                            downsampling=cfg.expressions.downsampling,
                        )
                except DatasetInitializationError as e:
                    print(f"Skipping the expression due to error: {e}")
                    continue

                dataset.inputs = dataset.inputs.to(device)
                dataset.targets = dataset.targets.to(device)

                if cfg.training.strategy == "kfold":

                    kfold = KFold(n_splits=cfg.training.k_splits, shuffle=True)
                    print(
                        "Running k-fold training with {} splits".format(
                            cfg.training.k_splits
                        )
                    )
                    for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset)):

                        with mlflow.start_run(
                            run_name=f"{fold}", nested=True
                        ) as childrun:
                            set_and_log_seeds()
                            train_dataloader, validation_dataloader = get_data_loaders(cfg, 
                                                                                       dataset, 
                                                                                       mode=cfg.training.strategy, 
                                                                                       train_val_ids=(train_ids, valid_ids),
                                                                                       fold=fold)
                            train(
                                cfg,
                                expression,
                                dataset,
                                train_dataloader,
                                validation_dataloader,
                                mlf_logger,
                            )

                elif cfg.training.strategy == "multirun":

                    for i in range(cfg.training.n_runs):

                        with mlflow.start_run(run_name=f"{i}", nested=True) as childrun:
                            
                            set_and_log_seeds()
                            train_dataloader, validation_dataloader = get_data_loaders(cfg, dataset, mode=cfg.training.strategy, i=i)
                            train(
                                cfg,
                                expression,
                                dataset,
                                train_dataloader,
                                validation_dataloader,
                                mlf_logger,
                            )

    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        # fflush everything
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()
