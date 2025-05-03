import importlib

# Used to activate or deactivate MLFlow tracking from the learning scripts

# Global state to store whether MLflow is enabled and whether we've done a "real" import yet.
_MLFLOW_FLAG = False
_ACTUAL_MLFLOW = None


class _MLFlowDummy:
    """Dummy element that can be called with everything."""

    def __getattribute__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def get_tracking_uri(self):
        return None


def set_mlflow(flag: bool):
    """
    Called once (in main.py, after reading Hydra config) to set whether
    or not we want real MLflow or dummy MLflow.
    """
    global _MLFLOW_FLAG
    _MLFLOW_FLAG = flag


def get_mlflow():
    """
    Returns either the real mlflow module (imported on demand)
    or a dummy mlflow object, depending on _MLFLOW_FLAG.
    """
    global _ACTUAL_MLFLOW
    if _ACTUAL_MLFLOW is None:
        if _MLFLOW_FLAG:
            # Do an on-demand import of mlflow
            _ACTUAL_MLFLOW = importlib.import_module("mlflow")
        else:
            # Return a dummy no-op mlflow
            _ACTUAL_MLFLOW = _MLFlowDummy()
    return _ACTUAL_MLFLOW
