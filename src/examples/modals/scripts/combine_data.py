import pandas as pd

from yaml import load
from tqdm import tqdm

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def yaml_to_dataframe(filename: str, keys: list[str]) -> pd.DataFrame:
    with open(filename, "r") as f:
        language_dicts = load(f, Loader=Loader)
    return pd.DataFrame.from_records(
        [{key: lang_dict[key] for key in keys} for lang_dict in language_dicts]
    )


if __name__ == "__main__":
    keys = ["name", "comm_cost", "complexity", "type", "degree_iff"]
    dominating_languages = yaml_to_dataframe(
        "modals/outputs/dominating_languages.yml", keys
    )
    explored_languages = yaml_to_dataframe(
        "modals/outputs/explored_languages.yml", keys
    )
    natural_languages = yaml_to_dataframe("modals/outputs/natural_languages.yml", keys)
    all_data = pd.concat(
        [explored_languages, dominating_languages, natural_languages], ignore_index=True
    )

    from ultk.effcomm.tradeoff import pareto_min_distances

    all_points = all_data[["complexity", "comm_cost"]].values
    pareto_points = all_data[all_data["type"] == "dominant"][
        ["complexity", "comm_cost"]
    ].values
    min_distances = pareto_min_distances(points=all_points, pareto_points=pareto_points)
    all_data["distance"] = min_distances

    all_data.to_csv("modals/outputs/combined_data.csv", index=False)
