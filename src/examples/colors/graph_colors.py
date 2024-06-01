"""
Functions for generating, displaying, and saving graphs representing the WCS data.
"""

import matplotlib.pyplot as plt
from pandas import DataFrame
from color_grammar import ColorLanguage
from ultk.language.language import Expression
from color_grammar import ColorLanguage
import plotnine as pn
import pandas as pd
from pathlib import Path
from skimage import io, color


def graph_expression(
    expression: Expression, name: str, current_dir: str, show: bool = False
):
    """Given a ULTK Expression, graph the meaning probabilitistic distribution of the expression.

    Args:
        expression (Expression): Expression to be graphed.
        name (str): Output file name.
        current_dir (str): System directory to be written to.
        show (bool, optional): Whether to show the final graph or not. Defaults to False.
    """
    Path(f"{current_dir}/outputs/lang-color/").mkdir(parents=True, exist_ok=True)

    referents = expression.meaning.referents
    # Set up axis limits for the plot
    plt.xlim(-2, 12)
    plt.ylim(-5, 45)
    for ref in referents:
        if expression.meaning._dist is not None:
            plt.scatter(
                ord(ref.V) - 64, int(ref.H), s=expression.meaning._dist[ref.name] * 10
            )
        else:
            plt.scatter(ord(ref.V) - 64, int(ref.H))
    if show:
        plt.show()

    plt.savefig(f"{current_dir}/outputs/lang-color/{name}.png")
    # Clear the plot
    plt.clf()


def graph_language_color_distribution(language: ColorLanguage, current_dir: str):
    """Given a ULTK Language, graph the meaning probabilistic distributions of the language.

    Args:
        language (Language): Language to be graphed.
        current_dir (str): System directory to be written to.
    """

    Path(f"{current_dir}/outputs/lang-color/").mkdir(parents=True, exist_ok=True)

    # Generate plot for color data across languages
    language_name = language.name

    language_color_data = []
    most_probable_referents = {}
    form_centroid_map = language.centroid()
    rgb_centroid_map = {}
    # Convert the form centroid map from CIELAB to RGB
    for form, centroid in form_centroid_map.items():
        rgb_trio = color.lab2rgb([[centroid]])[0][0]
        # Convert to hexadecimal
        rgb_hex = "#%02x%02x%02x" % (
            int(rgb_trio[0] * 255),
            int(rgb_trio[1] * 255),
            int(rgb_trio[2] * 255),
        )
        rgb_centroid_map[form] = rgb_hex

    for major_term in language.expressions:
        form = major_term.form
        for referent in major_term.meaning.referents:
            if referent.name in most_probable_referents:
                if (
                    major_term.meaning._dist[referent.name]
                    > most_probable_referents[referent.name][1]
                ):
                    most_probable_referents[referent.name] = (
                        form,
                        major_term.meaning._dist[referent.name],
                    )
            else:
                most_probable_referents[referent.name] = (
                    form,
                    major_term.meaning._dist[referent.name],
                )

    for referent, (form, prob) in most_probable_referents.items():
        language_color_data.append(
            (form, ord(referent[0]) - 96, int(referent[1:]), prob, rgb_centroid_map)
        )
    # Get the centroid color of the most probable expression associated with each referent
    language_color_data = pd.DataFrame(
        language_color_data, columns=["form", "V", "H", "prob", "rgb_centroid"]
    )
    plot = (
        pn.ggplot(pn.aes(x="H", y="V"))
        + pn.geom_point(language_color_data, pn.aes(color="form"))
        # + pn.scale_color_manual(values = rgb_centroid_map)
        + pn.ggtitle(f"Color Terms for {language_name}")
    )
    plot.save(
        f"{current_dir}/outputs/lang-color/color-terms-{language_name}.png",
        width=8,
        height=6,
        dpi=300,
    )


def graph_complexity(
    combined_data: DataFrame,
    ib_boundary_points: None,
    current_dir: str,
    show_labels: bool = False,
):
    """Given a DataFrame of complexities, graph the complexities.

    Args:
        complexity (DataFrame): DataFrame of complexities.
    """

    # Generate and save plots
    plot = pn.ggplot(pn.aes(x="complexity", y="comm_cost")) + pn.geom_point(
        combined_data, pn.aes(color="type")
    )
    if show_labels:
        plot += pn.geom_text(
            combined_data[combined_data["type"] == "natural"],
            pn.aes(label="name"),
            ha="left",
            size=5,
        )

    if ib_boundary_points is not None:
        plot += pn.geom_line(
            ib_boundary_points, pn.aes(color="type"), linetype="dashed"
        )

    plot.save(
        f"{current_dir}/outputs/complexity-commcost.png", width=8, height=6, dpi=300
    )

    plot = pn.ggplot(pn.aes(x="complexity", y="informativity")) + pn.geom_point(
        combined_data, pn.aes(color="type")
    )
    if ib_boundary_points is not None:
        plot += pn.geom_line(
            ib_boundary_points, pn.aes(color="type"), linetype="dashed"
        )
    if show_labels:
        plot += pn.geom_text(
            combined_data[combined_data["type"] == "natural"],
            pn.aes(label="name"),
            ha="left",
            size=5,
            nudge_x=0.1,
        )

    plot.save(
        f"{current_dir}/outputs/complexity-informativity.png",
        width=8,
        height=6,
        dpi=300,
    )

    plot = (
        pn.ggplot(pn.aes(x="informativity", y="comm_cost"))
        # + pn.geom_line(ib_boundary_points, pn.aes(color="type"), linetype="dashed")
        + pn.geom_point(combined_data, pn.aes(color="type"))
    )

    plot.save(
        f"{current_dir}/outputs/informativity-commcost.png", width=8, height=6, dpi=300
    )

    print("Finished")
