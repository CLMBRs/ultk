import pandas as pd
import plotnine as pn

if __name__ == "__main__":
    combined_data = pd.read_csv("modals/outputs/combined_data.csv")

    natural_data = combined_data[combined_data["type"] == "natural"]
    dominant_data = combined_data[combined_data["type"] == "dominant"]
    explored_data = combined_data[combined_data["type"] == "explored"]

    plot = (
        pn.ggplot(
            explored_data,
            pn.aes(
                x="complexity",
                y="comm_cost",
            ),
        )
        + pn.geom_point(
            # color="gray",
            pn.aes(
                fill="degree_iff",
                # shape="type",
            ),
            alpha=0.3,
            size=3,
        )
        + pn.geom_point(
            dominant_data,
            pn.aes(
                fill="degree_iff",
                # shape="type",
            ),        
            color="black",
            size=6,
        )

        + pn.geom_point(  # The natural languages
            natural_data,
            color="red",
            shape="+",
            size=4,
        )
        + pn.geom_label(
            natural_data,
            pn.aes(label="name"),
            ha="left",
            size=6,  # orig 9
            nudge_x=1,
            # color="white",
        )         
        + pn.scale_fill_continuous(
            "cividis",
            name="naturalness", 
        )
        + pn.theme_classic()
        + pn.xlab("Complexity")
        + pn.ylab("Commmunicative Cost")
        + pn.theme(axis_title=pn.element_text(size=14))
        + pn.theme(
            legend_position=(0.8, 0.95),
        )  # (x, y) in normalized figure coordinates
    )

    plot.save("modals/outputs/plot.png", width=8, height=6, dpi=300)

    import numpy as np
    from scipy.stats import ttest_1samp, linregress
    print(
        ttest_1samp(
            explored_data["distance"].values,
            natural_data["distance"].values.mean(),
        )
    )

    print(
        linregress(
            explored_data["distance"].values,
            explored_data["degree_iff"].values,
        )
    )
    # breakpoint()