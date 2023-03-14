import pandas as pd
import plotnine as pn

if __name__ == "__main__":
    combined_data = pd.read_csv("indefinites/outputs/combined_data.csv")
    plot = pn.ggplot(pn.aes(x="complexity", y="comm_cost")) + pn.geom_point(
        combined_data, pn.aes(color="type")
    )
    plot.save("indefinites/outputs/plot.png", width=8, height=6, dpi=300)

