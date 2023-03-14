import pandas as pd


if __name__ == "__main__":

    haspelmath_data = pd.read_csv("indefinites/data/raw/languages_real_40_updated.csv")
    # restrict to person
    haspelmath_data = haspelmath_data[haspelmath_data["PERSON"] == 1]
    # convert 6 -> 0 so that binary ops work properly
    feature_columns = [
        column
        for column in haspelmath_data.columns
        if column not in {"LANG", "ITEM", "DETER", "PERSON"}
    ]
    haspelmath_data[feature_columns] = haspelmath_data[feature_columns].applymap(
        lambda num: int(num != 6)
    )
    output_data = pd.DataFrame()
    output_data["language"] = haspelmath_data["LANG"].str.lower()
    output_data["expression"] = haspelmath_data["ITEM"]
    output_data["specific-known"] = haspelmath_data["spec.know"]
    output_data["specific-unknown"] = haspelmath_data["spec.unkn"]
    output_data["nonspecific"] = haspelmath_data["irr.nonsp"]
    output_data["freechoice"] = haspelmath_data["free.ch"]
    output_data["negative-indefinite"] = haspelmath_data["neg.frag"]
    output_data["npi"] = (
        (
            haspelmath_data["compar"]
            & (
                haspelmath_data["question"]
                | haspelmath_data["indir.neg"]
                | ~haspelmath_data["free.ch"]
            )
        )
        | (
            ~haspelmath_data["irr.nonsp"]
            & (haspelmath_data["question"] | haspelmath_data["indir.neg"])
        )
        | (
            haspelmath_data["dir.neg"]
            & ~haspelmath_data["neg.frag"]
            & ~haspelmath_data["irr.nonsp"]
        )
        | (
            haspelmath_data["condit"]
            & ~haspelmath_data["irr.nonsp"]
            & ~haspelmath_data["free.ch"]
        )
    )
    output_data = output_data.astype(
        {
            flavor: bool
            for flavor in (
                "specific-known",
                "specific-unknown",
                "nonspecific",
                "freechoice",
                "negative-indefinite",
                "npi",
            )
        }
    )
    output_data.to_csv("indefinites/data/natural_language_indefinites.csv", index=False)
