import argparse

GENERATE_IB_BOUND=False #True to generate the IB bound for the specified parameters
USE_RKK = False #Whether to use the RKK metric for complexity
USE_NOGA_ARRAYS = False #True to use the Zaslavsky data for the IB bound, false otherwise
GENERATE_LANG_COLOR_INFO=True #True to generate color information for each language
GENERATE_ADDITIONAL_COLOR_CHIPS = False #If true, will expand color terms based off color distance for both natural and artificial languages
USE_ONE_LANGUAGE = False #If true, just uses the first language in the list of languages
COLOR_CHIP_THRESHOLD = 5 #The minimum number of color chips a language must have to be included in the analysis


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate_ib_bound",
        "-i",
        action="store_true",
        help="Set to generate and display the IB bound for the specified parameters.",
    )
    parser.add_argument(
        "--use_rkk",
        "-r",
        action="store_true",
        help="Set to use the original RKK metric for complexity, false otherwise."
    )
    parser.add_argument(
        "--generate_lang_color_info",
        "-c",
        action="store_true",
        help="Set to generate major color term information for each language."
    )
    parser.add_argument(
        "--number_of_languages",
        type=int,
        default=-1,
        help="Number of languages to analyze from the WCS data. If -1, all languages will be generated."
    )
    parser.add_argument(
        "--color_chip_threshold",
        type=int,
        default=5,
        help="Number of chips required to be considered a major color term.",
    )
    parser.add_argument(
        "--save_languages",
        type=str,
        default="outputs/language_complexity.csv",
        help="Languages of agents and complexity will be saved to this file as CSV.",
    )
    args = parser.parse_args()
    return args