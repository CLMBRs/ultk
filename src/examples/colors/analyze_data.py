

#End to end analysis of the WCS color data. 
#Start here to run the full analysis of the WCS color data.
from pandas import DataFrame
import os
from generate_wcs_languages import generate_color_languages
from complexity import generate_color_complexity
from graph_colors import graph_complexity, graph_expression,graph_language_color_distribution
import pandas
import utils

GENERATE_IB_BOUND=False #True to generate the IB bound for the specified parameters
USE_RKK = False #Whether to use the RKK metric for complexity
# USE_NOGA_ARRAYS = False #True to use the Zaslavsky data for the IB bound, false otherwise
GENERATE_LANG_COLOR_INFO=True #True to generate color information for each language
# GENERATE_ADDITIONAL_COLOR_CHIPS = False #If true, will expand color terms based off color distance for both natural and artificial languages
USE_ONE_LANGUAGE = False #If true, just uses the first language in the list of languages
COLOR_CHIP_THRESHOLD = 5 #The minimum number of color chips a language must have to be included in the analysis

def main(args):

    GENERATE_IB_BOUND = args.generate_ib_bound
    USE_RKK = args.use_rkk
    GENERATE_LANG_COLOR_INFO = args.generate_lang_color_info
    NUM_LANGUAGES = args.number_of_languages
    COLOR_CHIP_THRESHOLD = args.color_chip_threshold

    current_dir = os.path.dirname(os.path.realpath(__file__))

    #Generate ULTK language structures from the WCS data
    print("Generating color languages...")
    languages, artificial_languages, meaning_dist = generate_color_languages(num_languages=args.number_of_languages, color_chip_threshold = args.color_chip_threshold)
    #Analyze the generated languages using the RKK+ algorithm
    print("Analyzing complexity/informativity of the color languages...")
    generate_color_complexity(nat_langs=languages, art_langs=artificial_languages, meaning_dist=meaning_dist, use_rkk = args.use_rkk, generate_ib_bound = args.generate_ib_bound)
    print("Graphing color data...")
    #Generate the per-language graphs of color terms
    if(GENERATE_LANG_COLOR_INFO):
        for nat_lang in languages:
            graph_language_color_distribution(nat_lang, current_dir)
        for art_lang in artificial_languages:
            graph_language_color_distribution(art_lang, current_dir)
    complexity_dataframe:DataFrame = pandas.read_csv(f"{current_dir}/outputs/complexity_data.csv")
    graph_complexity(complexity_dataframe, None, current_dir)
    print("Color analysis complete.")

if __name__ == "__main__":
    args = utils.get_args()
    main(args)