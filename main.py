import argparse

from argparse import ArgumentParser
from correlation import person_correlation
from data_preprocess import get_top_emitters, merge_dataframes, get_countries_from_data, download
from forcasting import linear_regression, arma_co2

from plot_service.plotting import plot_heatmap, plot_correlation, \
    visualize

parser = ArgumentParser()
parser.add_argument("-c", "--countries", nargs="+", help="List of countries to plot",
                    default=["Germany", "France", "Italy", "Spain", "United Kingdom"])
parser.add_argument("--correlation", default=False, help="Calculate correlation", action=argparse.BooleanOptionalAction)
parser.add_argument("-v", "--visualize", default=False,
                    help="Visualize data", action=argparse.BooleanOptionalAction)

args = parser.parse_args()

co2_df, ch4_df= download()


combined_emissions = merge_dataframes(co2_df, ch4_df)
co2_top_emitters = get_top_emitters(co2_df, 5)
ch4_top_emitters = get_top_emitters(ch4_df, 5)
selected_countries = get_countries_from_data(combined_emissions, args.countries)

linear_regression(selected_countries[2])
arma_co2(selected_countries[0])

# for country in selected_countries:
#     #if args.visualize:
#         # plot_correlation(selected_countries[country], f"clean/images/correlation/{country}.jpg")
#     person_correlation(country)
#
# if args.visualize:
#     visualize(selected_countries)
#     plot_heatmap(combined_emissions, "clean/images/heatmap/")

