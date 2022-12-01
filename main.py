import argparse

from argparse import ArgumentParser
from correlation import person_correlation, person_population_emissions
from data_preprocess import get_top_emitters, merge_dataframes, get_countries_from_data, download, get_population_data
from forcasting import linear_regression, ARMA_co2, SARIMA_co2, ARIMA_co2

from plot_service.plotting import plot_heatmap, plot_correlation, \
    visualize

parser = ArgumentParser()
parser.add_argument("-c", "--countries", nargs="+", help="List of countries to plot",
                    default=["Germany", "France", "United Kingdom", "Italy", "Spain", "Poland", "Netherlands", "Belgium",])
parser.add_argument("--correlation", default=False, help="Calculate correlation", action=argparse.BooleanOptionalAction)
parser.add_argument("-v", "--visualize", default=True,
                    help="Visualize data", action=argparse.BooleanOptionalAction)
parser.add_argument("--forcast", default=0, help="Position in the list of given countries", type=int)

args = parser.parse_args()

co2_df, ch4_df= download()
population_df = get_population_data()


combined_emissions = merge_dataframes(co2_df, ch4_df)
co2_top_emitters = get_top_emitters(co2_df, 5)
ch4_top_emitters = get_top_emitters(ch4_df, 5)
selected_countries = get_countries_from_data(combined_emissions, args.countries)

ARMA_co2(selected_countries[args.forcast])
SARIMA_co2(selected_countries[args.forcast])
ARIMA_co2(selected_countries[args.forcast])

for country in selected_countries:
    if args.visualize:
        plot_correlation(country, f"clean/images/correlation/{country.index.name}.jpg")
    person_correlation(country)
    person_population_emissions(country, population_df)
#
# if args.visualize:
#     visualize(selected_countries)
#     plot_heatmap(combined_emissions, "clean/images/heatmap/")

