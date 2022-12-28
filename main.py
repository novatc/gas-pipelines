import argparse

from argparse import ArgumentParser
from correlation import person_correlation, person_population_emissions
from data_preprocess import get_top_emitters, merge_dataframes, get_countries_from_data, download, get_population_data, \
    save_data
from forcasting_methodes.forcasting import ARMA_co2, SARIMA_co2, ARIMA_co2
from forcasting_methodes.keras_forcasting import keras_forcast
from forcasting_methodes.lr import linear_regression
from forcasting_methodes.pytorch_forcasting import pytorch_forcast

from plot_service.plotting import plot_correlation

parser = ArgumentParser()
parser.add_argument("-c", "--countries", nargs="+", help="List of countries to plot",
                    default=["Germany", "France", "United Kingdom", "Italy", "Spain", "Poland", "Netherlands",
                             "Belgium", ])
parser.add_argument("--correlation", default=False, help="Calculate correlation", action=argparse.BooleanOptionalAction)
parser.add_argument("--forcast", default=0, help="Position in the list of given countries", type=int)
parser.add_argument("--arma", default=True, help="Use arma for forcasting", action=argparse.BooleanOptionalAction)
parser.add_argument("--lr", default=False, help="Linear Regression", action=argparse.BooleanOptionalAction)
parser.add_argument("--keras", default=False, help="Use keras nodel for forcasting", action=argparse.BooleanOptionalAction)
parser.add_argument("--pytroch", default=True, help="Use pytroch model for forcasting", action=argparse.BooleanOptionalAction)

args = parser.parse_args()

co2_df, ch4_df = download()
# save the df
save_data(co2_df, "clean/co2.csv")
save_data(ch4_df, "clean/ch4.csv")



population_df = get_population_data()

combined_emissions = merge_dataframes(co2_df, ch4_df)
co2_top_emitters = get_top_emitters(co2_df, 5)
ch4_top_emitters = get_top_emitters(ch4_df, 5)
selected_countries = get_countries_from_data(combined_emissions, args.countries)

if args.arma:
    ARMA_co2(selected_countries[args.forcast])
    SARIMA_co2(selected_countries[args.forcast])
    ARIMA_co2(selected_countries[args.forcast])

if args.lr:
    print("Linear Regression")
    linear_regression(selected_countries[args.forcast])
if args.keras:
    print("Keras")
    keras_forcast(selected_countries[args.forcast])

if args.pytroch:
    print("Pytroch")
    pytorch_forcast(selected_countries[args.forcast])

if args.correlation:
    for country in selected_countries:
        plot_correlation(country, f"clean/images/correlation/{country.index.name}.jpg")
        person_correlation(country)
        person_population_emissions(country, population_df)
