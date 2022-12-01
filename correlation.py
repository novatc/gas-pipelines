import pandas as pd
from scipy.stats import pearsonr

from data_preprocess import calculate_sum_for_year
from plot_service.plotting import plot_correlation_population


def person_correlation(df: pd.DataFrame):
    corr, _ = pearsonr(df["co2"], df["ch4"])
    print(f'Pearsons correlation for {df.index.name}: %.3f' % corr)

def person_population_emissions(county_df, population_df):
    country = county_df.copy()
    population = population_df.copy()

    country = calculate_sum_for_year(country)

    country.drop(columns='ch4', inplace=True)

    population = population_df[population_df.columns[population_df.columns == country.index.name][0]]
    country = country.merge(population, left_index=True, right_index=True, how="outer")

    corr, _ = pearsonr(country["co2"], country[country.index.name])
    print(f'Pearsons correlation between CO2 and population for {country.index.name}: %.3f' % corr)
    plot_correlation_population(country, path="clean/images/correlation/population" + country.index.name + ".png")



