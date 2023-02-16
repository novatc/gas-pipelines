import pandas as pd
from scipy.stats import pearsonr

from data_preprocess import calculate_sum_for_year


def population_and_co2(country_name: str, country_df: pd.DataFrame, population_df: pd.DataFrame):
    country = country_df.copy()

    country = calculate_sum_for_year(country)

    population = population_df[population_df.columns[population_df.columns == country_name][0]]
    country = country.merge(population, left_index=True, right_index=True, how="outer")

    corr, _ = pearsonr(country["co2"], country[country_name])
    return country


def population_and_ch4(country_name: str, country_df: pd.DataFrame, population_df: pd.DataFrame):
    country = country_df.copy()

    country = calculate_sum_for_year(country)

    population = population_df[population_df.columns[population_df.columns == country_name][0]]
    country = country.merge(population, left_index=True, right_index=True, how="outer")

    corr, _ = pearsonr(country["ch4"], country[country_name])
    return country
