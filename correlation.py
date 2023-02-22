import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from data_preprocess import calculate_sum_for_year
from plot_service.plotting import plot_correlation_population


def person_correlation(df: pd.DataFrame):
    corr, _ = pearsonr(df["co2"], df["ch4"])
    print(f'Pearsons correlation for {df.index.name}: %.3f' % corr)

def simple_correlation():
    df_co2 = pd.read_csv('clean/co2.csv')
    df_ch4 = pd.read_csv('clean/ch4.csv')

    df_co2 = df_co2.T
    df_ch4 = df_ch4.T

    df_co2.columns = ['co2_' + str(i) for i in range(df_co2.shape[1])]
    df_ch4.columns = ['ch4_' + str(i) for i in range(df_ch4.shape[1])]

    country_name = 'United States'  # replace with the name of the country you want to plot
    df_country = pd.concat([df_co2.loc[country_name], df_ch4.loc[country_name]], axis=1)
    df_country.columns = ['co2', 'ch4']

    plt.scatter(df_country['co2'], df_country['ch4'])
    plt.xlabel('CO2 Emissions')
    plt.ylabel('CH4 Emissions')
    plt.title(f'CO2 vs CH4 Emissions for {country_name}')
    plt.show()


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


