import os

import pandas as pd
from matplotlib import pyplot as plt
from data_preprocess import download_data, cleanup_data, save_data, normalize_data, calculate_sum_emissions, \
    get_top_emitters, combine_emissions, make_directories
import warnings

from plot_service.plotting import plot_heatmap, plot_pairplot, plot_top_emitters, plot_historgam, plot_correlation

co2_data_dict = "data/co2_emissions_raw.zip"
ch4_data_dict = "data/ch4_emissions_raw.zip"

co2_unzip_path = "emissions_raw/CO2_2000_2021.xlsx"
ch4_unzip_path = "emissions_raw/CH4_2000_2021.xlsx"

co2_data_url = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EDGAR/datasets/v70_FT2021_GHG/v70_FT2021_CO2_m_2000_2021.zip"
ch4_data_url = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EDGAR/datasets/v70_FT2021_GHG/v70_FT2021_CH4_m_2000_2021.zip"
# ignore further warnings
warnings.filterwarnings("ignore")

# create data directory
make_directories()


download_data(co2_data_url, co2_data_dict)
download_data(ch4_data_url, ch4_data_dict)

co2_df = cleanup_data(co2_unzip_path)
ch4_df = cleanup_data(ch4_unzip_path)


save_data(co2_df, "clean/co2_emissions.csv")
save_data(ch4_df, "clean/ch4_emissions.csv")

# pick two countries to compare
countries = ["Germany", "China"]

plot_historgam(co2_df, countries, "clean/images/co2_emissions_histogram.png")
plot_historgam(ch4_df, countries, "clean/images/ch4_emissions_histogram.png")

co2_top_emitters = get_top_emitters(co2_df, 5)
ch4_top_emitters = get_top_emitters(ch4_df, 5)

plot_top_emitters(co2_top_emitters, "clean/images/co2_top_emitters.png")
plot_top_emitters(ch4_top_emitters, "clean/images/ch4_top_emitters.png")

grouped_c02 = co2_df.groupby(co2_df.index).sum()
grouped_ch4 = ch4_df.groupby(ch4_df.index).sum()

plot_heatmap(grouped_c02, "clean/images/grouped_c02_heatmap.png")
plot_heatmap(grouped_ch4, "clean/images/ch4_heatmap.png")

plot_heatmap(co2_top_emitters, "clean/images/heatmap/co2_top_emitters_heatmap.png")
plot_heatmap(ch4_top_emitters, "clean/images/heatmap/ch4_top_emitters_heatmap.png")

# add string to column names in dataframes co2_top_emitters and ch4_top_emitters to make them unique
co2_top_emitters.columns = [str(col) + '_co2' for col in co2_top_emitters.columns]
ch4_top_emitters.columns = [str(col) + '_ch4' for col in ch4_top_emitters.columns]

top_emitters = co2_top_emitters.merge(ch4_top_emitters, left_index=True, right_index=True, how="outer")
save_data(top_emitters, "clean/top_emitters.csv")

# create new df for each country that is contained in top_emitters with the ending _co2 and _ch4
# and merge them together and save as csv

dict_of_countries= combine_emissions(top_emitters)

for country in dict_of_countries:
    plot_correlation(dict_of_countries[country], f"clean/images/correlation/{country}.jpg")








