import os

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from data_preprocess import download_data, cleanup_data, save_data, normalize_data, calculate_sum_emissions, \
    get_top_emitters
import warnings

from plot_service.plotting import plot_heatmap, plot_pairplot, plot_top_emitters, plot_historgam

co2_data_dict = "data/co2_emissions_raw.zip"
ch4_data_dict = "data/ch4_emissions_raw.zip"

co2_unzip_path = "emissions_raw/CO2_2000_2021.xlsx"
ch4_unzip_path = "emissions_raw/CH4_2000_2021.xlsx"

co2_data_url = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EDGAR/datasets/v70_FT2021_GHG/v70_FT2021_CO2_m_2000_2021.zip"
ch4_data_url = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EDGAR/datasets/v70_FT2021_GHG/v70_FT2021_CH4_m_2000_2021.zip"
# ignore further warnings
warnings.filterwarnings("ignore")

# create data directory
os.makedirs("emissions_raw", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("clean", exist_ok=True)
os.makedirs("clean/images/", exist_ok=True)


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



# # group by index
# co2_grouped = co2_df.groupby("C_group_IM24_sh").sum()
# co2_grouped_contry = co2_df.groupby("Name").sum()
#
# ch4_grouped = ch4_df.groupby("C_group_IM24_sh").sum()
# ch4_grouped_contry = ch4_df.groupby("Name").sum()
#
# save_data(co2_grouped, "clean/co2_grouped.csv")
# save_data(co2_grouped_contry, "clean/co2_grouped_country.csv")
#
# save_data(ch4_grouped, "clean/ch4_grouped.csv")
# save_data(ch4_grouped_contry, "clean/ch4_grouped_country.csv")
#
# # normalize data
# co2_grouped_norm = normalize_data(co2_grouped)
# co2_grouped_norm_contry = normalize_data(co2_grouped_contry)
#
# ch4_grouped_norm = normalize_data(ch4_grouped)
# ch4_grouped_norm_contry = normalize_data(ch4_grouped_contry)
#
# sum_ch4 = calculate_sum_emissions(ch4_grouped)
# sum_co2 = calculate_sum_emissions(co2_grouped)
#
# # combine dataframes to one
# sum_df = pd.concat([sum_ch4, sum_co2], axis=1)
# sum_df.columns = ["CH4", "CO2"]
#
#
# top_5_co2 = get_top_emitters(co2_grouped_contry, 5)
# top_5_ch4 = get_top_emitters(ch4_grouped_contry, 5)
# plot_top_emitters(top_5_co2, "clean/images/top_5_co2.png")
# plot_top_emitters(top_5_ch4, "clean/images/top_5_ch4.png")

# # plot grouped data as heatmap
# plot_heatmap(co2_grouped_norm, "clean/images/co2_grouped_norm.png")
# plot_heatmap(co2_grouped_norm_contry, "clean/images/co2_grouped_norm_contry.png")
#
# plot_heatmap(ch4_grouped_norm, "clean/images/ch4_grouped_norm.png")
# plot_heatmap(ch4_grouped_norm_contry, "clean/images/ch4_grouped_norm_contry.png")

