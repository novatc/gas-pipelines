import os

from data_preprocess import download_data, cleanup_data, save_data, normalize_data
import warnings

from plot_service.plotting import plot_heatmap

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

month_column = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

download_data(co2_data_url, co2_data_dict)
download_data(ch4_data_url, ch4_data_dict)

co2_df = cleanup_data(co2_unzip_path)
ch4_df = cleanup_data(ch4_unzip_path)


co2_df.set_index("Year", inplace=True)
co2_df.sort_index(inplace=True)

ch4_df.set_index("Year", inplace=True)
ch4_df.sort_index(inplace=True)

# group by index
co2_grouped = co2_df.groupby("C_group_IM24_sh").sum()
co2_grouped_contry = co2_df.groupby("Name").sum()

ch4_grouped = ch4_df.groupby("C_group_IM24_sh").sum()
ch4_grouped_contry = ch4_df.groupby("Name").sum()

save_data(co2_grouped, "clean/co2_grouped.csv")
save_data(co2_grouped_contry, "clean/co2_grouped_contry.csv")

save_data(ch4_grouped, "clean/ch4_grouped.csv")
save_data(ch4_grouped_contry, "clean/ch4_grouped_contry.csv")

# normalize data
co2_grouped_norm = normalize_data(co2_grouped)
co2_grouped_norm_contry = normalize_data(co2_grouped_contry)

ch4_grouped_norm = normalize_data(ch4_grouped)
ch4_grouped_norm_contry = normalize_data(ch4_grouped_contry)


# plot grouped data as heatmap
plot_heatmap(co2_grouped_norm, "clean/images/co2_grouped_norm.png")
plot_heatmap(co2_grouped_norm_contry, "clean/images/co2_grouped_norm_contry.png")

plot_heatmap(ch4_grouped_norm, "clean/images/ch4_grouped_norm.png")
plot_heatmap(ch4_grouped_norm_contry, "clean/images/ch4_grouped_norm_contry.png")


