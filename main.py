import os

from data_preprocess import download_data, cleanup_data, save_data, normalize_data
import warnings

from plot_service.plotting import plot_heatmap

data_dict = "data/emissions_raw.zip"
unzip_path = "emissions_raw/CO2_2000_2021.xlsx"
co2_data_url = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EDGAR/datasets/v70_FT2021_GHG/v70_FT2021_CO2_m_2000_2021.zip"
ch4_data_url = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EDGAR/datasets/v70_FT2021_GHG/v70_FT2021_CH4_m_2000_2021.zip"
# ignore further warnings
warnings.filterwarnings("ignore")

# create data directory
os.makedirs("emissions_raw", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("clean", exist_ok=True)

month_column = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

download_data(co2_data_url, data_dict)
co2_df = cleanup_data(unzip_path)


co2_df.set_index("Year", inplace=True)
co2_df.sort_index(inplace=True)

# group by index
co2_grouped = co2_df.groupby("C_group_IM24_sh").sum()
co2_grouped_contry = co2_df.groupby("Name").sum()

save_data(co2_grouped, "clean/co2_grouped.csv")
save_data(co2_grouped_contry, "clean/co2_grouped_contry.csv")

# normalize data
co2_grouped_norm = normalize_data(co2_grouped)
co2_grouped_norm_contry = normalize_data(co2_grouped_contry)


# plot grouped data as heatmap
plot_heatmap(co2_grouped_norm, "clean/co2_grouped_norm.png")
plot_heatmap(co2_grouped_norm_contry, "clean/co2_grouped_norm_contry.png")


