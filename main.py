from data_preprocess import download_data, cleanup_data

data_dict = "data/emissions_raw.zip"
unzip_path = "emissions_raw/CO2_2000_2021.xlsx"
data_url = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EDGAR/datasets/v70_FT2021_GHG/v70_FT2021_CO2_m_2000_2021.zip"

download_data(data_url, data_dict)
co2_df = cleanup_data(unzip_path)
print(co2_df.head())