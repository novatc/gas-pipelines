import os
import sys

import matplotlib.pyplot as plt
import wget
from zipfile import ZipFile
import pandas as pd

def make_directories():
    os.makedirs("emissions_raw", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("clean", exist_ok=True)
    os.makedirs("clean/combined", exist_ok=True)
    os.makedirs("clean/images/", exist_ok=True)
    os.makedirs("clean/images/heatmap", exist_ok=True)
    os.makedirs("clean/images/pairplot", exist_ok=True)
    os.makedirs("clean/images/correlation", exist_ok=True)

def check_for_local_data(path: str):
    # check if data is already downloaded
    if not os.path.exists(path):
        # download data
        return False
    else:
        return True


def download_data(url: str, path: str):
    if not check_for_local_data(path):
        print("Downloading data...")
        wget.download(url, out=path)
        print("Download complete!")
    else:
        print("Data already downloaded!")
    with ZipFile(path, 'r') as zipObj:
        zipObj.extractall("emissions_raw")


def cleanup_data(path: str):
    month_column = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # read in xlsx file
    df = pd.read_excel(path, sheet_name="TOTALS BY COUNTRY", skiprows=9)
    # drop columns IPCC_annex, Country_code_A3
    df.drop(columns=["IPCC_annex", "Country_code_A3", "Substance"], inplace=True)

    df = df.dropna()
    df.set_index('Year', inplace=True)
    df.sort_index(inplace=True)
    df = df.pivot_table(index="Name", columns=["Year"], values=month_column)

    df.columns = [" ".join([str(x), str(y)]) for x, y in df.columns]
    df = df.transpose()
    # index to datetime with german format
    df.index = pd.to_datetime(df.index, format="%b %Y")
    df.sort_index(inplace=True)
    # get month name and year as index
    df.index = df.index.strftime("%b %Y")

    return df


def calculate_sum_emissions(df):
    # sum up all the rows and save in new row
    new_df = df.sum()
    return new_df


def save_data(df, path):
    # index as column
    # save data as csv
    df.to_csv(path, index=True, header=True, sep=",")


def normalize_data(df):
    return df.div(df.sum(axis=1), axis=0)


def get_top_emitters(df, n):
    # sum up all rows and save in new row
    new_df = df.sum()
    # return df but only with the top emitters
    return df[new_df.nlargest(n).index]

def combine_emissions(df):
    list_of_countries = {}
    for country in df.columns:
        if country.endswith("_co2"):
            country_name = country.split("_co2")[0]
            # check if country is in ch4_top_emitters
            if country_name + "_ch4" in df.columns:
                country_co2 = df[country]
                country_ch4 = df[country_name + "_ch4"]
                country_df = country_co2.to_frame().merge(country_ch4.to_frame(), left_index=True, right_index=True,
                                                          how="outer")
                # rename first column to date
                country_df.columns = ["co2", "ch4"]
                list_of_countries[country_name] = country_df
                save_data(country_df, "clean/combined/" + country_name + ".csv")

    return list_of_countries
