import os
import sys

import wget
from zipfile import ZipFile
import pandas as pd

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
    # read in xlsx file
    df = pd.read_excel(path, sheet_name="TOTALS BY COUNTRY", skiprows=9)
    # drop columns IPCC_annex, Country_code_A3
    df.drop(columns=["IPCC_annex", "Country_code_A3", "Substance"], inplace=True)
    df.to_csv(path, index=False)

    df = df.dropna()
    return df

def save_data(df, path):
    df.to_csv(path, sep=',', encoding='utf-8')

def normalize_data(df):
    return df.div(df.sum(axis=1), axis=0)