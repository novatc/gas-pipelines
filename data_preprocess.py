import os
import sys

import wget
from zipfile import ZipFile
import pandas as pd

def bar_progress(current, total):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()

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
        wget.download(url, out=path, bar=bar_progress)
        print("Download complete!")
    else:
        print("Data already downloaded!")
    with ZipFile(path, 'r') as zipObj:
        zipObj.extractall("emissions_raw")

def cleanup_data(path: str):
    # read in xlsx file
    df = pd.read_excel(path, sheet_name="TOTALS BY COUNTRY", skiprows=10)
    # drop nan values
    df = df.dropna()
    return df


