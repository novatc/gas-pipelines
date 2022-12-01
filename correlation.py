import pandas as pd
from scipy.stats import pearsonr

def person_correlation(df: pd.DataFrame):
    corr, _ = pearsonr(df["co2"], df["ch4"])
    print(f'Pearsons correlation for {df.index.name}: %.3f' % corr)