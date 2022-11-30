import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_heatmap(df, path):
    plt.figure(figsize=(20, 10))
    sns.heatmap(df, cmap="YlGnBu")
    plt.savefig(path)

def plot_pairplot(df, path):
    sns.pairplot(df)
    plt.title("Relationship between CO2 and CH4 emissions")
    plt.savefig(path)


def plot_top_emitters( df_top_emitters, path):
    # plot top emitters, each row a line
    df_top_emitters.plot(figsize=(20, 10))
    plt.title("Top 5 CO2 emitters")
    plt.savefig(path)



def plot_historgam(df: pd.DataFrame, countries: list, path):
    sns.histplot(data=df[countries], palette="tab10", kde=True)
    plt.title(f"CO2 emissions Histogram for {', '.join(countries)}")
    plt.savefig(path)

def plot_correlation(df, path):
    sns.lmplot(x="co2", y="ch4", data=df, fit_reg=True)
    plt.title(f"Correlation between CO2 and CH4 emissions")
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(path)
