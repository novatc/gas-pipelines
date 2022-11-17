import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_heatmap(df, path):
    plt.figure(figsize=(20, 10))
    sns.heatmap(df, cmap="YlGnBu")
    plt.savefig(path)

def plot_pairplot(df, path):
    penguins = sns.load_dataset("penguins")
    sns.pairplot(penguins, hue="species")
    plt.show()


def plot_top_emitters( df_top_emitters, path):
    # plot top emitters, each row a line
    df_top_emitters.plot(figsize=(20, 10))
    plt.title("Top 5 CO2 emitters")
    plt.savefig(path)



def plot_historgam(df: pd.DataFrame, countries: list, path):
    sns.histplot(data=df[countries], palette="tab10", kde=True)
    plt.title(f"CO2 emissions Histogram for {', '.join(countries)}")
    plt.savefig(path)
