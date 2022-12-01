import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_heatmap(df, path):
    # drop columns that end with "_ch4"
    df_co2 = df[[col for col in df.columns if not col.endswith("_ch4")]]
    df_ch4 = df[[col for col in df.columns if not col.endswith("_co2")]]
    plt.figure(figsize=(20, 10))

    co2_plot = sns.heatmap(df_co2)
    co2_plot.set_title(f"CO2 emissions")
    co2_plot.figure.savefig(path + "heatmap_co2.jpg")

    ch4_plot = sns.heatmap(df_ch4)
    ch4_plot.set_title(f"CH4 emissions")
    ch4_plot.figure.savefig(path + "heatmap_ch4.jpg")


def plot_top_emitters(df_top_emitters, path):
    # plot top emitters, each row a line
    df_top_emitters.plot(figsize=(20, 10))
    plt.title("Top 5 CO2 emitters")
    plt.savefig(path)


def plot_correlation(df, path):
    sns.lmplot(x="co2", y="ch4", data=df, fit_reg=True, height=6, aspect=1.5)
    plt.title(f"Correlation between CO2 and CH4 emissions for {df.index.name}")
    plt.xlabel("CO2 emissions")
    plt.ylabel("CH4 emissions")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(path)

def plot_correlation_population(df, path):
    sns.lmplot(x="co2", y=df.index.name, data=df, fit_reg=True, height=6, aspect=1.5)
    plt.title(f"Correlation between CO2 and population for {df.index.name}")
    plt.xlabel("CO2 emissions")
    plt.ylabel("Population")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()

def plot_arma(name, train, test, y_pred_out, path, methode):
    plt.plot(train, color='black', label='Train')
    plt.plot(test, color='red', label='Test')
    plt.plot(y_pred_out, color='green', label='Predictions')
    plt.xlabel('Time')
    plt.ylabel('CO2 Emissions')
    plt.legend()
    plt.title(f'Forecasting for {name} using {methode}')

    plt.xticks(train.index[::20])
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()


def visualize(df_list: list):
    # plot heatmap
    for country in df_list:
        plot_correlation(country, f"clean/images/correlation/{country.index.name}.jpg")
