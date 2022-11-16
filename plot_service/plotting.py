import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(df, path):
    plt.figure(figsize=(20, 10))
    sns.heatmap(df, cmap="YlGnBu")
    plt.savefig(path)