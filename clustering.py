import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# read csv and turn every column in a list of date and column value
df = pd.read_csv("clean/co2.csv", sep=",", index_col=0)
# drop the colum that has no name
df.reset_index(drop=True, inplace=True)
print(df.head())
df = df.T

# normalize data for each row
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# clustering preparation

# clustering
kmeans = KMeans(n_clusters=7, random_state=0).fit(df)
df["cluster"] = kmeans.labels_

# drop the first and the last row
df = df.iloc[1:-1, :]
df.to_csv("clean/co2_clustered.csv")