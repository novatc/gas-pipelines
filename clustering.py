import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# read csv and turn every column in a list of date and column value
df = pd.read_csv("clean/co2.csv", sep=",", index_col=0)

df_t = df.transpose()
df_t.dropna(inplace=True)

# Scale the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_t)

# Perform clustering
kmeans = KMeans(n_clusters=5, n_init=10, max_iter=300, random_state=0)
kmeans.fit(df_scaled)

# Reduce the dimensionality of the data
pca = PCA(n_components=2)
pca.fit(df_scaled)
X_pca = pca.transform(df_scaled)

