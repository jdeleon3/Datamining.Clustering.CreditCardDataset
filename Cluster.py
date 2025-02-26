from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
from Visualizer import Visualizer
from sklearn.preprocessing import StandardScaler

class Cluster:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.visualizer = Visualizer(df)

    def elbow_curve(self, max_clusters: int):
        inertia_values = []
        for i in range(1, max_clusters):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(self.df)
            inertia_values.append(kmeans.inertia_)
        self.visualizer.plot_elbow_curve(inertia_values)

    def silhouette_score(self, max_clusters: int):
        silhouette_values = []
        for i in range(2, max_clusters):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(self.df)
            silhouette_values.append(silhouette_score(self.df, kmeans.labels_))
        self.visualizer.plot_silhouette_score(silhouette_values)

    def cluster(self, n_clusters: int):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(self.df)
        self.df['cluster'] = kmeans.labels_

        centroids = kmeans.cluster_centers_
        centroids = pd.DataFrame(centroids, columns=self.df.columns.difference(['cluster']))
        centroids.to_csv('./data/centroids.csv')
        print("Centroids: ")
        print(self.df.columns)
        print(centroids)        

        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(self.df)
        self.visualizer.plot_clusters(pca_data, kmeans.labels_)                
        db_index = davies_bouldin_score(self.df, kmeans.labels_)
        print(f'Davies-Bouldin Index: {db_index}')
        self.df.to_csv('./data/clustered_data.csv', index=False)

if __name__ == '__main__':
    df = pd.read_csv('./data/CC GENERAL.csv')
    df.drop(columns=['CUST_ID'], inplace=True)
    df.fillna(df.mean(), inplace=True)
    df.drop_duplicates(inplace=True)
    cluster = Cluster(df)
    cluster.elbow_curve(10)
    cluster.silhouette_score(10)
    cluster.cluster(3)