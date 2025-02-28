from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
from visualizer import Visualizer


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
        print("\n\nCentroids: ")
        print(self.df.columns)
        print(centroids)

        
        print("\n\nCluster Distribution: ")
        print(self.df['cluster'].value_counts())

        # Use PCA to reduce the dimensions to 2 for visualization
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(self.df)
        self.visualizer.plot_clusters(pca_data, kmeans.labels_)                
        db_index = davies_bouldin_score(self.df, kmeans.labels_)

        print(f'\n\nDavies-Bouldin Index: {db_index}')
        self.df.to_csv('./data/clustered_data.csv', index=False)
 
    def elbow_curve_with_minimum_itemcount(self, max_clusters: int, min_itemcount: int):
        inertia_values = []
        for i in range(max_clusters, 0, -1):            
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(self.df)
            df_temp = pd.DataFrame(kmeans.labels_)
            if(df_temp[0].value_counts().min() >= min_itemcount):
                inertia_values.append(kmeans.inertia_)
        inertia_values.reverse()
        self.visualizer.plot_elbow_curve(inertia_values, min_clusters=1)
            

if __name__ == '__main__':
    df = pd.read_csv('./data/CC GENERAL.csv')
    df.drop(columns=['CUST_ID'], inplace=True)
    df.fillna(df.mean(), inplace=True)
    df.drop_duplicates(inplace=True)
    cluster = Cluster(df)
    cluster.elbow_curve(10)
    cluster.silhouette_score(10)
    cluster.cluster(3)