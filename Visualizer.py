import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from numpy import ndarray

class Visualizer:

    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def plot_histograms(self):
        figure = plt.figure(figsize=(20, 20))
        self.df.hist(ax=figure.gca())
        #plt.show()
        filename = './images/histograms.png'
        if os.path.exists(filename):
            os.remove(filename)
        figure.savefig(filename)
        figure.clear()
        plt.close()

    def plot_boxplots(self):
        for column in self.df.columns:
            box = sns.boxplot(self.df[column])
            #plt.show()
            filename = f'./images/boxplot_{column}.png'
            if os.path.exists(filename):
                os.remove(filename)
            box.figure.savefig(filename)
            box.figure.clear()
    
    def plot_correlation_heatmap(self):
        corr = self.df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        plt.figure(figsize=(40, 20))
        sns.heatmap(corr,mask=mask, square=True, annot=True)
        #plt.show()
        filename = './images/correlation_heatmap.png'
        if os.path.exists(filename):
            os.remove(filename)
        plt.savefig(filename)
        plt.close()
    
    def plot_elbow_curve(self, inertia_values: list, min_clusters: int = 2):
        plt.figure(figsize=(10, 5))
        plt.plot(range(min_clusters, len(inertia_values) + min_clusters), inertia_values, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Curve')
        #plt.show()
        filename = './images/elbow_curve.png'
        if os.path.exists(filename):
            os.remove(filename)
        plt.savefig(filename)
        plt.close()
    
    def plot_silhouette_score(self, silhouette_values: list):
        plt.figure(figsize=(10, 5))
        plt.plot(range(2, len(silhouette_values) + 2), silhouette_values, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score')
        #plt.show()
        filename = './images/silhouette_score.png'
        if os.path.exists(filename):
            os.remove(filename)
        plt.savefig(filename)
        plt.close()

    def plot_clusters(self, pca_data: pd.DataFrame, cluster_labels: list):
        plt.figure(figsize=(10, 5))
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_labels, cmap='viridis')
        
        plt.title('PCA - Cluster Visualization')
        #plt.show() 
        plt.savefig('./images/cluster_visualization.png')       
        plt.close()        

if __name__ == '__main__':
    dv = Visualizer(pd.read_csv('./data/CC GENERAL.csv'))
    dv.df.drop(columns=['CUST_ID'], inplace=True)
    dv.df.fillna(dv.df.mean(), inplace=True)
    dv.df.drop_duplicates(inplace=True)
    dv.plot_histograms()
    dv.plot_boxplots()
    dv.plot_correlation_heatmap()