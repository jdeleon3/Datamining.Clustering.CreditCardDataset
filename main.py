from cluster import Cluster
from datahandler import DataHandler
import os
import glob

# Delete all images in the images folder
files = glob.glob('./images/*')
for f in files:
    os.remove(f)

# Delete the clustered data file
if os.path.exists('./data/clustered_data.csv'):
    os.remove('./data/clustered_data.csv')
    os.remove('./data/centroids.csv')

# Load the data, clean it, standardize it, and cluster it
filename = './data/CC GENERAL.csv'
dh = DataHandler(filename)
dh.print_missing_values()
dh.print_data_info()
dh.clean_data()
dh.inspect_data()
dh.standardize_data()
cluster = Cluster(dh.get_data())
cluster.elbow_curve(10)
cluster.silhouette_score(10)
cluster.cluster(3)
