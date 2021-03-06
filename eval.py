
from arff import Arff
from Kmeans import KMEANSClustering
from HAC import HACClustering
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# label_count = 0 because clustering is unsupervised.
mat = Arff("./datasets/seismic-bumps_train.arff", label_count=0)

raw_data = mat.data
data = raw_data

### Normalize the data ###
scaler = MinMaxScaler()
scaler.fit(data)
norm_data = scaler.transform(data)

### KMEANS ###
KMEANS = KMEANSClustering(k=5, debug=True)
KMEANS.fit(norm_data)
KMEANS.save_clusters("evaluation_kmeans.txt")

### HAC SINGLE LINK ###
HAC_single = HACClustering(k=5, link_type='single')
HAC_single.fit(norm_data)
HAC_single.save_clusters("evaluation_hac_single.txt")

### HAC COMPLETE LINK ###
HAC_complete = HACClustering(k=5, link_type='complete')
HAC_complete.fit(norm_data)
HAC_complete.save_clusters("evaluation_hac_complete.txt")
