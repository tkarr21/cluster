
from arff import Arff
from Kmeans import KMEANSClustering
from HAC import HACClustering
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt



# label_count = 0 because clustering is unsupervised.
mat = Arff("./datasets/iris.arff", label_count=0)

raw_data = mat.data

data = raw_data[:, :-1]
data_w_label = raw_data
kmeanslog = []
haclog = []
klog = []
scaler = MinMaxScaler()
scaler.fit(data)
norm_data = scaler.transform(data)

'''### Normalize the data ###

### KMEANS ###

for k in range(2,8):
    KMEANS = KMEANSClustering(k=k, debug=False)
    KMEANS.fit(norm_data)
    #KMEANS.save_clusters("evaluation_kmeans.txt")
    print(KMEANS.clusters)
    kmeanslog.append(KMEANS.tot_err)
    klog.append(k)

    ### HAC complete LINK ###
    HAC_complete = HACClustering(k=k, link_type='complete')
    HAC_complete.fit(norm_data)
    haclog.append(HAC_complete.tot_err)

ax = plt.subplot(111)
ax.bar([1.8,2.8,3.8,4.8,5.8,6.8], kmeanslog, width=0.2, color='b', align='center', label='kmeans')
ax.bar(klog, haclog, width=0.2, color='r', align='center', label='hac-complete')
plt.xlabel('k values')
plt.ylabel('Sum squared error')
plt.legend()
plt.show()'''




'''### Normalize the data ###
scaler = MinMaxScaler()
scaler.fit(data_w_label)
norm_data = scaler.transform(data_w_label)
for k in range(2, 8):
    KMEANS = KMEANSClustering(k=k, debug=False)
    KMEANS.fit(norm_data)
    #KMEANS.save_clusters("evaluation_kmeans.txt")
    print(KMEANS.clusters)
    kmeanslog.append(KMEANS.tot_err)
    klog.append(k)

    ### HAC complete LINK ###
    HAC_complete = HACClustering(k=k, link_type='complete')
    HAC_complete.fit(norm_data)
    haclog.append(HAC_complete.tot_err)

ax = plt.subplot(111)
ax.bar([1.8,2.8,3.8,4.8,5.8,6.8], kmeanslog, width=0.2, color='b', align='center', label='kmeans')
ax.bar(klog, haclog, width=0.2, color='r', align='center', label='hac-complete')
plt.xlabel('k values')
plt.ylabel('Sum squared error')
plt.legend()
plt.show()'''


for k in range(5):
    KMEANS = KMEANSClustering(k=4, debug=False)
    KMEANS.fit(norm_data)
    klog.append(k)
    kmeanslog.append(KMEANS.tot_err)
    print('\nkmeans k=4 iter:{}'.format(k))
    print(KMEANS.clusters)
    print(KMEANS.tot_err)
    print(KMEANS.err)

ax = plt.subplot(111)
ax.bar(klog, kmeanslog,
       width=0.2, color='b', align='center', label='kmeans')
plt.xlabel('kmeans k=4, iterations')
plt.ylabel('Sum squared error')
plt.legend()
plt.show()
