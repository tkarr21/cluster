from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import silhouette_score, homogeneity_score


digits = load_digits()
data = digits.data

k = 9
kmeans = KMeans(n_clusters=k, random_state=5, n_init=30,
                max_iter=300, tol=0.0001).fit(data)
print('\nkmeans inertia')
print(kmeans.inertia_)

preds = kmeans.fit_predict(data)
centers = kmeans.cluster_centers_

score = silhouette_score(data, preds, metric='euclidean')
print("For n_clusters = {}, silhouette score is {})".format(k, score))

print(homogeneity_score(digits.target, preds))


clustering = AgglomerativeClustering(
    n_clusters=k, affinity='euclidean', linkage='ward').fit(data)

centroids = []
err = []
for i in range(k):
    
    cluster = np.argwhere(clustering.labels_ == i)
    centro = np.mean(data[cluster, :], axis=0)
    centroids.append(centro)
    err.append(np.sum((data[cluster,:] - centro)**2))

print('\nAgglomerative err')
print(sum(err))

preds = clustering.fit_predict(data)
score = silhouette_score(data, preds, metric='euclidean')
print("For k = {}, silhouette score is {})".format(k, score))
print(homogeneity_score(digits.target, preds))


