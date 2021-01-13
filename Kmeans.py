import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


class KMEANSClustering(BaseEstimator, ClusterMixin):

    def __init__(self, k=3, debug=False):  # add parameters here
        """
        Args:
            k = how many final clusters to have
            debug = if debug is true use the first k instances as the initial centroids otherwise choose random points as the initial centroids.
        """
        self.k = k
        self.debug = debug

    def fit(self, X, y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """

        # randomly grab initial k centroids

        # indices from X
        if self.debug:
            kindex = np.arange(5)
        else:
            kindex = np.random.randint(X.shape[0], size=self.k)
        # actual data instances from x (will become cluster mean)
        self.centroids = X[kindex, :]
        # the clusters in the form of lists of indices of X
        self.clusters = [[] for k in range(self.k)]

        while True:
            next_clustering = [[] for k in range(self.k)]
            for i in range(X.shape[0]):
                assingment = self.assign_cluster(X[i, :])
                next_clustering[assingment].append(i)

            # check for no change in clusters
            if next_clustering == self.clusters:
                # end algorithm on no change
                break
            
            # set new clustering
            self.clusters = next_clustering
            # calc new centroids
            for k in range(self.k):
                self.centroids[k] = np.mean(X[self.clusters[k],:], axis=0)
        
        self.sse(X)

        return self

    def assign_cluster(self, instance):
        # euclidean distance of instance from each centroid and return index of minimum
        return np.argmin(np.linalg.norm(self.centroids - instance, axis=1))
    
    def sse(self, X):
        print('k={} kmeans sse'.format(self.k))
        self.err = []
        for k in range(self.k):

            self.err.append(np.sum((X[self.clusters[k]] - self.centroids[k])**2))

        self.tot_err = sum(self.err)

    def save_clusters(self, filename):
        """
            f = open(filename,"w+") 
            Used for grading.
            write("{:d}\n".format(k))
            write("{:.4f}\n\n".format(total SSE))
            for each cluster and centroid:
                write(np.array2string(centroid,precision=4,separator=","))
                write("\n")
                write("{:d}\n".format(size of cluster))
                write("{:.4f}\n\n".format(SSE of cluster))
            f.close()
        """

        f = open(filename, "w+")
        f.write("{:d}\n".format(self.k))
        f.write("{:.4f}\n\n".format(self.tot_err))
        for i in range(len(self.centroids)):
            f.write(np.array2string(
                self.centroids[i], precision=4, separator=","))
            f.write("\n")
            f.write("{:d}\n".format(len(self.clusters[i])))
            f.write("{:.4f}\n\n".format(self.err[i]))
        f.close()

