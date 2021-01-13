import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

class HACClustering(BaseEstimator,ClusterMixin):

    def __init__(self, k=3, link_type='single', distance_type='euclidean'):  ## add parameters here
        """
        Args:
            k = how many final clusters to have
            link_type = single or complete. when combining two clusters use complete link or single link
        """
        self.link_type = link_type
        self.k = k
    def fit(self,X,y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        # calc distances
        d_mtrx = self.distance(X)

        # cluster
        self.hac(d_mtrx) if self.link_type == 'single' else self.hac(
            d_mtrx, single=False)
            
        # calc sum squared error
        self.sse(X)


        return self
        


    def hac(self, d_mtrx, single=True):

        # make every point its own cluster
        self.clusters = [[x]  for x in range(d_mtrx.shape[0])]

        # run the algorithm
        while len(self.clusters) > self.k:

            # to store distances between clusters
            clust_pairs = np.zeros((len(self.clusters), len(self.clusters)))
            for i in range(len(self.clusters)):
                for j in range(len(self.clusters)):
                    
                    #set unneeded vals to nan
                    if i >= j:
                        clust_pairs[i][j] = np.nan
                    
                    else:
                        clust_pairs[i][j] = np.nanmin(d_mtrx[np.ix_(self.clusters[i], self.clusters[j])]) \
                            if single else np.nanmax(d_mtrx[np.ix_(self.clusters[i], self.clusters[j])])



            # grab the closest clusters
            mergees = divmod(np.nanargmin(clust_pairs), clust_pairs.shape[1])
            #print("mergees: {}".format(mergees))

            self.clusters[mergees[0]].extend(self.clusters[mergees[1]])
            #print(self.clusters[mergees[0]])
            
            # remove merged cluster
            self.clusters.pop(mergees[1])


    def sse(self, X):
        print('k={} hac sse'.format(self.k))
        self.centroids = []
        self.err = []
        for cluster in self.clusters:
            #print("the cluster:")
            #print(cluster)
            if self.link_type == 'complete':
                print(cluster)
            centro = np.mean(X[cluster,:], axis=0)
            self.centroids.append(centro)

            self.err.append(np.sum((X[cluster] - centro)**2))

        self.tot_err = sum(self.err)
            

    def distance(self, X):
    
        ''' calculate distance matrix for all instances in X
        Args:
            X (array-like): 2D numpy array of shape (n,m)

        Returns: 
            d_mtrx (array-like): 2D numpy array of shape (n,n)
                row i has n distances from 
        '''

        d_mtrx = np.zeros((X.shape[0], X.shape[0]))

        for i in range(X.shape[0]):

            dist = np.zeros(X.shape[0])
            for j in range(X.shape[0]):

                if i >= j:
                    dist[j] = np.nan
                else:
                    dist[j] = np.linalg.norm(X[i, :] - X[j, :])
                
                
            d_mtrx[i, :] = dist

        return d_mtrx
    

    def save_clusters(self,filename):
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
            f.write(np.array2string(self.centroids[i], precision=4, separator=","))
            f.write("\n")
            f.write("{:d}\n".format(len(self.clusters[i])))
            f.write("{:.4f}\n\n".format(self.err[i]))
        f.close()





if __name__ == "__main__":

    hac = HACClustering()

    a = np.arange(50).reshape(5, 10)

    dist = hac.distance(a)
    print(dist)     

