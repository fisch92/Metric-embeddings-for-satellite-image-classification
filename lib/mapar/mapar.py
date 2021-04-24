import random
import numpy as np
import scipy as sc
from sklearn.neighbors import BallTree
from sklearn.metrics import adjusted_mutual_info_score
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error, max_error, coverage_error, label_ranking_average_precision_score, label_ranking_loss, silhouette_samples
from sklearn.neighbors import DistanceMetric
from scipy.special import kl_div
from sklearn.preprocessing import RobustScaler
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage


class Mapar(object):

    '''
        Musgrave, Kevin, S. Belongie und S.-N. Lim (2020a). A metric learning
        reality check . arXiv preprint arXiv:2003.08505.
    '''

    def kld(x,y):
        return kl_div(x,y).sum()#np.sum(x * np.log((x+1e-11)/(y+1e-11)))
    
    def jsd(x,y):
        m = (x+y)*0.5
        return np.where(np.abs(x-y).sum(axis=1) > 0, 0.5*(kl_div(x,m)+kl_div(y,m)).mean(), np.zeros(x.shape[0]))
    
    def mapar(X, y, k=10):
        
        x_tree = BallTree(X, leaf_size=2, metric=DistanceMetric.get_metric("l2"))
        xdist, xind = x_tree.query(X, k=k+1, sort_results=True)
        maxy = y.argmax(axis=1)
        maxy = np.expand_dims(maxy, axis=1)
        
        #maxy = np.repeat(maxy, maxy.shape[0], axis=1)
        R = k
        r = 0
        for ck in range(0, k):
            r += np.where(maxy[xind[:, 1:]][:,ck,0] == maxy[:,0], np.ones(maxy[xind[:, 1:]][:,ck,0].shape), np.zeros(maxy[xind[:, 1:]][:,ck,0].shape))
            
        
        P = r/R
        P = np.expand_dims(P, axis=1)
        P = np.repeat(P, R, axis=1)
        #print(r.shape, P.shape, xind.shape, maxy[xind[:, 1:]].shape)
        mapar = np.sum(P[:, 1:], axis=1)/R
        #print(P[:,0]-mapar)
        return mapar.mean()

    '''
        Extension with the key idea of gSil.

        Rawashdeh, Mohammad und A. Ralescu (2012). Center-wise intra-inter
        silhouettes. In: International Conference on Scalable Uncertainty Management, S.
        406–419. Springer.
    '''

    def mapar_ml(X, y, k=10):
        y=y+1e-8
        #y=y/np.expand_dims(y.max(axis=1), axis=1)
        x_tree = BallTree(X, leaf_size=2, metric=DistanceMetric.get_metric("l2"))
        xdist, xind = x_tree.query(X, k=k+1, sort_results=True)
        
        R = k
        r = 0
        for ck in range(0, k):
            intra = []
            inter = []
            for cluster in range(0, y.shape[1]):
                intra.append(np.min([y[xind[:, 1:]][:,ck, cluster],y[:, cluster]], axis=0))
                for compcluster in range(cluster+1, y.shape[1]):
                    inter1 = np.min([y[xind[:, 1:]][:,ck, cluster],y[:, compcluster]], axis=0)
                    inter2 = np.min([y[xind[:, 1:]][:,ck, compcluster],y[:, cluster]], axis=0)
                    inter.append(np.max([inter1, inter2], axis=0))
            inter = np.mean(inter, axis=0)
            intra = np.mean(intra, axis=0)
            #print(inter)
            r += np.max([intra-inter, np.zeros(intra.shape)], axis=0)/np.max([intra, inter], axis=0)
        
            
        
        P = r/R
        P = np.expand_dims(P, axis=1)
        P = np.repeat(P, R, axis=1)
        #print(r.shape, P.shape, xind.shape, maxy[xind[:, 1:]].shape)
        mapar = np.sum(P[:, :], axis=1)/R
        #print(P[:,0]-mapar)
        return mapar.mean()





    def score(X, y, k=10):
        score = Mapar.mapar_ml(X, y, k)
        
        return score
            
        

