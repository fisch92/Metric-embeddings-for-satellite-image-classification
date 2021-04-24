import os
import random
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from mxnet import gluon
from mxnet import autograd
from network.resnext import resnext50_32x4d, resnext18_32x4d
from utils.math import Distances
from mxnet.gluon.loss import L2Loss
from mxnet import lr_scheduler
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from utils.fuzzycmeans import FuzzyCMeans
from sklearn.neighbors import KDTree

from network.abstractNet import AbstractNet


class MagNet(AbstractNet):

    '''
        Rippel, Oren, M. Paluri, P. Dollar und L. Bourdev (2015). Metric lear-
        ning with adaptive density discrimination. arXiv preprint arXiv:1511.05939.
    '''

    def __init__(self, margin=0.5, **kwargs):
        super(MagNet, self).__init__(**kwargs)
        
        
        self.margin = margin

        self.n_clusters = 12
        self.clusterer = KMeans(n_clusters=self.n_clusters, max_iter=1000)
        

    def record_grad(self, magnet_batch, ctx):
        magnet_batch = nd.array(magnet_batch, ctx=ctx)
        
        cluster = None
        classes_mask = None
        classes_idx = None
        counter = 0
        mean_dist = 0

        for batch in magnet_batch:
            emb_class = self.predict(batch)
            tree = KDTree(emb_class.asnumpy(), metric='l2')
            neighbor_dists, _ = tree.query(emb_class.asnumpy(), k=int(emb_class.shape[0]/5))
            idx_neighbor_dists = np.argwhere(neighbor_dists[:, 1:] > 0)
            if len(idx_neighbor_dists) > 0:
                mean_dist += neighbor_dists[:, 1:][idx_neighbor_dists].mean()
            if cluster is None:
                cluster = batch
            else: 
                cluster = nd.concat(cluster, batch, dim=0)

            for sample in batch:
                if classes_mask is None:
                    cclass = nd.zeros(len(magnet_batch), ctx=ctx, dtype='float32')# + nd.abs(nd.random.normal(0,1e-8,len(magnet_batch),ctx=self.ctx))
                    cclass[counter] = 1# - nd.abs(nd.random.normal(0,1e-8,1,ctx=self.ctx))
                    classes_mask = nd.expand_dims(cclass, axis=0)
                else:
                    cclass = nd.zeros(len(magnet_batch), ctx=ctx, dtype='float32')# + nd.abs(nd.random.normal(0,1e-8,len(magnet_batch),ctx=self.ctx))
                    cclass[counter] = 1# - nd.abs(nd.random.normal(0,1e-8,1,ctx=self.ctx))
                    classes_mask = nd.concat(classes_mask, nd.expand_dims(cclass, axis=0), dim=0)
                if classes_idx is None:
                    cclass = nd.full(1, counter,ctx=ctx, dtype='int32')
                    classes_idx = cclass
                else:
                    cclass = nd.full(1, counter,ctx=ctx, dtype='int32')
                    classes_idx = nd.concat(classes_idx, cclass, dim=0)
            counter += 1
        classes_idx = classes_idx.asnumpy()
        mean_dist = mean_dist/len(magnet_batch)
        emb_cluster = self.predict(cluster)
        cluster_idx = self.clusterer.fit_predict(emb_cluster.asnumpy())
            
        class_means = None
        idx_class_cluster = np.full(((np.max(classes_idx)+1)*(np.max(cluster_idx)+1),2), -1)
        counter = 0
        for cclass in range(0, np.max(classes_idx)+1):
            comp_batch_class = np.argwhere(classes_idx==cclass)[:,0]
            
            for ccluster in range(-1, np.max(cluster_idx)+1):
                comp_batch_cluster = np.argwhere(cluster_idx==ccluster)[:,0]
                comp_batch_class_cluster = np.intersect1d(comp_batch_cluster, comp_batch_class)
                if len(comp_batch_class_cluster) > 0:
                    idx_class_cluster[counter][0] = cclass
                    idx_class_cluster[counter][1] = ccluster
                    counter+=1
                
                    mean_class = emb_cluster[comp_batch_class_cluster].mean(axis=0).asnumpy()
                    if class_means is None:
                        class_means = np.expand_dims(mean_class,axis=0)
                    else:
                        class_means = np.concatenate((class_means, np.expand_dims(mean_class,axis=0)), axis=0)
                
        
        tree = KDTree(class_means, metric='l2')
        neighbor_dists, ind = tree.query(emb_cluster.asnumpy(), k=class_means.shape[0])
        
        with autograd.record():
            emb_clusters = self.predict(cluster)
            mean = emb_clusters.mean(axis=0)
            var = nd.square(emb_clusters-mean).sum(axis=0)/(emb_clusters.shape[0]-1.0)
            var_const = -(2.0*var)
            loss = None
            
            for batch in range(0, emb_clusters.shape[0]):
                batch_cluster = cluster_idx[batch]
                if batch_cluster != -1:
                    batch_class = classes_idx[batch]

                    comp_batch_cluster = np.argwhere(cluster_idx==batch_cluster)[:,0]
                    comp_batch_sameclass = np.argwhere(classes_idx==batch_class)[:,0]
                    
                    comp_batch_cluster = np.intersect1d(comp_batch_cluster, comp_batch_sameclass)

                    if len(comp_batch_cluster) > 1:
                        mean_cluster = emb_clusters[comp_batch_cluster].mean(axis=0)
                        var_cluster = nd.square(emb_clusters[batch] - mean_cluster).mean(axis=0)
                        
                        var_other_class_all = None
                        #print(ind[batch_cluster])
                        counter = 0
                        for cclasscluster in ind[batch]:
                            cclass = idx_class_cluster[cclasscluster][0]
                            ccluster = idx_class_cluster[cclasscluster][1]
                            
                            comp_all_cluster = np.argwhere(cluster_idx==ccluster)[:,0]
                            comp_batch_class = np.argwhere(classes_idx==cclass)[:,0]
                            comp_batch_otherclass = np.argwhere(classes_idx!=batch_class)[:,0]
                            comp_batch_otherclass_cluster = np.intersect1d(comp_all_cluster, comp_batch_class)
                            comp_batch_otherclass_cluster = np.intersect1d(comp_batch_otherclass_cluster, comp_batch_otherclass)
                                
                            if len(comp_batch_otherclass_cluster) > 1 and counter < 5:
                                counter += 1
                                mean_other_class = emb_clusters[comp_batch_otherclass_cluster].mean(axis=0)
                                var_other_class = nd.square(emb_clusters[batch] - mean_other_class).mean(axis=0)
                                
                                if var_other_class_all is None:
                                    var_other_class_all = nd.expand_dims(var_other_class,axis=0)
                                else:
                                    var_other_class_all = nd.concat(var_other_class_all, nd.expand_dims(var_other_class,axis=0), dim=0)


                        if not var_other_class_all is None and counter>2:
                            intra = nd.exp(var_cluster/var_const-self.margin)
                            #print(var_other_class.shape)
                            inter = nd.exp(var_other_class_all/var_const).sum(axis=0)
                            batch_loss = nd.relu(-nd.log((intra+1e-8)/(inter+1e-8)))
                            if loss is None:
                                loss = batch_loss
                            else:
                                loss = nd.concat(loss, batch_loss, dim=0)

                        
            if loss is None:
                loss = nd.zeros(1,ctx=ctx)
                           
        loss.backward()

        return loss.mean().asnumpy()
