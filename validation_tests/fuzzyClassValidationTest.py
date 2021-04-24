import random
import copy
import numpy as np
import scipy as sc
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from lib.mapar.mapar import Mapar
from scipy.spatial.distance import euclidean
from sklearn.metrics import silhouette_score
from sklearn.metrics import mean_squared_log_error, max_error, coverage_error, label_ranking_average_precision_score, label_ranking_loss, silhouette_samples
from sklearn.feature_selection import mutual_info_regression
from utils.testDistributionGenerator import TestDistributionGenerator
from scipy.special import kl_div

from sklearn.neighbors import BallTree
from sklearn.neighbors import DistanceMetric
from scipy.ndimage.filters import gaussian_filter1d

matplotlib.use('Agg')


class FuzzyClassValidationTest():

    '''
        Plot validation scores for different dynamic data distributions. 
    '''

    def sampleData(config=None):
        if config is None:
            config = {
                'clusters': [{
                    'distribution': 'blob',
                    'samples': 100,
                    'label': 0,
                    'center': np.array([0.5, 0.5]),
                    'cluster_std': 0.5,
                    'random_noise': 0.0
                }, {
                    'distribution': 'blob',
                    'samples': 100,
                    'label': 1,
                    'center': np.array([0.5, 0.5]),
                    'cluster_std': 0.5,
                    'random_noise': 0.0
                }, {
                    'distribution': 'blob',
                    'samples': 100,
                    'label': 2,
                    'center': np.array([0.5, 0.5]),
                    'cluster_std': 0.5,
                    'random_noise': 0.0
                }, {
                    'distribution': 'blob',
                    'samples': 100,
                    'label': 3,
                    'center': np.array([0.5, 0.5]),
                    'cluster_std': 0.5,
                    'random_noise': 0.0
                }],
                'feature': 2,
                'labels': 4,
                'random_noise': 0.5
            }

        samples = None
        labels = None
        for cluster in config['clusters']:
            
            if cluster['distribution'] == 'blob':

                # np.repeat(cluster['center'], int(config['feature']/len(cluster['center'])))
                tcenter = np.zeros(config['feature'])
                tcenter[:len(cluster['center'])] = cluster['center']
                cluster['center'] = tcenter
                cluster_samples, cluster_label = TestDistributionGenerator.createBlob(
                    samples=cluster['samples'], center=cluster['center'][:config['feature']], std=cluster['cluster_std'], dims=config['feature'])
            if cluster['distribution'] == 'circle':
                # np.repeat(cluster['center'], int(config['feature']/len(cluster['center'])))
                tcenter = np.zeros(config['feature'])
                tcenter[:len(cluster['center'])] = cluster['center']
                cluster['center'] = tcenter
                # np.repeat(cluster['center'], int(config['feature']/len(cluster['center'])))
                tr = np.zeros(config['feature'])
                tr[:len(cluster['r'])] = cluster['r']
                cluster['r'] = tr
                cluster_samples, cluster_label = TestDistributionGenerator.createCircle(
                    samples=cluster['samples'], center=cluster['center'][:config['feature']], std=cluster['cluster_std'], r=cluster['r'], dims=config['feature'])
            if cluster['distribution'] == 'line':
                # np.repeat(cluster['center'], int(config['feature']/len(cluster['center'])))
                tstart = np.zeros(config['feature'])
                tstart[:len(cluster['start'])] = cluster['start']
                cluster['start'] = tstart
                tend = np.zeros(config['feature'])
                tend[:len(cluster['end'])] = cluster['end']
                cluster['end'] = tend
                cluster_samples, cluster_label = TestDistributionGenerator.createLine(
                    samples=cluster['samples'], start=cluster['start'][:config['feature']], end=cluster['end'][:config['feature']], std=cluster['cluster_std'], dims=config['feature'])
            # print(cluster_labels.shape)
            cluster_labels = np.zeros((cluster['samples'], config['labels']))
            #print(cluster_labels.shape, counter, cluster_label.shape)
            cluster_labels[:, cluster['label']] = 0.5+cluster_label
            if samples is None:
                samples = cluster_samples
                labels = cluster_labels
            else:
                samples = np.concatenate((samples, cluster_samples), axis=0)
                labels = np.concatenate((labels, cluster_labels), axis=0)

        for cluster in config['clusters']:
            if cluster['distribution'] == 'blob':
                # print(cluster['label'])
                tlabel = np.exp(-np.sum(np.square(samples-cluster['center'][:config['feature']])/(
                    np.square(cluster['cluster_std'])), axis=1))
                tlabel = np.clip(tlabel, 0, 1)
                labels[:, cluster['label']] += tlabel
                #labels[:, cluster['label']] = np.clip(labels[:, cluster['label']], 0, 1)
            if cluster['distribution'] == 'circle':
                #cluster['cluster_std'] = cluster['cluster_std']*np.pi

                dist = (
                    np.abs(samples-cluster['center'][:config['feature']])-cluster['r'])
                tlabel = np.exp(-np.sum(np.square(dist) /
                                        (2.*cluster['cluster_std']**2), axis=1))
                tlabel = np.clip(tlabel, 0, 1)
                labels[:, cluster['label']] += tlabel

            if cluster['distribution'] == 'line':
                #cluster['cluster_std'] = cluster['cluster_std']*np.pi

                dist = np.min(np.abs(np.expand_dims(np.linspace(
                    cluster['start'][:config['feature']], cluster['end'][:config['feature']]), axis=0)-np.expand_dims(samples, axis=1)), axis=1)
                # print(dist.shape)
                tlabel = np.exp(-np.sum(np.square(dist) /
                                        (2.*cluster['cluster_std']**2), axis=1))
                tlabel = np.clip(tlabel, 0, 1)
                labels[:, cluster['label']] += tlabel
                #labels[:, cluster['label']] = np.clip(labels[:, cluster['label']], 0, 1)
            #labels[:, cluster['label']] = (labels[:, cluster['label']]-labels[:, cluster['label']].min())/(labels[:, cluster['label']].max()-labels[:, cluster['label']].min())
            # print(cluster['random_noise'])
            #print(cluster['label'], labels[:, cluster['label']])
            labels[:, cluster['label']] = np.where(np.random.random_sample(
                labels.shape[0]) < cluster['random_noise'], np.random.random_sample(labels.shape[0]), labels[:, cluster['label']])
        labels = labels/np.expand_dims(labels.sum(axis=1), axis=1)
        # print(labels)

        return samples, labels

    def showData(X, y):
        plt.figure()
        idx_max_y = np.argmax(y, axis=1)
        y_unique = np.unique(idx_max_y)
        colors = cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
        for this_y, color in zip(y_unique, colors):
            this_X = X[idx_max_y == this_y]
            plt.scatter(this_X[:, 0], this_X[:, 1],
                        c=color[np.newaxis, :],
                        alpha=0.5, edgecolor='k',
                        label="Class %s" % this_y)

        plt.show()

    def silhouette(X, y):
        y = y+1e-8
        weights = y.sum(axis=1)
        y = y/np.expand_dims(y.sum(axis=1), axis=1)
        out = 0
        x_tree = BallTree(
            X, leaf_size=2, metric=DistanceMetric.get_metric("l2"))
        xdist, xind = x_tree.query(X, k=X.shape[0], sort_results=True)

        a_dists = []
        b_dists = []
        for cluster in range(0, y.shape[1]):
            intra = np.min([np.repeat(y[xind[:, :1]][:, :, cluster], xind[:,
                                                                          1:].shape[1], axis=1), y[xind[:, 1:]][:, :, cluster]], axis=0)
            # print(intra.shape)
            a_dist = 1.0/intra.sum(axis=1) * \
                np.sum(xdist[:, 1:] * intra, axis=1)
            a_dists.append(a_dist)
            for compcluster in range(0, y.shape[1]):
                if cluster != compcluster:
                    inter1 = np.repeat(
                        y[xind[:, :1]][:, :, cluster], xind[:, 1:].shape[1], axis=1)
                    inter2 = y[xind[:, 1:]][:, :, compcluster]

                    inter12 = np.min([inter1, inter2], axis=0)

                    inter3 = np.repeat(
                        y[xind[:, :1]][:, :, compcluster], xind[:, 1:].shape[1], axis=1)
                    inter4 = y[xind[:, 1:]][:, :, cluster]

                    inter34 = np.min([inter3, inter4], axis=0)
                    inter = np.max([inter12, inter34], axis=0)

                    b_dist = 1.0/inter.sum(axis=1) * \
                        np.sum(xdist[:, 1:] * inter, axis=1)

                    b_dists.append(b_dist)

        a_dist = np.min(a_dists, axis=0)
        b_dist = np.min(b_dists, axis=0)
        out = ((b_dist-a_dist)/np.max([b_dist, a_dist], axis=(0)))

        return (out*weights).sum()/weights.sum()

    def kld(x, y):
        return kl_div(x, y).mean()

    def jsd(x, y):
        m = (x+y)*0.5
        return 0.5*(KNNSilhouetteScore.kld(x, m)+KNNSilhouetteScore.kld(y, m))

    def score(X, y):

        silhouette_scores = FuzzyClassValidationTest.silhouette(X, y)
        print('silhouette_scores', silhouette_scores)

        mapa1 = Mapar.score(X, y, k=1)
        print('mapar', mapa1)
        mapa5 = Mapar.score(X, y, k=5)
        print('mapar', mapa5)
        mapa10 = Mapar.score(X, y, k=10)
        print('mapar', mapa10)

        dendro = linkage(y, method='single')
        dists = sc.spatial.distance.pdist(X)
        ccc, cophe_dists = sc.cluster.hierarchy.cophenet(dendro, dists)

        print('ccc', ccc)

        return silhouette_scores, ccc, mapa1, mapa5, mapa10


def unitTest():
    N = 1
    D = 2
    sigma = 3

    config = {
        'clusters': [{
            'distribution': 'line',
            'samples': 25,
            'label': 1,
            'start': np.array([-1.0, 0.0]),
            'end': np.array([-1.0, 2.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'line',
            'samples': 25,
            'label': 1,
            'start': np.array([-1.0, 2.0]),
            'end': np.array([0.0, 2.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'line',
            'samples': 25,
            'label': 1,
            'start': np.array([0.0, 2.0]),
            'end': np.array([0.0, 0.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'line',
            'samples': 25,
            'label': 0,
            'start': np.array([-0.5, 1.0]),
            'end': np.array([-0.5, -1.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'line',
            'samples': 25,
            'label': 0,
            'start': np.array([-0.5, -1.0]),
            'end': np.array([-1.5, -1.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'line',
            'samples': 25,
            'label': 0,
            'start': np.array([-1.5, -1.0]),
            'end': np.array([-1.5, 1.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }],
        'feature': D,
        'labels': 2,
        'random_noise': 0.0
    }

    # for sample in y:
    #   print(sample)

    silhouette_scores = np.zeros(len(np.arange(0, 1, 0.04)))
    cccs = np.zeros(len(np.arange(0, 1, 0.04)))
    mapa1_scores = np.zeros(len(np.arange(0, 1, 0.04)))
    mapa5_scores = np.zeros(len(np.arange(0, 1, 0.04)))
    mapa10_scores = np.zeros(len(np.arange(0, 1, 0.04)))
    for iteration in np.arange(0, N):
        counter = 0
        for noise in np.arange(0.0, 1, 0.04):
            config['clusters'][0]['random_noise'] = noise
            config['clusters'][1]['random_noise'] = noise
            config['clusters'][2]['random_noise'] = noise
            config['clusters'][3]['random_noise'] = noise
            config['clusters'][4]['random_noise'] = noise
            config['clusters'][5]['random_noise'] = noise
            X, y = FuzzyClassValidationTest.sampleData(copy.deepcopy(config))
            silhouette_score, ccc, mapa1, mapa5, mapa10 = FuzzyClassValidationTest.score(
                X, y)
            silhouette_scores[counter] += silhouette_score
            cccs[counter] += ccc
            mapa1_scores[counter] += mapa1
            mapa5_scores[counter] += mapa5
            mapa10_scores[counter] += mapa10
            counter += 1
            # FuzzyClassValidationTest.showData(X,y)

    silhouette_scores = gaussian_filter1d(silhouette_scores/N, sigma=sigma)
    cccs = gaussian_filter1d(cccs/N, sigma=sigma)
    mapa1_scores = gaussian_filter1d(mapa1_scores/N, sigma=sigma)
    mapa5_scores = gaussian_filter1d(mapa5_scores/N, sigma=sigma)
    mapa10_scores = gaussian_filter1d(mapa10_scores/N, sigma=sigma)
    plt.plot(np.arange(0, 1, 0.04), silhouette_scores,
             color='green', label='gSil')
    plt.plot(np.arange(0, 1, 0.04), cccs, color='blue', label='CCC')
    plt.plot(np.arange(0, 1, 0.04), mapa1_scores, color='purple',
             label='MAP@1 Score', linestyle=(0, (2, 5)))
    plt.plot(np.arange(0, 1, 0.04), mapa5_scores, color='turquoise',
             label='MAP@5 Score', linestyle=(2, (2, 5)))
    plt.plot(np.arange(0, 1, 0.04), mapa10_scores, color='orange',
             label='MAP@10 Score', linestyle=(4, (2, 5)))
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.title("label noise")
    plt.xlabel("likelihood")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig('results/noise.png')
    plt.close()
    # plt.show()

    config = {
        'clusters': [{
            'distribution': 'blob',
            'samples': 50,
            'label': 1,
            'center': np.array([0.0, 1.0]),
            'cluster_std': 0.05,
            'random_noise': 0.0
        }, {
            'distribution': 'blob',
            'samples': 50,
            'label': 1,
            'center': np.array([0.0, 1.0]),
            'cluster_std': 0.05,
            'random_noise': 0.0
        }, {
            'distribution': 'blob',
            'samples': 50,
            'label': 0,
            'center': np.array([0.0, 0.0]),
            'cluster_std': 0.05,
            'random_noise': 0.0
        }, {
            'distribution': 'blob',
            'samples': 50,
            'label': 0,
            'center': np.array([0.0, 0.0]),
            'cluster_std': 0.05,
            'random_noise': 0.0
        }],
        'feature': D,
        'labels': 2,
        'random_noise': 0.0
    }

    # for sample in y:
    #   print(sample)

    silhouette_scores = np.zeros(len(np.arange(0, 3, 0.08)))
    cccs = np.zeros(len(np.arange(0, 3, 0.08)))
    mapa1_scores = np.zeros(len(np.arange(0, 3, 0.08)))
    mapa5_scores = np.zeros(len(np.arange(0, 3, 0.08)))
    mapa10_scores = np.zeros(len(np.arange(0, 3, 0.08)))
    for iteration in np.arange(0, N):
        counter = 0
        for center_blobx in np.arange(0, 3, 0.08):
            config['clusters'][0]['center'][0] = center_blobx
            config['clusters'][1]['center'][0] = -center_blobx
            config['clusters'][2]['center'][0] = center_blobx
            config['clusters'][3]['center'][0] = -center_blobx
            X, y = FuzzyClassValidationTest.sampleData(copy.deepcopy(config))
            silhouette_score, ccc, mapa1, mapa5, mapa10 = FuzzyClassValidationTest.score(
                X, y)
            silhouette_scores[counter] += silhouette_score
            cccs[counter] += ccc
            mapa1_scores[counter] += mapa1
            mapa5_scores[counter] += mapa5
            mapa10_scores[counter] += mapa10
            counter += 1
            # FuzzyClassValidationTest.showData(X,y)
    silhouette_scores = gaussian_filter1d(silhouette_scores/N, sigma=sigma)
    cccs = gaussian_filter1d(cccs/N, sigma=sigma)
    mapa1_scores = gaussian_filter1d(mapa1_scores/N, sigma=sigma)
    mapa5_scores = gaussian_filter1d(mapa5_scores/N, sigma=sigma)
    mapa10_scores = gaussian_filter1d(mapa10_scores/N, sigma=sigma)
    plt.plot(np.arange(0, 3, 0.08), silhouette_scores,
             color='green', label='gSil')
    plt.plot(np.arange(0, 3, 0.08), cccs, color='blue', label='CCC')
    plt.plot(np.arange(0, 3, 0.08), mapa1_scores, color='purple',
             label='MAP@1 Score', linestyle=(0, (2, 5)))
    plt.plot(np.arange(0, 3, 0.08), mapa5_scores, color='turquoise',
             label='MAP@5 Score', linestyle=(2, (2, 5)))
    plt.plot(np.arange(0, 3, 0.08), mapa10_scores, color='orange',
             label='MAP@10 Score', linestyle=(4, (2, 5)))
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.title("splitted cluster")
    plt.xlabel("x")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig('results/splitted.png')
    plt.close()

    config = {
        'clusters': [{
            'distribution': 'line',
            'samples': 50,
            'label': 0,
            'start': np.array([-1.0, -1.0]),
            'end': np.array([-1.0, 1.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'line',
            'samples': 50,
            'label': 0,
            'start': np.array([1.0, -1.0]),
            'end': np.array([1.0, 1.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'blob',
            'samples': 50,
            'label': 1,
            'center': np.array([0.0, 0.0]),
            'cluster_std': 0.05,
            'random_noise': 0.0
        }, {
            'distribution': 'blob',
            'samples': 50,
            'label': 1,
            'center': np.array([0.0, 0.0]),
            'cluster_std': 0.05,
            'random_noise': 0.0
        }],
        'feature': D,
        'labels': 2,
        'random_noise': 0.0
    }

    # for sample in y:
    #   print(sample)

    silhouette_scores = np.zeros(len(np.arange(0, 3, 0.08)))
    cccs = np.zeros(len(np.arange(0, 3, 0.08)))
    mapa1_scores = np.zeros(len(np.arange(0, 3, 0.08)))
    mapa5_scores = np.zeros(len(np.arange(0, 3, 0.08)))
    mapa10_scores = np.zeros(len(np.arange(0, 3, 0.08)))
    for iteration in np.arange(0, N):
        counter = 0
        for center_blobx in np.arange(0, 3, 0.08):
            config['clusters'][2]['center'][0] = center_blobx
            config['clusters'][3]['center'][0] = -center_blobx
            X, y = FuzzyClassValidationTest.sampleData(copy.deepcopy(config))
            silhouette_score, ccc, mapa1, mapa5, mapa10 = FuzzyClassValidationTest.score(
                X, y)
            silhouette_scores[counter] += silhouette_score
            cccs[counter] += ccc
            mapa1_scores[counter] += mapa1
            mapa5_scores[counter] += mapa5
            mapa10_scores[counter] += mapa10
            counter += 1
            # FuzzyClassValidationTest.showData(X,y)
    silhouette_scores = gaussian_filter1d(silhouette_scores/N, sigma=sigma)
    cccs = gaussian_filter1d(cccs/N, sigma=sigma)
    mapa1_scores = gaussian_filter1d(mapa1_scores/N, sigma=sigma)
    mapa5_scores = gaussian_filter1d(mapa5_scores/N, sigma=sigma)
    mapa10_scores = gaussian_filter1d(mapa10_scores/N, sigma=sigma)
    plt.plot(np.arange(0, 3, 0.08), silhouette_scores,
             color='green', label='gSil')
    plt.plot(np.arange(0, 3, 0.08), cccs, color='blue', label='CCC')
    plt.plot(np.arange(0, 3, 0.08), mapa1_scores, color='purple',
             label='MAP@1 Score', linestyle=(0, (2, 5)))
    plt.plot(np.arange(0, 3, 0.08), mapa5_scores, color='turquoise',
             label='MAP@5 Score', linestyle=(2, (2, 5)))
    plt.plot(np.arange(0, 3, 0.08), mapa10_scores, color='orange',
             label='MAP@10 Score', linestyle=(4, (2, 5)))
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.title("splitted cluster")
    plt.xlabel("x")
    plt.ylabel("score")
    plt.tight_layout()
    plt.savefig('results/splitted2.png')
    plt.close()

    silhouette_scores = np.zeros(len(np.arange(2.0, 250, 10.0)))
    cccs = np.zeros(len(np.arange(2.0, 250, 10.0)))
    mapa1_scores = np.zeros(len(np.arange(2.0, 250, 10.0)))
    mapa5_scores = np.zeros(len(np.arange(2.0, 250, 10.0)))
    mapa10_scores = np.zeros(len(np.arange(2.0, 250, 10.0)))

    for iteration in np.arange(0, N):
        counter = 0
        config = {
            'clusters': [{
                'distribution': 'blob',
                'samples': 100,
                'label': 0,
                'center': np.array([-1.0, 0.0]),
                'cluster_std': 0.1,
                'random_noise': 0.0
            }, {
                'distribution': 'blob',
                'samples': 100,
                'label': 1,
                'center': np.array([1.0, 0.0]),
                'cluster_std': 0.1,
                'random_noise': 0.0
            }],
            'feature': D,
            'labels': 2,
            'random_noise': 0.0
        }

        for iteration in np.arange(2, 250, 10):

            config['clusters'][1]['samples'] = iteration
            X, y = FuzzyClassValidationTest.sampleData(copy.deepcopy(config))
            silhouette_score, ccc, mapa1, mapa5, mapa10 = FuzzyClassValidationTest.score(
                X, y)
            silhouette_scores[counter] += silhouette_score
            cccs[counter] += ccc
            mapa1_scores[counter] += mapa1
            mapa5_scores[counter] += mapa5
            mapa10_scores[counter] += mapa10
            counter += 1
            # FuzzyClassValidationTest.showData(X,y)
    silhouette_scores = gaussian_filter1d(silhouette_scores/N, sigma=sigma)
    cccs = gaussian_filter1d(cccs/N, sigma=sigma)
    mapa1_scores = gaussian_filter1d(mapa1_scores/N, sigma=sigma)
    mapa5_scores = gaussian_filter1d(mapa5_scores/N, sigma=sigma)
    mapa10_scores = gaussian_filter1d(mapa10_scores/N, sigma=sigma)
    plt.plot(np.arange(2.0, 250, 10.0), silhouette_scores,
             color='green', label='gSil')
    plt.plot(np.arange(2.0, 250, 10.0), cccs, color='blue', label='CCC')
    plt.plot(np.arange(2.0, 250, 10.0), mapa1_scores, color='purple',
             label='MAP@1 Score', linestyle=(0, (2, 5)))
    plt.plot(np.arange(2.0, 250, 10.0), mapa5_scores,
             color='turquoise', label='MAP@5 Score', linestyle=(2, (2, 5)))
    plt.plot(np.arange(2.0, 250, 10.0), mapa10_scores, color='orange',
             label='MAP@10 Score', linestyle=(4, (2, 5)))
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.title("cluster size")
    plt.xlabel("size")
    plt.ylabel("score")
    plt.tight_layout()
    plt.savefig('results/cluster_size.png')
    plt.close()

    silhouette_scores = np.zeros(len(np.arange(2.0, 21, 1.0)))
    cccs = np.zeros(len(np.arange(2.0, 21, 1.0)))
    mapa1_scores = np.zeros(len(np.arange(2.0, 21, 1.0)))
    mapa5_scores = np.zeros(len(np.arange(2.0, 21, 1.0)))
    mapa10_scores = np.zeros(len(np.arange(2.0, 21, 1.0)))

    for iteration in np.arange(0, N):
        counter = 0
        config = {
            'clusters': [{
                'distribution': 'line',
                'samples': 50,
                'label': 0,
                'start': np.array([1.0, 0.0]),
                'end': np.array([1.0, 1.0]),
                'cluster_std': 0.1,
                'random_noise': 0.0
            }],
            'feature': D,
            'labels': 2,
            'random_noise': 0.1
        }

        for iteration in np.arange(0.0, 19, 1.0):
            config['clusters'].append({
                'distribution': 'line',
                'samples': 50,  # int(100/iteration),
                'label': int(iteration+1) % 3,
                # np.array([1.0+iteration, 0.0]),
                'start': np.array([10*(1.0+iteration), 0.0]),
                # np.array([1.0+iteration, random.random()*5]),
                'end': np.array([10*(1.0+iteration), 1]),
                'cluster_std': 0.1,
                'random_noise': 0.0
            })
            config['labels'] = 3 if iteration + \
                1 > 1 else 2  # 1 + int(iteration)
            #config['feature'] = int(iteration*2)
            #config['clusters'][0]['cluster_std'] = config['clusters'][0]['cluster_std']*m
            X, y = FuzzyClassValidationTest.sampleData(copy.deepcopy(config))
            silhouette_score, ccc, mapa1, mapa5, mapa10 = FuzzyClassValidationTest.score(
                X, y)
            silhouette_scores[counter] += silhouette_score
            cccs[counter] += ccc
            mapa1_scores[counter] += mapa1
            mapa5_scores[counter] += mapa5
            mapa10_scores[counter] += mapa10
            counter += 1
            # FuzzyClassValidationTest.showData(X,y)
    silhouette_scores = gaussian_filter1d(silhouette_scores/N, sigma=sigma)
    cccs = gaussian_filter1d(cccs/N, sigma=sigma)
    mapa1_scores = gaussian_filter1d(mapa1_scores/N, sigma=sigma)
    mapa5_scores = gaussian_filter1d(mapa5_scores/N, sigma=sigma)
    mapa10_scores = gaussian_filter1d(mapa10_scores/N, sigma=sigma)
    plt.plot(np.arange(2.0, 21, 1.0), silhouette_scores,
             color='green', label='gSil')
    plt.plot(np.arange(2.0, 21, 1.0), cccs, color='blue', label='CCC')
    plt.plot(np.arange(2.0, 21, 1.0), mapa1_scores, color='purple',
             label='MAP@1 Score', linestyle=(0, (2, 5)))
    plt.plot(np.arange(2.0, 21, 1.0), mapa5_scores, color='turquoise',
             label='MAP@5 Score', linestyle=(2, (2, 5)))
    plt.plot(np.arange(2.0, 21, 1.0), mapa10_scores, color='orange',
             label='MAP@10 Score', linestyle=(4, (2, 5)))

    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.title("count of clusters")
    plt.xlabel("count of clusters")
    plt.ylabel("score")
    plt.tight_layout()
    plt.savefig('results/count_of_clusters.png')
    plt.close()

    config = {
        'clusters': [{
            'distribution': 'blob',
            'samples': 25,
            'label': 1,
            'center': np.array([-1.0, 0.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'blob',
            'samples': 25,
            'label': 1,
            'center': np.array([-1.0, 2.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'blob',
            'samples': 25,
            'label': 1,
            'center': np.array([0.0, 2.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'blob',
            'samples': 25,
            'label': 0,
            'center': np.array([-0.5, 1.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'blob',
            'samples': 25,
            'label': 0,
            'center': np.array([-0.5, -1.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'blob',
            'samples': 25,
            'label': 0,
            'center': np.array([-1.5, -1.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }],
        'feature': D,
        'labels': 2,
        'random_noise': 0.0
    }

    # for sample in y:
    #   print(sample)

    silhouette_scores = np.zeros(len(np.arange(0, 1, 0.04)))
    cccs = np.zeros(len(np.arange(0, 1, 0.04)))
    mapa1_scores = np.zeros(len(np.arange(0, 1, 0.04)))
    mapa5_scores = np.zeros(len(np.arange(0, 1, 0.04)))
    mapa10_scores = np.zeros(len(np.arange(0, 1, 0.04)))

    start0 = config['clusters'][0]['center']
    start1 = config['clusters'][1]['center']
    start2 = config['clusters'][2]['center']
    start3 = config['clusters'][3]['center']
    start4 = config['clusters'][4]['center']
    start5 = config['clusters'][5]['center']

    for iteration in np.arange(0, N):
        counter = 0

        config['clusters'][0]['center'] = start0
        config['clusters'][1]['center'] = start1
        config['clusters'][2]['center'] = start2
        config['clusters'][3]['center'] = start3
        config['clusters'][4]['center'] = start4
        config['clusters'][5]['center'] = start5

        for iteration in np.arange(0.0, 10, 0.4):
            m = 1.0+iteration
            config['clusters'][0]['center'] = start0*m
            config['clusters'][1]['center'] = start1*m
            config['clusters'][2]['center'] = start2*m
            config['clusters'][3]['center'] = start3*m
            config['clusters'][4]['center'] = start4*m
            config['clusters'][5]['center'] = start5*m

            config['clusters'][0]['cluster_std'] = m/10.0
            config['clusters'][1]['cluster_std'] = m/10.0
            config['clusters'][2]['cluster_std'] = m/10.0
            config['clusters'][3]['cluster_std'] = m/10.0
            config['clusters'][4]['cluster_std'] = m/10.0
            config['clusters'][5]['cluster_std'] = m/10.0
            X, y = FuzzyClassValidationTest.sampleData(copy.deepcopy(config))
            silhouette_score, ccc, mapa1, mapa5, mapa10 = FuzzyClassValidationTest.score(
                X, y)
            silhouette_scores[counter] += silhouette_score
            cccs[counter] += ccc
            mapa1_scores[counter] += mapa1
            mapa5_scores[counter] += mapa5
            mapa10_scores[counter] += mapa10
            counter += 1
            # FuzzyClassValidationTest.showData(X,y)
    silhouette_scores = gaussian_filter1d(silhouette_scores/N, sigma=sigma)
    cccs = gaussian_filter1d(cccs/N, sigma=sigma)
    mapa1_scores = gaussian_filter1d(mapa1_scores/N, sigma=sigma)
    mapa5_scores = gaussian_filter1d(mapa5_scores/N, sigma=sigma)
    mapa10_scores = gaussian_filter1d(mapa10_scores/N, sigma=sigma)
    plt.plot(np.arange(0, 1, 0.04), silhouette_scores,
             color='green', label='gSil')
    plt.plot(np.arange(0, 1, 0.04), cccs, color='blue', label='CCC')
    plt.plot(np.arange(0, 1, 0.04), mapa1_scores, color='purple',
             label='MAP@1 Score', linestyle=(0, (2, 5)))
    plt.plot(np.arange(0, 1, 0.04), mapa5_scores, color='turquoise',
             label='MAP@5 Score', linestyle=(2, (2, 5)))
    plt.plot(np.arange(0, 1, 0.04), mapa10_scores, color='orange',
             label='MAP@10 Score', linestyle=(4, (2, 5)))
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.title("cluster scale")
    plt.xlabel("scale factor")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig('results/cluster_scale.png')
    plt.close()

    config = {
        'clusters': [{
            'distribution': 'circle',
            'samples': 50,
            'label': 0,
            'center': np.array([0.0, 0.0]),
            'r': np.array([1.0, 1.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'blob',
            'samples': 50,
            'label': 1,
            'center': np.array([0.0, 0.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }],
        'feature': D,
        'labels': 2,
        'random_noise': 0.0
    }

    # for sample in y:
    #   print(sample)

    silhouette_scores = np.zeros(len(np.arange(-3, 3, 0.08)))
    cccs = np.zeros(len(np.arange(-3, 3, 0.08)))
    mapa1_scores = np.zeros(len(np.arange(-3, 3, 0.08)))
    mapa5_scores = np.zeros(len(np.arange(-3, 3, 0.08)))
    mapa10_scores = np.zeros(len(np.arange(-3, 3, 0.08)))
    for iteration in np.arange(0, N):
        counter = 0
        for center_blobx in np.arange(-3, 3, 0.08):
            config['clusters'][1]['center'][0] = center_blobx
            X, y = FuzzyClassValidationTest.sampleData(copy.deepcopy(config))
            silhouette_score, ccc, mapa1, mapa5, mapa10 = FuzzyClassValidationTest.score(
                X, y)
            silhouette_scores[counter] += silhouette_score
            cccs[counter] += ccc
            mapa1_scores[counter] += mapa1
            mapa5_scores[counter] += mapa5
            mapa10_scores[counter] += mapa10
            counter += 1
            # FuzzyClassValidationTest.showData(X,y)
    silhouette_scores = gaussian_filter1d(silhouette_scores/N, sigma=sigma)
    cccs = gaussian_filter1d(cccs/N, sigma=sigma)
    mapa1_scores = gaussian_filter1d(mapa1_scores/N, sigma=sigma)
    mapa5_scores = gaussian_filter1d(mapa5_scores/N, sigma=sigma)
    mapa10_scores = gaussian_filter1d(mapa10_scores/N, sigma=sigma)
    plt.plot(np.arange(-3, 3, 0.08), silhouette_scores,
             color='green', label='gSil')
    plt.plot(np.arange(-3, 3, 0.08), cccs, color='blue', label='CCC')
    plt.plot(np.arange(-3, 3, 0.08), mapa1_scores, color='purple',
             label='MAP@1 Score', linestyle=(0, (2, 5)))
    plt.plot(np.arange(-3, 3, 0.08), mapa5_scores, color='turquoise',
             label='MAP@5 Score', linestyle=(2, (2, 5)))
    plt.plot(np.arange(-3, 3, 0.08), mapa10_scores, color='orange',
             label='MAP@10 Score', linestyle=(4, (2, 5)))
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.title("surrounded cluster")
    plt.xlabel("x")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig('results/surrounded_cluster.png')
    plt.close()

    config = {
        'clusters': [{
            'distribution': 'line',
            'samples': 25,
            'label': 1,
            'start': np.array([-1.0, 0.0]),
            'end': np.array([-1.0, 2.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'line',
            'samples': 25,
            'label': 1,
            'start': np.array([-1.0, 2.0]),
            'end': np.array([0.0, 2.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'line',
            'samples': 25,
            'label': 1,
            'start': np.array([0.0, 2.0]),
            'end': np.array([0.0, 0.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'line',
            'samples': 25,
            'label': 0,
            'start': np.array([-0.5, 1.0]),
            'end': np.array([-0.5, -1.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'line',
            'samples': 25,
            'label': 0,
            'start': np.array([-0.5, -1.0]),
            'end': np.array([-1.5, -1.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'line',
            'samples': 25,
            'label': 0,
            'start': np.array([-1.5, -1.0]),
            'end': np.array([-1.5, 1.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }],
        'feature': D,
        'labels': 2,
        'random_noise': 0.0
    }

    # for sample in y:
    #   print(sample)
    # Problem im Aufbau, Inter Class Distance wächst mit

    silhouette_scores = np.zeros(len(np.arange(2, 50, 4)))
    cccs = np.zeros(len(np.arange(2, 50, 4)))
    mapa1_scores = np.zeros(len(np.arange(2, 50, 4)))
    mapa5_scores = np.zeros(len(np.arange(2, 50, 4)))
    mapa10_scores = np.zeros(len(np.arange(2, 50, 4)))
    for iteration in np.arange(0, N):
        counter = 0
        for dim in np.arange(2, 50, 4):
            config['feature'] = dim
            X, y = FuzzyClassValidationTest.sampleData(copy.deepcopy(config))
            silhouette_score, ccc, mapa1, mapa5, mapa10 = FuzzyClassValidationTest.score(
                X, y)
            silhouette_scores[counter] += silhouette_score
            cccs[counter] += ccc
            mapa1_scores[counter] += mapa1
            mapa5_scores[counter] += mapa5
            mapa10_scores[counter] += mapa10
            counter += 1
            # FuzzyClassValidationTest.showData(X,y)
    silhouette_scores = gaussian_filter1d(silhouette_scores/N, sigma=sigma)
    cccs = gaussian_filter1d(cccs/N, sigma=sigma)
    mapa1_scores = gaussian_filter1d(mapa1_scores/N, sigma=sigma)
    mapa5_scores = gaussian_filter1d(mapa5_scores/N, sigma=sigma)
    mapa10_scores = gaussian_filter1d(mapa10_scores/N, sigma=sigma)
    plt.plot(np.arange(2, 50, 4), silhouette_scores,
             color='green', label='gSil')
    plt.plot(np.arange(2, 50, 4), cccs, color='blue', label='CCC')
    plt.plot(np.arange(2, 50, 4), mapa1_scores, color='purple',
             label='MAP@1 Score', linestyle=(0, (2, 5)))
    plt.plot(np.arange(2, 50, 4), mapa5_scores, color='turquoise',
             label='MAP@5 Score', linestyle=(2, (2, 5)))
    plt.plot(np.arange(2, 50, 4), mapa10_scores, color='orange',
             label='MAP@10 Score', linestyle=(4, (2, 5)))
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.title("dimensions")
    plt.xlabel("dimensions")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig('results/dimensions.png')
    plt.close()

    config = {
        'clusters': [{
            'distribution': 'blob',
            'samples': 50,
            'label': 0,
            'center': np.array([1.0, 0.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'blob',
            'samples': 50,
            'label': 1,
            'center': np.array([0.0, 1.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'blob',
            'samples': 50,
            'label': 2,
            'center': np.array([-1.0, 0.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'blob',
            'samples': 50,
            'label': 3,
            'center': np.array([0.0, -1.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }],
        'feature': D,
        'labels': 4,
        'random_noise': 0.0
    }

    # for sample in y:
    #   print(sample)

    silhouette_scores = np.zeros(len(np.arange(0.1, 3, 0.04)))
    cccs = np.zeros(len(np.arange(0.1, 3, 0.04)))
    mapa1_scores = np.zeros(len(np.arange(0.1, 3, 0.04)))
    mapa5_scores = np.zeros(len(np.arange(0.1, 3, 0.04)))
    mapa10_scores = np.zeros(len(np.arange(0.1, 3, 0.04)))
    for iteration in np.arange(0, N):
        counter = 0
        for std in np.arange(0.1, 3, 0.04):
            config['clusters'][0]['cluster_std'] = std
            config['clusters'][1]['cluster_std'] = std
            config['clusters'][2]['cluster_std'] = std
            config['clusters'][3]['cluster_std'] = std
            X, y = FuzzyClassValidationTest.sampleData(copy.deepcopy(config))
            silhouette_score, ccc, mapa1, mapa5, mapa10 = FuzzyClassValidationTest.score(
                X, y)
            silhouette_scores[counter] += silhouette_score
            cccs[counter] += ccc
            mapa1_scores[counter] += mapa1
            mapa5_scores[counter] += mapa5
            mapa10_scores[counter] += mapa10
            counter += 1
            # FuzzyClassValidationTest.showData(X,y)
    silhouette_scores = gaussian_filter1d(silhouette_scores/N, sigma=sigma)
    cccs = gaussian_filter1d(cccs/N, sigma=sigma)
    mapa1_scores = gaussian_filter1d(mapa1_scores/N, sigma=sigma)
    mapa5_scores = gaussian_filter1d(mapa5_scores/N, sigma=sigma)
    mapa10_scores = gaussian_filter1d(mapa10_scores/N, sigma=sigma)
    plt.plot(np.arange(0.1, 3, 0.04), silhouette_scores,
             color='green', label='gSil')
    plt.plot(np.arange(0.1, 3, 0.04), cccs, color='blue', label='CCC')
    plt.plot(np.arange(0.1, 3, 0.04), mapa1_scores, color='purple',
             label='MAP@1 Score', linestyle=(0, (2, 5)))
    plt.plot(np.arange(0.1, 3, 0.04), mapa5_scores, color='turquoise',
             label='MAP@5 Score', linestyle=(2, (2, 5)))
    plt.plot(np.arange(0.1, 3, 0.04), mapa10_scores, color='orange',
             label='MAP@10 Score', linestyle=(4, (2, 5)))
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.xlabel("cluster standard deviation")
    plt.ylabel("Score")
    plt.title("Kompaktheit")
    plt.tight_layout()
    plt.savefig('results/cluster_standard_deviation.png')
    plt.close()

    config = {
        'clusters': [{
            'distribution': 'line',
            'samples': 25,
            'label': 1,
            'start': np.array([-1.0, 0.0]),
            'end': np.array([-1.0, 2.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'line',
            'samples': 25,
            'label': 1,
            'start': np.array([-1.0, 2.0]),
            'end': np.array([0.0, 2.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'line',
            'samples': 25,
            'label': 1,
            'start': np.array([0.0, 2.0]),
            'end': np.array([0.0, 0.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'line',
            'samples': 25,
            'label': 0,
            'start': np.array([-0.5, 0.0]),
            'end': np.array([-0.5, -2.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'line',
            'samples': 25,
            'label': 0,
            'start': np.array([-0.5, -2.0]),
            'end': np.array([-1.5, -2.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }, {
            'distribution': 'line',
            'samples': 25,
            'label': 0,
            'start': np.array([-1.5, -2.0]),
            'end': np.array([-1.5, 0.0]),
            'cluster_std': 0.1,
            'random_noise': 0.0
        }],
        'feature': D,
        'labels': 2,
        'random_noise': 0.0
    }

    # for sample in y:
    #   print(sample)

    silhouette_scores = np.zeros(len(np.arange(-2.0, 0.0, 0.04)))
    cccs = np.zeros(len(np.arange(-2.0, 0.0, 0.04)))
    mapa1_scores = np.zeros(len(np.arange(-2.0, 0.0, 0.04)))
    mapa5_scores = np.zeros(len(np.arange(-2.0, 0.0, 0.04)))
    mapa10_scores = np.zeros(len(np.arange(-2.0, 0.0, 0.04)))
    for iteration in np.arange(0, N):
        counter = 0
        for blobx in np.arange(-2.0, 0.0, 0.04):
            config['clusters'][3]['start'][1] = 2.0+blobx
            config['clusters'][3]['end'][1] = blobx
            config['clusters'][4]['start'][1] = blobx
            config['clusters'][4]['end'][1] = blobx
            config['clusters'][5]['start'][1] = blobx
            config['clusters'][5]['end'][1] = 2.0+blobx
            X, y = FuzzyClassValidationTest.sampleData(copy.deepcopy(config))
            silhouette_score, ccc, mapa1, mapa5, mapa10 = FuzzyClassValidationTest.score(
                X, y)
            silhouette_scores[counter] += silhouette_score
            cccs[counter] += ccc
            mapa1_scores[counter] += mapa1
            mapa5_scores[counter] += mapa5
            mapa10_scores[counter] += mapa10
            counter += 1
            # FuzzyClassValidationTest.showData(X,y)
    silhouette_scores = gaussian_filter1d(silhouette_scores/N, sigma=sigma)
    cccs = gaussian_filter1d(cccs/N, sigma=sigma)
    mapa1_scores = gaussian_filter1d(mapa1_scores/N, sigma=sigma)
    mapa5_scores = gaussian_filter1d(mapa5_scores/N, sigma=sigma)
    mapa10_scores = gaussian_filter1d(mapa10_scores/N, sigma=sigma)
    plt.plot(np.arange(2.0, 0.0, -0.04), silhouette_scores,
             color='green', label='gSil')
    plt.plot(np.arange(2.0, 0.0, -0.04), cccs, color='blue', label='CCC')
    plt.plot(np.arange(2.0, 0.0, -0.04), mapa1_scores, color='purple',
             label='MAP@1 Score', linestyle=(0, (2, 5)))
    plt.plot(np.arange(2.0, 0.0, -0.04), mapa5_scores,
             color='turquoise', label='MAP@5 Score', linestyle=(2, (2, 5)))
    plt.plot(np.arange(2.0, 0.0, -0.04), mapa10_scores,
             color='orange', label='MAP@10 Score', linestyle=(4, (2, 5)))
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.xlabel("cluster distance")
    plt.ylabel("score")
    plt.title("Umschließende Cluster")
    plt.tight_layout()
    plt.savefig('results/cluster_distance.png')
    plt.close()
