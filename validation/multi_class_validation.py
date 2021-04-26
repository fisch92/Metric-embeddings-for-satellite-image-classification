import cv2
import mxnet as mx
import numpy as np
import scipy as sc

from utils.math import Distances
from dataProcessor.tiffReader import GEOMAP
from validation.osmClasses import OSMClasses
from utils.labelProcessor import LabelProcessor
from validation.clcClasses import CLCClasses
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_log_error, normalized_mutual_info_score
from lib.mapar.mapar import Mapar
from sklearn.neighbors import BallTree
from sklearn.neighbors import DistanceMetric


class MultiClassValidation():

    '''
        Each tile is represented by the proportion taken by each class in the image. 
        A simple regressor trained supervised on the real proportion of labels. The quality of embeddings 
        is measured by the quality of the regressor.
        MAP@R and gSil determine the score without a additional regressor. 
    '''

    def __init__(self, size, classifiers={'knn': KNeighborsClassifier()}, distance=Distances.L2_Dist, validation_map=GEOMAP.OSM):
        self.distance = distance
        self.classifiers = classifiers
        self.labelProcessor = LabelProcessor(size, validation_map)
        if validation_map == GEOMAP.OSM:
            self.colorToLabel = OSMClasses.getLabel
            self.nb_labels = 13
        elif validation_map == GEOMAP.CLC:
            self.colorToLabel = CLCClasses.getLabel
            self.nb_labels = 45

    def train(self, embeddings, class_imgs):
        data, labels = self.getTrainData(embeddings, class_imgs)
        for key, classifier in self.classifiers.items():
            classifier.fit(data, labels)
        return self

    def predict(self, embeddings, class_imgs):
        data, labels = self.getTrainData(embeddings, class_imgs)
        for key, classifier in self.classifiers.items():
            pred_labels = classifier.predict(data)
            for clabel in range(0, len(labels)):
                print(labels[clabel], pred_labels[clabel])

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
            intra = np.min([np.repeat(y[xind[:, :1]][:, :, cluster], 
                xind[:,1:].shape[1], axis=1), y[xind[:, 1:]][:, :, cluster]], axis=0)
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

    def scores(self, embeddings, class_imgs):

        tsum_error = {}
        nmi = {}
        silhouette_scores = {}
        ccc = {}
        mapar1 = {}
        mapar5 = {}
        mapar10 = {}
        data, labels = self.getTrainData(embeddings, class_imgs)

        for key, classifier in self.classifiers.items():
            pred_labels = classifier.predict(data)
            pred_labels = pred_labels / \
                np.expand_dims(pred_labels.sum(axis=1), axis=1)
            pred_labels = np.clip(pred_labels, 0, 1)
            #err = mean_squared_log_error(pred_labels, labels)

            sum_error = 0
            for batch in range(0, len(labels)):
                sum_error += np.sum(np.abs(pred_labels[batch] - labels[batch]))
            sum_error = sum_error / len(labels)
            tsum_error[key] = sum_error

            mean_nmi_score = 0
            for batch in range(0, len(pred_labels)):
                score = normalized_mutual_info_score(pred_labels[batch].astype(
                    int), labels[batch].astype(int), average_method='arithmetic')
                mean_nmi_score += score
            nmi[key] = mean_nmi_score/len(pred_labels)

            silhouette_scores[key] = MultiClassValidation.silhouette(
                data, labels)

            dendro = sc.cluster.hierarchy.linkage(labels, 'single')
            dists = sc.spatial.distance.pdist(data)
            cophe_dists = sc.cluster.hierarchy.cophenet(dendro)

            ccc[key] = np.corrcoef(dists, cophe_dists)[0, 1]

            mapar1[key] = Mapar.score(data, labels, k=1)
            mapar5[key] = Mapar.score(data, labels, k=5)
            mapar10[key] = Mapar.score(data, labels, k=10)

        return tsum_error, nmi, silhouette_scores, ccc, mapar1, mapar5, mapar10

    def getTrainData(self, embeddings, class_imgs):
        data = None
        labels = None
        np_class_imgs = class_imgs
        np_embeddings = embeddings

        for np_class_img in range(0, len(np_class_imgs)):
            label = self.labelProcessor.getLabels(
                np_class_imgs[np_class_img], processed=True)
            # print(idx)
            if label.sum() > 0:
                if data is None:
                    data = np.expand_dims(np_embeddings[np_class_img], axis=0)
                    labels = np.expand_dims(label, axis=0)
                else:
                    data = np.concatenate((data, np.expand_dims(
                        np_embeddings[np_class_img], axis=0)), axis=0)
                    labels = np.concatenate(
                        (labels, np.expand_dims(label, axis=0)), axis=0)

        labels = labels/np.expand_dims(labels.sum(axis=1), axis=1)
        return data, labels


def unitTest():
    def mean(img):
        return img.mean(axis=(0, 1), exclude=True)

    images, coords, px_coords, valid = batchSampler.unitTest(32, 64)
    emb_pred, emb_pos, emb_neg = images
    class_pred, class_pos, class_neg = valid
    embs = nd.concat(mean(emb_pred), mean(emb_pos), dim=0)
    class_imgs = nd.concat(class_pred, class_pos, dim=0)

    embs = nd.concat(embs, mean(emb_neg), dim=0)
    class_imgs = nd.concat(class_imgs, class_neg, dim=0)

    validator = Validation(embs[:int(len(embs)/2)],
                           class_imgs[:int(len(embs)/2)])
    validator.train()
    acc = validator.accurancy(
        embs[int(len(embs)/2):], class_imgs[int(len(embs)/2):])
    print(acc)
