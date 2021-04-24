import cv2
import mxnet as mx
import numpy as np
import scipy as sc
import pickle

from utils.math import Distances
from utils.labelProcessor import LabelProcessor
from dataProcessor.tiffReader import GEOMAP
from validation.osmClasses import OSMClasses
from validation.clcClasses import CLCClasses
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import euclidean
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class SingleClassValidation():

    '''
        Each tile is represented by the class that appears most in the image. 
        A simple classifier trained supervised on labels. The quality of embeddings 
        is measured by the quality of the classifier.
    '''

    def __init__(self, size, classifiers=['knn'], distance=Distances.L2_Dist, validation_map=GEOMAP.OSM):
        self.distance = distance
        self.labelProcessor = LabelProcessor(size, validation_map)
        if validation_map == GEOMAP.OSM:
            self.colorToLabel = OSMClasses.getLabel
            self.labelToColor = OSMClasses.getColor
        elif validation_map == GEOMAP.CLC:
            self.colorToLabel = CLCClasses.getLabel
            self.labelToColor = CLCClasses.getColor
        self.classifiers = classifiers

    def predict(self, embeddings, class_imgs):
        colors = []
        #data, labels = self.getTrainData(embeddings, class_imgs)
        for key, classifier in self.classifiers.items():
            pred_labels = classifier.predict(embeddings)
            for clabel in range(0, len(pred_labels)):
                print(pred_labels[clabel])
                colors.append(self.labelToColor(pred_labels[clabel]))
        return colors

    def train(self, embeddings, class_imgs, file=None):
        if file:
            for key, classifier in self.classifiers.items():
                with open(file + key + '.pkl', 'rb') as fid:
                    self.classifiers[key] = pickle.load(fid)
        else:
            data, labels = self.getTrainData(embeddings, class_imgs)
            for key, classifier in self.classifiers.items():
                classifier.fit(data, labels)
                with open('single_class' + key + '.pkl', 'wb') as fid:
                    pickle.dump(classifier, fid) 
        return self


    def scores(self, embeddings, class_imgs):

        accs = {}
        accs_per_class = {}
        f1_scores = {}
        silhouette_scores = {}
        ccc = {}
        data, labels = self.getTrainData(embeddings, class_imgs)

        class_counts = {}
        for label in labels:
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1
        for key, classifier in self.classifiers.items():
            predicted = classifier.predict(data)
            f1score = f1_score(labels, predicted, average='weighted')
            acc = accuracy_score(labels, predicted)
            acc_per_class = balanced_accuracy_score(labels, predicted)#
            accs[key] = acc
            accs_per_class[key] = acc_per_class
            f1_scores[key] = f1score

            s_score = silhouette_score(data, labels.astype(int), metric='l2')
            silhouette_scores[key] = s_score

            X = [[i] for i in labels.astype(int)]
            dendro = sc.cluster.hierarchy.linkage(X, 'single')
            dists = sc.spatial.distance.pdist(data)
            cophe_dists = sc.cluster.hierarchy.cophenet(dendro)

            ccc[key] = np.corrcoef(dists, cophe_dists)[0,1]



        return accs, accs_per_class, f1_scores, silhouette_scores, ccc
        
        


    def getTrainData(self, embeddings, class_imgs):
        data = None
        labels = None
        np_class_imgs = class_imgs
        np_embeddings = embeddings

        for np_class_img in range(0, len(np_class_imgs)):
            label = self.labelProcessor.getLabels(np_class_imgs[np_class_img], processed=True)
            idx = np.argmax(label)
            #print(idx)
            if data is None:
                data = np.expand_dims(np_embeddings[np_class_img], axis=0)
                labels = np.expand_dims(idx, axis=0)
            else:
                data = np.concatenate((data, np.expand_dims(np_embeddings[np_class_img], axis=0)), axis=0)
                labels = np.concatenate((labels, np.expand_dims(idx, axis=0)), axis=0)

        return data, labels




def unitTest():
    def mean(img):
        return img.mean(axis=(0,1), exclude=True)

    images, coords, px_coords, valid = batchSampler.unitTest(32, 64)
    emb_pred, emb_pos, emb_neg = images
    class_pred, class_pos, class_neg = valid
    embs = nd.concat(mean(emb_pred), mean(emb_pos), dim=0)
    class_imgs = nd.concat(class_pred, class_pos, dim=0)

    embs = nd.concat(embs, mean(emb_neg), dim=0)
    class_imgs = nd.concat(class_imgs, class_neg, dim=0)

    validator = Validation(embs[:int(len(embs)/2)], class_imgs[:int(len(embs)/2)])
    validator.train()
    acc = validator.accurancy(embs[int(len(embs)/2):], class_imgs[int(len(embs)/2):])
    print(acc)
