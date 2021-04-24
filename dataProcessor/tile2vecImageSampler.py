import random
import cv2 
import os
import pickle
import numpy as np
import scipy as sc

from dataProcessor.tiffReader import TiffReader
from dataProcessor.tiffReader import GEOMAP
from dataProcessor.tiffReader import MissingDataError
from utils.labelProcessor import LabelProcessor
from sklearn.preprocessing import LabelEncoder


class Tile2VecImageSampler():

    def __init__(self, 
        size=50,
        tripletsDir='../tile2vecValidation/triplets',
        supervisedDir='../tile2vecValidation/data/tiles'
    ):
        self.labelProcessor = LabelProcessor(size, GEOMAP.TILE2VEC)
        self.triplets = {}
        triplets = os.listdir(tripletsDir)
        counter = 0
        for file in triplets:
            if counter % 1000 == 0:
                print('loadTriplet:', counter/len(triplets))
            if 'anchor' in file:
                key = int(file.split('anchor')[0])
                path = '/'.join([tripletsDir, file])
                data = np.load(path)
                data = data.transpose((2, 0, 1))
                if not key in self.triplets:
                    self.triplets[key] = {'pred': data}
                else:
                    self.triplets[key]['pred'] = data

            if 'distant' in file:
                key = int(file.split('distant')[0])
                path = '/'.join([tripletsDir, file])
                data = np.load(path)
                data = data.transpose((2, 0, 1))

                if not key in self.triplets:
                    self.triplets[key] = {'neg': data}
                else:
                    self.triplets[key]['neg'] = data

            if 'neighbor' in file:
                key = int(file.split('neighbor')[0])
                path = '/'.join([tripletsDir, file])
                data = np.load(path)
                data = data.transpose((2, 0, 1))

                if not key in self.triplets:
                    self.triplets[key] = {'pos': data}
                else:
                    self.triplets[key]['pos'] = data
            counter += 1
            #if counter/len(triplets) > 0.1:
            #    break

        self.validation = {}

        validation = os.listdir(supervisedDir)
        counter = 0
        for file in validation:
            print('loadVal:', counter/len(validation))
            if 'y.npy' == file:
                path = '/'.join([supervisedDir, file])
                data = np.load(path)
                data = LabelEncoder().fit_transform(data)
                self.ground_truth = data-1
            elif 'tile' in file:
                key = int(file.split('tile')[0])-1
                path = '/'.join([supervisedDir, file])
                data = np.load(path)
                self.validation[key] = data
            counter += 1

        self.valLabelIterator = 0
        self.size = size


    def getrandomPool(self, train):

        label_shuffle = [x for x in range(0, len(self.ground_truth))]
        random.shuffle(label_shuffle)
        for label in label_shuffle:
            if self.ground_truth[label] == self.valLabelIterator and label in self.validation:
                if train and not label % 5 == 0 or not train and label % 5 == 0:
                    
                    
                    pred = self.validation[label]
                    pred = pred.transpose((2, 0, 1))
                    pred_coord = (0,0)
                    pred_pxcoord = (0,0)
                    pred_valid = np.full(pred.shape, self.ground_truth[label]/self.labelProcessor.nb_labels*255.0)

                    self.valLabelIterator += 1
                    if self.valLabelIterator > self.labelProcessor.nb_labels:
                        self.valLabelIterator = 0

                    #print(pred_valid)
                    return pred, pred_coord, pred_pxcoord, pred_valid

        self.valLabelIterator += 1
        if self.valLabelIterator > self.labelProcessor.nb_labels:
            self.valLabelIterator = 0
        #print(train, self.valLabelIterator)
        return self.getrandomPool(train)


    def get_triplet_by_px(self):
        key = random.choice(list(self.triplets.keys()))
        if not 'pred' in self.triplets[key] or not 'pos' in self.triplets[key] or not 'neg' in self.triplets[key]:
            #print(self.triplets[key])
            return self.get_triplet_by_px()
        pred = self.triplets[key]['pred']
        pos = self.triplets[key]['pos']
        neg = self.triplets[key]['neg']

        return (pred, pos, neg), (0, 0, 0), ((0, 0), (0, 0), (0, 0)), (np.zeros(pred.shape), np.zeros(pos.shape), np.zeros(neg.shape))

    def get_quadruplet_by_px(self, iteration=0):
        
        key = random.choice(list(self.triplets.keys()))
        if not 'pred' in self.triplets[key] or not 'pos' in self.triplets[key] or not 'neg' in self.triplets[key]:
            #print(self.triplets[key])
            return self.get_triplet_by_px()
        pred = self.triplets[key]['pred']
        pos = self.triplets[key]['pos']
        neg = self.triplets[key]['neg']

        key = random.choice(list(self.triplets.keys()))
        if not 'pred' in self.triplets[key] and not 'pos' in self.triplets[key] and not 'neg' in self.triplets[key]:
            #print(self.triplets[key])
            return self.get_triplet_by_px()

        neg2 = self.triplets[key][random.choice(list(self.triplets[key].keys()))]
        
        return (pred, pos, neg, neg2), ((0, 0), (0, 0), (0, 0), (0, 0)), ((0, 0), (0, 0), (0, 0), (0, 0)), (np.zeros(pred.shape), np.zeros(pred.shape), np.zeros(pred.shape), np.zeros(pred.shape))


    def reset(self):
        pass