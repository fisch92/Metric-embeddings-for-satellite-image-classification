import random
import cv2 
import pickle
import numpy as np
import scipy as sc

from dataProcessor.tiffReader import TiffReader
from dataProcessor.tiffReader import GEOMAP
from dataProcessor.tiffReader import MissingDataError
from utils.labelProcessor import LabelProcessor

class SupervisedImageSampler():

    def __init__(self, 
        size, 
        geomap=GEOMAP.SENTINEL,
        validationmap=GEOMAP.OSM,
        pool='validationPool',
        singleClassTreshold=0.0
    ):
        self.validationmap = validationmap
        self.size = size

        self.start_lon = 5.6666654+0.3
        self.start_lan = 47.3537047+0.3
        self.end_lon = 15.3333306-0.3
        self.end_lan =  55.7054442-0.3

        self.tiff = TiffReader(geomap)
        self.validation = TiffReader(validationmap)
        self.labelProcessor = LabelProcessor(self.size, validationmap)
        if not pool is None:
            with open(pool+GEOMAP.toString(validationmap)+str(size)+'True'+str(singleClassTreshold)+'.pkl', 'rb') as fid:
                self.validationPool = pickle.load(fid)
        if not pool is None:
            with open(pool+GEOMAP.toString(validationmap)+str(size)+'False'+str(singleClassTreshold)+'.pkl', 'rb') as fid:
                self.trainPoolPool = pickle.load(fid)
            
        self.valLabelIterator = 0

    def getrandomPool(self, train):
        if train:
            pool = self.trainPoolPool
        else:
            pool = self.validationPool

        if self.valLabelIterator > self.labelProcessor.nb_labels:
            self.valLabelIterator = 0
        
        max_labels = pool[4].argmax(axis=1)
        idxs = np.argwhere(max_labels == self.valLabelIterator)
        #print(idx)
        while len(idxs) < 50:
            self.valLabelIterator += 1
            if self.valLabelIterator > self.labelProcessor.nb_labels:
                self.valLabelIterator = 0
            idxs = np.argwhere(max_labels == self.valLabelIterator)
            
        #print(idxs.shape)
        rand_val = idxs[random.randint(0, len(idxs)-1)][0]
       
        
        #print(self.valLabelIterator)
        self.valLabelIterator += 1
        #print('image', self.valLabelIterator)

        pred = pool[0][rand_val]
        pred_coord = pool[1][rand_val]
        pred_pxcoord = pool[2][rand_val]
        pred_valid = pool[3][rand_val]
        #label = self.labelProcessor.getLabels(pred_valid, processed=False)
        #idx = np.argmax(label)
        #print('image', idx)
        return pred, pred_coord, pred_pxcoord, pred_valid

    def getPoolbyLabel(self, labels, exclude=False):
       
        pool = self.trainPoolPool
        
        if exclude:
            tlabel = random.randint(0, self.labelProcessor.nb_labels-1)
            if tlabel in labels:
                return self.getPoolbyLabel(labels, exclude)
        else:
            tlabel = random.choice(labels)

        max_labels = pool[4].argmax(axis=1)
        idxs = np.argwhere(max_labels == tlabel)
            
        if len(idxs) == 0:
            return None, None, None, None, None
            
        rand_val = idxs[random.randint(0, len(idxs)-1)][0]

        pred = pool[0][rand_val]
        pred_coord = pool[1][rand_val]
        pred_pxcoord = pool[2][rand_val]
        pred_valid = pool[3][rand_val]
        return pred, pred_coord, pred_pxcoord, pred_valid, tlabel


    def get_single_by_px(self, iteration=0, validation=False):

        return self.getrandomPool(train=True)


    def get_triplet_by_px(self, iteration=0):
        tlabel = random.randint(0, self.labelProcessor.nb_labels-1)
        pred, pred_coord, pred_pxcoord, pred_valid, label = self.getPoolbyLabel([tlabel], exclude=False)
        pos, pos_coord, pos_pxcoord, pos_valid, label = self.getPoolbyLabel([tlabel], exclude=False)
        if pred is None or pos is None:
            return self.get_triplet_by_px()
        #print("supervised")

        neg = None
        while neg is None:
            neg, neg_coord, neg_pxcoord, neg_valid, label = self.getPoolbyLabel([tlabel], exclude=True)
        

        return (pred, pos, neg), (pred_coord, pos_coord, neg_coord), (pred_pxcoord, pos_pxcoord, neg_pxcoord), (pred_valid, pos_valid, neg_valid)

    def get_quadruplet_by_px(self, iteration=0):
        tlabel = random.randint(0, self.labelProcessor.nb_labels-1)
        pred, pred_coord, pred_pxcoord, pred_valid, label = self.getPoolbyLabel([tlabel], exclude=False)
        pos, pos_coord, pos_pxcoord, pos_valid, label = self.getPoolbyLabel([tlabel], exclude=False)
        if pred is None or pos is None:
            return self.get_quadruplet_by_px()

        neg = None
        while neg is None:
            neg, neg_coord, neg_pxcoord, neg_valid, label = self.getPoolbyLabel([tlabel], exclude=True)
        
        neg2 = None
        while neg2 is None:
            neg2, neg2_coord, neg2_pxcoord, neg2_valid, label = self.getPoolbyLabel([tlabel, label], exclude=True)

        return (pred, pos, neg, neg2), (pred_coord, pos_coord, neg_coord, neg2_coord), (pred_pxcoord, pos_pxcoord, neg_pxcoord, neg2_pxcoord), (pred_valid, pos_valid, neg_valid, neg2_valid)    

    def reset(self):
        pass


    def get_tiled_by_px(self, cols=4, rows=4, iteration=0):
        self.valLabelIterator += 1
    
        if self.valLabelIterator > self.labelProcessor.nb_labels-1:
            self.valLabelIterator = 0

        preds = []
        preds_valid = []
        for col in range(0,cols):
            for row in range(0,rows):
                pred, pred_coord, pred_pxcoord, pred_valid, _ = self.getPoolbyLabel([self.valLabelIterator])
                if pred is None:
                    return self.get_tiled_by_px(cols=cols, rows=rows, iteration=0)

                preds.append(pred)
                preds_valid.append(pred_valid)


        return preds, preds_valid

    
def unitTest():
    sampler = ImageSampler(256/8, 256/2, 256*10, 256*100, 256, 256)
    images, coords,_ = sampler.get_quadruplet_by_px()

if __name__ == '__main__':
    unitTest()
