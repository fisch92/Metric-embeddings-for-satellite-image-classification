import random
import cv2 
import pickle
import numpy as np
import scipy as sc

from dataProcessor.tiffReader import TiffReader
from dataProcessor.tiffReader import GEOMAP
from dataProcessor.tiffReader import MissingDataError
from utils.labelProcessor import LabelProcessor

class ImageSampler():


    '''
        Parameters
        ----------
        random_reset : float
            Sample in region of last image, till random value is smaller than random_reset.
        pool: string
            Name of validation pool file. (generated with createPool.py)
        valmod: int
            Reserve one tile on every [valmod] px for validation.
        singleClassTreshold:
            Validate only an images, that dominated (threshold) by one class.
    '''
    def __init__(self, 
        min_pos_dist, 
        max_pos_dist, 
        min_neg_dist, 
        max_neg_dist, 
        size, 
        geomap=GEOMAP.SENTINEL,
        validationmap=GEOMAP.OSM,
        random_reset=0.5,
        pool='validationPool',
        valmod=500,
        singleClassTreshold=0.0
    ):
        self.validationmap = validationmap
        self.min_pos_dist = min_pos_dist
        self.max_pos_dist = max_pos_dist
        self.min_neg_dist = min_neg_dist
        self.max_neg_dist = max_neg_dist
        self.size = size

        # ~ germany 
        self.start_lon = 5.6666654+0.3
        self.start_lan = 47.3537047+0.3
        self.end_lon = 15.3333306-0.3
        self.end_lan =  55.7054442-0.3

        self.tiff = TiffReader(geomap)
        self.validation = TiffReader(validationmap)
        self.labelProcessor = LabelProcessor(self.size, validationmap)
        self.valmod = valmod 
        if not pool is None:
            with open(pool+GEOMAP.toString(validationmap)+str(size)+'True'+str(singleClassTreshold)+'.pkl', 'rb') as fid:
                self.validationPool = pickle.load(fid)
        if not pool is None:
            with open(pool+GEOMAP.toString(validationmap)+str(size)+'False'+str(singleClassTreshold)+'.pkl', 'rb') as fid:
                self.trainPoolPool = pickle.load(fid)
            
        self.valLabelIterator = 0

        start_x, start_y = self.tiff.coord2Px(self.start_lon, self.end_lan)
        self.start_x = start_x
        self.start_y = start_y
        end_x, end_y = self.tiff.coord2Px(self.end_lon, self.start_lan)
        self.end_x = end_x-size
        self.end_y =  end_y-size

        self.lastpredx = None
        self.lastpredy = None

        self.random_reset = random_reset

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

    def get_single_by_px(self, iteration=0, validation=False):
        padding_end_x = self.end_x - self.min_pos_dist
        padding_start_x = self.start_x + self.min_pos_dist
        padding_end_y = self.end_y - self.min_pos_dist
        padding_start_y = self.start_y + self.min_pos_dist
        
        try:
            random_reset = random.random()
            if self.lastpredx is None or random_reset < self.random_reset: #if true --> random sampling
                pred_x = random.random()*(padding_end_x - padding_start_x) + padding_start_x
                pred_y = random.random()*(padding_end_y - padding_start_y) + padding_start_y
                pred = self.tiff.readTilePx(pred_x, pred_y, self.size, self.size)
                #print("sample random")
            else: # cluster sampling
                pred, _, (pred_x, pred_y), _ = self.get_pos(self.lastpredx, self.lastpredy)

            if pred_x%self.valmod < self.size and pred_y%self.valmod < self.size and not validation:
                return self.get_single_by_px(iteration, validation)
            elif pred_x%self.valmod > self.size and pred_y%self.valmod > self.size and validation:
                return self.get_single_by_px(iteration, validation)
            self.lastpredx, self.lastpredy = (pred_x, pred_y)
         
        except MissingDataError as e:
            if iteration < 6:
                return self.get_single_by_px(iteration=iteration+1, validation=validation)
            raise e

        pred_valid, pred_coord = self.getValidxy(pred_x, pred_y)

        return pred, pred_coord, (pred_x, pred_y), pred_valid

    def get_cluster_by_px(self, size, iteration=0, validation=False):
        padding_end_x = self.end_x - self.min_pos_dist
        padding_start_x = self.start_x + self.min_pos_dist
        padding_end_y = self.end_y - self.min_pos_dist
        padding_start_y = self.start_y + self.min_pos_dist
        
        try:
            pred_x = random.random()*(padding_end_x - padding_start_x) + padding_start_x
            pred_y = random.random()*(padding_end_y - padding_start_y) + padding_start_y
            pred = self.tiff.readTilePx(pred_x, pred_y, self.size, self.size)
            #print("sample random")
            
            self.lastpredx, self.lastpredy = (pred_x, pred_y)
         
        except MissingDataError as e:
            if iteration < 6:
                return self.get_single_by_px(iteration=iteration+1, validation=validation)
            raise e

        pred_valid, pred_coord = self.getValidxy(pred_x, pred_y)
        preds = [pred]
        preds_coord = [pred_coord]
        preds_px_coord = [(pred_x, pred_y)]
        preds_valid = [pred_valid]
        for iteration in range(0, size-1):
            pos, pos_coord, pos_px_coord, pos_valid = self.get_pos(pred_x, pred_y)
            preds.append(pos)
            preds_coord.append(pos_coord)
            preds_px_coord.append(pos_px_coord)
            preds_valid.append(pos_valid)
		
        return preds, preds_valid


    def get_triplet_by_px(self, iteration=0):
        padding_end_x = self.end_x - self.min_pos_dist
        padding_start_x = self.start_x + self.min_pos_dist
        padding_end_y = self.end_y - self.min_pos_dist
        padding_start_y = self.start_y + self.min_pos_dist
        
        try:
            random_reset = random.random()
            if self.lastpredx is None or random_reset < self.random_reset:
                pred_x = random.random()*(padding_end_x - padding_start_x) + padding_start_x
                pred_y = random.random()*(padding_end_y - padding_start_y) + padding_start_y
                pred = self.tiff.readTilePx(pred_x, pred_y, self.size, self.size)
                #print("sample random")
            else:
                pred, _, (pred_x, pred_y), _ = self.get_pos(self.lastpredx, self.lastpredy)

            if pred_x%self.valmod < self.size and pred_y%self.valmod < self.size:
                return self.get_triplet_by_px(iteration)
            pos, pos_coord, (pos_x, pos_y), pos_valid = self.get_pos(pred_x, pred_y)
            neg, neg_coord, (neg_x, neg_y), neg_valid = self.get_neg(pred_x, pred_y)
            self.lastpredx, self.lastpredy = (pred_x, pred_y)
        except MissingDataError as e:
            if iteration < 6:
                return self.get_triplet_by_px(iteration=iteration+1)
            raise e

        pred_valid, pred_coord = self.getValidxy(pred_x, pred_y)

        return (pred, pos, neg), (pred_coord, pos_coord, neg_coord), ((pred_x, pred_y), (pos_x, pos_y), (neg_x, neg_y)), (pred_valid, pos_valid, neg_valid)

    def get_quadruplet_by_px(self, iteration=0):
        padding_end_x = self.end_x - self.min_neg_dist
        padding_start_x = self.start_x + self.min_neg_dist
        padding_end_y = self.end_y - self.min_neg_dist
        padding_start_y = self.start_y + self.min_neg_dist
    
        try:
            random_reset = random.random()
            if self.lastpredx is None or random_reset < self.random_reset:
                pred_x = random.random()*(padding_end_x - padding_start_x) + padding_start_x
                pred_y = random.random()*(padding_end_y - padding_start_y) + padding_start_y
                pred = self.tiff.readTilePx(pred_x, pred_y, self.size, self.size)
            else:
                pred, _, (pred_x, pred_y), _ = self.get_pos(self.lastpredx, self.lastpredy)
                
            if pred_x%self.valmod < self.size and pred_y%self.valmod < self.size:
                return self.get_quadruplet_by_px(iteration)
            pos, pos_coord, pos_px_coord, pos_valid = self.get_pos(pred_x, pred_y)
            neg, neg_coord, neg_px_coord, neg_valid = self.get_neg(pred_x, pred_y)
            
            self.lastpredx, self.lastpredy = (pred_x, pred_y)
            
            neg2, neg2_coord, neg2_px_coord, neg2_valid = self.get_neg(pred_x, pred_y)

        except MissingDataError as e:
            if iteration < 6:
                return self.get_quadruplet_by_px(iteration=iteration+1)
            raise e


        pred_valid, pred_coord = self.getValidxy(pred_x, pred_y)
        
        return (pred, pos, neg, neg2), (pred_coord, pos_coord, neg_coord, neg2_coord), ((pred_x, pred_y), pos_px_coord, neg_px_coord, neg2_px_coord), (pred_valid, pos_valid, neg_valid, neg2_valid)

    def get_tiled_by_px(self, cols=4, rows=4, iteration=0):
        padding_end_x = self.end_x - self.min_pos_dist
        padding_start_x = self.start_x + self.min_pos_dist
        padding_end_y = self.end_y - self.min_pos_dist
        padding_start_y = self.start_y + self.min_pos_dist
        preds = []
        preds_valid = []
        try:
            start_pred_x = random.random()*(padding_end_x - padding_start_x) + padding_start_x
            start_pred_y = random.random()*(padding_end_y - padding_start_y) + padding_start_y
            for col in range(0, cols):
                for row in range(0, rows):
                    pred = self.tiff.readTilePx(start_pred_x+self.size*col, start_pred_y+self.size*row, self.size, self.size)
                    pred_valid, pred_coord = self.getValidxy(start_pred_x+self.size*col, start_pred_y+self.size*row)
                    #print(pred_valid.shape)
                    preds.append(pred)
                    preds_valid.append(pred_valid)
        except MissingDataError as e:
            if iteration < 6:
                return self.get_tiled_by_px(iteration=iteration+1)
            raise e

        return preds, preds_valid


    def get_neg(self, pred_x, pred_y):
        neg_x = random.random()*(self.end_x - self.start_x) + self.start_x
        neg_y = random.random()*(self.end_y - self.start_y) + self.start_y
        #neg_x = pred_x + random.choice([-1, 1]) * (random.random() * (self.max_neg_dist - self.min_neg_dist) + self.min_neg_dist)
        #neg_y = pred_y + random.choice([-1, 1]) * (random.random() * (self.max_neg_dist - self.min_neg_dist) + self.min_neg_dist)

        if neg_x%self.valmod < self.size and neg_y%self.valmod < self.size:
            return self.get_neg(pred_x, pred_y)
        
        
        neg = self.tiff.readTilePx(neg_x, neg_y, self.size, self.size)

        neg_valid, neg_coord = self.getValidxy(neg_x, neg_y)

        return neg, neg_coord, (neg_x, neg_y), neg_valid

    def get_pos(self, pred_x, pred_y):
        pos_x = pred_x + random.choice([-1, 1]) * (random.random() * (self.max_pos_dist - self.min_pos_dist) + self.min_pos_dist)
        pos_y = pred_y + random.choice([-1, 1]) * (random.random() * (self.max_pos_dist - self.min_pos_dist) + self.min_pos_dist)
        #print(pred_x, pos_x, pred_y, pos_y)
        
        pos_x = np.clip(pos_x, self.start_x, self.end_x)
        pos_y = np.clip(pos_y, self.start_y, self.end_y)
        if pos_x%self.valmod < self.size and pos_y%self.valmod < self.size:
            return self.get_pos(pred_x, pred_y)
        

        pos = self.tiff.readTilePx(pos_x, pos_y, self.size, self.size)

        pos_valid, pos_coord = self.getValidxy(pos_x, pos_y)
        #print(self.start_y, self.end_y)
        return pos, pos_coord, (pos_x, pos_y), pos_valid

    def getValidxy(self, x, y):
        src_start_lon, src_start_lan = self.tiff.px2Coord(x, y)
        src_end_lon, src_end_lan = self.tiff.px2Coord(x+self.size, y+self.size)

        start_x, start_y = self.validation.coord2Px(src_start_lon, src_start_lan)
        end_x, end_y = self.validation.coord2Px(src_end_lon, src_end_lan)

        valid = self.validation.readTilePx(start_x, start_y, abs(end_x-start_x), abs(end_y-start_y))
        return valid, (src_start_lon, src_start_lan)

    def getValidlonlan(self, lon, lan):
        x, y = self.tiff.coord2Px(lon, lan)
        return self.getValidxy(x, y)

    def reset(self):
        self.lastpredx, self.lastpredy = (None, None)
        self.lastpredvalx, self.lastpredvaly = (None, None)

    
def unitTest():
    sampler = ImageSampler(256/8, 256/2, 256*10, 256*100, 256, 256)
    images, coords,_ = sampler.get_quadruplet_by_px()

if __name__ == '__main__':
    unitTest()
