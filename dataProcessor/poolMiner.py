import os
import pickle
import numpy as np
from dataProcessor.imageSampler import ImageSampler
from dataProcessor.tiffReader import GEOMAP
from utils.labelProcessor import LabelProcessor


class PoolMiner():
    """docstring for PoolMiner"""
    def __init__(self, size, file, valmap, validation, singleClassTreshold, pool_size):
        self.file = file
        self.valmap = valmap
        self.pool = None
        self.pool_size = pool_size
        self.lastSave = 0
        self.size = size
        self.validation = validation
        self.singleClassTreshold = singleClassTreshold
        self.imageSampler = ImageSampler(
            0, 
            0, 
            0, 
            0, 
            size, 
            geomap=GEOMAP.SENTINEL,
            validationmap=valmap,
            random_reset=1.0,
            pool=None,
            valmod=500
        )
        self.finished = False
        self.labelProcessor = LabelProcessor(self.size, valmap)
        if os.path.isfile(self.file+
                        GEOMAP.toString(self.valmap)+
                        str(self.size)+str(self.validation)+
                        str(self.singleClassTreshold) +
                        '.pkl'):
            with open(self.file+
                        GEOMAP.toString(self.valmap)+
                        str(self.size)+str(self.validation)+
                        str(self.singleClassTreshold) +
                        '.pkl', 'rb') as fid:
                self.pool = pickle.load(fid)
        
    def mine(self):
        counter = 0
        weight = 0
        label_stats = 0
        while not self.finished:
            pred, pred_coord, (pred_x, pred_y), pred_valid = self.imageSampler.get_single_by_px(validation=self.validation)
            labels = self.fillPool(pred, pred_coord, (pred_x, pred_y), pred_valid)
            self.imageSampler.reset()
            if labels.sum()>0:
                label_stats += labels
                counter += 1
                weight += labels.sum()
                if counter%50==0:
                    print('label_stats', label_stats/weight)


    def fillPool(self, pred, pred_coord, pred_px_coord, pred_valid):
        labels = self.labelProcessor.getLabels(pred_valid, processed=False)
        data = [
            [pred], 
            np.expand_dims(pred_coord, axis=0),
            np.expand_dims(pred_px_coord, axis=0), 
            [pred_valid], 
            np.expand_dims(labels, axis=0)
        ]


        if self.pool is None: 
            self.pool = data
            return labels

        maxValue = data[4].max(axis=1)
        if maxValue < self.singleClassTreshold:
            #print("threshold")
            return labels

        dists = np.abs(self.pool[2]-pred_px_coord).sum(axis=1)
        if dists.min() < self.size:
            #print("dists")
            return labels

        if len(self.pool[0]) < self.pool_size:
            temp_Pool = self.pool
            temp_Pool[0].append(data[0][0])
            temp_Pool[1] = np.concatenate((temp_Pool[1], data[1]), axis=0)
            temp_Pool[2] = np.concatenate((temp_Pool[2], data[2]), axis=0)
            temp_Pool[3].append(data[3][0])
            temp_Pool[4] = np.concatenate((temp_Pool[4], data[4]), axis=0)
            self.pool = temp_Pool
            #print(len(self.pool[0]))
            return labels
        else:
            temp_Pool = self.pool.copy()
            unique, unique_indices, counts = np.unique(temp_Pool[4].argmax(axis=1), return_index=True, return_counts=True)
            skip = unique_indices[counts.argmax()]
            
            temp_Pool[0][skip] = data[0][0]
            temp_Pool[1][skip] = data[1][0]
            temp_Pool[2][skip] = data[2][0]
            temp_Pool[3][skip] = data[3][0]
            temp_Pool[4][skip] = data[4][0]

            if not data[4].argmax(axis=1) in unique:
                self.pool = temp_Pool
                return labels
            idx = np.argwhere(data[4].argmax(axis=1) == unique)
            #print(counts[idx][0][0], counts.max())
            if counts.max()-1 <= counts.min():
                self.finished = True
            if counts[idx][0][0] < counts.max()-2:
                self.validationPool = temp_Pool
                self.lastSave += 1
                print(counts)
                if self.lastSave > 100:
                    
                    with open(
                        self.file+
                        GEOMAP.toString(self.valmap)+
                        str(self.size)+str(self.validation)+
                        str(self.singleClassTreshold) +
                        '.pkl', 'wb'
                        ) as fid:
                        pickle.dump(self.pool, fid)
                    self.lastSave = 0 
                return labels


        return labels