import cv2
import time
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd

from enum import Enum
from dataProcessor.imageProcessor import ImageProcessor
from dataProcessor.imageSampler import ImageSampler
from dataProcessor.miningTypes import MiningTypes
from multiprocessing import Process, Queue

from dataProcessor.batchSampler.abstractBatchSampler import AbstractBatchSampler


class MagNetBatchSampler(AbstractBatchSampler):

    def __init__(self, **kwargs):
        super(MagNetBatchSampler, self).__init__(**kwargs)
        

    def prepareBatch(self):
        batch_cluster = []
        coord_cluster = []
        px_coord_cluster = []
        valid_cluster = []
        for iteration in range(0, 12):
            pred_batches = np.zeros((self.batch_size, self.channels, self.image_sampler.size, self.image_sampler.size))
            pred_coords = np.zeros((self.batch_size, 2))
            pred_px_coords = np.zeros((self.batch_size, 2))
            pred_valid_batches = np.zeros((self.batch_size, self.channels, self.image_sampler.size, self.image_sampler.size))
            
                
            images, valids = self.image_sampler.get_cluster_by_px(self.batch_size)
            prep_imgs = ImageProcessor.prepare_Images(images, ctx=self.ctx, bgr=False)
            prep_valid_imgs = ImageProcessor.prepare_Images(valids, size=self.image_sampler.size, ctx=self.ctx, bgr=True, validation=True)

            for batch in range(0, len(images)):
                pred_batches[batch] = prep_imgs[batch]
                pred_valid_batches[batch] = prep_valid_imgs[batch]


            batch_cluster.append(pred_batches)
            coord_cluster.append(pred_coords)
            px_coord_cluster.append(pred_px_coords)
            valid_cluster.append(pred_valid_batches)
            #print("reset")
            self.reset()

        
        return (batch_cluster, coord_cluster, px_coord_cluster, valid_cluster)