import cv2
import time
import random
import threading
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from multiprocessing import Process, Queue

from enum import Enum
from utils.labelProcessor import LabelProcessor
from dataProcessor.imageProcessor import ImageProcessor
from dataProcessor.imageSampler import ImageSampler
from dataProcessor.miningTypes import MiningTypes
from dataProcessor.tiffReader import GEOMAP

from dataProcessor.batchSampler.abstractBatchSampler import AbstractBatchSampler

class TripletBatchSampler(AbstractBatchSampler):

    def __init__(self, *, distance=None, mining=[], random_mining_iterations=5, ctx=mx.gpu(), **kwargs):
        self.distance = distance
        self.mining = mining
        self.random_mining_iterations = random_mining_iterations
        super(TripletBatchSampler, self).__init__(**kwargs)
        
    
    def prepareBatch(self):
        pred_batches = np.zeros((self.batch_size, self.channels, self.image_sampler.size, self.image_sampler.size))
        pos_batches = np.zeros((self.batch_size, self.channels, self.image_sampler.size, self.image_sampler.size))
        neg_batches = np.zeros((self.batch_size, self.channels, self.image_sampler.size, self.image_sampler.size))

        pred_coords = np.zeros((self.batch_size, 2))
        pos_coords = np.zeros((self.batch_size, 2))
        neg_coords = np.zeros((self.batch_size, 2))

        pred_px_coords = np.zeros((self.batch_size, 2))
        pos_px_coords = np.zeros((self.batch_size, 2))
        neg_px_coords = np.zeros((self.batch_size, 2))

        pred_valid_batches = np.zeros((self.batch_size, self.channels, self.image_sampler.size, self.image_sampler.size))
        pos_valid_batches = np.zeros((self.batch_size, self.channels, self.image_sampler.size, self.image_sampler.size))
        neg_valid_batches = np.zeros((self.batch_size, self.channels, self.image_sampler.size, self.image_sampler.size))

        for batch in range(0, self.batch_size):
            images, coords, px_coords, valid_images = self.image_sampler.get_triplet_by_px()

            prep_imgs = ImageProcessor.prepare_Images(images, ctx=self.ctx, bgr=False)
            prep_valid_imgs = ImageProcessor.prepare_Images(valid_images, size=self.image_sampler.size, ctx=self.ctx, bgr=True, validation=True)

            pred_img, pos_img, neg_img = prep_imgs
            pred_coord, pos_coord, neg_coord = coords
            pred_px_coord, pos_px_coord, neg_px_coord = px_coords
            pred_valid_img, pos_valid_img, neg_valid_img = prep_valid_imgs

            pred_batches[batch] = pred_img
            pred_coords[batch] = pred_coord
            pred_px_coords[batch] = pred_px_coord
            pred_valid_batches[batch] = pred_valid_img

            pos_batches[batch] = pos_img
            pos_coords[batch] = pos_coord
            pos_px_coords[batch] = pos_px_coord
            pos_valid_batches[batch] = pos_valid_img

            neg_batches[batch] = neg_img
            neg_coords[batch] = neg_coord
            neg_px_coords[batch] = neg_px_coord
            neg_valid_batches[batch] = neg_valid_img

        if MiningTypes.HARD_NEGATIVE in self.mining:
            arg_neg_batches = self.argDoHardMining(pred_batches, neg_batches)
            neg_batches = neg_batches[arg_neg_batches]
            neg_coords = neg_coords[arg_neg_batches]
            neg_valid_batches = neg_valid_batches[arg_neg_batches]
        if MiningTypes.HARD_POSITIVE in self.mining:
            arg_pos_batches = self.argDoHardMining(pred_batches, pos_batches, minimize=False)
            pos_batches = pos_batches[arg_pos_batches]
            pos_coords = pos_coords[arg_pos_batches]
            pos_valid_batches = pos_valid_batches[arg_pos_batches]

        if MiningTypes.RANDOM_HARD_NEGATIVE in self.mining:
            neg_batches, neg_coords, neg_px_coords, neg_valid_batches = self.doRandomHardMining(pred_batches, pred_px_coords)
        if MiningTypes.RANDOM_HARD_POSITIVE in self.mining:
            pos_batches, pos_coords, pos_px_coords, pos_valid_batches = self.doRandomHardMining(pred_batches, pred_px_coords, isNeg=False)

        self.reset()
        return ((pred_batches, pos_batches, neg_batches), (pred_coords, pos_coords, neg_coords), (pred_px_coords, pos_px_coords, neg_px_coords), (pred_valid_batches, pos_valid_batches, neg_valid_batches))


    def argDoHardMining(self, pred, hard, minimize=True):

        pred = nd.array(pred, ctx=self.ctx)
        hard = nd.array(hard, ctx=self.ctx)
        out_hard = nd.zeros(self.batch_size, ctx=self.ctx)
        emb_hard = self.net(hard)
        emb_pred = self.net(pred)
        for batch in range(0, len(pred)):
            single_pred = nd.expand_dims(emb_pred[batch], axis=0)
            
            distances = self.distance(single_pred, emb_hard)
            #print(distances.shape)
            sort = nd.argsort(distances)
            if minimize:
                out_hard[batch] = sort[0]
            else:
                out_hard[batch] = sort[-1]
        #print(out_hard.shape)
        return out_hard.asnumpy().astype('int32')

    def doRandomHardMining(self, pred, pred_px_coords, isNeg=True):

        out_hard = nd.zeros(pred.shape, ctx=self.ctx)
        out_hard_coords = nd.zeros((pred.shape[0], 2), ctx=self.ctx)
        out_hard_px_coords = nd.zeros((pred.shape[0], 2), ctx=self.ctx)
        out_hard_valid = nd.zeros(pred.shape, ctx=self.ctx)

        out_hard_distances = -nd.ones(pred.shape[0], ctx=self.ctx)

        emb_pred = self.net(pred)
        
        for iteration in range(0, self.random_mining_iterations):
            for batch in range(0, len(pred)):
            
                pred_x, pred_y = pred_px_coords[batch]
                if isNeg:
                    hard, hard_coord, hard_px_coord, hard_valid_img = self.image_sampler.get_neg(pred_x, pred_y)
                else:
                    hard, hard_coord, hard_px_coord, hard_valid_img = self.image_sampler.get_pos(pred_x, pred_y)
                
                hard = nd.expand_dims(ImageProcessor.prepare_Image(hard, ctx=self.ctx, bgr=False), axis=0)
                hard_valid = nd.expand_dims(ImageProcessor.prepare_Image(hard_valid_img, size=self.image_sampler.size, ctx=self.ctx, bgr=True, validation=True), axis=0)
                emb_hard = self.net(hard)
                
                distance = self.distance(emb_pred[batch], emb_hard)
                use = False
                if out_hard_distances[batch] == -1:
                    use = True

                if isNeg and distance < out_hard_distances[batch]:
                    use = True

                if not isNeg and distance > out_hard_distances[batch]:
                    use = True
                    
                if use:
                    out_hard_distances[batch] = distance
                    out_hard[batch] = hard
                    out_hard_coords[batch][0], out_hard_coords[batch][1] = hard_coord
                    out_hard_px_coords[batch][0], out_hard_px_coords[batch][1] = hard_px_coord
                    out_hard_valid[batch] = hard_valid

                #print(out_hard_distances)
        return out_hard, out_hard_coords, out_hard_px_coords, out_hard_valid