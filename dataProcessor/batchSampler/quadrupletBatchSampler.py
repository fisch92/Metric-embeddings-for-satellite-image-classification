import cv2
import random
from multiprocessing import Process, Queue
import time
import asyncio
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd

from enum import Enum
from dataProcessor.imageProcessor import ImageProcessor
from dataProcessor.imageSampler import ImageSampler
from dataProcessor.miningTypes import MiningTypes
from sklearn.neighbors import KDTree

from dataProcessor.batchSampler.abstractBatchSampler import AbstractBatchSampler


class QuadrupletBatchSampler():

    def __init__(self, distance=None, mining=[], random_mining_iterations=5):
        super(QuadrupletBatchSampler, self).__init__(**kwargs)
        self.distance = distance
        self.mining = mining
        self.random_mining_iterations = random_mining_iterations

    def prepareBatch(self):

        pred_batches = np.zeros((self.batch_size, self.channels, self.image_sampler.size, self.image_sampler.size))
        pos_batches = np.zeros((self.batch_size, self.channels, self.image_sampler.size, self.image_sampler.size))
        neg_batches = np.zeros((self.batch_size, self.channels, self.image_sampler.size, self.image_sampler.size))
        neg2_batches = np.zeros((self.batch_size, self.channels, self.image_sampler.size, self.image_sampler.size))

        pred_coords = np.zeros((self.batch_size, 2))
        pos_coords = np.zeros((self.batch_size, 2))
        neg_coords = np.zeros((self.batch_size, 2))
        neg2_coords = np.zeros((self.batch_size, 2))

        pred_px_coords = np.zeros((self.batch_size, 2))
        pos_px_coords = np.zeros((self.batch_size, 2))
        neg_px_coords = np.zeros((self.batch_size, 2))
        neg2_px_coords = np.zeros((self.batch_size, 2))

        pred_valid_batches = np.zeros((self.batch_size, self.channels, self.image_sampler.size, self.image_sampler.size))
        pos_valid_batches = np.zeros((self.batch_size, self.channels, self.image_sampler.size, self.image_sampler.size))
        neg_valid_batches = np.zeros((self.batch_size, self.channels, self.image_sampler.size, self.image_sampler.size))
        neg2_valid_batches = np.zeros((self.batch_size, self.channels, self.image_sampler.size, self.image_sampler.size))

        for batch in range(0, self.batch_size):
            images, coords, px_coords, valid_images = self.image_sampler.get_quadruplet_by_px()

            prep_imgs = ImageProcessor.prepare_Images(images, ctx=self.ctx, bgr=False)
            prep_valid_imgs = ImageProcessor.prepare_Images(valid_images, size=self.image_sampler.size, ctx=self.ctx, bgr=True)

            pred_img, pos_img, neg_img, neg2_img = prep_imgs
            pred_coord, pos_coord, neg_coord, neg2_coord = coords
            pred_px_coord, pos_px_coord, neg_px_coord, neg2_px_coord = px_coords
            pred_valid_img, pos_valid_img, neg_valid_img, neg2_valid_img = prep_valid_imgs

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

            neg2_batches[batch] = neg2_img
            neg2_coords[batch] = neg2_coord
            neg2_px_coords[batch] = neg2_px_coord
            neg2_valid_batches[batch] = neg2_valid_img

        if MiningTypes.HARD_NEGATIVE in self.mining:
            arg_neg_batches = self.argDoHardMining(pred_batches, neg_batches)
            neg_batches = neg_batches[arg_neg_batches]
            neg_coords = neg_coords[arg_neg_batches]
            neg_valid_batches = neg_valid_batches[arg_neg_batches]

            arg_neg2_batches = self.argDoHardMining(neg_batches, neg2_batches)
            neg2_batches = neg2_batches[arg_neg2_batches]
            neg2_coords = neg2_coords[arg_neg2_batches]
            neg2_valid_batches = neg2_valid_batches[arg_neg2_batches]
        if MiningTypes.HARD_POSITIVE in self.mining:
            arg_pos_batches = self.argDoHardMining(pred_batches, pos_batches, minimize=False)
            pos_batches = pos_batches[arg_pos_batches]
            pos_coords = pos_coords[arg_pos_batches]
            pos_valid_batches = pos_valid_batches[arg_neg_batches]

        if MiningTypes.RANDOM_HARD_NEGATIVE in self.mining:
            neg_batches, neg_coords, neg_px_coords, neg_valid_batches = self.doRandomHardMining(pred_batches, pred_px_coords, validation=validation)
        if MiningTypes.RANDOM_HARD_POSITIVE in self.mining:
            pos_batches, pos_coords, pos_px_coords, pos_valid_batches = self.doRandomHardMining(pred_batches, pred_px_coords, isNeg=False, validation=validation)

        
        self.reset()
        return ((pred_batches, pos_batches, neg_batches, neg2_batches), \
        (pred_coords, pos_coords, neg_coords, neg2_coords), \
        (pred_px_coords, pos_px_coords, neg_px_coords, neg2_px_coords), \
        (pred_valid_batches, pos_valid_batches, neg_valid_batches, neg2_valid_batches))

    def argDoHardMining(self, pred, hard, minimize=True):

        out_hard = nd.zeros(self.batch_size, ctx=self.ctx)

        emb_pred = self.net(pred)
        emb_hard = self.net(hard)

        tree = KDTree(emb_pred.asnumpy())
        dist, ind = tree.query(emb_hard.asnumpy(), k=1)
        for batch in range(0, len(pred)):
            if minimize:
                out_hard[batch] = ind[batch][0]
            else:
                out_hard[batch] = ind[batch][-1]

        return out_hard

    def doRandomHardMining(self, pred, pred_px_coords, isNeg=True, validation=False):

        out_hard = nd.zeros(pred.shape, ctx=self.ctx)
        out_hard_coords = nd.zeros((pred.shape[0], 2), ctx=self.ctx)
        out_hard_px_coords = nd.zeros((pred.shape[0], 2), ctx=self.ctx)
        out_hard_valid = nd.zeros(pred.shape, ctx=self.ctx)

        out_hard_distances = -nd.ones(pred.shape[0], ctx=self.ctx)

        for batch in range(0, len(pred)):
            single_pred = nd.expand_dims(pred[batch], axis=0)
            emb_pred = self.net(single_pred)
            pred_x, pred_y = pred_px_coords[batch]
            for iteration in range(0, self.random_mining_iterations):
                if isNeg:
                    hard, hard_coord, hard_px_coord, hard_valid_img = self.image_sampler.get_neg(pred_x, pred_y)
                else:
                    hard, hard_coord, hard_px_coord, hard_valid_img = self.image_sampler.get_pos(pred_x, pred_y)
                
                hard = nd.expand_dims(ImageProcessor.prepare_Image(hard, ctx=self.ctx, bgr=False), axis=0)
                hard_valid = nd.expand_dims(ImageProcessor.prepare_Image(hard_valid_img, size=self.image_sampler.size, ctx=self.ctx, bgr=True), axis=0)
                emb_hard = self.net(hard)
                
                distance = self.distance(emb_pred, emb_hard)
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
