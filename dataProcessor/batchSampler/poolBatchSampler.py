import cv2
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd

from enum import Enum
from utils.labelProcessor import LabelProcessor
from dataProcessor.imageProcessor import ImageProcessor
from dataProcessor.imageSampler import ImageSampler
from dataProcessor.miningTypes import MiningTypes
from dataProcessor.tiffReader import GEOMAP


class PoolBatchSampler():

	def __init__(self, batch_size, image_sampler, ctx=mx.cpu(), channels=3):
		self.batch_size = batch_size
		self.image_sampler = image_sampler
		self.ctx = ctx
		self.channels = channels

	def getBatch(self, validation=False):
		pred_batches = nd.zeros((self.batch_size, self.channels, self.image_sampler.size, self.image_sampler.size), ctx=self.ctx)
		pred_coords = nd.zeros((self.batch_size, 2), ctx=self.ctx)
		pred_px_coords = nd.zeros((self.batch_size, 2), ctx=self.ctx)
		pred_valid_batches = nd.zeros((self.batch_size, self.channels, self.image_sampler.size, self.image_sampler.size), ctx=self.ctx)

		for batch in range(0, self.batch_size):
			image, coord, px_coord, valid_image = self.image_sampler.getrandomPool(train=not validation)

			prep_img = ImageProcessor.prepare_Image(image, ctx=self.ctx, bgr=False)
			prep_valid_img = ImageProcessor.prepare_Image(valid_image, size=self.image_sampler.size, ctx=self.ctx, bgr=True, validation=True)

			pred_batches[batch] = nd.array(prep_img, ctx=self.ctx)
			pred_coords[batch] = coord
			pred_px_coords[batch] = px_coord
			pred_valid_batches[batch] = nd.array(prep_valid_img, ctx=self.ctx)

			#self.labelProcessor = LabelProcessor(self.image_sampler.size, GEOMAP.CLC)
			#label = self.labelProcessor.getLabels(prep_valid_img, processed=True)
			#idx = np.argmax(label)
			#print('batch', idx)
			

		return pred_batches, pred_coords, pred_px_coords, pred_valid_batches
