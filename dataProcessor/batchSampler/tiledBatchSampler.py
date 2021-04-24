import cv2
import mxnet as mx
import mxnet.ndarray as nd

from enum import Enum
from dataProcessor.imageProcessor import ImageProcessor
from dataProcessor.imageSampler import ImageSampler
from dataProcessor.miningTypes import MiningTypes


class TiledBatchSampler():

	def __init__(self, image_sampler, ctx=mx.cpu(), tiles=4):
		self.image_sampler = image_sampler
		self.ctx = ctx
		self.tiles = tiles

	def getBatch(self, validation=False, train=False):
		pred_batches = nd.zeros((self.tiles*self.tiles, 3, self.image_sampler.size, self.image_sampler.size), ctx=self.ctx)
		pred_valid_batches = nd.zeros((self.tiles*self.tiles, 3, self.image_sampler.size, self.image_sampler.size), ctx=self.ctx)
		
		counter = 0
		images, valid_images = self.image_sampler.get_tiled_by_px(cols=self.tiles, rows=self.tiles)

		for ctile in range(0, len(images)):
			#print(valid_images[ctile].shape)
			prep_imgs = ImageProcessor.prepare_Image(images[ctile], ctx=self.ctx, bgr=False)
			prep_valid_imgs = ImageProcessor.prepare_Image(valid_images[ctile], size=self.image_sampler.size, ctx=self.ctx, bgr=True, validation=True)

			pred_batches[counter] = prep_imgs
			pred_valid_batches[counter] = prep_valid_imgs

			counter += 1

		
		return pred_batches, pred_valid_batches
