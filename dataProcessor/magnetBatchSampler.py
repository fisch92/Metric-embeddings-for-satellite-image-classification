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


class MagNetBatchSampler():

	def __init__(self, batch_size, image_sampler, net=None, distance=None, mining=[], random_mining_iterations=5, ctx=mx.cpu(), channels=3):
		self.batch_size = batch_size
		self.image_sampler = image_sampler
		self.ctx = ctx
		self.net = net
		self.distance = distance
		self.mining = mining
		self.random_mining_iterations = random_mining_iterations
		self.channels = channels
		self.batches = Queue()
		for worker in range(0,8):
			fillDeamon = Process(target=self.fillDeamon, args=(self.batches,))
			fillDeamon.daemon = True
			fillDeamon.start()

	def fillDeamon(self, queue):
		while True:
			while queue.qsize() < 32:
				self.fill(queue)
			time.sleep(1)


	def fill(self, queue):
		queue.put(self.prepareBatch())

	def take(self):
		return self.batches.get()

	def getBatch(self, validation=False, file=None):
		

		if self.batches.qsize() > 0:
			return self.take()
		else:
			time.sleep(1)
			return self.getBatch(validation=validation, file=file)

	def prepareBatch(self, validation=False, train=False):
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


	def reset(self):
		self.image_sampler.reset()

def unitTest(batches=8, img_size=256):

	def mean(img):
		return img.mean(axis=(0,1), exclude=True)

	def l2_distance(a, b):
		return nd.sqrt(nd.square(a-b).sum(axis=0, exclude=True))

	sampler = ImageSampler(img_size/8, img_size/2, img_size*10, img_size*100, img_size)
	batch_sampler = BatchSampler(batches, sampler, net=mean, distance=l2_distance, mining=[MiningTypes.RANDOM_HARD_NEGATIVE, MiningTypes.RANDOM_HARD_POSITIVE])
	images, coords, px_coords, valid = batch_sampler.getTripletBatch()

	return images, coords, px_coords, valid

if __name__ == '__main__':
	unitTest()
