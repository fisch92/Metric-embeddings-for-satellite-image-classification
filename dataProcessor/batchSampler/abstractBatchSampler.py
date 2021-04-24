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


class AbstractBatchSampler():

	def __init__(self, batch_size, image_sampler, net=None, ctx=mx.cpu(), channels=3):
		self.batch_size = batch_size
		self.image_sampler = image_sampler
		self.ctx = ctx
		self.net = net
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
		return NotImplemented


	def reset(self):
		self.image_sampler.reset()
