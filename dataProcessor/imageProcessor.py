import cv2
import random
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd


class ImageProcessor():

	'''
		Color correction, resizing and argumentation
	'''

	def prepare_Image(img, ctx=mx.cpu(), size=None, bgr=False, norm=True, mx=True, validation=False):
		#print(img.shape)
		img = img.transpose((1, 2, 0))
		if bgr and img.shape[2] == 3:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		if not size is None and validation and img.shape[2] == 3:
			img = cv2.resize(img, (size, size), cv2.INTER_NEAREST)
		elif not size is None and not validation:
			img = cv2.resize(img, (size, size))

		if random.random() < 0.25:
			img = np.flip(img, axis=0)
		if random.random() < 0.25:
			img = np.flip(img, axis=1)
		if random.random() < 0.25:
			img = np.rot90(img, 1)
		if random.random() < 0.25:
			img = np.rot90(img, 3)

		if norm:
			img = img/127.5-1.0
		if mx:
			img = img.transpose((2, 0, 1))
			return img
		else:
			return img

	def deprepare_Image(img, bgr=False):
		if type(img).__module__ != np.__name__:
			img = img.asnumpy()
		img = img.transpose((1, 2, 0))
		if bgr:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		img = (img+1.0)*127.5
		return img

	def toOpenCV(batch):
		cv_batch = batch.asnumpy()
		cv_batch = (cv_batch + 1.0)*127.5
		cv_batch = cv_batch.transpose((1, 2, 0))
		return cv_batch.astype("uint8")


	def prepare_Images(imgs, ctx=mx.cpu(), size=None, bgr=False, norm=True, mx=True, validation=False):
		out = []
		for img in imgs:
			prepared = ImageProcessor.prepare_Image(img, ctx=ctx, size=size, bgr=bgr, norm=norm, mx=mx, validation=validation)
			out.append(prepared)
		return out


def unitTest():
	from dataProcessor.imageSampler import ImageSampler

	sampler = ImageSampler(256/8, 256/2, 256*10, 256*100, 256, 256)
	images, coords = sampler.get_quadruplet_by_px()

	prep_imgs = ImageProcessor.prepare_Images(images)

if __name__ == '__main__':
	unitTest()
