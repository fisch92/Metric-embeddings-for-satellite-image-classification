import cv2
import mxnet as mx
import mxnet.ndarray as nd

from dataProcessor.imageSampler import ImageSampler

class ImageProcessor():

	def prepare_Image(img, ctx=mx.cpu(), size=None):
		img = img.transpose((1, 2, 0))
		bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
		if not size is None:
			bgr = cv2.resize(bgr, (size, size))
		bgr = bgr.transpose((2, 0, 1))

		bgr = bgr/127.5-1.0
		return nd.array(bgr, ctx=ctx)

	def toOpenCV(batch):
		cv_batch = batch.asnumpy()
		cv_batch = (cv_batch + 1.0)*127.5
		cv_batch = cv_batch.transpose((1, 2, 0))
		return cv_batch.astype("uint8")


	def prepare_Images(imgs, ctx=mx.cpu(), size=None):
		out = []
		for img in imgs:
			prepared = ImageProcessor.prepare_Image(img, ctx=ctx, size=size)
			out.append(prepared)
		return out


def unitTest():
	sampler = ImageSampler(256/8, 256/2, 256*10, 256*100, 256, 256)
	images, coords = sampler.get_quadruplet_by_px()

	prep_imgs = ImageProcessor.prepare_Images(images)

if __name__ == '__main__':
	unitTest()