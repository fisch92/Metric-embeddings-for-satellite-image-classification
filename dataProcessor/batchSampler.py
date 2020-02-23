import cv2
import mxnet as mx
import mxnet.ndarray as nd

from enum import Enum
from dataProcessor.imageProcessor import ImageProcessor
from dataProcessor.imageSampler import ImageSampler

class MiningTypes(Enum):
	HARD_NEGATIVE = 0
	HARD_POSITIVE = 1
	RANDOM_HARD_NEGATIVE = 2
	RANDOM_HARD_POSITIVE = 3


class BatchSampler():

	def __init__(self, batch_size, image_sampler, net=None, distance=None, mining=[], random_mining_iterations=5, ctx=mx.cpu()):
		self.batch_size = batch_size
		self.image_sampler = image_sampler
		self.ctx = ctx
		self.net = net
		self.distance = distance
		self.mining = mining
		self.random_mining_iterations = random_mining_iterations

	def getTripletBatch(self):
		pred_batches = nd.zeros((self.batch_size, 3, self.image_sampler.size, self.image_sampler.size), ctx=self.ctx)
		pos_batches = nd.zeros((self.batch_size, 3, self.image_sampler.size, self.image_sampler.size), ctx=self.ctx)
		neg_batches = nd.zeros((self.batch_size, 3, self.image_sampler.size, self.image_sampler.size), ctx=self.ctx)

		pred_coords = nd.zeros((self.batch_size, 2), ctx=self.ctx)
		pos_coords = nd.zeros((self.batch_size, 2), ctx=self.ctx)
		neg_coords = nd.zeros((self.batch_size, 2), ctx=self.ctx)

		pred_px_coords = nd.zeros((self.batch_size, 2), ctx=self.ctx)
		pos_px_coords = nd.zeros((self.batch_size, 2), ctx=self.ctx)
		neg_px_coords = nd.zeros((self.batch_size, 2), ctx=self.ctx)

		pred_valid_batches = nd.zeros((self.batch_size, 3, self.image_sampler.size, self.image_sampler.size), ctx=self.ctx)
		pos_valid_batches = nd.zeros((self.batch_size, 3, self.image_sampler.size, self.image_sampler.size), ctx=self.ctx)
		neg_valid_batches = nd.zeros((self.batch_size, 3, self.image_sampler.size, self.image_sampler.size), ctx=self.ctx)

		for batch in range(0, self.batch_size):
			images, coords, px_coords, valid_images = self.image_sampler.get_triplet_by_px()

			prep_imgs = ImageProcessor.prepare_Images(images, ctx=self.ctx)
			prep_valid_imgs = ImageProcessor.prepare_Images(valid_images, size=self.image_sampler.size, ctx=self.ctx)

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
			pos_valid_batches = pos_valid_batches[arg_neg_batches]

		if MiningTypes.RANDOM_HARD_NEGATIVE in self.mining:
			neg_batches, neg_coords, neg_px_coords, neg_valid_batches = self.doRandomHardMining(pred_batches, pred_px_coords)
		if MiningTypes.RANDOM_HARD_POSITIVE in self.mining:
			pos_batches, pos_coords, pos_px_coords, pos_valid_batches = self.doRandomHardMining(pred_batches, pred_px_coords, isNeg=False)

		return (pred_batches, pos_batches, neg_batches), (pred_coords, pos_coords, neg_coords), (pred_px_coords, pos_px_coords, neg_px_coords), (pred_valid_batches, pos_valid_batches, neg_valid_batches)

	def getQuadrupletBatch(self):
		pred_batches = nd.zeros((self.batch_size, 3, self.image_sampler.size, self.image_sampler.size), ctx=self.ctx)
		pos_batches = nd.zeros((self.batch_size, 3, self.image_sampler.size, self.image_sampler.size), ctx=self.ctx)
		neg_batches = nd.zeros((self.batch_size, 3, self.image_sampler.size, self.image_sampler.size), ctx=self.ctx)
		neg2_batches = nd.zeros((self.batch_size, 3, self.image_sampler.size, self.image_sampler.size), ctx=self.ctx)

		pred_coords = nd.zeros((self.batch_size, 2), ctx=self.ctx)
		pos_coords = nd.zeros((self.batch_size, 2), ctx=self.ctx)
		neg_coords = nd.zeros((self.batch_size, 2), ctx=self.ctx)
		neg2_coords = nd.zeros((self.batch_size, 2), ctx=self.ctx)

		pred_px_coords = nd.zeros((self.batch_size, 2), ctx=self.ctx)
		pos_px_coords = nd.zeros((self.batch_size, 2), ctx=self.ctx)
		neg_px_coords = nd.zeros((self.batch_size, 2), ctx=self.ctx)
		neg2_px_coords = nd.zeros((self.batch_size, 2), ctx=self.ctx)

		pred_valid_batches = nd.zeros((self.batch_size, 3, self.image_sampler.size, self.image_sampler.size), ctx=self.ctx)
		pos_valid_batches = nd.zeros((self.batch_size, 3, self.image_sampler.size, self.image_sampler.size), ctx=self.ctx)
		neg_valid_batches = nd.zeros((self.batch_size, 3, self.image_sampler.size, self.image_sampler.size), ctx=self.ctx)
		neg2_valid_batches = nd.zeros((self.batch_size, 3, self.image_sampler.size, self.image_sampler.size), ctx=self.ctx)

		for batch in range(0, self.batch_size):
			images, coords, px_coords, valid_images = self.image_sampler.get_quadruplet_by_px()

			prep_imgs = ImageProcessor.prepare_Images(images, ctx=self.ctx)
			prep_valid_imgs = ImageProcessor.prepare_Images(valid_images, size=self.image_sampler.size, ctx=self.ctx)

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
		if MiningTypes.HARD_POSITIVE in self.mining:
			arg_pos_batches = self.argDoHardMining(pred_batches, pos_batches, minimize=False)
			pos_batches = pos_batches[arg_pos_batches]
			pos_coords = pos_coords[arg_pos_batches]
			pos_valid_batches = pos_valid_batches[arg_neg_batches]

		if MiningTypes.RANDOM_HARD_NEGATIVE in self.mining:
			neg_batches, neg_coords, neg_px_coords, neg_valid_batches = self.doRandomHardMining(pred_batches, pred_px_coords)
		if MiningTypes.RANDOM_HARD_POSITIVE in self.mining:
			pos_batches, pos_coords, pos_px_coords, pos_valid_batches = self.doRandomHardMining(pred_batches, pred_px_coords, isNeg=False)

		return (pred_batches, pos_batches, neg_batches, neg2_batches), \
		(pred_coords, pos_coords, neg_coords, neg2_coords), \
		(pred_px_coords, pos_px_coords, neg_px_coords, neg2_px_coords), \
		(pred_valid_batches, pos_valid_batches, neg_valid_batches, neg2_valid_batches)

	def argDoHardMining(self, pred, hard, minimize=True):

		out_hard = nd.zeros(self.batch_size, ctx=self.ctx)
		for batch in range(0, len(pred)):
			single_pred = nd.expand_dims(pred[batch], axis=0)
			emb_pred = self.net(single_pred)
			emb_hard = self.net(hard)
			distances = self.distance(emb_pred, emb_hard)
			sort = nd.argsort(distances)
			if minimize:
				out_hard[batch] = sort[0]
			else:
				out_hard[batch] = sort[-1]

		return out_hard

	def doRandomHardMining(self, pred, pred_px_coords, isNeg=True):

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
				
				hard = nd.expand_dims(ImageProcessor.prepare_Image(hard, ctx=self.ctx), axis=0)
				hard_valid = nd.expand_dims(ImageProcessor.prepare_Image(hard_valid_img, size=self.image_sampler.size, ctx=self.ctx), axis=0)
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