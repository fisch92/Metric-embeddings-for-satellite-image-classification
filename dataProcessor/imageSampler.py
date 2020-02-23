import random
import cv2 
import numpy as np

from dataProcessor.tiffReader import TiffReader
from dataProcessor.tiffReader import GEOMAP
from dataProcessor.tiffReader import MissingDataError

class ImageSampler():

	def __init__(self, 
		min_pos_dist, 
		max_pos_dist, 
		min_neg_dist, 
		max_neg_dist, 
		size, 
		geomap=GEOMAP.SENTINEL,
		validationmap=GEOMAP.OSM
	):
		self.min_pos_dist = min_pos_dist
		self.max_pos_dist = max_pos_dist
		self.min_neg_dist = min_neg_dist
		self.max_neg_dist = max_neg_dist
		self.size = size

		self.start_lon = 5.6666654+0.3
		self.start_lan = 47.3537047+0.3
		self.end_lon = 15.3333306-0.3
		self.end_lan =  55.7054442-0.3

		self.tiff = TiffReader(geomap)
		self.validation = TiffReader(validationmap)

		start_x, start_y = self.tiff.coord2Px(self.start_lon, self.start_lan)
		self.start_x = start_x
		self.start_y = start_y
		end_x, end_y = self.tiff.coord2Px(self.end_lon, self.end_lan)
		self.end_x = end_x-size
		self.end_y =  end_y-size

	def get_triplet(self, iteration=0):
		padding_end_lon = self.end_lon - self.min_neg_dist
		padding_start_lon = self.start_lon + self.min_neg_dist
		padding_end_lan = self.end_lan - self.min_neg_dist
		padding_start_lan = self.start_lan + self.min_neg_dist
		try:
			pred_lon = random.random()*(padding_end_lon - padding_start_lon) + padding_start_lon
			pred_lan = random.random()*(padding_end_lan - padding_start_lan) + padding_start_lan
			pred = self.tiff.readTileCoord(pred_lon, pred_lan, self.size, self.size)

			pos_lon = pred_lon + (random.random()-0.5) *2* (self.max_pos_dist - self.min_pos_dist) + self.min_pos_dist
			pos_lan = pred_lan + (random.random()-0.5) *2* (self.max_pos_dist - self.min_pos_dist) + self.min_pos_dist
			pos = self.tiff.readTileCoord(pos_lon, pos_lan, self.size, self.size)

			neg_lon = pred_lon + (random.random()-0.5) *2* (self.max_neg_dist - self.min_neg_dist) + self.min_neg_dist
			neg_lan = pred_lan + (random.random()-0.5) *2* (self.max_neg_dist - self.min_neg_dist) + self.min_neg_dist
			neg_lon = np.clip(neg_lon, self.start_lon, self.end_lon)
			neg_lan = np.clip(neg_lan, self.start_lan, self.end_lan)
			neg = self.tiff.readTileCoord(neg_lon, neg_lan, self.size, self.size)
		except MissingDataError as e:
			if iteration < 3:
				return self.get_triplet(iteration=iteration+1)
			raise e

		pred_valid = self.getValidlonlan(pred_lon, pred_lan)
		pos_valid = self.getValidlonlan(pos_lon, pos_lan)
		neg_valid = self.getValidlonlan(neg_lon, neg_lan)
		'''pred = pred.transpose((1, 2, 0))
		pos = pos.transpose((1, 2, 0))
		neg = neg.transpose((1, 2, 0))

		tile = np.concatenate((pred, pos, neg), axis=1)
		cv2.imshow('sample image',tile)
		cv2.waitKey(0)
		cv2.destroyAllWindows()'''

		return (pred, pos, neg), ((pred_lon, pred_lan), (pos_lon, pos_lan), (neg_lon, neg_lan)), (pred_valid, pos_valid, neg_valid)

	def get_triplet_by_px(self, iteration=0):
		padding_end_x = self.end_x - self.min_neg_dist
		padding_start_x = self.start_x + self.min_neg_dist
		padding_end_y = self.end_y - self.min_neg_dist
		padding_start_y = self.start_y + self.min_neg_dist

		try:
			pred_x = random.random()*(padding_end_x - padding_start_x) + padding_start_x
			pred_y = random.random()*(padding_end_y - padding_start_y) + padding_start_y
			pred = self.tiff.readTilePx(pred_x, pred_y, self.size, self.size)

			pos, pos_coord, (pos_x, pos_y), pos_valid = self.get_pos(pred_x, pred_y)
			neg, neg_coord, (neg_x, neg_y), neg_valid = self.get_neg(pred_x, pred_y)
		except MissingDataError as e:
			if iteration < 3:
				return self.get_triplet_by_px(iteration=iteration+1)
			raise e

		pred_valid, pred_coord = self.getValidxy(pred_x, pred_y)

		'''pred = pred.transpose((1, 2, 0))
		pos = pos.transpose((1, 2, 0))
		neg = neg.transpose((1, 2, 0))

		tile = np.concatenate((pred, pos, neg), axis=1)
		cv2.imshow('sample image',tile)
		cv2.waitKey(0)
		cv2.destroyAllWindows()'''


		return (pred, pos, neg), (pred_coord, pos_coord, neg_coord), ((pred_x, pred_y), (pos_x, pos_y), (neg_x, neg_y)), (pred_valid, pos_valid, neg_valid)

	def get_quadruplet_by_px(self, iteration=0):
		padding_end_x = self.end_x - self.min_neg_dist
		padding_start_x = self.start_x + self.min_neg_dist
		padding_end_y = self.end_y - self.min_neg_dist
		padding_start_y = self.start_y + self.min_neg_dist
		try:
			pred_x = random.random()*(padding_end_x - padding_start_x) + padding_start_x
			pred_y = random.random()*(padding_end_y - padding_start_y) + padding_start_y
			pred = self.tiff.readTilePx(pred_x, pred_y, self.size, self.size)

			pos, pos_coord, pos_px_coord, pos_valid = self.get_pos(pred_x, pred_y)
			neg, neg_coord, neg_px_coord, neg_valid = self.get_neg(pred_x, pred_y)

			neg2, neg2_coord, neg2_px_coord, neg2_valid = self.get_neg(pred_x, pred_y)

		except MissingDataError as e:
			if iteration < 3:
				return self.get_quadruplet_by_px(iteration=iteration+1)
			raise e


		pred_valid, pred_coord = self.getValidxy(pred_x, pred_y)

		'''pred = pred.transpose((1, 2, 0))
		pos = pos.transpose((1, 2, 0))
		neg = neg.transpose((1, 2, 0))
		neg2 = neg2.transpose((1, 2, 0))

		tile = np.concatenate((pred, pos, neg, neg2), axis=1)
		cv2.imshow('sample image',tile)
		cv2.waitKey(0)
		cv2.destroyAllWindows()'''

		return (pred, pos, neg, neg2), (pred_coord, pos_coord, neg_coord, neg2_coord), ((pred_x, pred_y), pos_px_coord, neg_px_coord, neg2_px_coord), (pred_valid, pos_valid, neg_valid, neg2_valid)

	def get_neg(self, pred_x, pred_y):
		neg_x = random.random()*(self.end_x - self.start_x) + self.start_x
		neg_y = random.random()*(self.end_y - self.start_y) + self.start_y
		#neg_x = pred_x + (random.random()-0.5)*2 * (self.max_neg_dist - self.min_neg_dist) + self.min_neg_dist
		#neg_y = pred_y + (random.random()-0.5)*2 * (self.max_neg_dist - self.min_neg_dist) + self.min_neg_dist
		neg = self.tiff.readTilePx(neg_x, neg_y, self.size, self.size)

		neg_valid, neg_coord = self.getValidxy(neg_x, neg_y)

		return neg, neg_coord, (neg_x, neg_y), neg_valid

	def get_pos(self, pred_x, pred_y):
		pos_x = pred_x + (random.random()-0.5)*2 * (self.max_pos_dist - self.min_pos_dist) + self.min_pos_dist
		pos_y = pred_y + (random.random()-0.5)*2 * (self.max_pos_dist - self.min_pos_dist) + self.min_pos_dist

		pos = self.tiff.readTilePx(pos_x, pos_y, self.size, self.size)

		pos_valid, pos_coord = self.getValidxy(pos_x, pos_y)

		return pos, pos_coord, (pos_x, pos_y), pos_valid

	def getValidxy(self, x, y):
		src_start_lon, src_start_lan = self.tiff.px2Coord(x, y)
		src_end_lon, src_end_lan = self.tiff.px2Coord(x+self.size, y+self.size)

		start_x, start_y = self.validation.coord2Px(src_start_lon, src_start_lan)
		end_x, end_y = self.validation.coord2Px(src_end_lon, src_end_lan)

		valid = self.validation.readTilePx(start_x, start_y, abs(end_x-start_x), abs(end_y-start_y))
		return valid, (src_start_lon, src_start_lan)

	def getValidlonlan(self, lon, lan):
		x, y = self.tiff.coord2Px(lon, lan)
		return self.getValidxy(x, y)
		

def unitTest():
	sampler = ImageSampler(256/8, 256/2, 256*10, 256*100, 256, 256)
	images, coords,_ = sampler.get_quadruplet_by_px()

if __name__ == '__main__':
	unitTest()