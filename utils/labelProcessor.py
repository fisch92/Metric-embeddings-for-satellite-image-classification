import cv2
import numpy as np

from dataProcessor.tiffReader import GEOMAP
from validation.osmClasses import OSMClasses
from validation.clcClasses import CLCClasses
from dataProcessor.imageProcessor import ImageProcessor

class LabelProcessor():

		def __init__(self, size, validation_map=GEOMAP.OSM):
			self.size = size
			if validation_map == GEOMAP.OSM:
				self.colorToLabel = OSMClasses.getLabels
				self.nb_labels = 13
			elif validation_map == GEOMAP.CLC:
				self.colorToLabel = CLCClasses.getLabels
				self.nb_labels = 45
			elif validation_map == GEOMAP.TILE2VEC:
				self.colorToLabel = self.getColor
				self.nb_labels = 27
		
		def getColor(self, unique_values):
			return (unique_values[0]/255.0*self.nb_labels).astype('uint8'), np.zeros(len(unique_values[0]))

		def getLabels(self, class_img, processed=False):
			
			if processed:
				rgb_class_img = ImageProcessor.deprepare_Image(class_img)
			else:
				rgb_class_img = ImageProcessor.prepare_Image(class_img, norm=False, mx=False, bgr=True, size=self.size, validation=True)
			
			tmp_img = rgb_class_img.reshape((rgb_class_img.shape[0] * rgb_class_img.shape[1], rgb_class_img.shape[2]))
			unique_values, unique_idx, unique_counts = np.unique(tmp_img, axis=0, return_index=True, return_counts=True)
			#count_sort_ind = np.argsort(unique_counts)
			#unique_values = unique_values[count_sort_ind]
			#unique_idx = unique_idx[count_sort_ind]
			#unique_counts = unique_counts[count_sort_ind]
			labels = np.zeros(self.nb_labels)
			idxs, dists = self.colorToLabel(unique_values)

			for value in range(0, len(unique_values)):
				#print(tmp_img.shape, unique_counts[value])
				#
				if dists[value] < 15:
					labels[idxs[value]] += unique_counts[value]/len(tmp_img)


			return labels
