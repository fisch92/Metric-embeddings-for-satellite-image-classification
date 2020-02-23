import cv2
import numpy as np

from dataProcessor.imageProcessor import ImageProcessor
import dataProcessor.batchSampler as batchSampler

class Visualizer():

	def showSamples(batches, valid):
		out_image = None
		for row in range(0, len(batches)):
			temp_row = None
			for col in range(0, len(batches[row])):
				cv_img = ImageProcessor.toOpenCV(batches[row][col])
				valid_img = ImageProcessor.toOpenCV(valid[row][col])
				cv_img = cv2.addWeighted(cv_img, 0.7, valid_img, 0.3, 0)
				if temp_row is None:
					temp_row = cv_img
				else:
					temp_row = np.concatenate((temp_row, cv_img), axis=1)

			if out_image is None:
				out_image = temp_row
			else:
				out_image = np.concatenate((out_image, temp_row), axis=0)

		cv2.imshow('sample image',out_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


def unitTest():
	images, coords, px_coords, valid = batchSampler.unitTest()
	Visualizer.showSamples(images, valid)

if __name__ == '__main__':
	unitTest()