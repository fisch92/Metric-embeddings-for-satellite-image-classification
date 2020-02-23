import cv2
import mxnet as mx
import numpy as np
import mxnet.ndarray as nd
from enum import Enum

from utils.math import Distances
import dataProcessor.batchSampler as batchSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

class OSMClasses(Enum):
	ARABLE_LAND = 0
	OPEN_SPACE = 1
	FOREST = 2
	PASTURES = 3
	SHRUB = 4
	WETLANDS = 5
	PERMANENT_CROPS = 6
	URBAN_FABRIC = 7
	INDUSTRIAL = 8
	ARTIFICIAL = 9
	WATER = 10
	MINE = 11
	COSTAL_WETLANDS = 12
	OTHER = 13

	def getLabel(color):
		osm_type = None
		if(color.tolist() == [250., 250., 160.]):
			osm_type =  OSMClasses.ARABLE_LAND
		elif(color.tolist() == [250., 250., 250.]):
			osm_type =  OSMClasses.OPEN_SPACE
		elif(color.tolist() == [70., 250.,   0.]):
			osm_type =  OSMClasses.FOREST
		elif(color.tolist() == [220., 220.,  70.]):
			osm_type =  OSMClasses.PASTURES
		elif(color.tolist() == [200., 240.,  70.]):
			osm_type =  OSMClasses.SHRUB
		elif(color.tolist() == [160., 160., 250.]):
			osm_type =  OSMClasses.WETLANDS
		elif(color.tolist() == [230., 130.,   0.]):
			osm_type =  OSMClasses.PERMANENT_CROPS
		elif(color.tolist() == [220.,   0.,  70.]):
			osm_type =  OSMClasses.URBAN_FABRIC
		elif(color.tolist() == [200.,  70., 240.]):
			osm_type =  OSMClasses.INDUSTRIAL
		elif(color.tolist() == [250., 160., 250.]):
			osm_type =  OSMClasses.ARTIFICIAL
		elif(color.tolist() == [0., 200., 240.]):
			osm_type =  OSMClasses.WATER
		elif(color.tolist() == [160. ,  0., 200.]):
			osm_type =  OSMClasses.MINE
		elif(color.tolist() == [220., 220., 220.]):
			osm_type =  OSMClasses.COSTAL_WETLANDS
		else:
			osm_type =  OSMClasses.OTHER

		return osm_type.value



class Validation():

	def __init__(self, classifiers={'knn': KNeighborsClassifier()}, distance=Distances.L2_Dist, ctx=mx.cpu()):
		self.ctx = ctx
		self.distance = distance
		self.classifiers = classifiers

	def train(self, embeddings, class_imgs):
		data, labels = Validation.getTrainData(embeddings, class_imgs)
		for key, classifier in self.classifiers.items():
			classifier.fit(data, labels)


	def accurancy(self, embeddings, class_imgs):

		accs = {}
		data, labels = Validation.getTrainData(embeddings, class_imgs)

		'''predicted_labels = self.classifier.predict(data)
		similarity = np.where(predicted_labels == labels, np.ones(labels.shape), np.zeros(labels.shape))
		acc = similarity.sum()/len(similarity)'''
		for key, classifier in self.classifiers.items():
			acc = classifier.score(data, labels)
			accs[key] = acc
		return accs
		
		


	def getTrainData(embeddings, class_imgs):
		data = None
		labels = None
		np_class_imgs = class_imgs.asnumpy()
		np_embeddings = embeddings.asnumpy()

		for np_class_img in range(0, len(np_class_imgs)):
			vis_img = np_class_imgs[np_class_img].transpose(1, 2, 0)
			
			tmp_img = vis_img.reshape((vis_img.shape[0] * vis_img.shape[1], 3))
			tmp_img = (tmp_img + 1)*127.5
			tmp_img = np.floor(tmp_img/10.0)*10.0
			unique_values, unique_idx, unique_counts = np.unique(tmp_img, axis=(0), return_index=True, return_counts=True)
			
			#print(unique_values)
			if unique_counts.max() > vis_img.shape[0] * vis_img.shape[1] * 0.5:
				idx = np.argmax(unique_counts)
				
				if data is None:
					data = np.expand_dims(np_embeddings[np_class_img], axis=0)
					labels = np.expand_dims(OSMClasses.getLabel(unique_values[idx]), axis=0)
				else:
					data = np.concatenate((data, np.expand_dims(np_embeddings[np_class_img], axis=0)), axis=0)
					labels = np.concatenate((labels, np.expand_dims(OSMClasses.getLabel(unique_values[idx]), axis=0)), axis=0)

		return data, labels




def unitTest():
	def mean(img):
		return img.mean(axis=(0,1), exclude=True)

	images, coords, px_coords, valid = batchSampler.unitTest(32, 64)
	emb_pred, emb_pos, emb_neg = images
	class_pred, class_pos, class_neg = valid
	embs = nd.concat(mean(emb_pred), mean(emb_pos), dim=0)
	class_imgs = nd.concat(class_pred, class_pos, dim=0)

	embs = nd.concat(embs, mean(emb_neg), dim=0)
	class_imgs = nd.concat(class_imgs, class_neg, dim=0)

	validator = Validation(embs[:int(len(embs)/2)], class_imgs[:int(len(embs)/2)])
	validator.train()
	acc = validator.accurancy(embs[int(len(embs)/2):], class_imgs[int(len(embs)/2):])
	print(acc)
