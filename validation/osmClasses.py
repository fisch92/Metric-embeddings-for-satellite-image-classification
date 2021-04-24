import numpy as np
from enum import Enum
from sklearn.neighbors import KDTree


class OSMClasses():
	"""ARABLE_LAND = 0
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
	COSTAL_WETLANDS = 12"""

	labels = np.array(
		[[255., 255., 168.],
		[230., 230., 230.],
		[77., 255.,   0.],
		[230., 230.,  77.],
		[204., 242.,  77.],
		[166., 166., 255.],
		[230., 128.,   0.],
		[230.,   0.,  77.],
		[204.,  77., 242.],
		[255., 166., 255.],
		[0., 204., 242.],
		[166. ,  0., 204.],
		[230., 230., 255.]]
	)
	tree = KDTree(labels, metric='l1')

	def getLabel(color):
		dist, ind = OSMClasses.tree.query(np.expand_dims(color, axis=0), k=1)
		return ind[0][0], dist

	def getLabels(colors):
		#osm_type = np.argmin(np.abs(color-CLCClasses.labelColors()).mean(axis=1))
		dist, ind = OSMClasses.tree.query(colors, k=1)
		return ind[:, 0], dist
		"""osm_type = OSMClasses.ARABLE_LAND
		osm_dist = np.abs(np.array([255., 255., 168.])-color).mean()
		
		if(np.abs(np.array([230., 230., 230.])-color).mean() < osm_dist):
			osm_type =  OSMClasses.OPEN_SPACE
			osm_dist = np.abs(np.array([230., 230., 230.])-color).mean()
		if(np.abs(np.array([77., 255.,   0.])-color).mean() < osm_dist):
			osm_type =  OSMClasses.FOREST
			osm_dist = np.abs(np.array([77., 255.,   0.])-color).mean()
		if(np.abs(np.array([230., 230.,  77.])-color).mean() < osm_dist):
			osm_type =  OSMClasses.PASTURES
			osm_dist = np.abs(np.array([230., 230.,  77.])-color).mean()
		if(np.abs(np.array([204., 242.,  77.])-color).mean() < osm_dist):
			osm_type =  OSMClasses.SHRUB
			osm_dist = np.abs(np.array([204., 242.,  77.])-color).mean()
		if(np.abs(np.array([166., 166., 255.])-color).mean() < osm_dist):
			osm_type =  OSMClasses.WETLANDS
			osm_dist = np.abs(np.array([166., 166., 255.])-color).mean()
		if(np.abs(np.array([230., 128.,   0.])-color).mean() < osm_dist):
			osm_type =  OSMClasses.PERMANENT_CROPS
			osm_dist = np.abs(np.array([230., 128.,   0.])-color).mean()
		if(np.abs(np.array([230.,   0.,  77.])-color).mean() < osm_dist):
			osm_type =  OSMClasses.URBAN_FABRIC
			osm_dist = np.abs(np.array([230.,   0.,  77.])-color).mean()
		if(np.abs(np.array([204.,  77., 242.])-color).mean() < osm_dist):
			osm_type =  OSMClasses.INDUSTRIAL
			osm_dist = np.abs(np.array([204.,  77., 242.])-color).mean()
		if(np.abs(np.array([255., 166., 255.])-color).mean() < osm_dist):
			osm_type =  OSMClasses.ARTIFICIAL
			osm_dist = np.abs(np.array([255., 166., 255.])-color).mean()
		if(np.abs(np.array([0., 204., 242.])-color).mean() < osm_dist):
			osm_type =  OSMClasses.WATER
			osm_dist = np.abs(np.array([0., 204., 242.])-color).mean()
		if(np.abs(np.array([166. ,  0., 204.])-color).mean() < osm_dist):
			osm_type =  OSMClasses.MINE
			osm_dist = np.abs(np.array([166. ,  0., 204.])-color).mean()
		if(np.abs(np.array([230., 230., 255.])-color).mean() < osm_dist):
			osm_type =  OSMClasses.COSTAL_WETLANDS
			osm_dist = np.abs(np.array([230., 230., 255.])-color).mean()

		return osm_type.value"""
		
	def getColor(osm_type):
		return OSMClasses.labels[osm_type]
		"""label = OSMClasses(label)
		if label == OSMClasses.ARABLE_LAND:
			return np.array([255., 255., 168.])
		if label ==  OSMClasses.OPEN_SPACE:
			return np.array([230., 230., 230.])
		if label ==  OSMClasses.FOREST:
			return np.array([77., 255.,   0.])
		if label ==  OSMClasses.PASTURES:
			return np.array([230., 230.,  77.])
		if label ==  OSMClasses.SHRUB:
			return np.array([204., 242.,  77.])
		if label == OSMClasses.WETLANDS:
			return np.array([166., 166., 255.])
		if label ==  OSMClasses.PERMANENT_CROPS:
			return np.array([230., 128.,   0.])
		if label ==  OSMClasses.URBAN_FABRIC:
			return np.array([230.,   0.,  77.])
		if label ==  OSMClasses.INDUSTRIAL:
			return np.array([204.,  77., 242.])
		if label ==  OSMClasses.ARTIFICIAL:
			return np.array([255., 166., 255.])
		if label ==  OSMClasses.WATER:
			return np.array([0., 204., 242.])
		if label ==  OSMClasses.MINE:
			return np.array([166. ,  0., 204.])
		if label ==  OSMClasses.COSTAL_WETLANDS:
			return np.array([230., 230., 255.])"""
