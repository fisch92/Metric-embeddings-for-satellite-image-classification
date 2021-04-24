import numpy as np
import mxnet.ndarray as nd
from sklearn.decomposition import PCA

class Distances():

	def L2_Dist(x, y):
		return nd.square(x-y).sum(axis=0, exclude=True)

	def MSE_Dist(x, y):
		return nd.square(x-y).mean(axis=0, exclude=True)
		
	def L1_Dist(x, y):
		return nd.abs(x-y).mean(axis=0, exclude=True)


def calc_MI(x, y, bins=5):
	if len(x.shape) > 1:
		if x.shape[1] > 3:
			pcax = PCA(n_components=3)
			x = pcax.fit_transform(x)
	else:
		x = np.expand_dims(x, axis=1)
	if len(y.shape) > 1:
		if y.shape[1] > 3:
			pcay = PCA(n_components=3)
			y = pcay.fit_transform(y)
	else:
		y = np.expand_dims(y, axis=1)
	c_x, bins_x = np.histogramdd(x, bins=bins)
	c_y, bins_y = np.histogramdd(y, bins=bins)

	xy = np.concatenate((x, y), axis=1)
	c_xy, bins_xy = np.histogramdd(xy, bins=bins)

	xdigs = np.zeros(x.shape, dtype='int32')
	ydigs = np.zeros(y.shape, dtype='int32')
	xydigs = np.zeros(xy.shape, dtype='int32')
	for dim in range(0, x.shape[1]):
		xdigs[:, dim] = np.digitize(x[:, dim], bins_x[dim][1:-1])
	for dim in range(0, y.shape[1]):
		ydigs[:, dim] = np.digitize(y[:, dim], bins_y[dim][1:-1])
	for dim in range(0, xy.shape[1]):
		xydigs[:, dim] = np.digitize(xy[:, dim], bins_xy[dim][1:-1])

	xydigs, xydigs_idx = np.unique(xydigs, axis=0, return_index=True)
	xdigs = xdigs[xydigs_idx]
	ydigs = ydigs[xydigs_idx]
	mi = 0
	for sample in range(0, xdigs.shape[0]):
		px = c_x[tuple(xdigs[sample])]/x.shape[0]
		py = c_y[tuple(ydigs[sample])]/x.shape[0]
		pxy = c_xy[tuple(ydigs[sample])][tuple(xdigs[sample])]/x.shape[0]
		#print(pxy)
		if pxy > 0:
			mi += pxy * np.log2(pxy/(px*py))

	
	return mi

def numpy_to_tuple(arr):
	return tuple(map(tuple, arr))

def unitTest():
	x = np.array([
		[1,2],
		[1,2], 
		[3,4],
		[3,4],
		])
	y = np.array([
		[1,2],
		[2,2], 
		[3,4],
		[3,4],
		])
	mi = calc_MI(x, y, bins=10)
	print(mi)
