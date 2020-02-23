import mxnet.ndarray as nd

class Distances():

	def L2_Dist(x, y):
		return nd.sqrt(nd.square(x-y).sum(axis=0, exclude=True))
		
	def L1_Dist(x, y):
		return nd.abs(x-y).mean(axis=0, exclude=True)