import skfuzzy as fuzz

class FuzzyCMeans():
	def __init__(self, n_clusters=5, maxiter=1000, error=0.005):
		self.n_clusters = n_clusters
		self.maxiter = maxiter
		self.error = error
		self.cntr = None

	def fit_predict(self, data):
		data = data.transpose((1,0))
		#if self.cntr is None:
		self.cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data, self.n_clusters, 2, error=self.error, maxiter=self.maxiter, init=None)
		'''else:
			u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(data, cntr, 2, error=self.error, maxiter=self.maxiter)'''
		return u.transpose((1,0))

	def fit(self, data):
		data = data.transpose((1,0))
		#if self.cntr is None:
		self.cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data, self.n_clusters, 2, error=self.error, maxiter=self.maxiter, init=None)
		'''else:
			u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(data, cntr, 2, error=self.error, maxiter=self.maxiter)'''
		return u.transpose((1,0))

	def predict(self, data):
		data = data.transpose((1,0))
		
		u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(data, self.cntr, 2, error=self.error, maxiter=self.maxiter)
		return u.transpose((1,0))