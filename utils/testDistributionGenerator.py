import numpy as np

class TestDistributionGenerator():
	
	def createBlob(samples=100, center=[0,0], std=1, dims=2):
		data = np.random.normal(loc=center, scale=std, size=(samples, dims))
		#labels[:, cluster['label']] += 1./(np.sqrt(2.*np.pi)*std)*np.exp(-np.sum(np.square(samples-center)/(2*np.square(std)), axis=1))
		#print(center.shape)
		labels = 1./(np.sqrt(2.*np.pi)*std)*np.exp(-np.sum(np.square(data-center)/(2*np.square(std)), axis=1))
			
		return data, labels

	def createCircle(samples=100, center=[0,0], std=1, r=[1.0, 0.5], dims=2):
		data = np.random.normal(loc=0, scale=1,   size=(samples, dims))
		noise = np.random.normal(loc=0, scale=std, size=(samples, dims))
		dist = np.sqrt(np.square(data).sum(axis=1))
		dist = np.expand_dims(dist, axis=1)
		out = data/dist*(r+noise)+center
		labels = 1./(np.sqrt(2.*np.pi)*std)*np.exp(-np.sum(np.square(noise)/(2*np.square(std)), axis=1))
		return out, labels

	def createLine(samples=100, start=[0,0], end=[0,0], std=1, dims=2):
		data = np.random.uniform(start, end, size=(samples, dims))
		noise = np.random.normal(loc=0, scale=std, size=(samples, dims))
		dist = np.sqrt(np.square(data).sum(axis=1))
		dist = np.expand_dims(dist, axis=1)
		out = data+noise
		labels = 1./(np.sqrt(2.*np.pi)*std)*np.exp(-np.sum(np.square(noise)/(2*np.square(std)), axis=1))
		return out, labels

	def createNoise(samples=100, min=0, max=1, dims=2):
		data = np.random.uniform(min, max, size=(samples, dims))
		return data, np.random.uniform(0, 1, size=(samples))