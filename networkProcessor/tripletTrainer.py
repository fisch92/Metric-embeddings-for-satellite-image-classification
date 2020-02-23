import mxnet as mx
import mxnet.ndarray as nd
from utils.math import Distances
from utils.converters import Converters
from tensorboardX import SummaryWriter

from network.resnext import resnext50_32x4d
from dataProcessor.imageProcessor import ImageProcessor
from dataProcessor.imageSampler import ImageSampler
from dataProcessor.batchSampler import BatchSampler
from dataProcessor.batchSampler import MiningTypes
from network.tripletnet import TripletNet
from utils.validation import Validation
from sklearn.neighbors import KNeighborsClassifier

class TripletTrainer():

	def __init__(self, 
		batch_size, 
		image_size, 
		min_pos_dist=64/8, 
		max_pos_dist=64, 
		min_neg_dist=64*5, 
		max_neg_dist=64*10, 
		mining=[MiningTypes.RANDOM_HARD_NEGATIVE], 
		lr=0.1, 
		ctx=mx.gpu(), 
		net=resnext50_32x4d(ctx=mx.gpu()), 
		margin=1.0, 
		validator=KNeighborsClassifier(),
		validation_nb=1000,
		validation_val=0.1
		):
		self.ctx = ctx
		self.batch_size = batch_size
		self.image_size = image_size

		self.tripletNet = TripletNet(ctx=ctx, margin=margin, net=net)

		self.imageSampler = ImageSampler(min_pos_dist, max_pos_dist, min_neg_dist, max_neg_dist, image_size)
		self.batchSampler = BatchSampler(batch_size, self.imageSampler, self.tripletNet.predict, Distances.L2_Dist, mining, random_mining_iterations=3, ctx=ctx)

		self.global_step = 0
		self.logger = SummaryWriter(logdir='./log_tripletNet')
		self.validator = validator
		self.validation_nb = validation_nb
		self.validation_val = validation_val
		

	def train(self, epochs=100, iterations=1000):
		for epoch in range(0, epochs):
			self.do_epoch(iterations=iterations)
			self.tripletNet.save()

	def do_epoch(self, iterations=1000):
		mean_loss = nd.zeros(self.batch_size, ctx=self.ctx)
		for iteration in range(0, iterations):
			images, coords, px_coords, valid = self.batchSampler.getTripletBatch()
			pred, pos, neg = images
			loss = self.tripletNet.train_step(pred, pos, neg)
			mean_loss += loss

			self.logger.add_scalars('TripletLoss', {
				'loss': nd.mean(loss).asscalar()
			}, global_step=self.global_step)

			
			self.global_step += 1

		self.validation(int(self.validation_nb/self.batch_size))
		print('step: ', self.global_step, 'loss: ', nd.mean(mean_loss/iterations).asscalar())

	def validation(self, iterations=20):
		embs = None
		val_imgs = None
		labels = []
		for iteration in range(0, iterations):
			images, coords, px_coords, valid = self.batchSampler.getTripletBatch()
			pred, pos, neg = images
			val_pred, val_pos, val_neg = valid
			emb_pred = self.tripletNet.predict(pred)
			emb_pos = self.tripletNet.predict(pos)
			emb_neg = self.tripletNet.predict(neg)


			if embs is None:
				embs = nd.concat(emb_pred, emb_pos, dim=0)
				val_imgs = nd.concat(val_pred, val_pos, dim=0)
			else:
				embs = nd.concat(embs, emb_pred, dim=0)
				embs = nd.concat(embs, emb_pos, dim=0)
				val_imgs = nd.concat(val_imgs, val_pred, dim=0)
				val_imgs = nd.concat(val_imgs, val_pos, dim=0)

			embs = nd.concat(embs, emb_neg, dim=0)
			val_imgs = nd.concat(val_imgs, val_neg, dim=0)

		for i in range(0, len(embs)):
			labels.append('img')

		split = int(self.validation_val * len(embs))
		self.validator.train(embs[split:], val_imgs[split:])
		accs = self.validator.accurancy(embs[:split], val_imgs[:split])

		for key, acc in accs.items():
			self.logger.add_scalars('Validation', {
				key: acc
			}, global_step=self.global_step)

		print('step: ', self.global_step, 'acc: ', accs)

		self.logger.add_embedding(embs[:int(len(embs)/2)], label_img=val_imgs[:int(len(embs)/2)], global_step=self.global_step, tag='imgs', metadata=labels[:int(len(embs)/2)])


def unitTest():
	trainer = TripletTrainer(2, 256, ctx=mx.cpu())
	trainer.train(epochs=2, iterations=5)
