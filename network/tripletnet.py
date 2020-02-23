import os
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon
from mxnet import autograd
from network.resnext import resnext50_32x4d
from utils.math import Distances
from mxnet.gluon.loss import L2Loss
from mxnet import lr_scheduler


class TripletNet():

	def __init__(self, name='tripletnet', ctx=mx.gpu(), loss=Distances.L2_Dist, learning_rate=0.02, margin=0.5, net=resnext50_32x4d(ctx=mx.gpu())):
		self.name = name
		self.ctx = ctx

		self.net = net
		self.load()
		
		scheduler = lr_scheduler.FactorScheduler(base_lr=learning_rate, step=100, factor=0.9)
		self.trainer = gluon.Trainer(self.net.collect_params(), mx.optimizer.Adam(learning_rate=learning_rate, lr_scheduler=scheduler))
		self.loss = loss
		
		self.margin = margin
		

	def predict(self, batch):
		return self.net(batch)

	def train_step(self, pred_batch, pos_batch, neg_batch):
		with autograd.record():
			emb_pred = self.net(pred_batch)
			emb_pos = self.net(pos_batch)
			emb_neg = self.net(neg_batch)

			loss = self.loss(emb_pred, emb_pos) - self.loss(emb_pred, emb_neg) + self.margin*nd.ones(pred_batch.shape[0], ctx=self.ctx)
			loss = nd.relu(loss)

		loss.backward()
		self.trainer.step(pred_batch.shape[0])

		return loss

	def save(self):
		self.net.save_parameters(self.name)

	def load(self, init=mx.init.Xavier()):
		if os.path.exists(self.name):
			self.net.load_parameters(self.name, ctx=self.ctx)
		else:
			self.net.initialize(init=init, ctx=self.ctx)