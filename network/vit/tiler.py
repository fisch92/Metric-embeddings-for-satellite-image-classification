from mxnet.gluon import nn
import mxnet as mx
import math
from mxnet import nd
import numpy as np
from mxnet.gluon.nn import Dense, Activation, Conv2D, Conv2DTranspose, Sequential,\
	BatchNorm, LeakyReLU, Flatten, HybridSequential, HybridBlock, Dropout, HybridLambda, InstanceNorm, SELU, LayerNorm, AvgPool3D, PReLU

class Tiler(HybridBlock):
	def __init__(self, img_size, size, in_channel):
		super(Tiler, self).__init__()
		self.size = size

		self.cls_token = self.params.get('cls_token', shape=(1,1,size*size*in_channel), init=mx.init.Normal(1))
		self.positions = self.params.get('position', shape=(1, (img_size // size) **2+1, size*size*in_channel), init=mx.init.Normal(1))

	def hybrid_forward(self, F, x, positions, cls_token):
		num = x.shape[2] // self.size
		x = F.expand_dims(x, axis=1)
		xs = F.split(x, axis=3, num_outputs=num)
		x = F.concat(*xs, dim=1)
		xs = F.split(x, axis=4, num_outputs=num)
		x = F.concat(*xs, dim=1)
		x = x.transpose((0,1,3,4,2))
		x = F.reshape(x, (x.shape[0], x.shape[1], x.shape[2]*x.shape[3]*x.shape[4]))
		cls_token = F.repeat(cls_token, x.shape[0], axis=0)
		x = F.concat(x,cls_token, dim=1) # maybe concat better
		#print(x.shape, positions.shape)
		return x+positions

class DeTiler(HybridBlock):
	def __init__(self, in_channel):
		super(DeTiler, self).__init__()
		self.in_channel = in_channel

	def hybrid_forward(self, F, x):
		x = F.reshape(x, (x.shape[0], x.shape[1], self.in_channel))

		return x