import mxnet as mx
import mxnet.ndarray as nd
from utils.math import Distances
from utils.converters import Converters
from tensorboardX import SummaryWriter

from dataProcessor.tiffReader import GEOMAP
from networkProcessor.trainer import Trainer
from network.resnext import resnext50_32x4d
from dataProcessor.imageProcessor import ImageProcessor
from dataProcessor.imageSampler import ImageSampler
from dataProcessor.supervisedImageSampler import SupervisedImageSampler
from dataProcessor.batchSampler.magnetBatchSampler import MagNetBatchSampler
from dataProcessor.miningTypes import MiningTypes
from network.magnet import MagNet


class MagNetTrainer(Trainer):

	def __init__(self, 
		batch_size, 
		image_size, 
		min_pos_dist=64/8, 
		max_pos_dist=64, 
		min_neg_dist=64*5, 
		max_neg_dist=64*10, 
		mining=[], 
		lr=0.1, 
		ctx=[mx.gpu()], 
		net=resnext50_32x4d(ctx=mx.gpu()), 
		margin=1.0,
		margin2=0.5,
		validation_map='osm',
		random_reset=0.0,
		load=True,
		singleClassTreshold=0.0,
		supervised=False,
		alt_loss=False,
		name=None,
		**kargs
		):
		if name is None:
			magnet = MagNet(ctx=ctx, margin=margin, margin2=margin2, net=net, load=load)
		else:
			magnet = MagNet(ctx=ctx, margin=margin, margin2=margin2, net=net, load=load, name=name)
		
		if supervised:
			image_sampler = SupervisedImageSampler(image_size, validationmap=validation_map, singleClassTreshold=singleClassTreshold)
		else:
			image_sampler = ImageSampler(min_pos_dist, max_pos_dist, min_neg_dist, max_neg_dist, image_size, validationmap=validation_map, random_reset=random_reset, singleClassTreshold=singleClassTreshold)
		batchSampler = MagNetBatchSampler(batch_size=batch_size, image_sampler=image_sampler, net=magnet.predict, ctx=ctx[0])
		
		super().__init__(
			imageSampler=image_sampler,
			batchSampler=batchSampler, 
			net=magnet,
			name='MagNet',
			batch_size=batch_size,
			ctx=ctx,
			validation_map=validation_map,
			**kargs
		)
