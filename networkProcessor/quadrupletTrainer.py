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
from dataProcessor.batchSampler.quadrupletBatchSampler import QuadrupletBatchSampler
from dataProcessor.supervisedImageSampler import SupervisedImageSampler
from dataProcessor.miningTypes import MiningTypes
from network.quadrupletnet import QuadrupletNet


class QuadrupletTrainer(Trainer):

	def __init__(self, 
		batch_size, 
		image_size, 
		min_pos_dist=64/8, 
		max_pos_dist=64, 
		min_neg_dist=64*5, 
		max_neg_dist=64*10, 
		mining=[MiningTypes.RANDOM_HARD_NEGATIVE], 
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
			quadrupletNet = QuadrupletNet(ctx=ctx, margin=margin, margin2=margin2, net=net, load=load, alt_loss=alt_loss)
		else:
			quadrupletNet = QuadrupletNet(ctx=ctx, margin=margin, margin2=margin2, net=net, load=load, alt_loss=alt_loss, name=name)
		
		if supervised:
			imageSampler = SupervisedImageSampler(image_size, validationmap=validation_map, singleClassTreshold=singleClassTreshold)
		else:
			imageSampler = ImageSampler(min_pos_dist, max_pos_dist, min_neg_dist, max_neg_dist, image_size, validationmap=validation_map, random_reset=random_reset, singleClassTreshold=singleClassTreshold)
		#batchSampler = TripletBatchSampler(batch_size, imageSampler, tripletNet.predict, Distances.L2_Dist, mining, random_mining_iterations=10, ctx=ctx)
		batchSampler = QuadrupletBatchSampler(batch_size=batch_size, imageSampler=imageSampler, net=quadrupletNet.predict, distance=Distances.L2_Dist, mining=mining, random_mining_iterations=3, ctx=ctx[0])
		
		super().__init__(
			imageSampler=imageSampler, 
			batchSampler=batchSampler, 
			net=quadrupletNet,
			name='QuadrupletNet',
			batch_size=batch_size,
			ctx=ctx,
			validation_map=validation_map,
			**kargs
		)
