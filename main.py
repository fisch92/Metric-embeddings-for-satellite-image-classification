import json
import mxnet as mx
from network.resnext import resnext50_32x4d
from networkProcessor.tripletTrainer import TripletTrainer
from networkProcessor.quadrupletTrainer import QuadrupletTrainer
from dataProcessor.batchSampler import MiningTypes
from utils.validation import Validation

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def main():
	with open('config.json') as config_file:
		config = json.load(config_file)

		imagesize = config["imagesize"]
		batchsize = config["batchsize"]
		min_pos_dist = config["min_pos_dist"]
		max_pos_dist = config["max_pos_dist"]
		margin = config["margin"]
		mining = config["mining"]
		validation_nb = config["validation_nb"]
		validation_val = config["validation_val"]

		ctx=mx.gpu()
		classifiers = {}
		if 'knn' in config["classifiers"]:
			classifiers['knn'] = KNeighborsClassifier()
		if 'svc' in config["classifiers"]:
			classifiers['svc'] = SVC(gamma='auto', C=0.5, cache_size=2048, max_iter=1000)
		if 'rfc' in config["classifiers"]:
			classifiers['rfc'] = RandomForestClassifier()
		if 'mlp' in config["classifiers"]:
			classifiers['mlp'] = MLPClassifier(max_iter=1000)

		net = resnext50_32x4d(ctx=ctx)
		validator = Validation(classifiers=classifiers, ctx=ctx)

		if config["loss"] == "quadrupletloss":
			trainer = QuadrupletTrainer(
				batch_size=batchsize, 
				image_size=imagesize, 
				ctx=ctx, 
				mining=[],#[MiningTypes.RANDOM_HARD_NEGATIVE], 
				min_pos_dist=imagesize*min_pos_dist, 
				max_pos_dist=imagesize*max_pos_dist, 
				min_neg_dist=imagesize*30, 
				max_neg_dist=imagesize*100, 
				margin=margin,
				net=net,
				validator=validator,
				validation_nb=validation_nb,
				validation_val=validation_val)
		elif config["loss"] == "tripletloss":
			trainer = TripletTrainer(
				batch_size=batchsize, 
				image_size=imagesize, 
				ctx=ctx, 
				mining=[],#[MiningTypes.RANDOM_HARD_NEGATIVE], 
				min_pos_dist=imagesize*min_pos_dist, 
				max_pos_dist=imagesize*max_pos_dist, 
				min_neg_dist=imagesize*30, 
				max_neg_dist=imagesize*100, 
				margin=margin,
				net=net,
				validator=validator,
				validation_nb=validation_nb,
				validation_val=validation_val)

		trainer.train(epochs=1000, iterations=25)

if __name__ == '__main__':
	main()