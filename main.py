import os
import sys
import json
import mxnet as mx
import numpy as np

from network.resnext import resnext50_32x4d
from network.vit.tranformer import TransformerNet
from networkProcessor.tripletTrainer import TripletTrainer
from networkProcessor.quadrupletTrainer import QuadrupletTrainer
from networkProcessor.magnetTrainer import MagNetTrainer
from dataProcessor.miningTypes import MiningTypes
from dataProcessor.tiffReader import GEOMAP
from validation.single_class_validation import SingleClassValidation
from validation.multi_class_validation import MultiClassValidation

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def loadConfig(config, name=None, valIteration=5):
    with open(config) as config_file:
        config = json.load(config_file)

        print(json.dumps(config, indent=4, sort_keys=True))

        config["min_pos_dist"] *= config["image_size"]
        config["max_pos_dist"] *= config["image_size"]
        config["min_neg_dist"] *= config["image_size"]
        config["max_neg_dist"] *= config["image_size"]
        config["margin2"] = config["margin"]*0.5
        mining = config["mining"]
        config["norm_output"] = config["norm_output"] == "True"
        config["supervised"] = config["supervised"] == "True"
        config["alt_loss"] = config["alt_loss"] == "True"
        config['mining'] = [MiningTypes.getType(x) for x in config['mining']]
        config['ctx'] = [mx.gpu(1), mx.gpu(0)]

        if config["validation_map"] == 'osm':
            config["validation_map"] = GEOMAP.OSM
        elif config["validation_map"] == 'clc':
            config["validation_map"] = GEOMAP.CLC
        elif config["validation_map"] == 'tile2vec':
            config["validation_map"] = GEOMAP.TILE2VEC
        else:
            raise Exception(('Map not found', config["validation_map"]))

        multi_class_classifiers = {}

        if 'knn' in config["multi_class_classifiers"]:
            multi_class_classifiers['knn'] = MultiOutputRegressor(
                KNeighborsRegressor())
        if 'rfc' in config["multi_class_classifiers"]:
            multi_class_classifiers['rfc'] = MultiOutputRegressor(
                RandomForestRegressor(n_estimators=150))
        if 'mlp' in config["multi_class_classifiers"]:
            multi_class_classifiers['mpl'] = MultiOutputRegressor(
                MLPRegressor(max_iter=2500))
        if 'dt' in config["multi_class_classifiers"]:
            multi_class_classifiers['dt'] = MultiOutputRegressor(
                DecisionTreeRegressor())

        single_class_classifiers = {}

        if 'knn' in config["single_class_classifiers"]:
            single_class_classifiers['knn'] = KNeighborsClassifier()
        if 'svc' in config["single_class_classifiers"]:
            single_class_classifiers['svc'] = SVC(
                gamma='auto', C=0.5, cache_size=2048, max_iter=2500)
        if 'rfc' in config["single_class_classifiers"]:
            single_class_classifiers['rfc'] = RandomForestClassifier(
                n_estimators=150)
        if 'mlp' in config["single_class_classifiers"]:
            single_class_classifiers['mlp'] = MLPClassifier(max_iter=2500)
        if 'dt' in config["single_class_classifiers"]:
            single_class_classifiers['dt'] = DecisionTreeClassifier()

        if config["network"] == "resnext50":
            net = resnext50_32x4d(
                ctx=config['ctx'], norm_output=config["norm_output"], classes=config["output_size"])
        elif config["network"] == "transformer":
            net = TransformerNet(
                norm_output=config["norm_output"], classes=config["output_size"], img_size=config["image_size"], size=8)

        single_class_validator = SingleClassValidation(
            config["image_size"], classifiers=single_class_classifiers, validation_map=config["validation_map"])
        multi_class_validator = MultiClassValidation(
            config["image_size"], classifiers=multi_class_classifiers, validation_map=config["validation_map"])

        config['net'] = net
        config['single_class_validator'] = single_class_validator
        config['multi_class_validator'] = multi_class_validator

        if config["loss"] == "quadrupletloss":
            trainer = QuadrupletTrainer(**config)
        elif config["loss"] == "tripletloss":
            trainer = TripletTrainer(**config)
        elif config["loss"] == "magnetloss":
            trainer = MagNetTrainer(**config)

        return trainer
        #trainer.train(epochs=1000, iterations=100)


if __name__ == '__main__':

    trainer = loadConfig('config.json', valIteration=1)
    trainer.train(epochs=1000, iterations=250)
