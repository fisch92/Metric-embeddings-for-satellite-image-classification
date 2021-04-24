from dataProcessor.poolMiner import PoolMiner
from dataProcessor.tiffReader import GEOMAP

def main():
	miner = PoolMiner(96, 'validationPool', GEOMAP.OSM, validation=False, singleClassTreshold=0.0, pool_size=10000)
	miner.mine()

if __name__ == '__main__':
	main()