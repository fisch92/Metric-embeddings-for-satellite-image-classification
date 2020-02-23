import dataProcessor.batchSampler as batchSampler
import utils.visualizer as visualizer
import dataProcessor.tiffReader as tiffReader
import networkProcessor.tripletTrainer as tripletTrainer
import utils.validation as validation

def main():
	#tiffReader.unitTest()
	#batchSampler.unitTest()
	#visualizer.unitTest()
	#tripletTrainer.unitTest()
	validation.unitTest()

if __name__ == '__main__':
	main()