import dataProcessor.batchSampler as batchSampler
import utils.visualizer as visualizer
import dataProcessor.tiffReader as tiffReader
import networkProcessor.tripletTrainer as tripletTrainer
import validation.single_class_validation as validation
import utils.math as math
import validation_tests.fuzzyClassValidationTest as fuzzyClassValidationTest


def main():
	#tiffReader.unitTest()
	#tripletBatchSampler.unitTest()
	#visualizer.unitTest()
	#tripletTrainer.unitTest()
	#validation.unitTest()
	#math.unitTest()
	fuzzyClassValidationTest.unitTest()

if __name__ == '__main__':
	main()