__author__ = 'jdk288'
import code.utils as utils
from config import EXAMPLE_DATA,TRIP_DATA_1,TRIP_DATA_2, TRAIN_DATA
import numpy as np

def problem1c():

	##Problem 1C Parameters
	k = 1
	str_fields = [5]
	float_fields = [10, 11, 12, 13, 9]
	observed_float_fields = [8]

	generatorCreatingFunctionForFilter = utils.createFunctionForGenerator(str_fields, float_fields+observed_float_fields)
	mean_dev_filter = utils.derive_filter(generatorCreatingFunctionForFilter(TRAIN_DATA))

	(training_data, neighborModel) = utils.trainModel(k, str_fields, float_fields, observed_float_fields, TRAIN_DATA, row_filter=mean_dev_filter, row_tranformer=utils.simplePickupTimeTransformer)
	#print len(training_data)
	training_observed = training_data[:,len(str_fields) + len(float_fields)]

	(rms, R, mae) = utils.testModel(neighborModel, training_observed, k, str_fields, float_fields, observed_float_fields,  
	TRIP_DATA_1, test_entry_count=100000, row_filter=mean_dev_filter, row_tranformer=utils.simplePickupTimeTransformer)

	print "********* RESULTS FOR 1.C **************\n"
	utils.printResults(rms, R, mae)
	print "****************************************\n"

if __name__ == '__main__':
	problem1c()